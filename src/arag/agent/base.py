"""Base agent implementation for ARAG."""

import json
from typing import Any, Dict, List, Optional, Tuple

import tiktoken

from arag.core.context import AgentContext
from arag.core.evidence_verifier import EvidenceVerifier
from arag.core.llm import LLMClient
from arag.tools.registry import ToolRegistry


class BaseAgent:
    """Base agent with tool calling and optional evidence-gated submission."""

    def __init__(
        self,
        llm_client: LLMClient,
        tools: ToolRegistry,
        system_prompt: str = None,
        max_loops: int = 10,
        max_token_budget: int = 128000,
        verbose: bool = False,
        require_task_completion: bool = False,
        evidence_verifier: Optional[EvidenceVerifier] = None,
    ):
        self.llm = llm_client
        self.tools = tools
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.max_loops = max_loops
        self.max_token_budget = max_token_budget
        self.verbose = verbose
        self.require_task_completion = require_task_completion
        self.evidence_verifier = evidence_verifier
        if self.require_task_completion and self.evidence_verifier is None:
            self.evidence_verifier = EvidenceVerifier(llm_client)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

    def _calculate_message_tokens(self, messages: List[Dict[str, Any]]) -> int:
        total = len(self.tokenizer.encode(self.system_prompt))
        for msg in messages:
            content = msg.get("content", "")
            if content:
                total += len(self.tokenizer.encode(str(content)))
        return total

    def _force_final_answer(
        self,
        messages: List[Dict[str, Any]],
        context: AgentContext,
        total_cost: float,
        reason: str,
    ) -> Tuple[str, float]:
        """Generate best-effort output only after the common gate has rejected exit."""

        compact_state = {
            "task_state": context.task_state,
            "last_verification": (
                context.verification_history[-1] if context.verification_history else {}
            ),
        }
        force_prompt = (
            "The runtime limit was reached and verification did not pass. Provide one concise "
            "best-effort answer from the gathered information. Do not claim that it was verified "
            "and do not call tools. Runtime reason: "
            f"{reason}. Compact state: {json.dumps(compact_state, ensure_ascii=False)}"
        )
        forced_messages = messages + [{"role": "user", "content": force_prompt}]
        try:
            response = self.llm.chat(
                messages=forced_messages,
                tools=None,
                temperature=0.0,
            )
            total_cost += float(response.get("cost", 0.0))
            final_answer = response["message"].get("content", "")
            if self.verbose:
                print(f"Forced answer: {final_answer[:200]}...")
        except Exception as exc:
            if self.verbose:
                print(f"Error getting forced answer: {exc}")
            candidate = context.task_state.get("candidate_answer", "").strip()
            final_answer = candidate or f"Error: {reason} and failed to generate final answer."
        return final_answer, total_cost

    @staticmethod
    def _missing_from_state(context: AgentContext) -> List[str]:
        missing = []
        for subtask in context.task_state.get("subtasks", []):
            if subtask.get("status") != "completed":
                missing.append(str(subtask.get("id") or subtask.get("question") or "unknown"))
        return missing

    def _ensure_limit_submission(self, context: AgentContext):
        state = context.task_state
        submission = context.answer_submission
        submission_is_current = (
            submission
            and int(submission.get("loop", -1)) >= int(state.get("updated_at_loop", 0))
            and submission.get("answer", "") == state.get("candidate_answer", "")
            and submission.get("answer_type", "") == state.get("expected_answer_type", "")
        )
        if submission_is_current:
            return
        context.submit_answer(
            answer=state.get("candidate_answer", ""),
            answer_type=state.get("expected_answer_type", ""),
            completed_subtask_ids=[
                str(item.get("id", ""))
                for item in state.get("subtasks", [])
                if item.get("status") == "completed"
            ],
            supporting_chunk_ids=state.get("candidate_evidence_chunk_ids", []),
        )

    def _verify(self, context: AgentContext) -> Tuple[Dict[str, Any], float]:
        if not self.evidence_verifier:
            raise RuntimeError("evidence verifier is required for task-completion gating")
        return self.evidence_verifier.verify(context)

    @staticmethod
    def _verification_reason(result: Dict[str, Any]) -> str:
        structural = result.get("structural_validation", {})
        failures = structural.get("structural_failures", [])
        if failures:
            return "; ".join(str(item) for item in failures)
        reasons = [
            str(item.get("reason", ""))
            for item in result.get("subtask_checks", [])
            if item.get("reason")
        ]
        reasons.extend(str(item) for item in result.get("failure_types", []))
        return "; ".join(dict.fromkeys(reasons)) or "semantic verification failed"

    @staticmethod
    def _correction_message(result: Dict[str, Any]) -> str:
        compact = {
            "stage": result.get("stage"),
            "subtask_checks": result.get("subtask_checks", []),
            "final_answer_supported": result.get("final_answer_supported", False),
            "final_answer_matches_question": result.get(
                "final_answer_matches_question", False
            ),
            "missing_subtasks": result.get("missing_subtasks", []),
            "conflicts": result.get("conflicts", []),
            "recommended_next_query": result.get("recommended_next_query", ""),
            "failure_types": result.get("failure_types", []),
            "structural_failures": result.get("structural_validation", {}).get(
                "structural_failures", []
            ),
        }
        return (
            "The submitted answer was rejected by the independent evidence gate. Correct the "
            "task state, read any needed chunks, and continue retrieval before submitting again. "
            f"Verification result: {json.dumps(compact, ensure_ascii=False)}"
        )

    @staticmethod
    def _result(
        answer: str,
        trajectory: List[Dict[str, Any]],
        total_cost: float,
        loop_count: int,
        context: AgentContext,
        **flags: Any,
    ) -> Dict[str, Any]:
        return {
            "answer": answer,
            "trajectory": trajectory,
            "total_cost": total_cost,
            "loops": loop_count,
            **flags,
            **context.get_summary(),
        }

    def _finish_at_limit(
        self,
        *,
        messages: List[Dict[str, Any]],
        context: AgentContext,
        trajectory: List[Dict[str, Any]],
        total_cost: float,
        loop_count: int,
        reason: str,
        flag_name: str,
    ) -> Dict[str, Any]:
        """Run the same gate for every non-normal termination path."""

        if not self.require_task_completion:
            final_answer, total_cost = self._force_final_answer(
                messages, context, total_cost, reason
            )
            context.record_finalization(forced=True, reason=reason)
            return self._result(
                final_answer,
                trajectory,
                total_cost,
                loop_count,
                context,
                **{flag_name: True},
            )

        self._ensure_limit_submission(context)
        verification, verification_cost = self._verify(context)
        total_cost += verification_cost
        if verification.get("valid"):
            context.record_finalization(forced=False)
            return self._result(
                context.answer_submission.get("answer", ""),
                trajectory,
                total_cost,
                loop_count,
                context,
                **{flag_name: True},
            )

        missing = list(dict.fromkeys(
            [str(item) for item in verification.get("missing_subtasks", [])]
            + self._missing_from_state(context)
        ))
        conflicts = list(dict.fromkeys(
            [str(item) for item in verification.get("conflicts", [])]
            + [str(item) for item in context.task_state.get("unresolved_conflicts", [])]
        ))
        context.record_finalization(
            forced=True,
            reason=reason,
            missing_subtasks=missing,
            unresolved_conflicts=conflicts,
        )
        final_answer, total_cost = self._force_final_answer(
            messages, context, total_cost, reason
        )
        return self._result(
            final_answer,
            trajectory,
            total_cost,
            loop_count,
            context,
            **{flag_name: True},
        )

    def run(self, query: str) -> Dict[str, Any]:
        context = AgentContext(original_question=query)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        trajectory: List[Dict[str, Any]] = []
        total_cost = 0.0
        loop_count = 0
        tool_schemas = self.tools.get_all_schemas()
        termination_reason = "Maximum loops exceeded"
        termination_flag = "max_loops_exceeded"

        if self.verbose:
            print(f"Question: {query}")

        for loop_idx in range(self.max_loops):
            loop_count = loop_idx + 1
            context.current_loop = loop_count
            current_tokens = self._calculate_message_tokens(messages)
            if current_tokens > self.max_token_budget:
                termination_reason = (
                    f"Token budget exceeded ({current_tokens} > {self.max_token_budget})"
                )
                termination_flag = "token_budget_exceeded"
                break

            if self.verbose:
                print(
                    f"Loop {loop_count}/{self.max_loops} "
                    f"(Tokens: {current_tokens}/{self.max_token_budget})"
                )

            try:
                response = self.llm.chat(messages=messages, tools=tool_schemas)
            except Exception as exc:
                termination_reason = f"LLM error: {exc}"
                termination_flag = "llm_error_forced"
                break

            total_cost += float(response.get("cost", 0.0))
            message = response["message"]
            messages.append(message)
            tool_calls = message.get("tool_calls")

            if not tool_calls:
                if not self.require_task_completion:
                    context.record_finalization(forced=False)
                    return self._result(
                        message.get("content", ""),
                        trajectory,
                        total_cost,
                        loop_count,
                        context,
                    )

                proposed_answer = message.get("content", "")
                reason = "an explicit submit_answer call is required"
                context.add_task_completion_guard_event(loop_count, reason, proposed_answer)
                messages.append({
                    "role": "user",
                    "content": (
                        "A natural-language response does not submit an answer. Update the task "
                        "state or call submit_answer with the candidate, answer type, completed "
                        "subtask IDs, and supporting read Chunk IDs."
                    ),
                })
                continue

            submitted_this_loop = False
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                try:
                    func_args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, TypeError):
                    func_args = {}

                try:
                    tool_result, tool_log = self.tools.execute(
                        func_name, context, **func_args
                    )
                except Exception as exc:
                    tool_result = f"Error executing tool: {exc}"
                    tool_log = {"retrieved_tokens": 0, "error": str(exc)}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                })
                trajectory.append({
                    "loop": loop_count,
                    "tool_name": func_name,
                    "arguments": func_args,
                    "tool_result": tool_result,
                    **tool_log,
                })
                submitted_this_loop = submitted_this_loop or bool(
                    tool_log.get("answer_submitted")
                )

            if submitted_this_loop and self.require_task_completion:
                verification, verification_cost = self._verify(context)
                total_cost += verification_cost
                if verification.get("valid"):
                    context.record_finalization(forced=False)
                    return self._result(
                        context.answer_submission.get("answer", ""),
                        trajectory,
                        total_cost,
                        loop_count,
                        context,
                    )

                reason = self._verification_reason(verification)
                context.add_task_completion_guard_event(
                    loop_count,
                    reason,
                    context.answer_submission.get("answer", ""),
                )
                messages.append({
                    "role": "user",
                    "content": self._correction_message(verification),
                })

        if self.verbose:
            print(f"{termination_reason}, checking gate before final output...")
        return self._finish_at_limit(
            messages=messages,
            context=context,
            trajectory=trajectory,
            total_cost=total_cost,
            loop_count=loop_count,
            reason=termination_reason,
            flag_name=termination_flag,
        )
