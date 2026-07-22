"""Independent semantic verification for evidence-grounded answer submission."""

import json
import re
from typing import Any, Dict, Optional, Tuple

from arag.core.context import AgentContext


VERIFIER_SYSTEM_PROMPT = """You are an independent evidence verifier for multi-hop QA.
You did not perform the retrieval and must not trust the agent's completion labels.
Use only the supplied question, task state, direct evidence quotes, and candidate answer.

Check every item rigorously:
- The quote directly entails the subtask answer and satisfies the stated criterion.
- Subject, relation, and object direction are correct.
- The answer has the requested type and scope (person, place, year, network, number,
  comparison result, or another explicitly requested type).
- A bridge entity is not used as the final answer when another relation is requested.
- The final answer follows from combining the completed subtasks.
- Contradictions, a missing hop, or a criterion tailored to an unsupported answer make it invalid.

Return ONLY one strict JSON object with this schema:
{
  "valid": false,
  "subtask_checks": [{
    "subtask_id": "h1",
    "evidence_entails_answer": false,
    "criterion_satisfied": false,
    "relation_direction_correct": false,
    "answer_type_correct": false,
    "reason": "concise explanation"
  }],
  "final_answer_supported": false,
  "final_answer_matches_question": false,
  "missing_subtasks": ["specific missing fact or hop"],
  "conflicts": [],
  "recommended_next_query": "concise retrieval query",
  "failure_types": ["evidence_not_entailed"]
}
Set valid=true only when every subtask and final-answer check passes."""


class EvidenceVerifier:
    """Call a separate LLM role to validate compact task evidence."""

    def __init__(self, llm_client):
        self.llm = llm_client

    @staticmethod
    def _extract_json(content: str) -> Dict[str, Any]:
        text = str(content or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise ValueError("verifier returned no JSON object")
            parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("verifier JSON must be an object")
        return parsed

    @staticmethod
    def _payload(context: AgentContext, submission: Dict[str, Any]) -> Dict[str, Any]:
        state = context.task_state
        return {
            "original_question": context.original_question,
            "expected_answer_type": state.get("expected_answer_type", ""),
            "subtasks": [
                {
                    "id": item.get("id", ""),
                    "question": item.get("question", ""),
                    "completion_criterion": item.get("completion_criterion", ""),
                    "depends_on": item.get("depends_on", []),
                    "answer": item.get("answer", ""),
                    "evidence_quote": item.get("evidence_quote", ""),
                    "evidence_chunk_ids": item.get("evidence_chunk_ids", []),
                }
                for item in state.get("subtasks", [])
            ],
            "candidate_answer": submission.get("answer", ""),
            "submitted_answer_type": submission.get("answer_type", ""),
        }

    @staticmethod
    def _normalize_result(result: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {
            "valid": bool(result.get("valid", False)),
            "subtask_checks": result.get("subtask_checks", []),
            "final_answer_supported": bool(result.get("final_answer_supported", False)),
            "final_answer_matches_question": bool(
                result.get("final_answer_matches_question", False)
            ),
            "missing_subtasks": [str(item) for item in result.get("missing_subtasks", [])],
            "conflicts": [str(item) for item in result.get("conflicts", [])],
            "recommended_next_query": str(result.get("recommended_next_query", "")),
            "failure_types": [str(item) for item in result.get("failure_types", [])],
        }
        if not normalized["final_answer_supported"]:
            normalized["valid"] = False
        if not normalized["final_answer_matches_question"]:
            normalized["valid"] = False
        if normalized["conflicts"]:
            normalized["valid"] = False
        return normalized

    def verify(
        self,
        context: AgentContext,
        submission: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], float]:
        """Run deterministic checks, then semantic checks when structurally sound."""

        active_submission = submission or context.answer_submission
        structural = context.deterministic_validation(active_submission)
        if not structural["structural_checks_passed"]:
            result = {
                "valid": False,
                "stage": "structural",
                "structural_validation": structural,
                "subtask_checks": structural.get("subtask_checks", []),
                "final_answer_supported": False,
                "final_answer_matches_question": False,
                "missing_subtasks": structural.get("missing_subtasks", []),
                "conflicts": structural.get("conflicts", []),
                "recommended_next_query": context.task_state.get("next_action", {}).get(
                    "query", ""
                ),
                "failure_types": ["structural_validation_failed"],
            }
            context.record_verification(result)
            return result, 0.0

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            self._payload(context, active_submission), ensure_ascii=False
                        ),
                    },
                ],
                tools=None,
                temperature=0.0,
            )
            result = self._normalize_result(
                self._extract_json(response["message"].get("content", ""))
            )
            result["stage"] = "semantic"
            result["structural_validation"] = structural
            cost = float(response.get("cost", 0.0))
        except Exception as exc:
            result = {
                "valid": False,
                "stage": "semantic",
                "structural_validation": structural,
                "subtask_checks": [],
                "final_answer_supported": False,
                "final_answer_matches_question": False,
                "missing_subtasks": [],
                "conflicts": [],
                "recommended_next_query": "",
                "failure_types": ["verifier_error"],
                "verifier_error": str(exc),
            }
            cost = 0.0

        context.record_verification(result)
        return result, cost

