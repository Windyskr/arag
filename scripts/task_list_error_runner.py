#!/usr/bin/env python3
"""Run the semantic evidence-gated task-list experiment."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from arag import BaseAgent, LLMClient
from arag.core.evidence_verifier import EvidenceVerifier
from arag.tools.submit_answer import SubmitAnswerTool
from arag.tools.update_task_state import UpdateTaskStateTool

from batch_runner import BatchRunner


SUMMARY_FIELDS = (
    "total_retrieved_tokens",
    "retrieval_logs",
    "chunks_read_count",
    "chunks_read_ids",
    "task_state",
    "task_state_updates",
    "task_state_complete",
    "task_state_completion_reason",
    "structural_checks_passed",
    "subtask_revision_history",
    "task_completion_guard_events",
    "answer_submission",
    "answer_submission_attempts",
    "verification_attempts",
    "verification_passed",
    "verification_history",
    "verification_failure_types",
    "forced_answer",
    "forced_reason",
    "missing_subtasks",
    "unresolved_conflicts",
    "max_loops_exceeded",
    "token_budget_exceeded",
    "llm_error_forced",
)


class SemanticTaskListRunner(BatchRunner):
    """Batch runner with explicit submission and independent evidence verification."""

    def __init__(self, *args, selected_qids=None, **kwargs):
        self.selected_qids = set(selected_qids or [])
        super().__init__(*args, **kwargs)

    def _load_questions(self) -> List[Dict[str, Any]]:
        questions = super()._load_questions()
        if self.selected_qids:
            questions = [
                item
                for item in questions
                if (item.get("qid") or item.get("id")) in self.selected_qids
            ]
        return questions

    def _init_shared_tools(self):
        tools = super()._init_shared_tools()
        tools.register(UpdateTaskStateTool())
        tools.register(SubmitAnswerTool())
        return tools

    @staticmethod
    def _client_from_config(config: Dict[str, Any], default_model: str) -> LLMClient:
        return LLMClient(
            model=config.get("model") or default_model,
            api_key=config.get("api_key") or os.getenv("ARAG_API_KEY"),
            base_url=config.get("base_url")
            or os.getenv("ARAG_BASE_URL", "https://api.openai.com/v1"),
            temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_tokens", 16384),
            reasoning_effort=config.get("reasoning_effort"),
            max_retries=config.get("max_retries", 5),
        )

    def _create_agent(self) -> BaseAgent:
        llm_config = self.config.get("llm", {})
        verifier_config = self.config.get("verifier", {})
        client = self._client_from_config(
            llm_config,
            os.getenv("ARAG_MODEL", "deepseek-v4-flash"),
        )
        verifier_client = self._client_from_config(
            verifier_config,
            os.getenv("ARAG_VERIFIER_MODEL", "gpt-5.4-mini"),
        )
        task_prompt_path = (
            Path(__file__).parent.parent / "src/arag/agent/prompts/task_list.txt"
        )
        task_prompt = task_prompt_path.read_text(encoding="utf-8")
        agent_config = self.config.get("agent", {})
        return BaseAgent(
            llm_client=client,
            tools=self._shared_tools,
            system_prompt=f"{self._system_prompt}\n\n{task_prompt}",
            max_loops=agent_config.get("max_loops", 10),
            max_token_budget=agent_config.get("max_token_budget", 128000),
            verbose=self.verbose,
            require_task_completion=True,
            evidence_verifier=EvidenceVerifier(verifier_client),
        )

    def _process_one(self, item: Dict[str, Any], agent: BaseAgent) -> Dict[str, Any]:
        qid = item.get("qid") or item.get("id")
        question = item.get("question", "")
        gold_answer = item.get("answer", item.get("gold_answer", ""))
        try:
            result = agent.run(question)
            prediction = {
                "qid": qid,
                "question": question,
                "trajectory": result["trajectory"],
                "gold_answer": gold_answer,
                "pred_answer": result["answer"],
                "total_cost": result["total_cost"],
                "loops": result["loops"],
                "experiment": "semantic_evidence_grounded_gate_v2",
                "agent_model": self.config.get("llm.model", "deepseek-v4-flash"),
                "verifier_model": self.config.get("verifier.model", "gpt-5.4-mini"),
            }
            prediction.update({field: result.get(field) for field in SUMMARY_FIELDS})
            for flag in (
                "max_loops_exceeded",
                "token_budget_exceeded",
                "llm_error_forced",
                "forced_answer",
                "verification_passed",
                "task_state_complete",
                "structural_checks_passed",
            ):
                prediction[flag] = bool(prediction.get(flag, False))
            return prediction
        except Exception as exc:
            prediction = {
                "qid": qid,
                "question": question,
                "trajectory": [],
                "gold_answer": gold_answer,
                "pred_answer": f"Error: {exc}",
                "total_cost": 0,
                "loops": 0,
                "error": str(exc),
                "experiment": "semantic_evidence_grounded_gate_v2",
                "agent_model": self.config.get("llm.model", "deepseek-v4-flash"),
                "verifier_model": self.config.get("verifier.model", "gpt-5.4-mini"),
            }
            prediction.update({field: None for field in SUMMARY_FIELDS})
            prediction["task_state_complete"] = False
            prediction["verification_passed"] = False
            prediction["forced_answer"] = False
            return prediction


class TaskListErrorRunner(SemanticTaskListRunner):
    """Semantic runner restricted to LLM-judge failures from a baseline file."""

    def __init__(self, *args, baseline_eval_file: str, selected_qids=None, **kwargs):
        self.baseline_eval_file = Path(baseline_eval_file)
        super().__init__(*args, selected_qids=selected_qids, **kwargs)

    def _load_questions(self) -> List[Dict[str, Any]]:
        failures = []
        with self.baseline_eval_file.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if (
                    row.get("llm_accuracy") == 0
                    and (not self.selected_qids or row["qid"] in self.selected_qids)
                ):
                    failures.append({
                        "qid": row["qid"],
                        "question": row["question"],
                        "answer": row["gold_answer"],
                    })
        if self.limit:
            failures = failures[: self.limit]
        return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--baseline-eval")
    source.add_argument("--questions")
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--limit", "-l", type=int)
    parser.add_argument("--workers", "-w", type=int, default=5)
    parser.add_argument("--qid", action="append", dest="selected_qids")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    from arag import Config

    common = {
        "config": Config.from_yaml(args.config),
        "output_dir": args.output,
        "limit": args.limit,
        "num_workers": args.workers,
        "verbose": args.verbose,
    }
    if args.baseline_eval:
        runner = TaskListErrorRunner(
            questions_file="unused-by-task-list-runner.json",
            baseline_eval_file=args.baseline_eval,
            selected_qids=args.selected_qids,
            **common,
        )
    else:
        runner = SemanticTaskListRunner(
            questions_file=args.questions,
            selected_qids=args.selected_qids,
            **common,
        )
    runner.run()


if __name__ == "__main__":
    main()
