#!/usr/bin/env python3
"""Rerun only baseline LLM-judge failures with the task-list experiment enabled."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from arag import BaseAgent, LLMClient
from arag.tools.update_task_state import UpdateTaskStateTool

from batch_runner import BatchRunner


class TaskListErrorRunner(BatchRunner):
    """Batch runner restricted to llm_accuracy=0 baseline records."""

    def __init__(self, *args, baseline_eval_file: str, selected_qids=None, **kwargs):
        self.baseline_eval_file = Path(baseline_eval_file)
        self.selected_qids = set(selected_qids or [])
        super().__init__(*args, **kwargs)

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
            failures = failures[:self.limit]
        return failures

    def _init_shared_tools(self):
        tools = super()._init_shared_tools()
        tools.register(UpdateTaskStateTool())
        return tools

    def _create_agent(self) -> BaseAgent:
        llm_config = self.config.get("llm", {})
        client = LLMClient(
            model=llm_config.get("model") or os.getenv("ARAG_MODEL", "gpt-4o-mini"),
            api_key=llm_config.get("api_key") or os.getenv("ARAG_API_KEY"),
            base_url=llm_config.get("base_url")
            or os.getenv("ARAG_BASE_URL", "https://api.openai.com/v1"),
            temperature=llm_config.get("temperature", 0.0),
            max_tokens=llm_config.get("max_tokens", 16384),
            reasoning_effort=llm_config.get("reasoning_effort"),
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
        )

    def _process_one(self, item: Dict[str, Any], agent: BaseAgent) -> Dict[str, Any]:
        qid = item["qid"]
        try:
            result = agent.run(item["question"])
            return {
                "qid": qid,
                "question": item["question"],
                "trajectory": result["trajectory"],
                "gold_answer": item["answer"],
                "pred_answer": result["answer"],
                "total_cost": result["total_cost"],
                "loops": result["loops"],
                "total_retrieved_tokens": result.get("total_retrieved_tokens", 0),
                "retrieval_logs": result.get("retrieval_logs", []),
                "chunks_read_count": result.get("chunks_read_count", 0),
                "chunks_read_ids": result.get("chunks_read_ids", []),
                "task_state": result.get("task_state", {}),
                "task_state_updates": result.get("task_state_updates", 0),
                "task_state_complete": result.get("task_state_complete", False),
                "task_state_completion_reason": result.get(
                    "task_state_completion_reason", ""
                ),
                "task_completion_guard_events": result.get(
                    "task_completion_guard_events", []
                ),
                "max_loops_exceeded": result.get("max_loops_exceeded", False),
                "token_budget_exceeded": result.get("token_budget_exceeded", False),
                "experiment": "persistent_task_list_v1",
            }
        except Exception as exc:
            return {
                "qid": qid,
                "question": item["question"],
                "trajectory": [],
                "gold_answer": item["answer"],
                "pred_answer": f"Error: {exc}",
                "total_cost": 0,
                "loops": 0,
                "total_retrieved_tokens": 0,
                "retrieval_logs": [],
                "chunks_read_count": 0,
                "chunks_read_ids": [],
                "task_state": {},
                "task_state_updates": 0,
                "task_state_complete": False,
                "task_completion_guard_events": [],
                "error": str(exc),
                "experiment": "persistent_task_list_v1",
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--baseline-eval", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--limit", "-l", type=int)
    parser.add_argument("--workers", "-w", type=int, default=5)
    parser.add_argument("--qid", action="append", dest="selected_qids")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    from arag import Config

    runner = TaskListErrorRunner(
        config=Config.from_yaml(args.config),
        questions_file="unused-by-task-list-runner.json",
        output_dir=args.output,
        limit=args.limit,
        num_workers=args.workers,
        verbose=args.verbose,
        baseline_eval_file=args.baseline_eval,
        selected_qids=args.selected_qids,
    )
    runner.run()


if __name__ == "__main__":
    main()
