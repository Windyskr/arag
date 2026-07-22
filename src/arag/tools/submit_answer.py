"""Explicit answer-submission tool for evidence-gated agents."""

from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from arag.tools.base import BaseTool

if TYPE_CHECKING:
    from arag.core.context import AgentContext


class SubmitAnswerTool(BaseTool):
    """Register an answer proposal for automatic verification by the agent runtime."""

    @property
    def name(self) -> str:
        return "submit_answer"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Explicitly submit the single final answer after update_task_state. The runtime "
                    "will run deterministic evidence checks and an independent semantic verifier. "
                    "Call this alone, only when all required subtasks are complete."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "answer_type": {"type": "string"},
                        "completed_subtask_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "supporting_chunk_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "answer",
                        "answer_type",
                        "completed_subtask_ids",
                        "supporting_chunk_ids",
                    ],
                },
            },
        }

    def execute(
        self,
        context: "AgentContext",
        answer: str,
        answer_type: str,
        completed_subtask_ids: List[str],
        supporting_chunk_ids: List[str],
    ) -> Tuple[str, Dict[str, Any]]:
        context.submit_answer(
            answer=answer,
            answer_type=answer_type,
            completed_subtask_ids=completed_subtask_ids,
            supporting_chunk_ids=supporting_chunk_ids,
        )
        return (
            "Answer submitted. The runtime will now perform structural and semantic verification.",
            {"retrieved_tokens": 0, "answer_submitted": True},
        )
