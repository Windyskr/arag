"""Persistent task-list state for multi-hop retrieval experiments."""

from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from arag.tools.base import BaseTool

if TYPE_CHECKING:
    from arag.core.context import AgentContext


class UpdateTaskStateTool(BaseTool):
    """Store concise subproblems, evidence, and the next retrieval action."""

    @property
    def name(self) -> str:
        return "update_task_state"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Update the persistent task list after reviewing the latest tool results. "
                    "Call this in the same assistant response as the next retrieval tool. Before "
                    "answering, mark every required subtask completed and cite read Chunk IDs for "
                    "the single candidate answer. Keep this state concise and evidence-based."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subtasks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "question": {"type": "string"},
                                    "completion_criterion": {
                                        "type": "string",
                                        "description": (
                                            "Exact subject-relation-object fact required to mark "
                                            "this subtask completed."
                                        ),
                                    },
                                    "status": {
                                        "type": "string",
                                        "enum": ["pending", "completed", "blocked"],
                                    },
                                    "answer": {"type": "string"},
                                    "evidence_quote": {
                                        "type": "string",
                                        "description": (
                                            "Short direct quote that satisfies the completion "
                                            "criterion; empty while pending."
                                        ),
                                    },
                                    "evidence_chunk_ids": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": [
                                    "id", "question", "completion_criterion", "status",
                                    "answer", "evidence_quote", "evidence_chunk_ids",
                                ],
                            },
                        },
                        "expected_answer_type": {"type": "string"},
                        "candidate_answer": {"type": "string"},
                        "candidate_evidence_chunk_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "unresolved_conflicts": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "next_action": {
                            "type": "object",
                            "description": "Concise next tool and query, or empty when complete.",
                        },
                    },
                    "required": [
                        "subtasks", "expected_answer_type", "candidate_answer",
                        "candidate_evidence_chunk_ids", "unresolved_conflicts", "next_action",
                    ],
                },
            },
        }

    def execute(
        self,
        context: "AgentContext",
        subtasks: List[Dict[str, Any]],
        expected_answer_type: str,
        candidate_answer: str = "",
        candidate_evidence_chunk_ids: List[str] = None,
        unresolved_conflicts: List[str] = None,
        next_action: Dict[str, Any] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        context.update_task_state(
            subtasks=subtasks,
            expected_answer_type=expected_answer_type,
            candidate_answer=candidate_answer,
            candidate_evidence_chunk_ids=candidate_evidence_chunk_ids,
            unresolved_conflicts=unresolved_conflicts,
            next_action=next_action,
        )
        complete, reason = context.task_completion_status()
        pending = [
            subtask["id"]
            for subtask in context.task_state["subtasks"]
            if subtask["status"] != "completed"
        ]
        result = (
            f"Task state updated. Complete: {complete}. Pending: {pending}. "
            f"Completion check: {reason}."
        )
        return result, {
            "retrieved_tokens": 0,
            "task_state_updated": True,
            "task_state_complete": complete,
        }
