"""Agent execution context for ARAG."""

from typing import Any, Dict, List, Set
from dataclasses import dataclass, field


@dataclass
class RetrievalLog:
    """Log entry for a retrieval operation."""
    tool_name: str
    tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentContext:
    """Context manager for agent execution state."""
    
    def __init__(self):
        # Token statistics
        self.total_retrieved_tokens: int = 0
        self.retrieval_logs: List[RetrievalLog] = []
        
        # State management
        self.read_chunk_ids: Set[str] = set()
        self.search_history: List[Dict[str, Any]] = []

        # Optional task-list state used by experimental agents.
        self.task_state: Dict[str, Any] = {}
        self.task_state_history: List[Dict[str, Any]] = []
        self.task_completion_guard_events: List[Dict[str, Any]] = []
    
    def add_retrieval_log(
        self,
        tool_name: str,
        tokens: int,
        metadata: Dict[str, Any] = None
    ):
        """Add a retrieval log entry."""
        log = RetrievalLog(
            tool_name=tool_name,
            tokens=tokens,
            metadata=metadata or {}
        )
        self.retrieval_logs.append(log)
        self.total_retrieved_tokens += tokens
    
    def mark_chunk_as_read(self, chunk_id: str):
        """Mark chunk as read."""
        self.read_chunk_ids.add(str(chunk_id))
    
    def is_chunk_read(self, chunk_id: str) -> bool:
        """Check if chunk has been read."""
        return str(chunk_id) in self.read_chunk_ids
    
    # Aliases for backward compatibility
    def add_read_chunk(self, chunk_id: str, content: str = None):
        """Alias for mark_chunk_as_read."""
        self.mark_chunk_as_read(chunk_id)
    
    def has_read_chunk(self, chunk_id: str) -> bool:
        """Alias for is_chunk_read."""
        return self.is_chunk_read(chunk_id)
    
    def get_read_chunk(self, chunk_id: str):
        """Check if chunk was read (returns None, content not stored)."""
        return None if not self.is_chunk_read(chunk_id) else ""

    def update_task_state(
        self,
        subtasks: List[Dict[str, Any]],
        expected_answer_type: str,
        candidate_answer: str = "",
        candidate_evidence_chunk_ids: List[str] = None,
        unresolved_conflicts: List[str] = None,
        next_action: Dict[str, Any] = None,
    ):
        """Persist the model's concise subproblem and evidence state."""
        normalized_subtasks = []
        for subtask in subtasks:
            normalized_subtasks.append({
                "id": str(subtask.get("id", "")),
                "question": str(subtask.get("question", "")),
                "completion_criterion": str(subtask.get("completion_criterion", "")),
                "status": str(subtask.get("status", "pending")).lower(),
                "answer": str(subtask.get("answer", "")),
                "evidence_quote": str(subtask.get("evidence_quote", "")),
                "evidence_chunk_ids": [
                    str(chunk_id) for chunk_id in subtask.get("evidence_chunk_ids", [])
                ],
            })

        state = {
            "subtasks": normalized_subtasks,
            "expected_answer_type": str(expected_answer_type),
            "candidate_answer": str(candidate_answer),
            "candidate_evidence_chunk_ids": [
                str(chunk_id) for chunk_id in (candidate_evidence_chunk_ids or [])
            ],
            "unresolved_conflicts": [
                str(conflict) for conflict in (unresolved_conflicts or [])
            ],
            "next_action": next_action or {},
        }
        self.task_state = state
        self.task_state_history.append(state)

    def task_completion_status(self) -> tuple:
        """Return whether the current task state is safe to answer from."""
        if not self.task_state:
            return False, "task state has not been initialized"

        subtasks = self.task_state.get("subtasks", [])
        if not subtasks:
            return False, "task list has no subtasks"

        pending = [
            subtask.get("id") or subtask.get("question")
            for subtask in subtasks
            if subtask.get("status") != "completed"
        ]
        if pending:
            return False, f"pending subtasks: {pending}"

        incomplete_support = [
            subtask.get("id") or subtask.get("question")
            for subtask in subtasks
            if (
                not subtask.get("completion_criterion", "").strip()
                or not subtask.get("answer", "").strip()
                or not subtask.get("evidence_quote", "").strip()
                or not subtask.get("evidence_chunk_ids", [])
            )
        ]
        if incomplete_support:
            return False, f"completed subtasks lack direct support: {incomplete_support}"

        subtask_evidence_ids = {
            chunk_id
            for subtask in subtasks
            for chunk_id in subtask.get("evidence_chunk_ids", [])
        }
        unread_subtask_ids = subtask_evidence_ids - self.read_chunk_ids
        if unread_subtask_ids:
            return False, (
                "subtask evidence chunks have not been read: "
                f"{sorted(unread_subtask_ids)}"
            )

        conflicts = self.task_state.get("unresolved_conflicts", [])
        if conflicts:
            return False, f"unresolved conflicts: {conflicts}"

        if not self.task_state.get("candidate_answer", "").strip():
            return False, "candidate answer is empty"

        evidence_ids = set(self.task_state.get("candidate_evidence_chunk_ids", []))
        if not evidence_ids:
            return False, "candidate answer has no evidence chunk IDs"

        unread_ids = evidence_ids - self.read_chunk_ids
        if unread_ids:
            return False, f"candidate evidence chunks have not been read: {sorted(unread_ids)}"

        return True, "all required subtasks and evidence checks are complete"

    def add_task_completion_guard_event(self, loop: int, reason: str, proposed_answer: str):
        """Record a blocked early-answer attempt for experiment analysis."""
        self.task_completion_guard_events.append({
            "loop": loop,
            "reason": reason,
            "proposed_answer": proposed_answer,
        })
    
    def reset(self):
        """Reset context for new query."""
        self.retrieval_logs = []
        self.read_chunk_ids = set()
        self.search_history = []
        self.total_retrieved_tokens = 0
        self.task_state = {}
        self.task_state_history = []
        self.task_completion_guard_events = []
    
    def get_summary(self) -> Dict[str, Any]:
        """Get context summary."""
        summary = {
            "total_retrieved_tokens": self.total_retrieved_tokens,
            "retrieval_logs": [
                {
                    "tool_name": log.tool_name,
                    "tokens": log.tokens,
                    "metadata": log.metadata
                }
                for log in self.retrieval_logs
            ],
            "chunks_read_count": len(self.read_chunk_ids),
            "chunks_read_ids": list(self.read_chunk_ids)
        }
        if self.task_state or self.task_state_history or self.task_completion_guard_events:
            complete, reason = self.task_completion_status()
            summary.update({
                "task_state": self.task_state,
                "task_state_updates": len(self.task_state_history),
                "task_state_complete": complete,
                "task_state_completion_reason": reason,
                "task_completion_guard_events": self.task_completion_guard_events,
            })
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Export context as dictionary."""
        return self.get_summary()
