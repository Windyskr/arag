"""Agent execution context for ARAG."""

import copy
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class RetrievalLog:
    """Log entry for a retrieval operation."""

    tool_name: str
    tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)


_QUOTE_TRANSLATION = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
    "\u00ab": '"',
    "\u00bb": '"',
})


def normalize_evidence_text(value: str) -> str:
    """Normalize text for deterministic direct-quote containment checks."""

    normalized = unicodedata.normalize("NFKC", str(value or ""))
    normalized = normalized.translate(_QUOTE_TRANSLATION).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


class AgentContext:
    """Context manager for per-question agent execution state."""

    def __init__(self, original_question: str = ""):
        self.original_question = str(original_question)
        self.current_loop = 0

        self.total_retrieved_tokens: int = 0
        self.retrieval_logs: List[RetrievalLog] = []

        self.read_chunk_ids: Set[str] = set()
        self.read_chunks: Dict[str, str] = {}
        self.search_history: List[Dict[str, Any]] = []

        self.task_state: Dict[str, Any] = {}
        self.task_state_history: List[Dict[str, Any]] = []
        self.subtask_revision_history: List[Dict[str, Any]] = []
        self.task_completion_guard_events: List[Dict[str, Any]] = []

        self.answer_submission: Dict[str, Any] = {}
        self.answer_submission_history: List[Dict[str, Any]] = []
        self.verification_history: List[Dict[str, Any]] = []
        self.verification_failure_types: List[str] = []
        self.verification_passed = False

        self.forced_answer = False
        self.forced_reason = ""
        self.final_missing_subtasks: List[str] = []
        self.final_unresolved_conflicts: List[str] = []

    def add_retrieval_log(
        self,
        tool_name: str,
        tokens: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add a retrieval log entry."""

        log = RetrievalLog(tool_name=tool_name, tokens=tokens, metadata=metadata or {})
        self.retrieval_logs.append(log)
        self.total_retrieved_tokens += tokens

    def mark_chunk_as_read(self, chunk_id: str, content: Optional[str] = None):
        """Mark a chunk as read and retain its body when supplied."""

        normalized_id = str(chunk_id)
        self.read_chunk_ids.add(normalized_id)
        if content is not None and normalized_id not in self.read_chunks:
            self.read_chunks[normalized_id] = str(content)

    def is_chunk_read(self, chunk_id: str) -> bool:
        """Check whether a chunk has been read."""

        return str(chunk_id) in self.read_chunk_ids

    def add_read_chunk(self, chunk_id: str, content: Optional[str] = None):
        """Backward-compatible alias for marking and storing a read chunk."""

        self.mark_chunk_as_read(chunk_id, content)

    def has_read_chunk(self, chunk_id: str) -> bool:
        """Backward-compatible alias for ``is_chunk_read``."""

        return self.is_chunk_read(chunk_id)

    def get_read_chunk(self, chunk_id: str) -> Optional[str]:
        """Return retained chunk content, if available."""

        return self.read_chunks.get(str(chunk_id))

    def _normalize_subtask(
        self,
        subtask: Dict[str, Any],
        previous: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        status = str(subtask.get("status", "pending")).lower()
        if status not in {"pending", "completed", "blocked", "unsupported"}:
            status = "pending"

        normalized = {
            "id": str(subtask.get("id", "")),
            "question": str(subtask.get("question", "")),
            "completion_criterion": str(subtask.get("completion_criterion", "")),
            "status": status,
            "answer": str(subtask.get("answer", "")),
            "evidence_quote": str(subtask.get("evidence_quote", "")),
            "evidence_chunk_ids": [
                str(chunk_id) for chunk_id in subtask.get("evidence_chunk_ids", [])
            ],
            "depends_on": [str(item) for item in subtask.get("depends_on", [])],
            "revision": int(previous.get("revision", 1) if previous else 1),
            "revision_reason": str(subtask.get("revision_reason", "")),
            "created_at_loop": int(
                previous.get("created_at_loop", self.current_loop) if previous else self.current_loop
            ),
            "completed_at_loop": None,
        }
        if previous:
            normalized["completed_at_loop"] = previous.get("completed_at_loop")

        changed_fields = [
            field_name
            for field_name in ("question", "completion_criterion", "answer")
            if previous and normalized[field_name] != previous.get(field_name, "")
        ]
        if changed_fields:
            normalized["revision"] += 1
            if not normalized["revision_reason"]:
                normalized["revision_reason"] = f"Changed {', '.join(changed_fields)}"
            self.subtask_revision_history.append({
                "loop": self.current_loop,
                "subtask_id": normalized["id"],
                "revision": normalized["revision"],
                "reason": normalized["revision_reason"],
                "changed_fields": changed_fields,
                "previous": copy.deepcopy(previous),
            })

            if "completion_criterion" in changed_fields:
                normalized["status"] = "pending"
                normalized["completed_at_loop"] = None

        if normalized["status"] == "completed" and normalized["completed_at_loop"] is None:
            normalized["completed_at_loop"] = self.current_loop
        elif normalized["status"] != "completed":
            normalized["completed_at_loop"] = None
        return normalized

    def update_task_state(
        self,
        subtasks: List[Dict[str, Any]],
        expected_answer_type: str,
        candidate_answer: str = "",
        candidate_evidence_chunk_ids: Optional[List[str]] = None,
        unresolved_conflicts: Optional[List[str]] = None,
        next_action: Optional[Dict[str, Any]] = None,
        removed_subtasks: Optional[List[Dict[str, Any]]] = None,
    ):
        """Persist task state while preserving established subtask obligations."""

        previous_by_id = {
            item.get("id", ""): item for item in self.task_state.get("subtasks", [])
        }
        normalized_subtasks = []
        seen_ids: Set[str] = set()
        consistency_failures: List[str] = []
        for subtask in subtasks:
            normalized = self._normalize_subtask(
                subtask, previous_by_id.get(str(subtask.get("id", "")))
            )
            if not normalized["id"]:
                consistency_failures.append("subtask ID cannot be empty")
                continue
            if normalized["id"] in seen_ids:
                consistency_failures.append(f"duplicate subtask ID: {normalized['id']}")
                continue
            seen_ids.add(normalized["id"])
            normalized_subtasks.append(normalized)

        removals = {
            str(item.get("id", "")): item
            for item in (removed_subtasks or [])
            if item.get("id")
        }
        for removed_id, previous in previous_by_id.items():
            if removed_id in seen_ids:
                continue
            removal = removals.get(removed_id, {})
            reason = str(removal.get("reason", "")).strip()
            if not reason:
                consistency_failures.append(
                    f"established subtask {removed_id} cannot disappear without a removal reason"
                )
                normalized_subtasks.append(copy.deepcopy(previous))
                continue
            self.subtask_revision_history.append({
                "loop": self.current_loop,
                "subtask_id": removed_id,
                "revision": int(previous.get("revision", 1)) + 1,
                "reason": reason,
                "change": "removed",
                "merged_into": str(removal.get("merged_into", "")),
                "previous": copy.deepcopy(previous),
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
            "state_consistency_failures": consistency_failures,
            "updated_at_loop": self.current_loop,
        }
        self.task_state = state
        self.task_state_history.append(copy.deepcopy(state))
        self.verification_passed = False

    def submit_answer(
        self,
        answer: str,
        answer_type: str,
        completed_subtask_ids: List[str],
        supporting_chunk_ids: List[str],
    ):
        """Register an explicit answer proposal for automatic validation."""

        submission = {
            "answer": str(answer),
            "answer_type": str(answer_type),
            "completed_subtask_ids": [str(item) for item in completed_subtask_ids],
            "supporting_chunk_ids": [str(item) for item in supporting_chunk_ids],
            "loop": self.current_loop,
        }
        self.answer_submission = submission
        self.answer_submission_history.append(copy.deepcopy(submission))
        self.verification_passed = False

    def _quote_location(self, quote: str, chunk_ids: List[str]) -> Optional[str]:
        normalized_quote = normalize_evidence_text(quote)
        if not normalized_quote:
            return None
        for chunk_id in chunk_ids:
            content = self.read_chunks.get(str(chunk_id))
            if content is not None and normalized_quote in normalize_evidence_text(content):
                return str(chunk_id)
        return None

    def deterministic_validation(
        self, submission: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run model-independent task, quote, and candidate checks."""

        failures: List[str] = []
        missing_subtasks: List[str] = []
        subtask_checks: List[Dict[str, Any]] = []
        state = self.task_state
        subtasks = state.get("subtasks", []) if state else []

        if not state:
            failures.append("task state has not been initialized")
        elif not subtasks:
            failures.append("task list has no subtasks")

        failures.extend(state.get("state_consistency_failures", []))
        for subtask in subtasks:
            subtask_id = subtask.get("id") or subtask.get("question") or "unknown"
            item_failures: List[str] = []
            if subtask.get("status") != "completed":
                item_failures.append(f"status is {subtask.get('status', 'pending')}")
            for field_name in ("question", "completion_criterion", "answer", "evidence_quote"):
                if not str(subtask.get(field_name, "")).strip():
                    item_failures.append(f"{field_name} is empty")

            evidence_ids = [str(item) for item in subtask.get("evidence_chunk_ids", [])]
            if not evidence_ids:
                item_failures.append("evidence_chunk_ids is empty")
            unread = sorted(set(evidence_ids) - self.read_chunk_ids)
            if unread:
                item_failures.append(f"evidence chunks have not been read: {unread}")
            quote_chunk_id = self._quote_location(subtask.get("evidence_quote", ""), evidence_ids)
            if subtask.get("evidence_quote", "").strip() and quote_chunk_id is None:
                item_failures.append("evidence_quote does not exist in a declared read chunk")

            quote_exists = quote_chunk_id is not None
            if item_failures:
                missing_subtasks.append(str(subtask_id))
                if subtask.get("status") == "completed":
                    subtask["status"] = "unsupported"
                    subtask["completed_at_loop"] = None
            subtask_checks.append({
                "subtask_id": str(subtask_id),
                "quote_exists": quote_exists,
                "quote_chunk_id": quote_chunk_id,
                "passed": not item_failures,
                "failures": item_failures,
            })

        conflicts = [str(item) for item in state.get("unresolved_conflicts", [])]
        if conflicts:
            failures.append(f"unresolved conflicts: {conflicts}")

        expected_type = str(state.get("expected_answer_type", "")).strip()
        candidate = str(state.get("candidate_answer", "")).strip()
        candidate_ids = [str(item) for item in state.get("candidate_evidence_chunk_ids", [])]
        if not expected_type:
            failures.append("expected answer type is empty")
        if not candidate:
            failures.append("candidate answer is empty")
        if not candidate_ids:
            failures.append("candidate answer has no evidence chunk IDs")
        unread_candidate = sorted(set(candidate_ids) - self.read_chunk_ids)
        if unread_candidate:
            failures.append(f"candidate evidence chunks have not been read: {unread_candidate}")

        active_submission = submission or self.answer_submission
        if active_submission:
            submitted_answer = str(active_submission.get("answer", "")).strip()
            submitted_type = str(active_submission.get("answer_type", "")).strip()
            completed_ids = set(active_submission.get("completed_subtask_ids", []))
            supporting_ids = set(active_submission.get("supporting_chunk_ids", []))
            required_ids = {str(item.get("id", "")) for item in subtasks}
            if not submitted_answer:
                failures.append("submitted answer is empty")
            if normalize_evidence_text(submitted_answer) != normalize_evidence_text(candidate):
                failures.append("submitted answer does not match task-state candidate answer")
            if not submitted_type:
                failures.append("submitted answer type is empty")
            elif expected_type and normalize_evidence_text(submitted_type) != normalize_evidence_text(
                expected_type
            ):
                failures.append("submitted answer type does not match expected answer type")
            missing_completed = sorted(required_ids - completed_ids)
            if missing_completed:
                failures.append(f"submission omits required subtasks: {missing_completed}")
            unread_support = sorted(supporting_ids - self.read_chunk_ids)
            if unread_support:
                failures.append(f"submission cites unread chunks: {unread_support}")
            if candidate_ids and not set(candidate_ids).issubset(supporting_ids):
                failures.append("submission omits candidate evidence chunks")

            by_id = {str(item.get("id", "")): item for item in subtasks}
            depended_on = {
                dependency
                for item in subtasks
                for dependency in item.get("depends_on", [])
            }
            leaf_tasks = [item for item_id, item in by_id.items() if item_id not in depended_on]
            if len(leaf_tasks) == 1:
                leaf_answer = str(leaf_tasks[0].get("answer", ""))
                non_leaf_answers = [
                    str(item.get("answer", ""))
                    for item_id, item in by_id.items()
                    if item_id in depended_on
                ]
                normalized_submission = normalize_evidence_text(submitted_answer)
                if (
                    normalized_submission in {normalize_evidence_text(item) for item in non_leaf_answers}
                    and normalized_submission != normalize_evidence_text(leaf_answer)
                ):
                    failures.append("submitted answer is an intermediate entity, not the final target")

        for check in subtask_checks:
            failures.extend(
                f"subtask {check['subtask_id']}: {failure}" for failure in check["failures"]
            )

        failures = list(dict.fromkeys(failures))
        return {
            "quote_exists": bool(subtask_checks) and all(
                item["quote_exists"] for item in subtask_checks
            ),
            "quote_chunk_id": next(
                (item["quote_chunk_id"] for item in subtask_checks if item["quote_chunk_id"]),
                None,
            ),
            "structural_checks_passed": not failures,
            "structural_failures": failures,
            "subtask_checks": subtask_checks,
            "missing_subtasks": list(dict.fromkeys(missing_subtasks)),
            "conflicts": conflicts,
        }

    def task_completion_status(self) -> tuple:
        """Return whether deterministic completion checks currently pass."""

        result = self.deterministic_validation()
        if result["structural_checks_passed"]:
            return True, "all deterministic task and evidence checks are complete"
        return False, "; ".join(result["structural_failures"])

    def record_verification(self, result: Dict[str, Any]):
        """Record one automatic structural/semantic verification attempt."""

        entry = copy.deepcopy(result)
        entry.setdefault("loop", self.current_loop)
        self.verification_history.append(entry)
        self.verification_passed = bool(entry.get("valid"))
        for failure_type in entry.get("failure_types", []):
            if failure_type not in self.verification_failure_types:
                self.verification_failure_types.append(failure_type)

        if not self.verification_passed:
            missing = [str(item) for item in entry.get("missing_subtasks", [])]
            recommended_query = str(entry.get("recommended_next_query", "")).strip()
            if self.task_state:
                self.task_state["verification_missing_subtasks"] = missing
                if recommended_query:
                    self.task_state["next_action"] = {
                        "tool": "search",
                        "query": recommended_query,
                        "reason": "semantic verification failed",
                    }

    def add_task_completion_guard_event(self, loop: int, reason: str, proposed_answer: str):
        """Record a blocked early-answer or rejected-submission attempt."""

        self.task_completion_guard_events.append({
            "loop": loop,
            "reason": reason,
            "proposed_answer": proposed_answer,
        })

    def record_finalization(
        self,
        *,
        forced: bool,
        reason: str = "",
        missing_subtasks: Optional[List[str]] = None,
        unresolved_conflicts: Optional[List[str]] = None,
    ):
        """Store final exit-path metadata for prediction analysis."""

        self.forced_answer = forced
        self.forced_reason = str(reason) if forced else ""
        self.final_missing_subtasks = [str(item) for item in (missing_subtasks or [])]
        self.final_unresolved_conflicts = [
            str(item) for item in (unresolved_conflicts or [])
        ]

    def reset(self):
        """Reset context for a new query."""

        original_question = self.original_question
        self.__init__(original_question=original_question)

    def get_summary(self) -> Dict[str, Any]:
        """Get context summary without embedding full read Chunk bodies."""

        structural_complete, reason = self.task_completion_status()
        complete = structural_complete and self.verification_passed
        return {
            "total_retrieved_tokens": self.total_retrieved_tokens,
            "retrieval_logs": [
                {
                    "tool_name": log.tool_name,
                    "tokens": log.tokens,
                    "metadata": log.metadata,
                }
                for log in self.retrieval_logs
            ],
            "chunks_read_count": len(self.read_chunk_ids),
            "chunks_read_ids": sorted(self.read_chunk_ids),
            "task_state": self.task_state,
            "task_state_updates": len(self.task_state_history),
            "task_state_complete": complete,
            "task_state_completion_reason": reason,
            "structural_checks_passed": structural_complete,
            "subtask_revision_history": self.subtask_revision_history,
            "task_completion_guard_events": self.task_completion_guard_events,
            "answer_submission": self.answer_submission,
            "answer_submission_attempts": len(self.answer_submission_history),
            "verification_attempts": len(self.verification_history),
            "verification_passed": self.verification_passed,
            "verification_history": self.verification_history,
            "verification_failure_types": self.verification_failure_types,
            "forced_answer": self.forced_answer,
            "forced_reason": self.forced_reason,
            "missing_subtasks": self.final_missing_subtasks,
            "unresolved_conflicts": self.final_unresolved_conflicts,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export context as a dictionary."""

        return self.get_summary()
