import json
from typing import Any, Dict, List, Tuple

from arag.agent.base import BaseAgent
from arag.core.context import AgentContext
from arag.core.evidence_verifier import EvidenceVerifier
from arag.tools.base import BaseTool
from arag.tools.registry import ToolRegistry
from arag.tools.submit_answer import SubmitAnswerTool
from arag.tools.update_task_state import UpdateTaskStateTool


def task_state_args(status="pending", candidate="", evidence=None, quote=None):
    return {
        "subtasks": [{
            "id": "hop_1",
            "question": "Find the requested network",
            "completion_criterion": "The source directly states the requested network",
            "status": status,
            "answer": candidate,
            "evidence_quote": quote if quote is not None else (
                "The answer is CBS" if candidate else ""
            ),
            "evidence_chunk_ids": evidence or [],
            "depends_on": [],
        }],
        "expected_answer_type": "network",
        "candidate_answer": candidate,
        "candidate_evidence_chunk_ids": evidence or [],
        "unresolved_conflicts": [],
        "next_action": {},
    }


def submission_args(answer="CBS"):
    return {
        "answer": answer,
        "answer_type": "network",
        "completed_subtask_ids": ["hop_1"],
        "supporting_chunk_ids": ["790"],
    }


def completed_context(content="The answer is CBS"):
    context = AgentContext("Which network?")
    context.mark_chunk_as_read("790", content)
    context.update_task_state(**task_state_args("completed", "CBS", ["790"]))
    context.submit_answer(**submission_args())
    return context


def test_task_state_requires_read_chunk_body_and_real_quote():
    context = AgentContext()
    context.update_task_state(**task_state_args("completed", "CBS", ["790"]))
    complete, reason = context.task_completion_status()
    assert complete is False
    assert "have not been read" in reason
    assert context.task_state["subtasks"][0]["status"] == "unsupported"

    context.mark_chunk_as_read("790", "This text does not contain the claimed quotation.")
    context.update_task_state(**task_state_args("completed", "CBS", ["790"]))
    complete, reason = context.task_completion_status()
    assert complete is False
    assert "does not exist" in reason


def test_quote_check_normalizes_case_whitespace_newlines_and_unicode_quotes():
    context = AgentContext()
    context.mark_chunk_as_read("790", "The network said \u201cThe   Answer\nis CBS\u201d yesterday.")
    args = task_state_args(
        "completed", "CBS", ["790"], quote='the answer is cbs'
    )
    context.update_task_state(**args)
    result = context.deterministic_validation()
    assert result["structural_checks_passed"] is True
    assert result["subtask_checks"][0]["quote_exists"] is True
    assert result["subtask_checks"][0]["quote_chunk_id"] == "790"


def test_read_chunk_bodies_are_internal_and_reset_per_question():
    context = AgentContext()
    context.add_read_chunk("1", "private evidence body")
    context.add_read_chunk("1", "replacement must not overwrite")
    assert context.get_read_chunk("1") == "private evidence body"
    assert "read_chunks" not in context.get_summary()
    context.reset()
    assert context.read_chunk_ids == set()
    assert context.read_chunks == {}


def test_update_task_state_preserves_removed_subtasks_and_revisions_criteria():
    context = AgentContext()
    first = task_state_args()
    first["subtasks"].append({
        "id": "hop_2",
        "question": "Resolve the second relation",
        "completion_criterion": "Source states relation B",
        "status": "pending",
        "answer": "",
        "evidence_quote": "",
        "evidence_chunk_ids": [],
        "depends_on": ["hop_1"],
    })
    context.update_task_state(**first)

    second = task_state_args()
    second["subtasks"][0]["completion_criterion"] = "A materially different criterion"
    second["subtasks"][0]["status"] = "completed"
    context.update_task_state(**second)

    by_id = {item["id"]: item for item in context.task_state["subtasks"]}
    assert set(by_id) == {"hop_1", "hop_2"}
    assert by_id["hop_1"]["status"] == "pending"
    assert by_id["hop_1"]["revision"] == 2
    assert context.task_state["state_consistency_failures"]
    assert len(context.subtask_revision_history) == 1


def test_deterministic_check_rejects_bridge_intermediate_entity():
    context = AgentContext("Which network originally aired the show containing this episode?")
    context.mark_chunk_as_read("1", "The episode belongs to Example Show.")
    context.mark_chunk_as_read("2", "Example Show originally aired on CBS.")
    context.update_task_state(
        subtasks=[
            {
                "id": "h1",
                "question": "Identify the show",
                "completion_criterion": "The source states the episode's show",
                "status": "completed",
                "answer": "Example Show",
                "evidence_quote": "The episode belongs to Example Show.",
                "evidence_chunk_ids": ["1"],
                "depends_on": [],
            },
            {
                "id": "h2",
                "question": "Find that show's original network",
                "completion_criterion": "The source states the show's original network",
                "status": "completed",
                "answer": "CBS",
                "evidence_quote": "Example Show originally aired on CBS.",
                "evidence_chunk_ids": ["2"],
                "depends_on": ["h1"],
            },
        ],
        expected_answer_type="network",
        candidate_answer="Example Show",
        candidate_evidence_chunk_ids=["1"],
        unresolved_conflicts=[],
        next_action={},
    )
    context.submit_answer("Example Show", "network", ["h1", "h2"], ["1"])
    result = context.deterministic_validation()
    assert result["structural_checks_passed"] is False
    assert any("intermediate entity" in item for item in result["structural_failures"])


def test_update_task_state_tool_records_history():
    context = AgentContext()
    result, log = UpdateTaskStateTool().execute(context, **task_state_args())
    assert "Task state updated" in result
    assert log["task_state_updated"] is True
    assert context.get_summary()["task_state_updates"] == 1


class FakeReadTool(BaseTool):
    @property
    def name(self) -> str:
        return "fake_read"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "parameters": {
                    "type": "object",
                    "properties": {"chunk_id": {"type": "string"}},
                    "required": ["chunk_id"],
                },
            },
        }

    def execute(
        self, context: AgentContext, chunk_id: str
    ) -> Tuple[str, Dict[str, Any]]:
        context.mark_chunk_as_read(chunk_id, "The answer is CBS")
        return "The answer is CBS", {"retrieved_tokens": 1}


class FakeLLM:
    def __init__(self, messages: List[Dict[str, Any]]):
        self.messages = iter(messages)

    def chat(self, **kwargs):
        return {"message": next(self.messages), "cost": 0}


class FakeVerifierLLM:
    def __init__(self, valid_sequence):
        self.valid_sequence = iter(valid_sequence)

    def chat(self, **kwargs):
        valid = next(self.valid_sequence)
        result = {
            "valid": valid,
            "subtask_checks": [{
                "subtask_id": "hop_1",
                "evidence_entails_answer": valid,
                "criterion_satisfied": valid,
                "relation_direction_correct": valid,
                "answer_type_correct": valid,
                "reason": "" if valid else "The relation is not established.",
            }],
            "final_answer_supported": valid,
            "final_answer_matches_question": valid,
            "missing_subtasks": [] if valid else ["Verify the requested network relation"],
            "conflicts": [],
            "recommended_next_query": "show original network" if not valid else "",
            "failure_types": [] if valid else ["evidence_not_entailed"],
        }
        return {
            "message": {"role": "assistant", "content": json.dumps(result)},
            "cost": 0.01,
        }


def tool_call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def registry():
    tools = ToolRegistry()
    tools.register(UpdateTaskStateTool())
    tools.register(SubmitAnswerTool())
    tools.register(FakeReadTool())
    return tools


def test_agent_requires_submit_and_accepts_independent_verification():
    agent_llm = FakeLLM([
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                tool_call("state-1", "update_task_state", task_state_args()),
                tool_call("read-1", "fake_read", {"chunk_id": "790"}),
            ],
        },
        {"role": "assistant", "content": "The CW"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                tool_call(
                    "state-2",
                    "update_task_state",
                    task_state_args("completed", "CBS", ["790"]),
                )
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call("submit-1", "submit_answer", submission_args())],
        },
    ])
    agent = BaseAgent(
        llm_client=agent_llm,
        tools=registry(),
        max_loops=5,
        require_task_completion=True,
        evidence_verifier=EvidenceVerifier(FakeVerifierLLM([True])),
    )
    result = agent.run("Which network?")
    assert result["answer"] == "CBS"
    assert result["loops"] == 4
    assert result["task_state_complete"] is True
    assert result["verification_passed"] is True
    assert result["forced_answer"] is False
    assert result["task_completion_guard_events"][0]["proposed_answer"] == "The CW"


def test_semantic_rejection_returns_feedback_and_allows_resubmission():
    agent_llm = FakeLLM([
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call("read", "fake_read", {"chunk_id": "790"})],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call(
                "state", "update_task_state", task_state_args("completed", "CBS", ["790"])
            )],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call("submit-1", "submit_answer", submission_args())],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call("submit-2", "submit_answer", submission_args())],
        },
    ])
    verifier = EvidenceVerifier(FakeVerifierLLM([False, True]))
    agent = BaseAgent(
        agent_llm, registry(), max_loops=5, require_task_completion=True,
        evidence_verifier=verifier,
    )
    result = agent.run("Which network?")
    assert result["answer"] == "CBS"
    assert result["verification_attempts"] == 2
    assert result["verification_passed"] is True
    assert "evidence_not_entailed" in result["verification_failure_types"]
    assert result["task_state"]["next_action"]["query"] == "show original network"


def test_max_loops_verifies_complete_state_without_marking_forced():
    agent_llm = FakeLLM([
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call("read", "fake_read", {"chunk_id": "790"})],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call(
                "state", "update_task_state", task_state_args("completed", "CBS", ["790"])
            )],
        },
    ])
    agent = BaseAgent(
        agent_llm, registry(), max_loops=2, require_task_completion=True,
        evidence_verifier=EvidenceVerifier(FakeVerifierLLM([True])),
    )
    result = agent.run("Which network?")
    assert result["answer"] == "CBS"
    assert result["max_loops_exceeded"] is True
    assert result["verification_passed"] is True
    assert result["forced_answer"] is False


def test_max_loops_incomplete_state_records_forced_failure_fields():
    agent_llm = FakeLLM([
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call("state", "update_task_state", task_state_args())],
        },
        {"role": "assistant", "content": "Best effort: CBS"},
    ])
    agent = BaseAgent(
        agent_llm, registry(), max_loops=1, require_task_completion=True,
        evidence_verifier=EvidenceVerifier(FakeVerifierLLM([])),
    )
    result = agent.run("Which network?")
    assert result["answer"] == "Best effort: CBS"
    assert result["forced_answer"] is True
    assert result["task_state_complete"] is False
    assert result["verification_passed"] is False
    assert result["forced_reason"] == "Maximum loops exceeded"
    assert result["missing_subtasks"] == ["hop_1"]


def test_semantic_failure_keeps_task_incomplete_on_forced_exit():
    agent_llm = FakeLLM([
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call("read", "fake_read", {"chunk_id": "790"})],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call(
                "state", "update_task_state", task_state_args("completed", "CBS", ["790"])
            )],
        },
        {"role": "assistant", "content": "Best effort: CBS"},
    ])
    agent = BaseAgent(
        agent_llm, registry(), max_loops=2, require_task_completion=True,
        evidence_verifier=EvidenceVerifier(FakeVerifierLLM([False])),
    )
    result = agent.run("Which network?")
    assert result["structural_checks_passed"] is True
    assert result["verification_passed"] is False
    assert result["task_state_complete"] is False
    assert result["forced_answer"] is True
    assert result["missing_subtasks"] == ["Verify the requested network relation"]


def test_token_budget_uses_same_forced_exit_metadata():
    agent = BaseAgent(
        FakeLLM([{"role": "assistant", "content": "No evidence available"}]),
        registry(),
        max_loops=3,
        max_token_budget=0,
        require_task_completion=True,
        evidence_verifier=EvidenceVerifier(FakeVerifierLLM([])),
    )
    result = agent.run("Which network?")
    assert result["token_budget_exceeded"] is True
    assert result["forced_answer"] is True
    assert result["task_state_complete"] is False
    assert result["verification_passed"] is False
    assert "Token budget exceeded" in result["forced_reason"]
