from typing import Any, Dict, List, Tuple

from arag.agent.base import BaseAgent
from arag.core.context import AgentContext
from arag.tools.base import BaseTool
from arag.tools.registry import ToolRegistry
from arag.tools.update_task_state import UpdateTaskStateTool


def task_state_args(status="pending", candidate="", evidence=None):
    return {
        "subtasks": [{
            "id": "hop_1",
            "question": "Find the answer",
            "completion_criterion": "The source directly states the answer",
            "status": status,
            "answer": candidate,
            "evidence_quote": "The answer is CBS" if candidate else "",
            "evidence_chunk_ids": evidence or [],
        }],
        "expected_answer_type": "entity",
        "candidate_answer": candidate,
        "candidate_evidence_chunk_ids": evidence or [],
        "unresolved_conflicts": [],
        "next_action": {},
    }


def test_task_state_requires_completed_read_evidence():
    context = AgentContext()
    context.update_task_state(**task_state_args())
    assert context.task_completion_status()[0] is False

    context.update_task_state(**task_state_args("completed", "CBS", ["790"]))
    complete, reason = context.task_completion_status()
    assert complete is False
    assert "have not been read" in reason

    context.mark_chunk_as_read("790")
    assert context.task_completion_status()[0] is True


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
        context.mark_chunk_as_read(chunk_id)
        return "CBS", {"retrieved_tokens": 1}


class FakeLLM:
    def __init__(self, messages: List[Dict[str, Any]]):
        self.messages = iter(messages)

    def chat(self, **kwargs):
        return {"message": next(self.messages), "cost": 0}


def tool_call(call_id, name, arguments):
    import json

    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def test_agent_blocks_early_answer_until_task_state_is_complete():
    registry = ToolRegistry()
    registry.register(UpdateTaskStateTool())
    registry.register(FakeReadTool())

    fake_llm = FakeLLM([
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
        {"role": "assistant", "content": "CBS"},
    ])
    agent = BaseAgent(
        llm_client=fake_llm,
        tools=registry,
        max_loops=5,
        require_task_completion=True,
    )

    result = agent.run("Which network?")

    assert result["answer"] == "CBS"
    assert result["loops"] == 4
    assert result["task_state_complete"] is True
    assert len(result["task_completion_guard_events"]) == 1
    assert result["task_completion_guard_events"][0]["proposed_answer"] == "The CW"
