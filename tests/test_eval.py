import importlib.util
import json
from pathlib import Path

import pytest
import requests


EVAL_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval.py"
SPEC = importlib.util.spec_from_file_location("arag_eval_script", EVAL_SCRIPT)
eval_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(eval_script)


class FakeJudge:
    def __init__(self, model="judge-a", base_url="https://judge.example/v1", responder=None):
        self.model = model
        self.base_url = base_url
        self.responder = responder or (lambda messages: "correct")
        self.calls = []

    def generate(self, messages, temperature):
        self.calls.append(messages)
        return self.responder(messages), 0.0


def write_predictions(path, answers=("Alpha", "Beta", "Gamma")):
    rows = [
        {
            "qid": "q{}".format(index),
            "question": "Question {}".format(index),
            "pred_answer": answer,
            "gold_answer": answer,
            "total_cost": 0.01,
            "total_retrieved_tokens": 10,
            "loops": 1,
        }
        for index, answer in enumerate(answers)
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return rows


def test_checkpointed_evaluation_resumes_without_rejudging(tmp_path):
    predictions_path = tmp_path / "predictions.jsonl"
    source_rows = write_predictions(predictions_path, answers=("Alpha", "Beta"))
    output_dir = tmp_path / "eval"

    first_judge = FakeJudge()
    evaluator = eval_script.Evaluator(first_judge, predictions_path)
    llm_accuracy, contain_accuracy = evaluator.evaluate(max_workers=2, output_dir=output_dir)

    assert (llm_accuracy, contain_accuracy) == (1.0, 1.0)
    assert len(first_judge.calls) == 2
    checkpoint_path = output_dir / "predictions_judge_checkpoint.jsonl"
    checkpoint_lines = [json.loads(line) for line in checkpoint_path.read_text().splitlines()]
    assert checkpoint_lines[0]["type"] == "manifest"
    assert [entry["source_index"] for entry in checkpoint_lines[1:]] == [0, 1]
    assert predictions_path.read_text(encoding="utf-8") == "".join(
        json.dumps(row) + "\n" for row in source_rows
    )

    resumed_judge = FakeJudge(
        responder=lambda messages: pytest.fail("resume should not call the judge")
    )
    resumed = eval_script.Evaluator(resumed_judge, predictions_path)
    resumed_llm, resumed_contain = resumed.evaluate(
        max_workers=2,
        output_dir=output_dir,
        resume=True,
    )

    assert (resumed_llm, resumed_contain) == (1.0, 1.0)
    assert resumed_judge.calls == []
    evaluated_rows = [
        json.loads(line)
        for line in (output_dir / "predictions.jsonl").read_text().splitlines()
    ]
    assert all(row["status"] == "answered" for row in evaluated_rows)
    summary = json.loads((output_dir / "predictions_eval_summary.json").read_text())
    assert summary["resumed_records"] == 2
    assert summary["newly_judged_records"] == 0


def test_resume_rejects_changed_judge_before_network_access(tmp_path):
    predictions_path = tmp_path / "predictions.jsonl"
    write_predictions(predictions_path, answers=("Alpha",))
    output_dir = tmp_path / "eval"

    eval_script.Evaluator(FakeJudge(model="judge-a"), predictions_path).evaluate(
        max_workers=1,
        output_dir=output_dir,
    )

    changed_judge = FakeJudge(
        model="judge-b",
        responder=lambda messages: pytest.fail("mismatched resume must not call judge"),
    )
    evaluator = eval_script.Evaluator(changed_judge, predictions_path)
    with pytest.raises(ValueError, match="Checkpoint provenance"):
        evaluator.evaluate(max_workers=1, output_dir=output_dir, resume=True)

    assert changed_judge.calls == []


def test_evaluation_refuses_to_overwrite_raw_predictions(tmp_path):
    predictions_path = tmp_path / "predictions.jsonl"
    original = write_predictions(predictions_path, answers=("Alpha",))
    evaluator = eval_script.Evaluator(FakeJudge(), predictions_path)

    with pytest.raises(ValueError, match="overwrite raw predictions"):
        evaluator.evaluate(max_workers=1, output_dir=tmp_path)

    assert predictions_path.read_text(encoding="utf-8") == "".join(
        json.dumps(row) + "\n" for row in original
    )


def test_terminal_error_is_checkpointed_and_bounded(tmp_path):
    predictions_path = tmp_path / "predictions.jsonl"
    original = write_predictions(predictions_path, answers=("terminal", "good", "later", "later"))
    output_dir = tmp_path / "eval"
    calls = []

    def responder(messages):
        prompt = messages[-1]["content"]
        calls.append(prompt)
        if "Generated answer: terminal" in prompt:
            response = type("Response", (), {"status_code": 404})()
            raise requests.HTTPError("HTTP 404 model_not_found", response=response)
        return "correct"

    evaluator = eval_script.Evaluator(FakeJudge(responder=responder), predictions_path)
    with pytest.raises(requests.HTTPError, match="HTTP 404"):
        evaluator.evaluate(max_workers=2, output_dir=output_dir)

    assert len(calls) <= 2
    assert all("Generated answer: later" not in prompt for prompt in calls)
    assert not (output_dir / "predictions.jsonl").exists()
    assert not (output_dir / "predictions_eval_summary.json").exists()
    assert predictions_path.read_text(encoding="utf-8") == "".join(
        json.dumps(row) + "\n" for row in original
    )

    checkpoint_entries = [
        json.loads(line)
        for line in (output_dir / "predictions_judge_checkpoint.jsonl").read_text().splitlines()
    ]
    assert checkpoint_entries[0]["type"] == "manifest"
    completed_indexes = {entry["source_index"] for entry in checkpoint_entries[1:]}
    assert completed_indexes <= {1}
