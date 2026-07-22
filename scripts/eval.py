#!/usr/bin/env python3
"""Evaluate ARAG predictions with a resumable LLM judge.

Usage:
    python scripts/eval.py \
        --predictions results/predictions.jsonl \
        --output results/eval \
        --workers 10
"""

import argparse
import hashlib
import json
import logging
import os
import re
import string
import tempfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from tqdm import tqdm

from arag import LLMClient

logger = logging.getLogger(__name__)

CHECKPOINT_SCHEMA_VERSION = 1
JUDGE_SYSTEM_PROMPT = "You are an expert evaluator."
JUDGE_USER_PROMPT = """Please evaluate if the generated answer is correct by comparing it with the gold answer.
Generated answer: {pred_answer}
Gold answer: {gold_answer}

The generated answer should be considered correct if it:
1. Contains the key information from the gold answer
2. Is factually accurate and consistent with the gold answer
3. Does not contain any contradicting information

Respond with ONLY 'correct' or 'incorrect'.
Response:"""
JUDGE_PROMPT_FINGERPRINT = hashlib.sha256(
    (JUDGE_SYSTEM_PROMPT + "\n" + JUDGE_USER_PROMPT).encode("utf-8")
).hexdigest()


def normalize_answer(s):
    """Normalize an answer for containment comparison."""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_number(record, key, converter=float):
    try:
        return converter(record.get(key, 0) or 0)
    except (TypeError, ValueError):
        return converter(0)


class Evaluator:
    """Evaluate predictions and persist each LLM judgment in a checkpoint."""

    def __init__(self, llm_client, predictions_path):
        self.llm_client = llm_client
        self.predictions_path = Path(predictions_path).expanduser().resolve()
        self.prediction_results = self.load_predictions()

    def load_predictions(self):
        """Load prediction objects from JSON or JSONL."""
        if self.predictions_path.suffix == ".jsonl":
            with self.predictions_path.open("r", encoding="utf-8") as handle:
                prediction_results = [
                    json.loads(line) for line in handle if line.strip()
                ]
        else:
            with self.predictions_path.open("r", encoding="utf-8") as handle:
                prediction_results = json.load(handle)

        if not isinstance(prediction_results, list):
            raise ValueError("Predictions must be a JSON array or JSONL records.")
        for index, prediction in enumerate(prediction_results):
            if not isinstance(prediction, dict):
                raise ValueError(
                    "Prediction record at index {} must be a JSON object, got {}.".format(
                        index, type(prediction).__name__
                    )
                )
        return prediction_results

    def calculate_llm_accuracy(self, pred_answer, gold_answer):
        """Use the configured LLM judge to assess one answer."""
        user_prompt = JUDGE_USER_PROMPT.format(
            pred_answer=pred_answer,
            gold_answer=gold_answer,
        )
        response, _ = self.llm_client.generate(
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        return 1.0 if response.strip().lower() == "correct" else 0.0

    def calculate_contain(self, pred_answer, gold_answer):
        """Check whether the normalized gold answer occurs in the prediction."""
        if not pred_answer or not gold_answer:
            return 0.0
        return float(normalize_answer(gold_answer) in normalize_answer(pred_answer))

    def evaluate_single(self, idx, prediction):
        """Evaluate one prediction without writing shared state."""
        pred_answer = prediction.get("pred_answer", "")
        gold_answer = prediction.get("gold_answer") or prediction.get("answer", "")

        if not isinstance(pred_answer, str) or not pred_answer.strip():
            return idx, 0.0, 0.0, "failed"

        llm_acc = self.calculate_llm_accuracy(pred_answer, gold_answer)
        contain_acc = self.calculate_contain(pred_answer, gold_answer)
        return idx, llm_acc, contain_acc, "answered"

    def _artifact_paths(self, output_dir):
        output_path = (
            Path(output_dir).expanduser().resolve()
            if output_dir
            else self.predictions_path.parent / "eval"
        )
        output_predictions_path = (output_path / self.predictions_path.name).resolve()
        if output_predictions_path == self.predictions_path:
            raise ValueError(
                "Evaluation output would overwrite raw predictions. "
                "Choose an output directory other than {}.".format(
                    self.predictions_path.parent
                )
            )

        stem = self.predictions_path.stem
        return {
            "output_dir": output_path,
            "predictions": output_predictions_path,
            "summary": output_path / "{}_eval_summary.json".format(stem),
            "checkpoint": output_path / "{}_judge_checkpoint.jsonl".format(stem),
        }

    def _provenance(self):
        provenance = {
            "source_sha256": _sha256_file(self.predictions_path),
            "source_record_count": len(self.prediction_results),
            "source_format": self.predictions_path.suffix.lower(),
            "judge_model": str(getattr(self.llm_client, "model", "unknown")),
            "judge_base_url": str(getattr(self.llm_client, "base_url", "")).rstrip("/"),
            "temperature": 0.0,
            "judge_prompt_sha256": JUDGE_PROMPT_FINGERPRINT,
        }
        return provenance, _sha256_text(_canonical_json(provenance))

    def _record_hashes(self):
        return [_sha256_text(_canonical_json(record)) for record in self.prediction_results]

    def _write_checkpoint_line(self, handle, record):
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    def _load_checkpoint(self, checkpoint_path, provenance, fingerprint, record_hashes):
        """Validate and load a matching checkpoint keyed by source index."""
        if not checkpoint_path.exists():
            return {}

        with checkpoint_path.open("r", encoding="utf-8") as handle:
            lines = [(line_number, line.strip()) for line_number, line in enumerate(handle, 1)]

        nonempty_lines = [(number, line) for number, line in lines if line]
        if not nonempty_lines:
            raise ValueError("Checkpoint {} is empty or malformed.".format(checkpoint_path))

        try:
            manifest = json.loads(nonempty_lines[0][1])
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Checkpoint {} has invalid manifest JSON: {}".format(checkpoint_path, exc)
            ) from exc

        if (
            manifest.get("type") != "manifest"
            or manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or manifest.get("provenance") != provenance
            or manifest.get("provenance_fingerprint") != fingerprint
        ):
            raise ValueError(
                "Checkpoint provenance does not match this input or judge configuration. "
                "Use a fresh output directory or resume with the original model, base URL, "
                "and prediction file."
            )

        completed = {}
        for line_number, line in nonempty_lines[1:]:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Checkpoint {} has invalid JSON on line {}: {}".format(
                        checkpoint_path, line_number, exc
                    )
                ) from exc

            index = entry.get("source_index")
            if (
                entry.get("type") != "result"
                or entry.get("provenance_fingerprint") != fingerprint
                or type(index) is not int
                or index < 0
                or index >= len(self.prediction_results)
                or entry.get("source_record_sha256") != record_hashes[index]
                or index in completed
                or entry.get("status") not in {"answered", "failed"}
            ):
                raise ValueError(
                    "Checkpoint {} has an invalid result entry on line {}.".format(
                        checkpoint_path, line_number
                    )
                )

            try:
                completed[index] = (
                    float(entry["llm_accuracy"]),
                    float(entry["contain_accuracy"]),
                    entry["status"],
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Checkpoint {} has invalid scores on line {}.".format(
                        checkpoint_path, line_number
                    )
                ) from exc

        return completed

    def _apply_result(self, index, llm_accuracy, contain_accuracy, status):
        prediction = self.prediction_results[index]
        prediction["llm_accuracy"] = llm_accuracy
        prediction["contain_accuracy"] = contain_accuracy
        prediction["status"] = status

    def _checkpoint_entry(
        self,
        index,
        llm_accuracy,
        contain_accuracy,
        status,
        fingerprint,
        record_hashes,
    ):
        return {
            "type": "result",
            "source_index": index,
            "source_qid": self.prediction_results[index].get("qid"),
            "source_record_sha256": record_hashes[index],
            "provenance_fingerprint": fingerprint,
            "llm_accuracy": llm_accuracy,
            "contain_accuracy": contain_accuracy,
            "status": status,
        }

    def _atomic_write(self, path, write_contents):
        path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temp_path = tempfile.mkstemp(
            prefix=".{}-".format(path.name),
            dir=path.parent,
            text=True,
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                write_contents(handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        except Exception:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            raise

    def _write_final_artifacts(self, paths, provenance, fingerprint, resumed_records, new_records):
        total_samples = len(self.prediction_results)
        statuses = [record["status"] for record in self.prediction_results]
        llm_scores = [float(record["llm_accuracy"]) for record in self.prediction_results]
        contain_scores = [float(record["contain_accuracy"]) for record in self.prediction_results]
        answered_samples = sum(status == "answered" for status in statuses)
        failed_samples = sum(status == "failed" for status in statuses)
        answer_rate = answered_samples / total_samples if total_samples else 0.0
        llm_accuracy = sum(llm_scores) / answered_samples if answered_samples else 0.0
        contain_accuracy = (
            sum(contain_scores) / answered_samples if answered_samples else 0.0
        )

        total_cost = sum(_safe_number(record, "total_cost") for record in self.prediction_results)
        total_retrieved_tokens = sum(
            _safe_number(record, "total_retrieved_tokens", int)
            for record in self.prediction_results
        )
        total_loops = sum(
            _safe_number(record, "loops", int) for record in self.prediction_results
        )
        avg_cost = total_cost / total_samples if total_samples else 0.0
        avg_retrieved_tokens = (
            total_retrieved_tokens / total_samples if total_samples else 0.0
        )
        avg_loops = total_loops / total_samples if total_samples else 0.0

        def write_predictions(handle):
            if self.predictions_path.suffix == ".jsonl":
                for prediction in self.prediction_results:
                    handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            else:
                json.dump(self.prediction_results, handle, ensure_ascii=False, indent=4)

        summary = {
            "total_samples": total_samples,
            "answered_samples": answered_samples,
            "failed_samples": failed_samples,
            "answer_rate": round(answer_rate, 4),
            "llm_accuracy": llm_accuracy,
            "contain_accuracy": contain_accuracy,
            "correct_by_llm": int(sum(llm_scores)),
            "correct_by_contain": int(sum(contain_scores)),
            "total_cost": round(total_cost, 6),
            "avg_cost_per_question": round(avg_cost, 6),
            "total_retrieved_tokens": int(total_retrieved_tokens),
            "avg_retrieved_tokens": round(avg_retrieved_tokens, 1),
            "total_loops": total_loops,
            "avg_loops": round(avg_loops, 2),
            "judge_provenance": provenance,
            "judge_provenance_fingerprint": fingerprint,
            "judge_checkpoint": str(paths["checkpoint"]),
            "resumed_records": resumed_records,
            "newly_judged_records": new_records,
        }

        self._atomic_write(paths["predictions"], write_predictions)
        self._atomic_write(
            paths["summary"],
            lambda handle: json.dump(summary, handle, ensure_ascii=False, indent=4),
        )
        logger.info("Evaluation Results:")
        logger.info("  Total Samples: %s", total_samples)
        logger.info("  Answered: %s (%.2f%%)", answered_samples, answer_rate * 100)
        logger.info("  Failed: %s", failed_samples)
        logger.info("  LLM Accuracy: %.4f", llm_accuracy)
        logger.info("  Contain Accuracy: %.4f", contain_accuracy)
        logger.info("  Total Cost: $%.4f", total_cost)
        logger.info("  Avg Loops: %.2f", avg_loops)
        logger.info("Summary saved to: %s", paths["summary"])
        return llm_accuracy, contain_accuracy

    def evaluate(self, max_workers, output_dir=None, resume=False):
        """Run a bounded-concurrency, checkpointed evaluation."""
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1.")

        paths = self._artifact_paths(output_dir)
        provenance, fingerprint = self._provenance()
        record_hashes = self._record_hashes()
        checkpoint_path = paths["checkpoint"]
        checkpoint_exists = checkpoint_path.exists()
        if checkpoint_exists and not resume:
            raise ValueError(
                "Checkpoint {} already exists. Re-run with --resume or choose a fresh "
                "output directory.".format(checkpoint_path)
            )

        completed = (
            self._load_checkpoint(checkpoint_path, provenance, fingerprint, record_hashes)
            if checkpoint_exists
            else {}
        )
        for index, (llm_accuracy, contain_accuracy, status) in completed.items():
            self._apply_result(index, llm_accuracy, contain_accuracy, status)

        paths["output_dir"].mkdir(parents=True, exist_ok=True)
        with checkpoint_path.open("a", encoding="utf-8") as checkpoint_handle:
            if not checkpoint_exists:
                self._write_checkpoint_line(
                    checkpoint_handle,
                    {
                        "type": "manifest",
                        "schema_version": CHECKPOINT_SCHEMA_VERSION,
                        "source_path": str(self.predictions_path),
                        "provenance": provenance,
                        "provenance_fingerprint": fingerprint,
                    },
                )

            pending_indexes = iter(
                index
                for index in range(len(self.prediction_results))
                if index not in completed
            )
            newly_judged_records = 0
            progress = tqdm(
                total=len(self.prediction_results),
                initial=len(completed),
                desc="Evaluating",
                unit="sample",
            )
            executor = ThreadPoolExecutor(max_workers=max_workers)
            futures = {}

            def submit_until_full():
                while len(futures) < max_workers:
                    try:
                        index = next(pending_indexes)
                    except StopIteration:
                        return
                    futures[executor.submit(self.evaluate_single, index, self.prediction_results[index])] = index

            def checkpoint_success(future):
                nonlocal newly_judged_records
                index = futures[future]
                result_index, llm_accuracy, contain_accuracy, status = future.result()
                if result_index != index:
                    raise RuntimeError(
                        "Judge returned source index {} for scheduled index {}.".format(
                            result_index, index
                        )
                    )
                self._write_checkpoint_line(
                    checkpoint_handle,
                    self._checkpoint_entry(
                        index,
                        llm_accuracy,
                        contain_accuracy,
                        status,
                        fingerprint,
                        record_hashes,
                    ),
                )
                self._apply_result(index, llm_accuracy, contain_accuracy, status)
                futures.pop(future)
                newly_judged_records += 1
                progress.update(1)

            try:
                submit_until_full()
                terminal_error = None
                terminal_index = None
                while futures and terminal_error is None:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        try:
                            checkpoint_success(future)
                        except Exception as exc:  # Preserve terminal provider errors.
                            terminal_error = exc
                            terminal_index = futures.pop(future, None)
                            break
                    if terminal_error is None:
                        submit_until_full()

                if terminal_error is not None:
                    logger.error(
                        "Judge failed for source index %s; no additional requests will be "
                        "scheduled: %s",
                        terminal_index,
                        terminal_error,
                    )
                    for future in futures:
                        future.cancel()
                    while futures:
                        done, _ = wait(futures, return_when=FIRST_COMPLETED)
                        for future in done:
                            if future.cancelled():
                                futures.pop(future)
                                continue
                            try:
                                checkpoint_success(future)
                            except Exception as drain_error:
                                logger.error(
                                    "Additional in-flight judge request failed during drain: %s",
                                    drain_error,
                                )
                                futures.pop(future, None)
                    raise terminal_error
            finally:
                progress.close()
                executor.shutdown(wait=True, cancel_futures=True)

        if any("status" not in prediction for prediction in self.prediction_results):
            raise RuntimeError("Evaluation finished without a judgment for every prediction.")

        return self._write_final_artifacts(
            paths,
            provenance,
            fingerprint,
            resumed_records=len(completed),
            new_records=newly_judged_records,
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate ARAG predictions")
    parser.add_argument(
        "--predictions", "-p", required=True, help="Predictions file path (.json or .jsonl)"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=10, help="Maximum in-flight judge requests"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Evaluation output directory (default: <predictions-parent>/eval)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("ARAG_JUDGE_MODEL", "gpt-5.4-mini"),
        help="LLM judge model (default: ARAG_JUDGE_MODEL or gpt-5.4-mini)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("ARAG_JUDGE_BASE_URL") or os.getenv("ARAG_BASE_URL"),
        help="LLM judge API base URL (default: ARAG_JUDGE_BASE_URL or ARAG_BASE_URL)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only a matching provenance-validated judge checkpoint.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    print("\n{}\nARAG Evaluation\n{}".format("=" * 60, "=" * 60))
    print("Predictions: {}".format(args.predictions))
    print("Workers: {}".format(args.workers))
    print("Judge model: {}".format(args.model))
    print("Output: {}".format(args.output or "<predictions-parent>/eval"))
    print("{}\n".format("=" * 60))

    llm_client = LLMClient(
        model=args.model,
        api_key=os.getenv("ARAG_API_KEY"),
        base_url=args.base_url or "https://api.openai.com/v1",
    )
    evaluator = Evaluator(llm_client, args.predictions)
    llm_acc, contain_acc = evaluator.evaluate(
        max_workers=args.workers,
        output_dir=args.output,
        resume=args.resume,
    )

    print("\n{}\nResults\n{}".format("=" * 60, "=" * 60))
    print("LLM Accuracy: {:.4f}".format(llm_acc))
    print("Contain Accuracy: {:.4f}".format(contain_acc))
    print("{}\n".format("=" * 60))


if __name__ == "__main__":
    main()
