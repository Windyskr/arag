#!/usr/bin/env python3
"""Compare the task-list rerun with the original 45 HotpotQA LLM-judge failures."""

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = ROOT / "analysis/hotpotqa_failure_cases.json"
EXPERIMENT_PATH = ROOT / "results/hotpotqa-task-list-v2/eval/predictions.jsonl"
OUTPUT_JSON = ROOT / "analysis/task_list_experiment_comparison.json"
OUTPUT_REPORT = ROOT / "analysis/task_list_experiment_report.md"

GENUINE_IMPROVEMENTS = {
    "5ae0ac1c5542993d6555ec07": "职业维度比较由 Pierre Chenal 修正为 Gareth Huw Evans。",
    "5ab82fe155429919ba4e225a": "从无关舰艇 Anteo 修正为直接证据支持的 PPA。",
    "5ae78f3b554299540e5a5608": "完成 Supergirl 的第二跳并读取 Chunk 790，由 The CW 修正为 CBS。",
    "5ae30aa05542992decbdcdd7": "从其他履历候选修正为直接证据中的 Dark Heresy。",
    "5ac3a47f554299391541386d": "主答案由 Norwood 修正为第二跳唯一确认的 College Park。",
}

JUDGE_FALSE_POSITIVES = {
    "5a7a5ec855429941d65f25e4": "仍回答 Helmand River，Gold 是 Sistan Basin。",
    "5ae234e85542994d89d5b395": "仍回答 Elisabeth of Bavaria，Gold 是 Isabella II。",
    "5ab9be4f554299753720f843": "仍回答 seven-thousanders，Gold 是 mountain。",
}

BASELINE_EVAL_ONLY_ERRORS = {
    "5ae3d67f5542992f92d8238f": "Louisiana Tech 原答案已与 Gold 等价；本次 judge 改判正确。",
    "5a7306a655429901807daf67": "Umaro Mokhtar Sissoco Embaló 仍是精确答案，但 judge 继续误判。",
    "5abd8ee05542993062266cc8": "Mineola 仍是精确答案，但 judge 继续误判。",
}


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def first_line(text):
    return " ".join(text.split())[:300].replace("|", "\\|")


def audit_label(qid, baseline, experiment):
    if qid in GENUINE_IMPROVEMENTS:
        return "genuine_improvement", GENUINE_IMPROVEMENTS[qid]
    if qid in JUDGE_FALSE_POSITIVES:
        return "judge_false_positive", JUDGE_FALSE_POSITIVES[qid]
    if qid in BASELINE_EVAL_ONLY_ERRORS:
        return "baseline_answer_already_correct", BASELINE_EVAL_ONLY_ERRORS[qid]
    if baseline["aggregate_bucket"] == "gold_data_issue":
        return "gold_data_issue_persists", "问题、Gold 或 supporting chain 问题仍然存在。"
    if experiment["llm_accuracy"] == 1:
        return "needs_review", "Judge 判为正确，但不在预先审计集合中。"
    return "system_error_persists", "Task list 执行后仍未得到当前 Gold 所需答案。"


def main():
    with BASELINE_PATH.open(encoding="utf-8") as handle:
        baseline_payload = json.load(handle)
    baseline = {case["qid"]: case for case in baseline_payload["cases"]}
    experiments = load_jsonl(EXPERIMENT_PATH)
    assert len(experiments) == 45
    assert set(baseline) == {row["qid"] for row in experiments}

    tool_counts = Counter()
    paired_state_updates = 0
    samples_with_paired_updates = set()
    comparisons = []
    for row in experiments:
        base = baseline[row["qid"]]
        label, note = audit_label(row["qid"], base, row)
        tool_counts.update(entry["tool_name"] for entry in row["trajectory"])
        tools_by_loop = defaultdict(list)
        for entry in row["trajectory"]:
            tools_by_loop[entry["loop"]].append(entry["tool_name"])
        for tool_names in tools_by_loop.values():
            update_count = tool_names.count("update_task_state")
            if update_count and any(name != "update_task_state" for name in tool_names):
                paired_state_updates += update_count
                samples_with_paired_updates.add(row["qid"])
        comparisons.append({
            "qid": row["qid"],
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "baseline_answer": base["pred_answer"],
            "task_list_answer": row["pred_answer"],
            "baseline_primary_failure_type": base["primary_failure_type"],
            "baseline_aggregate_bucket": base["aggregate_bucket"],
            "task_list_llm_accuracy": row["llm_accuracy"],
            "task_list_contain_accuracy": row["contain_accuracy"],
            "manual_outcome": label,
            "manual_note": note,
            "baseline_loops": base["loops"],
            "task_list_loops": row["loops"],
            "baseline_retrieved_tokens": base["total_retrieved_tokens"],
            "task_list_retrieved_tokens": row["total_retrieved_tokens"],
            "task_state_updates": row["task_state_updates"],
            "task_state_complete": row["task_state_complete"],
            "max_loops_exceeded": row["max_loops_exceeded"],
            "guard_events": row["task_completion_guard_events"],
            "final_task_state": row["task_state"],
        })

    measured_correct = sum(row["llm_accuracy"] for row in experiments)
    outcome_counts = Counter(row["manual_outcome"] for row in comparisons)
    by_bucket = defaultdict(lambda: {"n": 0, "judge_correct": 0, "genuine": 0})
    for comparison in comparisons:
        bucket = by_bucket[comparison["baseline_aggregate_bucket"]]
        bucket["n"] += 1
        bucket["judge_correct"] += int(comparison["task_list_llm_accuracy"])
        bucket["genuine"] += int(comparison["manual_outcome"] == "genuine_improvement")

    summary = {
        "sample_count": len(experiments),
        "measured_llm_accuracy": measured_correct / len(experiments),
        "measured_llm_correct": int(measured_correct),
        "manual_genuine_improvements": len(GENUINE_IMPROVEMENTS),
        "manual_genuine_improvement_rate_all_errors": len(GENUINE_IMPROVEMENTS) / 45,
        "manual_genuine_improvement_rate_actionable_system_errors": (
            len(GENUINE_IMPROVEMENTS) / 17
        ),
        "judge_false_positives": len(JUDGE_FALSE_POSITIVES),
        "baseline_eval_only_errors": len(BASELINE_EVAL_ONLY_ERRORS),
        "task_state_complete": sum(row["task_state_complete"] for row in experiments),
        "max_loops_exceeded": sum(row["max_loops_exceeded"] for row in experiments),
        "guard_event_count": sum(
            len(row["task_completion_guard_events"]) for row in experiments
        ),
        "avg_task_state_updates": sum(row["task_state_updates"] for row in experiments) / 45,
        "paired_state_updates": paired_state_updates,
        "samples_with_paired_updates": len(samples_with_paired_updates),
        "baseline_avg_loops": sum(case["loops"] for case in baseline.values()) / 45,
        "task_list_avg_loops": sum(row["loops"] for row in experiments) / 45,
        "baseline_avg_retrieved_tokens": sum(
            case["total_retrieved_tokens"] for case in baseline.values()
        ) / 45,
        "task_list_avg_retrieved_tokens": sum(
            row["total_retrieved_tokens"] for row in experiments
        ) / 45,
        "task_list_avg_tool_calls": {
            name: count / 45 for name, count in sorted(tool_counts.items())
        },
        "manual_outcomes": dict(outcome_counts),
        "by_baseline_bucket": dict(by_bucket),
    }
    payload = {"summary": summary, "comparisons": comparisons}
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# HotpotQA 错误样例 Persistent Task List 实验",
        "",
        "## 实验设置",
        "",
        "- 输入：原始评测中 `llm_accuracy == 0` 的全部 45 条记录。",
        "- 模型与检索配置：保持 `configs/test_hotpotqa.yaml` 不变。",
        "- 实验处理：持久化 task state、精确完成条件、直接证据引用、",
        "  已读 Chunk 约束和回答前完成状态门控。",
        "- 输出：`results/hotpotqa-task-list-v2/predictions.jsonl`。",
        "- 评测：沿用 `scripts/eval.py` 的 LLM judge 提示词和判定逻辑。",
        "",
        "## 核心结果",
        "",
        f"自动评测判对 **{summary['measured_llm_correct']}/45 "
        f"({summary['measured_llm_accuracy']:.1%})**。逐条复核答案和轨迹后，确认真正由 task list "
        f"带来的行为改善为 **{summary['manual_genuine_improvements']}/45 "
        f"({summary['manual_genuine_improvement_rate_all_errors']:.1%})**。排除 25 条数据/Gold 问题和 "
        "3 条原答案本已正确的 judge 假阴性后，在 17 条可修复系统错误中改善 **5/17 (29.4%)**。",
        "",
        "自动评测的 9 条与真实改善的 5 条不能混用：",
        "",
        "- 5 条因检索、推理或最终答案选择变化而真正修正。",
        "- 1 条原答案本已正确（`Louisiana Tech`），本次被 judge 接受。",
        "- 3 条是 judge 假阳性，答案仍不符合 Gold 所要求的关系。",
        "- 另两条本已正确的答案（`Umaro Embaló`、`Mineola`）仍被 judge 错判。",
        "",
        "## 真实改善样例",
        "",
        "| qid | 原始失败类型 | 改善 |",
        "|---|---|---|",
    ]
    for qid, note in GENUINE_IMPROVEMENTS.items():
        lines.append(f"| `{qid}` | {baseline[qid]['primary_failure_type']} | {note} |")

    lines.extend([
        "",
        "按可修复失败类型统计：证据充分后的推理错误 **3/13**，第二跳/工具决策错误 "
        "**1/1**，最终答案选择或格式错误 **1/3**。",
        "",
        "## 协议运行表现",
        "",
        f"- 最终 task state 在结构上完整：{summary['task_state_complete']}/45。",
        f"- 达到 max loops：{summary['max_loops_exceeded']}/45。",
        f"- 服务端门控拦截过早回答：{summary['guard_event_count']} 次。",
        f"- 每题平均更新 task state：{summary['avg_task_state_updates']:.2f} 次。",
        f"- 同一响应中“更新状态 + 调用下一工具”：{summary['paired_state_updates']}/257 次状态更新，"
        f"覆盖 {summary['samples_with_paired_updates']}/45 个样例。",
        f"- 平均 loops：{summary['baseline_avg_loops']:.2f} → {summary['task_list_avg_loops']:.2f}。",
        f"- 平均 retrieved tokens：{summary['baseline_avg_retrieved_tokens']:.1f} → "
        f"{summary['task_list_avg_retrieved_tokens']:.1f}。",
        "",
        "task list 完成不等于答案正确：31 条通过了结构门控，其中 24 条仍被判错。模型可能自行定义了错误的"
        "完成条件，也可能把直接引文绑定到错误关系。Sangin 样例最典型：task state 在结构上完整，"
        "但最终仍选择中间实体 Helmand River，而不是 Sistan Basin。",
        "",
        "## 结论",
        "",
        "task list 有效，但作用范围小于“解决全部多跳失败”。它对遗漏第二跳、或直接支持句与干扰候选竞争"
        "的样例最有效；它不能稳定修复问题解析、关系方向、错误 Gold，或模型自行批准但语义无效的证据。",
        "",
        "关键正例是 `5ae78f...`：仅加入普通 task list 时仍回答 The CW；加入精确完成条件和证据引用后，"
        "模型查询 `Supergirl TV series originally aired on what network`，读取 Chunk 790，最终回答 CBS。",
        "",
        "## 逐例对比",
        "",
        "| qid | judge | 人工结论 | 原答案 | task-list 答案 | loops | 完成/max |",
        "|---|---:|---|---|---|---:|---|",
    ])
    for row in comparisons:
        lines.append(
            f"| `{row['qid']}` | {int(row['task_list_llm_accuracy'])} | "
            f"{row['manual_outcome']} | {first_line(row['baseline_answer'])} | "
            f"{first_line(row['task_list_answer'])} | "
            f"{row['baseline_loops']}→{row['task_list_loops']} | "
            f"{row['task_state_complete']}/{row['max_loops_exceeded']} |"
        )

    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
