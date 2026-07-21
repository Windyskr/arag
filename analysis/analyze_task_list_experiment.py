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
        "### 实验对象与对照条件",
        "",
        "本实验不是重新抽取一批问题，而是从原始 HotpotQA 评测输出中严格筛选 "
        "`llm_accuracy == 0` 的全部 45 条记录，并只重跑这 45 条。每条实验输出都保留相同的 "
        "`qid`、question 和 gold answer，原始 prediction 作为逐题基线。因此这里测量的是：在同一批"
        "已知错误样例上，仅加入 task-list 协议后，Agent 行为是否改善。",
        "",
        "除下述 task-list 机制外，实验继续使用 `configs/test_hotpotqa.yaml`：LLM 为 "
        "`grok-4.5`，temperature 为 0，embedding 为 `Qwen/Qwen3-Embedding-0.6B`，"
        "`max_loops=15`，token budget 为 128000。chunks、embedding 索引、keyword search、"
        "semantic search、read_chunk 的实现和工具 schema 均未修改，也没有重建 embedding 索引；"
        "工具允许的 `top_k` 范围或默认值也未改，但模型在具体轮次中选择的工具参数仍可能因新协议而变化。"
        "原始 A-RAG system prompt 被保留，只在其后追加 task-list 协议。",
        "",
        "### 优化 1：增加跨轮持久化的结构化 task state",
        "",
        "新增 `update_task_state` 工具，让模型不再只依赖对话中隐含的计划。状态保存在当前问题的 "
        "`AgentContext` 中，跨检索轮次持续存在，并在每个新问题开始时清空。其主要字段为：",
        "",
        "- `subtasks`：最少必要子问题；每项包含 id、问题、完成条件、状态、子答案、证据引文和 Chunk ID。",
        "- `expected_answer_type`：最终答案预期类型，例如人物、地点、网络、日期或数值。",
        "- `candidate_answer` 与 `candidate_evidence_chunk_ids`：当前唯一候选答案及其证据。",
        "- `unresolved_conflicts`：尚未解决的实体、时间、关系方向或证据冲突。",
        "- `next_action`：下一步计划调用的工具及 query，用于把状态更新和工具选择绑定起来。",
        "",
        "同时保存 task state 的更新历史、最终完成状态、未通过原因以及被门控拦截的回答，供逐轨迹分析。"
        "状态中只记录简短子问题和证据事实，不要求模型暴露完整 chain-of-thought。",
        "",
        "### 优化 2：在同一次模型响应中更新状态并决定下一工具",
        "",
        "第一轮要求模型先把问题拆成最少必要子问题，然后在同一 assistant response 中同时调用 "
        "`update_task_state` 和第一个搜索工具。收到一批工具结果后，下一次响应必须先依据新证据修订 "
        "task state，再在同一响应中调用 keyword_search、semantic_search 或 read_chunk。Agent 会按顺序"
        "执行该响应中的多个 tool calls，所以不需要为更新 task list 单独增加一个纯规划轮。最终轨迹中，"
        "257 次状态更新有 159 次与下一工具调用处于同一轮，覆盖 34/45 个样例。",
        "",
        "整体流程为：`拆分子问题 + 首次检索 → 阅读工具结果 → 更新完成状态 + 决定下一工具 → "
        "继续检索/读取 → 最终完成检查 → 回答`。",
        "",
        "### 优化 3：把“完成”定义为精确关系得到直接证据",
        "",
        "每个子任务必须提供 `completion_criterion`，用 subject-relation-object 形式明确需要证明的事实。"
        "只有已经通过 read_chunk 读取的 Chunk 直接陈述该关系时，子任务才能标记为 completed；仅命中"
        "相关实体、依赖常识推测或存在相似表述都不算完成。completed 子任务还必须保存最短直接支持句"
        "到 `evidence_quote`，并保存对应 `evidence_chunk_ids`。",
        "",
        "对于 bridge 类型多跳问题，第一跳得到桥接实体后，协议要求使用“桥接实体 + 第二跳的精确关系”"
        "继续搜索，且禁止拿第一跳证据替代第二跳证据。回答前还要求复核关系方向、答案类型、比较/计数"
        "范围、实体身份以及日期或版本。发现冲突时必须写入 `unresolved_conflicts`，不能直接结束。",
        "",
        "### 优化 4：增加 read_chunk 约束和程序化回答门控",
        "",
        "实验 Agent 启用 `require_task_completion=True`。当模型准备在无工具调用的响应中直接给出最终"
        "答案时，BaseAgent 会执行结构化完成检查。只有同时满足以下条件才允许回答：",
        "",
        "- task state 已初始化且至少包含一个子任务；",
        "- 所有子任务均为 completed；",
        "- 每个子任务都有完成条件、子答案、直接证据引文和证据 Chunk ID；",
        "- 子任务及最终候选答案引用的所有 Chunk ID 都已经由 read_chunk 实际读取；",
        "- `unresolved_conflicts` 为空；",
        "- 最终候选答案非空，并绑定至少一个已读取的证据 Chunk。",
        "",
        "如果检查失败，Agent 不接受该次答案：它会记录被拦截的 loop、原因和 proposed answer，向模型"
        "追加一条纠正消息，并在剩余 loop 内继续更新 task state 或检索。这是代码层面的门控，而不只是"
        "提示词建议。需要注意，该门控只能验证字段完整性和 Chunk 是否已读，不能自动判断引文是否真的"
        "蕴含 completion criterion；这也是实验中出现“结构完成但语义仍错”的主要边界。",
        "",
        "### 从第一版到最终测试版的调整",
        "",
        "最初 pilot 只加入普通 task list。在 Supergirl 样例中，模型仍把“The CW television series”"
        "错误当成“该剧最初播出的网络”，自行把任务标记完成并回答 The CW。最终用于 45 条错误样例的"
        "版本因此进一步加入：精确完成条件、直接 evidence quote、证据 Chunk 必须已读、桥接实体后的"
        "第二跳规则，以及回答前程序化门控。调整后该样例会继续查询 "
        "`Supergirl TV series originally aired on what network`，读取 Chunk 790，并回答 CBS。",
        "",
        "需要特别说明：最终 task-list prompt 还显式写入了 Supergirl 这个反例，即“The CW television "
        "series Supergirl”不能证明该剧最初在哪个网络播出。该规则来自对 pilot 失败轨迹的观察，所以 "
        "`5ae78f...` 应视为定向回归测试，能证明“精确第二跳关系”机制生效，但不能作为独立 held-out "
        "泛化证据。若排除这个定向样例，非定向的真实改善是其余 44 条中的 4 条；在其余 16 条可修复"
        "系统错误中为 4 条。",
        "",
        "### 输出与评测",
        "",
        "实验生成结果写入 `results/hotpotqa-task-list-v2/predictions.jsonl`，除原有 trajectory、loops、"
        "retrieved tokens 和工具日志外，还记录完整 task state、更新次数、完成原因、门控事件、"
        "max-loops 和 token-budget 状态。评测沿用原始 `scripts/eval.py` 的 LLM judge 提示词与判定"
        "逻辑，没有为 task-list 实验改变判分标准。自动评测之后再逐条人工复核，区分真实改善、judge "
        "假阳性以及原答案本已正确的假阴性。",
        "",
        "生成阶段使用 5 个 worker。评测第一次以 10 个 worker 运行时遇到上游 502，未采用该次不完整"
        "输出；随后对全部 45 条以 3 个 worker 重新评测并成功完成。并发数只影响请求调度，不改变 judge "
        "prompt 或判分逻辑。本实验每个条件只运行一次，没有做多随机种子重复；即使 temperature 为 0，"
        "远端模型和 LLM judge 仍不保证完全确定性。另外，由于样本是按原始错误筛选的，本结果只说明"
        "“已知错误集的修复率”，不能直接外推为全部 1000 条 HotpotQA 的总体准确率提升。",
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
