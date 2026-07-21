#!/usr/bin/env python3
"""Build a deterministic evidence/trajectory inventory for HotpotQA LLM-judge errors."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


CHUNK_ID_RE = re.compile(r"Chunk ID:\s*([^\s,(]+)")

FT4 = "4. 第二跳或后续跳检索失败"
FT8 = "8. 已经读取正确证据，但推理错误"
FT10 = "10. 达到最大轮次后被强制回答"
FT11 = "11. 最终答案表达或格式问题"
FT12 = "12. Gold answer、数据或评测问题"


def annotation(
    primary: str,
    bucket: str,
    analysis: str,
    path: str,
    fix: str,
    reasoning: str,
    secondary: list[str] | None = None,
    early_stopped: bool = False,
) -> dict[str, Any]:
    return {
        "primary": primary,
        "bucket": bucket,
        "analysis": analysis,
        "path": path,
        "fix": fix,
        "reasoning": reasoning,
        "secondary": secondary or [],
        "early_stopped": early_stopped,
    }


ANNOTATIONS = {
    "5ae163b3554299422ee99678": annotation(
        FT12, "gold_data_issue",
        "问题询问歌曲作者与写作地点，Gold 却是 1994 年电视回顾节目的说明句；问题、Gold 与 supporting chain 不一致。模型对歌曲作者的回答不可能匹配该 Gold。",
        "两条 supporting facts 均已读取 → 问题与 Gold 错位 → LLM judge 按 Gold 判错",
        "修正问题或 Gold；不应通过检索/提示词迎合错误标注。",
        "按当前 supporting chain，只能回答该节目是向 Eric Morecambe 致敬的三集回顾，以及其艺名来自 Morecambe；无法推出 Gold 所要求的歌曲作者与地点。",
    ),
    "5ae7793c554299540e5a55c2": annotation(
        FT12, "gold_data_issue",
        "Wet 'n Wild Orlando 被 Universal's Volcano Bay 取代；Krakatau 是 Volcano Bay 内 200 英尺高的主题火山，不是公园名。模型答案在语义上正确，Gold 粒度错误。",
        "正确两跳证据已读取 → 模型给出真实公园名 → Gold 错把园内火山当公园",
        "将 Gold 改为 Universal's Volcano Bay，或把问题改为园内火山叫什么。",
        "Wet 'n Wild Orlando 是美国首个水上公园 → 后继公园是 Universal's Volcano Bay → Krakatau 只是园内主题火山。",
    ),
    "5a8efb5a55429918e830d172": annotation(
        FT12, "gold_data_issue",
        "问题字面问 country，答案应为 India；Gold Jamnagar 是 Poonamben Maadam 的议会选区/地区。",
        "完整证据已读取 → 模型按问题类型回答国家 India → Gold 标成地区 Jamnagar",
        "把问题中的 country 改为 constituency/district，或把 Gold 改为 India。",
        "Gujarat Legislative Assembly 位于印度 → 若问国家则 India；Poonamben Maadam 的议会席位才是 Jamnagar。",
    ),
    "5ae234e85542994d89d5b395": annotation(
        FT12, "gold_data_issue",
        "Isabella II 是 Conrad IV 的母亲，不是妻子；模型给出的 Elisabeth of Bavaria 是其配偶。Gold 与问题中的 married to 关系相反。",
        "完整证据已读取 → 模型正确区分母子与婚姻关系 → Gold 关系标注错误",
        "将问题改为 Conrad IV 的母亲是谁，或将 Gold 改为 Elisabeth of Bavaria。",
        "Conrad IV 是 Frederick II 与 Isabella II 之子；其妻为 Elisabeth of Bavaria。",
    ),
    "5a7ba11d554299294a54aa39": annotation(
        FT12, "gold_data_issue",
        "Gold Søren Lindsted 是问题主体而非俱乐部；supporting fact 表明其生涯起点是 Holbæk Boldklub，而 Copenhagen 描述对应后来效力的 KB，问题内部也冲突。",
        "完整证据已读取 → 问题约束互相冲突且 Gold 类型错误 → 无唯一可满足答案",
        "若问生涯起点，Gold 改为 Holbæk Boldklub 并删去 Copenhagen；若问 Copenhagen 俱乐部，删除 started his career at。",
        "Søren Lindsted 起步于 Holbæk Boldklub；KB 是 Copenhagen 的体育俱乐部，但不是其生涯起点。",
    ),
    "5a7a5ec855429941d65f25e4": annotation(
        FT8, "evidence_sufficient_reasoning",
        "Agent 找到并读取 Sangin→Helmand River 与 Helmand River→Sistan Basin 两跳，却把中间实体 Helmand River 当成最终答案，未解析 'primary watershed for' 的宾语方向。",
        "正确证据已读取 → 多跳关系方向解析错误 → 回答中间实体",
        "回答前显式写出每跳 subject-relation-object，并检查最终问题槽位。",
        "Sangin 位于 Helmand River 河谷 → Helmand River 是 Sistan Basin 的主要流域 → 答 Sistan Basin。",
    ),
    "5a7b8f5e554299294a54a9f6": annotation(
        FT12, "gold_data_issue",
        "问题问 Richmond town 的 2000 人口，Gold 1,864 来自 Richmond CDP；模型给出的 3,284 是 town 的外部真实数值，但超出当前语料。",
        "town 与 CDP 实体混淆写入 Gold → Agent 发现冲突并检索至上限 → 按外部知识作答",
        "统一 town/CDP 实体；若保持当前 Gold，应把问题明确改成 census-designated place。",
        "2010 年 3,411 对应 Richmond town；当前 supporting chain 将其错误接到 2000 年人口 1,864 的 Richmond CDP。",
        secondary=[FT10],
    ),
    "5a7f62405542992e7d278cf0": annotation(
        FT12, "gold_data_issue",
        "Teriade 出版的是艺术杂志 Verve，James Joyce 只是其早期投稿者；supporting facts 没有说明 Teriade 出版 Finnegans Wake。模型拒绝臆造是合理的。",
        "supporting chain 仅靠人物共现错误拼接 → Gold Teriade 无证据 → 模型按语料拒答",
        "替换为真正含 Finnegans Wake 出版者的证据，或重写问题为谁出版 Verve。",
        "Verve 由 Teriade 出版且 Joyce 曾投稿，不蕴含 Teriade 出版 Joyce 的 Finnegans Wake。",
    ),
    "5ae207a65542994d89d5b317": annotation(
        FT12, "gold_data_issue",
        "Lavatera 约 25 种；Oplismenus 有 100 多个描述名但仅 7 个正式承认种。问题未说明比较 described 还是 recognized species，Gold 选前者、模型选后者。",
        "完整证据已读取 → 计数口径未定义 → Gold 与模型选择不同口径",
        "在问题中明确 officially recognized 或 described species。",
        "若按正式承认种：Lavatera 25 > Oplismenus 7；若按历史描述名：Oplismenus 100+ > Lavatera 25。",
    ),
    "5a72224755429971e9dc92be": annotation(
        FT12, "gold_data_issue",
        "Berenberg Bank 创立于 1590；Dain Rauscher Wessels 的源头在 1920/1930 年代。模型答案 Berenberg Bank 明确受证据支持，Gold 相反。",
        "正确日期已读取 → 正确比较 → Gold 反标",
        "把 Gold 改为 Berenberg Bank。",
        "1590 早于 1920/1930 年代，因此 Berenberg Bank 先成立。",
    ),
    "5a8b4d5e5542995d1e6f135c": annotation(
        FT12, "gold_data_issue",
        "证据只说明 Shimkus 最后代表第 20 区，之后重划到第 19 区，以及后来代表第 15 区；Gold 第 15 区不是 'before the 20th'。问题时间方向与证据不符。",
        "完整证据已读取 → 时间关系无法支持 Gold → 模型判断第 20 区是首个选区",
        "将 before 改为 later/after，或补充其第 20 区之前的真实任职证据。",
        "第 20 区撤销后 Shimkus 被划入第 19 区；第 15 区是另一后续任职，不能由证据推出为之前。",
    ),
    "5ade89d855429975fa854ef7": annotation(
        FT12, "gold_data_issue",
        "问题询问两位合演演员，模型给出 Song Kang-ho 与 Jung Woo-sung；Gold 却是奖项短语 Best Actor prize，且问题称五个典礼、证据仅列三个。",
        "演员阵容已读取 → 模型回答正确实体类型 → Gold 类型及数量陈述错误",
        "Gold 改为 Song Kang-ho and Jung Woo-sung，并修正 three/five。",
        "Lee Byung-hun 是目标演员；2008 年 The Good, the Bad, the Weird 的另外两位主演是 Song Kang-ho 和 Jung Woo-sung。",
    ),
    "5a82edae55429966c78a6a9f": annotation(
        FT8, "evidence_sufficient_reasoning",
        "supporting fact 明确写 international 1986 single release，但模型被专辑/早期发行的 1985 信息干扰，选择了错误年份。",
        "正确证据已读取 → 相邻年份冲突未按问题中的 single release 消歧 → 推理错误",
        "日期问题要求引用与目标事件同一句证据，并区分 album year 与 single release year。",
        "Double 的代表单曲 The Captain of Her Heart 在欧洲作为单曲于 1986 年发行。",
    ),
    "5ae0ac1c5542993d6555ec07": annotation(
        FT8, "evidence_sufficient_reasoning",
        "Gold 按职业标签数量选择 Gareth Evans（导演、编剧、剪辑、动作指导），模型扩展为主观的跨国经历/题材多样性并选择 Pierre Chenal。",
        "两人职业证据已读取 → 比较标准自行改写 → 选择错误实体",
        "比较题优先使用 supporting sentence 中可枚举的同维度属性，不引入主观职业影响力。",
        "Gareth Evans 有四类明确职业，Pierre Chenal 只有导演与编剧，因此按数据集口径选 Gareth Huw Evans。",
    ),
    "5abbc96755429931dba1452c": annotation(
        FT8, "evidence_sufficient_reasoning",
        "Chunk 489 已读且包含 Fortune 专辑 'selling an overall three million copies worldwide'，模型只关注歌曲局部句并错误声称语料没有销量。",
        "正确 Chunk 已读取 → 长 Chunk 中跨文档桥接句被忽略 → 达上限后拒答",
        "回答前在已读 Chunk 内对 gold-like 数量单位（copies/sold/million）做证据复查；缩短 Chunk 或结构化文档边界。",
        "Ain't Thinkin' 'Bout You 关联 Chris Brown → 其专辑 Fortune 的销量为全球三百万张 → 答 three million copies worldwide。",
        secondary=[FT10],
    ),
    "5a7e3b585542995ed0d166df": annotation(
        FT8, "evidence_sufficient_reasoning",
        "数据集实体是 2016 年 Pete's Dragon，模型擅自切换到同名 1977 原版，因此把发行先后颠倒。",
        "正确的 2004/2016 证据已读取 → 同名实体消歧被外部知识覆盖 → 推理错误",
        "比较前绑定 title+year，禁止用未检索到的同名作品替换当前实体。",
        "Home on the Range 为 2004 年；当前上下文的 Pete's Dragon 为 2016 年；故 Home on the Range 更早。",
    ),
    "5abd516a5542992ac4f3825c": annotation(
        FT12, "gold_data_issue",
        "Gold Hindi 是语言/电影产业，不是 religion；问题字段显然把 region/language 写成 religion。模型据姓名猜 Hindu 又违反了语料约束。",
        "问题类型错误 → Gold 不是宗教类别 → 模型无证据猜测 Hindu",
        "将 religion 改为 language/film industry；并在提示词中禁止按姓名推断宗教。",
        "Krrish 是印度首个科幻电影系列 → 作曲者 Rajesh Roshan 是 Hindi cinema music composer → 数据集期望 Hindi。",
        secondary=[FT8],
    ),
    "5ac2f1d1554299218029dba4": annotation(
        FT8, "evidence_sufficient_reasoning",
        "已读 Chunk 555 说明 Lithocarpus 是树类属、Duranta 含 shrubs and small trees；Agent 后续被无关 candicine Chunk 897 带偏，并虚构共同化学成分。",
        "正确证据已读取 → 后续语义检索引入无关化学实体 → 证据选择与推理错误",
        "设置证据充分即停止规则；候选答案必须在两个目标实体的支持句中分别出现。",
        "Lithocarpus 包含树；Duranta 的 17 个物种为灌木或小乔木；共同类别是 trees。",
        secondary=[FT10],
    ),
    "5a83aaeb5542996488c2e483": annotation(
        FT8, "evidence_sufficient_reasoning",
        "Power 的证据同时给出 Dwele 和 MBDTF，另一句给出该专辑由 Roc-A-Fella 发行；模型却凭 Kanye 专辑先验选 Graduation。",
        "两跳证据同 Chunk 已读取 → 未对 Dwele 与唱片公司做联合约束 → 推理错误",
        "最终候选必须同时通过所有问题约束（Dwele、Roc-A-Fella、studio album）验证。",
        "Dwele 为 Power 提供额外人声 → Power 收录于 My Beautiful Dark Twisted Fantasy → 该专辑由 Roc-A-Fella Records 发行。",
    ),
    "5a7302b75542991f9a20c5fd": annotation(
        FT8, "evidence_sufficient_reasoning",
        "Miles Doleac 支持句列出 Sleepy Hollow，下一支持句明确其为 supernatural horror film；模型改选另一部 comedy-horror Don't Kill It。",
        "正确证据已读取 → 目标类型和 supporting title 未绑定 → 选择无关候选",
        "对答案候选执行类型检查，并优先选择在两条 supporting facts 中形成直接连接的实体。",
        "Miles Doleac 有 Sleepy Hollow 演出经历 → Sleepy Hollow 是 supernatural horror film → 答 Sleepy Hollow。",
    ),
    "5a8fa4d95542995b4424207b": annotation(
        FT12, "gold_data_issue",
        "Trenton–Mercer 有 794,000 客流，Palm Springs 只描述季节性，没有同口径客流，supporting facts 不足以比较 busier；Gold 单方面选 Trenton。",
        "完整证据已读取 → 缺少一个机场的可比数值 → 模型合理拒绝比较",
        "补充 Palm Springs 同期客流 supporting fact，或改写问题为哪个机场的客流被量化为 794,000。",
        "当前证据仅能确认 Trenton–Mercer 的 2014-2015 客流约 794,000，无法严谨比较两机场。",
    ),
    "5ab82fe155429919ba4e225a": annotation(
        FT8, "evidence_sufficient_reasoning",
        "Chunk 731 首句直接说明 PPA 是 Italian Navy 的 new Frigate class，模型被其他海军舰艇 Anteo 干扰。",
        "正确直接证据已读取 → 无关候选覆盖精确匹配 → 推理错误",
        "对 'new' 等限定词做硬约束；直接支持句优先于宽泛搜索得到的同类实体。",
        "PPA（Pattugliatore Polivalente d'Altura）被定义为 Italian Navy 的新护卫舰级。",
    ),
    "5ae3d67f5542992f92d8238f": annotation(
        FT12, "answer_format_or_eval",
        "预测 Louisiana Tech University (Louisiana Tech Bulldogs) 与 Gold Louisiana Tech 是明确别名且无矛盾；LLM judge 假阴性。",
        "正确证据已读取 → 语义等价答案 → LLM judge 误判",
        "对短答案先做规范化/别名匹配，并对 LLM judge 假阴性复核。",
        "Derek Dooley 当前任 Cowboys 外接手教练；他曾执教 Louisiana Tech。",
    ),
    "5a729cc25542991f9a20c532": annotation(
        FT8, "evidence_sufficient_reasoning",
        "Agent 将 'wrote about' 解释为小说创作，选择 Susan Choi；数据集链条把 journalist Paul Avery 对 Hearst kidnapping 的报道工作视为 wrote about。",
        "Paul Avery supporting fact 已读取 → 关系表达歧义下选择了语料外另一作者 → 推理偏离数据集链",
        "桥接问题优先沿 supporting corpus 中的直接实体关系，不额外搜索更符合常识但不在链上的候选。",
        "Patty Hearst 被 SLA 绑架 → Paul Avery 从事该绑架案报道 → 数据集答案 Paul Avery。",
    ),
    "5ae78f3b554299540e5a5608": annotation(
        FT4, "agent_tool_decision",
        "第一跳 Chunk 789 已给出剧名 Supergirl，但 Agent 直接把该集所在季的 The CW 当成剧集 originally aired 网络，未以 Supergirl 发起第二跳。原 query 下正确 Chunk 790 排名 132；改写为 'Supergirl originally aired network' 时排名第 1。",
        "第一跳得到 Supergirl → 未生成实体化第二跳 query → Chunk 790 未返回/未读取 → 过早回答 The CW",
        "第一跳得到实体后强制建立未完成子问题清单，并查询 'Supergirl originally aired network'；或自动读相邻 Chunk 790。",
        "Mr. & Mrs. Mxyzptlk 属于 Supergirl → Supergirl 最初在 CBS 播出 → 答 CBS。",
        secondary=["2. 语义查询生成错误", "7. 正确证据位于相邻 Chunk", "9. 过早停止"],
        early_stopped=True,
    ),
    "5adbe75455429944faac23af": annotation(
        FT11, "answer_format_or_eval",
        "模型主答案写 Mercury Records，但证据链与 Gold 要求 Blackheart Records；括号里的 'via Blackheart Records' 不能消除主答案冲突。",
        "正确证据已读取 → 分销关系方向未锁定 → 主答案与附带答案冲突",
        "最终答案只输出一个通过问题主语/宾语方向验证的实体；此题按 Gold 输出 Blackheart Records。",
        "Joan Jett（Joan Marie Larkin）与 Kenny Laguna 的 Blackheart Records 充当独立厂牌分销出口；数据集答案 Blackheart Records。",
        secondary=[FT8],
    ),
    "5aded04755429975fa854fa7": annotation(
        FT12, "gold_data_issue",
        "2007 Patriots 是 franchise 第 48 季，但 Super Bowl XLII 的胜者是 New York Giants。问题中的 won 与 Gold Patriots 直接矛盾，模型答案事实正确。",
        "完整证据已读取 → 正确识别比赛胜者 Giants → Gold 错把参赛且处于第 48 季的 Patriots 当胜者",
        "把 won 改为 played/appeared，或把 Gold 改为 New York Giants 并删除第 48 季限定。",
        "Patriots 的第 48 季进入 Super Bowl XLII，但 Giants 赢得比赛。",
    ),
    "5a7306a655429901807daf67": annotation(
        FT12, "answer_format_or_eval",
        "预测完整包含 Gold 'Umaro Mokhtar Sissoco Embaló'，并只附无重音别名，无事实冲突；LLM judge 假阴性。",
        "正确证据已读取 → 完全匹配 Gold → LLM judge 误判",
        "增加确定性 exact/normalized containment 预判，命中完整 Gold 时跳过或覆盖 LLM judge。",
        "面积 36,125 km² 的西非国家是 Guinea-Bissau；其当时总理是 Umaro Mokhtar Sissoco Embaló。",
    ),
    "5adefdd75542995ec70e8f4e": annotation(
        FT8, "evidence_sufficient_reasoning",
        "问题问 Jack McKeon 管理过的球队（Marlins）总共赢过几次 World Series；模型缩窄为他亲自执教夺冠次数，因此答 1 而非球队史 2。",
        "完整证据已读取 → 统计对象从 team history 错换成 manager tenure → 推理错误",
        "数量题显式标注计数主体与时间范围，回答前复述 'team has won'。",
        "Jack McKeon 管理 Florida Marlins → Marlins 队史赢得两次 World Series → 答 two。",
    ),
    "5ae7e8b855429952e35ea9dd": annotation(
        FT11, "answer_format_or_eval",
        "模型把专辑同名曲 Let's Make Sure We Kiss Goodbye 放在主答案，虽随后列出 Gold Feels Like Love，但给出多个候选使主答案错误。",
        "正确证据已读取 → 三首 single 未按数据集目标消歧 → 多答案且主答案错误",
        "最终抽取层限制为单一短答案；若证据列多个候选，按问题对应 supporting answer 选择 Feels Like Love 或报告歧义。",
        "Vince Gill 是美国乡村歌手/词曲作者/多乐器演奏者 → 2000 MCA Nashville 专辑包含单曲 Feels Like Love → 数据集答案该曲。",
        secondary=[FT8],
    ),
    "5ae30aa05542992decbdcdd7": annotation(
        FT8, "evidence_sufficient_reasoning",
        "Ross Watson supporting sentence 明确说 Lead Developer for Dark Heresy 且参与 Black Crusade；模型转而选择其其他 lead developer 项目并给出多个答案。",
        "正确直接证据已读取 → 忽略同句精确关系 → 被后续无关履历干扰",
        "直接同句证据优先；候选必须同时满足 Black Crusade designer 与 lead developer 两个关系。",
        "Black Crusade 设计团队成员 Ross Watson → 他是 Dark Heresy 的 Lead Developer → 答 Dark Heresy。",
    ),
    "5a8f45385542997ba9cb320a": annotation(
        FT8, "evidence_sufficient_reasoning",
        "两条支持句都以 'is a genus of' 开头；模型选择了同样出现的生长型 herbaceous perennial，而问题明确问 categorization，Gold 要层级 genus。",
        "正确证据已读取 → 共同属性有多个 → 未按问题类型选择分类层级",
        "对 'type of categorization/taxonomic' 提示优先抽取 genus/species/family 等分类词。",
        "Dahlia 是一个 genus；Aruncus 也是一个 genus；共同植物分类为 genus。",
    ),
    "5a8de02b554299068b959e13": annotation(
        FT12, "gold_data_issue",
        "Gold Republic of Ireland national team 只来自另一条生涯事实，不是 Inter Milan 之后的俱乐部转会；问题的 after 关系不能由 supporting chain 推出。",
        "两条事实均已读取但不存在 after 连接 → 模型按足球生涯常识选择 Tottenham → Gold 使用无关国家队事实",
        "补充真实转会时间线并修正 Gold，或改问 Robbie Keane 后来担任哪支国家队队长。",
        "当前证据仅说明 Keane 曾在 Inter Milan 及后来担任爱尔兰国家队队长，不能证明紧接着效力该国家队。",
    ),
    "5a8baf635542996e8ac889c6": annotation(
        FT12, "gold_data_issue",
        "Mike Campbell 来自 Chegutu district；Harare 是 Rhodesia/Zimbabwe 的首都，不是该 district 的首府。Gold 把国家/政体首都接到 district capital。",
        "实体 Chegutu 已找到 → supporting chain 在行政层级上错误拼接 → 检索至上限仍无法支持 Gold",
        "将问题改为其所在国家的现首都，或提供 Chegutu district 行政中心并修正 Gold。",
        "Mike Campbell 来自 Zimbabwe 的 Chegutu district；Harare 是 Zimbabwe（原 Salisbury）的国家首都，不是由证据支持的 district capital。",
        secondary=[FT10],
    ),
    "5ae234005542994d89d5b392": annotation(
        FT12, "gold_data_issue",
        "'highest scope in profession' 没有定义比较规则；两人的职业标签高度重合，Gold James Dean Bradfield 缺乏可验证判据，模型用奖项/影响力另建标准。",
        "完整人物简介已读取 → 比较维度未定义 → Gold 与模型采用不同主观标准",
        "将问题改为可计数属性，例如 supporting sentence 中列出的职业数量，并明确计数规则。",
        "现有证据无法客观定义谁的 profession scope 更高。",
    ),
    "5a711cb45542994082a3e59f": annotation(
        FT12, "gold_data_issue",
        "supporting facts 给 Lisa Raymond 6 个女双大满贯，Liezel Huber 5 个；模型选择 Lisa 与证据一致，Gold Liezel Huber 反标。",
        "正确计数证据已读取 → 6 对 5 比较正确 → Gold 相反",
        "把 Gold 改为 Lisa Raymond。",
        "Lisa Raymond 女双 6 冠；Liezel Huber 女双 5 冠；因此 Lisa Raymond 更多。",
    ),
    "5ae1a8c05542997f29b3c0ec": annotation(
        FT12, "gold_data_issue",
        "Liberty Square 实际只在 Florida 的 Magic Kingdom；Anaheim Disneyland 有 Mark Twain Riverboat。问题用共享的 Rivers of America 错误桥接并制造不存在的 California Liberty Square，模型指出假前提。",
        "两条 supporting facts 已进入上下文 → 错误实体桥接 → 模型拒绝假前提但包含 Anaheim",
        "改问 Disneyland 的 Mark Twain Riverboat 位于哪座城市，或把 Liberty Square 地点 Gold 改为 Lake Buena Vista, Florida。",
        "Magic Kingdom Liberty Square 有 Liberty Belle；Disneyland 的 Mark Twain Riverboat 位于 Anaheim；两者不是同一园区。",
    ),
    "5ab9be4f554299753720f843": annotation(
        FT12, "gold_data_issue",
        "问题短语 'what land elevation ... have in common' 含义不明；Gold mountain 是地貌类别而非 elevation，模型按数值海拔解释为都超过 7,000m。",
        "两座山证据已读取 → 问题把 landform/elevation 混写 → 模型与 Gold 选择不同语义",
        "若期望 mountain，改问 what type of landform；若问海拔共同点，Gold 改为 over 7,000 metres 并补数值证据。",
        "Khunyang Chhish 与 Ismoil Somoni Peak 都是 mountain；它们也都超过 7,000 米，但当前 Gold 只接受前者。",
    ),
    "5a7a103b5542990783324e27": annotation(
        FT12, "gold_data_issue",
        "Gold Section.80 是 Good Kid, M.A.A.D City 的前作；问题中的 'album succeeded ...' 通常问后继者，模型回答 To Pimp a Butterfly。关系方向与 Gold 相反。",
        "正确专辑时间线已读取 → 按 succeeded 的通常语义选择后作 → Gold 标前作",
        "将 succeeded 改为 preceded，或将 Gold 改为 To Pimp a Butterfly 并补直接证据。",
        "Money Trees 属于 Good Kid, M.A.A.D City；Section.80 在它之前，To Pimp a Butterfly 在它之后。",
    ),
    "5ac3a47f554299391541386d": annotation(
        FT11, "answer_format_or_eval",
        "模型正文包含 Gold College Park，但把 Norwood 放在加粗主答案并列出多个 suburb；问题要求一个由第二跳唯一确认的 suburb。",
        "正确证据已读取 → 未执行唯一答案抽取 → 主答案与 Gold 不同且多答案",
        "最终答案先做第二跳唯一性筛选，只输出 College Park；将解释放在答案后且不再列候选。",
        "Dunstan 包含 College Park → College Park 位于 City of Norwood Payneham St Peters → 答 College Park。",
        secondary=[FT8],
    ),
    "5ae635505542992ae0d16282": annotation(
        FT12, "gold_data_issue",
        "Gold Rochdale 是 Jimmy Cricket 当前居住地，不是出生地；证据只称其为 Irish comedian，未给具体出生城市。模型用 Ireland 作出生地仍是推断，但比 Gold 关系更接近问题。",
        "完整证据已读取 → residence 被 Gold 当 birthplace → Agent 无出生地证据后用国籍推断",
        "将问题改为 currently lives where，或补充出生地证据；提示词禁止把国籍当出生地。",
        "And There's More 主演是 Jimmy Cricket；当前语料只说明他是 Irish comedian 且现居 Rochdale，不能确定出生地点。",
        secondary=[FT8],
    ),
    "5ac272f95542992f1f2b38d3": annotation(
        FT12, "gold_data_issue",
        "Couroupita 明确原生中美洲；Graptopetalum 原生 Mexico 和 Arizona。是否把 Mexico 算作 Central America 取决于地域口径，常见地理口径并不包括 Mexico，Gold yes 有歧义。",
        "完整原生地证据已读取 → 地域边界口径未定义 → 模型按常见口径答 no",
        "明确 Central America 的定义，或改问是否都原生于 Americas。",
        "Couroupita 原生中南美洲；Graptopetalum 原生墨西哥和亚利桑那；按通常中美洲定义答案为 no。",
    ),
    "5abd8ee05542993062266cc8": annotation(
        FT12, "answer_format_or_eval",
        "预测主答案与 Gold 都是 Mineola，解释与 supporting facts 完全一致且无矛盾；这是明确的 LLM judge 假阴性。",
        "全部证据已读取 → 精确答案 Mineola → LLM judge 误判",
        "在调用 LLM judge 前执行规范化 exact/containment；本例应直接判正确。",
        "Mineola 是 Nassau County 村庄，名称意为 pleasant place，位于 Kathleen Rice 代表的纽约第 4 国会选区。",
    ),
    "5a84e0a45542991dd0999e11": annotation(
        FT12, "gold_data_issue",
        "Steel Venom 在 Shakopee, Minnesota，Wicked Twister 在 Sandusky, Ohio；前者纬度更高。模型答案正确，Gold Wicked Twister 反标。",
        "两个地点已读取 → 地理比较正确 → Gold 相反",
        "把 Gold 改为 Steel Venom，并可加入纬度证据降低歧义。",
        "Shakopee 约北纬 44.8°，Sandusky 约北纬 41.4°，所以 Steel Venom 更北。",
    ),
    "5a84bda45542992a431d1a96": annotation(
        FT12, "gold_data_issue",
        "Fred MacMurray 是八月出生的演员，他只在电影中扮演 professor，并非教授；Gold 把角色职业当成演员职业。模型列出真实教授，说明问题关系存在歧义。",
        "supporting facts 已读取 → 角色与演员属性错误合并 → Gold Fred MacMurray 不满足字面问题",
        "改问 who was born in August and played a professor，或将 Gold 换为真正教授并提供对应证据。",
        "Fred MacMurray 八月出生并扮演 Professor Ned Brainard；他本人职业是演员。",
    ),
}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def canonical(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).lower()
    return "".join(character for character in text if character.isalnum())


def load_chunks(path: Path) -> dict[str, str]:
    raw = load_json(path)
    chunks: dict[str, str] = {}
    for item in raw:
        if isinstance(item, dict):
            chunks[str(item["id"])] = item["text"]
        else:
            chunk_id, text = item.split(":", 1)
            chunks[chunk_id] = text
    return chunks


def map_fact_to_chunks(fact: str, chunks_canonical: dict[str, str]) -> list[str]:
    needle = canonical(fact)
    return [chunk_id for chunk_id, text in chunks_canonical.items() if needle in text]


def trajectory_chunk_ids(entry: dict[str, Any]) -> list[str]:
    tool_name = entry.get("tool_name")
    arguments = entry.get("arguments", {})
    if tool_name == "read_chunk":
        return [str(value) for value in arguments.get("chunk_ids", [])]
    return CHUNK_ID_RE.findall(entry.get("tool_result", ""))


def tool_call_counts(row: dict[str, Any]) -> Counter[str]:
    return Counter(entry.get("tool_name", "unknown") for entry in row.get("trajectory", []))


def build_case(
    row: dict[str, Any], gold: dict[str, Any], chunks: dict[str, str], chunks_canonical: dict[str, str]
) -> dict[str, Any]:
    context = {title: sentences for title, sentences in gold["context"]}
    supporting_facts = []
    correct_chunk_ids: set[str] = set()
    for title, sentence_id in gold["supporting_facts"]:
        sentence = context[title][sentence_id]
        chunk_ids = map_fact_to_chunks(sentence, chunks_canonical)
        correct_chunk_ids.update(chunk_ids)
        supporting_facts.append(
            {
                "title": title,
                "sentence_id": sentence_id,
                "text": sentence,
                "chunk_ids": chunk_ids,
            }
        )

    trajectory = []
    keyword_chunk_ids: set[str] = set()
    semantic_chunk_ids: set[str] = set()
    read_chunk_ids: set[str] = set()
    evidence_in_search_snippet: set[str] = set()
    for entry in row.get("trajectory", []):
        chunk_ids = trajectory_chunk_ids(entry)
        tool_name = entry.get("tool_name")
        if tool_name == "keyword_search":
            keyword_chunk_ids.update(chunk_ids)
        elif tool_name == "semantic_search":
            semantic_chunk_ids.update(chunk_ids)
        elif tool_name == "read_chunk":
            read_chunk_ids.update(chunk_ids)

        result_canonical = canonical(entry.get("tool_result", ""))
        for fact in supporting_facts:
            fact_key = f"{fact['title']}#{fact['sentence_id']}"
            if canonical(fact["text"]) in result_canonical:
                evidence_in_search_snippet.add(fact_key)

        trajectory.append(
            {
                "loop": entry.get("loop"),
                "tool_name": tool_name,
                "arguments": entry.get("arguments", {}),
                "returned_or_read_chunk_ids": chunk_ids,
                "tool_result": entry.get("tool_result", ""),
                "retrieved_tokens": entry.get("retrieved_tokens", 0),
                "chunks_found": entry.get("chunks_found"),
            }
        )

    last_tool_loop = max((entry.get("loop", 0) for entry in row.get("trajectory", [])), default=0)
    max_loops_reached = row.get("loops") == 15 and last_tool_loop == 15
    correct_ids = sorted(correct_chunk_ids, key=lambda value: int(value))
    keyword_ids = sorted(keyword_chunk_ids, key=lambda value: int(value))
    semantic_ids = sorted(semantic_chunk_ids, key=lambda value: int(value))
    read_ids = sorted(read_chunk_ids, key=lambda value: int(value))
    searched_ids = sorted(keyword_chunk_ids | semantic_chunk_ids, key=lambda value: int(value))
    for fact in supporting_facts:
        fact_ids = set(fact["chunk_ids"])
        fact_key = f"{fact['title']}#{fact['sentence_id']}"
        fact["keyword_returned"] = bool(fact_ids & keyword_chunk_ids)
        fact["semantic_returned"] = bool(fact_ids & semantic_chunk_ids)
        fact["read"] = bool(fact_ids & read_chunk_ids)
        fact["in_search_snippet"] = fact_key in evidence_in_search_snippet
        fact["entered_llm_context"] = fact["read"] or fact["in_search_snippet"]

    correct_read = all(fact["read"] for fact in supporting_facts)
    correct_retrieved = all(
        fact["keyword_returned"] or fact["semantic_returned"] for fact in supporting_facts
    )
    correct_evidence_in_context = all(fact["entered_llm_context"] for fact in supporting_facts)

    annotation_data = ANNOTATIONS[row["qid"]]
    uncovered = [
        f"{fact['title']}#{fact['sentence_id']}"
        for fact in supporting_facts
        if not fact["entered_llm_context"]
    ]
    case = {
        "qid": row["qid"],
        "question": row["question"],
        "question_type": gold.get("type"),
        "level": gold.get("level"),
        "gold_answer": row["gold_answer"],
        "pred_answer": row["pred_answer"],
        "llm_accuracy": row["llm_accuracy"],
        "contain_accuracy_auxiliary": row.get("contain_accuracy"),
        "primary_failure_type": annotation_data["primary"],
        "secondary_failure_types": annotation_data["secondary"],
        "aggregate_bucket": annotation_data["bucket"],
        "supporting_facts": supporting_facts,
        "correct_chunk_ids": correct_ids,
        "keyword_returned_chunk_ids": keyword_ids,
        "semantic_returned_chunk_ids": semantic_ids,
        "searched_chunk_ids": searched_ids,
        "read_chunk_ids": read_ids,
        "correct_evidence_retrieved": correct_retrieved,
        "correct_evidence_read": correct_read,
        "correct_evidence_in_context": correct_evidence_in_context,
        "all_supporting_facts_keyword_returned": all(
            fact["keyword_returned"] for fact in supporting_facts
        ),
        "all_supporting_facts_semantic_returned": all(
            fact["semantic_returned"] for fact in supporting_facts
        ),
        "supporting_fact_coverage": {
            "total": len(supporting_facts),
            "retrieved": sum(
                fact["keyword_returned"] or fact["semantic_returned"] for fact in supporting_facts
            ),
            "read": sum(fact["read"] for fact in supporting_facts),
            "entered_llm_context": sum(fact["entered_llm_context"] for fact in supporting_facts),
        },
        "supporting_facts_in_search_snippets": sorted(evidence_in_search_snippet),
        "loops": row.get("loops", 0),
        "total_retrieved_tokens": row.get("total_retrieved_tokens", 0),
        "tool_call_counts": dict(tool_call_counts(row)),
        "keyword_queries": [
            {
                "loop": entry["loop"],
                "keywords": entry["arguments"].get("keywords", []),
                "top_k": entry["arguments"].get("top_k", 5),
                "returned_chunk_ids": entry["returned_or_read_chunk_ids"],
            }
            for entry in trajectory
            if entry["tool_name"] == "keyword_search"
        ],
        "semantic_queries": [
            {
                "loop": entry["loop"],
                "query": entry["arguments"].get("query", ""),
                "top_k": entry["arguments"].get("top_k", 5),
                "returned_chunk_ids": entry["returned_or_read_chunk_ids"],
            }
            for entry in trajectory
            if entry["tool_name"] == "semantic_search"
        ],
        "read_requests": [
            {
                "loop": entry["loop"],
                "chunk_ids": entry["arguments"].get("chunk_ids", []),
            }
            for entry in trajectory
            if entry["tool_name"] == "read_chunk"
        ],
        "max_loops_reached": max_loops_reached,
        "token_budget_triggered": False,
        "token_budget_inference": "No saved flag; <=23,148 retrieved tokens is far below 128,000.",
        "early_stopped": annotation_data["early_stopped"],
        "final_answer_produced_at": (
            "forced_after_loop_15" if max_loops_reached else f"loop_{row.get('loops', 0)}"
        ),
        "trajectory": trajectory,
        "analysis": annotation_data["analysis"],
        "failure_path": annotation_data["path"],
        "recommended_fix": annotation_data["fix"],
        "correct_reasoning_chain": annotation_data["reasoning"],
        "evidence_gap_notes": (
            "未进入上下文的 supporting facts: " + ", ".join(uncovered)
            if uncovered
            else "全部 supporting facts 已通过搜索摘要或 read_chunk 进入 LLM 上下文。"
        ),
    }
    return case


def summary(all_rows: list[dict[str, Any]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    correct = [row for row in all_rows if row.get("llm_accuracy") == 1]
    wrong = [row for row in all_rows if row.get("llm_accuracy") == 0]

    def avg(rows: list[dict[str, Any]], field: str) -> float:
        return mean(row.get(field, 0) for row in rows) if rows else 0.0

    wrong_counts = Counter()
    for row in wrong:
        wrong_counts.update(tool_call_counts(row))

    primary_counts = Counter(case["primary_failure_type"] for case in cases)
    bucket_counts = Counter(case["aggregate_bucket"] for case in cases)
    return {
        "error_definition": "llm_accuracy == 0 in eval/predictions.jsonl",
        "total_samples": len(all_rows),
        "error_samples": len(wrong),
        "correct_samples": len(correct),
        "correct_avg_loops": avg(correct, "loops"),
        "error_avg_loops": avg(wrong, "loops"),
        "correct_avg_retrieved_tokens": avg(correct, "total_retrieved_tokens"),
        "error_avg_retrieved_tokens": avg(wrong, "total_retrieved_tokens"),
        "error_avg_tool_calls": {
            tool: wrong_counts[tool] / len(wrong)
            for tool in ("keyword_search", "semantic_search", "read_chunk")
        },
        "error_max_loops_reached": sum(case["max_loops_reached"] for case in cases),
        "all_max_loops_reached": sum(
            row.get("loops") == 15
            and max((entry.get("loop", 0) for entry in row.get("trajectory", [])), default=0) == 15
            for row in all_rows
        ),
        "correct_max_loops_reached": sum(
            row.get("loops") == 15
            and max((entry.get("loop", 0) for entry in row.get("trajectory", [])), default=0) == 15
            for row in correct
        ),
        "error_early_stopped": sum(case["early_stopped"] for case in cases),
        "error_token_budget_triggered": sum(case["token_budget_triggered"] for case in cases),
        "error_correct_evidence_retrieved": sum(case["correct_evidence_retrieved"] for case in cases),
        "error_correct_evidence_read": sum(case["correct_evidence_read"] for case in cases),
        "error_correct_evidence_in_context": sum(case["correct_evidence_in_context"] for case in cases),
        "error_all_supporting_facts_keyword_returned": sum(
            case["all_supporting_facts_keyword_returned"] for case in cases
        ),
        "error_all_supporting_facts_semantic_returned": sum(
            case["all_supporting_facts_semantic_returned"] for case in cases
        ),
        "primary_failure_types": {
            failure_type: {
                "count": count,
                "pct_of_errors": count / len(cases),
                "pct_of_all_samples": count / len(all_rows),
            }
            for failure_type, count in primary_counts.most_common()
        },
        "aggregate_buckets": {
            bucket: {
                "count": count,
                "pct_of_errors": count / len(cases),
                "pct_of_all_samples": count / len(all_rows),
            }
            for bucket, count in bucket_counts.most_common()
        },
    }


def md_cell(value: Any, limit: int | None = None) -> str:
    text = str(value).replace("\n", " ").replace("|", "\\|")
    if limit and len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def yes_no(value: bool) -> str:
    return "是" if value else "否"


DETAIL_NOTES = {
    "5ae78f3b554299540e5a5608": {
        "deviation": "第 1 轮已经从 Chunk 789 得到中间实体 Supergirl；第 2 轮读完后直接回答 The CW，没有围绕 Supergirl 的 originally aired network 发起第二跳。",
        "why": "默认提示词只笼统要求多跳分解，没有未完成子问题状态或回答前证据检查；The CW 与 episode 在同一句出现，形成了强局部诱因。",
        "ideal": "搜 Mr. & Mrs. Mxyzptlk → 读 789 得到 Supergirl → semantic_search('Supergirl originally aired network') → 读 790 → CBS。",
    },
    "5a7a5ec855429941d65f25e4": {
        "deviation": "检索与阅读均正确，最终将第一跳中间实体 Helmand River 当作答案，漏掉关系宾语 Sistan Basin。",
        "why": "提示词没有要求用 subject-relation-object 复核最终答案槽位；搜索片段中 Helmand River 重复频率更高。",
        "ideal": "搜 Sangin → 读 279 得到 Helmand River → 搜/读 278 → 提取 'primary watershed for Sistan Basin' → 只答 Sistan Basin。",
    },
    "5abbc96755429931dba1452c": {
        "deviation": "Chunk 489 已包含 three million copies worldwide，但 Agent 在长 Chunk 中只抓住歌曲句，后续 15 轮反复检索并声称没有销量。",
        "why": "Chunk 混合多个文档且很长；搜索-阅读循环没有对已读证据做结构化复查，达到上限时强制回答也没有证据回看。",
        "ideal": "搜歌名 → 读 489 → 在同 Chunk 锁定 Chris Brown 与 Fortune 销量句 → 验证单位 copies worldwide → 三百万张。",
    },
    "5ae7e8b855429952e35ea9dd": {
        "deviation": "已读证据列出三首 singles，模型将同名曲置于加粗主答案，同时把正确 Feels Like Love 放在括号列表。",
        "why": "当前提示词鼓励解释与引用，但没有唯一短答案抽取步骤；LLM judge 将主答案冲突视为错误。",
        "ideal": "定位 Vince Gill → 读 855/856 → 在候选 singles 中选数据集目标 Feels Like Love → 最终答案只输出该曲名。",
    },
    "5abd8ee05542993062266cc8": {
        "deviation": "没有系统偏离：模型主答案就是 Mineola，且解释完整支持 Gold。错误发生在 LLM judge。",
        "why": "eval.py 对所有有答案样例直接调用非确定性语义 judge，没有先用规范化 exact/containment 兜底，也没有 judge 复核。",
        "ideal": "现有检索轨迹不变 → normalize(pred) 包含 normalize(gold) → 直接判正确，不调用 LLM judge。",
    },
    "5ae7793c554299540e5a55c2": {
        "deviation": "模型正确回答替代 Wet 'n Wild 的公园 Universal's Volcano Bay；Gold 却取了园内火山 Krakatau。",
        "why": "supporting chain/Gold 抽取粒度错误，任何忠实回答问题的 Agent 都会与 Gold 冲突；提示词不应纠正为错误事实。",
        "ideal": "搜 Wet 'n Wild → 读 112/113 → 回答 Universal's Volcano Bay；评测 Gold 同步修正。",
    },
}


def render_report(payload: dict[str, Any], path: Path) -> None:
    stats = payload["summary"]
    cases = payload["cases"]
    lines: list[str] = []
    lines.extend(
        [
            "# A-RAG HotpotQA failure analysis",
            "",
            "> 分析口径：只把 `results/hotpotqa/eval/predictions.jsonl` 中 `llm_accuracy == 0` 的样例定义为错误。`contain_accuracy` 与官方 EM/F1 仅用于辅助解释，不扩展错误集合。分析未修改核心逻辑、原始 predictions、chunks 或 embedding 索引。",
            "",
            "## 1. 实验产物定位",
            "",
            "| 产物 | 路径 | 说明 |",
            "|---|---|---|",
            "| 配置 | `configs/test_hotpotqa.yaml` | grok-4.5；Qwen3-Embedding-0.6B；max_loops=15；token budget=128000 |",
            "| 问题 | `data/hotpotqa/questions.json` | 1000 条 HotpotQA hard distractor 子集 |",
            "| 原始预测 | `results/hotpotqa/predictions.jsonl` | 当前最终版，1000 条，SHA256 `e36cbb...84e6d` |",
            "| 重试前预测 | `results/hotpotqa/predictions.pre-retry.jsonl` | 与最终版有 15 条记录不同，不作为本报告输入 |",
            "| LLM 评测样例 | `results/hotpotqa/eval/predictions.jsonl` | 带 `llm_accuracy`/`contain_accuracy`，本报告直接输入 |",
            "| LLM 评测汇总 | `results/hotpotqa/eval/predictions_eval_summary.json` | LLM accuracy 0.955，即 45 错 |",
            "| 官方评测 | `results/hotpotqa/official-eval/` | 官方 answer EM/F1；SP 为空，不能作为本次错误集合 |",
            "| Chunks | `data/hotpotqa/chunks.json` | 1311 个长 Chunk |",
            "| Embedding 索引 | `data/hotpotqa/index/sentence_index.pkl` | 49,843 句，1024 维，模型元数据为 Qwen/Qwen3-Embedding-0.6B |",
            "| 运行日志 | 项目内未找到 | 回收站中的 `original-main-*.log` 是次日 `full-adaptive`/memory 运行且中途终止，不能与当前预测可靠对应，故不作为证据 |",
            "",
            "评测脚本 `scripts/eval.py` 的主逻辑是把 pred/gold 交给 LLM judge，要求只返回 `correct` 或 `incorrect`。当前输出中 45 条 `llm_accuracy=0`、70 条 `contain_accuracy=0`；按用户要求，本报告只分析前 45 条。官方 EM 为 0 的主要原因是预测普遍带长解释，不代表 1000 条都在本次口径下错误。",
            "",
            "## 2. 核心统计",
            "",
            f"- 总样例：{stats['total_samples']}；LLM judge 错误：{stats['error_samples']}（{stats['error_samples']/stats['total_samples']:.1%}）。",
            f"- 正确样例平均 loops：{stats['correct_avg_loops']:.2f}；错误样例：{stats['error_avg_loops']:.2f}。",
            f"- 正确样例平均 retrieved tokens：{stats['correct_avg_retrieved_tokens']:.1f}；错误样例：{stats['error_avg_retrieved_tokens']:.1f}。",
            f"- 错误样例平均工具调用：keyword {stats['error_avg_tool_calls']['keyword_search']:.2f}，semantic {stats['error_avg_tool_calls']['semantic_search']:.2f}，read_chunk {stats['error_avg_tool_calls']['read_chunk']:.2f}。",
            f"- 错误样例中 max-loops 强制回答 {stats['error_max_loops_reached']} 条；全体共 {stats['all_max_loops_reached']} 条，其中正确 {stats['correct_max_loops_reached']} 条。",
            f"- 明确过早停止 {stats['error_early_stopped']} 条；token budget 触发 {stats['error_token_budget_triggered']} 条。",
            "",
            "### Primary cause",
            "",
            "| Primary cause | 数量 | 占 45 错误 | 占 1000 全集 |",
            "|---|---:|---:|---:|",
        ]
    )
    for failure_type, values in stats["primary_failure_types"].items():
        lines.append(
            f"| {failure_type} | {values['count']} | {values['pct_of_errors']:.1%} | {values['pct_of_all_samples']:.1%} |"
        )
    lines.extend(
        [
            "",
            "高层互斥归并：Gold/数据本身问题 25 条（55.6%）；证据充分后的推理错误 13 条（28.9%）；答案格式或 LLM judge 问题 6 条（13.3%）；Agent 工具决策错误 1 条（2.2%）。若按行为口径把唯一的后续跳失败也计作检索未完成，则检索错误 1 条；它不是 embedding 固有召回失败。",
            "",
            "## 3. 证据覆盖",
            "",
            f"- 45 条的所有官方 supporting facts 都能映射到当前 chunks，语料缺证据为 0 条。",
            f"- keyword_search 完整返回所有 supporting facts：{stats['error_all_supporting_facts_keyword_returned']}/45。",
            f"- semantic_search 完整返回所有 supporting facts：{stats['error_all_supporting_facts_semantic_returned']}/45。",
            f"- keyword 或 semantic 联合完整返回：{stats['error_correct_evidence_retrieved']}/45。",
            f"- 所有 supporting facts 都被 read_chunk 完整读取：{stats['error_correct_evidence_read']}/45。",
            f"- 计入搜索摘要后，所有 supporting facts 实际进入 LLM 上下文：{stats['error_correct_evidence_in_context']}/45。",
            "",
            "两个非完整读取样例：",
            "",
            "- `5ae78f...`：Chunk 789 已读，但 Supergirl 原始播出网络所在 Chunk 790 未返回、未读。这是唯一真实的证据缺口。原 query 下 790 排名 132；改为 `Supergirl originally aired network` 后排名第 1。失败路径：`第一跳得到 Supergirl → 未发起实体化第二跳 → 正确 Chunk 未进入 top_k → 过早回答`。",
            "- `5ae1a8...`：Mark Twain Riverboat 支持句所在 Chunk 1013 被搜索返回且完整句已进入摘要，但未 read_chunk；模型已获得 Anaheim 信息。此样例的主因是数据错误桥接 Magic Kingdom Liberty Square 与 Disneyland，而非读取不足。",
            "",
            "因此不存在 `正确 Chunk 已返回但 Agent 漏读并导致正常样例失败` 的独立 primary case，也没有证据表明相邻 Chunk 普遍遗漏。相邻读取只可能直接帮助 `5ae78f...` 这 1 条。",
            "",
            "## 4. 全部错误索引",
            "",
            "| qid | question | gold | predicted（首行） | primary | loops/tokens | K/S/R | 证据 R/Read/Ctx | max |",
            "|---|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for case in cases:
        calls = case["tool_call_counts"]
        coverage = case["supporting_fact_coverage"]
        lines.append(
            "| {qid} | {question} | {gold} | {pred} | {primary} | {loops}/{tokens} | {k}/{s}/{r} | {ret}/{read}/{ctx} | {maxed} |".format(
                qid=case["qid"],
                question=md_cell(case["question"], 90),
                gold=md_cell(case["gold_answer"], 48),
                pred=md_cell(case["pred_answer"].splitlines()[0], 70),
                primary=md_cell(case["primary_failure_type"]),
                loops=case["loops"],
                tokens=case["total_retrieved_tokens"],
                k=calls.get("keyword_search", 0),
                s=calls.get("semantic_search", 0),
                r=calls.get("read_chunk", 0),
                ret=coverage["retrieved"],
                read=coverage["read"],
                ctx=coverage["entered_llm_context"],
                maxed=yes_no(case["max_loops_reached"]),
            )
        )

    lines.extend(
        [
            "",
            "## 5. 典型案例复盘",
            "",
        ]
    )
    selected = [
        "5ae78f3b554299540e5a5608",
        "5a7a5ec855429941d65f25e4",
        "5abbc96755429931dba1452c",
        "5ae7e8b855429952e35ea9dd",
        "5abd8ee05542993062266cc8",
        "5ae7793c554299540e5a55c2",
    ]
    by_qid = {case["qid"]: case for case in cases}
    for number, qid in enumerate(selected, 1):
        case = by_qid[qid]
        note = DETAIL_NOTES[qid]
        chunks_text = "; ".join(
            f"{fact['title']}#{fact['sentence_id']} → {','.join(fact['chunk_ids'])}"
            for fact in case["supporting_facts"]
        )
        actual = " → ".join(
            f"L{entry['loop']} {entry['tool_name']}({','.join(entry['returned_or_read_chunk_ids']) or 'none'})"
            for entry in case["trajectory"]
        )
        lines.extend(
            [
                f"### {number}. `{qid}`",
                "",
                f"1. 原始问题：{case['question']}",
                f"2. 正确答案（当前 Gold）：{case['gold_answer']}",
                f"3. 模型答案：{case['pred_answer']}",
                f"4. 正确多跳链：{case['correct_reasoning_chain']}",
                f"5. 正确证据 Chunk：{chunks_text}",
                f"6. 实际轨迹：{actual}",
                f"7. 开始偏离：{note['deviation']}",
                f"8. 未被工具/提示词纠正的原因：{note['why']}",
                f"9. 最小修改：{case['recommended_fix']}",
                f"10. 理想轨迹：{note['ideal']}",
                "",
            ]
        )

    lines.extend(
        [
            "## 6. 主要瓶颈判断",
            "",
            "1. **主要不是检索器，而是数据质量与证据后的决策。** 25/45 是问题/Gold/supporting chain 本身错误或歧义；再有 13/45 是完整证据进入上下文后的推理错误。只有 1/45 因 Agent 未完成第二跳而缺证据。若排除 25 条数据问题，20 条可归因系统错误中，13 条（65%）是证据充分后的推理，6 条（30%）是答案表达/评测，1 条（5%）是工具决策。",
            "2. **Embedding 召回不是主要瓶颈。** semantic 单独完整覆盖 38/45，keyword+semantic 覆盖 44/45；唯一缺失样例通过正确 query 可把目标 Chunk 从第 132 提到第 1，说明瓶颈是 query/第二跳决策。",
            "3. **Keyword search 没有显示实体名称变化导致的系统性失败。** keyword 单独完整覆盖 44/45；唯一例子不是别名失败，而是 Agent 没有搜索已知实体 Supergirl。",
            "4. **top_k 不偏小。** 目标证据已联合进入 top_k 的样例为 44/45；唯一缺失 Chunk 排名 132，简单从 10 增至 20 无效。增大 top_k 主要增加噪声和 tokens。",
            "5. **max_loops 不是限制复杂样例的主因。** 45 错误中 4 条被强制回答，但四条早已读到证据或存在数据冲突；增加轮次预计直接修复 0 条。错误样例 loops 更高（7.96 vs 4.68）说明它们更多是在冲突/歧义中反复搜索。",
            "6. **存在 1 条明确过早停止。** `5ae78f...` 在得到 Supergirl 后没有完成网络子问题。其余错误不是证据不完整时提前回答。",
            "7. **read_chunk 总体不是瓶颈。** 43/45 完整读取所有 supporting facts；1 条未读但摘要已含证据且主因是数据，另 1 条是第二跳未检索。自动相邻读取预计只直接覆盖 1 条。",
            "8. **最终答案表达/评测造成 6 条明显失分。** 3 条是多个/冲突主答案（Blackheart、Feels Like Love、College Park），3 条是 LLM judge 明确假阴性（Louisiana Tech、Umaro Embaló、Mineola）。",
            "9. **存在增加轮次也难解决的推理错误。** 13 条在证据充分后仍发生关系方向、实体消歧、比较标准、计数主体或答案类型错误；其中 3 条还已运行到第 15 轮。",
            "",
            "## 7. 改进建议",
            "",
            "覆盖数是对当前 45 个错误的保守估计，不等同于新实验中的净提升；同一样例可能被多个方案覆盖。",
            "",
            "| 优先级/方案 | 针对问题 | 当前预计覆盖 | 需要修改 | 改论文方法 | 重建索引 | 建议消融 | 副作用 |",
            "|---|---|---:|---|---|---|---|---|",
            "| P0 数据/Gold 审计与隔离 | 25 条问题、Gold 或 supporting chain 错误/歧义 | 25（评测有效性，不是模型修复） | `data/hotpotqa/questions.json` 的来源转换/过滤脚本；评测清单 | 否 | 否 | 原始1000 vs 清洗子集；报告两套分数 | 与论文原始子集不可直接横比，需保留原始结果 |",
            "| P0 确定性答案规范化 + judge 复核 | 3 条明确 LLM judge 假阴性；别名、冠词、Unicode | 3 | `scripts/eval.py` | 否 | 否 | LLM-only vs exact/alias-first vs 双 judge | containment 过宽可能放过带矛盾答案，需仅对精确短答案启用 |",
            "| P0 单一短答案抽取 | 3 条多答案/主答案冲突 | 3，另可降低官方 EM 格式损失 | `src/arag/agent/prompts/default.txt`；必要时 `src/arag/agent/base.py` 增加抽取步 | 提示词版否；二次调用版是轻微扩展 | 否 | 长解释 vs `Final answer:` 单槽 vs 独立抽取器 | 可能丢失必要限定或多答案问题 |",
            "| P0 回答前多跳/约束完成检查 | 关系方向、计数主体、类型选择；1 条漏第二跳 | 约 7-10/14 | 首选 `default.txt`；强制状态机需 `base.py` | 仅提示词否；状态机是扩展 | 否 | checklist on/off；按 bridge/comparison 分层 | 增加 loops、成本，可能对简单题过度检索 |",
            "| P0 必须绑定支持结论的完整 Chunk/句 | 直接证据被无关候选覆盖、长 Chunk 内漏句 | 约 5/13 | `default.txt`；可在 `context.py` 保存证据引用 | 轻微扩展 | 否 | 引用验证 on/off；完整 Chunk vs supporting sentence 摘要 | 当前 Chunk 很长，强制重读会增 token；无法修复错误 Gold |",
            "| P1 低置信度答案验证一次 | 比较、日期、计数、同名实体、多个候选 | 约 5-8/14 | `base.py`、`default.txt`，可能增加 verifier prompt | 是，增加验证阶段 | 否 | 仅 comparison/多候选触发 vs 全量触发 | 成本和延迟增加；验证模型可能坚持原错误 |",
            "| P1 搜索无结果/未完成子问题自动改写 | 唯一第二跳失败；query 未包含 Supergirl | 1 | `base.py`/`default.txt`，或工具编排层 | 自动策略是扩展 | 否 | 原 query vs 实体化 rewrite；记录目标 Chunk rank | 可能造成查询扩散与额外调用 |",
            "| P1 自动读高分结果相邻 Chunk | Chunk 789/790 跨边界 | 1 | `read_chunk.py` 或 Agent 编排 | 是 | 否 | ±1 邻居 vs 不读；按 chunk 边界样例统计 | 每次读取大 Chunk，token 增长显著且噪声更多 |",
            "| P1 实体别名/规范化 | 当前未观察到 keyword 名称变化造成 primary failure | 0（本轮）；可能改善泛化 | `keyword_search.py`，别名字典/规范化层 | 是 | 否（若仅查询扩展）；索引化别名则是 | alias on/off；精确率与召回率 | 歧义实体误匹配、keyword top_k 被挤占 |",
            "| P2 增大 top_k | 正确证据排序低 | 0；唯一缺失 rank=132，10→20 无效 | 配置/提示词中的 top_k 默认；工具 schema | 否 | 否 | k=5/10/20；tokens、准确率、读取率 | 噪声与 token 成本上升，可能加剧错误候选覆盖 |",
            "| P2 调整 max_loops | 4 条达到上限，但均非轮次不足 | 预计 0 | `configs/test_hotpotqa.yaml` | 否 | 否 | 10/15/20；强制回答率与成本 | 反复无效检索更严重；错误样例本已多 3.27 loops |",
            "| P2 更换/调整 embedding | 语义召回 | 预计 0 个明确 primary case | 配置、`scripts/build_index.py`、重建 index | 否（仍是原接口） | 是 | Qwen3 当前模型 vs 候选；固定 query 的 supporting recall@k | 重建成本高；可能降低其他 38 条完整语义覆盖 |",
            "| P2 增加 reranker | top_k 噪声/排序 | 当前明确覆盖 0；可能间接改善证据选择 | semantic tool 后处理、新模型配置 | 是 | 通常否，可复用向量候选 | dense only vs +reranker；Recall@k/MRR 与端到端 | 延迟、模型依赖；正确证据本已大量在 top_k |",
            "",
            "推荐前三项实验顺序：先修复/隔离 25 条数据问题以建立可信基线；再做确定性答案规范化与单一短答案抽取（直接覆盖 6 条）；随后测试回答前多跳约束检查（目标是 14 条系统性推理/工具错误中的 7-10 条）。不要先换 embedding、增大 top_k 或增加 max_loops。",
            "",
            "## 8. 逐例证据与完整轨迹",
            "",
            "以下附录逐条给出 supporting facts→Chunk 映射、覆盖状态、每轮工具参数、返回 Chunk ID 和原始搜索片段/read_chunk 内容。`final_answer_produced_at=forced_after_loop_15` 表示第 15 轮仍调用工具后由限制提示强制作答。",
            "",
        ]
    )

    for case in cases:
        lines.extend(
            [
                f"### `{case['qid']}`",
                "",
                f"- Question: {case['question']}",
                f"- Gold: {case['gold_answer']}",
                f"- Predicted: {case['pred_answer']}",
                f"- Primary: {case['primary_failure_type']}",
                f"- Secondary: {', '.join(case['secondary_failure_types']) or '无'}",
                f"- Loops/tokens/final: {case['loops']} / {case['total_retrieved_tokens']} / {case['final_answer_produced_at']}",
                f"- max_loops/token_budget/early_stop: {yes_no(case['max_loops_reached'])} / {yes_no(case['token_budget_triggered'])} / {yes_no(case['early_stopped'])}",
                f"- Read Chunk IDs: {', '.join(case['read_chunk_ids']) or '无'}",
                f"- Failure path: `{case['failure_path']}`",
                f"- Analysis: {case['analysis']}",
                f"- Correct chain: {case['correct_reasoning_chain']}",
                f"- Recommended fix: {case['recommended_fix']}",
                "",
                "Supporting facts:",
                "",
            ]
        )
        for fact in case["supporting_facts"]:
            lines.append(
                "- `{title}#{sid}` → Chunk `{chunks}`; keyword={kw}, semantic={sem}, read={read}, context={ctx}: {text}".format(
                    title=fact["title"],
                    sid=fact["sentence_id"],
                    chunks=",".join(fact["chunk_ids"]),
                    kw=yes_no(fact["keyword_returned"]),
                    sem=yes_no(fact["semantic_returned"]),
                    read=yes_no(fact["read"]),
                    ctx=yes_no(fact["entered_llm_context"]),
                    text=fact["text"],
                )
            )
        lines.extend(["", "<details>", "<summary>完整 trajectory（工具参数、Chunk ID、返回片段）</summary>", ""])
        for entry in case["trajectory"]:
            lines.extend(
                [
                    f"**Loop {entry['loop']} · `{entry['tool_name']}`**",
                    "",
                    f"Arguments: `{json.dumps(entry['arguments'], ensure_ascii=False)}`",
                    "",
                    f"Returned/read Chunk IDs: `{','.join(entry['returned_or_read_chunk_ids']) or 'none'}`; retrieved tokens: {entry['retrieved_tokens']}",
                    "",
                    "```text",
                    entry["tool_result"],
                    "```",
                    "",
                ]
            )
        lines.extend(["</details>", ""])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    rows = load_jsonl(args.predictions)
    wrong = [row for row in rows if row.get("llm_accuracy") == 0]
    missing_annotations = {row["qid"] for row in wrong} - set(ANNOTATIONS)
    extra_annotations = set(ANNOTATIONS) - {row["qid"] for row in wrong}
    if missing_annotations or extra_annotations:
        raise ValueError(
            f"Annotation mismatch: missing={sorted(missing_annotations)}, extra={sorted(extra_annotations)}"
        )
    gold_by_id = {item["_id"]: item for item in load_json(args.gold)}
    chunks = load_chunks(args.chunks)
    chunks_canonical = {chunk_id: canonical(text) for chunk_id, text in chunks.items()}
    cases = [build_case(row, gold_by_id[row["qid"]], chunks, chunks_canonical) for row in wrong]

    payload = {"summary": summary(rows, cases), "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    if args.report:
        render_report(payload, args.report)

    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
