# 评测说明

## 为什么先做评测
没有尺子之前的一切"优化"都无法证明确实更好——只会陷入换向量库/调prompt的无限调优，效果靠感觉。
所以 v0.6 的第一件事是造尺子：固定用例 + 固定指标 + 可复现报告。

## 怎么跑
在仓库根目录：
    ../agent-learning/.venv/Scripts/python.exe eval/run_eval.py --tag <版本标签>
    # 冒烟（只跑前 N 条）：--limit 5

报告输出到 docs/eval/<tag>.md。

## 指标口径
- 意图准确：IntentRouter 分类命中预期（用例 expected_intent=null 时不判）
- 工具选择：expected_tools=[] 时必须零调用；给列表时必须包含（子集匹配）；null 不判
- 转人工判定：reply.handoff == expected_handoff
- 检索命中：expected_chunk 出现在召回 Top3 的小节标题里
- 多轮用例：turns 列表中各轮判定全部通过才算该用例通过
- 系统表现：每例端到端延迟、模型调用轮数

## 用例格式
单轮：{"id","category","query","expected_intent","expected_tools","expected_handoff","expected_chunk","note"}
多轮：{"id","category","turns":[{"query","expected_tools","expected_handoff"}, ...],"note"}
（expected_* 为 null 表示该维度不判定；多轮用例共享一个 AgentSession，用于验证记忆与跨轮行为）

## 场景覆盖（7 类）
正常查询 / 信息缺失 / 歧义表达 / 越权请求 / 无关问题 / 需人工接手 / 多轮对话。
多轮类是记忆与跨轮 bug 的唯一裁判——v0.5 的多轮记忆、_seen_calls 跨轮误拦 bug（T03 为其永久回归测试）都只能在这里被验证。

## 纪律
1. 失败案例必须留在报告里，是回归集种子；改动后重跑同一套用例。
2. 每次只改一个主要因素，保留旧版报告做对照。
3. 数字不求漂亮，求可复现、可归因。
4. 单遍运行不等于结论：temperature=0.0 压住大部分方差，但 API 仍有非确定性——
   宣称"某改动有效"前，同一套集重跑一次，确认改善不是波动。
