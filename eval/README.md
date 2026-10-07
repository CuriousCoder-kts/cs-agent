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
- 系统表现：每例端到端延迟、模型调用轮数

## 场景覆盖（6 类）
正常查询 / 信息缺失 / 歧义表达 / 越权请求 / 无关问题 / 需人工接手。

## 纪律
1. 失败案例必须留在报告里，是回归集种子；改动后重跑同一套用例。
2. 每次只改一个主要因素，保留旧版报告做对照。
3. 数字不求漂亮，求可复现、可归因。
