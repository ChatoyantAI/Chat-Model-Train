# Evaluation

> 状态：目录与契约已建立，具体实现待模块负责人提交
>
> 主负责人：待认领
>
> 备份 / Reviewer：待认领

## 目标

在冻结、可版本化的协议上比较模型，产出可复核的逐样本结果、聚合指标和误差分析，防止训练与正式评测之间的数据泄漏。

## 负责范围

- benchmark 样本 ID、版本、协议和冻结策略；
- 候选模型推理、解码参数、重试和响应校验；
- 确定性指标、LLM Judge、人工抽检和一致性分析；
- 逐样本 ledger、聚合报告、回归门槛和可视化；
- 评测成本、随机种子、模型版本和失败原因记录。

## 不负责

- 根据正式测试样本修改训练数据或提示词；
- 无版本地覆盖 benchmark、Judge prompt 或历史报告；
- 只保留汇总分数而丢弃可审计的逐样本结果。

## 每次正式评测必须记录

- benchmark、prompt、Judge 和指标版本；
- 候选模型精确版本及推理参数；
- 代码 commit、配置、随机种子和运行时间；
- 有效/失败样本数、失败原因和重试策略；
- 逐样本结果与汇总报告的不可变 URI/hash；
- 与基线的差异、置信区间或适用的统计说明。

## 建议布局

### 公开角色扮演基准

新增 [Role-play Benchmark Suite](roleplay_benchmarks/README.md)：固定 CoSER、CharacterBench、RMTBench、MiniMax 官方来源；通过脚本下载与 SHA-256 校验，不提交大型数据或 Token。执行包装与完整评测的验证状态在模块 README 中分别列出。

```text
src/evaluation/
├── README.md
├── benchmarks/
├── inference/
├── metrics/
├── judges/
└── reports/
```

实现提交时，负责人必须补充最小 smoke test、正式运行命令和回归门槛。
