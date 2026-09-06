# Evaluation

> 状态：Elo、上下文重建、人工 / LLM Judge、Battle 对齐、单样本及会话过滤实现可用
>
> 主负责人：@gaoxiaomo
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

## 已有实现

- [elo_ranking](elo_ranking/README.md)：将线上匿名二选一胜负票批量拟合为无位置项的 Bradley–Terry 分数、排名和置信区间。
- [roleplay_judge_dataset](roleplay_judge_dataset/README.md)：按采集期生产逻辑从冻结 ELO 数据高保真重建候选上下文，并生成匿名单回复 Judge 数据。
- [roleplay_annotation_platform](roleplay_annotation_platform/README.md)：人工标注、投票回放、批量 LLM Judge、多模型一致性、Judge Battle 与可解释权重拟合。
- 同包独立服务：单样本过滤 V2（概率评分）与会话过滤（无 Gate、候选回复组评分）；包含排名、角色配额与 target 训练包导出。

## 标准、配置与验证

- [协议与完整量表索引](../../docs/evaluation/README.md)
- [服务配置模板](../../configs/evaluation/README.md)
- [Elo 冻结记录](../../docs/experiments/2026-08-27-data-check-elo-active-models-v2.md)
- [会话筛选与训练包记录](../../docs/experiments/2026-09-06-conversation-target-release-v1.md)

```bash
python -m pip install -r src/evaluation/roleplay_annotation_platform/requirements.txt
python -m unittest discover -s tests/evaluation -v
```

测试使用虚构数据与 mock，不发送付费请求。原始数据、SQLite、请求 / 响应、临时分析与日志不进入 Git；版本化代码和规则留在本目录，运行产物留在 `outputs/` 或受控外部存储。
