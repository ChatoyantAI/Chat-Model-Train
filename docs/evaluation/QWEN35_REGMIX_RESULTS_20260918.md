# Qwen3.5-9B 与 regmix 微调版评测结果

候选：Qwen/Qwen3.5-9B；ChatoyantAI/qwen3.5-9b-roleplay-sft-regmix-v1。
同一冻结输入、GLM 5.3 Flash judge、BF16、候选温度 0.7、关闭候选 thinking。属于项目裁判协议，不等同于官方榜单。

## 完成度及费用

| 项目 | 原版 | regmix |
|---|---:|---:|
| CharacterBench | 6496/6497 | 6497/6497 |
| CoSER | 196/200 | 198/200 |
| 累计 API 费用 | $1.057636 | $1.073285 |

费用不含 GPU；两任务均为 finished_partial。原始 CharacterBench 有 3 条共同排除的无效输入。

## CharacterBench（归一到 5 分，维度宏平均）

| 语言 | 原版 | regmix | 差值 |
|---|---:|---:|---:|
| en | 3.2582 | 3.3681 | +0.1099 |
| zh | 3.5646 | 3.3754 | -0.1892 |

| 维度 | 英文原版 | 英文 regmix | 中文原版 | 中文 regmix |
|---|---:|---:|---:|---:|
| attribute_consistency_bot | 3.680 | 3.960 | 3.835 | 3.885 |
| attribute_consistency_human | 3.555 | 3.625 | 3.650 | 3.615 |
| behavior_consistency_bot | 2.240 | 2.985 | 2.695 | 2.535 |
| behavior_consistency_human | 2.633 | 2.653 | 3.273 | 2.487 |
| boundary_consistency | 3.573 | 3.660 | 3.813 | 3.580 |
| emotion_self_regulation | 3.300 | 3.375 | 3.800 | 3.670 |
| empathetic_responsiveness | 2.766 | 3.062 | 3.250 | 3.005 |
| engagement | 2.552 | 2.472 | 3.284 | 2.648 |
| fact_accuracy | 1.770 | 1.865 | 1.980 | 2.030 |
| human_likeness | 3.780 | 3.276 | 3.956 | 3.336 |
| memory_consistency | 3.290 | 3.635 | 3.605 | 3.860 |
| morality_robustness | 4.637 | 4.597 | 4.488 | 4.610 |
| morality_stability | 4.580 | 4.620 | 4.710 | 4.620 |

原版唯一缺失为 zh-morality_robustness-0193（CallFailed）；微调版该题为 2.5 分。
扣除该题后，共同中文样本的 regmix 宏平均约 3.3761，原版 3.5646，下降方向不变。

## CoSER（原始 schema 分数）

| 维度 | 原版有效 196 | regmix 有效 198 | 原版共同 194 | regmix 共同 194 |
|---|---:|---:|---:|---:|
| Storyline Consistency | 27.9898 | 23.2298 | 27.7242 | 22.9253 |
| Anthropomorphism | 23.2347 | 13.1313 | 23.0232 | 13.0979 |
| Character Fidelity | 25.1531 | 22.3308 | 24.7809 | 22.0954 |
| Storyline Quality | 34.1199 | 25.6414 | 33.9175 | 25.5670 |
| avg | 27.6244 | 21.0833 | 27.3615 | 20.9214 |
| bleu | 0.0059 | 0.0524 | 0.0058 | 0.0528 |
| rouge_l | 0.1626 | 0.1715 | 0.1622 | 0.1720 |

BLEU/ROUGE-L 仅表示词面重叠，不能替代角色表现评分。

## 缺失与核验

- 原版 CoSER 缺 case 0040、0066、0165、0166。
- regmix CoSER 缺 case 0071、0116。
- 汇总中的 pending 对应无成功结果的场景，逐场景状态文件记录为 failed。
- regmix case 0027 状态文件保留旧失败，但已有成功结果，核验后有效数量仍为 198。
- 已读取 CoSER 结果重新聚合，与最终 summary 一致；共同 194 场景平均为原版 27.3615、regmix 20.9214。
- 旧缓存、账本、日志均保留；原版历史损坏账本已从保存响应重建且与暂停快照费用一致。本次只读取结果，无新增模型调用。

## 解读

英文 CharacterBench 略有提升，中文及 CoSER 下降。当前结果不支持把此微调版视为原版的全面升级。
实验尚不能区分数据质量、混合比例和训练参数的影响；不据此直接归因。
