# 评估实现与规范

负责人：`@gaoxiaomo`。本目录记录现行协议、评分标准、输入输出和复现方式；不收录开发复盘、试跑日志或生产聊天正文。

| 功能 | 实现 / 规范 | 评分与使用边界 |
|---|---|---|
| Elo / Bradley–Terry | [实现](../../src/evaluation/elo_ranking/README.md)、[计算规范](../ELO_RANKING.md) | 线上匿名胜负票形成模型排名；不是单样本质量分 |
| 上下文重建 | [实现](../../src/evaluation/roleplay_judge_dataset/README.md)、[数据契约](ROLEPLAY_EVALUATION_DATA_FORMAT_V0_1.md) | 生成匿名单回复与独立解盲标签 |
| 人工 / LLM Judge | [平台](../../src/evaluation/roleplay_annotation_platform/README.md)、[完整量表](ROLEPLAY_EVALUATION_DIMENSIONS_V0_2.md) | `roleplay-rubric-v0.18`：8 Gate、20 项；原文件名保留兼容 |
| Judge Battle 拟合 | [计算规范](ROLEPLAY_BATTLE_WEIGHT_FITTING_V1.md) | `battle-weight-fit-v2`：非负原子权重、N/A、Gate、五折验证 |
| 单样本过滤 V2 | [评分与概率](ROLEPLAY_SINGLE_SAMPLE_FILTER_RUBRIC_V2.md)、[服务](ROLEPLAY_SINGLE_SAMPLE_FILTER_V2_DEPLOYMENT.md) | `roleplay-single-sample-filter-v2.2`；独立库与页面 |
| 会话过滤 | [工作台规范](ROLEPLAY_CONVERSATION_FILTER_V1.md)、[完整量表](ROLEPLAY_CONVERSATION_FILTER_RUBRIC_V1.md) | `roleplay-conversation-filter-v1.0`；无 Gate；候选回复组整体评分 |
| target 训练包 | [格式与选择](ROLEPLAY_TARGET_EXPORT_V1.md) | 角色配额、严格历史前缀、target-only loss；非逐条独立评分 |

完整量表保留各档定义、N/A 条件和案例。自动化测试检查源文、生成 JSON 与文档一致性；不同协议的均值、概率分、胜率不能混作同一种指标。

正式产物记录放在 `docs/experiments/`，只包含配置、聚合统计、版本 / 哈希和受控存储路径。大型数据、SQLite、完整请求与原始响应保存在忽略的 `outputs/` 或外部存储；临时脚本和旧草案分别位于忽略的 `tmp/`、`.local/evaluation/archive/`。
