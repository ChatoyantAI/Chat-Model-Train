# 会话评分与角色配额 target 发布记录

记录 ID：`conversation-target-release-v1-20260906`。本记录描述已冻结数据版本和发布配置，不包含开发过程或原始聊天正文。

## 会话评分输入与协议

| 字段 | 值 |
|---|---|
| 来源 | `/home/shared/data/filter1/filter1_v1.parquet` |
| 来源 SHA-256 | `9dba235ba8afb95d529579f0483be655c547ec6c0ee0adbe32a8a9e3cec5fb89` |
| 粗筛候选 | 229,160 条 target，18,912 段会话 |
| 覆盖 | 2,759 个角色，13,760 个用户 |
| Judge | DeepInfra 直连 `zai-org/GLM-5.3-Flash` |
| 任务 | `d5b45395360be72a2b2249ad` |
| 协议 | `roleplay-conversation-filter-v1.0`；无 Gate；20 项连续 0/1/2/3 |
| 范围 | 完整历史中的粗筛候选 Assistant 回复组；其他历史仅作依据 |
| 并发 | 100 个滚动槽位 |
| 参数 | 不覆盖 temperature，不发 max_tokens、confidence 或 logprobs |

已完成结构化结果 18,912 段；其中 18,276 段进入数字均分排名，636 段疑似全量拒评单列。主任务记录费用约 `$34.393767`，另有 208 次未返回费用的尝试，不当作免费。数据和模型输出保留，不按排名删除原始记录。

冻结目录：`/home/shared/roleplay-conversation-filter/exports/d5b45395360be72a2b2249ad/`。

| 校验项 | SHA-256 |
|---|---|
| 会话数据版本 | `2f6b1193ff8cb667d638441195fcdb88e8083268082f4059f6163cad6d428950` |
| 会话量表 | `ac488d9ba7bc749e23320a414673d1454b29f503bb6d307184abfe8bc2515820` |
| 冻结 ranking.json | `0279bcbfb9d9764b54a4cf852f1c82a97032008b65c45d7e99aa93107d1d9772` |

## 训练包版本

发布目录：`/home/shared/data/roleplay_targets_top50000_cap200_v1_20260906/`。

按会话适用数字项均值降序、同分序号稳定排列，会话内按时间取 target；每角色最多 200 条，跨用户和语言累计。无额外低分或重大问题硬排除条件。配置为 `limit=50000, character_cap=200, shard_size=5000`。

| 指标 | 结果 |
|---|---:|
| target | 50,000 |
| 会话 | 4,186 |
| 角色 | 1,430 |
| 用户 | 3,474 |
| 分片 | 10 × 5,000 |
| 单角色最大条数 | 200 |
| 前 10 个角色占比 | 4.00% |
| 选择截止会话均分 | 2.1052631579 |
| 按 target 加权的会话均分 | 2.3255977140 |

语言分布：英文 25,815，西班牙文 13,918，葡萄牙文 3,557，法文 3,233，德文 2,025，繁体中文 1,452。字符长度与 token 长度分开，不用字符数估计训练器的实际截断长度。

发布包包含 `MANIFEST.json`、逐文件 `FILES.json`、训练分片、选择索引、完整会话评分审计、格式 / 分布文档与读取工具。构建时进行了全量回读，核对原始 target 正文、适用 System、严格前缀与分片大小。

原发布脚本快照由包内 `tools/` 保存；字节哈希为：

- builder：`8bf1b93071785013d1f83028d60b47b2eede18c939fd67e54ce85280bed886c7`
- selector：`e7ff86b26b299254d1c2737939e0bfbf6321c4f247f0e9ee18de202fdc361ced`

这些哈希指已发布包内的脚本，不宣称等于后续格式化或参数化后的仓库文件。新版脚本重建应写入新发布目录，新的哈希进入 manifest。

## 使用约束

- target 未逐条独立评分，`target_independently_scored=false`；对应会话有 0/1 项的 target 共 34,564 条，有重大问题标记的共 11,173 条，有复核标记的共 41,150 条。这些是会话级继承标记，不是逐条缺陷判定。
- `messages` 含适用 System、已有 Greeting 与当前 target 前的历史；`target` 单列，训练只计算 target loss。
- 不重新插入旧角色 prompt 或 User 尾部平台 guideline，不加入未来消息。
- 当前包未切分训练 / 验证，未启动训练；至少按会话分组划分，避免共享历史泄漏。
- 这是可追溯的排名选择包，不是“每条人工确认高质量”的金标集。

复现接口与字段见 [target 导出规范](../evaluation/ROLEPLAY_TARGET_EXPORT_V1.md)；评分口径见 [会话过滤规范](../evaluation/ROLEPLAY_CONVERSATION_FILTER_V1.md)。
