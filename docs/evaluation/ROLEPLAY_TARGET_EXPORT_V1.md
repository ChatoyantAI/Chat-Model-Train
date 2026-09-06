# 排名到 target 训练包 V1

Schema：`roleplay-target-messages-v1`。实现：`select_ranked_conversation_targets.py`、`export_target_training_bundle.py`。这些工具只读取已完成评分与源数据，不调用 LLM，不修改原文或启动训练。

## 排名与角色配额

1. 读取冻结会话排名，校验任务状态、数据 / 规则哈希与源 target 文件哈希。
2. 按会话适用数字项的精确均值降序；同分按会话序号稳定排列。
3. 在会话内按时间顺序取粗筛候选 target。按 `character_id` 累计额度，跨用户和语言共享同一角色上限。
4. 达到角色上限后跳过该角色后续候选，继续取其他角色，直到满足目标条数。不足时失败，不悄悄降低限制。
5. 配额或总数边界允许仅取一个会话的前若干候选，记录 `partial_conversation_selection`。

默认 50,000 条、每角色最多 200 条、每分片 5,000 条。无效结果、缺少数字分、疑似全量拒评不排名；低分、重大问题和复核标记保留在元数据，不额外作为这个排名选择器的硬过滤条件。

同一会话的 target 共享会话组评分。`target_independently_scored=false`，不得解释成每条回复独立拿到了该分数。

## 命令与目录约定

`--root` 为会话服务运行根目录，包含 `data/conversations-v1-platform.sqlite3`、`results/jobs.sqlite3` 及 `exports/JOB_ID/` 冻结目录。源 target 路径从索引 manifest 读取。

```bash
python -m src.evaluation.roleplay_annotation_platform.select_ranked_conversation_targets --root outputs/conversation-filter --job-id JOB_ID --limit 50000 --character-cap 200 --output outputs/target-selection-v1
python -m src.evaluation.roleplay_annotation_platform.export_target_training_bundle --root outputs/conversation-filter --job-id JOB_ID --limit 50000 --character-cap 200 --shard-size 5000 --output outputs/target-training-v1
```

第二条命令会包含选择审计，无需先执行第一条。可选 `--previous-ranking PATH/target_ranking.jsonl.gz` 仅用于计算与某次历史导出的 ID 重合量；未指定时不读取其他运行目录，比较输入哈希纳入构建签名。

相同输入重跑会校验既有产物；配置或源哈希不同则拒绝覆盖，必须指定新版本目录。

## 逐行格式

```json
{
  "schema_version": "roleplay-target-messages-v1",
  "sample_id": "target-001",
  "messages": [
    {"role": "system", "content": "Frozen platform system prompt."},
    {"role": "assistant", "content": "Welcome aboard."},
    {"role": "user", "content": "Show me the chart."}
  ],
  "target": "Here is the chart.",
  "metadata": {"target_independently_scored": false}
}
```

上例为格式示意，实际 metadata 含源 ID、角色 / 用户键、语言、System / 输入 / target 哈希、会话分数和质量标记。完整 JSON Schema 随每个训练包生成。

- `messages` 只包含该 target 之前的严格历史前缀，以本轮 User 结束；当前 target 与未来信息不进入输入。
- System 从适用该 target 的冻结 `system_contexts` 读取，并核对目标语言；不重新拼一份、也不插入旧 `prompt` 字段。
- 使用已有格式化 Greeting，避免重复插入；缺失不生成。历史包含当时的 `context_only` Assistant，它们只作输入。
- `target` 保持原始 Assistant 正文，按原 ID 与 SHA-256 核对。
- 平台 guideline 不重新附回 User；模型身份、评分与线上标签不混入 `messages`。
- `metadata` 不输入模型；训练仅对 `target` 计算 loss。`read_targets.py` 的消息转换辅助函数不自动设置训练器的 loss mask。

## 输出清单

```text
release/
├── MANIFEST.json                 # 构建签名、来源、配置与分片清单
├── FILES.json                    # 除清单自身外全部文件的大小 / SHA-256
├── data/targets-*.jsonl.gz        # UTF-8、gzip、逐行训练样本
├── docs/                         # 格式、schema、选择规则与分布
├── audit/selection/              # 选择索引、源候选、会话结果和统计
├── audit/verification.json       # 正文、System、前缀与分片核对
├── audit/training_statistics.json
└── tools/                        # 读取器与导出工具副本
```

生成后全量回读验证并核对哈希，再从临时构建目录发布到最终目录。保留语言、角色、用户、来源模型及长度分布；字符长度不是 tokenizer token 数。

训练 / 验证至少按 conversation_id 分组，不得随机拆 target 造成共享上下文泄漏。若需要评估跨角色泛化，再按角色分组。没有默认训练切分，不自动删除带质量标记的数据。
