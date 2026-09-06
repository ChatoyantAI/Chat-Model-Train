# 会话过滤评分规范 V1

协议：`roleplay-conversation-filter-v1.0`。实现位于 `src/evaluation/roleplay_annotation_platform/`，与单回复标注、过滤 V2 使用独立页面、Cookie 和结果库。

## 评估对象

读取截至最后一个粗筛候选的完整对话，每段一次请求，输出一套 20 项 `0/1/2/3` 或允许时的 `N/A`。评估对象是整段中的候选 Assistant 回复组，不是所有历史回复，也不是逐轮分别打分。

| 数据 | 作用 |
|---|---|
| `scope=candidate` 的 Assistant | 直接参与质量判断，在 `issues` 定位局部问题 |
| `scope=context_only` 的历史 | 提供前情，不直接算候选缺点 |
| Greeting | 初始状态、关系和语言依据，不作为候选评分 |
| `system_contexts` | 各候选适用的冻结平台生成指令 |
| 角色与 Persona | 有来源的角色、玩家设定，不是账号联系方式 |

完整档位、N/A 条件和校准案例见[会话量表](ROLEPLAY_CONVERSATION_FILTER_RUBRIC_V1.md)。该文档由 `conversation_filter.py` 生成，与网页、真实请求相同。后续信息不能反推前轮应该已知；缺失依据写入 `data_limitations`，不得编造。N1、N4、L6 不允许 N/A。

## 上下文构建

基础输入为原始会话 Parquet 分片、粗筛 target Parquet 与可选 Persona 快照。按消息 ID 匹配候选，保留中间未入粗筛的历史，不注入未来摘要。

Greeting 优先取角色资料中已有开场，按平台规则处理占位符和链接，不生成、续写或翻译。对应语言缺失时可使用已有英文资料，明确记录资料语言与来源；资料语言不覆盖候选要求的输出语言。

System 共用 `roleplay_judge_dataset.context_builder.build_platform_system_prompt()`：冻结模板、角色字段、Persona、聊天模式、第三人称及实验分支参与组装。不读取 `character_descriptions.prompt` / `characters.prompt`。这是有来源的离线上下文组装，不保证等于各轮历史请求的逐字副本。

从 User 正文分离平台 `<guideline>`，语言和格式等单列于 `platform_output_requirements`。N1 只看用户原始意图；所有项不检查字数 / Token 上限。目标语言以平台要求为准；约定状态栏与格式仍须检查。

### 索引构建

从仓库根目录执行，输出使用新文件，不覆盖已有索引。`SNAPSHOT_*` 代表受控快照路径，不是数据库连接。

```bash
python -m src.evaluation.roleplay_annotation_platform.conversation_filter_data --source SNAPSHOT_BUNDLES --candidates SNAPSHOT_TARGETS.parquet --personas SNAPSHOT_PERSONAS.parquet --output outputs/conversation-filter/data/base.sqlite3
python -m src.evaluation.roleplay_annotation_platform.conversation_filter_enrich --input outputs/conversation-filter/data/base.sqlite3 --output outputs/conversation-filter/data/enriched.sqlite3 --conversations SNAPSHOT_CONVERSATIONS.jsonl --recovery SNAPSHOT_OPENINGS.jsonl --profiles SNAPSHOT_PROFILES.parquet
python -m src.evaluation.roleplay_annotation_platform.migrate_conversation_platform --input outputs/conversation-filter/data/enriched.sqlite3 --output outputs/conversation-filter/data/conversations-v1-platform.sqlite3 --runtime SNAPSHOT_RUNTIME.jsonl --candidates SNAPSHOT_TARGETS.parquet
```

`enrich` 的会话 / 开场输入是逐行 JSON 的命名查询结果封装，读取 `conversations_*`、`first_messages_*`、`character_openings_*`；可用 `--english` 提供已有英文开场快照。原始字段契约由构建器定义，必须保留来源。`migrate` 默认只写新索引；可选 `--results` 会清理历史请求展示字段，使用前须停服务、备份结果库并确认范围。

索引的 `metadata.manifest` 包含来源与数据 SHA-256。任务冻结 `data_hash`、`rubric_hash`、请求模板和抽样配置；禁止替换原索引后继续同一任务。

## 启动与任务

复制 [conversation.env.example](../../configs/evaluation/conversation.env.example) 为同目录 `conversation.env`，填写私有路径与 DeepInfra Key。配置说明见[配置目录](../../configs/evaluation/README.md)。

```bash
python -m uvicorn src.evaluation.roleplay_annotation_platform.conversation_filter_app:app --host 127.0.0.1 --port 8768 --workers 1 --env-file configs/evaluation/conversation.env
```

页面提供会话目录、完整上下文、请求预览、量表、任务、逐项结果、排名与导出；按用户名隔离任务，用户名不是密码认证。

范围支持当前会话、序号、随机、角色温和平衡及筛选范围内全部。记录随机种子，可设每角色 / 用户上限；温和平衡按角色频次平方根分配倾向，不等于各角色等量。

`deepinfra::zai-org/GLM-5.3-Flash` 直连 DeepInfra，也支持已配置的 OpenRouter 模型。Key 仅存在服务器。浏览与预览不创建评分请求；任务可先准备为暂停状态，再由用户启动。

### 并发、重试与费用

- 单个 Uvicorn worker，全局 100 个滚动并发槽位，每任务 1–100；请求完成后继续取下一条，不等待整批完成。
- 每任务保存最大尝试数。网络 / 可重试服务器错误和无效结构按额度重试；429 尊重 `Retry-After` 并延后该任务派发。
- 可暂停、继续、取消或仅重试失败项，已完成项不重复评分。服务重启会暂停旧任务、标记中断，不直接重复扣费。
- 预算是停止派发阈值，不是账单硬上限，在途请求仍计费。实际费用与估算费用区分来源；未知费用不视作 0，必要时暂停。
- 默认不发 temperature、max_tokens、confidence 或 logprobs；会话模式没有 Gate 阶段。

可选 `supervise_conversation_run` 要求显式 `--authorize-failure-retries`，只恢复有界技术失败，保留手动暂停、取消、版本与预算限制，不改文本规避拒评。不会随 Web 服务自动启动。

## 输出与解释

```json
{
  "scores": {"N1": {"value": 2, "reason": "各轮问题得到明确回应。", "turn_ids": ["A1", "A2"]}},
  "issues": [],
  "data_limitations": []
}
```

示例仅展示一个项目，实际必须输出完整 20 项。`issues` 包含 `turn_ids/item_ids/severity/reason/affects_later`。分数非法或缺项属于无效结果；定位不够准确等可保留评分并记录 `warnings`。`data_limitations` 是模型报告的依据限制，需要核对，不等于整段未读取或技术失败。

过滤建议和排名分开：

- 建议结合最低分、重大问题与依据 / 校验警告，输出保留或复核建议，不删除原文。
- 排名使用完整、有效、有数字分且非疑似全量拒评的结果。拒评标记需要多个评分理由共同支持，单项 N/A 的解释不直接排除整段。
- 排名分为适用数字项等权均值：`sum(numeric_scores) / numeric_count`。N/A 不折 0；采用精确分数比较并列，序号仅作为稳定排列键。
- 随排名保留低分、重大问题、警告、适用项和 N/A 数量。高均分不等于每个 target 都合格，不同适用项组合也不完全可比。

## 导出与训练数据

```bash
python -m src.evaluation.roleplay_annotation_platform.export_conversation_run --result-db outputs/conversation-filter/results/jobs.sqlite3 --sample-db outputs/conversation-filter/data/conversations-v1-platform.sqlite3 --job-id JOB_ID --output-root outputs/conversation-filter/exports
```

只读导出终态任务；`--wait` 可等待结束。文件包含全量评分、排名、保留 / 复核 / 排除索引、汇总与校验清单，记录数据 / 规则哈希、任务状态及费用。

之后按排名与角色配额选择 target，输出 System + 历史 + target 训练包，见 [target 规范](ROLEPLAY_TARGET_EXPORT_V1.md)。必须标注 `target_independently_scored=false`；训练 / 验证至少按会话分组，避免共享前缀泄漏。

## 验证

```bash
python -m unittest discover -s tests/evaluation -v
python -m src.evaluation.roleplay_annotation_platform.build_conversation_docs --check
```

覆盖候选 / 背景边界、平台拼接、旧 prompt 排除、语言、Greeting、滚动队列、预算、429、排名、用户隔离、导出及 target 前缀；不访问生产数据或付费模型。
