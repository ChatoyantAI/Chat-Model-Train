# Roleplay Judge Dataset

> 状态：开发中
>
> 主负责人：`@gaoxiaomo`
>
> 备份 / Reviewer：待认领

## 目标

本模块将冻结的线上 ELO comparison 按采集期生产代码重建为候选级
`input_messages`，再拆成不含模型身份、左右位置和真人选择的单回复 Judge
输入。成功标准是对 13,269 场有效 comparison 生成 26,538 条可独立评分的
记录，并保留隔离标签、非正文审计和 manifest。

这里的结果是“按照采集期生产逻辑和冻结数据得到的高保真上下文重建”，
不是生产环境保存下来的原始请求包。生产系统没有持久化最终
`input_messages`。

## 范围与边界

本模块负责：

- 恢复候选生成前有效的消息历史；
- 复刻生产模板选择、角色卡、Persona、Greeting、记忆和 guideline 拼装；
- 按候选上下文窗口分别裁剪历史；
- 生成单回复 Judge 输入、隔离标签、上下文审计和 manifest；
- 检查匿名字段、数量、唯一 ID 和同场候选不相邻。

本模块不负责：

- 执行 Judge 评分；
- A/B 对比、ELO、权重拟合或模型排名；
- 从线上数据库修改任何数据；
- 把生产聊天正文或本地生成产物提交到 Git。

上游输入是冻结 ELO 数据和单独保存的最小运行时快照。下游消费者是人工
或 AI 单回复 Judge，以及评分完成后的隔离标签合并流程。

## 输入契约

| 输入 | 格式 | 来源 | 关键校验 |
|---|---|---|---|
| comparison | `comparisons.jsonl.gz` | 冻结导出 | `scoring.eligible=true` |
| 历史消息 | `context_messages.jsonl.gz` | 冻结导出 | 按 comparison 排序 |
| 图片解析 | `message_image_contexts.jsonl.gz` | 冻结导出 | 只使用 completed 且非空结果 |
| mutation | `message_mutation_events.jsonl.gz` | 冻结导出 | 回退生成之后的事件 |
| 运行时参数 | `prompt_reconstruction_runtime.jsonl.gz` | 生产只读当前快照 | 每个有效 comparison 恰好一条 |

运行时快照只保存 `chat_mod`、用户性别、测试账号布尔值、prompt 实验分组和
左右候选的 `context_window_k`。它不得保存邮箱、用户 ID 或数据库凭证。

## 输出契约

| 输出 | 格式 | 用途 |
|---|---|---|
| `judge_inputs.jsonl.gz` | 每行一个独立 Judge JSON | 提供给 Judge |
| `evaluation_labels.csv` | 隔离映射 | Judge 完成后解盲 |
| `evaluation_context_audit.jsonl.gz` | 无聊天正文的审计 JSONL | 排查重建分支和字段来源 |
| `manifest.json` | JSON | 数量、hash、版本和已知限制 |

Judge 输入只含：

```json
{
  "schema_version": 1,
  "evaluation_id": "eval_<opaque-random-id>",
  "rubric_version": "roleplay-rubric-v0.1",
  "trigger_type": "message | continue | auto_reply",
  "input_messages": [],
  "output_message": {"role": "assistant", "content": "..."}
}
```

## 目录与入口

```text
src/evaluation/roleplay_judge_dataset/
├── builder.py
├── cli.py
├── context_builder.py
├── io.py
└── templates/
```

- 单候选入口：`scripts/build_production_context.py`
- 批量入口：`scripts/build_roleplay_judge_dataset.py`
- Python API：`build_candidate_context()`、`restore_history()`、
  `build_judge_dataset()`

`templates/` 固化生产提交
`f8ab72c314fafa843677a94269e29dbdfb94f344` 的模板。运行时不访问 Git，也不
依赖 `lusychat` 工作区。

## 运行方法

以下示例使用 PowerShell。变量值按本机授权数据目录设置；不要把路径改写成
代码默认值。

```powershell
$dataRoot = $env:ROLEPLAY_ELO_DATA_ROOT
$rawDir = Join-Path $dataRoot 'elo_export_data_check_v2_20260827'
$runtimeFile = Join-Path $dataRoot 'prompt_reconstruction_runtime.jsonl.gz'

python scripts/build_production_context.py `
  --raw-dir $rawDir `
  --runtime-file $runtimeFile `
  --comparison-id '<comparison-id>' `
  --candidate left `
  --output (Join-Path $dataRoot 'context_sample.json')

python scripts/build_roleplay_judge_dataset.py `
  --raw-dir $rawDir `
  --runtime-file $runtimeFile `
  --output-dir $dataRoot
```

已有输出时，只有显式传 `--overwrite` 才会替换四个生成文件。构建先写临时
目录，全部完成后再原子移动到目标路径。

## 关键生产规则

- `chat_mod <= 50` 为 chat，`50 < chat_mod < 450` 为 scene，
  `chat_mod >= 450` 为 longText；
- `0 < context_window_k < 16` 不使用 summary，只保留最近 25 轮；
- 其他模型有 summary 时只保留 summary 边界之后的历史；
- 无 summary 且达到 bootstrap 条件时保留最近 25 轮；
- Continue 使用固定的瞬时 user 指令；
- 待评回复永远不进入 `input_messages`；
- 同场两个候选分别构造上下文，生成后使用随机不可反推 ID 并打散；
- Judge 输入不包含 pair、模型、左右位置或真人选择字段。

## 验证

```powershell
python -m unittest tests.evaluation.test_roleplay_judge_dataset -v
python -m unittest discover -s tests -v
git diff --check
```

正式数据还需验证：

- 普通消息、Continue、8K、summary 大上下文模型各抽一条；
- 8K 样本无 memory system，大上下文 summary 样本包含 memory system；
- `judge_inputs.jsonl.gz` 恰好 26,538 条；
- `evaluation_id` 唯一；
- Judge JSON 的键中不存在模型、左右、pair 或真人选择字段；
- 标签与审计均为 26,538 条；
- 同一个 `pair_group_id` 对应的两条记录不在 Judge 文件中相邻。

## 可复现与产物

模板和生产代码版本固定；冻结数据由上游 manifest/hash 标识。正式 manifest
记录输入快照、输出 hash、数量和生成时间。为满足匿名要求，`evaluation_id`、
`pair_group_id` 和打散顺序使用系统密码学随机源，因此重跑不会得到相同 ID 或
字节级相同输出。

## 已知限制与风险

- 生产没有保存最终模型请求包，只能按冻结字段和生产代码重建；
- 角色和 Persona 是冻结导出时快照，不保证逐字段等于生成时版本；
- `chat_mod`、用户性别、测试账号状态、prompt 分组和上下文窗口来自当前快照；
- 历史有效状态依赖冻结时间戳和 mutation event 回退；
- 当前数据没有 Character Card Lorebook，但实现保留采集期激活逻辑。

## 变更记录

- v0.1：建立候选级上下文重建和单回复 Judge 数据生成流程。
