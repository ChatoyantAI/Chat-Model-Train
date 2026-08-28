# 角色扮演双盲标注平台

## 目录约定

服务器部署根目录：`/home/shared/roleplay-blind-evaluation`

```text
/home/shared/roleplay-blind-evaluation/
├── README.md
├── data/
│   ├── judge_inputs.jsonl.gz
│   ├── manifest.json
│   └── samples.sqlite3
├── ground_truth/
│   ├── evaluation_labels.csv
│   └── evaluation_context_audit.jsonl.gz
├── app/
│   └── roleplay_annotation_platform/
└── results/
    ├── annotations.sqlite3
    └── by_user/<username>/<sequence_number>.json
```

`data/` 是平台实际读取的双盲数据。`ground_truth/` 保存解盲标签和审计数据，平台代码不会读取，也不会通过网页或 API 返回。

## 输入数据

### `judge_inputs.jsonl.gz`

- gzip 压缩的 UTF-8 JSONL；
- 每行是一条独立待评回复；
- 共 26,538 条；
- SHA-256：`558490bfa21a1fc073119e3dcfa2d37addfa79888cc32d7e11cc4adbcbaadfaf`。

每行结构：

```json
{
  "schema_version": 1,
  "evaluation_id": "eval_...",
  "rubric_version": "roleplay-rubric-v0.1",
  "trigger_type": "message | continue | auto_reply",
  "input_messages": [
    {"role": "system", "content": "..."},
    {"role": "assistant", "content": "角色 Greeting"},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "历史回复"},
    {"role": "user", "content": "当前输入"}
  ],
  "output_message": {"role": "assistant", "content": "本条待评回复"}
}
```

网页把 `input_messages` 中第一条 assistant 消息标记为 `Greeting`，其他 assistant 消息标记为历史回复；`output_message` 单独显示为“本条待评回复”。

Judge 文件中不包含模型名称、左右位置、pair ID、真人选择或胜负标签。

### `evaluation_labels.csv`

- UTF-8 CSV；
- 共 26,538 行；
- SHA-256：`3e490c4926bf0c30583dc6224b0168fdeaf2f4ec3958b4f3d330069ea7d06c32`；
- 表头：

```text
evaluation_id,pair_group_id,model_id,model_name,human_selected,source_completed_at
```

字段说明：

| 字段 | 含义 |
|---|---|
| `evaluation_id` | 与 Judge 数据、人工标注结果连接的唯一键 |
| `pair_group_id` | 同一次真人投票的两个候选共享该值 |
| `model_id` / `model_name` | 解盲后的模型身份 |
| `human_selected` | 该候选是否被线上用户选中；每组恰好一条为真 |
| `source_completed_at` | 原投票完成时间 |

标注期间不要把该文件放进网页目录，也不要交给标注员。人工结果冻结后，使用 `evaluation_id` 合并即可恢复模型身份与真人胜负。

### `evaluation_context_audit.jsonl.gz`

每条 Judge 数据对应一条上下文重建审计记录。该文件用于复核模板、summary、历史裁剪和运行时字段来源，不参与网页展示。

## 平台行为

- 用户只需输入固定用户名；
- 所有人按相同的 `seq=1..26538` 顺序看到相同数据；
- 每个用户名分别保存进度、草稿和提交结果；
- 页面与 API 不返回 `evaluation_id`、模型身份、配对关系或真人选择；
- 页面依次展示 System、Greeting、User/Assistant 历史和本条待评 Assistant 回复；
- 内置 Gate、5 个维度、25 个原子项以及每项的 0–3 分真实回复示例；
- 原子评分取值为 `0 / 1 / 2 / 3 / N/A`；
- 0、1、3 分提交时必须填写直接证据；
- 可保存草稿、修改已提交记录、前后翻页或按序号跳转；
- 每分钟自动保存一次未保存草稿。

用户名允许 1–40 个中文、字母、数字、下划线或连字符。平台没有密码验证；知道地址的人可以创建用户名，因此不要把地址公开传播。

## 结果格式

SQLite 是恢复进度的主索引；每次保存还会生成一份按用户名分目录的 JSON 快照：

```json
{
  "schema_version": 1,
  "rubric_version": "roleplay-rubric-v0.1-4point",
  "username": "annotator_01",
  "sequence_number": 1,
  "evaluation_id": "eval_...",
  "status": "draft | submitted",
  "gates": {
    "G-D1": false,
    "G-D2": false,
    "G-D3": false,
    "G-S1": "no",
    "G-S2": "N/A",
    "G-S3": "no",
    "G-P1": false,
    "G-P2": false,
    "G-P3": false
  },
  "scores": {
    "H1": {"value": 2, "evidence": "..."},
    "C1": {"value": "N/A", "evidence": "..."}
  },
  "gate_notes": "",
  "overall_notes": "",
  "created_at": "UTC ISO-8601",
  "updated_at": "UTC ISO-8601"
}
```

导出全部结果：

```bash
cd /home/shared/roleplay-blind-evaluation/app
python3 -m roleplay_annotation_platform.export_results \
  --database ../results/annotations.sqlite3 \
  --output ../results/annotations-export.jsonl
```

只导出一个用户名时增加 `--username annotator_01`。

## 数据索引重建

网页通过 `samples.sqlite3` 随机读取单条样本。需要重建时执行：

```bash
cd /home/shared/roleplay-blind-evaluation/app
python3 -m roleplay_annotation_platform.prepare_data \
  --input ../data/judge_inputs.jsonl.gz \
  --output ../data/samples.sqlite3 \
  --overwrite
```

构建器会验证必需字段，并拒绝包含模型、pair、左右位置或真人胜负键的数据。

## 服务管理

服务名：`roleplay-annotation.service`，监听 `0.0.0.0:8766`。

```bash
systemctl --user status roleplay-annotation.service
systemctl --user restart roleplay-annotation.service
journalctl --user -u roleplay-annotation.service -f
```

健康检查：

```bash
curl http://127.0.0.1:8766/health
```

## 双盲边界

这里的“双盲”只覆盖平台展示和平台 API：标注员不知道模型身份，也看不到真人胜负；平台为不同用户名隔离结果。拥有服务器 shell 权限并能读取 `ground_truth/` 的管理员仍可主动解盲，因此管理员不应参与盲标，且不应把 ground-truth 文件复制到网页可访问目录。
