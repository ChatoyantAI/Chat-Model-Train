# Data-Juicer independent_v1

这是 v2.2 候选母集的第一轮全量、CPU-only、独立样本初筛。它不使用 fork、parent 或 lineage 关系来改变比较范围：256 个 I/O bucket 合并成一个全局样本集合，每一行都视为一条独立完整对话。

## 输入与边界

- 输入版本：`lusy-rp-dialogue-data-juicer-v2.2`；
- 输入单位：一行一条 structurally valid 完整 dialogue branch；
- 时间窗：`[2026-03-20, 2026-07-20)` UTC；
- 预期输入：1,254,820 dialogues、18,999,811 retained turns、256 buckets；
- 原始 `01_constructed/dialogues/*.jsonl` 只读；
- 本轮不执行 benchmark/entity 排除、PII/安全清理、Judge 或最终训练集物化；
- 结果是候选审计分支，不是最终删除名单。

固定 RP benchmark 仍须在后续训练集物化前按 user、character、message/text 做永久排除。本轮只做内容重复和复读画像，不能据此声称已完成训练/评估隔离。

## 文本视图

`derive_views.py` 每个 bucket 原子生成一个 Parquet，并保留小型 report 以支持中断续跑：

| 字段 | 内容 | 用途 |
|---|---|---|
| `full_training_sha256` | canonical `[role, content]` messages（含 system、memory、Greeting）的 SHA-256 | 精确重复等价键，避免再物化一份完整训练文本 |
| `dialogue_body_text` | Greeting 之后的 user/assistant turns，保留角色标签，不含 system、memory、Greeting | MinHash、SimHash |
| `assistant_body_text` | Greeting 之后的 assistant replies | 单对话复读率 |
| `greeting_text` | 构造后的 Greeting | 后续 Greeting/回复格式专项分析 |

完整 messages 的 SHA-256 只是无损的精确等价键：Data-Juicer `document_deduplicator` 仍负责聚类和保留第一条；本轮不把 SHA-256 当作模糊相似文本。

## 运行顺序

```text
256 frozen JSONL buckets
  → bucket-level derived Parquet + profile manifest
  → document_deduplicator(full_training_sha256)
  ├→ document_minhash_deduplicator(dialogue_body_text)
  ├→ document_simhash_deduplicator(dialogue_body_text)
  └→ word/character_repetition_filter(assistant_body_text, score only)
```

MinHash 和 SimHash 都从 exact survivors 独立启动，二者不串联。`dialogue_body_text < 32` 的极短样本不进入模糊哈希，汇总中单列为 passthrough；这是为了规避空 shingle 和全零 SimHash，而不是删除它们。

参数：

- exact：大小写敏感，不移除非字母字符；
- MinHash：XLM-R multilingual SentencePiece、5-token shingle、256 permutations、Jaccard threshold 0.85；
- SimHash：character 6-gram、6 blocks、Hamming distance 4；
- repetition：word/character 10-gram，`min_ratio=0.0`、`max_ratio=1.0`，只打分不删除；
- Data-Juicer：v1.5.5 pinned，CPU-only，`np=96`。

## 产物

正式远端 run 登记在 `docs/DATA_ARTIFACTS.md`。目录契约：

```text
00_input_manifest/        配置副本、构造 manifest、代码 SHA-256、tokenizer 链接
01_derived_profile/       bucket Parquet、bucket reports、PROFILE_MANIFEST.json
02_exact_dedup/           exact survivors 与 tracer
03_minhash_branch/        MinHash 独立候选分支与 tracer
04_simhash_branch/        SimHash 独立候选分支与 tracer
05_repetition_profile/    score-only Parquet 与 stats sidecar
06_candidate_samples/     预留给后续人工复核样本
07_dashboard/             原子状态、compact summary、重复证据、复读 Top records
```

`run_full.sh` 需要外部传入 `SOURCE_DIR`、`RUN_ROOT`、`DJ_ROOT`、`TOKENIZER_MODEL`，不包含登录凭据。`run_status.json` 会在 bucket 完成和全局 stage 切换时更新，8988 Streamlit 页面读取的只是这些小型文件，不会扫描全量 Parquet。

## 已知限制

- Data-Juicer 三种文档去重算子默认保留 cluster 的第一条，不理解哪条 RP 质量更好；
- MinHash 近似的是 shingle set Jaccard，SimHash 更偏模板/词频指纹，两者都不是语义模型；
- 本轮忽略 fork 有利于先得到全局独立样本重复率，但不能替代后续 shared-prefix/fork 代表选择；
- 完整 system/角色卡没有进入模糊分支，避免共享角色设定掩盖实际对话变化；
- 循环率是整段 assistant 聚合分数，不能定位语义复读、同义改写或具体坏 turn；
- 在人工标注校准前，MinHash/SimHash/repetition 分数只能用于候选召回和分布画像。
