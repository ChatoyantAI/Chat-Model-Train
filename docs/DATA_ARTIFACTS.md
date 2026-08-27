# 远端数据路径

仓库不保存训练正文。下面是 4090 上的数据、报告和 manifest 位置。

## 总入口

```text
/home/gyr/lusychat-training/lusy-data-pipeline/README.md
```

该入口包含说明、代码镜像、归档和校验清单。

## 第一版 MDS/MMR 链路

| 内容 | 远端路径 | 状态 |
|---|---|---|
| pre-MDS targets 与脚本 | `/home/gyr/lusychat-training/pre-mds-context-v2` | 完整 |
| MDS 对话、embedding、聚类、MMR、union 和报告 | `/home/gyr/lusychat-training/mds-dialogue-rp-v1` | 完整 |
| Dialogue Judge Qwen | `/home/gyr/lusychat-training/judge-qwen36-35b-dialogue-evidence-v2` | 完整 |
| Dialogue Judge GLM | `/home/gyr/lusychat-training/judge-glm47-flash-dialogue-evidence-v2` | 完整 |
| Dialogue tier 与 Elite A | `/home/gyr/lusychat-training/judge-dialogue-selection-v1` | 完整 |
| Turn Judge 聚合 | `/home/gyr/lusychat-training/judge-deepinfra-glm47-turn-v47-softjson-direct64-a-all-aggregated-20260729` | 完整 |
| 原生归并后的候选池 | `/home/gyr/lusychat-training/a-unified-candidates-20260730-v2` | 完整 |
| 最终 RP 50K 选择结果 | `/home/gyr/lusychat-training/a-unified-50k-20260730-v2` | 结果与 report 完整；独立 selector 源码未冻结 |
| 60K 内容包 | `/home/gyr/lusychat-training/lora-content-60k-20260730-v1` | 完整 |
| v3.1 训练数据 | `/home/gyr/lusychat-training/lusy-sft-60k-20260802-v3.1` | 完整 |

最终 selector 的 selection report 记录了输入 296,959 个原生回复、286,418 个自动可选回复、质量排序、配额、caps、贪心和 swap repair；执行脚本没有单独保存。后续若重写 selector，需要标成新版本。

## 通用数据

| 内容 | 远端路径 |
|---|---|
| General 100K 候选池 | `/home/gyr/lusychat-training/general-sft-anchor-v1` |
| General 10K 选择与 60K 内容包 | `/home/gyr/lusychat-training/lora-content-60k-20260730-v1/01_selection/general_10k` |

General 10K selection manifest 冻结了 source、task、language、length decile 配额和 selected ID set hash。

## 后续修复与训练数据

| 内容 | 远端路径 |
|---|---|
| v4.0.1 修复后数据 | `/home/gyr/lusychat-training/lusy-sft-55k-20260810-v4.0.1` |
| v4.1 all-loss 代码和说明 | `/home/gyr/lusychat-training/lusy-data-pipeline/code/training-v4.1-coser-allloss` |
| 官方基模与 v4.1 同协议通用评测 | `/home/gyr/lusychat-training/lusy-data-pipeline/artifacts/eval-archive/20260812-official-vs-v4.1-allloss` |

## Data-Juicer v2.2

```text
/home/gyr/lusychat-training/data-juicer-dialogue-v2.2-20260816
```

关键内容：

```text
01_constructed/dialogues/bucket-000.jsonl ... bucket-255.jsonl
01_constructed/quarantine/
01_constructed/reports/
01_constructed/CONSTRUCTION_MANIFEST.json
00_inputs/fork_lineage_index.json
```

v2.2 当前是结构合格候选母集，不是高质量训练集。

第一轮全量独立样本 Data-Juicer 初筛：

```text
/home/gyr/lusychat-training/data-juicer-dialogue-v2.2-20260816/02_data_juicer/01_independent_v1
```

该 run 使用 Data-Juicer v1.5.5、CPU-only；256 buckets 全局合并，忽略 fork/lineage 关系。状态和紧凑结果位于 `07_dashboard/`，正式 Parquet 分支见 `02_exact_dedup/`、`03_minhash_branch/`、`04_simhash_branch/`、`05_repetition_profile/`。原始 `01_constructed` 未修改。

Turn 级 BGE-M3 语义向量：

```text
/home/gyr/lusychat-training/data-juicer-dialogue-v2.2-20260816/02_data_juicer/02_semantic_audit_v1
```

输入为 `01_independent_v1` exact survivors，共 1,254,267 条对话、18,958,570 个有效 turn pair。`01_turn_view/` 保存稳定 turn 元数据与原文，`assistant_embeddings/` 和 `user_embeddings/` 保存按 bucket 对齐的 1,024 维 FP16 归一化向量，`07_dashboard/` 保存实测吞吐、ETA 和运行状态。该版本只构建向量，不按 cosine 自动删除样本。后台服务为 `lusy-data-juicer-semantic-v1.service`。

## 固定 RP benchmark

| 内容 | 远端路径 |
|---|---|
| 冻结 500 对话 seed | `/home/gyr/lusychat-training/lusy-rp-eval-500-20260804` |
| 当前评估实现与 run ledger | `/home/gyr/lusychat-training/rp-benchmark-eval-v1-20260811` |

benchmark 的用户和角色必须永久排除出后续 SFT、DPO 和强化学习数据；模型输出、Bridge 续写和 Judge 结果也不得回流训练。
