# 代码地图

旧训练数据线：

```text
preprocessing
→ mds
→ judging/dialogue
→ judging/turn
→ General 10K + RP 50K
→ materialization
→ training（参数在 docs/TRAINING.md）
→ evaluation
```

`data_juicer` 是另一条从全量消息重新构造对话的线，不读取旧 MDS、Judge 或 RP 50K。

## `preprocessing/`

从冻结消息和候选回复构造因果可见的 assistant targets，再做语言过滤、格式修复和实体 cap。

| 文件 | 用途 |
|---|---|
| `build_context_plan_v2.py` | 为每个 target 决定 full history、bootstrap tail 或 memory tail；只生成计划，不重复物化长 prompt |
| `build_language_features_v2.py` | 用 Lingua 计算当前 user/assistant 的语言、置信度和 mismatch 原因 |
| `filter_liked_context_v2.py` | 对 liked/Claude/其他受保护来源做技术门禁，保留合格 target 和来源标记 |
| `materialize_messages_v2.py` | 按 context plan 拼出 system、Greeting、persona、memory、历史和最终 target |
| `format_repair.py` | 只修可确定恢复的引号、星号和嵌套 RP 格式 |
| `apply_diversity_caps_v2.py` | 保留 protected core，对 neutral 依次执行 conversation、user-character、user、character cap |
| `finalize_pre_mds_v2.py` | 删除 target 与可见历史的精确重复，重算 character share，写 pre-MDS manifest |

辅助模块：

| 文件 | 用途 |
|---|---|
| `lusy_training/cards.py` | 选择角色 description/persona，构造角色 system 和 image context |
| `lusy_training/preference.py` | 生成 prompt fingerprint |
| `lusy_training/text.py` | 去掉 provider CoT/think block |
| `lusy_extract/screening_text.py` | 空文本、provider error、控制 token、媒体占位符等硬门禁 |
| `language_alignment_support/build_language_alignment.py` | Lingua detector pool、分段检测和语言特征计算 |
| `language_alignment_support/language_alignment.py` | 文本清理、语言 group 和 user/assistant pair 分类规则 |
| `language_alignment_support/detector_worker.py` | 独立 Lingua worker，避免检测器阻塞主进程 |
| `*/__init__.py` | Python package 标记，不包含筛选逻辑 |

## `mds/`

把 targets 聚合成 replay dialogues，计算不含当前答案的三视图 embedding，按语言 KMeans，再在 cluster 内用 MMR 取 60%。

| 文件 | 用途 |
|---|---|
| `build_rp_mds_dialogues_v1.py` | 按 conversation、replay mode 和 memory boundary 聚合 targets，生成 dialogue 与 target map |
| `prepare_rp_mds_embedding_tasks.py` | 将 U/R/S 三视图切成可恢复的 embedding shards |
| `embed_rp_mds_shard_causal_v2.py` | 用 BGE-M3 编码单个 shard，写 embedding、token length 和 hash |
| `recent_view_safe.py` | 找出 current user 与 target assistant 边界，防止当前答案进入 embedding |
| `build_causal_recent_views.py` | 构造最近 3 组历史加当前 user 的 causal recent scene 文本 |
| `assemble_causal_recent_embedding.py` | 校验并合并 recent-scene shard |
| `assemble_fused_causal_v2.py` | 按 U 0.745 / R 0.005 / S 0.250 合并三视图向量 |
| `cluster_rp_mds_causal_v2_exactlang.py` | 21 种语言独立运行 MiniBatchKMeans |
| `select_rp_mds_mmr_causal_v2_exactlang.py` | cluster 内生成完整 MMR 排名和 50/60/70/80% selection |
| `build_post_mds_union_causal_v2.py` | 将 neutral MMR 结果与 protected core、memory bypass 合并并展开 targets |
| `validate_formal_causal_v2_exactlang.py` | 检查聚类、选择数量、语言隔离、hash 和 target leakage |
| `validate_post_mds_union_independent.py` | 独立重读 union，检查 bypass 完整性和 dialogue/target 账目 |
| `run_formal_causal_v2_exactlang.sh` | 串起 exact-language 聚类、MMR、union 和验证的远端入口 |

## `judging/`

### `judging/dialogue/`

| 文件 | 用途 |
|---|---|
| `dialogue_judge_contract_v2.py` | 定义对话级 Judge 的 evidence schema、0–4 分维度、prompt 和结构校验 |
| `run_dialogue_judge_25pct_v2.py` | 跑 25% dialogue slice；支持并发、重试、断点续跑和 token-balanced slot 分配 |
| `run_dialogue_judge_openrouter_v2.py` | 同一 Judge contract 的 OpenRouter runner |
| `run_judge_v6_evidence_gate.py` | 较早的 evidence-gate runner；保留用于核对历史 prompt，不与 v2 结果混算 |
| `build_selection_v1.py` | 合并 Qwen/GLM 有效结果，做 family 内 ECDF rank，生成 Elite A、Strong B 等层级 |

### `judging/turn/`

| 文件 | 用途 |
|---|---|
| `run_v46_promptonly_softjson_smoke.py` | v4.6 prompt/schema smoke；检查 DeepInfra soft JSON 行为 |
| `run_v47_first4k.py` | v4.7 首批 4K direct/batch 测试、上传/轮询/下载和结构验证 |
| `prepare_lusy_v47_full_a.py` | 将 Elite A 展开成全量逐 turn 请求和 expectation sidecar |
| `audit_full_a_turns.py` | 运行前核对 turn 数、上下文边界、长度和 prompt hash |
| `aggregate_a_all.py` | 合并两半 runner 和 retry 结果，输出 302,500 条 turn rows 与 dialogue 聚合 |
| `analyze_selection_policy.py` | 按原生 conversation/event key 分析重复评估行归并及质量/NSFW/语言/长度/实体分布，供 50K selector 设计使用；不做跨 conversation 文本去重 |

旧 50K selector 的独立执行文件没有保留下来；`analyze_selection_policy.py` 不是 selector 本体。

## `general/`

| 文件 | 用途 |
|---|---|
| `build_general_sft_anchor_v1.py` | 下载固定 revision 的 7 个公开数据源，做结构/长度/CoT/身份/精确重复过滤，按来源和语言配额构造 100K |
| `verify_general_sft_anchor_v1.py` | 独立检查 schema、source/language 数量、唯一性、think/身份残留和 SHA-256 |

100K 到 10K 的最终 selection manifest 在远端；每个来源精确取 10%，并保持 task、language 和 length decile 分层。

## `materialization/`

| 文件 | 用途 |
|---|---|
| `materialize_v4_0_1.py` | 从 v4.0 生成不可覆盖的 v4.0.1，移动指定 Thoughts block，并重算输出、audit 和 hash |

v3.1 的最初 materializer 没有作为独立通用脚本放在这里；v3.1 manifest 仍记录了 prompt、Greeting、宏替换和 split 结果。

## `evaluation/`

固定 500 对话 benchmark 的本地代码副本。

| 文件 | 用途 |
|---|---|
| `scripts/rp-eval` | CLI shell 入口 |
| `rp_eval/__main__.py` | 支持 `python -m rp_eval` |
| `rp_eval/cli.py` | `preflight`、`init-run`、`smoke`、`formal`、`status` 等命令 |
| `rp_eval/config.py` | 读取版本化 config 和环境变量中的 endpoint/key |
| `rp_eval/dataset.py` | 校验 500 seeds、hash、唯一性和 10/30/60 cohort |
| `rp_eval/roleplay_prompts.py` | 分别构造 Gemma 的训练同款 RP prompt 和 DeepSeek 线上 prompt |
| `rp_eval/candidates.py` | 候选请求/响应与 adapter 接口 |
| `rp_eval/candidate_clients.py` | OpenAI-compatible 候选模型请求、重试和错误分类 |
| `rp_eval/openrouter.py` | OpenRouter Bridge/Judge 客户端 |
| `rp_eval/bridge_policy.py` | 检查生成的后续 user 是否符合分支上下文和用户控制规则 |
| `rp_eval/prompts.py` | Bridge、单 turn Judge、trajectory Judge prompt 和 prompt hash |
| `rp_eval/schemas.py` | Pydantic Judge/result schema |
| `rp_eval/json_tools.py` | 从模型文本中提取 JSON object |
| `rp_eval/workflow.py` | 跑 A/B 与 B/A 双顺序盲评 |
| `rp_eval/scoring.py` | 将匿名 A/B 映射回模型名并聚合 position swap |
| `rp_eval/ledger.py` | append-only events 和不可覆盖 JSON 写入 |
| `rp_eval/pilot.py` | smoke、500 formal plan 和 matched replay 编排 |
| `rp_eval/selftest.py` | 不调用外部模型的结构自检 |
| `rp_eval/__init__.py` | package 标记 |
| `RUNBOOK.upstream.md` | 4090 评测项目的运行说明副本 |
| `SCORING_POLICY.upstream.md` | pairwise、uncertain 和 trajectory 聚合口径副本 |

## `data_juicer/`

### `data_juicer/construction/`

| 文件 | 用途 |
|---|---|
| `normalize.py` | 将 message versions 规范化成分桶 Parquet，并保留 model/provider 审计字段 |
| `schema.py` | Arrow schema 与 batch converter |
| `bundle.py` | 从规范化消息组成 logical turns 和 conversation bundles |
| `feedback.py` | 规范化 like/dislike/edit/regen，并生成保守的 preference candidates |
| `manifest.py` | 原子写入 shard 状态、计数和 hash，支持幂等续跑 |
| `build_fork_lineage_index.py` | 比较 turn signature 和 shared prefix，生成 fork/branch lineage |
| `vendor/build_dialogues_v2_1.py` | 冻结的 v2.1 renderer：角色卡、persona、memory、Greeting 和 messages 拼装 |
| `build_dialogues.py` | v2.2 policy layer：接 lineage、调整 source admission、调用 v2.1 renderer |
| `validate_sample.py` | 对单 bucket 和少量 accepted/quarantine 样本做结构检查 |
| `aggregate_manifest.py` | 汇总 256 buckets，重算 accepted/quarantine/turn 数量和最终 hash |
| `vendor/__init__.py` | vendor package 标记 |

Data-Juicer 的质量 operator 还没加入；后续代码应新增版本目录，不改写 v2.2 construction 的结果。
