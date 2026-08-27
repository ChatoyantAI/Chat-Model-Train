# Data-Juicer Restart

逐文件用途见 [`../README.md`](../README.md#data_juicer)。

`construction/` 保存 v2.2 前置全量对话构造的核心代码：

- `normalize.py`：物理 message-version 规范化与模型来源标记；
- `bundle.py`：logical turn 和 raw conversation bundle；
- `build_fork_lineage_index.py`：fork lineage；
- `build_dialogues.py`：统一 system、persona、memory、Greeting 和完整 turns；
- `validate_sample.py`：单 bucket/样本结构验证；
- `aggregate_manifest.py`：256 buckets 的最终账目与 manifest；
- `schema.py`、`manifest.py`、`feedback.py`：冻结 schema 和审计工具。

`construction/vendor/build_dialogues_v2_1.py` 是 v2.2 依赖的冻结 v2.1 renderer。v2.2 的 policy layer 会复用它，保留该文件是为了让构造规则可审计，而不是让新版本继续复制旧 renderer。

`independent_v1/` 是 v2.2 候选母集的第一轮全量初筛：忽略 fork 关系，把每行作为独立完整对话，依次派生算子视图、执行 exact，并从 exact survivors 独立运行 MinHash、SimHash 和 score-only repetition。运行契约和限制见 [`independent_v1/README.md`](independent_v1/README.md)。

`semantic_audit_v1/` 是其后的 turn 级多语言语义向量阶段：从 exact survivors 重新关联冻结 `messages`，为 18,958,570 个 user/assistant turn pair 生成 BGE-M3 FP16 向量。该阶段只构建可复用向量和审计状态，不设置 cosine 删除阈值；运行契约见 [`semantic_audit_v1/README.md`](semantic_audit_v1/README.md)。

远端项目：

```text
/home/gyr/lusychat-training/data-juicer-dialogue-v2.2-20260816
```

Data-Juicer `independent_v1` 已于 2026-08-20 完成。`semantic_audit_v1` 于 2026-08-21 作为独立后台服务启动，后续 cosine/FAISS 候选和阈值校准继续写入新版本目录。计划见 [`docs/DATA_JUICER_RESTART.md`](../../docs/DATA_JUICER_RESTART.md)。
