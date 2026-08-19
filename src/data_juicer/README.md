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

远端项目：

```text
/home/gyr/lusychat-training/data-juicer-dialogue-v2.2-20260816
```

Data-Juicer 的质量 recipe 尚未运行。后续 operator 写入新的版本目录，计划见 [`docs/DATA_JUICER_RESTART.md`](../../docs/DATA_JUICER_RESTART.md)。
