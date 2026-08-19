# Preprocessing

逐文件用途见 [`../README.md`](../README.md#preprocessing)。

第一版 pre-MDS 的技术过滤与因果上下文构造代码，建议阅读顺序：

1. `build_context_plan_v2.py`：确定 target 的 replay/memory 上下文计划。
2. `build_language_features_v2.py`：生成严格语言特征。
3. `filter_liked_context_v2.py`：整理受保护 liked 候选。
4. `materialize_messages_v2.py`：构造因果 OpenAI messages。
5. `format_repair.py`：仅做确定性的 RP 格式修复。
6. `apply_diversity_caps_v2.py`：执行 conversation/user/character concentration caps。
7. `finalize_pre_mds_v2.py`：处理精确历史重复并形成 pre-MDS parts。

随历史入口一并冻结的辅助实现：

- `lusy_training/`：角色卡拼装、prompt fingerprint 和 provider CoT 清理；
- `lusy_extract/`：确定性的文本硬门禁；
- `language_alignment_support/`：Lingua 检测池与语言一致性规则。

主脚本优先从上述仓库内目录导入；如需复现原远端布局，可用 `LUSY_PIPELINE_SRC` 和 `LUSY_LANGUAGE_ALIGNMENT_ROOT` 覆盖。

远端输入/输出根目录：

```text
/home/gyr/lusychat-training/pre-mds-context-v2
```

这些脚本对应第一版链路，数据默认路径和 schema 面向该远端目录。仓库只补了本地辅助模块发现方式，没有改变筛选逻辑。

## Fork 口径

第一版候选构造采用 root-only 范围：Fork/non-root conversation 在进入 target 池前直接排除。`filter_liked_context_v2.py` 对 liked/Claude 等受保护来源也显式检查 `is_fork_conversation` 和 `is_root_conversation`，命中后记录 `fork_conversation` 排除原因。

这不是 Fork-aware 去重：旧链路不会恢复 parent/child lineage，不比较父子质量，也不对 shared prefix 做 loss ownership。后续 MDS 产生的 replay/memory 技术 dialogue 属于同一 root conversation 的不同回放视图；它们在 Turn Judge 后按原生 conversation/event key 归并，和业务 Fork 是两种不同关系。不同 root conversations 之间的 exact/MinHash/semantic duplicate 也没有在 preprocessing 阶段全局归并。
