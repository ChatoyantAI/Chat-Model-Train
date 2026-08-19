# MDS / MMR

逐文件用途见 [`../README.md`](../README.md#mds)。

本目录保存第一版 causal exact-language MDS 实现。

执行关系：

```text
build_rp_mds_dialogues_v1.py
→ prepare_rp_mds_embedding_tasks.py
→ embed_rp_mds_shard_causal_v2.py
→ build_causal_recent_views.py
→ assemble_causal_recent_embedding.py
→ assemble_fused_causal_v2.py
→ cluster_rp_mds_causal_v2_exactlang.py
→ select_rp_mds_mmr_causal_v2_exactlang.py
→ build_post_mds_union_causal_v2.py
→ validate_*.py
```

`U/R/S` 三个视图均不包含当前 assistant target；语言独立聚类，protected core 和有效 memory 旁路保留。数据和执行报告位于：

`recent_view_safe.py` 是 `build_causal_recent_views.py` 使用的 target 防泄漏边界实现。

```text
/home/gyr/lusychat-training/mds-dialogue-rp-v1
```

`run_formal_causal_v2_exactlang.sh` 是历史入口，运行前必须先核对远端 manifest 和路径，不能在本地空目录直接执行。
