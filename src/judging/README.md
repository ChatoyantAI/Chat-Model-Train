# Judging

逐文件用途见 [`../README.md`](../README.md#judging)。

## Dialogue Judge

`dialogue/` 保存 Qwen/GLM 对话级 Judge contract、调用和 tier 构造代码。评估母集是 115,746 个 retention-60 UNION dialogues（111,465 neutral MMR + 3,851 protected core bypass + 430 summary bypass），不是纯 MMR 集合。

Qwen 和 GLM 各覆盖完整母集的 25%（各 28,937 条），样本互不重叠。抽样先按来源分配名额，再按 `source_partition × language_group × representative_topic × replay_mode × turn_bucket` 联合分层分配名额，最后在层内按固定 seed 的 `dialogue_id` SHA256 稳定排序取样。GLM 批次先排除 Qwen 批次，再从剩余数据重新分层抽取；它不是取剩余数据的 25%。这一步不按质量、MMR rank、点赞或模型来源择优，选中 dialogue 后保留全部可评分 assistant turns。`build_selection_v1.py` 把有效 Judge 结果转为 Elite A、Strong B 等 tier。

远端结果：

```text
/home/gyr/lusychat-training/judge-qwen36-35b-dialogue-evidence-v2
/home/gyr/lusychat-training/judge-glm47-flash-dialogue-evidence-v2
/home/gyr/lusychat-training/judge-dialogue-selection-v1
```

## Turn Judge

`turn/` 保存 v4.7 prompt contract 来源、首批执行、Elite A 全量准备、结果聚合和选择政策分析代码。脚本只从环境变量读取服务凭证。

Turn Judge 的 302,500 条有效结果是技术 evaluation rows。由于一个 root conversation 可以按 replay mode、memory 和 boundary 形成多个技术 dialogue，同一原生 assistant event 可能在不同可见 context 下被展开和评估多次。最终选择前按 `(conversation_id, assistant_increase_id)` 归并为 296,959 条 unique native replies；归并的是同一回复的重复评估视图，不是跨 conversation 文本去重，也不是 Fork 归并。代表行优先原始 body hash 一致和 context 更完整，不按多个 Judge 分数取最高值。

远端聚合结果：

```text
/home/gyr/lusychat-training/judge-deepinfra-glm47-turn-v47-softjson-direct64-a-all-aggregated-20260729
```

## 缺失文件

最终 50K selector 的结果和 selection report 已冻结，但独立执行脚本没有保留下来。本目录不提供一个后来重写的 selector 冒充历史代码；下一版 selector 应在新版本目录实现，并从 quality-pass 母集开始。
