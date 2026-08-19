# General SFT Anchor

逐文件用途见 [`../README.md`](../README.md#general)。

`build_general_sft_anchor_v1.py` 从固定 revision 的公开数据集构造 100K 通用能力候选池；不连接 LusyChat 生产服务。

构造包括：

- Aya、OASST2、Tulu IF、UltraChat、SmolTalk、OpenMath 和 BigCode 固定来源配比；
- 对话结构、长度、think/control token、AI 身份和精确重复过滤；
- source、task、language、license、provenance 和 selection key sidecar；
- 只有最后一个 assistant 计算 loss。

`verify_general_sft_anchor_v1.py` 对数量、schema、配比、唯一性和输出哈希做独立验证。

远端数据：

```text
/home/gyr/lusychat-training/general-sft-anchor-v1
```

第一版训练从 100K 中按 source、task、language 和 length decile 分层抽取 10K，selection manifest 位于：

```text
/home/gyr/lusychat-training/lora-content-60k-20260730-v1/01_selection/general_10k
```
