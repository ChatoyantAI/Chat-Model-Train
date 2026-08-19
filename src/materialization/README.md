# Materialization

逐文件用途见 [`../README.md`](../README.md#materialization)。

训练物化负责把选择结果转换为最终 `messages + train flag`，并验证：

- RP/General 数量；
- train/validation group leakage；
- target hash 唯一性；
- Greeting、system、宏替换和长度指导；
- General 文件在 RP repair 前后字节不变；
- 输出 manifest 与 SHA-256。

当前保存的 `materialize_v4_0_1.py` 是 v4.0.1 的后续修复物化脚本，不是最初 v3.1 selector。默认远端根目录为 `/home/gyr/lusychat-training`，可通过环境变量显式覆盖输入和输出路径。
