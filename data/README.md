# 数据资产与评测输入

四套固定评测输入集中在 [evaluation/roleplay_benchmarks/](evaluation/roleplay_benchmarks/README.md)，已纳入 Git，约 60.4 MiB。
包含 CoSER 测试集、CharacterBench 原始测试输入、RMTBench 两个版本的中英文输入，以及 MiniMax 的 95 条场景输入。
原文件按字节保留；来源 revision、文件大小和 SHA-256 见该目录的 `manifest.json`。

安装器优先校验这些文件，再恢复到原有运行路径。模型权重、训练集、API 密钥和评测响应缓存仍不进入 Git。

## 其他数据资产

除上述明确版本化的评测输入外，本目录保存数据契约与索引，例如：

- schema 和字段字典；
- 数据版本 manifest；
- 许可、来源和隐私说明；
- 不含敏感信息的统计摘要；
- 极小、脱敏、用于测试的固定样例。

`raw/`、`interim/`、`processed/` 和 `external/` 会被 Git 忽略。大型数据必须存放到团队认可的外部存储，并通过不可变 URI、版本和 hash 引用。

任何新增数据版本都应由 `src/data/` 中的可重复流程生成，并在 manifest 中记录代码 commit、输入版本、配置、随机种子、记录数、schema 版本和内容 hash。
