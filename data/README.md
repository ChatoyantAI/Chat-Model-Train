# 数据资产说明目录

本目录不是大型数据集存储。它只保存可进入 Git 的数据契约与索引，例如：

- schema 和字段字典；
- 数据版本 manifest；
- 许可、来源和隐私说明；
- 不含敏感信息的统计摘要；
- 极小、脱敏、用于测试的固定样例。

`raw/`、`interim/`、`processed/` 和 `external/` 会被 Git 忽略。大型数据必须存放到团队认可的外部存储，并通过不可变 URI、版本和 hash 引用。

任何新增数据版本都应由 `src/data/` 中的可重复流程生成，并在 manifest 中记录代码 commit、输入版本、配置、随机种子、记录数、schema 版本和内容 hash。

## 已有数据契约

- [elo_ranking](elo_ranking/vote_schema.json)：线上 ELO 有效投票标准化为 `winner_model` / `loser_model` 后的输入契约和虚构样例。
