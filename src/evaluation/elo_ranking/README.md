# ELO campaign ranking

## 状态与负责人

- 状态：计算实现可用；`data-check-elo-active-models-v2` 已完成冻结和正式出分；
- 主负责人：`@gaoxiaomo`；
- 备份 / Reviewer：待认领。

## 目标

将线上 ELO campaign 的有效二选一投票转换成可复核的模型分数、完整排名、两两胜负表和不确定性区间。

本模块使用标准 Bradley–Terry 批量极大似然，不使用按时间逐票更新的传统 Elo。UI 左右位置只在数据导出阶段用于确定胜方；进入本模块的输入只有 `winner_model` 和 `loser_model`，位置不会进入拟合、加权或校正。

完整口径与公式见 [计算和数据冻结方案](../../../docs/ELO_RANKING.md)。

## 输入

UTF-8 CSV，至少包含：

| 字段 | 必填 | 说明 |
|---|---|---|
| `user_id` | 是 | 用户标识，用于验证每个用户最多一票 |
| `winner_model` | 是 | 用户选择的模型，已经与左右位置解耦 |
| `loser_model` | 是 | 用户未选择的模型 |
| `comparison_id` | 建议 | 原始比较记录标识，用于追溯 |
| `campaign_id` | 建议 | campaign 标识 |
| `selected_at` | 建议 | 投票完成时间 |

生产聊天正文、角色内容、提示词和左右位置不属于计算输入，也不得提交到 Git。数据契约见 [vote schema](../../../data/elo_ranking/vote_schema.json)，极小虚构样例见 [sample_votes.csv](../../../data/elo_ranking/sample_votes.csv)。

## 输出

- `ranking.csv`：分数、排名、原始胜负和95%区间；
- `pairwise.csv`：每个无序模型对合并左右后的胜负票；
- `summary.json`：计算口径、输入 hash、随机种子、拟合元数据和完整排名。

运行产物应写入被 Git 忽略的 `outputs/`，正式报告再登记到实验记录或不可变对象存储。

## 运行

从仓库根目录执行虚构样例：

```powershell
python -m src.evaluation.elo_ranking `
  data/elo_ranking/sample_votes.csv `
  --output-dir outputs/elo-ranking/sample `
  --campaign-id example-campaign `
  --expected-model-count 3 `
  --bootstrap-samples 200
```

正式11模型数据：

```powershell
python -m src.evaluation.elo_ranking `
  data/raw/data-check-elo-active-models-v2-votes.csv `
  --output-dir outputs/elo-ranking/data-check-elo-active-models-v2 `
  --campaign-id data-check-elo-active-models-v2 `
  --expected-model-count 11 `
  --bootstrap-samples 2000 `
  --seed 20260827
```

也可以使用薄入口：

```powershell
python scripts/compute_elo_ranking.py --help
```

默认要求11个模型的55个模型对全部有有效票，并在发现重复用户时失败。这样不会静默吞掉数据问题。

## 验证

```powershell
python -m unittest discover -s tests/evaluation -p "test_*.py" -v
```

最低验收：

- 两模型60:40时分差约为70.44；
- 所有输出分数平均值为1000；
- 输入顺序不改变结果；
- 重复用户、自身比较、模型数错误和缺失模型对必须显式失败；
- 相同输入和随机种子生成相同的置信区间。
