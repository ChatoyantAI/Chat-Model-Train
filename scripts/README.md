# 脚本目录

`scripts/` 保存面向开发者或运维人员的薄封装入口，例如环境检查、调用正式 CLI 或批量编排。核心数据、训练和评测逻辑必须位于 `src/<module>/`，以便复用和测试。

每个脚本应：

- 提供 `--help` 或文件头说明；
- 校验参数并在失败时返回非零退出码；
- 不硬编码密钥、本机路径或生产资源；
- 在模块 README 中有对应运行示例；
- 尽量调用正式模块接口，而不是复制实现。

## 已有入口

- `compute_elo_ranking.py`：调用 `src/evaluation/elo_ranking`，从标准胜负票 CSV 生成批量 Bradley–Terry 排名。
- `build_production_context.py`：从冻结 ELO 数据单独重建一场 comparison 的一个候选上下文。
- `build_roleplay_judge_dataset.py`：批量生成单回复 Judge 输入、隔离标签、上下文审计和 manifest。
