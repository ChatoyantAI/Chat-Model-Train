# Chat Model Train

本仓库用于维护聊天模型训练相关的数据处理、训练、评测代码与可复现记录。`main` 是团队唯一的稳定集成分支；所有开发都必须从最新 `main` 创建独立分支，并通过 Pull Request 合并回 `main`。

> 原仓库内容保存在 `back` 分支。`back` 只用于查阅与迁移，不作为新开发的起点。

## 公开角色扮演评测

[统一评测运行说明](docs/evaluation/UNIFIED_PUBLIC_EVALUATION.md)：克隆仓库后运行安装脚本，配置候选模型与裁判的 API 端点，即可启动 CharacterBench、CoSER、MiniMax、RMTBench。评测不负责模型部署。

[四套评测数据](data/evaluation/roleplay_benchmarks/README.md)已随仓库提供，约 60.4 MiB；包含固定原始输入、来源版本和逐文件校验信息。

## 基本原则

1. **分支开发**：禁止直接向 `main` 提交或推送代码。
2. **职责明确**：每个模块必须有负责人、边界、输入输出、运行方式和验证方法。
3. **变更可复现**：训练和评测结果必须能追溯到代码提交、数据版本、配置与随机种子。
4. **代码与产物分离**：代码、配置、固定评测输入、schema、manifest 和小样例可以进 Git；训练数据、模型权重、密钥及大型运行产物不得进入 Git。
5. **文档与代码同步**：接口、目录、配置或流程变化时，相关 README 必须在同一个 PR 中更新。

详细规则见 [CONTRIBUTING.md](CONTRIBUTING.md)，模块负责人登记见 [docs/RESPONSIBILITIES.md](docs/RESPONSIBILITIES.md)。

## 仓库结构

```text
Chat-Model-Train/
├── src/
│   ├── data/                 # 数据抽取、清洗、转换、质量检查
│   ├── training/             # 训练入口、训练编排与 checkpoint 管理逻辑
│   ├── evaluation/           # 固定评测集、推理、指标与报告生成
│   └── common/               # 跨模块且稳定的公共能力
├── configs/                  # 可版本化的运行配置
├── data/                     # 固定评测输入、数据规范、manifest、schema 和小样例
├── tests/                    # 单元、集成与回归测试
├── scripts/                  # 面向人的薄封装脚本
├── docs/                     # 流程、职责、设计与实验记录
└── .github/                  # PR 模板等 GitHub 协作配置
```

目录的详细边界见 [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md)。新模块必须使用 [模块 README 模板](docs/templates/MODULE_README_TEMPLATE.md) 建立自己的说明文件。

## 标准开发流程

```bash
git switch main
git pull --ff-only origin main
git switch -c feat/<module>-<short-description>

# 开发、测试、同步文档
git add <files>
git commit -m "feat(<module>): <summary>"
git push -u origin feat/<module>-<short-description>
```

随后在 GitHub 创建指向 `main` 的 PR，完整填写负责人、变更范围、输入输出、验证结果和产物位置。一个 PR 应只解决一个清晰问题。

## 新模块的最低交付

如果负责的模块还不存在，不要把代码临时堆在仓库根目录。至少需要在同一个 PR 中提交：

- `src/<module>/README.md`：负责人、边界、输入输出、入口、运行方法；
- `src/<module>/` 下的实现代码；
- `configs/<module>/` 下的示例或默认配置（适用时）；
- `tests/<module>/` 下的最小验证（适用时）；
- 相关 schema、manifest 或极小脱敏样例（适用时）；
- [docs/RESPONSIBILITIES.md](docs/RESPONSIBILITIES.md) 中的负责人登记。

若当前 PR 只建立规范或接口、暂时没有实现代码，必须在模块 README 中明确状态、未完成项和后续负责人，不能用空目录代替说明。
