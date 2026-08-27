# 贡献与协作规范

本文定义本仓库的分支、模块职责、PR、测试、数据和实验记录要求。除紧急恢复操作外，所有变更都必须遵循本流程。

## 1. 分支与同步

所有分支必须从最新远端 `main` 创建：

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git switch -c <type>/<module>-<purpose>
```

推荐分支格式：

- `feat/data-add-normalizer`
- `fix/evaluation-metric-denominator`
- `docs/training-runbook`
- `refactor/common-config-loader`
- `chore/repository-tooling`

分支不得长期偏离 `main`。创建 PR 前应合入或 rebase 最新 `origin/main`，解决冲突并重新运行验证。禁止直接向 `main` push，禁止在共享分支上无说明地强推。

## 2. 模块职责

每个模块必须在 [docs/RESPONSIBILITIES.md](docs/RESPONSIBILITIES.md) 登记：

- 主负责人及 GitHub 账号；
- 备份负责人或主要 reviewer；
- 负责目录；
- 维护范围和明确不负责的边界；
- 当前状态。

负责人对模块的可运行性、文档、接口稳定性、review 响应和结果可追溯性负责，但不代表其他人不能贡献。跨模块变更必须同时邀请相关模块负责人 review。

新模块不存在时，提交者必须创建规范目录，并按照 [模块 README 模板](docs/templates/MODULE_README_TEMPLATE.md) 补齐说明。禁止只提交孤立脚本、没有 README 的模块目录，或把临时代码放在仓库根目录。

## 3. 代码和目录要求

- 可复用业务逻辑放在 `src/<module>/`；`scripts/` 只能是调用模块代码的薄入口。
- 配置与代码分离，配置放在 `configs/<module>/`，不得硬编码本机路径、token、模型地址或数据版本。
- 测试按模块放在 `tests/<module>/`，测试目录应能看出它验证的模块。
- 公共代码只有在至少两个模块真实复用且接口稳定时才能进入 `src/common/`。
- 输入输出格式必须有 schema、类型定义或明确的字段表；不允许只靠口头约定。
- CLI 或入口脚本失败时必须返回非零退出码，并输出可定位问题的信息。

## 4. 数据、模型和隐私

Git 中允许保存：

- 数据 schema、manifest、校验规则和统计摘要；
- 不含隐私、可公开且足够小的测试样例；
- 数据或模型产物的版本号、hash、对象存储路径和生成命令。

Git 中禁止保存：

- 生产聊天正文、个人身份信息或未脱敏用户数据；
- 大型 Parquet/Arrow 数据集、模型权重、checkpoint 和运行缓存；
- API key、密码、cookie、私钥、服务账号文件或 `.env`；
- 无法说明来源、许可或生成过程的数据。

产物应放在团队约定的对象存储或训练服务器，并在文档中记录不可变路径、hash、生成时间和权限要求。任何疑似密钥或隐私数据都必须在 PR 前移除；若已经提交，立即通知仓库管理员处理历史，而不是只做一次普通删除提交。

## 5. 训练与评测可复现性

训练或评测 PR 至少记录：

- 代码 commit；
- 数据集名称、版本、时间窗口及 manifest/hash；
- 完整配置文件和重要命令行覆盖项；
- 模型、tokenizer、依赖环境与硬件信息；
- 随机种子；
- 输出产物位置与 hash；
- 关键指标、失败样本和已知限制。

推荐复制 [实验记录模板](docs/templates/EXPERIMENT_RECORD_TEMPLATE.md) 创建 `docs/experiments/<experiment-id>.md`。不得用同一个版本号覆盖已有数据、模型或评测结果。

## 6. Commit 规范

推荐使用以下格式：

```text
<type>(<module>): <summary>
```

常用 `type`：`feat`、`fix`、`docs`、`test`、`refactor`、`perf`、`chore`。示例：

```text
feat(data): add dialogue schema validator
fix(evaluation): exclude invalid judge responses
docs(training): document checkpoint resume contract
```

Commit 应保持单一意图，不要混入无关格式化、生成产物或个人环境文件。

## 7. Pull Request 要求

所有 PR 必须以 `main` 为目标分支，并完整填写 `.github/pull_request_template.md`。最低要求：

- 明确模块负责人和 reviewer；
- 解释为什么改、改了什么、不改什么；
- 列出输入、输出及任何 schema/接口变化；
- 提供实际执行过的验证命令和结果；
- 说明数据、模型、配置、依赖和兼容性影响；
- 同步更新模块 README、职责表和实验记录；
- 确认没有密钥、隐私数据、大文件或本地绝对路径。

禁止以“未测试”“同上”“看代码”等文字代替必要信息。不适用的条目应写明原因。

## 8. Review 与合并

PR 满足以下条件后才可以合并：

- 自动检查通过；
- 至少一名非作者 reviewer 批准；
- 涉及模块获得该模块负责人或备份负责人的批准；
- 阻塞性评论已解决；
- 文档、测试和实现保持一致；
- PR 分支没有未解决冲突。

默认使用 Squash Merge，使 `main` 上一个 PR 对应一个清晰提交。合并后删除远端功能分支。仓库管理员应在 GitHub 为 `main` 开启分支保护，至少要求 PR、审批和状态检查。

## 9. 完成定义

代码“写完”不等于任务完成。一个模块变更只有在实现、验证、配置、文档、负责人登记和可追溯记录全部补齐后，才算完成。
