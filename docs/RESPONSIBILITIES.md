# 模块职责登记

本文件是模块所有权的唯一登记表。负责人变化、目录边界变化或新模块创建时，必须在同一个 PR 中更新本表。

“负责人”表示对模块质量和维护结果负责，不表示独占代码；“备份 / Reviewer”负责在主负责人不可用时保持 review 和故障处理连续性。请填写真实 GitHub `@账号`，不得只写昵称或部门名。

## 当前登记

| 模块 | 负责目录 | 主负责人 | 备份 / Reviewer | 维护范围 | 状态 |
|---|---|---|---|---|---|
| Data | `src/data/`, `data/`, `configs/data/`, `tests/data/` | 待认领 | 待认领 | 数据契约、抽取、清洗、版本、manifest、质量校验 | 待认领 |
| Training | `src/training/`, `configs/training/`, `tests/training/` | 待认领 | 待认领 | 训练入口、配置、恢复、日志、checkpoint 产物契约 | 待认领 |
| Evaluation | `src/evaluation/`, `configs/evaluation/`, `tests/evaluation/` | 待认领 | 待认领 | benchmark、推理、指标、Judge、报告与回归门槛；公开角色扮演基准下载与协议适配 | 统一端点评测入口与独立安装已验证；66项回归通过；小样本及真实96并发验证通过，非全量官方复现；见统一运行说明 |
| Common | `src/common/`, `tests/common/` | 待认领 | 待认领 | 稳定的跨模块公共类型、配置、日志与 I/O 能力 | 待认领 |
| Tooling | `scripts/`, 仓库级开发工具 | 待认领 | 待认领 | 本地入口、检查脚本、开发体验 | 待认领 |
| Repository governance | `README.md`, `CONTRIBUTING.md`, `.github/`, `docs/` | 待认领 | 待认领 | 结构规范、协作流程、模板和文档一致性 | 待认领 |

## 认领或移交流程

1. 从最新 `main` 创建分支。
2. 在本表中填写主负责人、备份负责人、准确目录和维护边界。
3. 检查对应模块 README 是否完整；不完整时在同一个 PR 中补齐。
4. 邀请新的主负责人和备份负责人 review。
5. PR 合并后，负责人开始承担该模块后续 review 和文档维护责任。

模块移交不能只修改姓名。移交 PR 还应列出当前运行状态、开放问题、产物位置、关键权限和近期风险。

## 新模块登记规则

新模块必须新增一行，并明确它与现有模块的边界。若同一功能同时改动多个模块，应指定一个牵头负责人，并在 PR 中列出其他模块 reviewer。无法确定所有权的代码不得长期放入 `common` 或 `scripts` 作为规避手段。
