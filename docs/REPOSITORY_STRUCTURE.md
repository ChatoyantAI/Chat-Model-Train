# 仓库结构与模块边界

本仓库采用“实现、配置、测试、文档、产物”分离的结构。模块名在这些目录中应保持一致，方便通过路径识别负责人和 review 范围。

## 标准布局

```text
src/<module>/                 # 可复用实现和正式入口
configs/<module>/             # 可提交、可审查的配置
tests/<module>/               # 单元、集成和回归验证
docs/                         # 跨模块规范、设计和实验记录
data/                         # schema、manifest、统计和极小脱敏样例
scripts/                      # 调用 src 中能力的薄封装
```

### `src/data/`

负责数据源适配、抽取、清洗、格式转换、数据契约、manifest 和质量检查。它输出有版本、可校验的数据，不负责模型训练或用最终 benchmark 调参。

### `src/training/`

负责消费版本化数据和配置，启动/恢复训练，记录运行元数据并产出 checkpoint。它不应私自改变数据语义或评测口径。

### `src/evaluation/`

负责固定 benchmark、模型推理、Judge、指标、显著性/误差分析和报告。正式测试集不得被训练代码依赖，评测实现不得暗中修改训练数据。

### `src/common/`

仅保存已被多个模块复用、含测试且接口稳定的基础能力。模块专用 helper 先留在所属模块；不要因为暂时不知道放哪里就放进 `common`。

### `configs/`

配置必须可读、可 diff，默认使用相对路径或环境变量引用。每个正式运行都应保存最终解析后的配置副本。敏感值只能通过秘密管理或环境变量注入。

### `data/`

只提交 schema、manifest、许可说明、统计摘要和极小脱敏测试样例。`raw/`、`interim/`、`processed/`、`external/` 已被 `.gitignore` 排除。大型产物应使用外部不可变存储，并在 manifest 中记录 URI 和 hash。

### `scripts/`

脚本只做参数解析、环境准备和调用 `src/` 正式接口。核心逻辑不得只存在于 shell 脚本或 notebook 中。

### `tests/`

测试路径应与模块对应。修复缺陷时优先增加能复现缺陷的回归测试；外部服务测试必须清楚标记，不能默认消耗生产额度。

## 创建新模块

新模块 PR 至少包含：

```text
src/<module>/README.md
src/<module>/<implementation>
configs/<module>/<example-config>       # 适用时
tests/<module>/<test>                   # 适用时
```

并同步更新：

- `README.md` 的结构或模块导航；
- `docs/RESPONSIBILITIES.md` 的负责人和边界；
- 输入输出 schema、运行说明和示例；
- 依赖声明及必要的实验记录。

Git 不跟踪真正的空目录，因此不能只“创建目录”。若实现将在后续 PR 提交，当前 PR 至少应放入完整 README，明确状态、负责人和后续工作。
