# 评测配置

新任务从 **`unified.example.json`** 开始，推荐用统一入口的 `init` 命令生成模型配置。
端点和密钥填写到忽略的本地环境文件，格式参考 `endpoints.env.example`。

| 文件 | 用途 |
|---|---|
| `unified.example.json` | 统一运行器的完整配置样例 |
| `endpoints.env.example` | actor/judge/simulator 端点环境变量 |
| `sources.lock.json` | 官方仓库提交与数据 revision，作为输入来源依据 |
| `requirements-unified.lock.txt` | 统一运行环境依赖 |
| `minimax.example.json`、`rmtbench.example.json` | 统一入口内部使用的 benchmark 基础配置 |
| `minimax.prompts.json` | MiniMax 本地补充提示词与来源说明 |
| `characterbench.example.json`、`coser.example.json` | 原生协议工具的参考配置，不是统一入口的模型配置 |
| 其他 requirements、`.env.example`、`api-smoke.example.json` | 安装或原生审计工具的配套文件 |
| [historical/](historical/README.md) | 过去按模型/checkpoint 保存的实验配置 |

选择角色格式、心理可见性等实验条件时，生成新的配置和运行目录。更换模型无需复制 benchmark 代码。
