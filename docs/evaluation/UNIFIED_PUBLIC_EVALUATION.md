# 统一公开评测入口

## 从哪里看

- 导读：[`src/evaluation/roleplay_benchmarks/README.md`](../../src/evaluation/roleplay_benchmarks/README.md)。
- 入口：`scripts/evaluation/run_public_evaluation.py` 是启动器，参数实现在包内 `cli.py`；也可用 `python -m src.evaluation.roleplay_benchmarks`。
- 配置示例：`configs/evaluation/roleplay_benchmarks/unified.example.json`。
- 配置、调度、请求：包内 `config.py`、`unified.py`、`endpoint_transport.py`。
- benchmark：CharacterBench 在 `characterbench.py`，CoSER 在 `frozen_backend.py`，MiniMax/RMT 在 `community.py`。
- 固定历史协议：`scripts/evaluation/frozen_public_v1/`。三份脚本按原字节哈希锁定，不修改。
- 异步写入与独立费用进程的实现在包内 `storage.py`；原脚本导入路径为历史兼容保留。
- 原生官方包装入口另见模块 README；统一入口的 CharacterBench 使用历史 GLM judge，不能混称 CharacterJudge 官方榜单成绩。

模型部署与评测分离。更换模型只新建配置，指定模型 ID、版本、端点环境变量；不复制或修改 benchmark 代码。
运行目录是数据和记录，不是每个模型重新写一套脚本。旧任务继续用其冻结启动器和原缓存，新入口拒绝误读旧版 manifest。

## 从干净仓库启动

首次使用从 `main` 克隆。恢复旧任务时检出其 manifest 记录的 commit，避免混用协议或绕过代码哈希校验。
Python 3.11+ 和 Git 是系统前置条件；无需 CUDA、torch 或 vLLM。以 Linux 为例：

```bash
git clone --branch main https://github.com/ChatoyantAI/Chat-Model-Train.git
cd Chat-Model-Train
python3 scripts/evaluation/setup_public_evaluation.py --benchmarks characterbench coser
```

安装器会建立用户目录下的独立虚拟环境、安装锁定依赖、初始化固定 benchmark 子模块、准备 CoSER tokenizer。
默认环境位于 `~/.cache/chat-model-train/eval-venv`，不是服务器专有路径。
要准备四套，省略 `--benchmarks`；RMT 数据需要自己的 Hugging Face 访问授权，使用 `--env-file` 提供 `HF_TOKEN`。
若已有数据，用 `--cache-from /已有仓库根目录` 复用哈希匹配的下载文件，既不重复下载也不复制旧评测结果。
安装不会发模型请求，不会启动模型服务。Windows 使用 `python` 安装，再用虚拟环境的 `Scripts/python.exe` 调用 Python 入口即可。

```bash
mkdir -p .local
cp configs/evaluation/roleplay_benchmarks/endpoints.env.example .local/evaluation.env
# 编辑 .local/evaluation.env：actor / judge 的 base URL 和 API key
bash scripts/evaluation/run_public_evaluation.sh init \
  --config .local/model.json --actor-model YOUR_SERVED_MODEL_ID \
  --actor-revision YOUR_MODEL_COMMIT --benchmarks characterbench coser
# API 候选模型加 --actor-kind remote；默认候选是 local。
bash scripts/evaluation/run_public_evaluation.sh --env-file .local/evaluation.env check \
  --config .local/model.json --check-api
bash scripts/evaluation/run_public_evaluation.sh --env-file .local/evaluation.env run \
  --config .local/model.json --run-dir /tmp/roleplay-evaluation/my-model-v1 --execute
```

`init` 只生成模型配置，默认 judge 为 GLM 5.3 Flash，可用 `--judge-model` 指定。
新配置默认启用 `assessment.version=20261018`。原有配置不自动改变；需要复现历史协议时加 `--assessment legacy`。
新 CoSER/HER 对照必须使用相同的 `--coser-role-thoughts include` 或 `exclude`，默认 `include`。
它统一裁判看到的内容，不修改 actor 的原生/HER 提示格式。修正依据与限制见[评分审查](PROTOCOL_AUDIT_20261018.md)。
角色端点在密钥文件中设置，本地和 API 使用同样的协议；URL 填 base URL，不附加 `/chat/completions`。
`check --check-api` 只 GET `/models`，不产生付费补全；不提供模型列表的兼容服务可省略 `--check-api`。
`run --config` 首次自动 prepare，以后原样运行就是恢复；已有目录若配置不一致会拒绝，避免混合不同模型结果。
没有 `--execute` 只做准备和预览。仍保留显式 `prepare` 子命令，方便预先准备输入。

`--actor-revision` 应填写模型提交/训练 checkpoint 版本。省略时记录为 endpoint alias，不能当作不可变权重证明。
CharacterBench/CoSER 固定 actor 温度0.7、helper温度0、max_tokens8192；默认 local16 / remote96。
MiniMax、RMT 协议选项沿用已验证配置，不因清理代码调整。
新评分将未完成的 MiniMax 重复/RMT 场景放入 `partial_scores`，`scores` 只使用完整单位；完整性见 `score_coverage`。
MiniMax 快速测试若只选一条 session，应显式设 `repeats=1`；选择不足完整重复数时可以完成调用，但没有完整种子的 headline 分数。

可不调用模型审查已保存的 CoSER JSON（支持嵌套在本地审查文件中的结果）：

```bash
python scripts/evaluation/run_public_evaluation.py audit-results \
  --inputs /path/to/result.json /path/to/paired-cases.json \
  --output /path/to/new-audit.json
```

审查输出包含输入哈希、空内容/角色名占位、缺失角色前缀、同角色台词精确重复、分数公式复算。
这是诊断，不会自动扣分、修改结果、重新生成或发起 Judge 请求；输出文件必须是新文件。

自定义安装目录可传 `setup ... --venv /本地路径`，随后设置 `EVALUATION_PYTHON=/该路径/bin/python`。
网络盘 noexec 时，代码和数据可以持久化，但虚拟环境要装在本机可执行磁盘；换镜像后重跑安装命令。
如果已设置 `NLTK_DATA`，安装和运行请保持同一值。

## 请求、评分、写盘

1. 固定 benchmark 产生原始请求；CharacterBench 为生成后评分，CoSER 按场景推进角色、环境、speaker和评分。
2. 同一进程内所有 benchmark 共享 local16 / remote96 并发门限。每套生产者默认32；额度是上限，不保证短任务始终有96个请求。
3. 按样本、模型版本、实际端点和完整请求查缓存。已完成样本直接跳过；紧凑缓存损坏/缺失时可重放已保存成功响应。
4. 发送前写小型 intent；接收后先保存完整原始 outcome，再交给后续评分。都在本地盘操作，不把网络盘放在请求热路径。
5. 轨迹、判断、进度使用有界异步写入队列（256、4线程）；费用流水由独立进程批量写入（队列512、每批128、最长1秒）。
6. 结束时等待写入确认并汇总一次。进程账本失败时仍能从原始 outcome 恢复费用，不把费用缺失当0。

意图/原始响应的少量同步持久化是断点计费边界；其余派生记录异步。队列满会产生背压，写入失败停止新调用，不无限堆内存。
同一服务器第二个统一入口若使用同一个端点 URL，会拒绝启动，避免并发相加；不限制其他软件或其他服务器的请求。

`/tmp` 是当前机器本地盘，`/dev/shm` 是内存盘，两者都不是迁机备份。
本轮结束并确认进程退出后，完整复制运行目录到 `/workspace`；恢复需包含原始日志、缓存、上游副本、manifest，不能只复制 summary。
用户可以自行转移；运行器不在后台同步网络盘。迁机后先把目录复制回新机器本地盘再恢复。

确定的 HTTP 错误/格式无效可以有限重试，每次保留独立证据；timeout、断连或有 intent 无 outcome 的请求计费状态未知，隔离并报告，不自动重发。
不能保证服务商在超时前没有计费。不要为了恢复直接删除原始请求文件；遗留 `.run.lock` 只在确认所属进程退出后处理。

## thinking 与协议一致性

本地请求使用 `chat_template_kwargs.enable_thinking=false`，若返回 reasoning字段/tokens则停止该运行并保留证据。
API 保留用户后来确认的旧方式，即 `thinking.remote={}`，不额外传关闭参数。它不等于已关闭思考。
2026-09-19 实测 GLM 5.3 Flash 返回 reasoning；[Z.ai 官方说明](https://docs.z.ai/guides/capabilities/thinking)也明确此型号不支持关闭。
不能把 `reasoning_effort=none` 当作可靠关闭方式；本次没有改旧评测策略。
若另测参数，使用新的配置和目录，并记录它与旧成绩不同。

## 产物位置

| 位置（相对运行目录） | 内容 |
|---|---|
| `manifest.json` / `progress.json` | 固定配置、各 benchmark 状态、实际请求数、峰值并发 |
| `transport-journal/<分桶>/<请求hash>/` | 意图、每次原始响应/错误、当前状态；重试不覆盖旧证据 |
| `usage-ledger/batch-*.jsonl` | 独立进程写入的不可变费用流水 |
| `usage-summary.json` | 从原始请求核对的累计请求数、重试、费用未知数、隔离请求 |
| `characterbench/outputs/full-public-v1/candidate/characterbench/` | CharacterBench结果与缓存（实际详细位置见对应 summary） |
| `coser/outputs/full-public-v1/candidate/coser/` | 场景状态、attempt目录、原始完整对话、评分、summary |
| `minimax/`、`rmtbench/` | 各自 cases、请求、judgments、summary |

各套分开报告，不生成四套混合总分。状态完成只在所有所选有效样本完成且异步写入确认后发布。
CharacterBench 原始6500条中有3条英文输入无效，单独记录 source_invalid；可处理6497条，不伪造这3条的成绩。

## 远端固定入口（整理后）

- `/workspace/projects/public-evaluation/`：与发布分支对应的完整仓库副本，代码直接从这里运行。
- `/workspace/services/public-evaluation/current`：指向上述仓库。
- `bash /workspace/services/public-evaluation/run.sh`：同一入口的便捷包装。
- `/workspace/projects/Chat-Model-Train/scripts/evaluation/run_public_evaluation.py`：训练仓库中的便利链接。
- `/workspace/services/public-evaluation/validation/20260919/`：前次实际 API 验证结果与原始记录归档。
- `/workspace/.secrets/evaluation.env`：用户授权保存的密钥；不在 Git 中。

源码不依赖 `/workspace/migrations`、旧 service release、特定模型部署目录或其他工作区。
新服务器无需恢复这些历史目录即可启动新评测。历史结果如需续跑，仍需完整结果目录和对应旧运行器版本。

## 清理边界

2026-10-18 的整理将 CLI、配置和存储实现收回 `src/evaluation/roleplay_benchmarks/`；删除未被调用的旧 `entry.py`。
当前主流程改为显式传入请求客户端和异步存储，移除三组全局函数替换；CharacterBench 运行逻辑从冻结版等价迁出，冻结原件保留。
16 份模型实验配置移动到 `configs/evaluation/roleplay_benchmarks/historical/`，原始字节保持不变。
既有缓存键和请求日志布局不变；已准备任务仍遵守原代码哈希，使用其记录版本恢复，不直接套用新代码或修改 manifest。
历史报告、结果、未提交的其他实验代码未清除。下列内容是此前清理的历史记录。

已移除按 Qwen/Gemma 模型命名的启动、部署、迁移和一次性恢复脚本及配套运行配置；Git 历史 `7bc2bfd` 保留原件。
本地未提交的旧实验脚本移到 `.local/archive/evaluation-legacy-20260919/`，不再作为入口、不打入新发布。
旧远端评测 release/重复脚本移到 `/workspace/archives/evaluation-legacy-20260919/`；请求缓存、模型、训练任务和原始结果保留。

继续保留三份按字节哈希冻结的协议文件、已验证的异步写入器/费用进程、恢复用的原子写入和 CoSER 缓存函数。
它们仍被当前入口引用，不是可随意删除的过时代码。底层原生协议检查和报告库用于审计，不再提供四个竞争的启动脚本。

## 已完成验证与边界

2026-10-18 目录/依赖整理：147 项离线回归全部通过，包含本地 HTTP 并发、缓存重放、异步写入、独立费用进程、HER/CoSER 评分契约与旧缓存恢复。
新增检查逐条比较迁出的 CharacterBench 与冻结版请求、重试及分数，并确认旧请求身份的原始响应可复用且新增调用为 0。
16 份历史配置内容、三份冻结协议源码均保持不变。本次没有请求真实模型、产生 API 费用或更新远端服务。

发布到 `main` 前，在独立 PR 工作区仅导入已提交的评测文件，初始化两个固定子模块，并通过安装器 `prepare_assets(..., cache_from=...)` 复用、核验公开数据缓存。再次执行全部 147 项回归，70.779 秒完成，0 失败、0 跳过；缓存不随 Git 提交。

以下是此前真实 API 验证的记录：

- 真实 CharacterBench 8条 + CoSER case0068完整场景：60请求、全部完成、0请求错误。再次运行新增0请求。
- 独立GLM API探针：96并发、96/96有效、0失败、峰值96；约22.74秒，p50约2.79秒、p95约20.11秒。
  返回的费用估算总计0.0007069美元，重放新增0调用。
- 首次探针为max_tokens128：96个HTTP成功但4条length截断；原记录保留。上述复测仅探针上限改为正式评测已有的8192。
- 传输测试覆盖502保留证据、无效结果重试、同样本并发去重、未知响应隔离、写入失败停止、端点占用拒绝、真实HTTP96并发和缓存重放。
- 此次不是四套全量评测，也不是完整长上下文96场景压测；不能据此承诺供应商未来零故障或推算全量必在几十秒完成。

来源校验只校验固定版本的受版本控制源码；运行过程中生成的未跟踪日志不再阻塞检查，修改过的源码仍会被拒绝。
