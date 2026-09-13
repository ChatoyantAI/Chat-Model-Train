# Role-play Benchmark Suite

> 状态：四套公开源已同步；下载、版本锁定、校验与执行包装已实现。31 项离线测试通过；两模型 GLM 裁判小样本试跑完成，完整官方协议评测尚未完成。
>
> 所属模块：Evaluation；负责人 / Reviewer 沿用仓库职责登记，未另外指定。

## 目标与边界

整合 CoSER、CharacterBench、RMTBench、MiniMax Role-Play Bench 的官方材料，保留各自协议，不把我们的量表替代为它们的官方评分。
逐项运行命令、配置边界和全量输入检查结果见 [运行手册](../../../../docs/evaluation/ROLEPLAY_BENCHMARK_RUNBOOK.md)。四个独立入口为 `scripts/evaluate_coser.py`、`scripts/evaluate_characterbench.py`、`scripts/evaluate_rmtbench.py`、`scripts/evaluate_minimax_roleplay.py`。前两项调用官方代码；后两项为基于官方数据与公开方法补充的实现，均支持生成、评分、汇总和断点续跑。
第三方源码只通过固定 commit 的 Git 子模块引用；大型数据、Token、模型与评测产物不进入本仓库 Git 历史。

本模块负责来源锁定、可重复下载、文件 SHA-256、独立运行目录和结果汇总。
不负责训练、不自动启动付费评测、不自动接受数据访问条款。

真实模型试跑范围、适配差异及结果见 [2026-09-13 试跑摘要](../../../../docs/evaluation/ROLEPLAY_SUITE_PILOT_SUMMARY_20260913.md)。本模块通用入口与服务器上的实验专用脚本不是同一个执行协议，不能混称官方成绩。

## 一键下载

需要 Python 3.11+、Git，以及可访问 GitHub / Hugging Face 的网络；下载本身不需要 CUDA、PyTorch、OpenAI SDK。

```bash
# 默认同步四套，自动初始化两个固定版本的 Git 子模块
python scripts/download_roleplay_benchmarks.py

# 只下载一套
python scripts/download_roleplay_benchmarks.py minimax

# 使用你自己的本地 Token 文件（不执行其中的 shell 内容）
python scripts/download_roleplay_benchmarks.py --env-file .local/roleplay-benchmarks.env

# 对已下载的所有源校验
python scripts/roleplay_benchmarks.py verify all
```

复制 `configs/evaluation/roleplay_benchmarks/.env.example` 到 `.local/roleplay-benchmarks.env`，填写自己的 `HF_TOKEN`。
也可以自行设置进程环境变量 `HF_TOKEN`，其优先级高于文件。Token 不作为命令行参数，不写进 manifest，不输出到日志。
本机的实际 Token 文件被 `/.local/` 与 `*.env` 排除；示例只有空字段。

RMTBench 需要账号具有官方数据访问权限。401 / 403 返回明确失败，不自动同意协议、不绕过 gated access。
重复执行会验证 SHA-256 并跳过已完成文件；中断后按文件继续，未完成的单个文件重新下载，不承诺字节级续传。
跨域下载重定向会移除 Authorization，不把 HF Token 传给文件 CDN。

## 大小与目录

2026-09-11 当前固定版本本地统计，含公开材料，不含模型权重：

| 来源 | 大小（约，十进制 MB） | 本地目录 | 内容 |
|---|---:|---|---|
| CoSER | 124.1 | `vendor/benchmarks/CoSER/` | 官方源码、200 个测试场景和上游附带资料 |
| CharacterBench | 54.2 | `vendor/benchmarks/CharacterBench/` | 官方源码、13 个测试文件共 3,250 条双语样本、完整评分提示词 |
| RMTBench | 4.0 | `data/external/roleplay_benchmarks/rmtbench/` | 论文版和更新版各中英文，共四个文件，每个 80 个角色 |
| MiniMax | 259.0 | `data/external/roleplay_benchmarks/minimax/` | 95 个场景种子、3,135 段已发布轨迹及评分，JSONL 与 Parquet 原件 |
| 合计 | **约 441 MB / 421 MiB** | | 不含裁判模型与运行目录 |

Git 子模块对象缓存约 56 MB；合计实际占用约 498 MB / 475 MiB。建议初始预留 1 GB。
MiniMax 两种格式为同一批数据的不同表示，不应重复计数；完整镜像保留两者。
每次 `prepare` 会复制一份独立上游工作目录，运行输出和模型权重还会额外占用空间。

## 固定来源与输入契约

`configs/evaluation/roleplay_benchmarks/sources.lock.json` 保存来源 URL、不可变 commit、文件清单、协议状态、阻塞与限制。

- Git：`git submodule update --init` 获取父仓库锁定的版本，随后与 lock 核对；已存在但被修改的上游拒绝使用。
- HF：只下载锁定 revision 的文件，不使用浮动 `main`；每个文件的实际大小、SHA-256 写入被 Git 忽略的 `snapshot.json`。
- 新 revision 不覆盖现有数据；必须另存旧快照后更新版本清单。
- RMTBench 的 Claude 3.5 文件属于论文版，Claude 4.5 文件另列，不混算。

所有文本和角色资料均视为数据，不是执行指令。

## 来源与运行状态

| 来源 | 已同步 | 可用执行接口 | 尚缺 |
|---|---|---|---|
| CoSER | 完整固定代码树及官方测试文件 | `prepare`、`run --stage evaluate` | GLM 适配小样本已运行；官方裁判/辅助模型协议仍需验证 |
| CharacterBench | 固定源码、完整公开测试文件、评分提示词 | generate → prompts → judge → aggregate | GLM 适配小样本已运行；官方 CharacterJudge 仍未验证；全量预处理发现 3 个英文源输入错误，严格模式在付费前阻塞 |
| RMTBench | 四份官方公开数据及论文裁判模板 | 本地实现 generate → judge → aggregate | 未公开的 Poison 数据与确切对话组合；本实现固定并记录公开子集组合 |
| MiniMax | 全部公开数据文件 | 本地实现动态 self-play → 分块 judge → aggregate | 官方完整模拟器和裁判文案未公开；本实现提供可替换的本地文案 |

访问权限解决不代表许可冲突解决：RMTBench 数据卡 Apache 标识与非商业 gate 同时存在；MiniMax 元数据与正文许可不一致；需作者澄清再作商业使用/再分发决定。
本仓库仅保存上游 Git 引用和下载清单，不重新提交这些材料的正文。

## 运行方式

先复制示例配置，填写模型不可变版本、精度和实际服务 ID。不要把实际 API Key 填进 JSON。

```bash
python scripts/roleplay_benchmarks.py prepare --config configs/evaluation/roleplay_benchmarks/coser.example.json --run-dir outputs/roleplay_benchmarks/coser-example
python scripts/roleplay_benchmarks.py run --run-dir outputs/roleplay_benchmarks/coser-example --stage evaluate
```

`run` 默认只是 dry-run。真正执行必须另加 `--execute`，且示例占位符必须全部替换。
模型服务配置由 `BENCHMARK_BASE_URL`、`BENCHMARK_API_KEY` 环境变量提供。CoSER 上游使用同一个兼容端点访问 actor、judge、environment、next-speaker 四个模型；不能把只提供 actor 的 vLLM 地址当成全部服务。

CharacterBench 分四个阶段：

```bash
python scripts/roleplay_benchmarks.py prepare --config configs/evaluation/roleplay_benchmarks/characterbench.example.json --run-dir outputs/roleplay_benchmarks/character-example
python scripts/roleplay_benchmarks.py run --run-dir outputs/roleplay_benchmarks/character-example --stage generate
python scripts/roleplay_benchmarks.py run --run-dir outputs/roleplay_benchmarks/character-example --stage prompts
python scripts/roleplay_benchmarks.py run --run-dir outputs/roleplay_benchmarks/character-example --stage judge
python scripts/roleplay_benchmarks.py run --run-dir outputs/roleplay_benchmarks/character-example --stage aggregate
```

Judge 使用本地 `CHARACTERJUDGE_PATH`，指向自己下载并固定版本的 `thu-coai/CharacterJudge`。权重不由数据下载脚本自动获取。
两套依赖分开安装，示例 requirements 在同一配置目录下；正式环境版本需在目标机器验证和冻结。

## 工程适配与原件保留

不修改 `vendor`。`prepare` 只对 `outputs/.../upstream/` 副本应用精确锚点补丁，并保存完整 `engineering.patch`：

- CoSER：从进程环境读取连接配置，避免在源码 config 中落地明文密钥；不改 prompt、采样温度、评分、thought 清理和重复清理。
- CharacterBench：让 API 包装尊重实际模型参数；去掉无限重试；用 `ast.literal_eval` 替代模型输出的 `eval`；修复第五次成功仍被丢弃的问题；耗尽重试明确报错；裁判路径从环境读取。
- 上游 CharacterBench 英文 prompt 的 15 characters 与中文 30 字限制原样保留，不能私自改成我们平台的篇幅口径。
- 原始裁判/聚合代码保留，工程适配差异记录在 `engineering.patch` 中。

## 输出契约与报告

| 文件 | 内容 |
|---|---|
| 数据 `snapshot.json` | revision、文件字节数、SHA-256、下载错误 |
| 运行 `manifest.json` | 配置、源 commit、准备副本哈希、补丁哈希、阶段状态 |
| `engineering.patch` | 上游副本的全部工程差异 |
| `<stage>.log` | 执行日志；保存在本机 outputs，不提交 |
| `coverage.json` | CharacterBench 覆盖率、重复 ID、解析错误，不改变原分数 |
| 汇总 JSON / Markdown | 各自官方分数、覆盖状态、按协议分组，不计算四套混合总分 |

```bash
python scripts/roleplay_benchmarks.py report --runs outputs/roleplay_benchmarks/coser-example outputs/roleplay_benchmarks/character-example --output outputs/roleplay_benchmarks/report.json
python -m pytest tests/evaluation/test_roleplay_benchmark_downloads.py -q
# 或使用标准库测试下载、原生包装、两套新增流程，不依赖 pytest
python -m unittest discover -s tests/evaluation -p 'test_benchmark_*.py' -v
```

GLM 适配小样本已完成；具体范围见试跑摘要。完整官方裁判流程仍待执行，不能用小样本通过替代全量验证。

## 分支与此前工作恢复

本分支为 `feature/evaluation-gyr`，从 `origin/main` 的 `4fdcb4c` 创建。
此前 `codex/roleplay-sft-prep` 的未提交改动已保存在本机命名 stash：
`snapshot: codex/roleplay-sft-prep before benchmark integration 2026-09-11`。
该 stash 未上传；恢复前先保存当前工作，再切回旧分支并核对 `git stash list` 后 apply 对应快照。不要盲目 pop 最新 stash。
