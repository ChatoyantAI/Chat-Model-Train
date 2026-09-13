# 四套角色扮演基准：运行手册与官方对齐记录

核对日期：2026-09-11。来源以 `configs/evaluation/roleplay_benchmarks/sources.lock.json` 为准。

运行状态更新（2026-09-13）：31 项离线测试通过；DeepSeek V4 Flash 与 Gemma 4 12B IT 的 GLM 替代裁判小样本完成。详见 [试跑摘要](ROLEPLAY_SUITE_PILOT_SUMMARY_20260913.md)。下文的通用入口与官方协议说明，不代表该试跑使用了全部官方模型和参数。

## 评测入口

| 基准 | 对应脚本 | 当前可执行阶段 |
|---|---|---|
| CoSER | `scripts/evaluate_coser.py` | inspect、prepare、run（evaluate） |
| CharacterBench | `scripts/evaluate_characterbench.py` | inspect、prepare、run（generate/prompts/judge/aggregate） |
| RMTBench | `scripts/evaluate_rmtbench.py` | inspect、prepare、run（generate/judge/aggregate/evaluate） |
| MiniMax | `scripts/evaluate_minimax_roleplay.py` | inspect、prepare、run（generate/judge/aggregate/evaluate） |

CoSER 和 CharacterBench 的评测代码从官方 GitHub 仓库以固定 commit 的 Git 子模块拉取，本仓库的脚本负责配置、启动和结果检查。RMTBench 和 MiniMax 使用官方 Hugging Face 数据，本仓库补充了可运行的流程；具体协议和补充设置见各自章节。
所有 `run` 默认 dry-run，只有显式 `--execute` 才调用模型。

## 0. 下载与固定版本

```bash
python scripts/download_roleplay_benchmarks.py --env-file .local/roleplay-benchmarks.env
python scripts/roleplay_benchmarks.py verify all
```

HF_TOKEN 只用于下载，不是推理 API Key。其他使用者填写自己的 Token。
数据下载没有启动模型、下载 GPU 权重或花费 LLM API 额度。

## 1. CoSER

官方：<https://github.com/Neph0s/CoSER>。

流程：200 个指定情境 → 被测模型扮演各角色 → 环境模型补环境 → next-speaker 模型决定发言 → 4 个维度分别裁判 → 汇总四项 0–100 分和 BLEU/ROUGE-L。

原始官方入口在 CoSER 根目录：

```bash
python gca_evaluation/main.py --test_file data/test/test_set.json --actor_model MODEL --judge_model gpt-4o-2024-08-06 --env_model gpt-4o-2024-08-06 --nsp_model gpt-4o-mini-2024-07-18
```

本仓库不直接在 vendor 里运行，以免缓存、日志和临时配置污染官方 checkout：

```bash
python scripts/evaluate_coser.py inspect --output outputs/roleplay_benchmarks/coser-preflight.json
# 先复制 coser.example.json 到自己的配置文件并填写模型版本、精度和模型 ID
python scripts/evaluate_coser.py prepare --config PATH_TO_YOUR_CONFIG --run-dir outputs/roleplay_benchmarks/coser-RUN_ID
python scripts/evaluate_coser.py run --run-dir outputs/roleplay_benchmarks/coser-RUN_ID
# 确认 dry-run 后，在独立环境中正式执行
python scripts/evaluate_coser.py run --run-dir outputs/roleplay_benchmarks/coser-RUN_ID --python PATH_TO_ENV_PYTHON --execute
```

必须准备：

- `BENCHMARK_BASE_URL`、`BENCHMARK_API_KEY`，端点要能访问配置里的 actor、judge、environment、next-speaker 四种模型。
- 如与论文比较，使用作者指定的 GPT-4o 2024-08-06 裁判。配置中的环境与 next-speaker 快照是显式配置，不表示当前服务仍提供这些模型。
- CoSER requirements 之外还需要 `nltk`、`rouge` 等，且旧 LangChain 导入路径需要兼容版本，不能盲装最新版。
- NLTK 的 `punkt` / `punkt_tab` 分词资源按实际 NLTK 版本准备。

保留官方：首次 temperature=0，格式再生成使用 temperature=1；fictional thoughts、去 thoughts、非 GPT/Claude 输出的重复清理、扣分公式全部保留。
`num_workers` 在原脚本中主要按模型并行；单一 actor 配置不是同时并行 200 个场景。
每次重复独立建运行目录，避免上游模拟输出路径复用。

产物：`upstream/exp/simulation/`、`upstream/exp/evaluation/`、`evaluate.log`、`coverage.json`。

## 2. CharacterBench

官方：<https://github.com/thu-coai/CharacterBench>。
官方裁判：<https://huggingface.co/thu-coai/CharacterJudge>（约 8B、BF16；权重不属于前述 441 MB 数据）。

流程：13 个测试文件，每个 250 条；每条生成中英文回复 → 原始脚本构造对应维度 Judge prompt → 本地 CharacterJudge 评分 → 原始 all_eval.py 汇总。
全部双语是 **6,500 个候选生成请求**，不含重试与裁判生成。

```bash
python scripts/evaluate_characterbench.py inspect --output outputs/roleplay_benchmarks/characterbench-preflight.json
python scripts/evaluate_characterbench.py prepare --config PATH_TO_YOUR_CONFIG --run-dir outputs/roleplay_benchmarks/characterbench-RUN_ID
python scripts/evaluate_characterbench.py run --run-dir outputs/roleplay_benchmarks/characterbench-RUN_ID --stage generate
python scripts/evaluate_characterbench.py run --run-dir outputs/roleplay_benchmarks/characterbench-RUN_ID --stage prompts
python scripts/evaluate_characterbench.py run --run-dir outputs/roleplay_benchmarks/characterbench-RUN_ID --stage judge
python scripts/evaluate_characterbench.py run --run-dir outputs/roleplay_benchmarks/characterbench-RUN_ID --stage aggregate
```

上述默认只输出官方命令。正式执行各阶段时加 `--execute --python PATH_TO_ENV_PYTHON`。
生成用候选 API；裁判阶段用 `CHARACTERJUDGE_PATH` 指向已下载、固定 revision 的本地模型。

### 官方评测口径

- 官方中文 prompt 为 30 字以内，英文为 **15 characters**，不是 15 words。
- 官方请求要求 JSON 对象；发布代码会把解析后的对象整体交给评分 prompt，而不是显式只取 `response` 字段。当前没有擅自改变这个表示。
- Judge payload 字典写 temperature=0，但本地 `model.generate()` 只传 `max_new_tokens`，没有传 temperature 或 do_sample。因此要同时核对模型自带 generation_config；不能声称实际必定为 temperature=0。
- 官方模型当前 generation_config 包含 `do_sample=true, temperature=0.7, top_p=0.8, top_k=20`。来源：<https://huggingface.co/thu-coai/CharacterJudge/blob/c3437c5/generation_config.json>。正式下载需使用完整 revision 并记录文件哈希。
- 上游 all_eval.py 的部分解析失败分支会返回默认分数。包装另做覆盖/解析审计，不能把解析失败当真实 3 分。

### 全量离线检查发现的 3 个问题

以下行号是 JSON 数组 **从 0 开始** 的索引，不是文本文件行号：

| 文件 | 索引 | 语言 | 官方预处理问题 |
|---|---:|---|---|
| empathetic_responsiveness_test.json | 208 | en | 最后出现未匹配已知用户/角色名称的 speaker |
| morality_robustness_test.json | 133 | en | 对话最后一条被识别为 assistant，违反末条必须是 user 的断言 |
| morality_robustness_test.json | 215 | en | 同上 |

中文 3,250/3,250 通过，英文 3,247/3,250 通过。由于官方生成脚本把中英文放进同一个 try，英文异常还会影响同条中文的执行。
当前严格模式会先产生 `input_audit.json` 并阻止付费调用；没有自动删掉这 3 条、调换角色或续写问题。
修复数据或显式排除后，都需要新数据版本/协议名和公开分母，不能继续说是原封不动的官方全集。

## 3. RMTBench

官方数据：<https://huggingface.co/datasets/xiangh/RMTBENCH>。
论文：<https://aclanthology.org/2025.findings-emnlp.730/>，第 4.3、5.1、5.2 节与附录 E。

```bash
python scripts/evaluate_rmtbench.py inspect --output outputs/roleplay_benchmarks/rmtbench-preflight.json
python scripts/evaluate_rmtbench.py prepare --config configs/evaluation/roleplay_benchmarks/rmtbench.example.json --run-dir outputs/roleplay_benchmarks/rmt-RUN_ID
python scripts/evaluate_rmtbench.py run --run-dir outputs/roleplay_benchmarks/rmt-RUN_ID
```

官方方法是：读取角色资料和预设用户话语，逐轮让被测模型自己回复，将自己的历史回复带入下一轮；再由 Qwen2.5-72B-Instruct 对指定维度评分。
论文普通质量维度为 1–5 分；角色身份、安全和用户偏好判断使用各自的 Yes/No 模板。不是把用户话语单独发给模型，也不是替换成参考 Assistant 历史。

论文使用 Claude 3.5 文件；Claude 4.5 是更新版，不混合。
当前可以读取原始 blocks，但完整论文的两段长对话/角色涉及四个 block 的抽选和拼接；精确拼接表、随机状态及完整 runnable harness 没有在发布材料中找到。Poison 数据明确未公开。
附录用户偏好模板的 Answer 区域写成了 `{response}`，与描述的参考答案含义冲突，也需要作者确认或明确记录修正。

### 本仓库实现：rmtbench-public-paper-v1

- `prepare` 下载并校验固定 SHA-256 的论文 v2 HTML，自动提取附录 E 表 8、10、11、12 的裁判模板与量表。正文不复制进 Git；提取结果保存到运行目录的 `prompts.json`。
- 表 10 的 Answer 占位符由 `{response}` 修正为 `{answer}`，实际使用数据里的参考答案；该修正写入运行记录。参考答案和参考问题不进入 actor 请求。
- Actor 的角色卡包装由本仓库提供；生成时保留模型自己的全部历史，不截断，不改预设用户消息。
- 默认 `mixed_four_blocks`：每个角色每次从四个公开类别各选一个 block，然后打乱 block 顺序；块内顺序不变。默认两个组合，随机种子 42。组合清单冻结到 `cases.json`，并非作者当年的确切组合表。
- 可选 `single_blocks`：分别评测全部独立 block，每块清空历史。此模式不使用 `repeats` 复制 block，与长对话模式分开比较。
- EE、EC、PA、CU 对每条回复分别请求一次裁判，输出理由与 `Score: [[1..5]]`。CM 只评价 character_maintenance 块，`[[No]]`（未暴露 AI 身份）计成功；UPA 只评价偏好块末尾对应参考问题的回复，`[[Yes]]`（匹配参考答案）计成功。
- Claude 3.5 偏好题按源数据顺序对应最后两轮，Claude 4.5 对应最后一轮；该映射也记录在 cases 中供检查。
- 评分保留原模板的“角色资料、当前问题、当前回复”输入；不额外给 Judge 拼接历史。SEC 没有公开数据，不计分。
- 按语言、维度分别输出有效评分均值与数量。1–5 原分保留；辅助百分制为 `mean / 5 × 100`，不是 `(mean - 1) / 4`。CM/UPA 输出成功率。没有补一个包含缺失 SEC 的七维总分。
- Actor `sampling={}` 表示不传温度等采样字段，使用服务端默认值；因此对比时须固定并记录服务端生成配置。Judge 示例 temperature=0、max_tokens=1024 是本地执行配置，不冒充论文未说明的参数。

固定用户脚本可能与自由生成的剧情发生衔接冲突。此指标应解释为固定脚本下的多轮表现，配合 MiniMax 动态交互结果看。

## 4. MiniMax Role-Play Bench

官方：<https://huggingface.co/datasets/MiniMaxAI/role-play-bench>。

```bash
python scripts/evaluate_minimax_roleplay.py inspect --output outputs/roleplay_benchmarks/minimax-preflight.json
python scripts/evaluate_minimax_roleplay.py prepare --config configs/evaluation/roleplay_benchmarks/minimax.example.json --run-dir outputs/roleplay_benchmarks/minimax-RUN_ID
python scripts/evaluate_minimax_roleplay.py run --run-dir outputs/roleplay_benchmarks/minimax-RUN_ID
```

流程概述：95 个公开场景种子 → 用户模拟器与被测模型交替对话 → 每个场景三次 → 按 20-turn chunk 裁判 → Worlds / Stories / Preferences 汇总。
原文称 100 dialogue turns；实际所有已发布轨迹的 num_turns 都是 102 条消息（含开场等），不能误当作 100 次问答对。

已发布模型轨迹、评分可直接检查，**不能把这些用户后续回复固定照搬成对新模型的自适应模拟用户**；那会变成另一个协议。

缺：用户模拟器完整 prompt、实际裁判模型、评分 prompt 与完整聚合代码。
官方讨论区截至本次核对仍未给出这些问题的解答：

- <https://huggingface.co/datasets/MiniMaxAI/role-play-bench/discussions/2>
- <https://huggingface.co/datasets/MiniMaxAI/role-play-bench/discussions/3>

### 本仓库实现：minimax-public-method-v1

完整流程已经实现，提示词保存在 `configs/evaluation/roleplay_benchmarks/minimax.prompts.json`：角色生成、用户模拟器和裁判文案均为本地补充，依据公开维度编写，后续可替换成作者模板。修改后的提示词会产生新的协议标识。

1. 以种子里的 AI 开场和第一条用户消息启动；模拟用户根据当前实际对话生成后续消息，不使用官方旧轨迹中的用户续句。
2. 默认生成 100 条新消息，即 50 条 actor 回复和 50 条模拟用户回复；加两条种子消息共 102 条。最后一条模拟用户回复保留在轨迹中，不产生额外 actor 请求。这是本实现的明确计数约定。
3. 默认每个种子独立运行三次。100、20、3 均可配置，但不同配置分别比较。
4. 将新消息每 20 条组成一个 target chunk；前面完整对话作为 prior_context。只评分 target chunk 中新生成的 actor 回复，种子开场和模拟用户不评分。
5. 每块一次 Judge 请求，返回六维 `score + reason` JSON。维度与官方名称对应；0–100 的具体缺陷程度档位为本地补充，见提示词文件。
6. 同一会话按块内 actor 消息数加权平均；同一种子先平均重复实验，再按种子等权平均，中文英文分别汇总。
7. `Worlds = mean(basics, logic, knowledge)`；`Stories = mean(diversity, content_logic)`；`Preferences = interaction`；`Overall = Worlds×0.5 + Stories×0.25 + Preferences×0.25`。50/25/25 来自公开榜单，类别内部等权为本地选择。

部分完成时，只纳入六维均有效且全部块评分完成的会话，显示覆盖率；缺失会话不补零。最终模型对比应等待相同样本全部完成。这套分数用于本协议下的模型横向比较，不直接接入原官方榜单。

## 5. RMTBench / MiniMax 的配置与执行

仅需 Python 3.11+ 标准库，调用 OpenAI 兼容的 `/v1/chat/completions` 服务；不依赖本机 CUDA。Actor、Judge、Simulator 可使用不同端点与 Key。

先复制示例配置到自己的配置文件，替换所有 `REPLACE_WITH...`，填写模型 ID、不可变 revision 和采样设置。`base_url_env` 与 `api_key_env` 填环境变量名，密钥只存在进程环境中。`prompt_file` 相对仓库根目录解析；JSON 模板中非占位符的花括号应写成 `{{` / `}}`。

PowerShell 设置示例（值自行填写，不把实际密钥提交到仓库）：

```powershell
$env:ACTOR_BASE_URL = 'http://127.0.0.1:8188/v1'
$env:ACTOR_API_KEY = 'YOUR_ACTOR_KEY'
$env:JUDGE_BASE_URL = 'https://YOUR_JUDGE_HOST/v1'
$env:JUDGE_API_KEY = 'YOUR_JUDGE_KEY'
# MiniMax 还需要：
$env:SIMULATOR_BASE_URL = 'https://YOUR_SIMULATOR_HOST/v1'
$env:SIMULATOR_API_KEY = 'YOUR_SIMULATOR_KEY'
```

正式执行前，先用 `limit_cases` 选少量案例，保留默认请求上限；确认结果后另建全量运行目录。

```bash
# 用填写完成的配置准备运行目录
python scripts/evaluate_rmtbench.py prepare --config PATH_TO_YOUR_CONFIG --run-dir outputs/roleplay_benchmarks/rmt-RUN_ID
# 默认预览请求数量和协议；不调用模型
python scripts/evaluate_rmtbench.py run --run-dir outputs/roleplay_benchmarks/rmt-RUN_ID
# 生成、裁判、汇总全流程；这次最多允许 100 个实际请求（含失败重试）
python scripts/evaluate_rmtbench.py run --run-dir outputs/roleplay_benchmarks/rmt-RUN_ID --stage evaluate --execute --max-requests 100
# 重复同一条命令会继续未完成项；有效请求命中缓存，不重复计费
# 确认预算后可调高 --max-requests；它是调用次数上限，不是美元额度

# MiniMax 使用相同阶段接口
python scripts/evaluate_minimax_roleplay.py run --run-dir outputs/roleplay_benchmarks/minimax-RUN_ID --stage evaluate --execute --max-requests 100
# 也可分开执行 --stage generate / judge / aggregate
```

- `concurrency` 是同时进行的会话数；一个会话内部严格按顺序生成。示例为 4，可按服务能力调高。
- 示例 `max_attempts=3`：429、部分临时 HTTP 错误、网络错误、空内容、截断和解析错误会重试；429 使用 Retry-After，单次最多等 300 秒。401 等配置错误不重试。
- 请求、最终 content、finish_reason、usage、解析结果都存入 `cases/<id>/requests/`。不保存 Authorization、推理正文或原始 HTTP 错误正文。
- 请求上限按单次命令计，断点续跑会重置本次计数。缓存减少重复调用；但服务端成功、客户端未收到回复的网络故障可能导致再次计费，无法保证 exactly-once。
- 完整输出缓存可跨本次运行的失败恢复；模型、数据、提示词或代码修改需另建运行目录。运行期间同一角色的服务地址也不得变更。
- `progress.json` 查看进度，`summary.json` / `summary.md` 看结果，`trajectory.json` 看对话，`judgments.json` 看逐项理由。失败或缺失不当成 0 分。
- 未完成时命令退出码为 2；完成为 0。同一运行有进程锁，崩溃后需确认旧进程已停止再删除 `.run.lock`。
- 费用仅统计服务端实际返回的 cost；没有返回则为未知，不能按 0 美元解释。Token 用量包含已记录的失败尝试，但不含超时后未返回的计费信息。

## 汇总与验证

```bash
python scripts/roleplay_benchmarks.py report --runs outputs/roleplay_benchmarks/coser-RUN_ID outputs/roleplay_benchmarks/characterbench-RUN_ID --output outputs/roleplay_benchmarks/report.json
python -m unittest discover -s tests/evaluation -p 'test_benchmark_*.py' -v
```

结果分 benchmark、语言、协议 ID 展示，不把不同基准的分数平均成一个总分。RMTBench/MiniMax 的运行目录也可传入统一 report 命令。
调用失败、缺样本、输入不可处理与模型真实低分必须分开。GLM 适配的小样本 GPU/API 试跑已完成，四套全量官方协议评测尚未完成。
