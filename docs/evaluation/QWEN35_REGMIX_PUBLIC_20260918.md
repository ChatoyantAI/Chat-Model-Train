# Qwen3.5-9B 原版与 regmix 微调版公开评测

只运行 CharacterBench（6,497 条有效双语输入）和 CoSER（200 个场景）。
复用迁移任务中冻结的输入、提示词、评分解析与 GLM 5.3 Flash 裁判；
这属于项目的 GLM 裁判协议，不是官方榜单成绩。

| 候选模型 | HF revision | 服务 | GPU |
| --- | --- | --- | --- |
| Qwen/Qwen3.5-9B | c202236235762e1c871ad0ccb60c8ee5ba337b9a | qwen35-base / 8200 | 0 |
| ChatoyantAI/qwen3.5-9b-roleplay-sft-regmix-v1 | b198c30431e45c356424c27760f06d7535f76d8e | qwen35-regmix / 8201 | 1 |

微调仓库是 rank 16 LoRA（116,429,720 字节），加载在同一个原版 checkpoint 上。
两者 BF16、不量化、actor 温度 0.7、max_tokens 8,192、64K 服务上下文、
enable_thinking=false。CoSER 最大 20 轮，seed=42+场景编号。
GLM 请求沿用已恢复成功的历史 provider-default 参数，不添加关闭思考文本或额外 thinking 字段。
小说角色的内心活动仍遵循 CoSER 原提示，与模型 reasoning 模式分别记录。

模型永久保存于 `/workspace/sft_data/models`；`/dev/shm/models/Qwen3.5-9B`
是临时推理副本。每个分片核对 HF 字节数、完整 safetensors 元数据与 safe_open 校验；
检查 LoRA 与原版 tokenizer 的词表一致，并验证非思考模板。
首次启动缺少 PATH 中的 ninja 已修正，vLLM 日志及失败启动证据保留。

## 启动与恢复

现有服务器运行环境：vLLM `/opt/vllm-0.29`，评测 `/opt/eval-recovery`。
`DEEPINFRA_API_KEY` 从环境变量或容器初始进程环境读取，不写入脚本、日志或 Git。
先将以下脚本及已有的 `recovery_queue_guard.py`、`async_checkpoints.py`
放到 `/workspace/services/qwen-pair-20260918/`：

同目录 `configs/` 放入仓库的 `public_gemma_verified_v1.json` 和
`qwen35_public_pair.json`。恢复模块使用仓库中已提交的版本；哈希不一致会在请求前退出。

```bash
/opt/vllm-0.29/bin/python /workspace/services/qwen-pair-20260918/prepare_qwen_pair.py --launch
bash /workspace/services/qwen-pair-20260918/launch_qwen_pair_public.sh
```

准备脚本依赖同目录的 `inspect_qwen_pair.py`。评测引用
`/workspace/migrations/old-full-run-20260914` 的冻结协议与 CoSER source，
以及 `/workspace/projects/Chat-Model-Train` 的已有评测依赖。
不删除旧任务目录、不触碰 GPU 2/3 的训练进程。

固定流程配置是 `configs/evaluation/public_gemma_verified_v1.json`，模型入口配置是
`configs/evaluation/qwen35_public_pair.json`。更换候选模型只改后者的模型 ID、revision、
服务端口和部署身份，并使用独立结果目录。冻结输入、提示词、评分 schema、采样、
裁判、异步落盘和缓存恢复机制不随模型变化。vLLM 部署配置属于模型服务，另行管理。

流程配置依据已验证的 Gemma deployment-fb270adf9e1f41c980b760a868785cfc：
API 总上限 128、本地总上限 16、请求超时 600 秒、排队超时 30 秒，
异步 checkpoint 上限 256、写入线程 4，复用相同 SHA256 的两个恢复模块。
两模型并行时各分配 API 64、本地 8，不将多个任务的请求上限重复叠加。
CharacterBench 32 worker、CoSER 8 worker 是冻结脚本原值；此前的 96 是 RMT worker，
本次不运行 RMT。启动入口不提供隐式并发覆盖参数。
预算保护关闭，仍记录 usage、费用、响应、错误、尝试次数；H100 租金另计。
缓存与请求证据保留在原持久运行目录，使用 Gemma 的异步 checkpoint 发布机制，
不引入内存盘工作目录或新的归档格式。
异步发布派生成绩，付费请求响应先保存再交付。

重复执行启动命令可恢复相同目录；文件锁阻止重复 supervisor。
跳过已完成样本，复用请求 checkpoint、原始响应 journal、CoSER 的成功部分缓存。
失败证据按 attempt 保留，不覆盖；已发出但无响应/账本确认的中断请求隔离，
不会自动重发。保存了响应但账本未完成的请求在恢复时仅补记账、不再调用模型。

## 输出目录与进度

```text
/workspace/runs/roleplay-evaluation/qwen35-base-public-20260918/
/workspace/runs/roleplay-evaluation/qwen35-regmix-public-20260918/
```

每个目录包含 `logs/` 与 `outputs/full-public-v1/`。后者包含：

- `qwen-protocol.json`：输入哈希、revision、采样与调度参数；
- `supervisor.json`、`broker-status.json`、`broker-final.json`：状态、调用与费用；
- `qwen35-base/` 或 `qwen35-regmix/` 中各 benchmark 的 progress/summary 和逐样本证据；
- `transport-repair-v1/`：付费请求、响应、账本和恢复 journal。

已有 `qwen-protocol.json` 原样保留，不覆盖首次运行记录；每次启动的 `deployment-*.json`
与 supervisor 中保存当前流程配置、模型配置、脚本哈希和有效调度参数。
恢复可调整明确指定的调度配置，但模型身份、采样、输入、评分协议变化会拒绝复用旧目录。

只有两套覆盖均完整才标记 `completed`；正常退出但缺样本为 `finished_partial`。
按 benchmark 分别比较平均分、各维度、有效/失败数与相同输入交集，不混合总分。

离线验证：`/opt/eval-recovery/bin/python -m unittest -v test_qwen_pair_public`。
验证协议隔离、模型 revision/LoRA 入口、防止中断请求重复计费；测试不调用 API。

2026-09-18 原版恢复时发现两份旧账本异常（NUL 前缀、EIO）。一次性维护脚本
`recover_qwen_ledger_20260918.py` 从已有完整响应重建 212 条账目，核对到暂停前的
668 次本地、841 次 judge 调用和 $0.081795325 费用，未调用模型。
仅在所有调用数和费用与已保存快照一致时应用；异常原文件保留在 `ledger-evidence/`。
这不是正常评测启动步骤，也不改变已验证的恢复模块或评分协议。
