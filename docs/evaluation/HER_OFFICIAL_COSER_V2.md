# HER 官方 CoSER 接入：有 / 无 system_thinking

统一入口仍是 `scripts/evaluation/run_public_evaluation.py`。新协议为
`her-coser-official-v2`，固定引用 HER 提交
`1e5fd856b9e23d3af41445e3c4ca734e24e012ef` 的实际 `prompts.py` 和 `utils.py`。
源码及 MIT 许可证随仓库提供，`her_upstream/provenance.json` 记录 SHA-256。
旧 `her-official-v1` 只是历史格式替换模式，不再用于新 HER 实验。

## 两种配置

| 设置 | `with-system` | `without-system` |
| --- | --- | --- |
| 对应官方 model_type | `her` | `her_nosys` |
| 角色提示词 | 官方完整 system + role 模板 | 官方完整无 system 模板 |
| 输出长度提示 | 约 200 词、单角色、单轮 | 相同 |
| 发送方式 | 官方 Qwen ChatML → `/v1/completions` | `/v1/chat/completions`，服务器 tokenizer |
| repetition_penalty | 不额外指定 | 不额外指定；2026-09-23 按用户要求取消此前自动注入的 1.15 |
| 原生 thinking | 不注入原生 `<think>`，绕过服务器 chat template | `enable_thinking=false` |
| 自己的历史 | 保留完整 system/role 标签原文 | 保留 role 标签；按官方 chat 处理移除意外 system/native 分析 |
| 别人的历史 | 移除心理活动，保留 HER 动作标签和角色名 | 相同 |
| Judge 输入 | 删除 system，role thought → `[]`，action → `()` | 相同 |

这里的“无 system”是没有 **system_thinking 输出段**，不是删除 system message。
有 system 模式直接采用官方 Qwen ChatML，只适用于兼容该 tokenizer 的模型；
不要把该原始 token 模板直接用于 Gemma。无 system 模式由模型服务应用自身 tokenizer。
两种模式都保留角色心理活动供 Judge 评分。模式选择不会重写基座模型的历史配置。

为 8040 上的 27000 checkpoint 提供了两个完整配置：

- `configs/evaluation/roleplay_benchmarks/historical/qwen3-0.6b-her-27000-coser-official-with-system.json`
- `configs/evaluation/roleplay_benchmarks/historical/qwen3-0.6b-her-27000-coser-official-without-system.json`

两者保留原先 actor 温度 0.7、最多 16384 新 token、32768 上下文、GLM 5.3 Flash
Judge、local 64 / remote 48 / scene 128 上限。不要叠加启动共享相同端点的任务。
训练为 no-system 的 27000 checkpoint 正常使用 `without-system`；另一份用于对照。

## 新模型：只生成配置并填写端点

```bash
python scripts/evaluation/run_public_evaluation.py init \
  --config her-model.json --actor-model YOUR_MODEL --actor-kind local \
  --benchmarks coser --her-system-thinking without-system

python scripts/evaluation/run_public_evaluation.py --env-file /workspace/.secrets/evaluation.env \
  check --config her-model.json --check-api

python scripts/evaluation/run_public_evaluation.py --env-file /workspace/.secrets/evaluation.env \
  run --config her-model.json --run-dir /workspace/runs/roleplay-evaluation/NEW_RUN --execute
```

通用 `init` 保持原有默认并发和 8192 token 上限；上面的 27000 专用配置才是已部署的
64/48/128、16384 设置。新模式必须使用新 run-dir，禁止把新协议结果写入旧运行。
同一目录重复运行会复用成功缓存；请求内容、端点、revision 与模式均进入可核查记录。

## 实现与边界

- `her_official.py`：三种可见视图、官方请求封装、隔离 worker 的补丁。
- `her_upstream/`：未修改的官方 prompt / 工具函数快照；不引入另一套调度系统。
- `frozen_backend.py`：仅修改本次 run 的 worker 副本；vendor 与冻结脚本不变。
- `endpoint_transport.py`：完整原始响应先记账留证，再把原始角色内容返回 worker。
  官方 completions 的 `choices[].text` 也原样留存，不伪装成原始 chat 响应。
- 原有本地 journal、独立进程记账、异步派生文件落盘、共享并发限制继续使用。

这对齐的是 **HER 官方提示词、历史可见性、NSP/环境提示词和 Judge rubric**。
数据仍是本仓库锁定的 CoSER 原始 200 场景，Judge 仍是 GLM 5.3 Flash，
因此不是 HER 论文分数的完整复现。没有换成 HER 训练集，也没有变更 Judge 模型。
有几项明确保留的工程差异：

0. 不自动注入官方 chat 分支的 `repetition_penalty=1.15`，由服务默认值处理。
   先前 `official-nosystem-full-20260923` 全量结果使用了 1.15；取消参数后必须新建
   结果目录，不能继续写入该旧实验。提示词、历史、后处理和 Judge 均保持不变。

1. 空回复、未闭合标签、非法 JSON 必须重试/记失败；不沿用官方解析失败时给 50 分。
2. 官方 `convert_to_coser_format` 只识别 `her`，遗漏 `her_nosys`。
   我们两个分支都转换 role 标签，符合该评测函数“统一为 CoSER 格式”的意图。
3. Judge 视图为缺少前缀的回复补角色名；不会把补过前缀的文本送回模型自己的历史。
4. 保留现有严格缓存和计费证据、固定 seed、token 预算和共享并发；不修改评分公式。
5. 四个评分维度按既有 worker 调度，跨场景并发；不额外叠加一层无限评分并发。

## 验证

`tests/evaluation/test_her_official.py` 直接执行隔离 worker 的模拟和评分函数，
验证两种官方角色模板、自身/他人历史、真实 Judge 输入及原始 completions 缓存回放。
一并运行旧格式、端点并发/缓存、统一入口和 bootstrap 回归测试。

2026-09-23：本地和远端各 36 项回归通过，包含真实 loopback HTTP 96 并发、
成功响应零调用回放；另验证了 NSP 无效预测回退时的官方历史记录行为。
在 8040 的 27000 checkpoint 上，各模式完整跑通原始 CoSER case 0：

| 模式 | 完成 / 预期 | HTTP 错误 | 总请求 | 记录的重试 | API 报告费用 USD |
| --- | --- | --- | --- | --- | --- |
| with-system | 1 / 1 | 0 | 43 | 1 | 0.002881875 |
| without-system | 1 / 1 | 0 | 44 | 0 | 0.002882375 |

以上仅是功能验证，不是全量模型分数；费用不包含 GPU 租赁。
远端记录在 `/workspace/runs/roleplay-evaluation/her-official-v2-smoke-20260923-{with-system,without-system}`。
临时 smoke 启动器首次缺少 multiprocessing main guard，在发送请求前失败；
修复启动器后恢复，原日志和锁文件证据保留，未重发已成功样本。

上游依据：
[角色与 Judge 提示词](https://github.com/cydu24/HER/blob/1e5fd856b9e23d3af41445e3c4ca734e24e012ef/eval_code/benchmarks/multi_turn/coser/prompts.py)、
[多轮流程](https://github.com/cydu24/HER/blob/1e5fd856b9e23d3af41445e3c4ca734e24e012ef/eval_code/benchmarks/multi_turn/coser/benchmark.py)、
[API 分支及 repetition penalty](https://github.com/cydu24/HER/blob/1e5fd856b9e23d3af41445e3c4ca734e24e012ef/eval_code/models/vllm_models.py)。
