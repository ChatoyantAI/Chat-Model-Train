# HER 输出转为 CoSER 格式

脚本：`scripts/evaluation/convert_her_to_coser.py`，只需要 Python 标准库。

```bash
python scripts/evaluation/convert_her_to_coser.py \
  --input response.json --input-format openai-json --output converted.json
```

纯文本输入省略 `--input-format`。输出同时保存 `raw_content` 与 `coser_content`，拒绝覆盖已有文件。
空回复仍保存证据，但返回退出码 2，不能当作有效评分结果。

规则对应 [HER 官方代码固定版本](https://github.com/cydu24/HER/blob/1e5fd856b9e23d3af41445e3c4ca734e24e012ef/eval_code/benchmarks/multi_turn/coser/utils.py)：

- 删除系统思考块，包括官方支持的标签变体与未闭合块。
- `<role_thinking>心理</role_thinking>` 转成 `[心理]`，保留心理内容供裁判评分。
- `<role_action>动作</role_action>` 转成 `(动作)`。
- `<role_speech>台词</role_speech>` 只去掉标签，保留台词。
- 与官方一样整理空行和行首尾空白；不改写台词。

脚本导出 `HER_OUTPUT_INSTRUCTION`，是同版本官方 HER 生成格式指令。
单条试跑可用它替换原提示词中的格式要求，保留人物、场景与其他实验条件。
这仅对齐 HER 的输出格式与清理规则，不代表复现其完整评测协议。

统一入口通过独立的 `protocol: "her-coser-format-v1"` 配置启用，仅允许 CoSER。
历史协议 `frozen-gemma-glm-v1` 保持原样。新协议仅转换 actor 输出，全部请求保持原样。
生成继续使用原 CoSER 提示词，不调用 `HER_OUTPUT_INSTRUCTION`；历史中的括号格式也不转换。
自己的角色心理保留，其他角色不能看到该心理；系统思考不进入后续对话和裁判输入。
裁判收到转换后的角色心理、动作和台词；原始响应由现有 journal 独立保存，缓存恢复不会重复请求。

锁定的 CoSER 源码在评分前也删除角色心理；新协议在独立 worker 副本中取消这一删除，
与 HER 官方保留角色心理的评分方式一致。BLEU/ROUGE 仍单独使用去心理的文本，维持原有口径。
LLM 裁判输入发生修复，因此新评分版本与旧版需分别标记，旧缓存不混入新任务。
worker 副本与适配代码哈希记录在 manifest，恢复时校验。冻结脚本和 vendor 源码不改动。

2026-09-20 单条验证使用 CoSER case 0068 第一轮，模型 `gemma4-her-ckpt4000`。
使用 HER 官方格式指令，并保留原试跑的温度 0.7、max_tokens 8192、原生 enable_thinking=false
及 60 词简洁回复要求。只有一条 actor 请求，不调用 judge，不输出完整场景分数。

## 本地生成超时后的补跑

对已停止的 CoSER 任务，确认本地 vLLM 已无在途请求后，可以显式放行本地
`TimeoutError` 并延长传输超时。此操作仅用于确认不按请求计费的本地 vLLM，
不要用于转发收费 API 的本地代理。

```bash
python scripts/evaluation/recover_local_coser_timeouts.py \
  --env-file /path/to/endpoints.env --run-dir /path/to/run \
  --config /path/to/config.json --timeout-seconds 1200 --execute
python scripts/evaluation/run_public_evaluation.py \
  --env-file /path/to/endpoints.env run --run-dir /path/to/run --execute
```

省略 `--execute` 只查看恢复计划。端点环境变量沿用原任务。脚本只修改传输超时，
不修改提示词、模型、采样、输出长度、并发或评分规则；冻结代码保持原样。
旧状态指针归档到 `recovery/`，原始 intent/outcome 和失败记录不删除。
若发现同一请求已有有效响应，则恢复该响应的缓存指针，不重新生成。
远程 API 的不确定请求继续隔离。统一 runner 会跳过完成场景，并复用失败场景的成功请求。

同一 GLM 端点仍被另一个评测使用时，启动命令可加 `--wait-for-endpoints`。
新任务会等待已有任务释放端点，再自动启动；不修改评测配置，也不绕过全局端点占用保护。
只有端点被占用这一错误会等待重试，缓存或配置错误仍立即退出。

明确需要两个评测独立并行时，改用 `--independent-endpoints`。它与等待选项互斥，
仅放开跨任务的独占锁；每个任务仍遵守自己的 local/remote 并发上限，单个结果目录
仍禁止重复启动。此时多个任务的 API 并发会叠加，运行模式会记录到 manifest。

## 显式提高本地 HER 输出上限

`her-coser-format-v1` 的本地 actor 允许将 `actor.sampling.max_tokens` 从默认 8192
提高到最多 32768；需同时设置更大的 `actor_context_window`，并使用支持 `/tokenize`
的 vLLM `/v1` 接口。发送前按相同 messages 和 chat_template_kwargs 获取实际输入 token
数，将输出限制为 `min(max_tokens, actor_context_window - input_tokens - 1)`。
不截断或改写输入，judge 的上限和其他冻结协议保持不变。

冻结 worker 仍使用原逻辑请求身份，因此旧的有效回复和场景缓存可继续复用；
只有尚未成功的生成采用新上限。journal 记录实际发送的新 payload 和独立哈希，
旧的截断、失败和不确定请求证据均保留，不确定请求仍隔离。
对已开始的任务更改上限，须先暂停场景生产、排空在途请求，再保存旧 manifest、
配置、源码哈希与完成场景哈希，记录 runtime_change_history 后恢复。
这类任务包含调整前后两个阶段，不能表述为全程统一使用新上限。

## 使用 HER 输出格式指令的新评测

显式设置 `actor_prompt_format: "her-official-v1"` 可在该协议内启用 HER 输出指令。
只替换 actor 首个 system 消息中锁定的 CoSER `===Requirements===` 输出格式段，
人物、场景、其他角色、历史、60 词要求及采样参数逐字保留。judge/helper 不替换。
输入结构或原格式段不符合锁定版本时直接报错，不猜测替换位置。

这会改变生成输入，必须使用新的结果目录和明确的 protocol_label；不能把旧格式场景
结果或请求缓存复制进来冒充新评测结果。原格式协议默认行为不变。
实际替换后的请求在 journal 中记录并参与缓存哈希，恢复不会重复成功请求。
输出继续按原转换规则处理，原生 thinking 仍由请求参数单独关闭。
