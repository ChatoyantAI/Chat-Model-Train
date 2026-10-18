# Gemma 4 12B regmix v2：CharacterBench / CoSER

启动日期：2026-09-18。本次是独立候选模型评测，不覆盖 Gemma 原版、Qwen 或迁移结果。

## 模型与服务

- HF：`ChatoyantAI/gemma-4-12b-roleplay-sft-regmix-v2`
- 固定 revision：`b102f90a67873b85ad1acbaf8eecb3217a58de36`
- 基座：`google/gemma-4-12B-it`，revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`
- LoRA：r=32 / alpha=64，文件 524,648,560 bytes；SHA-256 `ee7c8a2845d69e63870c64dc1c8c01dc3840cd342eadf8c05b70fdb2b8a89c08`。
- 持久化适配器：`/workspace/sft_data/models/gemma-4-12b-roleplay-sft-regmix-v2`
- 复用已暂存基座：`/dev/shm/models/base-it` → `/dev/shm/gemma-4-12B-it`。机器更换后需要恢复该暂存副本；原始模型保存在 `/workspace/sft_data/models/gemma-4-12B-it`。
- GPU 0 → 8300，GPU 1 → 8301；负载均衡入口 8310；请求代理 18084。
- 实际请求模型名 `gemma4-regmix-v2`，vLLM 模型列表的 parent 为 `gemma4-base-it`，root 为上述适配器路径。
- 复用旧 Gemma 参数：FP8 per tensor、BF16、64K、每副本 max_num_seqs=8、max_num_batched_tokens=2048、KV auto、eager、chunked prefill、language_model_only。
- 两个已完成评测且没有在途请求的 Qwen 服务被替换；GPU 2/3 训练未操作。

## 评测协议

复用 `run_qwen_pair_public.py`，只扩展其配置校验以支持一个模型独占现有总配额；双模型配置和旧任务目录不变。冻结脚本、请求恢复模块、异步写盘模块均未修改。

- CharacterBench：6,497 条有效输入，原始 6,500 条中有 3 条源数据无效。
- CoSER：200 场景，20 轮上限，种子 `42 + scenario_index`。
- Actor 温度 0.7，helper/judge 温度 0，单次最大输出 8,192。
- GLM-5.3-Flash 评分；沿用供应商默认参数，不额外传 thinking 参数。
- Actor `enable_thinking=false`；没有向 benchmark 用户输入添加关闭思考指令。
- 工作线程：CharacterBench 32，CoSER 8；本地请求总上限 16，远程总上限 128。
- 沿用请求和原始响应的同步持久化、普通 checkpoint 的异步保存（4 writers / 256 pending paths），预算保护关闭。
- 同一次评测恢复仍使用同目录及缓存。未知收费状态请求隔离，不自动重发；成功响应可重建未完成的派生缓存。

## 路径和启动

远端部署目录：`/workspace/services/gemma-regmix-v2-20260918`

独立结果根目录：`/workspace/runs/roleplay-evaluation/gemma4-regmix-v2-public-20260918`

样本结果位于 `outputs/full-public-v1/gemma4-regmix-v2/{characterbench,coser}`；请求证据位于同一输出目录的 `transport-repair-v1`。输入身份记录为 `qwen-protocol.json`（保留公共 runner 的原文件名），实际配置与模块哈希记录为 `deployment-*.json`。

```bash
# 基座暂存和 vLLM / eval 环境须已恢复；先下载、校验并启动两个副本。
/opt/vllm-0.29/bin/python /workspace/services/gemma-regmix-v2-20260918/prepare_gemma_regmix_v2.py --launch

# 两个端口健康、适配器身份与实际生成检查通过后启动/恢复评测。
bash /workspace/services/gemma-regmix-v2-20260918/launch_gemma_regmix_v2_public.sh
```

## 启动验证

- 基座及适配器 safetensors 文件长度、header 覆盖范围和 safe_open 检查通过；适配器 SHA-256 与 HF LFS 对象一致。
- 适配器和基座词表、chat template 完全一致；禁用 thinking 时模板关闭 thought 通道。
- 8300、8301、8310 返回正确适配器身份；两个副本各进行一次本地生成，正常 stop、非空回复，无 reasoning 内容。
- 公共 runner 六项离线测试通过，包括单模型总配额、旧输入/缓存不变、模型身份、已付费响应恢复、未知请求隔离。
- Supervisor 已启动，状态 running，PID 284859；该 PID 仅代表本次启动，后续以 supervisor.json 为准。
- 启动证据：服务目录中的 `model-audit.json`、`smoke.json`、`vllm-8300.json`、`vllm-8301.json` 和对应日志。

本文是部署记录，不是最终评分报告。

## 用户授权提高 CoSER 场景并发

用户随后要求提高推进场景数量，尽量利用 GLM 并发。新增独立调度配置 `configs/evaluation/public_gemma_coser96_v1.json`，仅将 CoSER 同时推进场景数从 8 提高至 96。CharacterBench 32、本地总请求 16、GLM 总请求 128、模型生成参数、写盘配置及评分协议不变。启动脚本现指向该配置，原 8 场景配置保留。

切换使用请求代理已有的 `/control/drain`：停止发出新请求，等待在途响应保存，随后在同一结果目录恢复。切换前状态、费用及协议哈希保存在服务目录的 `switch-coser96-before.json`。新增配置等价性测试通过；公共 runner 共七项测试通过。

实际切换核对：2,046 次已发送操作均有请求和原始响应文件；旧调度器无在途请求、无未知收费。账本中缺少 9 条记录，原始响应仍完整存在；按其 usage 计算的恢复费用与旧进程快照一致，均为 $0.083616925。旧调度器在暂停后仍给未运行题目生成失败占位记录，因此在上述核对后终止旧主进程，保留磁盘证据，通过现有恢复逻辑补建派生缓存。

新进程 PID 290883，`supervisor.json` 确认 96 场景配置生效，恢复 9 条已保存响应的账本记录，隔离未知请求数 0。恢复过程没有为这 9 条记录重发请求。详细证据在 `switch-coser96-drained.json`，负载采样在 `coser96-load-sample.json`。

### 实测回调到 32 场景

96 场景确实全部启动，但启动突发造成代理本地队列积压（观察到 84 个等待请求），超过现有 30 秒排队期限的请求被拒绝；期间 GPU 仍间歇空闲，代理快照也出现数十秒延迟。因此没有保留这个更大的配置作为默认值。第二次排空前后记录在 `switch-coser32-before.json` / `switch-coser32-drained.json`。

最终启动脚本改用 `public_gemma_coser32_v1.json`，场景并发 32，是最初的 4 倍，其余协议和限流不变。96 场景试跑记录中有 422 次发送前退回；这不是 422 次上游 API 失败，也不代表新增 422 次计费。试跑结束时所有 2,546 次已发送操作都有原始响应，费用核对为 $0.105271850；现有恢复逻辑还需补回 2 条账本记录，没有未知收费请求。旧 worker 在在途请求清空后停止，缓存继续保留。

## 独立记账进程优化

用户要求优化磁盘写入。实测 vLLM 服务平均约 2.3–2.6 秒，旧代理最近平均约 10 秒；24 个请求的文件时间戳分段估算中，响应缓存写入至进入全局记账锁的间隔中位数约 8.2 秒。该分段是估算，不是逐阶段埋点。

新增 `scripts/evaluation/process_accounting.py`，通过可选包装器替换 broker 的账本持久化方法；冻结评测器、`recovery_queue_guard.py`、异步 checkpoint 模块及缓存身份不变。生产配置为 `public_gemma_coser32_process_accounting_v1.json`，继续使用 32 场景、32 CharacterBench 线程、本地 16、API 128。

- 账本由独立进程批量保存，最多 512 条待确认记录，每批最多 128 条，收集窗口 1 秒。主进程提交不等待队列空间，也不在全局锁内做文件 I/O。
- 每批写入新的不可变 JSONL 文件，使用原来的临时文件、fsync、原子替换实现，避免反复同步追加同一账本。旧账本不修改。
- 请求意图、原始成功响应及恢复缓存仍在交付前确认持久化；这部分有单请求等待，没有改成不确认落盘的纯内存队列。
- 成功响应的账本允许稍后提交，崩溃后由既有恢复流程从原始响应补回费用；失败请求等待错误账本保存，但等待发生在调度锁释放之后。
- 写入进程出错或队列满时停止继续发出请求，避免无界内存积压和费用记录丢失。运行状态增加 `accounting_writer`，包含 PID、待确认条数、已保存条数、批次数和错误。
- 新模块 6 项测试及现有 runner 7 项测试全部通过：慢盘不占调度锁、不可变批量保存、队列有界、失败证据保存、写入进程失败阻止新调用、写入进程崩溃后费用只恢复一次。

代码部署到 `/workspace/services/gemma-regmix-v2-20260918/process-accounting-v1`，启动脚本使用新部署，旧代码与配置保留。切换前证据为 `switch-process-accounting-before.json`。

切换时核对 4,734 次已发送操作均有原始响应；补回旧账本缺失的 8 条记录，费用为 $0.244793175，与排空快照一致。新主进程 PID 318433，独立记账进程 PID 318566；没有未知收费请求。

初步实测（相同 32 场景配置，不同样本的前后短窗口，并非重发同一批样本的控制实验）：旧代理最近 100 个 Gemma 请求平均 9.15 秒，新代理已保存的 81 个请求平均 3.60 秒；状态查询约 2.7–3.9 毫秒。该窗口无新增上游 HTTP 错误或发送前退回。证据为 `process-accounting-load-sample.json`、`process-accounting-latency-comparison.json`，后续队列观察为 `process-accounting-steady-health.json`。这些数据不等同于整轮完成时间按相同比例缩短。

## 当前运行目录：内存盘（用户自行导出）

后续检查发现每请求的缓存目录检查、创建及文件发布仍受网络盘 FUSE 元数据操作拖慢：目录存在性检查单次约 12 秒、列目录约 17 秒，主进程 33 个线程处于磁盘等待，普通 checkpoint 队列达到 256 上限。独立费用账本进程没有覆盖这些操作。

用户明确允许先写内存盘，并自行转移到网络盘。因此采用更直接的运行目录迁移，不扩展文件虚拟化或自动回传体系。当前启动脚本使用：

- 运行目录：`/dev/shm/roleplay-evaluation/gemma4-regmix-v2-public-20260918`
- 程序目录：`/workspace/services/gemma-regmix-v2-20260918/ram-v1`
- 配置：`configs/evaluation/public_gemma_coser32_ram_v1.json`
- 原始网络盘目录保留：`/workspace/runs/roleplay-evaluation/gemma4-regmix-v2-public-20260918`

模型生成、评分请求、原始响应、恢复缓存、账本、结果及运行日志均写入新的内存盘运行目录。普通 checkpoint 仍后台写入，费用账本仍由独立进程批量写入；请求意图和原始响应仍在交付前写入，但目标已是内存盘。模型服务、GPU 分配、32/32 工作并发、本地 16/API 128 总限额和评分协议不变。

`evaluation_storage.py` 仅将新运行目录的请求 scope 映射回旧目录的 scope，主进程及 CoSER worker 都在构造缓存键前安装映射，保留已付费请求的身份。冻结脚本和原恢复模块未修改。`stage_evaluation_run.py` 复制已停止任务，保留源目录，每个普通文件比较复制前后 SHA-256，拒绝覆盖已有目标目录，并输出 `storage-copy-verified.json`。启动器要求这个校验标记存在，避免机器重启后在空目录意外开始一轮全新付费任务。

内存盘不保证关机后的数据保存。用户需要在关机前导出**整个运行目录**，包括 `outputs/full-public-v1/transport-repair-v1`，不能只转移分数汇总。程序不自动回写网络盘，网络盘旧进度也不会继续更新；服务目录的 `active-run-location.json` 记录当前入口。

复制时发现停止旧进程后，FUSE 仍列出部分无法读取的原子写入临时文件（UUID `.tmp`/`.partial` 以及 CoSER `.json.tmp`）。这些尚未发布的临时文件单独列在复制清单中，原网络盘文件保留；正式请求、响应、缓存或结果文件读取失败仍会阻止启动。中断复制可通过匹配来源的 in-progress 标记恢复，正式文件重新核对字节内容后才生成验证标记。
