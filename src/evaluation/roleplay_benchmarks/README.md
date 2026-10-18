# 公开角色扮演评测

**从这里开始读。** 当前评测实现都在本目录；`scripts/evaluation/` 的运行命令只是启动器。
模型部署、训练和 GPU 分配由其他模块负责，评测只连接 OpenAI 兼容端点。

## 先读三个文件

1. [config.py](config.py)：选哪个 benchmark、模型/judge、并发、采样和协议约束。
2. [unified.py](unified.py)：`prepare()` 固定输入；`run()` 启动任务、共享限流、等待落盘、汇总状态。
3. 按需要进入 [characterbench.py](characterbench.py)、[frozen_backend.py](frozen_backend.py)（CoSER），或 [community.py](community.py)（MiniMax/RMT）。

命令参数在 [cli.py](cli.py)，配置样例在 [unified.example.json](../../../configs/evaluation/roleplay_benchmarks/unified.example.json)。
安装和启动命令见[运行说明](../../../docs/evaluation/UNIFIED_PUBLIC_EVALUATION.md)。

## 一次评测怎么走

```text
cli.py → config.py → unified.prepare / unified.run
                         │
                         ├─ CharacterBench → characterbench.run
                         ├─ CoSER          → frozen_backend.coser → 独立场景 worker
                         └─ MiniMax / RMT  → community.run
                                │
                         endpoint_transport.py
                      请求去重、共享限流、原始响应日志
                                │
                    生成 → 评分 → 异步存储 → 最终汇总
```

当前流程显式传入请求客户端和存储对象，不再全局替换 `Client`、`community` 读写函数或 RMT 配置/样本构建函数。
CoSER 的上游适配集中在 `frozen_backend.py`，只修改新任务中的 worker 副本。

## 各部分在哪

| 职责 | 文件 |
|---|---|
| 命令入口、安装检查 | `cli.py`、`bootstrap.py` |
| 配置、调度、共享并发 | `config.py`、`unified.py`、`execution.py` |
| HTTP、缓存重放、失败隔离 | `endpoint_transport.py`、`benchmark_api.py` |
| 原始证据持久化、异步派生文件、独立费用进程 | `storage.py` |
| CharacterBench | `characterbench.py`；输入准备复用锁定的历史协议 |
| CoSER | `frozen_backend.py`；官方生成/评分在 `vendor/benchmarks/CoSER/gca_evaluation/` |
| MiniMax / RMT | `community.py`、`community_protocols.py`、`paper_templates.py` |
| HER 提示格式、输出转换 | `her_official.py`、`her_coser.py`、`her_format.py` |
| 评分校验与汇总口径 | `assessment.py` |
| 已有结果审查、重新评分 | `assessment_audit.py`、`coser_rejudge.py` |
| 来源下载与核验 | `sources.py`、`sources_cli.py` |
| 原生协议审计、历史结果报告 | `preflight.py`、`runtime.py`、`reports.py` |

`runtime.py` 保留原生协议的准备、审计工具；新评测统一走 `cli.py → unified.py`。
三份按字节锁定的历史文件保留在 `scripts/evaluation/frozen_public_v1/`，仍有实际调用。
`recovery_queue_guard.py` 和 `process_accounting.install()` 是历史任务的修复适配，当前调度器不调用它们。

## 写盘和恢复

- 原始请求意图和响应：`storage.atomic_*` 在本地盘持久化后才交付，保留重复计费的恢复边界。
- 轨迹、评分、进度：`AsyncCheckpoints` 有界异步队列，同一路径的中间状态可合并。
- 费用流水：`ProcessLedger` 独立进程批量写入，保留 ACK 和结束时 flush。
- 评分前确认本场景轨迹可读，最终汇总前确认派生文件写入；不会每生成一轮就扫描全部历史请求。
- 成功请求按原有请求身份复用；未知计费状态的中断请求仍隔离，不自动重发。

本次整理没有改变采样、并发默认值、缓存键、原始日志布局、提示词或评分公式。
代码哈希校验仍有效：历史任务用 manifest 对应的代码版本恢复，不绕过校验。

## 协议与历史实验

| Benchmark | 数据/协议范围 |
|---|---|
| CharacterBench | 官方样本与 rubric，历史 GLM API judge；6500 源样本中 6497 条可处理，3 条源输入无效单列 |
| CoSER | 固定官方场景；原生/HER actor 格式由配置明确选择；默认 200 场景 |
| MiniMax | 官方公开数据；模拟器和部分 judge 文案由本地实现，未做人类校准 |
| RMTBench | Claude45 长对话，四块、每角色两次、seed42；排除未公开 SEC/Poison |

新 `init` 默认 `assessment.version=20261018`：CoSER/HER 使用一致的显式心理可见性与角色标签；MiniMax/RMT headline 只纳入完整单位，局部结果另存。
原有配置不自动迁移，`--assessment legacy` 保留历史协议。依据见[协议审查](../../../docs/evaluation/PROTOCOL_AUDIT_20261018.md)。
API judge 沿用 `thinking.remote={}`，不能宣称已关闭 thinking。HER/原生 actor 提示仍有差异，共同评分不等于严格权重消融。
四套分开报告，不合成跨 benchmark 总分，也不冒称官方排行榜复现。

过去按模型命名的配置已集中到 [historical/](../../../configs/evaluation/roleplay_benchmarks/historical/README.md)，内容原样保留。
历史报告和实验结论见[文档导航](../../../docs/evaluation/README.md)。
