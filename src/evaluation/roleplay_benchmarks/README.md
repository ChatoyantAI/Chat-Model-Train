# 公开角色扮演评测

统一入口：`scripts/evaluation/run_public_evaluation.py`。安装：`scripts/evaluation/setup_public_evaluation.py`。
完整克隆、配置、启动、恢复、产物位置见[运行说明](../../../../docs/evaluation/UNIFIED_PUBLIC_EVALUATION.md)。
本地 vLLM 和远程 API 都作为 OpenAI 兼容端点；模型部署、训练和 GPU 分配属于其他模块。
负责人及 Reviewer 沿用仓库 Evaluation 职责登记。

## 固定协议

| Benchmark | 当前协议 |
|---|---|
| CharacterBench | 历史 Gemma/GLM 适配器；GLM API judge；6500源样本中6497条可处理，3条源输入无效单列 |
| CoSER | 固定上游场景与提示词；历史 actor / environment / next-speaker / judge 路由；默认200场景 |
| MiniMax | 官方公开数据和公开方法的本地实现；缺失的官方完整模拟器文案以已记录的本地文案实现 |
| RMTBench | Claude45数据，四块长对话，每角色两次，seed42；不包含未公开Poison/SEC |

这些是历史可比较实验协议，不混称完整官方排行榜成绩；四套分开报告，不制造一个总分。
冻结源码位于 `scripts/evaluation/frozen_public_v1/`，输入来源固定在 `sources.lock.json`。
benchmark用户输入和评分方式不会因为换候选模型而改动。GLM沿用历史provider默认请求方式，不能标成已关闭thinking。

## 活跃代码

- `bootstrap.py`：安装、固定源准备、配置生成、免费端点检查。
- `unified.py`：四套编排，共享local16/remote96限制。
- `endpoint_transport.py`：原始意图/响应、去重重放、请求隔离、费用核对。
- `frozen_backend.py`：连接固定CharacterBench/CoSER/RMT协议。
- `community.py`、`community_protocols.py`：固定MiniMax/RMT实现。
- `scripts/evaluation/async_checkpoints.py`、`process_accounting.py`、`recovery_queue_guard.py`：复用已有写盘及恢复基础设施。

派生记录异步写盘，费用用独立进程；小型原始请求日志先在本地盘持久化再交付，避免崩溃后重复计费。
不在启动时扫描全部网络盘历史。旧任务目录、密钥和模型不打入仓库。
原生协议检查/报告库仍用于审计；新任务只有上述统一入口。

## 数据来源

Git子模块固定 CoSER 和 CharacterBench 上游提交；Hugging Face 数据固定 revision，逐文件保存并校验SHA-256。
`setup_public_evaluation.py` 自动完成所选数据准备，重复运行复用已有有效文件。
RMT需要个人账号授权；不自动接受 gated 条款。数据、权重、密钥、输出不进入Git。
下载与核验也可单独执行 `scripts/download_roleplay_benchmarks.py`、`scripts/roleplay_benchmarks.py verify all`；二者不是评测入口。

## 验证

2026-09-19：CharacterBench 8条、CoSER 1完整场景通过；真实GLM 96并发96/96有效，恢复新增0调用。
这是小样本与传输测试，不是四套全量结果。具体统计及限制见运行说明。

```bash
python -m unittest tests.evaluation.test_evaluation_bootstrap tests.evaluation.test_endpoint_transport tests.evaluation.test_unified_runner tests.evaluation.test_process_accounting tests.evaluation.test_async_checkpoints tests.evaluation.test_benchmark_public_runners tests.evaluation.test_benchmark_rmt45_migration
```
