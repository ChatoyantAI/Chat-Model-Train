# 评测命令

**读实现请从 [src/evaluation/roleplay_benchmarks/README.md](../../src/evaluation/roleplay_benchmarks/README.md) 开始。**
本目录保留命令启动器和需要原路径的历史兼容工具。

| 命令/文件 | 用途 |
|---|---|
| `setup_public_evaluation.py` | 安装 CPU 评测环境与固定输入 |
| `run_public_evaluation.py` / `.sh` | 唯一的新评测入口；实现是包内 `cli.py` |
| `convert_her_to_coser.py` | 独立格式转换命令；实现是包内 `her_format.py` |
| `probe_public_api.py` | 显式请求的 API 并发探针，会产生调用 |
| `build_storyline_comparison.py` / `storyline_comparison.html` | 已保存轨迹的 HTML 对比 |
| `recover_local_coser_timeouts.py` | 专门处理本地超时的恢复工具，不是普通启动步骤 |
| `frozen_public_v1/` | 按字节哈希锁定的协议源码，当前仍引用其中的输入和 CoSER worker |
| `async_checkpoints.py` | 历史导入兼容，实际实现已移到包内 `storage.py` |
| `process_accounting.py`、`recovery_queue_guard.py` | 历史运行器适配；当前入口直接使用包内存储与请求客户端 |

历史调查脚本不是正式运行的依赖。未提交的实验文件留在原处，避免影响其他在做的工作。
新模型只生成配置，通过同一入口运行，不再新增按模型命名的启动器。
