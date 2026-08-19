# RP Evaluation

逐文件用途见 [`../README.md`](../README.md#evaluation)。

当前代码来自 4090 上的固定 500 对话评估项目：

```text
/home/gyr/lusychat-training/rp-benchmark-eval-v1-20260811
```

入口：

```text
scripts/rp-eval
rp_eval/cli.py
```

模块职责：

- `dataset.py`：冻结 seed 校验和 10/30/60 分层计划；
- `roleplay_prompts.py`：Gemma RP system 构造；
- `candidate_clients.py`：候选模型调用；
- `prompts.py`：Bridge 和 Judge prompt；
- `workflow.py`：A/B、B/A 双顺序评判；
- `scoring.py`：匿名结果映射和 pairwise 聚合；
- `ledger.py`：append-only 运行账本；
- `pilot.py`：smoke、formal 和 matched replay 编排。

本仓库不包含 endpoint 配置、密钥、benchmark 正文或历史 run 结果。`RUNBOOK.upstream.md` 与 `SCORING_POLICY.upstream.md` 是 4090 项目的说明副本。
