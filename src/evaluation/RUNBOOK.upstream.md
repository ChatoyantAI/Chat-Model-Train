# Runbook

1. `./scripts/rp-eval preflight`：验证 SHA-256、500/500 数量、sample ID 顺序与 staged plan。
2. `./scripts/rp-eval smoke-openrouter`：验证 GLM-5.2 与 StreamLake 路由。
3. `./scripts/rp-eval smoke-scorers`：验证双顺序 Judge。
4. `./scripts/rp-eval smoke-pipeline`：验证 Gemma、DeepSeek、Luna 纯文本续句和第二轮生成。
5. `./scripts/rp-eval run-pilot20 --count 20 --concurrency 20 --max-turns 10`：运行固定前 20 条校准。
6. `./scripts/rp-eval run-formal500 --concurrency 20`：运行正式固定 500 条 10/30/60 分层评测。

每个 run 必须使用新目录，不能覆盖。主要产物为 `run_contract.json`、`responses.jsonl`、`events.jsonl`、`errors.jsonl`、`progress.json` 和 `summary.json`。

正式 500 条评测已于 2026-08-11 经用户明确批准，`formal_run_enabled=true`。评测数据和评分禁止进入任何训练集。
