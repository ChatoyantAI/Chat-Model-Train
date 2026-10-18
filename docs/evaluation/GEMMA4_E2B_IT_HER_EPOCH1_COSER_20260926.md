# Gemma 4 E2B IT versus HER Epoch 1: CoSER

User-requested fresh pair, 2026-09-26. Both use the pinned original 200 CoSER
scenes, GLM 5.3 Flash judge/helpers, actor temperature 0.7 and max_tokens 16384.
Native thinking is explicitly false. Neither sends repetition_penalty.

| Setting | Original IT | HER Epoch 1 |
| --- | --- | --- |
| API model | gemma4-e2b-it | gemma4-e2b-her-epoch1 |
| Local endpoint | http://127.0.0.1:8050/v1 | http://127.0.0.1:8051/v1 |
| GPUs | 0,1 independent replicas (DP2, TP1) | 2,3 independent replicas (DP2, TP1) |
| Weights | BF16 base | BF16 base + checkpoint-10161 LoRA |
| Actor protocol | CoSER native prompt | HER official without-system |
| Context | 65536 | 65536 |

HER uses `/v1/chat/completions` and the server's Gemma tokenizer/template,
not raw Qwen ChatML. Its own history retains role tags. System analysis is
removed; role thoughts/actions become brackets/parentheses for the judge.
Both judge views retain role thoughts, matching the previous requested pair;
the original IT branch is not the untouched CoSER thought-stripping protocol.
This comparison includes differences in actor prompts/history, not only weights.

Existing asynchronous persistence, per-request journal, accounting and recovery
remain unchanged. Each run has local concurrency 64, remote 48 and scene workers
128. The two jobs explicitly share remote capacity with a total ceiling of 96;
do not add other jobs using that endpoint without adjusting the shared budget.
The user changed deployment from TP2 to two full replicas per model after the
initial launch. The public ports, model aliases, BF16 weights and 64K context
remain unchanged. Training stays paused; evaluation does not restart deployment.

The initial evaluation exited during the deployment switch, with 151/200 IT
and 156/200 HER scenes completed; sampled remaining failures show HTTP 503.
Resume the existing run directories to retain successful generations/judgments
and prior failure evidence, rather than starting another paid full repeat.
Results therefore span the initial TP2 and resumed DP2 deployment phases.

Host: `root@211.72.13.201:42123`. Repository:
`/workspace/projects/public-evaluation`.

Configs: `configs/evaluation/roleplay_benchmarks/`
`gemma4-e2b-it-coser-20260926.json` and
`gemma4-e2b-her-epoch1-coser-20260926.json`.

Run directories under `/workspace/runs/roleplay-evaluation/`:

- `gemma4-e2b-it-coser-native-20260926`
- `gemma4-e2b-her-epoch1-coser-official-nosystem-no-rp-20260926`

Launch receipts and logs: `.local/gemma4-e2b-coser-pair-20260926/`.
Supervisor services: `eval-coser-gemma-e2b-it` and `eval-coser-gemma-e2b-her-epoch1`.
Summaries: `coser/outputs/full-public-v1/candidate/coser/summary.json` in each run.
Accounting: `usage-summary.json`. No prior-model responses are imported; within
each run, successful responses remain cached and failures retain their evidence.

Base HF revision: `3e22461f65e89153144f8adb70e3b8c2cc9845a7`.
Deployed adapter SHA-256:
`b01ce31bc44ad3a18e2f3ef308d27f3b230e878ebaa65c486dfdc116ced91253`.

## Independent repeat 2

On 2026-09-26 the user requested another full run to assess score variation.
Both models reuse their respective unchanged JSON configs and all 200 scenes;
IT stays on native CoSER and HER on official without-system, with no injected
repetition penalty. This is not the proposed cross-format ablation.

New run directories and Supervisor service names append `-repeat2` to the names
above. Launch receipt/logs are in `.local/gemma4-e2b-coser-repeat2-20260926/`.
The repeat uses the current DP2/TP1 deployments throughout, whereas the first
round spanned TP2 and DP2. No previous-run response caches are imported into
this explicitly requested fresh generation/judging repeat. Successful responses
within repeat 2 remain reusable on recovery. Previous results remain intact.
