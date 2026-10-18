# CoSER native original vs HER official no-system checkpoint 27000

User-authorized new full evaluation pair, started 2026-09-23 09:45 UTC
(17:45 Asia/Shanghai). This is a fresh generation/judging repeat, not a resume
of older scores or the one-scene integration smoke tests.

| Setting | Original Qwen3-0.6B | HER checkpoint 27000 |
| --- | --- | --- |
| Endpoint | `http://127.0.0.1:8041/v1` | `http://127.0.0.1:8040/v1` |
| GPUs | 2, 3 | 0, 1 |
| Actor prompt | Pinned CoSER native, including original format examples | Pinned HER official `her_nosys`, no system_thinking instruction |
| Protocol | `her-coser-format-v1`, actor `coser-native-v1` | `her-coser-official-v2`, `without-system` |
| Length instruction | Original CoSER 60 words | Official HER approximately 200 words |
| Extra repetition penalty | Unspecified, existing server behavior | Official HER chat default 1.15 |
| Own history | Original CoSER representation | HER role tags retained |
| Judge view | Role thoughts retained, as in the previous paired baseline | System analysis removed, role thoughts retained and converted to `[]` |
| Supervisor | `eval-coser-native-vs-her-original` | `eval-coser-native-vs-her-27000` |

Both consume the same original CoSER 200-scene file (SHA-256
`eb175779d1d5adf99c893ff1f82de9b18ada912d3f385b3f5419195afdfb374a`).
Actor temperature is 0.7, token cap 16384, context 32768, native thinking false.
GLM 5.3 Flash remains the judge and helper, temperature 0, max_tokens 8192;
provider thinking options remain the proven empty remote override.

Original means the deployed `qwen3-0.6b-original`, not a newly downloaded
pretraining-only checkpoint. Its native prompt/output configuration is unchanged
from the previous pair. Its thought-aware judging is not the unmodified CoSER
official thought-stripping evaluation. The 27000 run uses the official HER
prompt/history/NSP/environment/critic adaptations documented in
`HER_OFFICIAL_COSER_V2.md`. The comparison therefore includes these requested
protocol differences, not only model weight changes.

## Limits and persistence

Each run retains local concurrency 64, remote concurrency 48 and 128 scene
workers. Exactly these two jobs share the remote endpoint using
`--independent-endpoints`, for a total configured API ceiling of 96.
All other evaluation jobs were exited before launch. Training and model services
were not restarted. Existing asynchronous writers and accounting are reused.

Prior response caches were not imported into this explicitly requested fresh
repeat. Successful responses within each new run remain replayable on recovery;
errors, raw responses, usage, retries and ambiguous outcomes remain recorded.
Never overwrite older runs or copy mismatching model/protocol caches into them.

## Locations

Host: `root@211.72.13.201:42123`.
Repo: `/workspace/projects/public-evaluation`.
Runtime code commit: `08f7b9464f83024a4bb18e4a15f2acc9aee66baa`.

Configs, under `configs/evaluation/roleplay_benchmarks/`:

- `qwen3-0.6b-original-coser-pair-20260923.json`
- `qwen3-0.6b-her-27000-coser-official-without-system.json`

Run directories under `/workspace/runs/roleplay-evaluation/`:

- `qwen3-0.6b-original-coser-native-vs-her-official-20260923`
- `qwen3-0.6b-her-27000-coser-official-nosystem-full-20260923`

Each run has a frozen manifest and per-request evidence. Per-benchmark progress
and scores: `coser/outputs/full-public-v1/candidate/coser/summary.json`.
Final accounting: `usage-summary.json`; raw events: `transport-journal/`.

Launch receipt, config/input hashes, scripts and logs:
`/workspace/projects/public-evaluation/.local/coser-native-her-official-pair-20260923/`.
The existing private `.local/coser-pair-20260923/endpoints.env` is reused without
copying secrets into Git. The launchers call the public CLI, with its proper
multiprocessing entrypoint, and Supervisor manages both jobs.

## Verification and reporting

Preflight GET `/models` confirmed both exact actor aliases and the judge alias;
both prepared manifests contain 200 scenes and identical input hashes. Runtime
and input hashes are frozen separately per run. Both services entered RUNNING.
Completion must be established from each final manifest/summary, not Supervisor
EXITED alone. Report coverage, all four scores, failures, missing cases and costs;
compare common successful cases when coverage differs.
