# Qwen3 0.6B original vs single-no-system checkpoint 27000

The user authorized simultaneous, independent 200-scene CoSER runs on 2026-09-23.
The original uses CoSER's native actor prompt and unchanged native output. The
fine-tuned model uses HER role tags but does not receive the instruction to
generate system thinking. This is a new protocol comparison, not a continuation
of the previous HER two-part prompt run.

| Setting | Original | Checkpoint 27000 |
| --- | --- | --- |
| API model | qwen3-0.6b-original | qwen3-0.6b-her-single-no-system-27000 |
| Endpoint | http://127.0.0.1:8041/v1 | http://127.0.0.1:8040/v1 |
| GPUs | 2, 3: independent replicas | 0, 1: independent replicas |
| actor_prompt_format | coser-native-v1 | her-no-system-v1 |
| Supervisor | eval-coser-pair-original | eval-coser-pair-single27000 |

Both use actor temperature 0.7, an output cap of 16384, a 32768-token context,
and native `enable_thinking=false`. Exact vLLM token counting reduces the output
cap only when required to fit the original, untruncated benchmark input.
The judge and simulator/helper remain GLM 5.3 Flash, temperature 0, output cap
8192, with the existing provider-compatible remote options unchanged.

Both use the same role-thought-aware judging adapter and CoSER scoring formula;
the original model's **actor prompt and output** are native, while the judging
visibility matches the HER run. This is therefore not an unmodified upstream
leaderboard reproduction. Role thoughts are retained for the judge and hidden
from other characters. HER role thoughts become `[]`, actions become `()`, and
properly delimited system analysis is removed. Previous history handling remains
unchanged. Both retain the benchmark's 60-word response instruction.

`actor_output_validation=role-content-v2` rejects name-only actor responses in
both runs. The HER path also rejects mismatched or unclosed control tags, rather
than silently truncating and accepting the remaining speaker name. Raw responses
remain in immutable transport outcomes; invalid responses may be retried through
the existing worker limits, while valid successful responses are replayed.
No judge scores or frozen benchmark scripts were changed.

## Capacity and persistence

Each run has local concurrency 64, remote concurrency 48, and 128 CoSER scene
workers. Thus these two jobs together have a hard configured remote ceiling of
96. They use the existing `--independent-endpoints` option because they share
the judge endpoint; do not add further overlapping jobs without reallocating
the remote budget. Both evaluation jobs were inactive before this pair started.

The existing asynchronous derived-file writer and separate accounting process
are unchanged. Request intent/outcome journals retain their existing durability
boundary. Old result directories are preserved; caches from different prompts
or models are not copied into these new runs.

## Locations and restart

Host: `root@211.72.13.201:42123`.
Repository: `/workspace/projects/public-evaluation`.
Private environment, preflight responses, jobs manifest, and Supervisor logs:
`/workspace/projects/public-evaluation/.local/coser-pair-20260923/`.
Credentials are in `endpoints.env` with mode 0600 and are not committed.

Configs under `configs/evaluation/roleplay_benchmarks/`:

- `qwen3-0.6b-original-coser-pair-20260923.json`
- `qwen3-0.6b-her-single-no-system-27000-coser-pair-20260923.json`

Results under `/workspace/runs/roleplay-evaluation/`:

- `qwen3-0.6b-original-coser-pair-20260923`
- `qwen3-0.6b-her-single-no-system-27000-coser-pair-20260923`

Each contains the frozen manifest, raw transport journal, progress and accounting.
CoSER summaries are at `coser/outputs/full-public-v1/candidate/coser/summary.json`.
Resume an exited job through its existing Supervisor definition after inspecting
its manifest and confirming no other process owns its run. Never copy a previous
prompt's successful actor results into this run. Compare common successful cases
and separately report failed/invalid cases.

## Verification

- `/models` confirms the expected API names at both endpoints.
- Original weights SHA-256:
  `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
- Checkpoint 27000 weights SHA-256:
  `737ce006e2138d79f4168ea4e39b53a7e66e678b08230662b492a71188594d7b`.
- 27 focused tests pass both locally and remotely, including 96-concurrent
  transport, no-call cache replay, strict invalid-response retry and raw evidence
  retention, no-system prompt substitution, and native output preservation.
- One actual local generation per model passed before launch; both requests lack
  the system-thinking instruction, disable native thinking, finish with `stop`,
  and pass output validation. These probes are archived separately from scores.
- Both full jobs started under Supervisor; training and deployment were unchanged.

## Independent repeat 2

After the first pair finished (original: 200/200, checkpoint 27000: 197/200),
the user requested an independent repeat to measure variability. Both models
were restarted as evaluations, without restarting their deployment. Model file
SHA-256 values, the exact saved configuration, and frozen runtime/input hashes
were verified against the first pair before launch. No sampling, prompt,
validation, scoring, or concurrency settings changed.

- Supervisor: `eval-coser-repeat2-original`, `eval-coser-repeat2-single27000`.
- Runs: `/workspace/runs/roleplay-evaluation/` followed by
  `qwen3-0.6b-original-coser-pair-repeat2-20260923` or
  `qwen3-0.6b-her-single-no-system-27000-coser-pair-repeat2-20260923`.
- Logs and the repeat receipt: `.local/coser-pair-repeat2-20260923/` in the repo.
- The original pair's configs and private endpoint file are reused unchanged.
- No prior response caches are imported: this explicitly requested replicate
  generates and judges new trajectories. Normal within-repeat cache recovery
  remains enabled to avoid accidental duplicate paid requests.

Compare the intersection of successful case IDs between repetitions for each
model, and the common IDs across all four results for a paired model comparison.
Report the overall and per-dimension score differences, per-case variation,
invalid responses, failures, and cost. Two repeats give an observed difference,
not a reliable estimate of the distribution of repeated full-run means. This
experiment combines generation, simulator and judge variation; it does not
isolate judge-only variation on fixed trajectories.
