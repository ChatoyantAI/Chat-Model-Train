# Qwen3 0.6B single-no-system checkpoint 27000: CoSER

User supplied a deployed endpoint and authorized evaluation of this checkpoint.
This independent 200-scene CoSER run changes only actor model and revision from
`qwen3-0.6b-her-no-system-4500-coser-herprompt-20260922.json`.

- Host: `root@211.72.13.201:42123`.
- Endpoint: `http://127.0.0.1:8040/v1`.
- Model: `qwen3-0.6b-her-single-no-system-27000`.
- Pinned checkpoint: `/workspace/training/qwen3-her-single-no-system-5090-v1/pinned-checkpoints/checkpoint-27000`.
- Weights SHA-256: `737ce006e2138d79f4168ea4e39b53a7e66e678b08230662b492a71188594d7b`.
- Config: `configs/evaluation/roleplay_benchmarks/historical/qwen3-0.6b-her-single-no-system-27000-coser-herprompt-20260923.json`.

The same original CoSER scenes, HER-format instruction substitution and output
postprocessing are retained. Native actor thinking is false, actor temperature
0.7, output cap 16384 with a 32768-token context. GLM 5.3 Flash helper/judge uses
temperature 0 and 8192 output tokens, with unchanged remote provider parameters.
Shared limits remain local 64 / remote 96, with 128 scene workers. No deployment,
training, benchmark source, scoring schema or storage implementation was modified.

The existing HER instructions still request custom system/role tags; a no-system
training checkpoint does not implicitly rewrite the evaluation prompt. As before,
system-thinking is removed and role thoughts/actions are converted. This retains
the previous comparison protocol, not an exact reproduction of all upstream HER
history handling.

## Runtime

- Repo: `/workspace/projects/public-evaluation`.
- Supervisor: `eval-qwen-single-nosystem27000-full`.
- Run: `/workspace/runs/roleplay-evaluation/qwen3-06b-her-single-no-system-27000-coser-herprompt-20260923`.
- Launch log, private endpoints file and weights receipt:
  `/workspace/projects/public-evaluation/.local/qwen-single-no-system-27000-20260923/`.
- Summary: `coser/outputs/full-public-v1/candidate/coser/summary.json` under the run.

Both model catalogs passed preflight. The user already reported generation and
streaming checks, so no separate duplicate pilot was run. New results use a fresh
directory; the old model's response caches are not imported. Within this run,
normal immutable request journaling, asynchronous derived writes, separate
accounting and resume behavior remain enabled. Results are on the mounted xfs
volume under `/workspace`, not volatile RAM.

The prior checkpoint 4500 run is terminal partial: 197/200 successful, failures
101/163/197, average 6.142766497461929. Its final dimensions were 9.98730964467005
storyline consistency, 4.738578680203045 anthropomorphism, 3.065989847715736
character fidelity and 6.779187817258883 storyline quality. Its full-run phase
took about 18.8 minutes, excluding the earlier pilot. Compare new results on the
common successful cases before attributing differences to training.

To resume this exact run after verifying the Supervisor job has exited:

```bash
cd /workspace/projects/public-evaluation
bash scripts/evaluation/run_public_evaluation.sh \
  --env-file .local/qwen-single-no-system-27000-20260923/endpoints.env run \
  --config configs/evaluation/roleplay_benchmarks/historical/qwen3-0.6b-her-single-no-system-27000-coser-herprompt-20260923.json \
  --run-dir /workspace/runs/roleplay-evaluation/qwen3-06b-her-single-no-system-27000-coser-herprompt-20260923 \
  --execute
```
