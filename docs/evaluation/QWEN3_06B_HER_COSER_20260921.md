# Qwen3 0.6B original / HER checkpoint-13500 CoSER runs

Both evaluations were launched together on 2026-09-21 using the existing unified
runner, with no changes to benchmark code, prompts, scoring, or sampling.

| Actor | GPU | Endpoint | Config |
| --- | --- | --- | --- |
| qwen3-0.6b-original | 0 | http://127.0.0.1:8010/v1 | configs/evaluation/roleplay_benchmarks/historical/qwen3-0.6b-original-coser-20260921.json |
| qwen3-0.6b-her-13500 | 1 | http://127.0.0.1:8011/v1 | configs/evaluation/roleplay_benchmarks/historical/qwen3-0.6b-her-13500-coser-20260921.json |

Each run selects all 200 CoSER scenes. Protocol `her-coser-format-v1` retains
the original CoSER generation prompts and uses output-only HER conversion;
role thoughts remain visible to the judge and hidden from other characters.
Both actors use the same conversion and scoring rules. This is the project's
HER-format CoSER protocol, not a reproduction of the paper's different judge.

Actor temperature is 0.7, helper temperature is 0, and max_tokens is 8192.
Judge/helper is `zai-org/GLM-5.3-Flash`. Native thinking is explicitly disabled
for local requests. Learned tags in response content are handled by the existing
converter; disabling native thinking does not prevent learned textual analysis.
Unexpected output formats and length-limited responses must remain in the audit.

Each evaluator has 128 scene workers, local request concurrency 64 and remote
request concurrency 48. The pair's combined remote limit is 96. Explicit
`--independent-endpoints` permits the pair to run together; do not add more
evaluations against the same API without reallocating this total concurrency.
Services were already BF16, 32K context, max_num_seqs=64, with GPU memory
utilization 0.4. No deployment parameters were changed for this launch.

Remote code: `/workspace/projects/public-evaluation`.
Results and replay caches are separate under:

```text
/dev/shm/roleplay-evaluation/qwen3-0.6b-original-coser-thoughts-v2-20260921
/dev/shm/roleplay-evaluation/qwen3-0.6b-her-13500-coser-thoughts-v2-20260921
```

The system disk had only 5.7 GB free, so the hot path uses RAM storage with the
existing synchronous request journal, asynchronous derived checkpoints, and
separate accounting process. RAM results do not survive a machine restart;
copy them to persistent storage before stopping the machine.

Each directory contains `manifest.json`, `launch.json`, `deployment-receipt.json`,
`supervisor-*.log`, `transport-journal/`, and the `coser/` child evaluation.
Deployment receipts hash the safetensors files, and configs bind the actor
revision to that hash manifest. Secrets are loaded from
`/workspace/.secrets/evaluation.env`; no credentials are stored in this repo.

Resume an existing run using the matching model and port (example for HER):

```bash
cd /workspace/projects/public-evaluation
EVAL_PAIR_ACTOR_BASE_URL=http://127.0.0.1:8011/v1 \
EVAL_PAIR_ACTOR_API_KEY=local-not-required \
  /root/.cache/chat-model-train/eval-venv/bin/python \
  scripts/evaluation/run_public_evaluation.py \
  --env-file /workspace/.secrets/evaluation.env run \
  --run-dir /dev/shm/roleplay-evaluation/qwen3-0.6b-her-13500-coser-thoughts-v2-20260921 \
  --execute --independent-endpoints
```

Initial validation: both actor and judge model catalogs passed, both supervisors
were alive and sending real requests, and early HTTP errors were zero. The HER
actor produced substantially longer responses, including some 8192-token
truncations. At 64 local requests its KV cache approached capacity and vLLM
reported preemptions; do not infer spare capacity from model parameter count.
These observations are provisional, not final benchmark results.

## User-requested HER output budget increase

After the initial launch, the user requested a larger HER output allowance.
The HER actor cap was raised from 8192 to 16384 with exact server-side token
counting against its 32768-token context. Existing completed scenes and valid
responses are retained; new generations and truncated retries use the larger
budget when context space permits. Original Qwen and the judge remain at 8192.
This is therefore a staged run, not a uniformly 16384-token rerun.

GPU 1's service was drained and restarted with GPU memory utilization 0.8
(previously 0.4), preserving its other arguments and 64-sequence batch capacity.
This increases KV cache capacity for the longer generations. GPU 0 was not
restarted. The audit, prior manifests, cache hashes, service command, and restart
log are in the HER result directory's `token-limit-16384-switch/` subdirectory.
