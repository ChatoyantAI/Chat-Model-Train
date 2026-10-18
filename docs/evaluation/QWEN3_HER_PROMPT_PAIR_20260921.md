# CoSER with HER output instructions: checkpoints 9000 and 13500

User requested both deployed Qwen3-0.6B HER checkpoints be evaluated using the
format substitution verified in the separate two-case diagnostic.

| Checkpoint | GPU | Port | API model |
| --- | --- | --- | --- |
| 9000 | 0 | 8010 | qwen3-0.6b-her-9000 |
| 13500 | 1 | 8011 | qwen3-0.6b-her-13500 |

Both runs select all 200 official CoSER test scenes through the existing unified
runner. They use `protocol=her-coser-format-v1`,
`actor_prompt_format=her-official-v1`, and
`protocol_label=coser-her-official-prompt-v1`.

Only the pinned CoSER output-format requirements in the actor system message are
replaced with `HER_OUTPUT_INSTRUCTION`. Profiles, scenes, other characters,
history and the existing 60-word instruction remain unchanged. Native thinking
is false. Output conversion removes system_thinking, preserves role thoughts for
the judge, and hides those thoughts from other characters. Judge/helper prompts
are unchanged. This uses the project's GLM judge and is not the paper's exact
benchmark protocol.

Identical settings for both checkpoints:

- BF16, 32768-token service context.
- Actor temperature 0.7, max_tokens 16384. The existing server tokenizer check
  reduces the output allowance when needed to fit the context without truncating
  the benchmark input.
- Judge/helper: zai-org/GLM-5.3-Flash, temperature 0, max_tokens 8192.
- Each run: 128 scene workers, 64 local requests, 48 remote requests. Combined
  remote request concurrency is at most 96 after the prior run finishes.
- `--independent-endpoints` explicitly permits these two bounded runs to overlap.
- Existing RAM journal, asynchronous derived checkpoints and separate accounting
  process are reused without changing storage behavior.

Checkpoint 9000 can start alongside the previous run (48 remote slots each).
The new checkpoint-13500 run starts after its previous supervisor exits,
so concurrent evaluations do not exceed the configured API cap. Historical directories and
logs remain in place. New actor prompts imply new conversations and judgments;
old-format results are not imported into these new runs.

Remote repository: `/workspace/projects/public-evaluation`.

Configurations:

```text
configs/evaluation/roleplay_benchmarks/historical/qwen3-0.6b-her-9000-coser-herprompt-20260921.json
configs/evaluation/roleplay_benchmarks/historical/qwen3-0.6b-her-13500-coser-herprompt-20260921.json
```

Separate result directories:

```text
/dev/shm/roleplay-evaluation/qwen3-0.6b-her-9000-coser-herprompt-v1-20260921
/dev/shm/roleplay-evaluation/qwen3-0.6b-her-13500-coser-herprompt-v1-20260921
```

Each contains `manifest.json`, `launch.json`, `deployment-receipt.json`,
`supervisor-*.log`, `transport-journal/`, and `coser/`. Weight file hashes bind
the model revisions. RAM results must be copied to persistent storage before
machine shutdown. Credentials remain outside the repository.

Resume example (substitute model/port for checkpoint 9000):

```bash
cd /workspace/projects/public-evaluation
EVAL_PAIR_ACTOR_BASE_URL=http://127.0.0.1:8011/v1 \
EVAL_PAIR_ACTOR_API_KEY=local-not-required \
  /root/.cache/chat-model-train/eval-venv/bin/python \
  scripts/evaluation/run_public_evaluation.py \
  --env-file /workspace/.secrets/evaluation.env run \
  --run-dir /dev/shm/roleplay-evaluation/qwen3-0.6b-her-13500-coser-herprompt-v1-20260921 \
  --execute --independent-endpoints
```

Validation: 24 existing and focused tests passed, including exact preservation
of actor context/history/sampling, untouched helper requests, actual rewritten
prompt journaling, successful cache replay, context-aware output limits, and
retention of role thoughts for judging.

Capacity adjustment during the run: GPU 0 checkpoint 9000 initially retained
`gpu_memory_utilization=0.4`, whereas GPU 1 used `0.8`. After 287 cumulative
KV-cache preemptions, GPU 0 was raised to `0.8`. Its 128 scene workers were
paused, local in-flight requests drained, and the same service restarted with
only that memory allocation changed. The broker and remote API requests stayed
alive. All 30 previously completed scene-status hashes were preserved, and all
128 workers resumed. Local concurrency remains 64; sampling and scoring are
unchanged.

Audit receipts are under the checkpoint-9000 run's
`memory80-switch-1789973165/` directory, with persistent copies under
`/workspace/services/public-evaluation/deployment-adjustments/`.
The first observed 30-second post-resume window averaged 4,150 generated
tokens/second, with zero cache preemptions and no local queue. This short window
does not establish a throughput multiplier over the earlier fluctuating rate.
