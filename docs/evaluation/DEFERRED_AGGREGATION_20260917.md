# Let judging precede request-accounting scans

The live RMT worker's diagnostic stack was in `community.aggregate`, reading
each `*.attempt-*.json` from the network volume. The runner called this full
scan after generation, after judging, and again for its aggregate stage.
Consequently, completed generation could sit behind thousands of serial file
reads before any new judge requests were sent.

The fix moves the existing aggregate call outside the stage loop. An
`evaluate` run now generates, judges, and aggregates once. Explicit generate,
judge, and aggregate commands still publish their summaries. Per-case
checkpoints and progress continue to be saved while work is running. Final
accounting still reads all retained attempts, including historical failures;
it has not been replaced with an incomplete or estimated cost total.

The normal repository runner contains this fix. The current migrated run
uses the identical two-line stage-order change in memory, guarded against an
unexpected source layout. Its frozen engine files, manifest hashes, prompts,
sampling, parsers, and successful request-cache keys remain unchanged.

Validation: 50 local public-runner/recovery tests passed, including a complete
offline MiniMax run against both the new source and patched frozen loop.
Both send all judge requests before the single accounting scan, retain the
same scores and costs, and resume with zero new upstream calls. The remote
host also passed 33 recovery tests and the frozen-source preflight.

Deployment drains only the evaluation load balancer's ingress using its
graceful shutdown; vLLM engines remain loaded. Existing requests must return
and active broker counts must remain zero before stopping the supervisor.
Local connection failures during this maintenance window are preserved as
attempt evidence. Cached prefixes could still lead to new simulator requests,
so after the real LB exited, a temporary private empty model catalog engaged
the existing broker admission guard. Its `local_model_identity_changed` halt
is an intentional deployment pause, not an actual model replacement; the
temporary catalog forwarded no inference requests and closed immediately.
`maintenance-admission.json` records this distinction. The load balancer and supervisor are then restarted with
the same models, concurrency, budget, and existing caches. Old logs/statuses
and stale locks are archived rather than deleted.

Remote deployment archive:
`/workspace/migrations/old-full-run-20260914/outputs/full-public-v1/interruption-20260917T024221Z-deferred-aggregation/`

This removes the pre-judge bottleneck. Final accounting can still take time
on the network volume, and the full RMT judge workload is still large; neither
generation progress nor a completed request is treated as a completed score.

## Independent generation tail and optional judge-only recovery

The deployed aggregation fix runs as supervisor 90922. Its live resumed
generation checks reuse 319 completed RMT trajectories. Case `zh-016-000`
remains incomplete (38 messages); repeated actor attempts reach the unchanged
8192-token output cap and fail validation after about 250–260 seconds. This
generation tail is distinct from the removed accounting scan.

The user accepted leaving rare missing samples unfilled. An explicit
`--rmt-judge-existing-only` recovery option therefore runs the standard RMT
judge stage once on existing trajectories; it preserves missing-generation
coverage rather than declaring the suite complete. Other suites continue
normally. Local tests now total 51; remote recovery tests total 34, and the
flag passed frozen-source preflight.

The second pause/restart command was rejected by automatic approval review
(`blocked by policy`, no more specific reason supplied). It did not execute.
Thus the option is available in the staged code but is **not active** in
supervisor 90922. Its existing four-attempt policy remains active; after that
case finishes or exhausts retries, the already-deployed stage-order fix leads
directly to judging. Do not claim that judge-only recovery is running.
