# Gemma evaluation queue recovery, 2026-09-17

## Incident and evidence

The migrated runner admitted six concurrent remote requests while MiniMax and
RMT still submitted from 32 worker threads each, alongside eight CoSER workers.
The old launch admitted 64 remote requests. This is a scheduling configuration
change, not evidence that high concurrency itself is unsuitable.

The frozen broker waited indefinitely on a non-FIFO semaphore, then started a
300-second upstream timeout. Its callers timed out after 1,800 seconds including
queue time. Disconnected callers remained queued. The captured process had 59
server-side CLOSE-WAIT sockets and repeated BrokenPipe errors. RMT's valid
judgments remained at 15 while successful broker responses continued increasing.
The old broker did not save those raw responses before trying to deliver them.

CoSER started a new per-case attempt directory without importing earlier partial
request caches. Identical successful payloads appeared in three separate attempts
for cases 170, 172 and 175. Those duplicate calls are independently actionable;
changing concurrency alone cannot repair this recovery behavior.

A previously failed RMT request, sent directly with the same payload and sampling,
completed in 182.115 seconds (884 input / 3,495 output tokens; reported cost
$0.00094005). Its response had no returned reasoning content and zero reported
reasoning tokens. This confirms that the 1,800-second client failure was not a
measurement of that request's direct API latency. The successful diagnostic
response is imported for reuse; it must not be requested again.

Remote evidence directory:

```
/workspace/migrations/old-full-run-20260914/outputs/full-public-v1/
  interruption-20260916T165640Z-broker-queue-audit/
```

This contains captured process/socket/state files, the direct probe request and
response, its ledger entry, and an offline old/new transport compatibility check.
Secrets are absent from these artifacts. The stopped supervisor and eight CoSER
children were retained as interruption evidence. Six possibly billed in-flight
remote calls added a conservative $0.60 unknown-cost reservation, not reported
spend. Existing reservations and the $100 cap remain intact.

## Scope of the fix

`run_gemma_queue_recovery.py` wraps the existing recovery launcher and pins its
SHA-256, as well as the original full runner. The long-RMT extension remains
pinned by the original launcher. Neither frozen file nor the engine source is
edited. Gemma is the only candidate allowed by the repaired broker; GLM 5.1
remains stopped, and DeepSeek is not run.

The runtime changes are:

- Match the sum of remote-capable producer pools to the global admission limit.
  Local and remote admissions remain separate. CharacterBench must already be
  complete before this recovery's producer budget is applicable.
- Use FIFO admission, cancel disconnected queued clients, and reject queue waits
  after 30 seconds. Allow 600 seconds upstream, below the unchanged 1,800-second
  caller deadline. Distinguish an upstream timeout (504) from an upstream HTTP 502.
- Combine simultaneous retries of the same request within the same case into
  one upstream call. Distinct cases/repeats remain independent even when their
  prompt text is identical; the scope is a loopback HTTP header, not a prompt
  modification. Write request
  intent and the complete raw response before delivery. Recover interrupted
  responses only when the original parser accepts them. Invalid/truncated output
  is never promoted into a successful score.
- Copy CoSER's earliest successful partial request cache for the same case into
  a new attempt. Preserve existing destinations. Allocate attempts after all
  existing attempt directories, including interrupted attempts without status.
- Retain failure records and prior attempt files. Persist reported cost and full
  usage; use a valid estimated_cost if cost is null. Never clear historical
  reservations or silently raise the budget.
- Preserve the previously used transport setting
  `chat_template_kwargs.enable_thinking=false`; reject unexpected returned
  reasoning. No instruction is added to benchmark messages.

Inputs, role/model names, revisions, temperature, max_tokens, seeds, prompt
templates, scoring parsers, aggregation rules and request fingerprints are
unchanged. Network timeouts and worker counts affect execution, not scoring.
This preserves the protocol; it does not imply deterministic model outputs or
guarantee that an external provider will never return errors.

`audit_gemma_recovery.py` records and later verifies existing request caches,
request attempts, CoSER attempt evidence, CharacterBench results, frozen input
files, engine hashes and DeepSeek summaries. Manifest execution status may
change, but its protocol fields must not change. It refuses to replace an
existing baseline. It is read-only with respect to evaluation data and makes no
API calls.

## Validation and operation

Offline validation:

```
python -m unittest discover -s tests/evaluation -p test_recovery_queue_guard.py -v
python -m unittest discover -s tests/evaluation -p test_benchmark_public_runners.py -v
```

The 20 new regression tests cover FIFO fairness, disconnected sockets, bounded
queueing, single billing of duplicate requests, response replay after failed
delivery, CoSER reuse, payload identity, invalid/truncated output, thinking,
budget accounting, independent repeated samples and disk failure. The 16 existing public-runner tests pass.
Fourteen of 15 broader stdlib download/runtime checks pass; the remaining
existing check refuses the pre-existing dirty CharacterBench vendor checkout.
That unrelated checkout is left untouched.

The deployed host also runs the regression tests and validates frozen engine
hashes and the local served-model identity before execution. The offline check
using the actual frozen Broker and an actual recorded request returns identical
upstream JSON, response and reported cost before and after the repair, without
making any paid calls.

The wrappers are stored on the network volume under
`/workspace/services/queue-repair-v1/`. A controlled launch can use 12 global
remote slots (5 MiniMax + 5 RMT + 2 CoSER producers), keeping local capacity at 16:

```
/opt/eval-recovery/bin/python /workspace/services/queue-repair-v1/run_gemma_queue_recovery.py \
  --root /workspace/migrations/old-full-run-20260914 \
  --code-root /workspace/projects/Chat-Model-Train \
  --remote-concurrency 12 --public-workers 5 --coser-workers 2 \
  --local-concurrency 16 --port 18081 --budget 100 --execute --detach
```

Before restarting after interruption, verify the old supervisor and all its
children are stopped, then archive stale per-benchmark `.run.lock` files instead
of deleting evidence. The original launcher archives stale supervisor state.
No recurring monitor is created. Each launch records wrapper hashes, arguments
and runtime limits in `transport-repair-v1/deployment-*.json`.

Inspect queue depth, admission rejections, raw response journals and newly valid
judgments, not merely process liveness. Coverage remains incomplete until every
benchmark's required judgments are validated. Historic client attempt counts,
actual upstream calls, reported spend and unknown-cost reserves are different
quantities and must be reported separately.
