# Increase RMT workers within the shared API cap

The previous recovery launcher capped each public suite at 28 workers. RMT
processes each case's judgments sequentially, so its request concurrency could
not exceed 28 even when the broker had unused capacity. At a measured mean
18.41 seconds per judge call, 28 workers imply approximately 5,475 calls/hour,
matching the observed 5,480/hour in a three-minute sample. Filesystem lookup
stalls had already been removed; the per-suite worker count was now limiting
RMT throughput.

Following the user's request to increase concurrency, `--rmt-workers 48`
explicitly overrides RMT's runtime pool size, including the frozen manifest's
32-worker setting. MiniMax retains `--public-workers 28`; local Gemma stays
at 16 in-flight requests. All remote calls still share the broker's FIFO gate
of 64. Producer pools can exceed that gate in aggregate because MiniMax
alternates local actor work with remote simulator work. Queued producers remain
bounded; each pool is capped at the remote limit and their sum at twice that
limit. Existing queue cancellation and admission deadlines remain active.

The RMT override is scoped to each `community.run` invocation with a ContextVar,
so two suites running concurrently cannot accidentally inherit one another's
worker settings. Frozen manifests, prompts, sampling, cache hashes, and scoring
remain unchanged. Broker/supervisor status and deployment metadata expose the
individual producer limits. Budget protection remains disabled as requested;
costs and unknown-cost records are still retained.

65 local tests passed, including concurrent RMT/MiniMax selection (48 versus
28), preservation of a smaller frozen pool when no explicit override is given,
and an eight-producer mock proving actual upstream concurrency never exceeds
its shared two-slot gate while every successful call is accounted for. The
remote preflight confirmed RMT 48 / MiniMax 28 / CoSER 8 and the unchanged
64-slot remote gate. CoSER is already complete and sends no new requests.

Deployment uses the loopback maintenance drain, waits for in-flight calls and
pending checkpoint writes to reach zero, and preserves all old logs/statuses
and caches before replacing only the supervisor. vLLM engines stay loaded.

Supervisor 97825 was launched from `rmt48-20260917`. The maintenance evidence is
under `outputs/full-public-v1/interruption-20260917T042340Z-rmt48/`; the runtime
log is `supervisor-run-20260917T042340Z-rmt48.log`. The drain waited for the
remaining long local generations and all checkpoint writes; no uncertain
interruption charges or additional quarantined requests were introduced.

Live status reached 64 active remote calls with RMT 48 / MiniMax 28 reported,
and subsequently fluctuated around 56–64. No new broker errors, pre-dispatch
rejections, or persistence failures appeared in the initial observation window.
In the minute ending at Unix time 1789619189.52, 169 RMT judge calls completed
(10,140/hour if extrapolated), plus 34 Gemma actor and 35 simulator calls.
RMT mean/median turnaround in that sample was 18.84/13.74 seconds. This is a
short-window measurement, not a guarantee of sustained throughput or exact ETA;
it demonstrates the higher pool is active and the shared API cap is being used.

## Next measured step: RMT 64, shared API 80

The user requested a further increase. Before changing the configuration, the
48/64 run had completed 618 remote calls with no broker errors or admission
rejections. A three-minute window contained 456 RMT judge calls (9,120/hour),
with mean/median turnaround 17.96/15.00 seconds and no checkpoint backlog.

The next configuration changes only `--rmt-workers 64` and
`--remote-concurrency 80`. MiniMax remains 28 workers, local Gemma 16, and CoSER
8 (already completed). The budget guard stays disabled. The existing scheduler
code supports these values without further implementation changes; the remote
frozen-source preflight confirmed RMT 64 / MiniMax 28 / CoSER 8 with an 80-slot
shared remote gate. The same drain and cache-preserving replacement procedure
is used, with deployment arguments retained on the network volume.

This configuration runs as supervisor 100065 from `rmt64-api80-20260917`.
The drain/replacement record is in
`outputs/full-public-v1/interruption-20260917T043259Z-rmt64-api80/`, and the log
is `supervisor-run-20260917T043259Z-rmt64-api80.log`. All outstanding responses
and checkpoints were saved before replacement; no inference engines restarted.

Initial live measurements reached 70–72 active remote calls. In the minute
ending at Unix time 1789619788.72, 273 RMT judge calls completed (16,380/hour
when extrapolated), plus 31 Gemma actor and 36 simulator calls. RMT mean/median
turnaround was 13.95/11.53 seconds. The broker had completed 592 remote calls
with no errors or pre-dispatch rejections, and no pending checkpoint backlog at
that snapshot. These are short windows at different times, so the improvement
cannot be attributed exclusively to worker-count changes or treated as a
guarantee of sustained throughput. The 64/80 configuration remains active.

## Further measured step: RMT 96, shared API 128

Following another user request to raise concurrency, the next runtime uses
`--rmt-workers 96 --remote-concurrency 128`. Before draining, the 64/80 run
had completed 2,566 remote calls with no broker errors or admission rejections.
A recent three-minute window contained 715 RMT judge calls (14,300/hour),
with mean/median turnaround 15.89/14.35 seconds. Actual remote concurrency
was around 72–77, below its 80-slot gate; the RMT producer pool remained the
constraint that this change addresses.

The unchanged launcher passed remote preflight with RMT 96 / MiniMax 28 /
CoSER 8 and a 128-slot shared API gate. Local Gemma stays at 16, and budget
protection remains disabled. Supervisor 104111 runs from
`/workspace/services/queue-repair-v1/rmt96-api128-20260917`; its log is
`supervisor-run-20260917T044706Z-rmt96-api128.log`. The replacement record and
drained broker snapshot are in
`outputs/full-public-v1/interruption-20260917T044706Z-rmt96-api128/`.

The prior supervisor stopped only after both active calls and pending
checkpoints reached zero. Its final counts were 2,643 remote and 305 local
successful calls, with no broker errors. Quarantine count (43) and unknown
cost reserve ($59.40) were unchanged across replacement. Existing caches,
responses, accounting, frozen inputs, and vLLM processes were retained.

### Diagnostic crash and recovery

The first 96/128 process was still replaying cached progress with the bounded
checkpoint queue full, and had not reached steady throughput. A SIGUSR1
thread diagnostic was requested. Its log contains a partial thread dump
followed by `Fatal Python error: Segmentation fault`; the process exited.
The exact native cause is unproven, so this is neither evidence of API
throttling nor a successful validation of 96-worker throughput. Further
signal-triggered thread dumps were stopped.

Before restarting, all request intents created by this process were checked
against durable ledger entries and raw responses. There were 44 unaccounted
intents (41 remote, 3 local), none with a saved raw response. Their keys were
quarantined to prevent automatic redispatch; $4.10 was added to the existing
unknown-cost reserve, not to provider-reported spending. Reconciliation rows
mark elapsed duration as unknown and distinguish the estimate from measured
request duration. Successful cached responses and original crash evidence
remain retained.

Supervisor 104836 resumed with the same 96/128 arguments after preflight.
Its log is `supervisor-run-20260917T045342Z-rmt96-api128.log`, and crash recovery
evidence is in `interruption-20260917T045342Z-diagnostic-crash/`. The first
status confirmed 87 quarantined keys, a $63.50 unknown-cost reserve, no budget
guard, and no new broker errors. Throughput must be measured after replay
finishes; the configuration increase alone does not establish a speedup.

By Unix time 1789621125.12, replay had cleared: live remote concurrency was
101, local concurrency 13, and pending checkpoints 1. The replacement process
had completed 597 remote and 47 local calls with no broker errors, admission
rejections, or halt. In the preceding minute, 303 RMT judge calls completed
(18,180/hour), with mean/median turnaround 15.81/13.79 seconds; MiniMax had
39 simulator and 31 Gemma actor completions. This short window is roughly
27% above the prior 14,300/hour observation, but includes startup ramp-up and
does not establish a sustained speedup or final score validity. The 96/128
configuration is left running; GLM 5.1 candidate evaluation remains paused.
