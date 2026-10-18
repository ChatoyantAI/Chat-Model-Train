# Storage interruption and recovery

The Gemma supervisor stopped dispatching after an OSError around
2026-09-16 21:14:55 UTC (05:14:55 Beijing). Its in-memory halt was
`transport_persistence_failure_OSError`; active requests subsequently drained
to zero, but the supervisor still said `running`. The frozen callers continued
iterating missing judgments against the halted broker.

The exception location is narrowed to accounting persistence: the outer
handler set the halt after dispatch had updated the success counters; there
were no recorded dispatch errors. The previous handler discarded errno and
path, so the historical kernel/storage error cannot be determined exactly.
Do not claim that the disk was full or that a particular network error was
proven. A fresh durable-write probe succeeded.

`/workspace` is a GeeseFS network mount. The original append-only ledger was
43,486,325 bytes. Each successful request synchronized that growing file while
holding the shared broker lock. Three writes to an isolated copy took 3.072,
1.030 and 2.823 seconds; tiny-file append/sync controls took 0.034–0.036 seconds.
This identifies excessive I/O and lock contention; it does not recover the
missing errno from the historical interruption.

## Repair

- Leave the original ledger immutable; write new entries to bounded JSONL
  segments under `transport-repair-v1/ledgers/`. Load both original and segment
  charges/reserves at startup. Rotate after 256 KiB.
- On an ambiguous append/sync failure, repair the exact complete segment,
  preventing duplicate charges. Retry transient atomic writes at most three
  times, without repeating a paid API call. Permanent failures still halt.
- Record errno, affected path and phase. Report `halted` in both broker and
  supervisor state. A halted-broker header causes public runners to stop the
  current case instead of retrying every missing judgment for hours.
- Preserve successful caches, frozen prompts, scoring parsers and old errors.
  GLM judge uses the user-authorized provider default with no thinking fields.
  Gemma retains thinking off. Remote/local concurrency remains 64/16.

The actual segment writer on the network volume measured 0.067, 0.080 and
0.057 seconds for three diagnostic entries. An atomic replacement of a
256 KiB segment took 1.342 seconds; this more expensive path is reserved for
repair, rather than used for every normal append.

## Reconciliation and validation

Every recorded operation request ID was found in the original ledger; there
were no malformed ledger lines or unaccounted operation IDs. Its SHA256 was
`0961b9625de297fad7d88b9c79deb79fd37f50c741111a89674c8ff381cd65e5`.
Provider-reported cumulative cost was $37.199015647 and unknown-cost reserve
$58; the existing $100 guard is unchanged. Reserve is not confirmed spend.

33 offline recovery tests pass on Windows and the Linux host, including an
ambiguous successful append followed by EIO: one upstream call, one ledger
entry, one charge. Coverage also includes segment-cost reload, retained legacy
ledger bytes, precise failure reporting, and halted supervisor/client behavior.
The 16 existing public-runner tests pass locally.

Remote evidence is under:
`/workspace/migrations/old-full-run-20260914/outputs/full-public-v1/interruption-20260917T014446Z-storage/`
No successful benchmark sample was rerun for diagnostics. The interrupted
supervisor was stopped only after verifying zero active requests, and stale
locks/logs were archived. GLM 5.1 and DeepSeek were not restarted.
