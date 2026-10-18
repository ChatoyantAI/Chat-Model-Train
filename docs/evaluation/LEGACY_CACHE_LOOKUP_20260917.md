# Avoid repeated negative lookups in the legacy response directory

After asynchronous checkpoints were deployed and the budget guard was removed,
the live diagnostic stack still showed 46 threads waiting in `Path.exists()`
inside `valid_journal_response.locate`, checking the legacy response directory.
The background checkpoint workers were idle. Only a small fraction of the
configured upstream concurrency was active. The write-path improvement alone
therefore did not fix the end-to-end bottleneck.

The recovery layer checks both current and historical GLM payload hashes before
dispatch. Those extra negative lookups in the old flat network directory were
absent from the original DeepSeek path. Removing an eager full-directory scan
from startup had left the per-request negative lookups in place; that was an
incomplete optimization, not evidence that the GPU or provider must be slow.

## Fix and cache consistency

`build_legacy_response_index.py --journal <transport-repair-v1>` enumerates the
immutable legacy directory once, without reading response bodies. It publishes
an atomic index only after the enumeration completes and the directory mtime
remains unchanged. The index records its absolute source directory, mtime,
response filenames, and a filename-list checksum. The loader refuses an index
whose directory, mtime, schema, or checksum does not match.

Asynchronous recovery uses that index for legacy membership checks. New sharded
responses remain dynamically discoverable, and cached response payloads, scope,
finish status, and parsing are validated as before. Synchronous rollback ignores
the index because it can still write to the old directory. If more legacy files
are migrated later, rebuild the index before resuming. With no legacy directory,
a fresh run uses an empty set; with an existing but unindexed legacy directory,
the old lookup remains the conservative fallback.

The live index contains 6,345 historical response filenames. Building it while
the old queries ran was blocked in the filesystem; after draining and stopping
the old supervisor, enumeration completed. The first attempt detected a changed
mtime and correctly refused to publish; a second complete enumeration succeeded.
No successful response, scoring input, parser, or cost record was deleted.

Deployment: supervisor 94920, stage `async-io-index-20260917`; maintenance archive
`outputs/full-public-v1/interruption-20260917T040531Z-legacy-index/`. All upstream
requests and pending writes were drained before replacement. The no-budget flag
and remote/local caps of 64/16 were retained. Deployment metadata records the
index file hash. The default recovery service files were updated with backups.

62 local tests passed, including successful replay with legacy `exists()` calls
forbidden, a negative cache lookup without filesystem probing, stale-index
rejection, and detection of a corrupted filename list. The remote frozen-source
and index preflight passed before launch.

## Historical timing comparison

In 160 historical DeepSeek attempt records from the first English RMT case,
31 successful actor requests had a median 3.67 seconds and 129 judge requests
had a median 8.27 seconds. Judge input/output medians were 680/196 tokens.
In a recent Gemma-run sample of 50 RMT judge requests, broker turnaround had a
median 19.84 seconds, with input/output medians 1,164.5/203.5 tokens. These are
different time periods and partial samples; broker timing also includes durable
recording, so they do not isolate provider latency or prove an intrinsic model
speed ratio. Both runs use the same judge model and sampling configuration.
DeepSeek's saved long-RMT summary confirms all 47,240 judgments completed.

## Initial live verification after deployment

The post-change snapshot reached 38–40 active remote calls and up to 16 local
calls, with no new broker errors. In a 60-second window ending at Unix time
1789618164.75, 64 RMT judge calls, 30 Gemma actor calls, and 27 MiniMax simulator
calls completed. That is 3,840 / 1,800 / 1,620 calls per hour respectively when
linearly extrapolated. The earlier four-minute sample had 78 RMT judgments
(1,170/hour), so this initial observed RMT throughput is about 3.3 times higher.
These short windows are operational evidence, not a controlled performance test
or a completion-time guarantee. In the new diagnostic snapshot, workers mostly
waited on model HTTP responses rather than negative legacy filesystem lookups.
The roughly 39–58 hour estimate based on the previous bottleneck is superseded;
about 45,000 remaining judgments at 3,840/hour would take roughly 12 hours if
that rate holds, before final accounting and any tail retries.
