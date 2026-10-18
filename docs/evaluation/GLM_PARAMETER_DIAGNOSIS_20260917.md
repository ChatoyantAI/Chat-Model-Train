# GLM slowdown: controlled parameter diagnosis

The migration launcher injected `chat_template_kwargs.enable_thinking=false`
into GLM-5.3-Flash requests. The original frozen full-run broker did not.
The queue repair initially retained that migration behavior; it was not
equivalent to the original DeepSeek run's upstream request.

## Live evidence

At 2026-09-16 18:11 UTC, three independent diagnostic requests used identical
messages, temperature 0, max_tokens 1024, model and endpoint. Concurrency was
at most two. No successful benchmark sample was requested again.

| Parameter | Seconds | Output tokens | Result |
| --- | ---: | ---: | --- |
| Omitted, as in original broker | 0.993 | 24 | Direct JSON, no reported reasoning |
| `chat_template_kwargs.enable_thinking=false` | 45.427 | 123 | Analysis mixed into content |
| `reasoning_effort=none` | 44.312 | 123 | Same analysis mixed into content |

Both explicit-off variants reported zero reasoning tokens despite visible
analysis. Zero reported reasoning tokens does not establish that thinking is
disabled. One request per variant establishes a reproduction, not a latency
distribution or a 64-concurrency capacity test.

A fourth call retried only a previously truncated real request, operation
`4013be51bd974d92beaeeac4974397e2`. The only payload change was removing the
injected `enable_thinking=false`. Original messages and max_tokens 8192 were
preserved. The saved original response exhausted 8192 output tokens and ended
with `length`. The diagnostic completed in 86.814 seconds with 454 output
tokens and `stop`, returning a complete scoring JSON. However, it reported
**38 reasoning tokens and 175 characters of reasoning_content**. It is kept
as diagnostic evidence, not imported as a valid non-thinking benchmark score.

Provider-reported estimated costs across these four calls total $0.00037645.
This is API usage metadata, not an independently reconciled invoice.
Requests, full responses and per-call summaries are retained under:

`/workspace/migrations/old-full-run-20260914/outputs/full-public-v1/diagnostics/glm-parameter-ab-20260916T181131Z/`

## Interpretation and limits

The injected parameter reproduces a material change in behavior and latency;
the slowdown cannot be blamed solely on remote capacity or concurrency.
Earlier ledger analysis found 202 of 204 newer 502 entries around the original
300-second deadline. The frozen broker labels general transport exceptions
502, so that count is not proof of 204 actual provider HTTP 502 responses.
Excessively long outputs explain a plausible path to those timeouts, while
the independent queue starvation and error-labeling problems remain real.

The vLLM issue describes the same parameter/parser mismatch:
https://github.com/vllm-project/vllm/issues/54744
Its GLM-5.3-Flash recipe documents always-on thinking:
https://recipes.vllm.ai/zai-org/GLM-5.3-Flash
DeepInfra's generic `reasoning_effort=none` documentation did not match the
observed behavior for this model. These sources do not establish which
backend implementation or version DeepInfra actually serves.

## Code and operational state

The wrapper now removes the migration-injected false flag for this exact GLM
model, restoring the original broker payload. Gemma retains its explicit
non-thinking flag. Frozen scripts, prompts, scoring schema and input data are
unchanged. Existing scoped response journals remain readable using their
legacy keys, and completed caches are not deleted or rewritten.

The reasoning-response guard remains active. Restoring the old request is
**not** sufficient to claim compliance with strict non-thinking requirements.
The evaluation stays paused; the GLM 5.1 candidate run also stays paused.
Before a full paid restart, resolve the distinction between reproducing the
old provider-default protocol and requiring verified zero reasoning. Do not
silently change judges, scoring prompts, or relax that requirement.

The same pending repair includes durable checkpoint writes and skipping
already-completed trajectories; it does not add benchmark samples or judge
calls. Validation: 27 offline recovery tests, including the exact GLM payload
regression and same-case legacy journal reuse without cross-case reuse.

## Follow-up: model contract versus unverified provider internals

The model author's own API guide explicitly says GLM-5.3-Flash only supports
`thinking.type=enabled` and thinking cannot be disabled:
https://docs.z.ai/guides/vlm/glm-5.3-flash
The official model card lists `low`, `high`, and `max`, with other effort
values falling back to `max`:
https://huggingface.co/zai-org/GLM-5.3-Flash
Inspection of the author's actual template confirms it never reads
`enable_thinking`; it unconditionally opens `<think>` for generation:
https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/chat_template.jinja

This establishes a model-specific contract mismatch without assuming that
DeepInfra uses vLLM. DeepInfra's generic reasoning-off documentation is not
proof that this particular model supports an off mode. The prior diagnostics
called the DeepInfra HTTPS endpoint directly with urllib, bypassing our
broker, benchmark parser and local Gemma vLLM. The analysis text was already
present in the provider's raw `message.content` response; our local parser
did not insert it. The three synthetic requests have identical message hashes.
The responses identify GLM-5.3-Flash but expose no system fingerprint.

Read-only GETs of DeepInfra's model info and default native schema succeeded;
both raw documents are saved alongside the diagnostics as
`provider-model-info.json` and `provider-schema.json`. They do not establish
which serving engine, parser implementation, or per-request worker handled
the OpenAI-compatible requests. Do not label a specific vLLM bug as the
confirmed DeepInfra root cause. The supported conclusion is: our injected
off parameter is incompatible with the model contract, and the provider's
observable handling does not actually disable reasoning. Whether that
handling changes the template, parsing, or routing requires provider-side
evidence. A single A/B per setting does not assign all historical 502s or
their exact latency contribution to that parameter.

No further paid inference calls were made for this follow-up. Strict
non-thinking and retaining this judge are not currently a verified compatible
combination. `low` is still thinking and must not silently replace the frozen
judge configuration or be represented as equivalent to the previous run.

## Authorized historical-policy restart

The user subsequently explicitly requested omitting GLM thinking parameters,
using the old concurrency, and restarting. The wrapper now accepts and retains
the judge's separate reasoning fields under this historical policy; Gemma's
non-thinking guard is unchanged. Supervisor metadata no longer claims all
models have thinking disabled. Neither benchmark messages nor scoring parsers
change. The already-paid real-request diagnostic is available in its original
scoped journal for normal frozen-parser validation instead of another API call.

Restart settings: global remote limit 64, MiniMax/RMT 28 producers each,
CoSER 8, local Gemma limit 16 across GPUs 0/1. The existing $100 budget and
unknown-cost reserve remain intact. The GLM 5.1 candidate task stays paused.
No scheduled monitor is created. Preflight verifies frozen hashes and served
model identity; 28 offline recovery tests pass, including judge reasoning
acceptance/replay and continued rejection for models requiring thinking off.
