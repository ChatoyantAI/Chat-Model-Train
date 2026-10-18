# HER-32B: HER format instructions and output postprocessing

User requested a separate full CoSER run replacing the original output-format
instruction with the existing `actor_prompt_format=her-official-v1` adapter.
Configuration: `configs/evaluation/roleplay_benchmarks/historical/her-32b-coser-herprompt-20260921.json`.

Only the actor output-format instruction and protocol label change relative to
the original-prompt HER-32B configuration. The same model revision, GPU 0/1 BF16
TP2 deployment, 200 CoSER scenes, character profiles, scenario, 60-word request,
temperature 0.7, context-aware 16384-token allowance, native thinking=false,
GLM 5.3 Flash helper/judge, local concurrency 64, remote concurrency 96 and 128
scene workers are retained. No forced assistant prefill is added.

Output postprocessing removes system thinking, converts role thoughts to square
brackets and actions to parentheses. Role thoughts are available to the judge
and hidden from other characters. As in our previous runs, the actor's own
history receives the processed response. HER upstream instead keeps the raw
response in its own history: this run tests the format-instruction change and
does not claim to reproduce the entire upstream evaluation protocol.

The earlier single-request probe produced HER system/role tags with native
thinking disabled: 164 output tokens, stop finish, 2.54 seconds. The probe was
unscored. Its raw response remains in the original run's diagnostics directory.

Baseline: `her-32b-coser-originalprompt-v1-20260921`, 200/200 successful,
CoSER average 43.868125. New independent result directory:

```text
/dev/shm/roleplay-evaluation/her-32b-coser-herprompt-v1-20260921
```

The existing unified runner prepares fresh inputs and resumes within the new
directory; no previous conversations, judgments or request caches are imported.
The runtime copy under `/dev/shm/her32b-eval-runtime` is used while network-volume
reads are unreliable. Source is committed locally; RAM results need persistence
before shutdown. The same original pinned CoSER repository revision is used:
`624efc50db7be3a4e37cec6a1c58ed434dc9f2ce`.
