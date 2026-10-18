# Gemma E2B full fine-tuning epoch 1: CoSER

User-requested evaluation of `gemma4-e2b-her-full-epoch1` at
`http://127.0.0.1:8051/v1` on 2026-09-29. This is the full fine-tuned
checkpoint-10161, not the earlier LoRA checkpoint with the same step number.
Training remains paused; serving uses four independent replicas, BF16,
65536 context, with native thinking disabled.

The deployment export receipt records model SHA-256
`03c71d1dde7cb612de269d20fd166687154e957e16a3a2dcd36d7ba582b11e4f`.
Source: `/workspace/deployment-records/gemma4-e2b-full-epoch1-dp4-v1/export-complete.json`.

Configuration: `configs/evaluation/roleplay_benchmarks/historical/gemma4-e2b-her-full-epoch1-coser-20260929.json`.
Only actor identity, revision and run label differ from the previous HER epoch 2
configuration. The frozen CoSER 200 scenes and HER official without-system
protocol are unchanged. Role thoughts are retained for judging, role tags are
converted, and system-thinking analysis is removed. Native thinking is false;
no repetition penalty is supplied. Actor temperature is 0.7, output limit 16384.
GLM 5.3 Flash judges with temperature 0 and output limit 8192.

Concurrency remains local 64, remote 48, scene workers 128. Existing asynchronous
persistence and request accounting are unchanged. New model responses have an
independent run directory; prior results are preserved. Successful responses in
this run remain reusable for recovery.

Remote repository: `/workspace/projects/public-evaluation`,
`root@211.72.13.201:42123`.

- Run: `/workspace/runs/roleplay-evaluation/gemma4-e2b-her-full-epoch1-coser-official-nosystem-no-rp-20260929`
- Supervisor: `eval-coser-gemma-e2b-her-full-epoch1`
- Launch script, receipt and log: `.local/gemma4-e2b-her-full-epoch1-coser-20260929/`
- Coverage and scores: `coser/outputs/full-public-v1/candidate/coser/summary.json` within the run.
- Final accounting: `usage-summary.json` within the run.

API model availability checks passed for actor and judge before launch. Starting
the service is not evidence of full completion; inspect scene coverage and final
accounting before reporting results.
