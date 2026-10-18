# Qwen3 0.6B no-system checkpoint 4500: CoSER

Target host: `root@211.72.13.201:42123`.
Endpoint: `http://127.0.0.1:8040/v1`.
Served model: `qwen3-0.6b-her-no-system-4500`.
Weights: `/workspace/training/qwen3-her-no-system-5090-v1/train/checkpoint-4500/model.safetensors`.
SHA-256: `50d14f7da496cbbaf5c50b51fb2f0456178a8064ab798449bf1817b6ce3124e7`.

The independent configuration is
`configs/evaluation/roleplay_benchmarks/historical/qwen3-0.6b-her-no-system-4500-coser-herprompt-20260922.json`.
It uses the existing HER-format CoSER evaluation path, the original 200 CoSER
scenes, GLM 5.3 Flash helper/judge, actor temperature 0.7, actor token cap 16384
within a 32768 context window, helper temperature 0 and token cap 8192.
Native actor thinking is disabled. Remote thinking parameters remain the
previous provider-compatible empty object; this does not establish disabled
judge reasoning. HER system-thinking is removed from visible output; role
thoughts and actions are converted using the existing adapter. The actor's own
history remains postprocessed, as in previous comparisons, rather than claiming
an exact reproduction of upstream HER history handling.

Shared endpoint limits are local 64 / remote 96, with 128 CoSER scene workers.
This is one evaluation task; no second independent run is stacked on the judge.
No model deployment, training process, benchmark prompt code, or storage code is
changed. The stored HER output instructions are intentionally unchanged for
comparison with prior HER-format runs, including their system-thinking request;
the checkpoint's no-system training name does not change the evaluation prompt.

## Locations and launch

- Repo: `/workspace/projects/public-evaluation`.
- Credentials and endpoint variables:
  `/workspace/projects/public-evaluation/.local/qwen-no-system-4500-20260922/endpoints.env` (0600).
- Pilot: `/workspace/runs/roleplay-evaluation/qwen3-06b-her-no-system-4500-coser-herprompt-pilot-20260922-v2` (case 0068).
- Full: `/workspace/runs/roleplay-evaluation/qwen3-06b-her-no-system-4500-coser-herprompt-20260922`.
- Supervisor names: `eval-qwen-nosystem4500-pilot`, `eval-qwen-nosystem4500-full`.
- Supervisor logs stay outside newly prepared run directories, under the `.local`
  directory above. The runner requires a fresh directory when preparing.

The pilot must complete and replay without new API operations before launching
the full run. Its exact successful case files, raw request journal and accounting
records can then be imported into the fresh, separately prepared full run after
checking identical configuration apart from scene selection and identical frozen
sources apart from `coser-cases.json`. A SHA-256 import receipt records every copied
file. Full-run totals include the pilot's costs; do not add pilot and full totals.
The successful pilot scene is skipped during the full run, avoiding a second charge.

```bash
cd /workspace/projects/public-evaluation
bash scripts/evaluation/run_public_evaluation.sh \
  --env-file .local/qwen-no-system-4500-20260922/endpoints.env run \
  --config configs/evaluation/roleplay_benchmarks/historical/qwen3-0.6b-her-no-system-4500-coser-herprompt-20260922.json \
  --run-dir /workspace/runs/roleplay-evaluation/qwen3-06b-her-no-system-4500-coser-herprompt-20260922 \
  --execute
```

Run through Supervisor on this host for continued operation after SSH disconnects.
Do not execute the command concurrently with the Supervisor job. Outputs use the
mounted local xfs `/workspace` volume, preserving the existing asynchronous writer
and separate accounting process without depending on volatile `/dev/shm` storage.
