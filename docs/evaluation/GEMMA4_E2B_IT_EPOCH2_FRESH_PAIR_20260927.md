# Gemma E2B IT and HER Epoch 2: fresh CoSER pair

On 2026-09-27 the user explicitly requested new full evaluations of both models.
The earlier Epoch 2 run remains partial (153/200) with shutdown-related ambiguous
remote requests quarantined. It is not overwritten or relabeled as complete.

| Setting | Base IT | HER Epoch 2 |
| --- | --- | --- |
| API model | gemma4-e2b-it | gemma4-e2b-her-epoch2 |
| Endpoint | http://127.0.0.1:8050/v1 | http://127.0.0.1:8051/v1 |
| GPUs | 0,1 independent replicas | 2,3 independent replicas |
| Actor protocol | CoSER native | HER official without-system |
| Judge | GLM 5.3 Flash | GLM 5.3 Flash |

Both use the pinned original CoSER 200 scenes, BF16 serving, 65536 context,
temperature 0.7, max_tokens 16384 and native thinking false. No repetition penalty
is injected. Both judge views retain role thoughts. HER system analysis is
removed, role tags are converted for judging, and self-history keeps role tags.
Prompt/history protocols differ between models as previously requested.

Configs are unchanged from `gemma4-e2b-it-coser-20260926.json` and
`gemma4-e2b-her-epoch2-coser-20260926.json` in
`configs/evaluation/roleplay_benchmarks/`. Each has local concurrency 64, remote
48 and 128 scene workers. Only these two evaluations run, using the explicit
independent-endpoints option with aggregate remote ceiling 96.

New run directories under `/workspace/runs/roleplay-evaluation/`:

- `gemma4-e2b-it-coser-native-fresh-pair-20260927`
- `gemma4-e2b-her-epoch2-coser-official-nosystem-no-rp-fresh-pair-20260927`

No prior responses are imported into this user-requested fresh repeat. The
existing caches, successes and ambiguous request evidence remain in the old
directories. New successful responses remain reusable on within-run recovery.
Existing asynchronous persistence and accounting are unchanged.

Remote repository: `/workspace/projects/public-evaluation` on
`root@211.72.13.201:42123`. Receipts, config hashes, scripts and logs:
`.local/gemma4-e2b-it-epoch2-fresh-pair-20260927/`.

Supervisor services:

- `eval-coser-gemma-e2b-it-fresh-20260927`
- `eval-coser-gemma-e2b-her-epoch2-fresh-20260927`

The deployed Epoch 2 adapter hash was verified before launch:
`370ed6d2d499995f27cf6154b852a83b11de35a70ade8b189c75df497c189efc`.
Read each run's `coser/outputs/full-public-v1/candidate/coser/summary.json` for
coverage and scores, and `usage-summary.json` for final accounting. Report common
successful scenes if coverage differs; Supervisor exit alone is not completion.
