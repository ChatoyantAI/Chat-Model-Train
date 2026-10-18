# CoSER speaker-label paired rejudge

The user requested adding speaker labels and evaluating IT and full epoch 1
together. This run isolates labels by reusing existing successful trajectories;
it sends no actor, environment or NSP generation requests. Both deployed model
identities were verified by GET /models on 8050 and 8051.

The unified CLI now has a `coser-rejudge` command. It preserves each source's
exact accepted judge payload, adding missing actor name prefixes only to the
user transcript. Existing plain/Markdown speaker prefixes and environment text
are preserved. Misattributed text produced by the model is not rewritten.
Original judge system prompts, sampling, schema and score formula are unchanged.
Consequently this isolates label effects, not all CoSER/HER protocol differences.

Sources under `/workspace/runs/roleplay-evaluation/`:

- Group 0, IT: `gemma4-e2b-it-coser-native-fresh-pair-20260927` (200 successful).
- Group 1, full: `gemma4-e2b-her-full-epoch1-coser-official-nosystem-no-rp-20260929` (199 successful).

Common subset: 199 scenes, excluding case 67. Four dimensions per model:
1,592 planned judge requests before retries. All 796 IT judge transcripts gain
at least one missing actor label. All 796 HER judge transcripts remain identical
to their previous input; new judgments form a repeat-control condition.
This deliberately requests a fresh control repeat; subsequent recovery of this
new run replays already successful journal responses without new charges.

Both groups run concurrently, 48 requests each and a shared global maximum 96,
under exclusive judge-endpoint ownership. Existing JournalSender, DerivedStore
and process-isolated UsageLedger provide cache recovery, async checkpoints and
accounting. Ambiguous charged requests remain quarantined. Old runs are read-only;
inputs/source-result hashes and original request paths are captured in inputs.json.

Prepare (no paid calls):

```bash
python scripts/evaluation/run_public_evaluation.py --env-file PRIVATE_ENV \
  coser-rejudge --source-runs IT_RUN FULL_RUN --run-dir NEW_RUN
```

Execute/resume:

```bash
python scripts/evaluation/run_public_evaluation.py --env-file PRIVATE_ENV \
  coser-rejudge --run-dir NEW_RUN --execute
```

Remote result directory:
`/workspace/runs/roleplay-evaluation/gemma-e2b-it-full-epoch1-speaker-rejudge-20260929`.
Supervisor: `eval-coser-gemma-e2b-speaker-rejudge`.
Launch/log: `/workspace/projects/public-evaluation/.local/gemma-e2b-speaker-rejudge-20260929/`.
Read `progress.json`, `summary.json`, `usage-summary.json`; judgments are separated
under `model-0/` and `model-1/`. A scene contributes to its model's summary only
after all four dimensions succeed. If failures differ, compare the intersection
of successful scenes, not unmatched means.

Validation: local and remote each passed seven tests, including original HER
semantics, label preservation, strict critique parsing, original scoring,
immutable source evidence, journal replay without repeat charges, and rejection
of modified frozen inputs. No frozen benchmark scripts or upstream source edited.
