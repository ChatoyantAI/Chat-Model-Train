# Offline Gemma E2B storyline comparison

Artifact: `GEMMA_E2B_STORYLINE_COMPARISON_20260929.html` in this directory.
Open directly in Chrome or Edge; no server, API, CDN, or network requests.
The generated 14.3 MB artifact is kept outside the code commit, like the earlier
trajectory exports. The builder and template are versioned.

## Scope

199 common scenes from the September 27 IT run and September 29 full epoch 1
checkpoint-10161 run. Case 67 failed in the original full-model run and is
excluded from both sides. The page includes all saved judge-view dialogue turns,
reference context, original and speaker-label-rejudge scores and critiques,
exact old/new user transcripts, and clearly separated Chinese review notes.
These are processed judge views, not raw API generations.

Speaker rejudge finished with 1,591 successful dimension judgments and one
failure: HER case 102, Storyline Consistency, HTTP 400. No zero imputation or
silent substitution with the original score. Storyline Quality is complete for
both models across all 199 scenes: IT 39.93969849, full 32.27638191; full is lower
in 125 scenes, higher in 72, equal in two. The runner's all-four-dimensions summary
excludes HER case 102 and therefore reports a slightly different SQ mean; this
page explicitly uses the matched 199 SQ results.

The page highlights real internal contradictions (196, 127), a positive example
(46), and an unreliable high-score example (93). It documents the remaining
generation-protocol confounds and the missing thought/motivation input mapping.
It does not claim to isolate training as a causal explanation.

## Rebuild

```powershell
python scripts/evaluation/build_storyline_comparison.py
```

Default inputs under `.local/storyline-audit-20260929/`:

- `results.json`: immutable exports of five historical runs.
- `label-rejudge-results.json`: per-dimension rows, manifest and summary from the
  completed speaker-rejudge job.
- Locked dataset: `vendor/benchmarks/CoSER/data/test/test_set.json`.

Use `--input-dir` and `--output` to change export locations. Source hashes and
remote run paths are embedded in the HTML. No credentials are embedded.
The script validates scene matching, exact saved transcript reconstruction,
and every annotated quotation before writing the page.

## Verification

Chrome/Playwright checked desktop 1512×1050 and mobile 390×844 views, and
screenshots were visually inspected. No JavaScript errors, external HTTP
requests, or mobile horizontal overflow. Verified search, positive-case
filtering, annotation jumps, thought/environment switches, critique-round
switching, exact input display, and per-case JSON export. All 3,183 available
dimension scores (old and new) were independently recomputed in the browser
from saved severities and actor-turn counts, with no mismatches.

Local QA artifacts: `.local/storyline-audit-20260929/html-qa.json` and
`comparison-*.png`. Building and viewing the page makes no paid requests and
does not alter any evaluation results or service configuration.
