# TAIDE T4 inference baseline

This phase performs offline inference and human evaluation only. It does not train,
accept model license terms, provision GPUs, or deploy a service.

## Configuration

`fp16` remains the default. Explicit `bnb4` uses NF4, double quantization and FP16
compute on one idle GPU, with a 12,000 MiB free-memory admission threshold.
The threshold is not a loading guarantee. FP16 retains its 24,000 MiB threshold.
Input plus reserved output is capped at 2,048 tokens for bnb4 (4,096 for FP16).
Inputs and outputs that exceed their limits are rejected, never silently truncated.
Generation is greedy, sequential, local-files-only, and records immutable model
revision, prompts, quantization, timings and CUDA allocated/reserved peaks.
Protocols from different load modes cannot be directly paired by the evaluator.

The API accepts `loadMode: "bnb4"` on `POST /quality/evaluations` when `generate`
is true. Omitting it preserves FP16. Execution uses the existing supervisor and
persistent 1,800 GPU-second ledger. Load time and failures consume this budget;
cancellation and timeout terminate only owned workers. Atomic per-answer
checkpoints retain completed answers through termination.

## Fixed baseline

The baseline freezes 30 prepared TMMLU+ questions plus 10 synthetic binary
customer-service cases and 10 synthetic response-writing cases before generation.
The positive class is `需升級處理`. Five classification examples are positive and
five negative. Each writing example carries factual, instruction-following and
Taiwan-language rubrics. No fabricated human judgments are supplied.

Frozen revisions prepared on 2026-10-06:

- TAIDE: `6738923786667dfe00363b1e2d225536817b429e`
- TMMLU+ v1.1: `466ccd9752ce6889f35c47f875ac7a65012e2269`

Run commands in the trainer directory using its configured Python environment.
On the existing VM, use its existing runtime root and HF cache configuration.
For example, `--runtime-root ./data` is correct only when it is the same directory
used by the existing backend; using a different root would create a different ledger.

After the model owner has granted access and credentials are configured, download
separately (this checks existing access; it never accepts terms):

```text
python quality_models.py prepare model --revision 6738923786667dfe00363b1e2d225536817b429e
```

Freeze without generating:

```text
python quality_smoke.py --runtime-root ./data --benchmark-revision 466ccd9752ce6889f35c47f875ac7a65012e2269 --model-revision 6738923786667dfe00363b1e2d225536817b429e --load-mode bnb4 --prepare-only
```

Remove `--prepare-only` to run. A single-question smoke test must finish before the
remaining 49 questions are attempted. Both loads share the requested time allowance
and the existing aggregate budget, with a maximum of 1,800 remaining GPU seconds.
No other model is substituted. `--seconds` can reduce this allowance.

## Evidence and human decisions

Artifacts are under `data/quality/baselines/<manifest hash>/`, never committed.
`manifest.json` fixes cases, references, rubrics, policy, revisions and settings.
`answers.json` preserves original outputs. `run.json` and timestamped `attempts/`
retain status, failures and budget information. Failed zero-answer attempts can
be retried; completed or partial answers are preserved for inspection.

`report.json` separates TMMLU+ accuracy/invalid answers, classification
accuracy/precision/recall/F1, and writing pass/fail/pending counts. It includes
every case, missing answers and language findings. Metric denominators count
answered items; missing coverage is explicit. Invalid predictions are incorrect;
an invalid prediction for a positive reference counts as a false negative.
Undefined precision/recall/F1 is null. Small samples are exploratory and cannot
establish leaderboard performance or improvement from corrected training data.

Edit a copy of `reviews.json`. For writing, set each of `facts`, `instructions`
and `taiwanUsage` to `pass`, `fail` or `needs_review`. Record `reviewer` and `reason`
for completed decisions. For language suggestions, select `accepted`, `ignored`
or `pending` and give a reason for accepted/ignored suggestions. Preserve each
finding and output hash. Accepted suggestions do not rewrite model outputs.

Rebuild the report without inference by adding `--review <edited reviews path>`
to the same baseline command. Each validated review is saved as a timestamped
snapshot, retaining decisions and reasons. Automated language suggestions remain
separate from human correctness judgments.

## Current live-validation status

On 2026-10-06, the existing VM's Hugging Face access check returned HTTP 401 for
TAIDE, including when using its configured HF cache location. Weights are missing.
The 50-question baseline is prepared locally; no TAIDE GPU inference or human
scoring has completed. Successful T4 inference must not be claimed until the
owner configures an account/token with access and the smoke/full run succeeds.
