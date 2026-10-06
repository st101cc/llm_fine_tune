# ForgeTune pipeline safeguards and task pilots

## Outcome

Implemented task-first review, configurable evaluation size, reproducible
whole-dataset sampling, exact-input leakage checks, response-only SFT, and
report-only final evaluation. No trained adapter was deployed.

| Synthetic task | Base | Tuned | Interpretation |
| --- | --- | --- | --- |
| Ticket classification | 30/30 correct | 30/30 correct | No observed benefit from fine-tuning |
| JSON extraction | 29/30 exact matches | 30/30 exact matches | One additional correct answer; too little evidence for a general improvement |
| Summarization | 30 answers saved | 30 answers saved | Requires rubric-based human review; no automatic accuracy claimed |

These fixtures vary identifiers and small numeric fields within simple
templates. The final inputs differ from training inputs, but the templates
are shared. They validate pipeline mechanics, not generalization to real
support tickets, unseen writing styles, or production data.

## Experiment controls

- Cached Qwen2.5-3B-Instruct, QLoRA rank 8 / alpha 16.
- One epoch, learning rate 0.0001, batch size 1, accumulation 8.
- 96 training / 30 development / 30 final rows per task; 12 optimizer steps.
- Sequence length 256; no training rows truncated in the completed pilots.
- Same final questions, prompts, greedy decoding and 96-output-token budget
  for paired base/adapter answers.
- Actual TRL collator labels checked: context tokens were ignored and target
  tokens remained trainable.
- Offline execution, no hosted APIs, no model downloads, no adapter deployment.
- Pilots, including failed attempts, used about 16.7 minutes elapsed runtime.
  A VM unit-test run also briefly loaded models; counting its entire 220-second
  run on both GPUs gives a conservative combined allowance of about 24.1
  GPU-minutes, below the approved 30-minute cap. This is not a billing measurement.

## Summarization spot check

Inspected final rows 126, 135, 145 and 155. The tuned answers followed the
reference wording. They also copied its grammatical error, "1 days".
Some base answers added wording that the seller would not provide a new
delivery date, which the input did not establish. This is a four-example
qualitative inspection, not an accuracy score. The reference grammar needs
correction before using these examples beyond a smoke test.

## Safeguards

New API-created workflows pause for a task description and success metric
before querying GPU model candidates. The brief is recorded, not automatically
converted into a validated custom scorer. Users can cancel at this checkpoint.

Evaluation is configurable from 1 to 200 examples per phase, default 50 and
limited by the available split. Small sets are labelled exploratory. Audits
sample 600 rows with a fixed seed across the dataset; this is not a full
row-by-row audit or a complete privacy guarantee.

Repeated normalized inputs stay in one split even when target answers differ.
External test overlap is rejected; legacy manifests are checked before
training. These checks do not identify paraphrases, shared entities or
time-based leakage.

Chat SFT supervises the final assistant turn, with earlier turns as context.
Prompt/completion and classification supervise the answer; plain-text data
remains full-sequence language modelling. Incompatible token boundaries and
truncation leaving no answer targets fail clearly.

Final quality scores and judge failures cannot trigger retraining. Training
and artifact failures may be retried before final evaluation. Existing saved
workflow decisions are not rewritten.

## Verification and operational findings

- 66 backend tests passed locally and on the VM with CUDA disabled.
- 4 Compass Node tests passed.
- 9 browser scripts passed, including the new task checkpoint, configurable
  sample count, large-benchmark hosted guard, cancellation control, and mobile
  layout checks.
- The new regression tests failed before the corresponding fixes.
- After activation, a live 60-row synthetic upload stopped at task review
  with the default 50-example setting, then cancelled successfully without
  model inference or training (workflow `a5c906eb-531b-404c-a591-1f64f664badf`).
- The first GPU attempts failed because another user's process occupied GPU 0.
  It was left untouched; the pilots moved to GPU 1.
- Classification exposed a datasets-library compatibility issue after training:
  adding two Column objects is unsupported. Iterating both columns fixed it;
  the subsequent classification pilot completed successfully.
- Existing inference-cleanup test doubles were updated with the tokenizer and
  tensor-shape methods used by the current implementation.
- An independent reviewer could not start due to model capacity; the final
  review was performed locally, not represented as an independent approval.

## Evidence and reproduction

The runnable harness is `trainer/pipeline_pilot.py`. It requires an existing
local model snapshot and writes into a new runtime directory. It enforces
`--seconds` (maximum 1800), honors `CUDA_VISIBLE_DEVICES`, and accepts
`--tasks classification extraction summarization`.

Local paired answers, metrics and token-retention evidence are under
`.forge-data/pilot-results/2026-10-05/{classification,extraction,summarization}/`
at the repository root.

VM artifacts remain under `~/forgetune-backend/data/pilots/`:

- `pipeline-safeguards-20261005`: initial GPU-conflict failures.
- `pipeline-safeguards-20261005-gpu1`: extraction and summarization successes,
  plus the classification post-training compatibility failure.
- `pipeline-safeguards-20261005-classification`: successful classification retry.

Backend rollback files are in
`~/forgetune-backend/data/deploy/pipeline-safeguards-20261005/backup/`.

## Next useful experiment

Use a curated sample of real intended tasks, reviewed reference answers,
entity/time-aware splits where appropriate, and a larger untouched test set.
Keep the base model unless measured benefits justify tuning and its operating
cost. Do not adopt these synthetic-pilot adapters as production improvements.
