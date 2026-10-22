# Original-vs-reviewed comparison — design

Status: approved in conversation 2026-10-22 (owner: cherrie.chen). Implementation plan to follow.

## Goal

From any reviewed dataset version, one action trains two TAIDE QLoRA adapters that differ only in the review's accepted edits, evaluates base / original-trained / reviewed-trained models, and shows whether the review improved Taiwan language usage without hurting knowledge.

It serves two audiences:
- a demo: one clear result page;
- a routine pre-training check: configurable sizes, a cost estimate, and a history of comparisons.

This generalizes `trainer/taiwanchat_experiment.py`, whose TaiwanChat result (base 4.0 → original 7.6 → reviewed 1.6 listed mainland terms per 1k chars) is the reference.

## Scope and assumptions

- Works on any reviewed version produced by the Dataset quality pipeline: zh-TW review policy (OpenCC + optional customer term list), base model TAIDE (`taide/Llama-3.1-TAIDE-LX-8B-Chat`, pinned revision), 4-bit QLoRA.
- Other languages or base models are out of scope. The quality pipeline is zh-TW/TAIDE-specific.
- Supported row formats: `messages` (single- or multi-turn; the final assistant turn is supervised, as in `train.py`) and `prompt`/`completion`.
- Plain `text` datasets have no prompt to evaluate on. For those, the action is disabled and the reason is shown.

## 1. Data and split

All of the following derive from the version's own audit. Nothing is dataset-specific.

- **Edited row:** a row where at least one *accepted* finding changed the answer. Original and reviewed rows align 1:1 (`rowLineage`).
- **Eval (held-out):** up to `evalPrompts` (default 60) edited rows.
  - The prompt must be a single user turn (generation takes one prompt string) and fit the 400-token prompt budget.
  - These rows are never used for training or dev.
  - TMMLU+ smoke subset (30 MCQ) runs alongside as the knowledge-regression check.
- **Dev:** 50 random rows drawn from the rows not held out for eval, edited or not, before training rows are selected. They are used only for per-epoch validation loss.
- **Train:** all remaining eligible rows when they fit the budget, otherwise sampled down to `trainRows` (default: all available, capped at 500). A share `editedShare` (default 0.5) are edited rows.
  - Rows over 480 tokens (TAIDE chat template) are skipped, never truncated.
  - Repeated prompts are grouped with `input_key`, so a prompt never lands in two splits.
- **Variants:** original and reviewed training/dev files use identical row indices and order. Only answer text differs.
- **Seed:** 42, recorded with row indices and file hashes in a split manifest.
- **Minimums:** at least 100 train rows and 20 eval prompts. Otherwise the job is refused before any GPU use, with a plain reason (e.g. "only 8 rows were edited").
- **Disclosure:** eval prompts are deliberately drawn from edited rows (targeted, not random), and the edited share may be boosted. Both facts appear on the result page.

## 2. Job

A new `comparison` record lives in the quality store. Its files go under `<root>/quality/comparisons/<id>/`, and it runs in a background thread. Steps persist status, timing, GPU-seconds and message.

| # | Step | GPU |
|---|---|---|
| 1 | Prepare: verify version and source fingerprints; write split, train/dev files, eval cases | none |
| 2 | Train original: normal run under `runs/<id>/`, TAIDE 4-bit QLoRA via `run_with_pilot_budget(load_mode="bnb4")` | T4 #1 |
| 3 | Train reviewed: identical settings and seed, in parallel with step 2 | T4 #2 |
| 4 | Evaluate base, original, reviewed: TMMLU+ + held-out prompts, `writing_sample_tokens=128`, two at a time | both |
| 5 | Report | none |

- **GPU assignment:** idle GPUs are listed once and assigned explicitly to parallel steps. With one idle GPU, steps run sequentially. Only one comparison runs at a time.
- **Pre-flight estimate:** from measured T4 speed, about 1.3 s per row-epoch plus model load, plus about 8 s per eval prompt. Refuse to start when:
  - the ledger cannot cover the estimate, or
  - one training run would exceed the 1,800 s per-run cap. In that case, suggest fewer rows or epochs.
- **Restart:** an unfinished comparison resumes at its first incomplete step. Completed runs and evaluations are never repeated. A step interrupted mid-run is marked failed and retried once; its GPU time stays charged.
- **Cancel:** terminates only the job's own workers and keeps finished artifacts.
- **Failure:** the job stops at the failing step with its message and keeps all evidence.

## 3. API

Routes live under `/quality/comparisons`, following existing patterns (`@guarded`, store records, background job).

| Route | Behaviour |
|---|---|
| `POST /estimate` | Body: `versionId`, `trainRows`, `editedShare`, `epochs`, `evalPrompts`. CPU-only dry run that creates nothing. Returns `feasible`, `reasons`, split counts (edited rows, eligible eval prompts, rows skipped for length), GPU estimate (per run, evaluation, total, parallel wall-clock), ledger remaining and per-run cap. |
| `POST /` | Same body. Re-checks feasibility. Returns `202` with the record, `409` while another comparison runs, `422` if infeasible. |
| `GET /` | List: id, dataset name, status, created, headline metrics. |
| `GET /{id}` | Settings, split summary, steps, run IDs, report. |
| `POST /{id}/cancel` | Cancel. |
| `GET /{id}/export` | One JSON file: settings, split manifest, all answers, report. |

- **Validation:**
  - The version must be complete, and both version and source must still match their recorded fingerprints.
  - Bounds: `trainRows` 100–5,000; `epochs` 1–3; `editedShare` 0–1; `evalPrompts` 20–200.
- **Runs:** both are normal `runs/<id>/run.json` records, so they appear in `/runs` and Training runs and are usable as `adapterRunId`. They are tagged `managedBy: <comparisonId>` and `variant`. `POST /runs/{id}/cancel` returns `409` for managed runs and points to the comparison cancel.
- **Existing fix:** `app.py` `run_training` passes `load_mode="bnb4"` for QLoRA quality runs. Today they are refused on GPUs under 24 GB.

## 4. UI

- **Entry points:**
  - the review page, after "New version saved";
  - a "Compare" link per item in Dataset quality's reviewed-versions list;
  - a new **Comparisons** tab beside Dataset quality and Evaluations.
- **Start screen** (`#/quality-comparisons/new/<versionId>`): the four settings with defaults, and a live estimate panel (counts, GPU-seconds, wall-clock, ledger remaining). When infeasible, plain-language reasons are shown and Start is disabled.
- **Progress** (`#/quality-comparisons/<id>`): five-step tracker with status, elapsed time and GPU-seconds; links to both runs; Cancel; polling.
- **Result** (same URL):
  1. headline sentence;
  2. table for Base / Original / Reviewed: term-list rate and OpenCC rate per 1k chars, answers with any hit (n/N), TMMLU+, training loss;
  3. HTML/CSS bar chart of the term-list rate, following the dataviz guidance;
  4. paired counts (original hit & reviewed clean, and the reverse);
  5. 3–5 side-by-side examples with highlighted terms;
  6. "How this was measured" box: edited share, train size, targeted eval selection, sample size and seed, truncated samples, revisions, GPU used;
  7. Export.

## 5. Metric

Each writing answer is inspected with the audit's own policy (`quality_policy.inspect_text`). Self-containing replacements are excluded automatically, as in `approved_policy`.

- **Term-list rate:** customer-term findings (`CUSTOMER_TERM_REASON`) per 1,000 characters.
- **OpenCC rate:** OpenCC editable findings per 1,000 characters.
- Both are reported with the number of answers containing at least one hit.
- TMMLU+ accuracy uses the leading-letter parser.
- All three models answer identical prompts with identical protocol.

## 6. Code layout

- `trainer/comparison.py` (new): split, file writing, metric and report logic, moved out of `taiwanchat_experiment.py`. The experiment CLI becomes a thin wrapper over it.
- `trainer/quality_api.py`: routes and the job runner (threads, GPU assignment, resume).
- `trainer/app.py`: bnb4 admission for QLoRA quality runs, plus `409` on cancelling managed runs.
- `public/quality.js`: Comparisons tab, start, progress and result views.
- `public/quality.css`: result-page styles.

## 7. Testing

- **Unit (`comparison.py`):**
  - split determinism;
  - train/dev/eval disjoint by row and prompt;
  - edited share honoured;
  - identical indices across variants;
  - minimum refusals;
  - separate term and OpenCC counts;
  - no double-counting of self-containing replacements.
- **API (GPU mocked):**
  - steps advance, including the parallel path;
  - resume after restart without retraining;
  - cancel;
  - concurrent comparison returns `409`;
  - managed-run cancel returns `409`;
  - estimate refuses on ledger and per-run cap;
  - fingerprint mismatch is rejected.
- **`app.py`:** QLoRA quality runs request bnb4 admission.
- **Browser:** start, live estimate, progress, result.
- **Acceptance (real, VM):** corrected TaiwanChat confirmation run through the UI. Audit a TaiwanChat subset with the amended term list, bulk-accept, create a version, run the comparison. This replaces the earlier run affected by the 演演算法 bug and fits the remaining ledger (~4,300 s of 9,302 s).

## Out of scope

- Other languages or base models.
- Multi-seed runs and significance testing; the paired counts are shown, but no p-value.
- Comparing more than two variants.
- Deploying adapters.
