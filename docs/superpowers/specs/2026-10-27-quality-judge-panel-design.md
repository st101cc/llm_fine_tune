# Local judge panel for quality evaluations — design

Status: approved in conversation 2026-10-27 (owner: cherrie.chen). Implementation plan to follow.

## Goal

Term rates and TMMLU+ show whether a review changed *which words* a model uses, and whether knowledge regressed. They do not show whether the free-form answers are actually good. The judge panel grades saved writing answers on two separate questions:

1. **Correctness:** does the answer do what the prompt asked, accurately?
2. **zh-TW language quality:** is it natural Traditional Chinese as written in Taiwan (vocabulary, phrasing, script), not mainland usage or machine-sounding text?

Three local models judge, and their grades are labelled as estimates. They never replace or mix with human judgments.

## Scope and assumptions

- Owner decisions:
  - The panel runs **only when the user starts it**, on a saved quality evaluation or a finished comparison. Comparisons do not run it automatically and cost no extra GPU time.
  - Correctness is decided by majority vote.
  - Language quality is decided by **Breeze**.
  - The judges are the builder's choice, as recommended below.
- Local only. This keeps the quality pipeline's "no hosted judging" promise, and the quality page notice stays true. No DeepEval or Promptfoo.
- Only `writing` answers are judged. `mcq`, `json` and `classification` already have exact scores, which stay authoritative.
- zh-TW / TAIDE pipeline only, like the rest of the quality pages.

## Non-goals and anti-goals

- **Non-goals:**
  - Retraining or regenerating answers. The panel grades what is saved.
  - Statistical significance.
  - More than three judges.
  - Hosted judges.
- **Anti-goals.** It's a failure even if it runs:
  - A model grade is shown or exported as verified accuracy, or overwrites a human `judgment`.
  - A judge failure, invalid output or abstention is silently counted as pass or fail.
  - Anything downloads without an explicit, separate preparation step.
  - The panel runs on the GPU without charging the shared ledger, or exceeds it.

## 1. Judges

| Judge | Hugging Face ID | Role | Why |
|---|---|---|---|
| Breeze | `MediaTek-Research/Breeze-7B-Instruct-v1_0` | correctness vote, **decides language** | Trained for Taiwan Mandarin. |
| Qwen | `Qwen/Qwen2.5-7B-Instruct` | correctness vote, language vote is advisory | Already ForgeTune's default judge (`evaluator.DEFAULT_MODEL`). Strong Chinese. |
| Yi | `01-ai/Yi-1.5-9B-Chat` | correctness vote, language vote is advisory | A different model family, so its errors differ from Qwen's. Strong Chinese reasoning. |

- **TAIDE is excluded** because every compared answer comes from TAIDE, so it would be grading itself (builder's call).
- None of the three is cached on the VM today; only TAIDE is. Preparation is explicit and reuses `quality_models.prepare_model(model_id)` from the CLI (`python trainer/quality_models.py prepare judges`). It pins each judge to an immutable revision, recorded like TAIDE's, and never accepts license terms. These repos are not gated.
- **Admission (builder's call):** `prepare judges` writes a manifest under `quality/judges.json` with each judge's resolved revision. A judge counts as ready when its manifest revision is cached locally. Otherwise it is reported as `missing_weights` on the start screen, and Start is disabled. Nothing is substituted. The panel's evaluator identity includes these revisions.
- **Candidates of unknown origin (builder's call):**
  - Uploaded evaluation answers can come from any model.
  - If a candidate's name contains a judge's model name (for example "qwen", "breeze" or "yi-1.5"), the start screen and result show a self-grading warning for that candidate. All three votes still count, so the majority rule stays fixed.
  - Uploaded answers carry no truncation flag and are treated as complete (`truncated: false`). The result page says so.
- Each judge is loaded in 4-bit NF4 the way `evaluator.LocalJudge` does it: one judge at a time on one GPU, closed and freed before the next.

## 2. What gets graded

**Inputs.** Two entry points share one panel job:
- **A saved quality evaluation** (`/quality/evaluations/{id}`): its frozen `writing` cases (input, optional reference) and every candidate's answers.
- **A finished comparison** (`/quality/comparisons/{id}`): `eval-cases.jsonl` plus `eval/{base,original,reviewed}.json`, writing answers only. The candidates are the three models.

**Prompts to grade (owner decision).** The start screen has a "prompts to grade" setting.
- Default: 20 for a comparison, or all writing cases for an evaluation if fewer than 20.
- Maximum: every writing case in the source.
- Prompts are sampled with seed 42, and every candidate is graded on the same prompts.
- At the default, a comparison costs about 3 × (60 × 12 + 90) ≈ 2,400 GPU-s, which fits the current ledger. Grading all 60 needs the ledger raised first, which is the owner's call.

**Payload per grade.** This is the same untrusted-data framing as `evaluator.SYSTEM`. The JSON carries:
- the question,
- the reference, **only for evaluation sources**,
- the answer,
- `truncated: true/false`.

**No reference for comparisons (owner decision).** A comparison's reference is the reviewed dataset's answer, which is the reviewed model's own training target. Showing it would reward the reviewed model for matching its training text. For comparison sources, judges grade correctness against the prompt alone. The rubric is the writing rubric without its reference sentence, and the evaluator identity records the `reference: none` mode. The result page states this.

Candidate identity is hidden. Answer order across candidates is shuffled per case with a fixed seed.

**Truncated answers.** Comparison writing samples are cut at 128 tokens by design. When `truncated` is true, the rubric tells the judge to grade only the visible text and not to fail for incompleteness (builder's call). This is disclosed on the result page.

**One call per judge per answer** returns:
```json
{"correctness": {"verdict": "pass|fail|abstain", "reason": "..."},
 "language":    {"verdict": "pass|fail|abstain", "reason": "..."}}
```
- **Correctness rubric:** reuse `evaluator.RUBRICS["writing"]`, plus the truncation rule.
- **Language rubric (new):** pass when the text reads as natural Taiwan Traditional Chinese, and fail for mainland vocabulary or phrasing, Simplified characters, or clearly unnatural text. Quotes, code and proper names are not penalised.
- **Parsing:** strict, extending `evaluator.parse_grade` to the two-part shape. Reasons are capped at 600 characters.
- **Grading call:** generation is greedy, with up to 256 new tokens. If the input is over 4,096 tokens, the judge abstains on both grades instead of truncating, as `LocalJudge` does.

## 3. Aggregation

For each answer, each judge's vote is pass, fail, abstain or error. Error means invalid output or a crash.

- **Correctness = majority of the three judges.**
  - pass if at least 2 judges say pass.
  - fail if at least 2 judges say fail.
  - Otherwise `needs_review`. That covers any split involving an abstain or error, such as pass/fail/abstain.
- **Language = Breeze's verdict.**
  - pass or fail as Breeze says.
  - Breeze abstain or error → `needs_review`.
  - Qwen's and Yi's language votes are recorded and shown, but never change the grade.
- **Disagreement:** an answer is flagged when the correctness votes are not unanimous, or when Breeze's language vote differs from both others. Flagged answers are listed for a human to read.
- **Rates** are pass / (pass + fail) per candidate, per dimension. Coverage (graded vs `needs_review`) is always shown beside each rate. If coverage is below 100%, the result says so plainly; no rate is shown without its coverage.
- **Human judgments:**
  - Existing human `judgment` values on evaluation answers stay in their own column and their own metrics. The panel never writes `judgment`.
  - Where both exist, the result shows agreement (panel vs human, correctness only) as a calibration hint.

## 4. Job

The job copies `comparison_job.py`'s shape:
- It is a new `panel` record kind in the quality store, with files under `<root>/quality/panels/<id>/`.
- It runs in a background thread and can be cancelled.
- It resumes at the first unfinished judge after a restart.
- Only one panel or comparison GPU job runs at a time. A panel is refused while a comparison is active, and vice versa.

| # | Step | GPU |
|---|---|---|
| 1 | Prepare: freeze the case and answer set with a fingerprint, shuffle order, and write `items.jsonl` | none |
| 2 | Judge Breeze: all items | 1 GPU |
| 3 | Judge Qwen: all items | 1 GPU |
| 4 | Judge Yi: all items | 1 GPU |
| 5 | Aggregate and report | none |

- **Isolation and ledger:**
  - Each judge step runs as an owned worker subprocess through `quality_pilot.run_with_pilot_budget(load_mode="bnb4")`. That is the same supervisor, GPU lease, deadline and **shared GPU-second ledger** that generation and training use. It never calls `LocalJudge` in-process inside the API server.
  - The worker checkpoints each grade to `grades-<judge>.jsonl`. After a restart or retry, already-graded items are skipped.
- **Parallelism (builder's call):** with two idle T4s, steps 2 and 3 run in parallel and step 4 follows. With one idle GPU, steps run sequentially.
- **Chunking (builder's call):** a judge step is split into as many budgeted runs as needed, each under the 1,800 s per-run cap. The per-run cap is never raised. Each chunk reloads the model and its load time is charged. `grades-<judge>.jsonl` makes chunks resumable.
- **Estimate and admission:**
  - Before Start, a CPU-only estimate counts items (answers × judges) and multiplies by a measured per-grade time plus model load per chunk. The builder measures these once on the T4 and records the constants.
  - The planning figure is **about 12 s per grade** plus about 90 s load per chunk. This is based on comparison 9ed7a2c3, where TAIDE took 6–9 s per 128-token answer, and judge calls have longer inputs.
  - For a 60-prompt × 3-model comparison, that's about 3 × (180 × 12 + 2 × 90) ≈ 7,000 GPU-s.
  - Start is disabled, with a plain reason, if:
    - the ledger's remaining seconds can't cover the estimate, or
    - a judge is not prepared.
  - Raising the ledger (`BUDGET_LIMIT_SECONDS`, now 15,000 with about 4,274 s left) is an owner decision, never automatic.
- **Failure:** a judge step that fails after one retry stops the job. Grades already written are kept, and the report can still be built from the judges that finished. In that case everything is marked incomplete, and no majority is claimed with fewer than three judges.

## 5. API

These routes go under `/quality/panels` in `quality_api.py`, following the comparison routes:

| Route | Behaviour |
|---|---|
| `POST /estimate` | Body: `source` (`{"evaluationId"}` or `{"comparisonId"}`), optional `prompts` (count to grade). CPU-only. Returns `feasible`, `reasons`, item counts per candidate, per-judge readiness, GPU estimate, ledger remaining. |
| `POST /` | Same body. Re-checks feasibility and returns 202 with the panel record. |
| `GET /` | History list. |
| `GET /{id}` | Record with step status and, when complete, the report. |
| `POST /{id}/cancel` | Cancels only owned workers and keeps grades. |
| `GET /{id}/export` | JSON: every judge's raw verdicts and reasons, aggregates, judge revisions, rubric text and version, fingerprints. |

The evaluator identity is a hash of the panel version, judge IDs and revisions, rubrics and the aggregation rule, the way `evaluator.evaluate_results` builds `evaluatorId`. Panel results are comparable only when this identity matches.

## 6. UI

Changes go in `public/quality.js`, in the same style as the comparison pages.

- **Entry points:**
  - A **"Grade with judge panel"** button on a saved evaluation page and on a finished comparison result page.
  - A **Judge panels** link in the quality nav.
- **Start screen:** shows the source, the "prompts to grade" setting with a live re-estimate, item counts, judge readiness, the GPU estimate against ledger remaining, and the truncation note when it applies. Start is disabled with reasons when the run isn't feasible.
- **Progress:** a per-judge step list with graded/total counts, and Cancel.
- **Result:**
  - A table per candidate with correctness rate, language rate, and coverage for each.
  - A clear "estimated by local models, not verified" label.
  - Flagged disagreements, each showing the answer text and every judge's verdict and reason.
  - Human-vs-panel agreement, when human judgments exist.
  - Export.

## 7. Testing

- **Unit (`trainer/test_judge_panel.py`), with fake judges and no GPU:**
  - Aggregation truth table: every pass/fail/abstain/error combination for correctness, and Breeze-decides for language.
  - Parsing of the two-part verdict, including invalid output.
  - Truncation flag carried into the payload.
  - Identity hidden and order shuffled deterministically.
  - Resume skips graded items.
  - Ledger refusal and the per-run cap.
  - Human judgments untouched.
- **API:** the estimate refuses a missing judge, an exhausted ledger, a non-writing-only source and an active comparison. Export shape.
- **Browser:** a `test/test-quality-ui.cjs`-style check of the start, progress and result views with controlled API responses.
- **Acceptance on the VM:** after the owner approves judge preparation (three downloads of about 15–18 GB each), run the panel on comparison 9ed7a2c3 with the default 20 prompts. Record the rates, coverage, disagreements, GPU-seconds and measured per-grade time in `docs/quality-pilot-validation.md`.

## Left to the builder

- Exact language rubric wording (in English with zh-TW examples, or bilingual), keeping the pass/fail criteria above.
- Estimate constants, once measured.
- Module layout. Suggested: `trainer/judge_panel.py` for the payload, parsing, aggregation and job, with the worker entry in the same file or in `quality_pilot.py`.
