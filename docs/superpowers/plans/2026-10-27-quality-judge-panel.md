# Local Judge Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On a saved quality evaluation or a finished comparison, a user-started job has three local judges (Breeze, Qwen, Yi) grade each sampled writing answer for correctness and zh-TW language quality. It aggregates the votes (majority for correctness, Breeze decides language) and shows estimated rates with coverage, disagreements and an export.

**Architecture:** The work is split across these modules:
- `judge_panel.py`: pure logic. Judges, rubrics, item loading from both sources, sampling, payload, parsing, aggregation, estimate, identity, report. No GPU.
- `panel_job.py`: runs the steps with an injected judge runner, and handles chunking, resume, two-GPU parallelism and cancel. It mirrors `comparison_job.py`.
- `quality_pilot.py`: gains a `judge` worker and `judge_bounded(...)`, the budgeted subprocess wrapper, mirroring `generate_bounded`.
- `quality_models.py`: gains judge preparation and readiness.
- `quality_api.py`: gains `/quality/panels` routes.
- `quality.js`: gains the views.

**Tech Stack:** Python 3.11, FastAPI, pytest, Transformers + bitsandbytes (4-bit NF4), vanilla JS, Playwright browser checks.

**Spec:** `docs/superpowers/specs/2026-10-27-quality-judge-panel-design.md`

## Global Constraints

- **Judges**, in this order, exactly these IDs:
  - `breeze`: `MediaTek-Research/Breeze-7B-Instruct-v1_0`
  - `qwen`: `Qwen/Qwen2.5-7B-Instruct`
  - `yi`: `01-ai/Yi-1.5-9B-Chat`
  - TAIDE is never a judge, and no model is substituted.
- **Local only:** no hosted judging and no DeepEval/Promptfoo. Nothing downloads outside the explicit `prepare judges` CLI.
- **Scope:** only `task == "writing"` answers are graded. Exact scores for other tasks are untouched.
- **Human judgments:** the panel never writes or changes an answer's `judgment`. Panel results live only in the `panel` record and its files.
- **Labelling:** model grades are labelled "estimated by local models, not verified" in the UI and export.
- **Rates:** pass / (pass + fail). No rate is shown or exported without its coverage.
- **Correctness:** pass if at least 2 of 3 judges say pass, fail if at least 2 say fail, otherwise `needs_review`. Abstain and error never count as pass or fail.
- **Language:** Breeze's verdict. Breeze abstain or error → `needs_review`. Qwen's and Yi's language votes are recorded but never change the grade.
- **Comparison sources:** no reference is sent to judges (`referenceMode: "none"`). Evaluation sources send the case reference when present (`referenceMode: "case"`).
- **Truncation:** comparison answers carry `truncated` from the saved answer. Uploaded evaluation answers are treated as `truncated: false`, and the result page says so. When `truncated` is true, the rubric grades only the visible text and doesn't fail for incompleteness.
- **Sampling:** default 20 prompts. Maximum: every writing case in the source. Seed `42`. The same prompts are used for every candidate. Answer order is shuffled per case with seed `42`, and candidate identity is hidden.
- **Judge calls:**
  - Model loaded in 4-bit NF4, one judge per GPU at a time.
  - Greedy decoding, `max_new_tokens=256`.
  - An input over `4096` tokens gives abstain on both grades; nothing is truncated.
  - Reasons are capped at `600` characters.
- **GPU budget:**
  - Every judge run goes through `quality_pilot.run_with_pilot_budget(load_mode="bnb4")` and the shared ledger.
  - Per-run `max_seconds` ≤ `1800`; this cap is never raised.
  - Estimate constants: `SECONDS_PER_GRADE = 12`, `LOAD_SECONDS = 90` per chunk.
- **One GPU job at a time:** only one panel or comparison may be active. Either is refused with 409 while the other is active.
- **Identity:** `PANEL_VERSION = "judge-panel-v1"`. `evaluatorId` = first 16 hex characters of sha256 over `[PANEL_VERSION, judges with revisions, rubrics, aggregation rule, referenceMode]`.
- **Testing on the VM:** tests run on the VM; Windows has no `.venv`.
  - Sync: `tar cf - <files> | ssh gcp-dev "tar xf - -C /home/jupyter/dev-01/cherrie.chen/llm_fine_tune"`
  - Run: `.venv/bin/python -m pytest trainer/<file> -q` from the repo root.
  - The full suite is green (207 passed) before this work starts; keep it green.
- **After API changes,** restart the VM backend. Kill the PID from `pgrep -u $USER -f "[u]vicorn app:app --app-dir trainer --host 127.0.0.1 --port 18000"`, then run `trainer/start-vm-backend.sh`.
- **Style:** no comments that restate code. Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A source with zero writing answers** (an MCQ-only evaluation, or TMMLU+ only): the estimate returns `feasible: false` with the reason "this source has no writing answers to grade". Pinned in Task 1.
2. **A judge returning prose, Markdown or half-JSON:** that vote is `error` for that item, not a crashed step, and the item's correctness falls to `needs_review` if fewer than 2 agreeing votes remain. Pinned in Task 1 (parse) and Task 3 (worker keeps going).
3. **A comparison that isn't `complete`, or is missing `eval/*.json`:** refused with 422 "comparison has no saved evaluation answers", not a 500. Pinned in Task 1.
4. **A backend restart mid-judge:** the panel resumes, already-graded items are skipped, and finished chunks are not re-run or re-charged. Pinned in Task 4.
5. **Starting a comparison while a panel runs, or the reverse:** 409. Pinned in Task 5.

---

### Task 1: Panel core (`judge_panel.py`)

**Files:**
- Create: `trainer/judge_panel.py`
- Test: `trainer/test_judge_panel.py`

**Interfaces:**
- Consumes: `evaluator.RUBRICS["writing"]`, `evaluator.parse_grade` (pattern only), `quality_store.QualityStore.get/directory`, `quality_store.digest`, `quality_store.identifier`.
- Produces:
  - `JUDGES: tuple[dict]`: `({"name": "breeze", "modelId": ...}, {"name": "qwen", ...}, {"name": "yi", ...})`.
  - `PANEL_VERSION`, `SECONDS_PER_GRADE`, `LOAD_SECONDS`, `MAX_RUN_SECONDS = 1800`, `DEFAULT_PROMPTS = 20`, `SEED = 42`.
  - `SYSTEM: str`, the panel system prompt with the two-part JSON contract and the untrusted-data framing copied from `evaluator.SYSTEM`.
  - `CORRECTNESS_RUBRIC(reference_mode) -> str` and `LANGUAGE_RUBRIC: str`.
  - `load_source(store, source: dict) -> dict` returns `{"kind", "id", "referenceMode", "cases": [{"id","input","reference"|None}], "answers": [{"caseId","candidate","output","truncated": bool,"human": bool|None}]}`. `human` is the evaluation row's `correct` for writing rows (a human pass or fail). It is always `None` for comparisons.
    - For `{"evaluationId"}`: cases come from `record["cases"]` with `task == "writing"`; answers from `record["result"]["candidates"][c]["results"]` (writing rows only). `truncated` is looked up from `record.get("generation", {}).get("answers", [])` by `(caseId, candidate)`, default `False`.
    - For `{"comparisonId"}`: cases come from `<dir>/eval-cases.jsonl` with `reference` dropped. Answers come from `<dir>/eval/{base,original,reviewed}.json`, with `candidate` set to the file stem, writing case IDs only, and `truncated` from the answer.
    - Raises `ValueError` for a missing, incomplete or answer-less source.
  - `sample_items(source: dict, prompts: int | None) -> list[dict]` returns items `{"itemId": f"{caseId}|{candidate}", "caseId", "candidate", "question", "reference", "answer", "truncated"}`. Prompts are sampled with `random.Random(SEED)`. The order is shuffled per case with the same RNG, so candidates are interleaved, and the result is deterministic.
  - `payload(item: dict, reference_mode: str) -> str` returns JSON with `question`, `answer`, `truncated`, and `reference` only when `reference_mode == "case"`. It never includes `candidate`.
  - `parse_panel_grade(raw: str) -> dict` returns `{"correctness": {"verdict","reason"}, "language": {"verdict","reason"}}`. It strips code fences like `parse_grade`, raises `ValueError` on any other shape, and caps reasons at 600.
  - `aggregate(votes: dict[str, dict]) -> dict`. `votes` is keyed by judge name, and each value is a parsed grade or `{"error": str}`. Returns `{"correctness": "pass|fail|needs_review", "language": ..., "flagged": bool, "votes": votes}`.
  - `estimate(items: int, remaining_gpu_seconds: float, judges_ready: dict[str, bool]) -> dict` returns `{"feasible", "reasons", "gradesPerJudge", "chunksPerJudge", "gpuSeconds", "ledgerRemaining"}`.
  - `chunk_size() -> int` returns `floor((MAX_RUN_SECONDS - LOAD_SECONDS) / SECONDS_PER_GRADE)`.
  - `evaluator_id(revisions: dict[str, str], reference_mode: str) -> str`.
  - `self_grading_warnings(candidates: list[str]) -> dict[str, str]` maps a candidate to its judge name when the lowercased candidate contains `"breeze"`, `"qwen"` or `"yi-1.5"`.
  - `build_report(items, grades_by_judge: dict[str, dict[itemId, vote]], *, complete_judges: set[str], human: dict[(caseId, candidate), bool] | None) -> dict`. Per candidate it returns `{"correctness": {"pass","fail","needsReview","rate","coverage"}, "language": {...}}`, plus `"flagged": [items with votes]`, `"incomplete": bool` (true unless all three judges are complete; no majority is claimed when incomplete), `"humanAgreement": {"compared","agreed"} | None` and `"selfGrading"`.

- [ ] **Step 1: Write failing tests** in `trainer/test_judge_panel.py`. Use a `QualityStore(tmp_path)` with hand-written `evaluation` and `comparison` records and files.
  - `test_aggregate_correctness_truth_table`, parametrized over all 4³ combinations of pass/fail/abstain/error. Expect pass iff the pass count ≥ 2, fail iff the fail count ≥ 2, otherwise `needs_review`.
  - `test_language_is_breezes_verdict`:
    - Breeze `fail` with qwen and yi `pass` → `fail`, flagged.
    - Breeze `abstain` → `needs_review`.
    - Breeze `{"error"}` → `needs_review`.
  - `test_flagged_when_correctness_not_unanimous`.
  - `test_parse_panel_grade_accepts_fenced_json_and_caps_reason`: a reason of 700 characters comes back with length 600.
  - `test_parse_panel_grade_rejects_prose_and_partial`: `"這個回答很好"`, missing `language`, verdict `"maybe"` → `ValueError`.
  - `test_payload_hides_candidate_and_reference_for_comparison`: `"candidate"` is not in the JSON, `"reference"` is absent for mode `none` and present for mode `case`, and `truncated` is carried through.
  - `test_comparison_source_drops_reference_and_keeps_truncated`.
  - `test_evaluation_source_uses_writing_only_and_defaults_truncated_false`: a mixed writing+mcq evaluation → only writing items. A generated answer with `truncated: True` in `record["generation"]["answers"]` → `True`.
  - `test_source_without_writing_answers_is_refused`: `ValueError` with the message "this source has no writing answers to grade" (Review Focus 1).
  - `test_incomplete_comparison_is_refused`: status `running`, or no `eval/` → `ValueError("comparison has no saved evaluation answers")` (Review Focus 3).
  - `test_sample_items_is_deterministic_and_paired`: 60 cases × 3 candidates with `prompts=20` → 60 items, 20 distinct case IDs, each case present for all 3 candidates, and two calls are equal.
  - `test_estimate_default_comparison`: `estimate(60, 4274, all ready)` → `chunksPerJudge == 1`, `gpuSeconds == 3 * (60 * 12 + 90)`, feasible.
  - `test_estimate_refuses_missing_judge_and_short_ledger`: reasons include `"qwen: missing_weights"` and `"GPU ledger has 100 s left; this panel needs"`.
  - `test_estimate_chunks_large_selection`: 180 items → `chunksPerJudge == ceil(180 / chunk_size())`, and the GPU seconds include one `LOAD_SECONDS` per chunk.
  - `test_report_rates_coverage_and_incomplete`:
    - 2 candidates with mixed votes → the hand-computed rates and coverage.
    - With `complete_judges={"breeze","qwen"}` → `incomplete: True`.
  - `test_self_grading_warning`: `"Qwen2.5-7B@base"` → `{"Qwen2.5-7B@base": "qwen"}`. `"taide@base"` → no warning.
  - `test_evaluator_id_changes_with_reference_mode_and_revision`.

- [ ] **Step 2: Run and confirm they fail.** Run `.venv/bin/python -m pytest trainer/test_judge_panel.py -q`. Expected: collection error or failures, since the module is missing.

- [ ] **Step 3: Implement `trainer/judge_panel.py`** with the interfaces above.
  - `LANGUAGE_RUBRIC` text: pass when the text reads as natural Taiwan Traditional Chinese. Fail for mainland vocabulary or phrasing, Simplified characters, or clearly unnatural text. Never penalise quotes, code or proper names.
  - The truncation sentence is appended to both rubrics inside the payload's instruction when `truncated` is true.
  - The exact wording, including zh-TW examples, is the implementer's choice within those criteria.

- [ ] **Step 4: Run** `.venv/bin/python -m pytest trainer/test_judge_panel.py -q`. Expected: all pass.

- [ ] **Step 5: Commit** `trainer/judge_panel.py trainer/test_judge_panel.py`. Message: "Add judge panel core: sources, sampling, parsing, aggregation, estimate".

---

### Task 2: Judge preparation and readiness

**Files:**
- Modify: `trainer/quality_models.py`: add functions, and add a `prepare judges` choice in `main()`.
- Test: `trainer/test_quality_models.py`

**Interfaces:**
- Consumes: `judge_panel.JUDGES`, `quality_models.prepare_model(model_id) -> {"modelId","revision","path",...}`, `get_model_status(model_id, revision=...)`.
- Produces:
  - `prepare_judges(runtime_root: Path) -> dict` calls `prepare_model` for each judge in order. It writes `<runtime_root>/quality/judges.json` = `{"judges": {name: {"modelId","revision"}}, "preparedAt"}` atomically (tmp + replace), merging with the existing file. It is a network action used only by the CLI.
  - `judge_status(runtime_root: Path) -> dict[str, dict]` returns, per judge name, `{"modelId","revision","status","path"}`. `status` is `"ready"` only when the manifest has a revision and `get_model_status(modelId, revision=rev)["status"] == "ready"`. Otherwise it is `"missing_weights"`. This is offline only.

- [ ] **Step 1: Write failing tests.**
  - `test_judge_status_without_manifest_is_missing`: all three report `missing_weights`, and `get_model_status` is never called with `revision=None`.
  - `test_prepare_judges_writes_manifest_then_ready`: monkeypatch `prepare_model` to return fixed SHAs and `get_model_status` to return ready for those SHAs → the manifest has three entries and all three report `ready`.
  - `test_judge_status_missing_when_cached_revision_differs`.

- [ ] **Step 2: Run them and confirm they fail.**

- [ ] **Step 3: Implement** both functions. Extend `main()` so that `python trainer/quality_models.py prepare judges` prints the manifest. Running `prepare` without `judges` keeps its current behaviour.

- [ ] **Step 4: Run** `trainer/test_quality_models.py`. Expected: pass.

- [ ] **Step 5: Commit.** Message: "Add explicit judge preparation and offline readiness".

---

### Task 3: Judge worker and budgeted runner

**Files:**
- Modify: `trainer/evaluator.py`: `LocalJudge.__init__(self, model_id, *, system=SYSTEM, revision=None)`. It passes `revision` to both `from_pretrained` calls and uses `system` in place of the module `SYSTEM`. Existing callers are unchanged.
- Modify: `trainer/quality_pilot.py`: add the `"judge"` worker choice and `judge_bounded`.
- Modify: `trainer/judge_panel.py`: add `grade_items`.
- Test: `trainer/test_judge_worker.py`

**Interfaces:**
- Consumes: `judge_panel.SYSTEM`, `payload`, `parse_panel_grade`, `quality_models.judge_status`; `run_with_pilot_budget`.
- Produces:
  - `grade_items(items: list[dict], call: Callable[[str], str], out_path: Path, reference_mode: str, cancelled=lambda: False) -> int` lives in `judge_panel.py`. It skips item IDs already present in `out_path` (JSONL, one `{"itemId", "vote"}` per line). For each remaining item, it calls `call(payload(...))` and parses the result. On `ValueError` or `json.JSONDecodeError` the vote is `{"error": "invalid_output"}`; on any other exception from `call` it is `{"error": type(exc).__name__}`. Each line is appended and flushed immediately. Returns the number graded in this call.
  - The `_worker` phase `"judge"` reads `request.json` = `{"judge", "modelId", "revision", "items", "referenceMode", "gradesPath"}` and builds `LocalJudge(modelId, system=judge_panel.SYSTEM, revision=revision)`. It runs `grade_items` with the worker deadline already armed, then calls `close()`. Add `"judge"` to the `--worker` choices.
  - `judge_bounded(items, *, runtime_root, judge: str, grades_path: Path, reference_mode: str, cancelled, max_seconds: float, gpu=None) -> dict`.
    - It validates `max_seconds` in (0, 1800], and checks that `judge_status(runtime_root)[judge]["status"] == "ready"`; otherwise it returns `{"status": "unavailable", "reason": "missing_weights"}`.
    - It writes `request.json` under `_runtime_directory(runtime_root) / "judging" / <uuid>` and calls `run_with_pilot_budget(..., load_mode="bnb4")`.
    - It returns `{"status", "graded": count of lines in grades_path, "run", "budget"}`.

- [ ] **Step 1: Write failing tests** in `trainer/test_judge_worker.py`.
  - `test_grade_items_appends_and_resumes`:
    - First pass with 3 items, a fake `call` that returns valid JSON → 3 lines.
    - Second pass with 5 items → returns 2, the file has 5 lines, and the fake was called 5 times in total.
  - `test_grade_items_records_invalid_output_and_continues` (Review Focus 2): the fake returns prose for item 2 → line 2 is `{"error": "invalid_output"}` and item 3 is still graded.
  - `test_grade_items_stops_when_cancelled`.
  - `test_judge_bounded_refuses_unprepared_judge`: no manifest → `unavailable/missing_weights`, and no `run_with_pilot_budget` call (monkeypatched to raise if called).
  - `test_judge_bounded_rejects_over_cap`: `max_seconds=1801` → `ValueError`.
  - `test_local_judge_accepts_system_and_revision`: construct with kwargs and check the attributes. No model load.

- [ ] **Step 2: Run them and confirm they fail.**

- [ ] **Step 3: Implement** the three changes.

- [ ] **Step 4: Run** `trainer/test_judge_worker.py trainer/test_auto_evaluator.py trainer/test_judge_panel.py`. Expected: pass. Existing `LocalJudge` users are unaffected.

- [ ] **Step 5: Commit.** Message: "Add budgeted judge worker with resumable grade files".

---

### Task 4: Panel job runner (`panel_job.py`)

**Files:**
- Create: `trainer/panel_job.py`
- Modify: `trainer/quality_store.py:143`: recovery also skips `kind == 'panel'`, so the API can resume it, like `comparison`.
- Test: `trainer/test_panel_job.py`

**Interfaces:**
- Consumes: Task 1 (`load_source`, `sample_items`, `estimate`, `chunk_size`, `build_report`, `evaluator_id`, `self_grading_warnings`), Task 3 (`judge_bounded`), `quality_pilot.idle_gpus`, `budget_status`.
- Produces:
  - `STEPS = ("prepare", "judge-breeze", "judge-qwen", "judge-yi", "report")`.
  - `PHASES = (("prepare",), ("judge-breeze", "judge-qwen"), ("judge-yi",), ("report",))`.
  - `new_record(source: dict, prompts: int, display_name: str) -> dict`, with status `queued` and steps that persist `status`, `startedAt`, `finishedAt`, `gpuSeconds`, `graded`, `total` and `message`.
  - `run_panel(store, root, panel_id, stop: threading.Event, *, gpus=idle_gpus, judge=judge_bounded) -> None`.
    - Files go under `<root>/quality/panels/<id>/`: `items.jsonl`, `grades-<judge>.jsonl`, `report.json`.
    - Prepare freezes the items, with their `human` value, and records the `sourceFingerprint` (`digest` of the items), the `evaluatorId` with judge revisions, `referenceMode` and `selfGrading`. The report step passes `human={(caseId, candidate): human}` for the non-`None` entries to `build_report`.
    - Each judge step loops over chunks of `chunk_size()` ungraded items. Each chunk is a `judge(...)` call with `max_seconds = min(1800, LOAD_SECONDS + len(chunk) * SECONDS_PER_GRADE)`, and the step's `gpuSeconds` accumulates.
    - Before each chunk, if `budget_status(root)["remainingGpuSeconds"]` is below that chunk's `max_seconds`, the step fails with "GPU ledger has N s left; this chunk needs M s".
    - With two idle GPUs a phase's steps run in parallel threads with explicit `gpu=`. With one, they run sequentially.
    - A chunk that returns a non-complete status is retried once, then the step fails and the job stops. The report is still built from the complete judges, with `incomplete: true`, and the status is `failed`.
    - On cancel, `stop` is set: owned workers stop through `cancelled=stop.is_set`, and the status becomes `cancelled`.
    - `save()` uses the same `BEGIN IMMEDIATE` and never overwrites `cancelling` with `running`, as in `comparison_job.run_comparison`.

- [ ] **Step 1: Write failing tests** in `trainer/test_panel_job.py`. Use a real `QualityStore(tmp_path)`, a hand-written complete comparison with 3 × 25 writing answers, a fake `gpus`, and a fake `judge` that writes votes into `grades_path` and returns `complete`.
  - `test_panel_runs_all_judges_and_reports`: status `complete`, three grade files of 60 lines, and a report with three candidates.
  - `test_breeze_and_qwen_run_in_parallel_on_two_gpus`: the fake records `gpu` and overlapping intervals.
  - `test_large_selection_is_chunked`: `prompts=25` with `chunk_size` monkeypatched to 30 → 3 calls per judge, and the sum of `max_seconds` matches.
  - `test_resume_skips_graded_items_and_finished_steps` (Review Focus 4):
    - First run: the fake judge writes all 60 Breeze lines. For Qwen it writes 10 lines, sets `stop`, and returns `{"status": "cancelled"}`.
    - Set the record back to `running`, then call `run_panel` again with a fresh event → the fake is never called for Breeze, and Qwen receives only the remaining 50 items.
  - `test_failed_judge_keeps_partial_report`: Yi returns `unavailable` twice → status `failed`, and the report has `incomplete: True` with Breeze and Qwen grades kept.
  - `test_ledger_short_before_chunk_fails_step`.
  - `test_cancel_marks_cancelled_and_keeps_grades`.
  - `test_store_recover_leaves_panel_running` (in `quality_store`).

- [ ] **Step 2: Run them and confirm they fail.**

- [ ] **Step 3: Implement `panel_job.py`** and the one-line `quality_store` change.

- [ ] **Step 4: Run** `trainer/test_panel_job.py trainer/test_quality_store.py trainer/test_comparison_job.py`. Expected: pass.

- [ ] **Step 5: Commit.** Message: "Add resumable judge panel job with chunking and two-GPU steps".

---

### Task 5: API routes and restart behaviour

**Files:**
- Modify: `trainer/quality_api.py`
- Test: `trainer/test_quality_api.py`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces:
  - `class PanelRequest(StrictRequest)`: `source: dict` (exactly one of `evaluationId` or `comparisonId`) and `prompts: int | None` (≥ 1).
  - **Routes:**
    - `POST /quality/panels/estimate` → `{"feasible","reasons","items","candidates","judges": judge_status(...),"estimate","selfGrading","referenceMode","truncatedUnknown": bool}`. `truncatedUnknown` is true for uploaded evaluation answers.
    - `POST /quality/panels` returns 202 and the record. It re-checks feasibility, and returns 409 when a panel or comparison is active.
    - `GET /quality/panels` returns the list, each entry as `{id, displayName, status, createdAt, source}`.
    - `GET /quality/panels/{id}`.
    - `POST /quality/panels/{id}/cancel`.
    - `GET /quality/panels/{id}/export` streams JSON: `{"panel": record minus report, "items", "grades": {judge: [...]}, "report", "label": "estimated by local models, not verified"}`.
  - `POST /quality/comparisons` additionally returns 409 while a panel is active.
  - A `launch_panel(target, panel_id)` thread mirrors `launch_comparison`. The startup `recover()` relaunches panels in `queued`, `running` or `cancelling`. `shutdown()` sets their events and joins them.
  - `ValueError` from `load_source` maps to 422 through the existing `guarded`.

- [ ] **Step 1: Write failing tests** in `trainer/test_quality_api.py`. Use `TestClient`, monkeypatch `panel_job.run_panel` to a no-op, and monkeypatch `judge_status` to all-ready.
  - `test_panel_estimate_for_comparison_defaults_to_20_prompts`: `items == 60`, `referenceMode == "none"`.
  - `test_panel_estimate_refuses_missing_judge`: `feasible: false`, and the reason names the judge.
  - `test_panel_estimate_rejects_unknown_or_double_source`: 422.
  - `test_panel_estimate_for_incomplete_comparison_is_422` (Review Focus 3).
  - `test_panel_and_comparison_are_mutually_exclusive` (Review Focus 5): an active panel → `POST /quality/comparisons` 409, and an active comparison → `POST /quality/panels` 409.
  - `test_panel_export_shape_and_label`.
  - `test_panel_never_changes_evaluation_judgments`: after a panel record completes (stubbed), `GET /quality/evaluations/{id}` is byte-identical to its state before the panel.
  - `test_startup_relaunches_running_panel`.

- [ ] **Step 2: Run them and confirm they fail.**

- [ ] **Step 3: Implement.**

- [ ] **Step 4: Run the full suite:** `.venv/bin/python -m pytest trainer -q`. Expected: all pass.

- [ ] **Step 5: Sync, restart the VM backend, and check the live estimate.** Run `curl -s -X POST localhost:18000/quality/panels/estimate -H 'content-type: application/json' -d '{"source":{"comparisonId":"<9ed7a2c3 full id>"}}'`. Expected: `feasible: false`, with three `missing_weights` reasons, because the judges are not prepared yet.

- [ ] **Step 6: Commit.** Message: "Add /quality/panels API with resume and comparison exclusion".

---

### Task 6: UI — start, progress, result

**Files:**
- Modify: `public/quality.js`, `public/styles.css` (only if a new class is needed; reuse the comparison classes first).
- Test: `test/test-quality-panel.cjs`, in the style of `test/test-quality-ui.cjs` with mocked `/api/**` routes.

**Interfaces:**
- Consumes: the Task 5 routes.
- Produces:
  - **Routes:**
    - `#/quality-panels` is the history list.
    - `#/quality-panels/new/evaluation/<id>` and `#/quality-panels/new/comparison/<id>` are the start screens.
    - `#/quality-panels/<id>` shows progress, then the result.
  - A **"Judge panels"** nav link next to "Comparisons".
  - A **"Grade with judge panel"** button on the saved evaluation page and on a complete comparison result page.

- [ ] **Step 1: Load the `dataviz` skill** before drawing the rate bars. Bars are HTML/CSS, with values labelled in text, the same as the comparison page.

- [ ] **Step 2: Implement the start view.** It shows:
  - The source name.
  - A "Prompts to grade" number input (min 1, max the writing case count, default 20) that re-estimates on change, debounced, as on the comparison start screen.
  - Item counts.
  - Readiness per judge, showing "not prepared: run `python trainer/quality_models.py prepare judges` on the backend".
  - GPU seconds vs ledger remaining.
  - Notes, each shown only when it applies:
    - Truncation: "Comparison answers were cut at 128 tokens; judges grade only what is visible."
    - Unknown truncation: "Uploaded answers are treated as complete."
    - Reference: "No reference shown to judges for comparisons."
    - Self-grading warnings.
  - Start is disabled when the run isn't feasible, with the reasons listed.

- [ ] **Step 3: Implement the progress view.** Each judge step shows its status and graded/total. There is a Cancel button, and the view polls like the comparison progress view.

- [ ] **Step 4: Implement the result view.**
  - A headline label: "Estimated by local models, not verified".
  - A table per candidate: correctness rate, language rate, and the coverage of each ("18 of 20 graded").
  - Bars.
  - An "Incomplete: fewer than three judges finished" banner when `incomplete`.
  - Flagged disagreements, each showing the answer text and all six votes with reasons.
  - Human agreement, when present.
  - Export.

- [ ] **Step 5: Write `test/test-quality-panel.cjs`.**
  - Start screen: Start is disabled with mocked missing judges and enabled with all ready. Changing prompts to 40 triggers a re-estimate with `prompts: 40`.
  - Result: the "not verified" label is visible, coverage text is shown, and a flagged item shows Breeze's language reason.
  - No page errors.

- [ ] **Step 6: Run** `node --check public/quality.js` and `node test/test-quality-panel.cjs`, with the server at 127.0.0.1:4173 and `PLAYWRIGHT_MODULE` set if needed. Also run `node test/test-quality-ui.cjs`. Expected: pass.

- [ ] **Step 7: Verify in the browser** against the restarted VM backend. The comparison 9ed7a2c3 page shows the button, and the start screen shows three "not prepared" judges.

- [ ] **Step 8: Commit.** Message: "Add judge panel UI: start with live estimate, progress, result". Include a screenshot of the result view (from mocked data) in the PR.

---

### Task 7: Acceptance on the VM (owner-gated)

This task waits on two explicit approvals from the owner in chat: downloading the three judges (roughly 15–18 GB each; check `df -h` on the VM first), and spending about 2,400 GPU-s of the ledger.

- [ ] **Step 1: Check disk space,** then run `.venv/bin/python trainer/quality_models.py prepare judges` on the VM. Expected: `quality/judges.json` lists three revisions.
- [ ] **Step 2: Start a panel** on comparison 9ed7a2c3 with 20 prompts from the UI. Expected: Breeze and Qwen run in parallel, then Yi, ending `complete`.
- [ ] **Step 3: Record results.** Note the measured per-grade seconds and load seconds. If they differ from 12 s and 90 s by more than 25%, update `SECONDS_PER_GRADE` and `LOAD_SECONDS` in `judge_panel.py`, and update the matching test expectation.
- [ ] **Step 4: Write up the run** in `docs/quality-pilot-validation.md`: rates, coverage, flagged count, GPU-seconds against the estimate, ledger remaining, and limitations (20 prompts, truncated answers, no reference, estimates only). Commit and push.
