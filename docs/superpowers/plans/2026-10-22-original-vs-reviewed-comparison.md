# Original-vs-Reviewed Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** From any reviewed dataset version, one UI action trains two TAIDE QLoRA adapters that differ only in the accepted review edits, evaluates base / original / reviewed, and shows a comparison result page.

**Architecture:** The work is split across four modules:
- `comparison.py`: pure split, estimate, file and report logic. No GPU.
- `comparison_job.py`: runs the steps with injected GPU functions, handles resume, parallelism and cancel.
- `quality_api.py`: routes under `/quality/comparisons`.
- `quality.js`: the views.

Training writes normal run records under `<root>/runs/<id>/` and runs `train.py` through the existing GPU ledger.

**Tech Stack:** Python 3.11, FastAPI, pytest, Transformers tokenizer (TAIDE), vanilla JS UI, Node static server.

**Spec:** `docs/superpowers/specs/2026-10-22-original-vs-reviewed-comparison-design.md`

## Global Constraints

- Base model: `quality_models.TAIDE_ID` at the pinned revision (`get_model_status`); 4-bit QLoRA. No other model is substituted.
- Every GPU step goes through `quality_pilot.run_with_pilot_budget` / `generate_bounded` with `load_mode="bnb4"`.
- Per-step `max_seconds` ≤ 1800. The aggregate ledger is `quality_pilot.BUDGET_LIMIT_SECONDS`.
- Split seed `42`; dev rows `50`; max train row tokens `480`; max eval prompt tokens `400`; writing samples `128` tokens.
- Minimums: `100` train rows, `20` eval prompts.
- Request bounds:
  - `trainRows`: 100–5000, or omitted (meaning all available, capped at 500);
  - `editedShare`: 0–1, default 0.5;
  - `epochs`: 1–3, default 2;
  - `evalPrompts`: 20–200, default 60.
- Hyperparameters match the TaiwanChat run: rank 16, alpha 32, lr 2e-4, batch 1, grad-accum 8, max_sequence_length 512.
- Only one comparison runs at a time. A comparison runs in its own thread, not the 2-worker quality pool.
- Tests run on the VM. Local Windows Python lacks the ML dependencies.
  - Sync changed files to the VM with `tar cf - <files> | ssh gcp-dev "tar xf - -C /home/jupyter/dev-01/cherrie.chen/llm_fine_tune"`.
  - Run from the repo root: `.venv/bin/python -m pytest trainer/<file> -q`.
  - Pre-existing failures to ignore: 3 HF-import tests (`test_dataset_flow`, `test_hf_import` ×2).
- After API changes, restart the VM backend: kill PID from `pgrep -u $USER -f "[u]vicorn app:app --app-dir trainer --host 127.0.0.1 --port 18000"`, then `trainer/start-vm-backend.sh`.
- No comments restating code. Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Source dataset file deleted or moved after review:** respond with a 422 "source dataset is missing; start a new audit", not a 503 storage error.
2. **No idle GPU when a GPU step starts:** the step fails with the `gpu_busy` message from `run_with_pilot_budget`. The comparison is `failed` and resumable by a new comparison. It must not spin.
3. **Every edited row has a multi-turn prompt:** refuse with the reason "no edited row has a single-turn prompt to evaluate".
4. **`prompt`/`completion` versions:** split, files and eval cases work like `messages` versions.
5. **The ledger shrinks mid-job because another evaluation ran:** before each GPU step, if `budget_status(...)["remainingGpuSeconds"]` is below that step's `max_seconds`, the step fails with "GPU ledger has N s left; this step needs M s" instead of starting with a short deadline.

---

### Task 1: Comparison core (`comparison.py`)

**Files:**
- Create: `trainer/comparison.py`
- Modify: `trainer/taiwanchat_experiment.py` (import `leading_letter`, `scoring_policy`, `policy_hits` from `comparison`; delete its local copies; `approved_policy` becomes `scoring_policy`)
- Modify: `trainer/test_taiwanchat_experiment.py` (import from `comparison`)
- Test: `trainer/test_comparison.py`

**Interfaces:**
- Produces:
  - `load_pair(store, version_id) -> tuple[list[dict], list[dict], dict, dict]`
    - Returns (original rows, reviewed rows, audit, version) via `quality_store.iter_rows`.
    - Raises `ValueError` when the version is not `complete`.
    - Raises `ValueError("Source dataset is missing; start a new audit.")` when the source file does not exist.
    - Raises `ValueError` when `fingerprint(source) != version["provenance"]["sourceFingerprint"]` or `fingerprint(export_path) != version["provenance"]["outputFingerprint"]`. The message includes "start a new audit".
    - Raises `ValueError` when row counts differ.
  - `answer_text(row) -> str | None`: final assistant content, or `completion`. `None` for `text` rows.
  - `single_prompt(row) -> str | None`:
    - for messages rows, the user content when `messages[:-1]` is exactly one `user` message;
    - for prompt/completion rows, the `prompt`;
    - otherwise `None`.
  - `plan_split(original, reviewed, token_count, *, train_rows=None, edited_share=0.5, eval_prompts=60, seed=42) -> dict`
    - `token_count(row) -> int` gives full-row tokens; `token_count({"prompt_only": str}) -> int` gives prompt tokens.
    - Returns `{"eval": [int], "dev": [int], "train": [int], "counts": {"rows", "edited", "eligibleEval", "skippedLong", "trainEdited", "meanTrainTokens"}, "reasons": [str], "settings": {...resolved}}`.
  - `estimate(split, mean_train_tokens, epochs, ledger_remaining) -> dict`
    - Returns `{"trainSecondsPerRun", "evalSecondsPerModel", "totalGpuSeconds", "wallClockSeconds", "stepMaxSeconds": {"train", "evaluate"}, "feasible": bool, "reasons": [str], "ledgerRemaining", "perRunCap": 1800}`.
  - `write_files(directory, original, reviewed, split) -> dict[str, str]`
    - Writes `original-train.jsonl`, `reviewed-train.jsonl`, `dev-original.jsonl`, `dev-reviewed.jsonl`, `eval-cases.jsonl`, `split.json`.
    - Returns name → sha256.
    - Eval case: `{"id": "row:<index>", "task": "writing", "input": <single_prompt>, "reference": <reviewed answer>}`.
  - `scoring_policy(policy) -> dict`: the audit policy, with replacements that contain their own source term added to `exclusions`.
  - `policy_hits(text, policy) -> {"terms": [str], "opencc": [str]}`
    - `terms`: editable findings whose `reason == CUSTOMER_TERM_REASON`;
    - `opencc`: other editable findings.
  - `leading_letter(text) -> str | None`: moved unchanged from `taiwanchat_experiment.py`.
  - `build_report(answers_by_model: dict[str, list[dict]], eval_cases: list[dict], mcq_cases: list[dict], policy, losses: dict[str, float | None]) -> dict`
    - Returns `{"models": {name: {"termsPer1k", "openccPer1k", "answersWithTerm", "answersWithOpencc", "writingAnswered", "truncated", "tmmluplusAccuracy", "tmmluplusAnswered", "trainLoss"}}, "paired": {"originalOnly", "reviewedOnly", "both", "neither"}, "examples": [{"caseId", "prompt", "answers": {model: text}, "hits": {model: [str]}}], "headline": str}`.

**Constants** (module-level in `comparison.py`):
- `MIN_TRAIN = 100`, `MIN_EVAL = 20`, `DEV_ROWS = 50`
- `MAX_ROW_TOKENS = 480`, `MAX_PROMPT_TOKENS = 400`, `DEFAULT_TRAIN_CAP = 500`
- `TRAIN_LOAD_SECONDS = 90`, `SECONDS_PER_TOKEN_EPOCH = 0.0065`
- `EVAL_LOAD_SECONDS = 40`, `MCQ_SECONDS = 1.5`, `WRITING_SECONDS = 8`
- `RESERVATION_FACTOR = 1.5`, `PER_RUN_CAP = 1800`

**Split algorithm** (the tests pin it; this order is the decision):
1. `rng = random.Random(seed)`; shuffle indices.
2. Group by `advisor.input_key` of the original row. A group goes wholly to one split.
3. An edited row is one where `answer_text(original[i]) != answer_text(reviewed[i])`.
4. **Eval:** take groups whose first row is edited, has a `single_prompt` and has prompt tokens ≤ 400, up to `eval_prompts`. Use one row per group.
5. **Dev:** the next 50 groups' first rows, any edit status, with full tokens ≤ 480.
6. **Train:** from the remaining rows with tokens ≤ 480, fill `round(cap * edited_share)` edited rows and the rest unedited rows.
   - `cap = train_rows or min(available, 500)`.
   - If one pool runs short, take the remainder from the other.
   - Record the realised `trainEdited`.
   - Count rows over 480 in `skippedLong`.
7. **Reasons:**
   - `len(train) < 100` → reason naming the count;
   - `len(eval) < 20` → reason naming the count;
   - `counts["edited"] == 0` → "the review changed no answers";
   - edited rows exist but none has a single-turn prompt → "no edited row has a single-turn prompt to evaluate".
   - When every `answer_text` is `None` (plain `text` rows), the only reason is "plain text rows have no prompt to evaluate; comparison needs messages or prompt/completion rows".

- [x] **Step 1: Write the failing tests** in `trainer/test_comparison.py`. Use synthetic rows and `token_count` = character length. Assertions:

```python
def test_split_is_deterministic_and_disjoint(): # same inputs twice -> identical dict; eval/dev/train pairwise disjoint; no input_key in two splits
def test_variants_share_indices_and_differ_only_in_answers(tmp_path): # write_files -> original-train vs reviewed-train same prompts/order; answers differ exactly on edited rows
def test_edited_share_is_honoured(): # 300 edited + 700 unedited rows, train_rows=200, edited_share=0.5 -> counts["trainEdited"] == 100
def test_eval_rows_are_edited_single_turn_and_never_trained(): # every eval index is edited, single_prompt not None, not in train/dev
def test_minimums_and_reasons(): # 8 edited rows -> reasons mention "8" and not feasible; multi-turn-only edited rows -> "no edited row has a single-turn prompt to evaluate"
def test_prompt_completion_rows_split_like_messages(): # prompt/completion rows produce eval cases whose input == prompt
def test_plain_text_rows_are_refused_with_reason(): # text-only rows -> reasons == ["plain text rows have no prompt to evaluate; comparison needs messages or prompt/completion rows"]
def test_long_rows_skipped_not_truncated(): # rows over 480 tokens absent from train/dev, counted in skippedLong
def test_estimate_refuses_per_run_cap_and_ledger(): # tokens making trainSecondsPerRun > 1800 -> not feasible, reason mentions 1800; ledger_remaining below total -> reason mentions ledger
def test_policy_hits_separates_terms_and_opencc(): # policy {"terms": {"數據": "資料"}} on "數據與软件" -> terms ["數據"], opencc ["软件"]
def test_scoring_policy_protects_self_containing_replacements(): # {"算法": "演算法"} -> policy_hits("演算法", ...)["terms"] == []
def test_load_pair_rejects_missing_or_changed_source(tmp_path): # deleted source -> ValueError "Source dataset is missing; start a new audit."; modified source -> ValueError containing "start a new audit"
def test_report_counts_and_paired(): # hand-built answers: termsPer1k, answersWithTerm, paired originalOnly/reviewedOnly match hand counts; headline contains both rates
```

- [x] **Step 2: Run and confirm they fail.**
  `.venv/bin/python -m pytest trainer/test_comparison.py -q`. Expected: ImportError / failures.
- [x] **Step 3: Implement `trainer/comparison.py`** per the Interfaces block.
  - `load_pair` uses `store.get("version", id)`, `store.get("audit", version["provenance"]["auditId"])`, `store.source_path(audit["dataset"])`, `store.export_path(id)` and `quality_store.fingerprint`.
- [x] **Step 4: Point `taiwanchat_experiment.py` and its test at the shared functions.**
  Run `.venv/bin/python -m pytest trainer/test_comparison.py trainer/test_taiwanchat_experiment.py -q`. Expected: all pass.
- [x] **Step 5: Commit** `trainer/comparison.py trainer/test_comparison.py trainer/taiwanchat_experiment.py trainer/test_taiwanchat_experiment.py`. Message: "Add shared comparison core: split, estimate, files, metric, report".

### Task 2: GPU plumbing and managed runs

**Files:**
- Modify: `trainer/quality_pilot.py` (add `idle_gpus`; `generate_bounded` gains `gpu=None`)
- Modify: `trainer/app.py:460-480` (`run_training` passes `load_mode="bnb4"` when `record["config"]["parameter_method"] == "qlora"` on the quality path) and `trainer/app.py:887-895` (`cancel_run`)
- Test: `trainer/test_quality_models.py`, `trainer/test_quality_training.py`

**Interfaces:**
- Produces:
  - `quality_pilot.idle_gpus(load_mode="bnb4") -> list[str]`: GPU UUIDs from `nvidia-smi --query-gpu=uuid`, each passing `_select_idle_gpu(uuid, load_mode=load_mode)`.
  - `generate_bounded(..., gpu=None)`: forwards `gpu` to `run_with_pilot_budget`.
  - `POST /runs/{id}/cancel` → `409 {"detail": "This run is managed by comparison <id>; cancel the comparison instead."}` when `run.json` has `managedBy`.

- [x] **Step 1: Write failing tests.**
  - `test_idle_gpus_lists_only_idle_large_enough`: mock `subprocess.check_output` with two GPUs, one with a compute pid. Assert only the idle UUID is returned.
  - `test_generate_bounded_pins_requested_gpu`: monkeypatch `run_with_pilot_budget` to capture kwargs. Assert `gpu == "GPU-b"`.
  - `test_quality_qlora_training_requests_bnb4_admission`: follow the setup in `trainer/test_quality_training.py`. Capture `run_with_pilot_budget` kwargs; assert `load_mode == "bnb4"`.
  - `test_cancel_managed_run_returns_409`: write `run.json` with `managedBy`; POST cancel; assert 409 and the detail contains "comparison".
- [x] **Step 2: Run them and confirm they fail.**
- [x] **Step 3: Implement the three changes.**
- [x] **Step 4: Run** `trainer/test_quality_models.py trainer/test_quality_training.py trainer/test_taide_bnb4.py`. Expected: pass.
- [x] **Step 5: Commit.** Message: "Pin GPUs for bounded generation, admit QLoRA quality runs in 4-bit, protect managed runs".

### Task 3: Comparison job runner (`comparison_job.py`)

**Files:**
- Create: `trainer/comparison_job.py`
- Test: `trainer/test_comparison_job.py`

**Interfaces:**
- Consumes: Task 1 (`load_pair`, `plan_split`, `estimate`, `write_files`, `build_report`, `scoring_policy`) and Task 2 (`idle_gpus`, `generate_bounded(gpu=)`).
- Produces:
  - `STEPS = ("prepare", "train-original", "train-reviewed", "evaluate-base", "evaluate-original", "evaluate-reviewed", "report")`
  - `new_record(version_id, settings) -> dict`: `{"id", "versionId", "settings", "status": "queued", "createdAt", "steps": [{"name", "status": "pending", "startedAt": None, "completedAt": None, "gpuSeconds": 0, "message": None, "runId": None, "attempts": 0}], "split": None, "estimate": None, "report": None}`
  - `run_comparison(store, root, comparison_id, stop: threading.Event, *, gpus=idle_gpus, train=run_with_pilot_budget, generate=generate_bounded, token_count=None) -> None`
    - `token_count` defaults to a cached TAIDE tokenizer counter (`tokenizer.apply_chat_template(..., tokenize=True)` length).
    - Files go in `<root>/quality/comparisons/<id>/`.
  - Managed run record at `<root>/runs/<runId>/run.json`, in app.py's schema plus:
    - `"managedBy": <comparisonId>`, `"variant": "original"|"reviewed"`, `"qualityProvenance": {"comparisonId", "versionId", "variant"}`;
    - `config.dataset = {"source": "upload", "name": <train file>, "extension": "jsonl", "displayName": "<version displayName> · <variant>"}`;
    - `config.test_dataset` = the dev file;
    - `qualityModel = get_model_status(revision=...)`.
    - `metrics` are read from `metrics.json` exactly as app.py does: `eval_loss`, plus `perplexity = round(exp(eval_loss), 3)`.

**Behaviour** (each item is a test):
1. Steps run in order.
   - Training: with ≥2 idle GPUs, train-original and train-reviewed run concurrently on distinct GPUs; with 1, they run sequentially.
   - Evaluate steps: two concurrent, then the third.
2. Before each GPU step, it refuses with the Review Focus #5 message when the ledger is below the step's `max_seconds`. `max_seconds = min(1800, ceil(estimate × RESERVATION_FACTOR))`.
3. **Resume:**
   - steps already `complete` are skipped;
   - a step found `running` is retried once (`attempts` < 2), otherwise marked `failed`;
   - a train step whose run has `adapter/adapter_model.safetensors` and status `complete` is treated as complete.
4. **Stop event:**
   - if the record status is `cancelling` → `cancelled`;
   - otherwise (shutdown) the status stays `running` so startup can resume.
5. A failed step marks the comparison `failed` with that step's message. No later step runs.
6. Completion: `status = complete`; `report` = `build_report(...)` output plus the run IDs.

- [x] **Step 1: Write failing tests** in `trainer/test_comparison_job.py`. Use a real `QualityStore(tmp_path)` with a small audited and versioned dataset (setup pattern: `trainer/test_quality_store.py::setup` plus `accept_customer_terms`), fake `gpus`/`train`/`generate`, and `token_count=len`. Tests:
  - `test_happy_path_parallel_training_on_distinct_gpus`: fake `train` records `(gpu, start, end)`; assert the two intervals overlap and the GPUs differ; status `complete`; report has three models.
  - `test_single_gpu_runs_sequentially`
  - `test_managed_runs_are_normal_run_records`: both `run.json` exist with `managedBy`, `variant`, `status: complete`, `metrics.eval_loss`.
  - `test_resume_skips_completed_and_retries_interrupted_once`
  - `test_cancel_vs_shutdown`: status `cancelling` + stop → `cancelled`; stop alone → still `running`.
  - `test_step_failure_stops_job`: `train` returns `{"status": "unavailable", "message": "gpu_busy ..."}` → `failed`, the evaluate steps are still `pending`.
  - `test_ledger_shortfall_fails_step_before_launch`: monkeypatch `budget_status` low; assert `train` was never called and the message mentions "GPU ledger".
- [x] **Step 2: Run them and confirm they fail.**
- [x] **Step 3: Implement `comparison_job.py`.**
  - Persist the record with `store.put("comparison", record)` after every state change.
  - Run parallel steps with `concurrent.futures.ThreadPoolExecutor(max_workers=2)` inside the job thread.
- [x] **Step 4: Run** `trainer/test_comparison_job.py trainer/test_comparison.py`. Expected: pass.
- [x] **Step 5: Commit.** Message: "Add resumable comparison job runner with parallel T4 steps".

### Task 4: API routes and restart behaviour

**Files:**
- Modify: `trainer/quality_store.py:143-145` (`recover()` leaves `kind == "comparison"` records untouched)
- Modify: `trainer/quality_api.py` (request model, routes, own threads, startup resume, shutdown)
- Test: `trainer/test_quality_api.py`, `trainer/test_quality_store.py`

**Interfaces:**
- Consumes: `comparison_job.new_record`, `run_comparison`, `STEPS`; `comparison.load_pair`, `plan_split`, `estimate`.
- Produces:
  - `ComparisonRequest(StrictRequest)` fields:
    - `versionId: str`
    - `trainRows: int | None = Field(None, ge=100, le=5000)`
    - `editedShare: float = Field(0.5, ge=0, le=1)`
    - `epochs: int = Field(2, ge=1, le=3)`
    - `evalPrompts: int = Field(60, ge=20, le=200)`
  - Routes:
    - `POST /quality/comparisons/estimate` → `{"split": counts+reasons, "estimate": ..., "feasible": bool}`
    - `POST /quality/comparisons` → 202 record | 409 "Another comparison is running" | 422 infeasible (detail joins the reasons)
    - `GET /quality/comparisons` → list (`id`, `versionId`, `displayName`, `status`, `createdAt`, `headline`)
    - `GET /quality/comparisons/{id}`
    - `POST /quality/comparisons/{id}/cancel` → sets status `cancelling`, then the event
    - `GET /quality/comparisons/{id}/export` → JSON attachment `comparison-<id>.json` with the record, split manifest, all answers and the report

- [x] **Step 1: Write failing tests.**
  - `test_recover_leaves_comparisons_resumable` (store): a `running` comparison stays `running` after `recover()`; an audit still becomes `interrupted`.
  - `test_estimate_and_bounds` (api): 422 for `epochs: 4` and for `trainRows: 50`; estimate on a too-small version returns `feasible: false` with reasons.
  - `test_create_conflict_and_cancel`: monkeypatch `comparison_job.run_comparison` with a blocking fake; second POST → 409; cancel → record status `cancelling`, then `cancelled`.
  - `test_startup_resumes_running_comparison`: put a `running` record, build the app, assert the fake `run_comparison` was called with that id.
  - `test_missing_source_is_422_not_503`.
- [x] **Step 2: Run them and confirm they fail.**
- [x] **Step 3: Implement.**
  - Each comparison runs in a `threading.Thread(daemon=True, name="comparison-<id>")` tracked in a dict under the existing `lock`.
  - The startup hook resumes `queued`/`running` comparisons.
  - The shutdown hook sets their events without changing status.
  - The estimate uses the default TAIDE `token_count` from `comparison_job`, monkeypatched in tests.
- [x] **Step 4: Run the full suite** `.venv/bin/python -m pytest trainer -q`. Expected: only the 3 known HF-import failures (since fixed: `trainer/conftest.py` now isolates HF caches, so the suite is fully green).
- [x] **Step 5: Restart the VM backend.** Then `curl -s -X POST localhost:18000/quality/comparisons/estimate -H 'content-type: application/json' -d '{"versionId":"<the ui-term-list-test version id>"}'`. Expected: `feasible: false`, with reasons about the 4-row size.
- [x] **Step 6: Commit.** Message: "Add /quality/comparisons API with resume across restarts".

### Task 5: UI — Comparisons tab, start, progress, result

**Files:**
- Modify: `public/quality.js` (nav link in `qualityHeader`; `qualityComparisons(id)`; start / progress / result renderers; entry links in the versions list and after "New version saved")
- Modify: `public/app.js:573-576` (dispatch `page === "quality-comparisons"`, including sub-route `new/<versionId>`)
- Modify: `index.html:22` (sidebar link `◇ Comparisons`)
- Modify: `public/quality.css` (result table, bars, step tracker, highlights; must work at 700 px and below)

**Interfaces:**
- Consumes: Task 4 routes.
- Produces: routes `#/quality-comparisons`, `#/quality-comparisons/new/<versionId>`, `#/quality-comparisons/<id>`.

Copy fixed by the spec:
- headline from `report.headline`;
- the "How this was measured" box lists:
  - edited share and train size;
  - "Eval prompts were chosen from rows the review changed, not at random";
  - sample size and seed 42;
  - truncated samples;
  - model and revisions;
  - GPU-seconds used.

Highlighting: wrap each `hits[model]` term occurrence in `<mark>` after `qualityEscape`.

- [x] **Step 1: Load the `dataviz` skill** before writing the bar chart. Bars are HTML/CSS (`width: <rate/max*100>%`), with values labelled in text.
- [x] **Step 2: Implement the start view.**
  - Debounce the estimate (400 ms) on input change.
  - Disable Start when `feasible` is false and list the reasons.
  - Show GPU-seconds, wall-clock minutes and ledger remaining.
- [x] **Step 3: Implement the progress view.**
  - Step tracker with status, elapsed time and GPU-seconds.
  - Run links `#/runs/<runId>`; Cancel.
  - Poll every 3 s via `qualityTimer`.
- [x] **Step 4: Implement the result view**: headline, table, bars, paired counts, examples, measured box, export.
- [x] **Step 5: Verify in the browser** (`node server.mjs` already running at 127.0.0.1:4173 with the tunnel).
  - The start screen for the test version shows the refusal reasons with Start disabled.
  - The list view loads.
  - Render the result view by intercepting `GET /api/quality/comparisons/<id>` in-page with a record built from the TaiwanChat numbers (debug only).
  - Check the console has no errors.
  - Check 375 px width.
- [x] **Step 6: Run** `node --check public/quality.js public/app.js`, `node test/test-methods-page.cjs` and `node test/test-auto-evaluator.cjs`.
- [x] **Step 7: Commit.** Message: "Add Comparisons UI: start with live estimate, progress, result page".

### Task 6: Acceptance — corrected TaiwanChat comparison through the UI

**Files:**
- Create: `validation/taiwanchat-subset-2000.jsonl` is not committed. It is runtime data, generated on the VM under `data/datasets/`.
- Modify: `docs/quality-pilot-validation.md` (append a 2026-10-22 section with the real numbers)
- Modify: `docs/quality-pilot-operations.md` (a short "Compare original vs reviewed" usage paragraph; also fix the stale `hackathon/` paths there)

- [x] **Step 1: Generate the subset on the VM.**
  - 2,000 single-turn `messages` rows from `yentinglin/TaiwanChat@d766aa2…`, seed 42, full tokens ≤ 480. Reuse `taiwanchat_experiment` loading.
  - Upload it through the UI's Dataset quality upload.
- [x] **Step 2: In the UI, audit it with the full term list** from `validation/taiwanchat-terms.json` (including 大數據→巨量資料). Then "Accept all term-list suggestions" and create the version.
- [x] **Step 3: Start a comparison with the defaults.**
  - Confirm the estimate is feasible and under the ledger remaining (≈9,302 s).
  - Watch progress to `complete`.
- [x] **Step 4: Check the result page shows all sections.** Export the JSON. Confirm both runs appear on Training runs with `managedBy`.
- [x] **Step 5: Write the numbers and limitations into `docs/quality-pilot-validation.md`.** Commit and push.
