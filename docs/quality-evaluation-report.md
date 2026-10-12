# Traditional Chinese quality sidecar verification

Date: 2026-10-05. Branch: `codex/traditional-chinese-quality`. No commits, live model downloads, hosted inference, training, or real GPU execution were performed by this sidecar task.

## Owned files

- `hackathon/trainer/quality_evaluation.py`
- `hackathon/trainer/quality_models.py`
- `hackathon/trainer/quality_pilot.py`
- `hackathon/trainer/test_quality_evaluation.py`
- `hackathon/trainer/test_quality_models.py`
- This report.

Existing changes and the primary agent's policy/store/routes/UI/training integration remain outside sidecar ownership. Browser/API tests below exercise that concurrent integration but do not claim sidecar authorship of it.

## Integration contracts

Imports use the trainer directory on the Python import path, matching existing trainer modules.

```python
from quality_evaluation import evaluate
from quality_models import TAIDE_ID, get_model_status, generate_answers, prepare_benchmark, load_benchmark
from quality_pilot import generate_bounded, run_with_pilot_budget, budget_status

evaluate(cases, answers, policy, *, cancelled=lambda: False) -> dict
get_model_status(model_id=TAIDE_ID, *, revision=None) -> dict
generate_answers(cases, model_id=TAIDE_ID, adapter_path=None,
                 cancelled=lambda: False, *, revision=None) -> list[dict]
generate_bounded(cases, *, runtime_root, model_id=TAIDE_ID, adapter_path=None,
                 cancelled=lambda: False, max_seconds=1800) -> dict
run_with_pilot_budget(root, commands, *, gpu=None, cancelled=lambda: False,
                      max_seconds=1800, log_dir=None) -> dict
budget_status(runtime_root) -> dict
prepare_benchmark(root=BENCHMARKS, *, all_subjects=False) -> dict
load_benchmark(revision, *, runtime_root, all_subjects=False,
               expected_checksum=None) -> dict
```

`TAIDE_ID` is `taide/Llama-3.1-TAIDE-LX-8B-Chat`. The unbounded low-level `generate_answers` is for an already supervised worker, not a route handler. Route handlers call `generate_bounded`; training integration passes server-constructed command argument lists to `run_with_pilot_budget`. Do not accept arbitrary shell commands from clients. Wrapping the existing training entry point does not change its training parameters; the standalone pilot uses its own fixed eight-step worker.

Generation returns `status`, `answers`, `message`, `budget`, and where relevant `reason`, `model`, or `run`. Status is `complete`, `cancelled`, `budget_exhausted`, or `unavailable`. Actionable reasons include `missing_weights`, `missing_license`, `gpu_busy`, `insufficient_vram`, `no_gpu`, `gpu_unavailable`, `worker_failed`, and `generation_failed`. The command wrapper additionally returns `failed` for child failure, `phases`, and the invocation's `chargedGpuSeconds`. Budget fields are `limitGpuSeconds`, aggregate `chargedGpuSeconds`, and `remainingGpuSeconds`.

Status is read-only and local-only: `ready`, `missing_weights`, or `missing_license`. Resolved snapshots include `path`, immutable `revision`, and license ID/card/license-file hashes where present. `license.acceptance` is always `not_checked_offline`; cache presence is not evidence of accepting terms or current gate access.

Answers contain `caseId`, `candidate`, `output`, `protocol`, and `provenance`. Provenance records model SHA, license metadata, adapter content hash, and prompt hash. Passing `revision` fixes one snapshot across successive calls. Pilot phases resolve once and reuse that SHA. No mutable remote revision is used during loading.

## Evaluation semantics

Duplicate case IDs, duplicate candidate/case pairs, unknown IDs, unsupported tasks, invalid judgments/references, and outputs over 65,536 characters are rejected. No truncated policy inspection is represented as a pass. Missing answers are explicit in each candidate's coverage; when no answers exist, candidate names cannot be inferred and top-level `expectedCaseIds` exposes the expected set.

Classification uses exact text equality. JSON requires strict parsing (including nested duplicate-key rejection and rejection of NaN/Infinity/overflow), then structural equality: object key order does not matter, but booleans do not equal numbers. MCQ accepts only one A-D character after surrounding whitespace removal; explanations, lowercase, and punctuation fail. Writing requires an explicit human `pass`/`fail` judgment, otherwise it needs review. Missing references remain unscored.

The primary-owned `normalize_policy` and `inspect_text` functions are called lazily. Findings are returned without rewriting output, and informational `mixed_script` entries retain `replacement=None` and `editable=False`. Policy findings never become correctness scores. Normalized policy ID/version/OpenCC version and hashes are recorded.

Paired comparisons require identical case coverage, per-case generation protocol, policy provenance, and scored judgment coverage. Missing protocol metadata is not comparable. Per-task differences in prompts/token budgets are allowed only if both candidates use the same protocol for each case. Coverage/protocol mismatches are explained under `unpaired`. Cancellation returns partial results and suppresses pairing. No leaderboard, statistical significance, or quality-improvement claim is made.

## Benchmark preparation and loading

Default `BENCHMARKS` is `<FORGETUNE_DATA_ROOT>/quality/benchmarks`, or `hackathon/trainer/data/quality/benchmarks` when that environment variable is absent. It is separate from training datasets.

Preparation is an explicit network action:

```powershell
.venv/Scripts/python.exe -B hackathon/trainer/quality_models.py prepare benchmark
.venv/Scripts/python.exe -B hackathon/trainer/quality_models.py prepare benchmark --all-subjects
# Separate, explicit model download; requires pre-existing HF gate authorization:
.venv/Scripts/python.exe -B hackathon/trainer/quality_models.py prepare model
```

The sidecar agent did NOT run these preparation commands live. Offline tests replace the Hugging Face network boundary. The primary agent subsequently performed public benchmark preparation; its resulting artifact was independently loaded and verified below. Model preparation checks existing gate authorization and never accepts terms. Benchmark preparation resolves public `ikala/tmmluplus` at `v1.1` to a commit SHA, then uses only that SHA for source downloads.

The public refs endpoint reports `94d86f1d013d59f389b710a902af30b7a0aa8c8f` for `v1.1`, while `HfApi.dataset_info(..., revision="v1.1")` returns `466ccd9752ce6889f35c47f875ac7a65012e2269`. This is NOT a moved tag or an incorrect fallback: `v1.1` is an annotated Git tag. Read-only Git verification established the distinction:

```powershell
git ls-remote https://huggingface.co/datasets/ikala/tmmluplus 'refs/tags/v1.1*'
# 94d86f1d013d59f389b710a902af30b7a0aa8c8f  refs/tags/v1.1
# 466ccd9752ce6889f35c47f875ac7a65012e2269  refs/tags/v1.1^{}
```

`94d86f...` identifies the tag object; the dereferenced immutable dataset commit is **466ccd9752ce6889f35c47f875ac7a65012e2269**. The current preparation code correctly records the latter. `main` is a different commit, `639a64db464b4661feb91817b9edfe43932bdf18`. No fallback branch or invented SHA is used.

Sources: [public Git refs metadata](https://huggingface.co/api/datasets/ikala/tmmluplus/refs), [immutable v1.1 commit tree](https://huggingface.co/datasets/ikala/tmmluplus/tree/466ccd9752ce6889f35c47f875ac7a65012e2269), [TAIDE model card](https://huggingface.co/taide/Llama-3.1-TAIDE-LX-8B-Chat).

Default selection is the first 10 test questions in each of `computer_science`, `finance_banking`, and `official_document_management` (30 total). `--all-subjects` explicitly selects 10 test questions from every available subject; it does not mean every question. The prompt is fixed zero-shot, generation is greedy, maximum MCQ output is eight tokens, parsing is strict A-D. This bounded sample is not leaderboard-comparable.

Artifacts are `tmmluplus/<SHA>/default-30.json` or `all-subjects.json`, corresponding `.sha256` files, and pinned source CSV files. Preparation returns `manifestChecksum`. The loader accepts a SHA rather than an arbitrary filename and checks runtime containment, manifest checksum, optional caller-held checksum, source hashes, case hash, and dataset/version/revision provenance. Missing or altered artifacts raise `ValueError`. The checksum is a local integrity check; stronger adversarial authenticity requires retaining the expected checksum separately.

The primary agent's live-prepared 30-case artifact under `hackathon/trainer/data/quality/benchmarks` was successfully validated with `load_benchmark("466ccd9752ce6889f35c47f875ac7a65012e2269", runtime_root="hackathon/trainer/data", expected_checksum="148433ea475aeb2fd466a1e13367807c1c230c9f5e7965c1451c02a7240f7d38")`. The loader checked all source hashes and returned cases hash `842f7fca93cdd4b1653cce9270c938198ab73074b572a74dbc4298b0f72d815c` and the matching supplied manifest checksum. This read did not download or regenerate artifacts.

## Budget and process isolation

Both route-facing execution functions and the standalone pilot share `<runtime_root>/quality/gpu-budget.sqlite`. Transactions reserve allowance before starting a worker; settled runs charge measured time including load, failures, process startup, and cleanup. Unfinished reservations remain charged after a crash. Restarting the API does not reset the aggregate 1,800-second budget. No reset API is supplied.

The default selects an idle GPU by UUID, with no unrelated compute processes and at least 24,000 MiB free VRAM. Explicit GPU selection still requires admission checks. CUDA sees only that one GPU. OS leases prevent concurrent sidecar ownership of the same GPU. Availability is checked again before later phases; prior charges are retained if the GPU becomes busy. External workloads can race the check; the code does not change system-wide NVIDIA scheduling policy.

Commands run sequentially, with Hugging Face offline flags. Each has a separate supervisor carrying the common monotonic deadline; quality workers additionally arm an independent cutoff. Deadline/cancellation terminates only the owned process tree, never a PID discovered in the GPU inventory. No CPU training/inference fallback or hosted API fallback exists. The 1,800-second deadline triggers termination; OS termination latency is reported in `terminationOverrunSeconds` and charged, not hidden or clipped. Admission is conservative and does not guarantee freedom from OOM.

## Frozen structured-extraction pilot

One authored Traditional Chinese order-extraction task is used throughout: return JSON containing `order_id`, `item`, and integer `quantity`. Frozen partitions are 8 training, 2 development, and 4 final-test examples, with distinct order IDs and inputs. Development is held aside; the fixed pilot does not tune against it. Public benchmark cases are evaluation-only.

Original training targets contain explicit simplified item values. Reviewed targets are derived by applying recorded accepted span decisions, not by a hidden cleaner or an independently authored replacement training dataset. Every decision records completion offsets, original/replacement text, acceptance, reason, and reviewer `synthetic-fixture-author`. The eight decisions are:

| Order | Original item | Accepted item | Quantity |
|---|---|---|---:|
| TW101 | 铅笔 | 鉛筆 | 2 |
| TW102 | 电脑 | 電腦 | 1 |
| TW103 | 雨伞 | 雨傘 | 4 |
| TW104 | 书包 | 書包 | 3 |
| TW105 | 纸袋 | 紙袋 | 9 |
| TW106 | 红茶 | 紅茶 | 5 |
| TW107 | 绿豆 | 綠豆 | 7 |
| TW108 | 铁盒 | 鐵盒 | 6 |

These are documented fixture-author decisions, not fabricated production audit IDs or a claim that a human reviewed them in the UI. The primary agent owns the production audit/review/version provenance pipeline. The pilot validates the resulting JSON against frozen references and rechecks row IDs, whitespace-normalized inputs, and extracted order IDs across all partitions after applying decisions.

Base, original-adapter, and reviewed-adapter evaluations use identical held-out cases and generation protocol. Both tiny training runs use exactly 8 optimizer steps, LoRA rank 8, alpha 16, learning rate 0.0001, max sequence length 512, seed 42, batch size 1, accumulation 1, and dropout 0.05. The existing `training_text_dataset` completion-mask helper is reused without truncation; overlength data is rejected. Dataloader workers are zero. Separate processes reload the same immutable base, preventing optimizer/adapter leakage. This tiny shared-template fixture tests mechanics, not real-world generalization.

## Actual RED/GREEN evidence

All Python commands used the project `.venv/Scripts/python.exe`, `-B`, and pytest's disabled cache provider to avoid touching tracked bytecode or adding pytest cache. Windows shell calls required escalation; file patches used the documented `codex.exe --codex-run-as-apply-patch` fallback after the native patch helper encountered Windows error 1385.

Initial RED command (before implementation):

```powershell
.venv/Scripts/python.exe -B -m pytest hackathon/trainer/test_quality_evaluation.py hackathon/trainer/test_quality_models.py -q -p no:cacheprovider --tb=no
```

Result: **38 failed in 15.53s**, assertion failures for missing sidecar functionality. An earlier exploratory run produced 38 fixture-setup errors; the tests were corrected to fail in their bodies and RED was rerun before implementation.

Shared RED/GREEN regression command:

```powershell
.venv/Scripts/python.exe -B -m pytest hackathon/trainer/test_quality_evaluation.py hackathon/trainer/test_quality_models.py -q -p no:cacheprovider --tb=short
```

Observed milestones: initial **38 passed in 24.25s**; mixed-task/cache regression **2 failed, 40 passed in 15.73s**, then **42 passed in 13.53s**; route-budget/loader/output-limit RED **6 failed, 45 passed in 18.11s**, then **51 passed in 17.03s**; structured-fixture/deadline GREEN **52 passed in 18.34s**; shared-wrapper GREEN **54 passed in 25.12s**; final **55 passed in 18.66s**.

Additional targeted RED commands:

```powershell
.venv/Scripts/python.exe -B -m pytest hackathon/trainer/test_quality_models.py::test_eight_step_training_uses_safe_local_boundary -q -p no:cacheprovider --tb=short
.venv/Scripts/python.exe -B -m pytest hackathon/trainer/test_quality_models.py::test_fixture_partitions_are_fixed_and_independent hackathon/trainer/test_quality_models.py::test_worker_has_independent_deadline_if_parent_disappears -q -p no:cacheprovider --tb=short
.venv/Scripts/python.exe -B -m pytest hackathon/trainer/test_quality_models.py::test_training_command_wrapper_uses_shared_budget_and_offline_env hackathon/trainer/test_quality_models.py::test_supervisor_enforces_deadline_without_api_parent -q -p no:cacheprovider --tb=short
.venv/Scripts/python.exe -B -m pytest hackathon/trainer/test_quality_models.py::test_busy_between_phases_does_not_refund_previous_gpu_use -q -p no:cacheprovider --tb=short
```

Results respectively: **1 failed in 9.13s** (missing revision-aware worker interface); **2 failed in 0.72s** (wrong fixture task and missing worker cutoff); **2 failed in 1.54s** (missing shared wrapper/supervisor); **1 failed in 0.56s** (prior phase charge incorrectly refunded). All are included in final GREEN.

During trainer-boundary test setup, missing TRL and CPU-only Torch/PEFT CUDA initialization caused failures. The test was corrected to mock the actual TRL/PEFT execution boundary, while preserving real datasets and the existing completion-mask formatter. An intermediate full suite had **1 failed, 125 passed** (`test_eight_step_training_uses_safe_local_boundary`); that test harness failure was fixed and the full suite rerun.

Final full trainer/API verification:

```powershell
.venv/Scripts/python.exe -B -m pytest hackathon/trainer -q -p no:cacheprovider --tb=short
```

Result: **143 passed, 21 warnings in 45.71s**. Warnings are Starlette/AnyIO and FastAPI lifecycle deprecations in existing/concurrent application integration. No failing test was left hidden.

Fresh streaming/browser verification:

```powershell
node --test hackathon/test-quality-stream.mjs
$env:PLAYWRIGHT_MODULE = 'C:/Users/SPTTW10665/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'
node hackathon/test-quality-ui.cjs
```

Results: **1 streaming test passed**; browser output **Quality review and evaluation UI passed**. The browser smoke used headless Edge with mocked API responses, verified accept/version/export navigation and the evaluations page, and reported no page errors. It is not a live model or live inference test. The first browser attempt without `PLAYWRIGHT_MODULE` failed because project-local Playwright was absent; the bundled-runtime retry passed without installing dependencies.

## Live check and remaining limits

```powershell
.venv/Scripts/python.exe -B hackathon/trainer/quality_models.py status
```

Actual local result: `missing_weights`, revision null, `localOnly: true`, license acceptance `not_checked_offline`, with instructions to use explicit preparation after HF authorization. No model load was attempted. The user additionally reported a read-only VM preflight showing two idle Tesla T4 devices with 14,918 MiB free each and no TAIDE cache. Each is below this FP16 admission threshold; the sidecar does not combine GPUs or silently quantize/change the protocol. TRL is also absent in the checked local environment.

Real GPU use for this task: **0 seconds**. Budget tests launched CPU-only sleep/exit subprocesses or mocked the actual external process/GPU boundaries. No measured base/original/reviewed model scores exist. Training convergence, real TAIDE generation, and peak VRAM remain unvalidated; software and policy/API/browser checks do not substitute for those measurements. The live model pilot is explicitly blocked by prerequisites, not marked successful. The public benchmark artifact was prepared by the primary agent and validated successfully, as recorded above.

Final scope check found the six owned files untracked, with no staging or commits. A repository-wide `git diff --check` also flagged existing/concurrent trailing whitespace at `hackathon/trainer/app.py:764`; that file is outside sidecar ownership and was not edited here.
