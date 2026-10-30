# ForgeTune

A local-first fine-tuning studio for instruction and completion datasets, with a
Traditional Chinese (Taiwan) data-quality pilot built on TAIDE. The browser UI talks
to a small Node server, which proxies to a FastAPI/LangGraph trainer running on a
local NVIDIA GPU or on the `gcp-dev` VM.

## How it runs

```
Browser ──► Node web server (server.mjs, :4173) ──► Trainer API (trainer/app.py, FastAPI + LangGraph)
                                                             │
                                                             ├──► Compass API (optional, hosted)
                                                             ├──► local GPU ≥24 GB (:8000), or
                                                             └──► gcp-dev VM, 2× T4 (:18000 via SSH tunnel)
                                                                      │
                                         runtime data: datasets, runs, adapters, quality.sqlite, GPU ledger
```

- **Web server** (`server.mjs`, `public/`, `index.html`): serves the UI and proxies `/api/*` to the trainer; it holds no API keys.
- **Trainer** (`trainer/`): dataset import and audit, LangGraph workflow, training, evaluation, and the quality pilot.
- **Runtime data** (`trainer/data/`, `.forge-data/`): never committed.

## Quick start

From the repository root on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File start-auto.ps1
```

The launcher picks a local GPU with at least 24 GB, otherwise the `gcp-dev` VM, opens the SSH
tunnel when needed, and serves the UI at http://127.0.0.1:4173. Use `-CheckOnly` to inspect the
target and `-SelfTest` to check GPU selection. On the VM the backend is started (or restarted, after
stopping the running `uvicorn app:app` process) with `trainer/start-vm-backend.sh`.

To run only the UI (preview mode without a trainer): `npm start`.

## Two workflows

| Workflow | Steps | Where |
|---|---|---|
| Fine-tune a model | Import data (file, GitHub, Hugging Face) → analyse and benchmark → approve and train (LoRA/QLoRA) → evaluate base vs tuned | Workspace |
| Traditional Chinese quality | Audit for mainland terms → review into a new dataset version → compare original vs reviewed training → grade answers with the judge panel | Dataset quality, Comparisons, Judge panels |

Nothing trains or uses the GPU until you approve it, and no adapter is ever deployed automatically.

## Traditional Chinese quality pilot

A zh-TW / TAIDE-specific pipeline (`trainer/quality_*.py`, `public/quality.js`). Operations guide:
[docs/quality-pilot-operations.md](docs/quality-pilot-operations.md).

1. **Audit** a local or imported dataset (OpenCC suggestions plus an optional customer term list such as
   `validation/taiwanchat-terms.json`). Full scans cover every row; quick samples are capped at 600.
2. **Review** suggestions, accept or ignore them, and **create a dataset version**. Prompts, labels,
   code, URLs and quotations stay unchanged; the original file is never modified.
3. **Fine-tune** from that version with the **Fine-tune** button. It opens the normal fine-tune workflow
   with the reviewed data, tagged with its provenance: local-only, TAIDE only (no substitution), and the
   legacy judge is disabled.
4. **Compare original vs reviewed** trains two QLoRA adapters that differ only in the accepted edits
   (in parallel on the two T4s) and evaluates base, original-trained and reviewed-trained models on
   listed-term usage plus a TMMLU+ smoke subset. On TaiwanChat (2026-10-22) listed mainland terms per
   1,000 characters went 2.6 (base) → 8.3 (original-trained) → 0.4 (reviewed-trained), with no TMMLU+ change.
5. **Judge panel** grades saved writing answers from a quality evaluation or a finished comparison.
   Three local judges run in 4-bit, one per GPU: Breeze-7B, Qwen2.5-7B and Yi-1.5-9B. Correctness is a
   majority vote; zh-TW language quality is Breeze's verdict. Grades are labelled as estimates, shown
   with their coverage, and never overwrite human judgments. Comparisons send no reference answer to
   the judges. By default 20 prompts are graded.

All GPU work in the pilot charges one persistent GPU-second ledger (`quality_pilot.BUDGET_LIMIT_SECONDS`,
currently 15,000 s) with a 1,800 s cap per run. Start buttons stay disabled, with reasons, when an
estimate does not fit. Only one comparison or judge panel runs at a time.

**Preparing the judges** is an explicit, one-time download (about 15–18 GB per model; check `df -h`
first). Run it on the backend from the repository root with the backend's own cache settings:

```bash
FORGETUNE_DATA_ROOT=$PWD/data HF_HOME=$PWD/data/huggingface .venv/bin/python trainer/quality_models.py prepare judges
```

To grade a model trained through the Fine-tune button with the judge panel, first generate its answers
on **Evaluations** (TAIDE with that run's adapter), then start a judge panel on that evaluation.

## Tests

```powershell
python -m pytest trainer -q
node test/test-quality-panel.cjs
```

Test files (`test/`, `test-*.mjs`, `trainer/test_*.py`, `trainer/conftest.py`) are kept in local
checkouts and are not published to this repository; the commands here and in the feature sections
below assume a checkout that has them.

Python tests need the trainer dependencies; this Windows checkout has no `.venv`, so they are run on the VM
(`.venv/bin/python -m pytest trainer -q`). Browser tests need the web server on :4173 and Playwright
(set `PLAYWRIGHT_MODULE` when it is not installed in the project).

## Feature reference

The sections below document each feature and its safeguards in detail.

## What is included

### Pipeline safeguards (2026-10-05)

New workflows ask for an intended task and success metric before model
benchmarking. These are a recorded experiment brief, not an automatically
validated custom scoring function. Evaluation size is configurable from 1 to
200 examples per phase (default 50, bounded by available rows). Under 30
examples is explicitly exploratory; even larger synthetic samples are not
evidence of production quality. Compass models approved in the model plan
answer the same development questions as the local models.

Dataset audits sample up to 600 rows reproducibly across the entire dataset,
not just its prefix. Privacy findings retain original row IDs. Splits keep
normalized identical inputs together even when answers differ; external test
overlap is rejected. Existing manifests are checked again before training.
This is exact-input protection, not semantic near-duplicate or entity/time
leakage detection. Existing saved decisions are not rewritten.

SFT uses explicit response masks: prompt/completion and classification learn
the target; chat learns the final assistant turn, with earlier turns as context.
Plain text still uses full-sequence language modeling. Unsupported token
boundaries and sequences truncated to zero target tokens fail clearly.
Final evaluation is report-only: poor scores or judge failures never trigger
retraining. Operational retries are only permitted before final evaluation.

The offline `trainer/pipeline_pilot.py` exercises ticket classification, JSON
extraction and summarization with synthetic fixtures, a cached local model,
and a hard timeout. It preserves splits, configurations, logs, adapters and
paired answers in a new runtime output directory. It never deploys an adapter.
Use `--seconds` to reduce the default 1,800-second budget and select an idle
GPU via `CUDA_VISIBLE_DEVICES`. Summarization requires human review.

- Guided dataset, model/method, training, and evaluation/export workflow
- CSV, JSONL, Parquet, and Hugging Face dataset entry points
- PEFT, LoRA, and memory-efficient 4-bit QLoRA selections
- 3B, 7B, and guarded 14B model choices for a 24 GB GPU
- A one-job GPU queue, persisted run metadata/logs, cancellation, and adapter artifacts
- Ollama Modelfile generation alongside adapters and evaluation metrics
- LangGraph workflow orchestration with resumable approvals, shared evaluation splits, bounded retries, and candidate comparison

## Start the web app

    npm start

Open http://127.0.0.1:4173. The interface remains usable in preview mode when no trainer is running.

## Enable real GPU training in WSL2

From a WSL terminal, enter the same project folder and run:

    cd trainer
    chmod +x start.sh
    ./start.sh

The first run creates a virtual environment and installs the Python dependencies. Keep it running while the browser submits jobs. Training data, logs, adapters, metrics, and exports are stored under trainer/data/.

## Guided LangGraph workflow

When the local trainer is online, the workspace starts a guided workflow that profiles the dataset, creates a reproducible held-out split, runs base-model baselines, and pauses before GPU training so the proposed candidates and methods can be reviewed. It then trains eligible candidates serially, retries failed evaluations at most twice, and saves base-vs-tuned comparisons plus a final candidate leaderboard under trainer/data/workflows/.

The workflow API is available at `/workflow/runs`, `/workflow/runs/{id}`, and `/workflow/runs/{id}/resume`. Existing `/runs` endpoints remain available for single-model runs.

## Optional cloud classification

ForgeTune can use an OpenAI-compatible gateway only when local profiling is uncertain. Copy trainer/.env.example to trainer/.env and enter the base URL, API key, and model ID there. Do not place the API key in the browser or commit the .env file. The advisor sends at most 20 locally redacted sample rows only after cloud assistance is requested.

The local dataset auditor works without this configuration and remains the authoritative source for dataset quality and GPU feasibility checks.

## Dataset schemas

The trainer accepts:

- messages: chat messages compatible with the model chat template
- prompt + completion: paired supervised examples
- text: complete text examples

When no test dataset is supplied, the trainer creates a reproducible 90/10 train/evaluation split using seed 42.

## Ollama

After conversion of the merged model to GGUF, use the generated Modelfile:

    ollama create forge-support -f Modelfile
    ollama run forge-support

Ollama is not currently installed on this computer, so install it before importing a completed GGUF export.

## Prompt review inside the dataset workflow

After uploading or importing a dataset, review the audit and any clarification.
**Test base model & prompts** prepares fixed development examples automatically
and runs the base model. Try improved instructions or reference context in that
same workflow, inspect recorded responses, and select a prompt version. Choose
**Keep base model** to finish without training, or **Review a small fine-tune** to
open the separate training approval. No manual example entry is required.

Prompt trials and decisions are checkpointed with the dataset workflow. Each
workflow allows up to 10 trials. Missing model dependencies or inference errors
remain visible and cannot be accepted as successful results. The old standalone
prompt page has been removed; existing legacy API experiment records are retained.

New split manifests keep development examples separate from training and final
evaluation. Both base and tuned models use the selected instructions/context on
the same final examples. Token overlap remains a screening metric rather than
proof that a model is ready to deploy. Review task-specific quality before release.

## Dataset-first workspace

Upload CSV, JSONL, or Parquet on the workspace to start a LangGraph analysis-only
run. Local profiling checks the schema and up to 600 rows for quality, duplicates,
language, and likely task. Uncertain intent pauses for a task and explanation;
optional cloud assistance can suggest an interpretation but cannot confirm it for
you. Blocking data issues require replacement or explicitly approved cleanup.

Analysis stops at **Dataset ready** before any baseline inference or training.
**Update dataset** uploads a new file and starts a new analysis linked to its
previous version. Earlier files and clarification records remain available in
version history. Updates replace the file for a new version; they do not merge
rows or edit the prior version.

SQLite checkpoints preserve pending questions across restarts. CPU-only dataset
analysis can run with FastAPI, uvicorn, datasets, python-dotenv, python-multipart,
langgraph, langgraph-checkpoint-sqlite, and aiosqlite installed; no GPU model is
loaded until you explicitly continue to model evaluation. Run the API with
`python -m uvicorn app:app --app-dir trainer --host 127.0.0.1 --port 8000`.
`FORGETUNE_DATA_ROOT` optionally selects a separate storage directory. This local
session uses the repository's ignored `.forge-data/` directory.

Checks: `python -m pytest trainer -q` and `node test/test-datasets.cjs` (Playwright and
the web server required). The API tests exercise real file uploads and SQLite
restart/resume; browser tests use controlled API responses.

## Import from GitHub

Choose **GitHub** in the dataset import panel and paste a public file URL such as
`https://github.com/owner/repo/blob/main/data.jsonl`, or its
`raw.githubusercontent.com` URL. CSV, JSONL, and Parquet files up to 50 MB are
supported. Repository/folder links and private repositories are not supported.

`POST /datasets/github` saves a new local copy and retains its canonical source
URL. The UI then starts the same analysis-only LangGraph workflow used for file
uploads, including clarification and version history. Imports never execute
repository code, do not follow redirects, and remove partial downloads on failure.

## Hugging Face Hub imports

Choose **Hugging Face**, paste a dataset page URL or `owner/dataset` ID, and select
its split (normally `train`). **Full dataset (all subsets)** is the default:
**Import Full Data** downloads every configuration/subset for that split and
automatically audits each one. There is no upfront subset choice. All snapshots
use the same pinned Hub revision, with no sample row/size caps.

After every subset has been processed, **Choose subsets for training** shows
each audit and any download/analysis failures. Select one or more analyzed
subsets, then **Continue with selected subsets**. Each gets a separate saved
workflow; different schemas are never silently merged. Repeating selection
reuses the same workflows. Open a workflow to review its data and proceed through
model/prompt evaluation and training approval. Selection itself starts no GPU
work. A subset without the requested split is reported as failed rather than
silently substituted with test or validation data.

Full imports run in the background with row/byte progress and cancellation.
Reloading or returning to the workspace restores the current import in the same
browser. Each subset receives a local quality audit of up to 600 rows. Selected
workflows also run the existing privacy review; split preparation and later
training can use every row in the corresponding saved subset. Sampled audits
do not certify every row.
Large files may take additional time and disk space to prepare the Arrow cache
and train/development/final split; row-index manifests still grow with row count.

**Small sample** retains the first 30–10000 rows (default 600), capped at 50 MiB.
A prefix sample is not necessarily representative of all domains. Both modes
preserve chat `messages`, source metadata, and the resolved Hub revision.

The importer reserves 1 GiB of free space on the trainer plus an estimate for the
later dataset cache. Disk exhaustion, network errors, and cancellation remove
unfinished files; no partial import is handed to analysis. Cancellation takes
effect after the current network read returns. A trainer restart marks running
imports interrupted and removes their partial downloads; restart those imports
from the beginning. Completed subset snapshots are retained, including when
another subset fails or the batch is cancelled.

`POST /datasets/huggingface` with `mode:"all"` returns HTTP 202 and an import ID.
Poll `GET /datasets/imports/{id}`; cancel with
`POST /datasets/imports/{id}/cancel`. After completion, post
`{"configurations":["chat","math"]}` to `/datasets/imports/{id}/select` to prepare
separate analysis-only workflows. Job metadata is saved in
`datasets/.imports/` under the configured trainer data directory. Only one full
import runs per trainer process. The existing sample API remains synchronous and
defaults to sample mode for older clients; `mode:"full"` continues to import
one selected configuration for existing clients. Update the backend as well as the UI
(including the VM deployment when using `gcp-dev`). No new dependencies or
environment variables are required.

The supplied `nvidia/Nemotron-Cascade-2-SFT-Data` link was verified with a temporary
30-row import of its `chat` / `train` subset. Larger or long-context subsets may
need a smaller row count. Model inference currently truncates inputs to 4096 tokens
and generates at most 96 new tokens, so this is a bounded local comparison, not a
reproduction of Nemotron's full training setup.

References consulted:
- [Fine-tuning topic](https://github.com/topics/finetuning-large-language-models)
- [Data preparation, training and evaluation course](https://github.com/ksm26/Finetuning-Large-Language-Models)
- [Nemotron dataset card](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data)
- [Hugging Face streaming](https://huggingface.co/docs/datasets/stream)

Verification: `python -m pytest trainer -q`, `node test/test-datasets.cjs`, and
`node test/test-workflow-prompts.cjs`. Browser tests require the web server and Playwright
(or `PLAYWRIGHT_MODULE` pointing to the installed module). GPU inference must be
verified separately in the WSL training environment.


### Automatic model analysis
New dataset workflows automatically test the baseline and two task-specific prompt strategies after data clarification. Results and the selected test are saved in the same LangGraph workflow. Older workflows can use **Run automatic analysis**. Manual prompt editing is optional; training still requires plan approval.

Ground-truth label accuracy (with precision, recall, F1, and errors) and structured JSON exact match are measured for both base and tuned models where supported. Word overlap remains a diagnostic and cannot qualify a fine-tune for recommendation. Unsupported free-form correctness requires a task rubric. Review requires at least a 5 percentage point score gain on the same fully scored held-out examples without format regression; it never deploys automatically. Prompt trials prefer task scores when available. An unavailable inference runtime produces no recommendation to train. Final evaluation remains separate from prompt development.

No independent retrieval corpus is connected to the website workflow yet. `trainer/quality_probe.py` provides a separate reproducible, small-corpus RAG experiment using `validation/rag-corpus.json`; its results must not be confused with production retrieval. The workflow generates up to 512 tokens per answer and records whether that limit was reached.


### Prefer local 24 GB GPU, otherwise use dev-01
Run `powershell -ExecutionPolicy Bypass -File start-auto.ps1` from the repository. Selection happens at startup: a local NVIDIA GPU reporting at least 24,000 MiB is preferred; otherwise the launcher uses the existing `gcp-dev` SSH alias. Two smaller cards are not counted as one 24 GB card. Use `-CheckOnly` to inspect the target and `-SelfTest` to check the selection rule.

Local mode requires CUDA-enabled PyTorch in `.venv`. VM mode requires a Python 3.11+ CUDA environment at `.venv` (or `.venv311`) in the deployment directory (Python 3.10 loses LangGraph interrupt context). The `gcp-dev` deployment lives at `/home/jupyter/dev-01/cherrie.chen/llm_fine_tune` and is started with `trainer/start-vm-backend.sh`; to restart, stop the running `uvicorn app:app --app-dir trainer` process first, because the script exits when port 18000 already answers. The API binds to VM loopback port 18000 and the launcher forwards it over SSH. No public API port is opened. Dataset and checkpoint migration is a separate approved operation; the launcher does not synchronize data or fail over during running jobs. Restart the launcher to change targets. Existing independently started website processes must be stopped before the launcher can take port 4173.


### Advisor API configuration and quality validation

The backend loads `trainer/.env`, then `.env` in its startup directory without overriding existing variables. Copy `trainer/.env.example` there and fill in `ADVISOR_BASE_URL`, `ADVISOR_API_KEY`, and `ADVISOR_MODEL` for an OpenAI-compatible endpoint. The advisor hook assists dataset analysis. The trainer uses the same Compass key to pick models, check that they respond, and answer benchmark questions. `GET /advisor/status` reports whether all three values are present. Use the exact variable names in the example. In an earlier version, where the website also called Compass, the root `.env` listed 261 Compass model IDs, and an approved benchmark completed nine gpt-4.1-mini answers and eight Qwen2.5-32B-Instruct answers; the last Qwen question remained rate-limited. This does not configure the VM advisor automatically.

The tested trainer uses TRL 0.22 and Transformers 4.x.


## Integrated Nemotron workflow

Open a dataset workflow to use automatic base/prompt/RAG trials, and view live training steps and token retention. Reference uploads accept independent text or Markdown documents; retrieval is per question, and zero matching sources is reported without concluding that RAG is unnecessary. Prompt/RAG results remain visible while training runs and after it completes.

The website does not call Compass and needs no `ADVISOR_*` values. Only the trainer does: it reads `ADVISOR_BASE_URL` (which must be `https://compass.llm.shopee.io/compass-api/v1`) and `ADVISOR_API_KEY` (an active Compass key) from `trainer/.env` (never committed), then `.env` in its startup directory, as described above, for model picking, model checks and Compass benchmark answers. The key is never sent to the browser. `ADVISOR_MODEL` still selects the optional dataset advisor model on a backend where that configuration is installed.

The Compass catalog can list models that this key cannot run. Before showing the model plan, the trainer sends each Compass model the picker proposed one short request (20-second timeout, no retry) and moves any that fail to **Not suggested**. A definite refusal (an HTTP 4xx other than 429, such as "model not supported") also removes the model from the options for that run; a timeout, rate limit or server error leaves it available to add back by hand. Adding a Compass model by hand calls `GET /api/compass/check?model=<id>` (the trainer's `GET /compass/check`) first and adds only models that answer.

The picker learns from earlier runs through a model history in `model-history.json` under the trainer's data root (`trainer/data/` by default). After each development benchmark, the trainer records one entry per model: task type, date, status, score, cost, mean latency, question count, and an error that is either the quick check's message or a fixed text. It also records Compass models that a quick check definitely refused. A benchmark refusal counts as definite only when a dataset-free quick check of the same model is refused too. The history holds no questions, answers or dataset text, and keeps the newest 500 entries. When the history is first used, it imports results from existing workflows, and those imported results never hide a model. When picking, the agent sees a per-model summary of past results for the same task type, including how many questions they cover. A Compass model whose latest result was a definite refusal is hidden for 7 days and listed under **Not suggested** with the date and error. Deleting the file resets the history: it is rebuilt from existing workflows, and recent refusals are cleared.

Training now uses the development split for epoch validation. The final split is reserved for the paired base/adapter comparison after training. Full tuned answers, source retrievals, training metrics, and actual token retention are retained. A completed adapter is not automatically deployed.

The general-assistant Nemotron pilot uses 200 shuffled conversations that fit 2,048 tokens, excludes the earlier test questions, and splits into 170 training / 10 development / 20 final examples. Three development examples are used for the bounded prompt/RAG checks. It is a pilot on shorter conversations, not a full-dataset or long-context benchmark.

Checks: `python -m pytest trainer -q`; browser scripts `test/test-integrated-tools.cjs`, `test/test-workflow-prompts.cjs`, and `test/test-datasets.cjs` (set `PLAYWRIGHT_MODULE` if using a bundled Playwright installation).


### Development benchmark model selection

New workflows benchmark every GPU-eligible catalog model before prompt experiments. `candidate_limit` limits training candidates, not the initial benchmark. All candidates receive the same development IDs, instruction and 256-output-token cap; a local input exceeding 4,096 tokens is rejected instead of truncated. No final test questions are used to choose the model.

Selection requires at least two completed models, full correctness coverage and the same scoring metric. Label accuracy and exact JSON matching are computed automatically. Free-form answers use the local task evaluator described below; incomplete grades require review. A shared human rubric can override development grades for every answer across all completed models. Word overlap, parameter count and training loss are never quality substitutes. Failed candidates remain visible but cannot win.

### Automatic task evaluation

`trainer/evaluator.py` grades recorded answers during model benchmarking, baseline/prompt/RAG trials, and the LangGraph `tuned_accuracy_reviewer` node before final comparison. It chooses a rubric per question (math, code, summarization, writing, factual knowledge, or general assistance), using dataset metadata and the original request. Structured targets keep their deterministic metrics. Other answers receive a reference-guided pass/fail/abstain verdict and a reason, with candidate model identity hidden.

The default judge is the locally cached `Qwen/Qwen2.5-7B-Instruct`, loaded in 4-bit mode on the backend GPU. Set `EVALUATOR_MODEL` on the backend to use another cached model. Evaluation sends no data to a hosted API and downloads no model automatically. Code grading is static review, not execution. Ambiguous references, unavailable models, invalid judge output, and overlong context remain ungraded. Model grades are shown as **estimated correctness**, not verified accuracy. Calibrate these estimates with human review before deployment.

Every recorded evaluation answer is graded; the website shows five detailed examples. Comparison requires the same evaluator version, metric, questions and full grading coverage for both models. The existing 5-percentage-point improvement threshold only proposes human review, never deployment. Development questions remain separate from final evaluation.

For completed workflows, **Run automatic evaluation** calls `POST /api/workflow/runs/{id}/evaluate`. It grades saved base/tuned answers without retraining or regenerating them, and preserves the original workflow in `before-automatic-evaluation.json`. Missing or uncertain grades produce **Needs review**, not a fabricated score.

The highest correctness score defines a shortlist within two percentage points. Lower cost, then lower mean answer latency, breaks ties within that shortlist. This is a provisional small-sample policy, not a statistical significance claim. Models on your VM count as free ($0). Compass models cost the `usage.price_cost` that Compass reports for each answer, or recorded token usage times the input/output rates in USD per million tokens when no charge is reported. These are inference charges, not training costs or amortized serving costs. If a Compass model reports no price, cost is not compared and latency breaks the tie.

Compass models approved in the model plan answer the same development questions, instruction and output cap as the local models. A successful hosted candidate can win; the workflow retains it as the untuned baseline and never sends a hosted model to local fine-tuning. Local winners proceed through the prompt/RAG workflow and training approval.

Existing saved workflows retain their original decisions. **Benchmark models on this dataset** creates a new workflow using the same dataset and split seed. Previously inspected final results remain historical; use a fresh untouched final set before claiming a new model improvement.

Checks: `.venv/Scripts/python.exe -m pytest trainer -q` and `node test/test-model-selection.cjs` (set `PLAYWRIGHT_MODULE` if Playwright is supplied by the desktop runtime).
