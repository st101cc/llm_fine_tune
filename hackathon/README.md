# ForgeTune

A local-first fine-tuning studio for instruction and completion datasets. The browser runs on Windows; model training runs through a WSL2 Python service on your NVIDIA GPU.

## What is included

### Pipeline safeguards (2026-10-05)

New workflows ask for an intended task and success metric before model
benchmarking. These are a recorded experiment brief, not an automatically
validated custom scoring function. Evaluation size is configurable from 1 to
200 examples per phase (default 50, bounded by available rows). Under 30
examples is explicitly exploratory; even larger synthetic samples are not
evidence of production quality. Hosted Compass comparisons retain their
9-question limit and cannot join a larger local benchmark.

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

Checks: `python -m pytest trainer -q` and `node test-datasets.cjs` (Playwright and
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

Verification: `python -m pytest trainer -q`, `node test-datasets.cjs`, and
`node test-workflow-prompts.cjs`. Browser tests require the web server and Playwright
(or `PLAYWRIGHT_MODULE` pointing to the installed module). GPU inference must be
verified separately in the WSL training environment.


### Automatic model analysis
New dataset workflows automatically test the baseline and two task-specific prompt strategies after data clarification. Results and the selected test are saved in the same LangGraph workflow. Older workflows can use **Run automatic analysis**. Manual prompt editing is optional; training still requires plan approval.

Ground-truth label accuracy (with precision, recall, F1, and errors) and structured JSON exact match are measured for both base and tuned models where supported. Word overlap remains a diagnostic and cannot qualify a fine-tune for recommendation. Unsupported free-form correctness requires a task rubric. Review requires at least a 5 percentage point score gain on the same fully scored held-out examples without format regression; it never deploys automatically. Prompt trials prefer task scores when available. An unavailable inference runtime produces no recommendation to train. Final evaluation remains separate from prompt development.

No independent retrieval corpus is connected to the website workflow yet. `trainer/quality_probe.py` provides a separate reproducible, small-corpus RAG experiment using `validation/rag-corpus.json`; its results must not be confused with production retrieval. The workflow generates up to 512 tokens per answer and records whether that limit was reached.


### Prefer local 24 GB GPU, otherwise use dev-01
Run `powershell -ExecutionPolicy Bypass -File hackathon/start-auto.ps1` from the repository. Selection happens at startup: a local NVIDIA GPU reporting at least 24,000 MiB is preferred; otherwise the launcher uses the existing `gcp-dev` SSH alias. Two smaller cards are not counted as one 24 GB card. Use `-CheckOnly` to inspect the target and `-SelfTest` to check the selection rule.

Local mode requires CUDA-enabled PyTorch in `.venv`. VM mode requires a Python 3.11+ CUDA environment at `~/forgetune-backend/.venv311` (Python 3.10 loses LangGraph interrupt context), with the backend deployed under `~/forgetune-backend`, started with `bash ~/forgetune-backend/trainer/start-vm-backend.sh`. The API binds to VM loopback port 18000 and the launcher forwards it over SSH. No public API port is opened. Dataset and checkpoint migration is a separate approved operation; the launcher does not synchronize data or fail over during running jobs. Restart the launcher to change targets. Existing independently started website processes must be stopped before the launcher can take port 4173.


### Advisor API configuration and quality validation

The backend loads `trainer/.env`, then `.env` in its startup directory without overriding existing variables. Copy `trainer/.env.example` there and fill in `ADVISOR_BASE_URL`, `ADVISOR_API_KEY`, and `ADVISOR_MODEL` for an OpenAI-compatible endpoint. The advisor hook assists dataset analysis. The separate Compass controls described below run external model comparisons from the workflow page. `GET /advisor/status` reports whether all three values are present. Use the exact variable names in the example. The corrected root `.env` successfully listed 261 Compass model IDs. A separate approved benchmark completed nine gpt-4.1-mini answers and eight Qwen2.5-32B-Instruct answers; the last Qwen question remained rate-limited. See `training-rag-validation-2026-09-10.md` for results and limitations. This does not configure the VM advisor automatically.

See [actual training and quality validation](training-rag-validation-2026-09-10.md) for the three sample datasets, training artifacts, prompt/RAG comparisons, and limitations. The tested trainer uses TRL 0.22 and Transformers 4.x.


## Integrated Nemotron workflow

Open a dataset workflow to use automatic base/prompt/RAG trials, view live training steps and token retention, and compare hosted models with Compass. Reference uploads accept independent text or Markdown documents; retrieval is per question, and zero matching sources is reported without concluding that RAG is unnecessary. Prompt/RAG results remain visible while training runs and after it completes.

The local website server reads the root `.env` using Node's built-in environment loader (Node 22+). `ADVISOR_BASE_URL` must be `https://compass.llm.shopee.io/compass-api/v1`, and `ADVISOR_API_KEY` must hold an active Compass key. The key stays on the local website server; it is not copied to the VM or sent to the browser. `ADVISOR_MODEL` still selects the optional dataset advisor model on a backend where that configuration is installed. The Compass comparison lets users choose one or two IDs returned by the live model catalog independently.

`GET /api/compass/models` lists IDs. `GET/POST /api/workflow/runs/{id}/compass?phase=development|heldout` reads or starts a resumable comparison, limited to 1–9 questions and 256 output tokens each. Final reference answers are removed; final-test questions are unavailable until the local workflow completes. Each comparison button sends the chosen questions to Compass. Results stay in the root `.forge-data/compass/` directory; repeat the same request to resume only missing answers after a rate limit or restart. Models that appear in the catalog can still reject generation or lack permissions. These exploratory free-form comparisons do not produce a validated accuracy score.

Training now uses the development split for epoch validation. The final split is reserved for the paired base/adapter comparison after training. Full tuned answers, source retrievals, training metrics, and actual token retention are retained. A completed adapter is not automatically deployed.

The general-assistant Nemotron pilot uses 200 shuffled conversations that fit 2,048 tokens, excludes the earlier test questions, and splits into 170 training / 10 development / 20 final examples. Three development examples are used for the bounded prompt/RAG checks. It is a pilot on shorter conversations, not a full-dataset or long-context benchmark.

Checks: `python -m pytest trainer -q`; `node --test test-compass.mjs`; browser scripts `test-integrated-tools.cjs`, `test-workflow-prompts.cjs`, and `test-datasets.cjs` (set `PLAYWRIGHT_MODULE` if using a bundled Playwright installation).


### Development benchmark model selection

New workflows benchmark every GPU-eligible catalog model before prompt experiments. `candidate_limit` limits training candidates, not the initial benchmark. All candidates receive the same development IDs, instruction and 256-output-token cap; a local input exceeding 4,096 tokens is rejected instead of truncated. No final test questions are used to choose the model.

Selection requires at least two completed models, full correctness coverage and the same scoring metric. Label accuracy and exact JSON matching are computed automatically. Free-form answers use the local task evaluator described below; incomplete grades require review. A shared human rubric can override development grades for every answer across all completed models. Word overlap, parameter count and training loss are never quality substitutes. Failed candidates remain visible but cannot win.

### Automatic task evaluation

`trainer/evaluator.py` grades recorded answers during model benchmarking, baseline/prompt/RAG trials, and the LangGraph `tuned_accuracy_reviewer` node before final comparison. It chooses a rubric per question (math, code, summarization, writing, factual knowledge, or general assistance), using dataset metadata and the original request. Structured targets keep their deterministic metrics. Other answers receive a reference-guided pass/fail/abstain verdict and a reason, with candidate model identity hidden.

The default judge is the locally cached `Qwen/Qwen2.5-7B-Instruct`, loaded in 4-bit mode on the backend GPU. Set `EVALUATOR_MODEL` on the backend to use another cached model. Evaluation sends no data to a hosted API and downloads no model automatically. Code grading is static review, not execution. Ambiguous references, unavailable models, invalid judge output, and overlong context remain ungraded. Model grades are shown as **estimated correctness**, not verified accuracy. Calibrate these estimates with human review before deployment.

Every recorded evaluation answer is graded; the website shows five detailed examples. Comparison requires the same evaluator version, metric, questions and full grading coverage for both models. The existing 5-percentage-point improvement threshold only proposes human review, never deployment. Development questions remain separate from final evaluation.

For completed workflows, **Run automatic evaluation** calls `POST /api/workflow/runs/{id}/evaluate`. It grades saved base/tuned answers without retraining or regenerating them, and preserves the original workflow in `before-automatic-evaluation.json`. Missing or uncertain grades produce **Needs review**, not a fabricated score.

The highest correctness score defines a shortlist within two percentage points. Lower cost, then lower mean answer latency, breaks ties within that shortlist. This is a provisional small-sample policy, not a statistical significance claim. `gpu_hourly_cost_usd` can be supplied when creating a workflow; the review screen also accepts a VM hourly rate and Compass input/output rates in USD per million tokens. Local cost is measured answer-generation time (including prefill) times the VM hourly rate. It excludes loading and diagnostic loss passes. Hosted cost uses recorded token usage. These are rate-based inference estimates, not invoices, training costs or amortized serving costs. Unknown costs block automatic selection unless the user explicitly chooses to exclude cost for all models.

At the model review checkpoint, the existing Compass form locks the split, number of questions and prompt to this benchmark. Run one or two text models, then use **Use these results in model selection**. A successful hosted candidate can win; the workflow retains it as the untuned baseline and never sends a hosted model to local fine-tuning. Local winners proceed through the prompt/RAG workflow and training approval.

Existing saved workflows retain their original decisions. **Benchmark models on this dataset** creates a new workflow using the same dataset and split seed. Previously inspected final results remain historical; use a fresh untouched final set before claiming a new model improvement.

Checks: `.venv/Scripts/python.exe -m pytest hackathon/trainer -q`, `node --test hackathon/test-compass.mjs`, and `node hackathon/test-model-selection.cjs` (set `PLAYWRIGHT_MODULE` if Playwright is supplied by the desktop runtime).
