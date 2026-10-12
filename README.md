# ForgeTune

A local-first fine-tuning studio for instruction and completion datasets. The browser runs on Windows; model training runs through a WSL2 Python service on your NVIDIA GPU.

## What is included

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
