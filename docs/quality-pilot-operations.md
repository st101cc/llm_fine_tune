# ForgeTune Traditional Chinese quality pilot

This extends the existing browser and FastAPI application. It is a public/synthetic-data pilot, not a production-readiness or Taiwan-residency certification.

## Start and use

Install the trainer requirements, including exactly `OpenCC==1.4.2`. Run the existing trainer and Node server as usual. The Node `/api` proxy exposes `/quality`; no additional production service is required. Runtime metadata lives under `FORGETUNE_DATA_ROOT/quality/quality.sqlite`; derivative JSONL files live in the existing `datasets` directory. Back up both together. Never commit runtime data, logs or adapters.

Open **Dataset quality** from the workspace sidebar. Choose a previously imported local dataset (including completed Hub subsets) or upload a file. Use **Background full scan** before review; **Quick sample** is explicitly limited to 600 rows. Existing full-Hub import analysis and subset selection are unchanged; failed subset imports remain visible in that import workflow. Quality scans are launched separately for each imported subset, not as an aggregate collection-level audit.

Review each suggestion, accept or ignore it, then create a dataset version. Only accepted, eligible text edits apply. Prompts, metadata and labels remain unchanged. Raw-text correction requires an explicit opt-in. OpenCC candidates are not evidence of linguistic error; shared/ambiguous characters are conservatively skipped, and names require reviewer judgment or exclusions. Code, URLs and quoted passages are protected. Structured JSON strings are consequently not auto-corrected as ordinary prose. Bounded 256-character conversion segments may miss terminology spanning segment boundaries. Customer mappings can make an intended preference explicit.

Full audit scans every row with disk-backed duplicate/conflict tracking. Text fields above 65,536 characters are marked skipped and prevent full-coverage correction. Processing memory is proportional to the largest decoded row/Parquet batch and finding page, not the total row count; unusually large individual rows still need pre-validation. Cancellation and interrupted states are not passes. Restart preserves completed work without restarting expensive jobs.

If a process exits while writing a derivative, startup preserves unrecorded partial files in `quality/recovery-orphans` so a retry can proceed. Published versions with durable intent are reconciled by fingerprint. The archive is recoverable evidence, not a completed dataset or a reason to treat an interrupted audit as passed.

Exports use server-issued IDs. Original bytes remain untouched; derivatives preserve row order and unchanged fields. Reports retain the decisions attached to that version even if later reviews change. Source and output fingerprints detect stale data. Fine-tune opens the existing approval workflow, carries verified provenance, disables cloud assistance and selects only locally cached TAIDE. Dataset replacement after review requires a new audit.

## Independent evaluation

Upload JSONL cases with `id`, `input`, `task` (`classification`, `json`, `writing`, `mcq`), optional `reference`, and optional `subject`. Upload answers with `candidate`, `caseId`, `output`, optional `judgment` (`pass`, `fail`, `needs_review`) and optional explicit `protocol`. Duplicate IDs and inconsistent cases are rejected. Missing answers are reported. Cases are frozen in the evaluation record with a hash; there is no endpoint for editing them.

Exact classification, strict JSON validity/structural equality, MCQ accuracy, language findings and human judgments remain separate. Free-form correctness needs human review. No DeepEval/Promptfoo integration, optional local-judge panel, Taiwan Truthful QA integration or radar chart is included. Existing legacy workflow judging is not silently enabled for quality evaluations. Saved legacy answers retain a legacy provenance label; missing protocols prevent unsupported paired comparisons.

You may also load a saved ForgeTune workflow by its ID or explicitly generate from cached TAIDE, optionally using an existing completed adapter run ID. No arbitrary adapter/export filesystem path is accepted by these APIs. Local generation never downloads a model or substitutes another one.

## Public preparation versus offline execution

Preparation is an explicit network action, separate from quality processing:

```powershell
.venv/Scripts/python.exe -B hackathon/trainer/quality_models.py prepare benchmark
.venv/Scripts/python.exe -B hackathon/trainer/quality_models.py prepare benchmark --all-subjects
# Only after reviewing the license and obtaining required Hugging Face access:
.venv/Scripts/python.exe -B hackathon/trainer/quality_models.py prepare model
```

TMMLU+ v1.1 resolves to an immutable commit during preparation. Default selection is the first ten test questions each from computer science, finance/banking and official-document management. The all-subjects option selects ten questions per subject, not a full benchmark. Sources and manifests have checksums and remain outside trainable datasets. Evaluation uses fixed zero-shot instructions, greedy generation, strict whole-answer A–D parsing and no silent input truncation. Results are per-subject and per-question aggregates, not official leaderboard scores.

The current supervised model runner uses FP16, a 4,096-token input-plus-output ceiling and conservative admission of an idle GPU with at least 24,000 MiB free. A T4 alone is therefore unavailable for this path, even though separate quantized training configurations might fit. There is no automatic multi-GPU fallback. No new compute is provisioned.

## Bounded live pilot

```powershell
.venv/Scripts/python.exe -B hackathon/trainer/quality_pilot.py --output .forge-data/quality-pilot-new --runtime-root hackathon/trainer/data --seconds 1800
```

Use a new output directory. The frozen synthetic fixture is structured extraction with disjoint train/development/test cases. Original and fixture-author-reviewed variants use identical eight-step training settings. Fixture decisions are explicit synthetic author approvals, not invented production reviewer records. Test material is never used for training; overlap is checked again after corrections.

The persistent GPU budget is aggregate across quality inference, pilot phases and quality-approved training, including load time and failed attempts. Unfinished reservations remain charged after a crash. Do not delete or reset the budget ledger to obtain another allowance. The supervisor terminates only its owned worker tree at its deadline; no unrelated GPU workload is stopped. No adapter is automatically deployed. Actual tokenizer/mask/reload verification remains incomplete until authorized weights and adequate existing hardware are available.
