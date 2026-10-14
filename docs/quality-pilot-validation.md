# ForgeTune Traditional Chinese quality pilot — validation

Date: 2026-10-05. Branch: `codex/traditional-chinese-quality`.

## Outcome

The software workflow is implemented and verified locally: import an existing/uploaded dataset, audit it, explicitly review corrections, create a derivative, export, independently evaluate, or open the existing training-approval flow. Existing user changes were preserved; nothing was committed or deployed to the existing VM backend. The existing port-4173 application remains unchanged as a running service. A separate validation preview is available at `http://127.0.0.1:4174/#/quality`, backed by port 18001 and isolated synthetic runtime data.

Live TAIDE model validation was **blocked** at the time of this report, not passed. No production-readiness, Taiwan-residency, commercial-quality, or model-improvement claim is made. See the 2026-10-14 update below.

## Update 2026-10-14: live T4 runs and TaiwanChat comparison

The owner approved the bnb4 (NF4, double-quant) path. On the `gcp-dev` VM (2× Tesla T4, 15 GB), TAIDE `6738923…` is cached and runs end to end.

- **TMMLU+ baseline (v4 manifest):** 8/30 (26.7%). The MCQ token budget was raised from 8 to 200 and scoring reads the leading A–D letter, because TAIDE echoes the chosen option text and sometimes explains despite the instruction. Raw outputs are stored unmodified.
- **GPU ledger:** raised from 1,800 to 6,000 aggregate seconds by owner approval; each run is still capped at 1,800 s. 5,698 s used after this update.
- **TaiwanChat original-vs-reviewed QLoRA** (`trainer/taiwanchat_experiment.py`, dataset `yentinglin/TaiwanChat@d766aa2`, CC-BY-NC-4.0, internal research only): 500 single-turn rows with ≤480 tokens, 250 of them containing terms from the owner-approved list `validation/taiwanchat-terms.json` (38 mappings, approved by cherrie.chen). The reviewed variant applies only those mappings to final answers (787 replacements). Settings: rank 16, alpha 32, lr 2e-4, 2 epochs, 126 steps, ~1,320 GPU-s per run. Train loss was 1.51 for both variants.
- **Evaluation:** 60 held-out TaiwanChat prompts whose reference answers used listed terms, scored with 128-token writing samples (`writing_sample_tokens`, kept and marked truncated), plus the TMMLU+ smoke subset.

| Model | Listed mainland terms / 1k chars | Answers with any listed term | TMMLU+ |
|---|---|---|---|
| Base TAIDE | 4.0 | 32/60 | 8/30 |
| Trained on original data | 7.6 | 37/60 | 8/29 (1 overrun) |
| Trained on reviewed data | 1.6 | 13/60 | 8/30 |

Unreviewed data increased mainland usage; reviewed data reduced it below base, with no TMMLU+ change. Limitations: 60 prompts, one seed and greedy decoding, with no significance test. Prompts drawn from TaiwanChat contain listed terms, so all models partly echo them. Classification and writing answers still need human review.

A bug in the 算法→演算法 replacement also rewrote existing 演算法, which corrupted 15 of the 500 reviewed training rows into 演演算法. The reviewed model emitted 演演算法 once. Self-containing replacements are now excluded automatically, and the scores above use the corrected scoring. The reviewed adapter was not retrained because only 302 GPU-seconds remain.

## Verified software checks

- Final full trainer suite: **156 passed**, 25 FastAPI/Starlette deprecation warnings, 46.68 seconds. Command: `.venv/Scripts/python.exe -m pytest hackathon/trainer -q`.
- Real browser → Node proxy → FastAPI → SQLite: synthetic upload, full audit, individual acceptance, derivative creation, unchanged prompts/metadata, streamed export, independent JSON evaluation, complete coverage and mobile layout. Passed again after restarting the validation backend. Browser requests to external origins were blocked; none were attempted.
- Stream test confirms the first export chunk arrives before the upstream response finishes and preserves the download filename.
- Existing browser tests passed for dataset imports, full-Hub subset selection and failures, refresh/cancellation, model selection, evaluation summary, prompt workflow and pipeline safeguards.
- Compass unit suite: four tests passed; the quality workflow does not call Compass.
- JavaScript syntax checks passed for the new quality UI, existing application and Node server.
- Original baseline before changes: 66 trainer tests passed in the project virtual environment. System Python lacked project dependencies and was not used for final validation.

New subsystem regressions were observed failing before implementation. Independent review reproduced further failures; fixes include bounded diff computation, more complete protected spans, atomic cancellation, queue rejection without abandoned queued jobs, recoverable version publication, cancelled-training guards, managed-version provenance checks, pinned model revisions, and strict comparison gates. Orphaned partial derivatives are preserved in `quality/recovery-orphans`, not deleted. Completed originals are never modified.

Both independent review seats closed all findings within their respective scopes after rechecks, including strict pairing for prompt-trial recommendations and policy attestation for controlled pilot outputs. Final E2E passed again on the last restarted backend. Preview budget endpoint confirmed `chargedGpuSeconds: 0`, `remainingGpuSeconds: 1800`; the pinned 30-question benchmark is visible in the preview.

## Public benchmark preparation

Successfully prepared and verified **TMMLU+ v1.1 smoke subset**: 30 fixed questions, ten each from computer science, finance/banking and official-document management.

- Immutable commit: `466ccd9752ce6889f35c47f875ac7a65012e2269`.
- Annotated tag object: `94d86f1d013d59f389b710a902af30b7a0aa8c8f`; the distinct object ID is expected.
- Cases hash: `842f7fca93cdd4b1653cce9270c938198ab73074b572a74dbc4298b0f72d815c`.
- Manifest checksum: `148433ea475aeb2fd466a1e13367807c1c230c9f5e7965c1451c02a7240f7d38`.
- Preparation was an explicit public-network step. Subsequent loading verifies local manifest and source-file checksums. Material remains separate from trainable datasets.
- All-subject selection is supported in code and mocked preparation tests; only the default three-subject public fixture was prepared live.

This is a ForgeTune deterministic zero-shot protocol with strict A–D parsing, not an official leaderboard reproduction. No model answers or accuracy figures were fabricated.

## Live checks and blockers

| Check | Result |
|---|---|
| Local audit/review/export without CUDA | Passed |
| Saved-output evaluation without training | Passed |
| Public TMMLU+ immutable preparation | Passed |
| Existing VM inventory, read-only | Two idle Tesla T4s, 15,360 MiB total and about 14,918 MiB free each |
| Cached TAIDE weights | Not found locally or in the checked VM cache |
| TAIDE license/gated access | Not accepted or inferred on the user's behalf; access remains unverified |
| Current single-GPU FP16 admission | Requires at least 24,000 MiB free; a T4 does not meet it |
| Actual tokenizer/chat template/response masks | Blocked by missing model assets |
| Base versus original-data versus reviewed-data training | Blocked; no phases launched |
| Actual adapter reload/inference | Blocked |
| Actual GPU deadline/teardown behavior | Not exercised live; mocked/process-level software tests only |
| Aggregate GPU validation consumed | **0 / 1,800 seconds** |
| Adapter deployment | None |

Preflight evidence is saved locally in `.forge-data/quality-pilot-validation-2026-10-05/summary.json`. No new compute was provisioned, unrelated workloads stopped, hosted judging invoked or alternate model silently substituted.

## Deliberate limitations and remaining acceptance work

The current FP16 runner cannot complete the live acceptance test on one available T4. Completion needs authorized cached TAIDE weights and suitable existing single-GPU memory, or a separately approved quantized execution change with its own verification. The frozen synthetic fixture tests structured extraction plumbing; its explicit correction decisions are fixture-author approvals, not invented production review records.

Quality scans are launched per local/imported subset. Existing import-level partial failures remain visible; there is no aggregate collection-level quality audit job. Language fields over 65,536 characters are skipped and prevent a full-coverage correction pass. Peak memory still depends on a single decoded row/Parquet batch. Quotes and code remain protected, including JSON quoted strings, and names need human judgment.

The requested three-tier evaluation concept is only partially covered: native automated checks are implemented; DeepEval/Promptfoo is not integrated. Quality evaluation intentionally does not launch the legacy local judge; an explicit judge panel is not implemented. TMMLU+ smoke support exists, but Taiwan Truthful QA and localized radar charts are not included.

Existing dirty tracked Python cache files and an unrelated pre-existing trailing-space finding remain untouched. The passing test suite does not constitute a clean repository, a deployment, or completion of the blocked live model checks.

See `quality-pilot-operations.md` for usage and `quality-evaluation-report.md` for detailed model/evaluation contracts and test evidence.
