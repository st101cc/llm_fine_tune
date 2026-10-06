# ForgeTune Traditional Chinese Quality Pilot

Authority: user-approved implementation plan in this conversation (2026-10-05).

## Constraints
- Preserve existing uncommitted work; user chose current folder/new feature branch.
- Public/synthetic data only; no Taiwan residency or production-readiness claim.
- No cloud processing for quality jobs; explicit public asset preparation only.
- Aggregate live GPU validation cap: 1800 GPU-seconds, including loading/failures.
- No automatic adapter deployment or acceptance of gated model licenses.

## Tasks
1. Audit policy, persistent jobs, review, derivative export, regression tests.
2. Independent evaluation, TAIDE/local generation, TMMLU+ preparation and protocol.
3. Browser quality workspace, streaming exports, offline UI and training handoff.
4. Integration/security review, full regression suite, bounded live smoke evidence.

## Shared interfaces
- Audit jobs consume existing local dataset references; new exports use server UUIDs.
- Frozen evaluation cases and candidate answers use stable case IDs.
- Policy module supplies `inspect_text(text, policy)` findings to audit and evaluation.
- Quality data lives in separate runtime SQLite; no legacy run rewrites.

## Progress
- Baseline: project `.venv` passes all 66 original trainer tests. System Python lacked dependencies; it was not used for validation.
- Ruling: preserve dirty working-tree changes and do not auto-commit them; reviewers inspect task-owned additions/patches directly.
- Branch: `codex/traditional-chinese-quality`, current folder per user approval.
- Audit/review/export and independent evaluation routes/UI implemented; original training/import workflows retained.
- First independent audit review found six safety issues; all were addressed with bounded diffing, broader protection, atomic cancellation, explicit queue rejection, and durable version-publication recovery. Regression evidence recorded in `test_quality_hardening.py`.
- Full trainer suite passed 146 tests before the final hardening additions. Live browser→Node→FastAPI→SQLite audit/review/version/export/evaluation passed using synthetic data; mobile layout and no-external-browser-call checks passed.
- Existing import/full-subset/model-selection/evaluation-summary/prompt/safeguard browser checks passed.
- Public TMMLU+ v1.1 preparation succeeded: 30 fixed questions; commit `466ccd9752ce6889f35c47f875ac7a65012e2269` (annotated tag object `94d86f1d013d59f389b710a902af30b7a0aa8c8f`).
- Live preflight: local TAIDE weights missing; existing VM has two idle T4 GPUs (~15 GB each), below this pilot's single-GPU FP16 admission threshold. No model substitution, license acceptance, downloads of gated weights, or GPU execution. GPU charge: 0/1800 seconds.
- Separate validation preview uses ports 4174/18001 and isolated `.forge-data/quality-validation`; existing 4173 app and SSH tunnel remain untouched. This is test infrastructure, not a second production service.
- User's later three-tier panel question: native automated metrics and TMMLU+ support are in scope; DeepEval/Promptfoo, an optional judge panel, Taiwan Truthful QA and radar reporting are not implemented expansions.
- Final verification: 156 trainer tests passed (25 deprecation warnings); final restarted-preview E2E passed. Both independent review seats closed all scoped findings. GPU budget endpoint remains 0/1800 seconds consumed. Live model acceptance remains blocked, not complete; see `quality-pilot-validation.md`.
