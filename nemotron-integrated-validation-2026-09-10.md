# Nemotron general-assistant pilot — 2026-09-10

Workflow: `ce968a6a-eb11-4bc7-92c4-13e01494fe50`<br>
Training run: `1fd4835c-a982-42db-9a0e-c37a6107fe2d`

## Scope and data

This is a bounded general-assistant pilot using **Nemotron training data**, not training the Nemotron model itself. The trainable base is `Qwen/Qwen2.5-3B-Instruct`, selected to fit the available VM. The VM has two Tesla T4 GPUs (15 GB each); training used one GPU.

Source: `nvidia/Nemotron-Cascade-2-SFT-Data`, `chat` configuration, `train` split, revision `9f36020daf067f1a8b39336bf619fe30af30bb02`. Streaming shuffle used seed 20260910 and a 1,000-row buffer. Of 418 scanned conversations, 200 were retained; 211 exceeded the 2,048-token budget and seven were duplicate or previously used questions. All 30 questions from the preceding validation runs were excluded.

The deterministic split contains 170 training conversations, 10 development conversations, and 20 final evaluation conversations. Development data, rather than the final evaluation split, supplies epoch validation. The maximum retained conversation length was 2,047 tokens. This short-conversation sample does not represent all Nemotron tasks or long-context capability.

## Before training

Four actual local trials ran on the same three development questions (row IDs 173, 104, and 4): base prompt, two automatically generated prompt alternatives, and a prompt with retrieval from the connected eight-document corpus. Retrieval matched zero of three questions. This is an insufficient-coverage result, not evidence that RAG is unnecessary.

The bounded pilot selected: “Follow the user's task and requested output format exactly. Give a direct, concise answer and do not invent missing facts.” The final base and tuned comparisons use this identical prompt without retrieval.

## Actual training and artifact checks

QLoRA: rank 16, alpha 32, learning rate 0.0001, two epochs, batch size one, gradient accumulation eight, maximum sequence length 2,048. Training completed all 44 optimizer steps between 04:13:35 and 04:27:40 UTC.

- Trainable parameters: 29,933,568.
- Training loss: 1.6347758743.
- Development loss: 1.4240553379; development token accuracy: 0.6395776868. These are training diagnostics, not general-assistant answer accuracy.
- Training rows truncated: zero; retained token fraction: 1.0.
- Adapter: 119,801,528 bytes, 504 tensors, all finite; all 252 B tensors nonzero.
- Adapter SHA-256: `4c0f2fede6a8db3216e4715db27df7476b47a2155a11153eb876395b2b48ffda`.

## Website integration

The dataset workflow now includes recorded prompt/RAG trials, reference uploads, retrieved-source coverage, actual training records, token-retention diagnostics, and Compass comparisons. Compass credentials remain in the local server environment. Comparisons persist per workflow and phase; resuming skips completed answers. One or two available models can be compared on up to nine development questions, with final questions gated until workflow completion. Each answer is capped at 256 output tokens.

The live model catalog returned 261 models. A new three-question generation test has not been sent: automatic approval review requires fresh authorization because the preceding approval covered different questions. Mock gateway and browser tests exercise the integration without transmitting data.

Correctness fixes additionally keep incomplete final results out of public workflow responses, retain the references belonging to a selected RAG trial, and bound retrieved context so it does not displace the user question. The small-corpus retriever retains document prefixes under the token budget; passage retrieval remains a limitation for larger documents.

## Final quality evaluation

The paired evaluation completed on identical 20-question IDs with a shared 512-output-token budget. Both model calls succeeded. The workflow correctly returned `review_required` with no winner and no measured task-accuracy score.

| Diagnostic | Base | Fine-tuned |
|---|---:|---:|
| Answer word overlap with dataset reference | 0.132 | 0.167 |
| Answers reaching output limit | 2/20 | 5/20 |
| Recorded evaluation time | 290.3 s | 805.1 s |

Word overlap is not accuracy. The evaluator's 4.2376 versus 2.3296 loss is computed over input tokens, not held-out answer targets, and must not be interpreted as answer correctness. Timings include different model/adapter overhead and are not a controlled serving benchmark.

An unblinded qualitative review inspected all 20 pairs using rubric dimensions defined before outputs were available. It found eight regressions, two mixed changes, nine ties (including shared failures or partial answers), and one malformed-conversation exclusion. These judgments are diagnostic, not a validated win rate or production accuracy estimate.

Concrete findings:

- Row 1: tuned loops in a reasoning block and never produces the requested Spanish answer.
- Row 9: tuned adds Markdown fencing to a JSON-only task. Both invent a website absent from the paragraph; the required exact date is underspecified.
- Row 58: tuned invents a 2015 founding year and other company details absent from the input.
- Row 99: both miss an explicit passage fact identifying two characters as brothers.
- Row 116: tuned adds bold formatting despite a single-letter-only instruction.
- Row 184: both appropriately request the missing idea.
- Row 164 is excluded because the imported conversation ends in consecutive assistant turns, including a reasoning-only turn.

A post-run content audit found reasoning tags in 155 of 170 training targets, eight of ten development targets, and 19 of 20 final references. This is a likely contributor to learned reasoning markup and inflated overlap; causal confirmation requires a controlled follow-up, not a claim from this one run.

**Decision: retain the adapter for inspection; do not deploy it as an improvement.** The run proves training, artifact verification, paired inference, persistence, and website results work. It does not prove improved general-assistant precision or accuracy.

Before another quality attempt, normalize reasoning-tagged assistant targets for the intended answer style, repair or reject malformed turn sequences, resolve underspecified extraction labels, and validate a larger task-balanced development set. Any model selection using these observed final answers requires a fresh untouched final test set for a new improvement claim.

## Verification

33 backend tests passed locally and on the VM after the review fixes. Four Compass tests and three browser suites passed. The live workflow page loaded without browser errors or horizontal overflow. VM backend restart occurred only after training and final evaluation completed; saved results survived the restart.

Local evidence is in `.forge-data/nemotron-integrated/`: preparation and content audits, workflow checkpoint, paired outputs, adapter integrity audit, and manual review. Raw examples remain local for inspection; no new Compass prompts were sent without fresh approval.
