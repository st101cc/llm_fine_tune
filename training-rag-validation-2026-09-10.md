# Actual training, prompt, and RAG validation — 10 September 2026

**All three sample fine-tunes completed on dev-01. None demonstrated a clear correctness improvement.** The local comparison produced 63 answers across two base models, automatic prompt changes, independent retrieval, and three trained adapters. Compass authentication now works and returned 261 model IDs. The approved API comparison added 17 answers: all nine from gpt-4.1-mini and eight from Qwen2.5-32B-Instruct; the final Qwen question remained rate-limited.

## Training actually ran

The same three randomly selected dataset samples were used: nvidia/Nemotron-Cascade-2-SFT-Data (chat), HuggingFaceH4/no_robots, and Open-Orca/SlimOrca. Each contains the first 30 rows; dataset selection used seed 20260909, while the split used seed 42. Each split has 25 training rows, 2 prompt-development rows, and 3 held-out rows (10, 14, 19). The sets are disjoint. SlimOrca was losslessly converted from `conversations` roles to `messages` in a separate dataset version. Direct automatic SlimOrca conversion remains outside the import workflow.

The trained base was Qwen/Qwen2.5-Coder-7B-Instruct, using 4-bit QLoRA, rank 8, alpha 16, learning rate 0.0001, one epoch, batch size 1, gradient accumulation 8, and a 512-token training window. These are deliberately small functional pilots. The VM has two 15 GB Tesla T4 GPUs; training selects one GPU, rather than combining their memory.

| Dataset | Optimizer steps | Train loss | Evaluation loss | Training rows truncated | Training tokens retained |
|---|---:|---:|---:|---:|---:|
| nemotron | 4 | 2.1655 | 2.0084 | 22/25 | 15.8% |
| no_robots | 4 | 2.5992 | 3.2298 | 5/25 | 91.3% |
| slimorca | 4 | 2.1286 | 2.2310 | 7/25 | 86.0% |

Each run had **20,185,088 trainable parameters** and finite gradients/loss. All three saved adapters were reloaded successfully; every one of their 196 LoRA B tensors contained nonzero weights, and all saved weights were finite. Adapter hashes and metrics are retained in the raw evidence. Lower token loss alone is not evidence that answers became more accurate.

## Comparison method

The independent quality probe used the same three held-out questions per dataset, greedy decoding, up to 4096 input tokens and 256 output tokens for every arm. It compared the original selected prompt, a concise general-purpose prompt, original prompt plus retrieved reference material, and the original prompt with the saved adapter. The general-purpose prompt was tested after observing initial failures, so its results are exploratory; a new untouched test set is needed before selecting it for production.

The second local model was [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), evaluated with base, improved-prompt, and RAG arms. Its model card uses the Qwen Research license; this was a local evaluation, not a deployment. The two model families and all arms used the same generation budget.

RAG used BM25 over eight short, manually curated summaries from primary references, selected from question topics without using the gold answers. It retrieved at most two documents and abstained when fewer than two meaningful terms matched. The final assistant answer was removed before retrieval and generation. This is a **small, curated retrieval experiment**, not a production knowledge index or a feature already connected to the website workflow.

## Observed answer quality

These heterogeneous chat examples do not support a single meaningful classification precision score. The observations below inspect actual answers against the task, including factual errors and truncation; they are not an automated LLM-judge score.

| Example | Observation |
|---|---|
| Nemotron: ambiguous gender question | The retriever missed the relevant document. Several outputs made unsupported medical generalizations instead of resolving ambiguity. Further review is required. |
| Nemotron: Ohio partnership scenario | Retrieval introduced relevant statutes, but answers remained incomplete or overconfident. Some 3B answers invented that Julio signed the contract. No arm is acceptable as a reliable answer to the full scenario. |
| Nemotron: LCM of 240 and 630 | The correct answer is 5040. The 7B responses exhausted the token budget without delivering the requested answer. 3B with the concise prompt returned 1260; 3B with RAG returned 15120. Correct boxing did not mean correct mathematics. |
| no_robots: Aberdeen | RAG explained the use of local granite in buildings. The 7B base and adapter gave the same less direct industry-focused answer; the revised 7B prompt added an unsupported historical date. The concise 3B prompt and its RAG arm gave a direct explanation. |
| no_robots: French translation | The 7B base, prompt, RAG, and adapter all wrote `Je admire` rather than the required elision. The 3B model wrote `J’admire` and preserved the intended meaning. |
| no_robots: article summary | The 7B base and adapter conveyed substantially the same summary. The concise prompt shortened it without needing retrieval. No extra context was retrieved. |
| SlimOrca: sliding a box | The 7B base and adapter both incorrectly said a slippery floor could make pushing harder. Both the prompt change and RAG corrected the explanation. The 3B base also gave the correct relationship. |
| SlimOrca: fossil-fuel sequence | The 7B base and adapter incorrectly selected liquefaction as the first step. Prompt/RAG changed the answer to burial. The question is ambiguous about whether it asks for the first listed step or a missing prior step; review the reference before using this as a training target. |
| SlimOrca: winter-driving passage | All arms identified option D. Retrieval was unnecessary; it also returned an irrelevant oil document. The concise prompt reduced answer length. |

Retrieval found a relevant source for 6 of the 7 questions with topical coverage in the curated corpus; it correctly abstained on both questions without a matching source. Of eight documents retrieved overall, seven were relevant. These small-corpus diagnostics do not establish production retrieval precision.

The adapters did not repair the observed grammar, arithmetic, or friction errors. The original workflow retained the base model for all three runs. Under the corrected gate, these stored free-form results require review because they lack scored task correctness. Historical run records retain their original decisions.

## Fixes implemented and checked

- Updated the trainer for the installed TRL interface (`SFTConfig`, `processing_class`, `max_length`) and constrained compatible TRL/Transformers versions.
- Passed the base model and LoRA configuration to TRL together, preventing all adapter parameters from being frozen. Added a trainable-parameter guard and actual optimizer/row metrics.
- Moved workflow inference into short-lived processes so GPU contexts are released before training. Training chooses a free GPU before CUDA initializes.
- Converted training input to canonical text columns before TRL format detection. This fixed no_robots metadata being mistaken for a prompt/completion dataset.
- Replaced the code detector’s ordinary-word matches with code syntax. All three real samples now classify as assistant data; the candidate heuristic selects a general-purpose model. This heuristic is still not a benchmark result.
- Added ground-truth classification accuracy, precision, recall, F1, and error records to both base and tuned evaluation, plus structured JSON exact match. Fixed label substring collisions such as safe/unsafe. Free-form correctness remains unmeasured without a task rubric.
- Fine-tune recommendations now require a task-score gain of at least five percentage points on the same fully scored held-out examples, with no measured format regression. Word overlap and lower prompt loss cannot qualify a model. This only requests review; it does not deploy.
- Increased workflow generation from 96 to 512 tokens and recorded limit hits. The controlled quality probe intentionally kept its original 256-token limit for comparability.
- Displayed task correctness separately from word overlap and labelled successful artifacts as trained rather than eligible. The backend also checks a `.env` in its startup directory after its trainer-local file.

Verification: **29 backend tests passed locally and on the VM**, including process isolation, mixed-column training data, label boundaries, and rejection of overlap-only quality claims. Both browser regression suites passed. The three live workflows completed, adapters reloaded, and the website/API were reachable after restart.

## Compass comparison

The corrected root `.env` successfully authenticated to the [Compass Models API](https://compass.llm.shopee.io/docs/API_REFERENCE/models), which returned **261 model IDs**. Earlier 401 failures are resolved. The catalog includes text, image, audio, and embedding models; listing an ID does not verify generation access or suitability. The complete returned list is saved in `.forge-data/training-rag-validation/compass-model-ids.json`.

After explicit approval, the probe compared `Qwen2.5-32B-Instruct` (a larger general-purpose Qwen baseline) and `gpt-4.1-mini` (a different provider baseline) on the same nine public questions, using the same concise instruction, temperature 0, and a maximum of 256 output tokens. It retained original nonempty system/chat context and omitted the gold final assistant answer. Compass rejects empty system messages, so the probe removes those. No RAG context or trained adapter was sent to these API models. These hosted models differ from the local quantized models in size and serving configuration; this is an exploratory baseline comparison, not an isolated measurement of fine-tuning effects or a search across all 261 models.

| Model | Answers completed | Answers hitting 256-token limit | Remaining |
|---|---:|---:|---|
| gpt-4.1-mini | 9/9 | 1 | None |
| Qwen2.5-32B-Instruct | 8/9 | 1 | SlimOrca row 19: HTTP 429 after retries |

The probe preserves completed answers on resume and backs off after rate limits. Qwen's final request remained blocked by `42900: Rate limit exceeded`; it has no answer to score. Recorded request durations include retry waits and must not be interpreted as model inference latency. The credential was not printed, rewritten, or transferred to the VM, and no model was deployed.

| Example | Observed API answer quality |
|---|---|
| Nemotron: ambiguous gender question | GPT asked which interpretation the user intended. Qwen assumed gender transition. This is a useful difference in clarification behavior, not an objective factual score for the ambiguous question. |
| Nemotron: Ohio partnership | Both answers hit the token limit before covering both concerns and supplied unreliable legal citations. Neither is an acceptable complete answer. |
| Nemotron: LCM | Both returned the correct result, **5040**, within the cap. The prior local arms had failed to provide a correct completed result. |
| no_robots: Aberdeen | Both correctly linked the nickname to locally quarried granite used in the city's buildings, without the earlier unsupported historical detail. |
| no_robots: French translation | Both used grammatical `J'admire` and `les Francais` (with the appropriate French cedilla in the saved outputs), avoiding the local 7B model's `Je admire` error. |
| no_robots: article summary | Both gave concise summaries of the supplied article. Neither added the prior local 3B output's spurious numerical claim. |
| SlimOrca: sliding box | Both identified weight and surface friction and correctly described smoother surfaces as easier to slide across. |
| SlimOrca: fossil-fuel sequence | Both selected burial, matching the stored reference. The question's ambiguity remains; Qwen added a broad explanation that should not be treated as a validated training target. |
| SlimOrca: winter-driving passage | GPT correctly selected D. Qwen's answer is unavailable due to the rate limit. |

The API baselines repair several observed failures without additional training. That supports comparing stronger base models before investing in another fine-tune; it does **not** establish a production accuracy score or show that fine-tuning is unnecessary for a defined task. These nine heterogeneous examples are too small and informal for an aggregate precision claim. The app's existing advisor hook still assists dataset analysis only; this external-model benchmark was run separately and is not automatically integrated into the website workflow.

## Evidence and rerun

Raw workflows, adapter hashes, training metrics, all 63 local generated answers, 17 Compass answers, retrieval references, and the prepared nine-question Compass input are in `.forge-data/training-rag-validation/`. `quality-summary.json` is the compact numeric record; the `quality-*.json` files retain individual outputs. The reproducible local probe is `trainer/quality_probe.py`; the curated corpus is `validation/rag-corpus.json`.

Successful run IDs:

- nemotron: `fb6d6292-948a-411f-a5f0-a30728fcb5d2`
- no_robots: `2825e8f4-3370-42b7-9720-f438d5f03392`
- slimorca: `3deab8b2-9a8b-4c17-a97e-a712e42e97e2`

For a meaningful next fine-tune, choose one production task, validate its target answers, preserve complete training targets, and evaluate on a substantially larger untouched set against the best base/prompt/RAG alternative. These 25-row pilots do not establish that a larger fine-tune will improve accuracy.
