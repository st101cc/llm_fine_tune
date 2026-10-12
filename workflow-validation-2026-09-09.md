# Workflow validation — 9 September 2026

**Result: supported chat workflows pass; SlimOrca needs schema conversion.** Tests ran through the local website and SSH tunnel against the real `dev-01` GPU backend. Training was not started.

## Selection and scope

Python random seed `20260909` selected three datasets from this pool: `nvidia/Nemotron-Cascade-2-SFT-Data` (chat/train), `HuggingFaceH4/ultrachat_200k` (train_sft), `trl-lib/Capybara` (train), `Open-Orca/SlimOrca` (train), and `HuggingFaceH4/no_robots` (train). Each import used the first 30 rows of the selected subset. Dataset selection was random; the imported rows were prefix samples.

| Dataset | Import | Automatic trials | Final checkpoint |
|---|---|---|---|
| [nvidia/Nemotron-Cascade-2-SFT-Data](http://127.0.0.1:4173/#/workflow/86c42ba3-4fbf-4c34-ac0a-01de7ae39f35) | 30 rows | Baseline + 2 prompts completed | Training approval; no training started |
| [HuggingFaceH4/no_robots](http://127.0.0.1:4173/#/workflow/9b2e5add-5c35-457b-97ab-cbb23fb4263c) | 30 rows | Baseline + 2 prompts completed | Training approval; no training started |
| [Open-Orca/SlimOrca](http://127.0.0.1:4173/#/workflow/4b553973-4b56-424c-b136-5cb33636610b) | 30 rows | Blocked: unsupported conversations schema | Data review required |

## Measured prompt diagnostics

Both supported samples were classified as code by the existing heuristic and evaluated with `Qwen/Qwen2.5-Coder-7B-Instruct`. This was not a benchmark-based model choice. Each trial used the same two development rows.

| Dataset | Baseline overlap | Prompt 1 | Prompt 2 | Selected |
|---|---:|---:|---:|---|
| nvidia/Nemotron-Cascade-2-SFT-Data | 0.165 | 0.174 | 0.181 | Test 3 |
| HuggingFaceH4/no_robots | 0.16 | 0.147 | 0.137 | Test 1 |

Both recommendations were **More evidence needed before choosing RAG or fine-tuning**. Selecting the highest overlap trial does not establish a worthwhile improvement.

## Failures found and corrections verified

1. **Review checkpoints failed on the VM’s Python 3.10 runtime.** All three initial runs failed with `Called get_config outside of a runnable context`; two isolated VM tests reproduced the failure. ForgeTune now starts from an isolated Python 3.11 CUDA environment (`.venv311`). The same checkpoint tests and live runs pass. The startup script checks the Python version. LangGraph documents [limitations in async context propagation before Python 3.11](https://docs.langchain.com/oss/python/langgraph/streaming).
2. **GPU memory accumulated between model evaluations.** In the first Python 3.11 run, no_robots completed its baseline but both improved prompt trials failed with CPU/disk dispatch errors. Evaluation now clears model/tensor references, collects reference cycles, and empties the CUDA cache in `finally`, including failure paths. Both baseline and tuned evaluation paths were updated. Two new tests verify cleanup ordering on success and failure. Six real trials across the two datasets then passed consecutively in one backend session.
3. **SlimOrca is not supported directly.** Its `conversations` schema is imported but not normalized into `messages`. Validation correctly pauses with an unsupported-schema blocker; automatic conversion remains unimplemented.

## Checks passed

- VM and local backend suites: **23 passed each** (5 deprecation warnings each).
- Package compatibility check and a real CUDA tensor operation in Python 3.11.
- Both browser regression suites: dataset upload, GitHub/Hugging Face flow, clarification, prompt selection, and mobile layout.
- Real VM results rendered in the browser without JavaScript errors or mobile overflow.
- Live ambiguity test: pauses for clarification, rejects an invalid task value, and persists a valid answer.
- Live duplicate-data test: pauses for review without applying cleanup or starting training.
- First live 30-row split verified: 25 training, 2 prompt-development, 3 held-out evaluation rows; all sets disjoint.
- All final workflows retained their checkpoints after a backend restart. A synthetic clarification checkpoint resumed successfully after restart.
- Both successful workflows reached training-plan approval with no training jobs created.

## Limits

This was a functional smoke test, not a model-quality benchmark. There were only two development examples per supported dataset and generation is capped at 96 tokens with 4096 input tokens. Full-dataset processing, training, tuned-model evaluation, deployment, and actual retrieval were not tested. No independent RAG knowledge corpus is connected.

The backend still retained approximately 7 GB of GPU memory across the two cards after the six-trial run, down from approximately 23 GB at the earlier failure. The rerun demonstrates that these trials complete; it is not a long-running memory stability test. Another VM process used GPU memory and was left untouched. The final persistence check restarted the backend.

## Evidence and rerun

Raw results, before-fix failures, workflow snapshots, scripts, and a browser screenshot are stored in `.forge-data/workflow-validation/`. The scripts create new test datasets and workflows; model evaluation uses VM GPU resources.

```powershell
.venv/Scripts/python.exe -u .forge-data/workflow-validation/run.py
.venv/Scripts/python.exe .forge-data/workflow-validation/run-edges.py
.venv/Scripts/python.exe -m pytest hackathon/trainer -q --disable-warnings
```
