"""Resumable, cancellable original-vs-reviewed comparison job over local T4 GPUs."""
import functools
import json
import math
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from comparison import build_report, estimate, load_pair, plan_split, write_files
from quality_models import TAIDE_ID, get_model_status, load_benchmark
from quality_pilot import budget_status, generate_bounded, idle_gpus, run_with_pilot_budget
from quality_store import identifier

STEPS = ("prepare", "train-original", "train-reviewed", "evaluate-base", "evaluate-original", "evaluate-reviewed", "report")
PHASES = (("prepare",), ("train-original", "train-reviewed"), ("evaluate-base", "evaluate-original"), ("evaluate-reviewed",), ("report",))
GPU_STEPS = {"train", "evaluate"}
TRAIN_SCRIPT = Path(__file__).resolve().parent / "train.py"
WRITING_SAMPLE_TOKENS = 128
MAX_EVAL_ATTEMPTS = 6
GPU_BUSY = "gpu_busy: no idle GPU is available for this step; wait for an idle GPU and start the comparison again."


def now():
    return datetime.now(timezone.utc).isoformat()


def new_record(version_id, settings):
    step = {"status": "pending", "startedAt": None, "completedAt": None, "gpuSeconds": 0, "message": None, "runId": None, "attempts": 0}
    return {"id": str(uuid.uuid4()), "versionId": version_id, "settings": settings, "status": "queued", "createdAt": now(),
            "steps": [{"name": name, **step} for name in STEPS], "split": None, "estimate": None, "report": None}


@functools.lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(get_model_status(TAIDE_ID)["path"], local_files_only=True)


def taide_tokens(row):
    tokenizer = _tokenizer()
    if "prompt_only" in row:
        return len(tokenizer.apply_chat_template([{"role": "user", "content": row["prompt_only"]}], tokenize=True, add_generation_prompt=True))
    messages = row.get("messages")
    if not isinstance(messages, list):
        if not isinstance(row.get("prompt"), str) or not isinstance(row.get("completion"), str):
            return math.inf
        messages = [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["completion"]}]
    return len(tokenizer.apply_chat_template(messages, tokenize=True))


def benchmark_revision(root):
    for path in sorted((Path(root) / "quality" / "benchmarks" / "tmmluplus").glob("*/default-30.json")):
        try:
            load_benchmark(path.parent.name, runtime_root=root)
            return path.parent.name
        except ValueError:
            continue
    raise ValueError("No prepared TMMLU+ benchmark; prepare the default 30-question subset first.")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_done(run_dir):
    try:
        status = json.loads((run_dir / "run.json").read_text(encoding="utf-8")).get("status")
    except (OSError, json.JSONDecodeError):
        return False
    return status == "complete" and (run_dir / "adapter" / "adapter_model.safetensors").is_file()


def run_comparison(store, root, comparison_id, stop, *, gpus=idle_gpus, train=run_with_pilot_budget, generate=generate_bounded, token_count=None):
    root = Path(root).resolve()
    directory = root / "quality" / "comparisons" / identifier(comparison_id)
    record = store.get("comparison", comparison_id)
    if record["status"] in ("complete", "failed", "cancelled"):
        return
    steps = {s["name"]: s for s in record["steps"]}
    run_dir = lambda variant: root / "runs" / steps[f"train-{variant}"]["runId"]

    lock = threading.Lock()

    def save():
        # The cancel API writes `cancelling` to the store concurrently; never overwrite it with `running`.
        with lock, store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM records WHERE kind='comparison' AND id=?", (record["id"],)).fetchone()
            if row and json.loads(row[0]).get("status") == "cancelling" and record["status"] == "running":
                record["status"] = "cancelling"
            store.put("comparison", record, db)

    def ledger_check(kind):
        need = record["estimate"]["stepMaxSeconds"][kind]
        left = budget_status(root)["remainingGpuSeconds"]
        if left < need:
            raise RuntimeError(f"GPU ledger has {math.floor(left)} s left; this step needs {need} s")
        return need

    def prepare(step, gpu):
        settings = record["settings"]
        original, reviewed, _, version = load_pair(store, record["versionId"])
        split = plan_split(original, reviewed, token_count or taide_tokens, train_rows=settings.get("trainRows"),
                           edited_share=settings.get("editedShare", 0.5), eval_prompts=settings.get("evalPrompts", 60))
        plan = estimate(split, split["counts"]["meanTrainTokens"], settings.get("epochs", 2), budget_status(root)["remainingGpuSeconds"])
        if not plan["feasible"]:
            raise ValueError("; ".join(plan["reasons"]))
        model = get_model_status(TAIDE_ID)
        if model["status"] != "ready":
            raise ValueError(model.get("message") or "TAIDE is not ready locally.")
        hashes = write_files(directory, original, reviewed, split)
        record.update(split=split, estimate=plan, files=hashes, modelRevision=model["revision"], benchmarkRevision=benchmark_revision(root),
                      policy=version["provenance"]["policy"], displayName=version["dataset"].get("displayName", version["id"]))

    def train_step(step, gpu):
        variant = step["name"].removeprefix("train-")
        need = ledger_check("train")
        step["runId"] = step["runId"] or str(uuid.uuid4())
        folder = run_dir(variant)
        dataset = lambda name: {"source": "upload", "name": str(directory / name), "extension": "jsonl"}
        hyperparameters = {"rank": 16, "alpha": 32, "learning_rate": 0.0002, "epochs": record["settings"].get("epochs", 2),
                           "batch_size": 1, "gradient_accumulation": 8, "max_sequence_length": 512}
        run = {"id": step["runId"], "status": "running", "createdAt": now(), "startedAt": now(),
               "config": {"dataset": {**dataset(f"{variant}-train.jsonl"), "displayName": f"{record['displayName']} · {variant}"},
                          "base_model": TAIDE_ID, "base_model_revision": record["modelRevision"], "parameter_method": "qlora",
                          "hyperparameters": hyperparameters, "test_dataset": dataset(f"dev-{variant}.jsonl"), "split_seed": 42,
                          "workflow_id": None, "candidate_id": None, "attempt": step["attempts"], "idempotency_key": None,
                          "split_manifest_id": None, "split_manifest_path": None},
               "hardware": {"gpu": gpu}, "metrics": {}, "artifacts": {}, "workflowId": None, "candidateId": None,
               "attempt": step["attempts"], "idempotencyKey": None, "splitManifestId": None, "splitManifestPath": None,
               "qualityProvenance": {"comparisonId": record["id"], "versionId": record["versionId"], "variant": variant},
               "qualityModel": get_model_status(revision=record["modelRevision"]), "managedBy": record["id"], "variant": variant}
        write_json(folder / "run.json", run)
        result = train(root, [[sys.executable, "-u", str(TRAIN_SCRIPT), "--run-dir", str(folder)]], gpu=gpu, cancelled=stop.is_set,
                       max_seconds=need, log_dir=folder, load_mode="bnb4")
        step["gpuSeconds"] += result.get("chargedGpuSeconds") or 0
        done = result["status"] == "complete" and (folder / "adapter" / "adapter_model.safetensors").is_file()
        run.update(status="complete" if done else "cancelled" if result["status"] == "cancelled" else "failed",
                   completedAt=now(), qualityBudget=result, message=result.get("message"))
        if done:
            metrics_path = folder / "metrics.json"
            if metrics_path.is_file():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                loss = metrics.get("eval_loss")
                run["metrics"] = {**metrics, "eval_loss": loss, "perplexity": round(math.exp(loss), 3) if loss is not None else None,
                                  "classification": metrics.get("classification")}
            run["artifacts"] = {"adapter": "adapter/", "mergedModel": "merged/", "ollamaModelfile": "exports/Modelfile"}
        write_json(folder / "run.json", run)
        if not done:
            raise RuntimeError(result.get("message") or f"Training {variant} ended with status {result['status']}.")

    def evaluate_step(step, gpu):
        model = step["name"].removeprefix("evaluate-")
        need = ledger_check("evaluate")
        adapter = None if model == "base" else run_dir(model) / "adapter"
        remaining = load_benchmark(record["benchmarkRevision"], runtime_root=root)["cases"] + read_jsonl(directory / "eval-cases.jsonl")
        answers, over_limit, attempts, failure = [], [], [], None
        while remaining and len(attempts) < MAX_EVAL_ATTEMPTS:
            result = generate(remaining, runtime_root=root, adapter_path=adapter, cancelled=stop.is_set, max_seconds=need,
                              revision=record["modelRevision"], load_mode="bnb4", writing_sample_tokens=WRITING_SAMPLE_TOKENS, gpu=gpu)
            step["gpuSeconds"] += (result.get("run") or {}).get("chargedGpuSeconds") or 0
            attempts.append({k: result.get(k) for k in ("status", "message", "reason")})
            answered = result.get("answers") or []
            answers += answered
            if result["status"] == "complete":
                remaining = []
            elif "token limit" in (result.get("message") or ""):
                # The strict generator refuses truncated text; record the overrun and resume after it.
                over_limit.append(remaining[len(answered)]["id"])
                remaining = remaining[len(answered) + 1:]
            else:
                failure = result.get("message") or f"Evaluation ended with status {result['status']}."
                break
        write_json(directory / "eval" / f"{model}.json", {"model": model, "adapter": str(adapter) if adapter else None, "complete": not remaining,
                                                          "answers": answers, "overLimit": over_limit, "unanswered": [c["id"] for c in remaining], "attempts": attempts})
        if failure or remaining:
            raise RuntimeError(failure or f"Evaluation of {model} stopped after {MAX_EVAL_ATTEMPTS} attempts with {len(remaining)} cases unanswered.")

    def report_step(step, gpu):
        answers = {m: json.loads((directory / "eval" / f"{m}.json").read_text(encoding="utf-8"))["answers"] for m in ("base", "original", "reviewed")}
        losses = {}
        for variant in ("original", "reviewed"):
            metrics = json.loads((run_dir(variant) / "metrics.json").read_text(encoding="utf-8"))
            losses[variant] = metrics.get("train_loss", metrics.get("eval_loss"))
        mcq = load_benchmark(record["benchmarkRevision"], runtime_root=root)["cases"]
        report = build_report(answers, read_jsonl(directory / "eval-cases.jsonl"), mcq, record["policy"], losses)
        record["report"] = {**report, "runs": {v: steps[f"train-{v}"]["runId"] for v in ("original", "reviewed")}}

    handlers = {"prepare": prepare, "train": train_step, "evaluate": evaluate_step, "report": report_step}

    def execute(step, gpu=None):
        step.update(status="running", startedAt=now(), completedAt=None, message=None, attempts=step["attempts"] + 1)
        save()
        try:
            handlers[step["name"].split("-")[0]](step, gpu)
            step.update(status="complete", completedAt=now())
        except Exception as error:
            # A stopped step stays `running`: on shutdown the next start retries it once; on cancel it is closed below.
            if not stop.is_set():
                step.update(status="failed", completedAt=now(), message=str(error))
        save()

    record["status"] = "running"
    for step in record["steps"]:
        if step["name"].startswith("train-") and step["status"] != "complete" and step["runId"] and run_done(root / "runs" / step["runId"]):
            step.update(status="complete", completedAt=step["completedAt"] or now())
        elif step["status"] == "running" and step["attempts"] >= 2:
            step.update(status="failed", completedAt=now(), message="Step was interrupted twice; start a new comparison.")
    save()

    for phase in PHASES:
        if stop.is_set() or record["status"] == "cancelling" or any(s["status"] == "failed" for s in record["steps"]):
            break
        pending = [steps[name] for name in phase if steps[name]["status"] != "complete"]
        if not pending:
            continue
        if pending[0]["name"].split("-")[0] not in GPU_STEPS:
            execute(pending[0])
            continue
        free = gpus()
        if not free:
            pending[0].update(status="failed", completedAt=now(), message=GPU_BUSY)
            save()
        elif len(free) >= 2 and len(pending) == 2:
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(execute, pending, free[:2]))
        else:
            for step in pending:
                execute(step, free[0])
                if step["status"] != "complete" or stop.is_set():
                    break

    save()
    failed = next((s for s in record["steps"] if s["status"] == "failed"), None)
    if failed:
        record.update(status="failed", message=f"{failed['name']}: {failed['message']}")
    elif record["status"] == "cancelling":
        record["status"] = "cancelled"
        for step in record["steps"]:
            if step["status"] == "running":
                step.update(status="cancelled", completedAt=now())
    elif all(s["status"] == "complete" for s in record["steps"]):
        record.update(status="complete", message=None)
    save()
