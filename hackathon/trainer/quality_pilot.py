"""Explicit offline synthetic pilot with a shared 1800 single-GPU-second budget.

Parent owns timing and terminates only its worker process tree. Workers are
sequential, CPU data loading uses no subprocesses, and CUDA sees one GPU.
Synthetic fixtures test plumbing, not Taiwanese language benchmark quality.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from quality_models import TAIDE_ID


def _runtime_directory(runtime_root):
    root = Path(runtime_root).resolve()
    directory = (root / "quality").resolve()
    if not directory.is_relative_to(root):
        raise ValueError("Quality runtime directory escapes its root")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@contextmanager
def _budget_db(runtime_root):
    directory = _runtime_directory(runtime_root)
    path = (directory / "gpu-budget.sqlite").resolve()
    if not path.is_relative_to(directory):
        raise ValueError("GPU budget path escapes runtime area")
    with sqlite3.connect(path, timeout=10) as db:
        db.execute("CREATE TABLE IF NOT EXISTS reservations(id TEXT PRIMARY KEY, charge REAL NOT NULL, status TEXT NOT NULL)")
        yield db


def budget_status(runtime_root):
    """Persistent aggregate budget; unfinished reservations remain fully charged."""
    with _budget_db(runtime_root) as db:
        charged = db.execute("SELECT COALESCE(SUM(charge),0) FROM reservations").fetchone()[0]
    return {"limitGpuSeconds": 1800, "chargedGpuSeconds": charged, "remainingGpuSeconds": max(0, 1800 - charged)}


def _reserve_budget(runtime_root, seconds):
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not math.isfinite(seconds) or not 0 < seconds <= 1800:
        raise ValueError("Requested GPU seconds must be > 0 and <= 1800")
    with _budget_db(runtime_root) as db:
        db.execute("BEGIN IMMEDIATE")
        charged = db.execute("SELECT COALESCE(SUM(charge),0) FROM reservations").fetchone()[0]
        allowance = min(seconds, max(0, 1800 - charged))
        token = str(uuid.uuid4()) if allowance else None
        if token:
            db.execute("INSERT INTO reservations VALUES(?,?,?)", (token, allowance, "reserved"))
    return {"id": token, "reservedGpuSeconds": allowance}


def _settle_budget(runtime_root, reservation, charged):
    if not math.isfinite(charged) or charged < 0:
        raise ValueError("Invalid measured GPU charge")
    with _budget_db(runtime_root) as db:
        db.execute("UPDATE reservations SET charge=?,status='settled' WHERE id=? AND status='reserved'", (charged, reservation["id"]))


def _select_idle_gpu(gpu=None):
    try:
        if gpu is not None and (not isinstance(gpu, str) or not re.fullmatch(r"(?:[0-9]+|GPU-[A-Za-z0-9-]+)", gpu)):
            raise ValueError("Select one GPU")
        command = ["nvidia-smi"] + (["-i", gpu] if gpu is not None else []) + ["--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"]
        inventory = subprocess.check_output(command, text=True, timeout=5)
        free_but_small = False
        busy = False
        for row in inventory.splitlines():
            device, free = [part.strip() for part in row.split(",")]
            if not re.fullmatch(r"GPU-[A-Za-z0-9-]+", device):
                continue
            processes = subprocess.check_output(["nvidia-smi", "-i", device, "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=5).strip()
            if processes:
                busy = True
                continue
            if int(free) >= 24000:
                return device, None, None
            free_but_small = True
        if free_but_small:
            return None, "insufficient_vram", "An idle GPU with at least 24000 MiB free VRAM is required for local FP16 TAIDE; free memory or use a larger GPU."
        if busy:
            return None, "gpu_busy", "All available GPUs have unrelated compute processes. Wait for an idle GPU; no process was stopped."
        return None, "no_gpu", "No usable local NVIDIA GPU was found. Configure a CUDA host before generation."
    except (OSError, ValueError, subprocess.SubprocessError):
        return None, "gpu_unavailable", "Cannot inspect GPU availability. Check NVIDIA drivers and nvidia-smi; no GPU was selected."


def run_with_pilot_budget(root, commands, *, gpu=None, cancelled=lambda: False, max_seconds=1800, log_dir=None):
    """Server-owned argument lists only; shares the generation budget ledger.

    Does not alter training parameters or permit network/model preparation.
    Never expose arbitrary command construction directly to an API client.
    """
    if isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float)) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 1800:
        raise ValueError("max_seconds must be > 0 and <= 1800")
    commands = list(commands)
    if any(not isinstance(command, (list, tuple)) or not command or any(not isinstance(part, str) for part in command) for command in commands):
        raise ValueError("Commands must be argument lists, never shell strings")
    def response(status, message, **extra):
        return {"status": status, "phases": [], "chargedGpuSeconds": 0, "message": message, **extra, "budget": budget_status(root)}
    if cancelled():
        return response("cancelled", "Cancelled before launching a worker.")
    if budget_status(root)["remainingGpuSeconds"] <= 0:
        return response("budget_exhausted", "Persistent 1800 GPU-second budget exhausted.")
    device, reason, message = _select_idle_gpu(gpu)
    if device is None:
        return response("unavailable", message, reason=reason)
    reservation = _reserve_budget(root, max_seconds)
    if not reservation["reservedGpuSeconds"]:
        return response("budget_exhausted", "Persistent 1800 GPU-second budget exhausted.")
    charged = reservation["reservedGpuSeconds"]
    try:
        try:
            result = run_budgeted(commands, seconds=charged, gpu=device, cancelled=cancelled, log_dir=log_dir)
        except RuntimeError as exc:
            # Only admission failures occur before run_budgeted starts workers.
            charged = 0
            _settle_budget(root, reservation, charged)
            return response("unavailable", str(exc), reason="gpu_busy")
        charged = result["chargedGpuSeconds"]
        _settle_budget(root, reservation, charged)
        return {**result, "message": result.get("message", "Bounded local command run " + result["status"] + "."), "budget": budget_status(root)}
    finally:
        _settle_budget(root, reservation, charged)


def generate_bounded(cases, *, runtime_root, model_id=TAIDE_ID, adapter_path=None, cancelled=lambda: False, max_seconds=1800, revision=None):
    """Route-safe offline inference: reserved persistent budget + owned subprocess.

    Status reads do not infer license acceptance. No download or training occurs.
    24000 MiB is a conservative admission threshold, not a guarantee against OOM.
    Input plus output is limited to 4096 tokens by the generation worker.
    """
    from quality_models import get_model_status
    cases = list(cases)
    if not cases or len(cases) > 1000 or any(not isinstance(c, dict) or not isinstance(c.get("id"), str) or not c["id"] or not isinstance(c.get("input"), str) or len(c["input"]) > 65536 or c.get("task") not in ("classification", "json", "writing", "mcq") for c in cases):
        raise ValueError("Generation requires 1-1000 valid cases, each input at most 65536 characters")
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate generation case IDs")
    if isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float)) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 1800:
        raise ValueError("max_seconds must be > 0 and <= 1800")
    def response(status, message, **extra):
        return {"status": status, "answers": [], "message": message, **extra, "budget": budget_status(runtime_root)}
    if cancelled():
        return response("cancelled", "Generation cancelled before launch.")
    if budget_status(runtime_root)["remainingGpuSeconds"] <= 0:
        return response("budget_exhausted", "The persistent 1800 GPU-second budget has been exhausted.")
    status = get_model_status(model_id, **({'revision': revision} if revision else {}))
    if status["status"] != "ready":
        return response("unavailable", status["message"], reason=status["status"], model=status)
    directory = _runtime_directory(runtime_root) / "generation" / str(uuid.uuid4())
    if not directory.resolve().is_relative_to(_runtime_directory(runtime_root)):
        raise ValueError("Generation directory escapes runtime area")
    directory.mkdir(parents=True)
    request = {"cases": cases, "modelId": model_id, "revision": status["revision"], "adapterPath": str(Path(adapter_path).resolve()) if adapter_path else None}
    (directory / "request.json").write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--output", str(directory), "--model", model_id, "--revision", status["revision"], "--worker", "generate"]
    run = run_with_pilot_budget(runtime_root, [command], max_seconds=max_seconds, cancelled=cancelled, log_dir=directory)
    if run["status"] in ("cancelled", "budget_exhausted", "unavailable"):
        return response(run["status"], run["message"], reason=run.get("reason"), run=run)
    result_path = directory / "answers.json"
    if run["status"] != "complete" or not result_path.is_file():
        return response("unavailable", "Local generation worker failed. Inspect its local phase-0.log; verify CUDA, Transformers, PEFT (for adapters), and available VRAM.", reason="worker_failed", run=run)
    generated = json.loads(result_path.read_text(encoding="utf-8"))
    return response(generated["status"], generated["message"], **{key: value for key, value in generated.items() if key not in ("status", "message")}, run=run)


def training_parameters():
    return {"max_steps": 8, "rank": 8, "alpha": 16, "learning_rate": 0.0001, "max_sequence_length": 512, "seed": 42, "batch_size": 1, "gradient_accumulation_steps": 1}


def fixtures():
    """Frozen synthetic structured extraction; clean targets derive from decisions.

    Decisions are fixture-author approvals, not claims of independent human
    review or records from the production audit store. No heuristic policy
    correction is performed. Keys, IDs, quantities and prompts are unchanged.
    """
    entries = {
        "train": [("TW101", "鉛筆", "铅笔", 2), ("TW102", "電腦", "电脑", 1), ("TW103", "雨傘", "雨伞", 4), ("TW104", "書包", "书包", 3), ("TW105", "紙袋", "纸袋", 9), ("TW106", "紅茶", "红茶", 5), ("TW107", "綠豆", "绿豆", 7), ("TW108", "鐵盒", "铁盒", 6)],
        "development": [("TW201", "棉被", "棉被", 8), ("TW202", "白米", "白米", 2)],
        "test": [("TW301", "毛巾", "毛巾", 11), ("TW302", "水杯", "水杯", 3), ("TW303", "鉛筆", "鉛筆", 12), ("TW304", "電腦", "電腦", 2)],
    }
    partitions = {}
    for split, values in entries.items():
        rows = []
        for order_id, item, original_item, quantity in values:
            reference = json.dumps({"order_id": order_id, "item": item, "quantity": quantity}, ensure_ascii=False, separators=(",", ":"))
            original = json.dumps({"order_id": order_id, "item": original_item, "quantity": quantity}, ensure_ascii=False, separators=(",", ":"))
            row = {"id": f"{split}-{order_id}", "input": f"請擷取訂單資料，只輸出 JSON，欄位為 order_id、item、quantity。訂單 {order_id} 訂購 {item}，數量為 {quantity}。", "task": "json", "reference": reference}
            if split == "train":
                start = original.index(original_item)
                row.update({"original": original, "reviewDecisions": [{"field": "completion", "start": start, "end": start + len(original_item), "original": original_item, "replacement": item, "decision": "accept", "reviewer": "synthetic-fixture-author", "reason": "Unambiguous script correction in the item value; preserve the stated item, ID, quantity and JSON keys."}]})
            rows.append(row)
        partitions[split] = rows
    return partitions


def _reviewed_output(row):
    text = row["original"]
    previous_start = len(text)
    for decision in sorted(row["reviewDecisions"], key=lambda d: d["start"], reverse=True):
        start, end = decision["start"], decision["end"]
        if decision["decision"] != "accept" or decision["field"] != "completion" or not 0 <= start < end <= previous_start or text[start:end] != decision["original"]:
            raise ValueError("Invalid or overlapping fixture review decisions")
        text = text[:start] + decision["replacement"] + text[end:]
        previous_start = start
    if json.loads(text) != json.loads(row["reference"]):
        raise ValueError("Reviewed fixture does not match its frozen extraction reference")
    return text


def validate_partitions(partitions):
    """Recheck IDs, normalized prompts and extracted order IDs after review."""
    seen_ids, seen_inputs, seen_orders = set(), set(), set()
    for split in ("train", "development", "test"):
        if not partitions.get(split):
            raise ValueError("Each frozen partition must be nonempty")
        for row in partitions[split]:
            if row["task"] != "json":
                raise ValueError("Pilot partitions must all use structured extraction")
            reference = _reviewed_output(row) if split == "train" else row["reference"]
            order = json.loads(reference)["order_id"]
            prompt = " ".join(row["input"].split())
            if row["id"] in seen_ids or prompt in seen_inputs or order in seen_orders:
                raise ValueError("Frozen partitions overlap after corrections")
            seen_ids.add(row["id"])
            seen_inputs.add(prompt)
            seen_orders.add(order)


def training_rows(variant, split="train"):
    if variant not in ("original", "cleaned") or split != "train":
        raise ValueError("Only fixed original/cleaned training partitions are trainable")
    partitions = fixtures()
    validate_partitions(partitions)
    return [{"messages": [{"role": "user", "content": row["input"]}, {"role": "assistant", "content": row["original"] if variant == "original" else _reviewed_output(row)}]} for row in partitions[split]]


def _arm_worker_deadline(deadline):
    """Independent worker cutoff survives a lost parent; no child data loaders."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        os._exit(124)
    watchdog = threading.Timer(remaining, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()


@contextmanager
def _gpu_lease(uuid):
    """OS lock shared by pilot processes; never signal unrelated GPU owners."""
    from hashlib import sha256
    path = Path(tempfile.gettempdir()) / ("quality-gpu-" + sha256(uuid.encode()).hexdigest() + ".lock")
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("GPU is already leased by another quality pilot") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _gpu_uuid(gpu):
    try:
        rows = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=5).strip().splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("Cannot inspect local GPU; install/configure nvidia-smi before the pilot") from exc
    if len(rows) != 1 or not rows[0].startswith("GPU-"):
        raise RuntimeError("Select exactly one physical CUDA GPU")
    return rows[0].strip()


def _assert_idle(uuid):
    try:
        pids = subprocess.check_output(["nvidia-smi", "-i", uuid, "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("Cannot inspect GPU processes; refusing to run") from exc
    if pids:
        raise RuntimeError("GPU is busy with other processes; choose an idle GPU. No unrelated process will be stopped.")


def _kill_owned_tree(process):
    if os.name == "nt":
        # PID is obtained from this Popen, never from the GPU process inventory.
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2, check=False, creationflags=subprocess.CREATE_NO_WINDOW)
        finally:
            if process.poll() is None:
                process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=5)


def run_budgeted(commands, *, seconds=1800, gpu="0", cancelled=lambda: False, log_dir=None):
    """Charge elapsed wall time conservatively as seconds on one exclusive GPU.

    Includes startup/load, failures, gaps and termination. The deadline signals
    immediate termination, not a cooperative trainer callback. OS teardown
    latency is reported (never clipped to the requested cap).
    """
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0 < seconds <= 1800:
        raise ValueError("GPU-second cap must be > 0 and <= 1800")
    if not isinstance(gpu, str) or not re.fullmatch(r"(?:[0-9]+|GPU-[A-Za-z0-9-]+)", gpu):
        raise ValueError("Select one GPU, not a list")
    result = {"status": "complete", "budgetGpuSeconds": seconds, "chargedGpuSeconds": 0.0, "phases": [], "accounting": "conservative-single-GPU-wall-time-including-load-and-failure"}
    if cancelled():
        return {**result, "status": "cancelled"}
    uuid = _gpu_uuid(gpu)
    with _gpu_lease(uuid):
        _assert_idle(uuid)
        started = time.monotonic()
        deadline = started + seconds
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": uuid, "QUALITY_WORKER_DEADLINE": str(deadline), "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "WANDB_DISABLED": "true", "PYTHONDONTWRITEBYTECODE": "1"}
        for index, command in enumerate(commands):
            if cancelled() or time.monotonic() >= deadline:
                result["status"] = "cancelled" if cancelled() else "budget_exhausted"
                break
            if index:
                try:
                    _assert_idle(uuid)
                except RuntimeError as exc:
                    result.update({"status": "unavailable", "reason": "gpu_busy", "message": str(exc)})
                    break
            log = None
            process = None
            phase_started = time.monotonic()
            phase = {"index": index, "exitCode": None}
            result["phases"].append(phase)
            try:
                if log_dir:
                    log = (Path(log_dir) / f"phase-{index}.log").open("w", encoding="utf-8")
                options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
                if time.monotonic() >= deadline:
                    result["status"] = "budget_exhausted"
                    break
                supervised = [sys.executable, "-B", str(Path(__file__).resolve()), "--supervise", *command]
                process = subprocess.Popen(supervised, env=env, stdout=log or subprocess.DEVNULL, stderr=subprocess.STDOUT, **options)
                while process.poll() is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or cancelled():
                        result["status"] = "budget_exhausted" if remaining <= 0 else "cancelled"
                        _kill_owned_tree(process)
                        break
                    try:
                        process.wait(timeout=min(0.05, remaining))
                    except subprocess.TimeoutExpired:
                        pass
                phase["exitCode"] = process.returncode
                if result["status"] == "complete" and process.returncode != 0:
                    result["status"] = "budget_exhausted" if process.returncode == 124 else "failed"
            except OSError as exc:
                phase["error"] = str(exc)
                result["status"] = "failed"
            finally:
                if process is not None and process.poll() is None:
                    _kill_owned_tree(process)
                if log:
                    log.close()
                phase["chargedGpuSeconds"] = time.monotonic() - phase_started
                result["chargedGpuSeconds"] = time.monotonic() - started
            if result["status"] != "complete":
                break
        result["chargedGpuSeconds"] = time.monotonic() - started
        if result["status"] == "complete" and result["chargedGpuSeconds"] >= seconds:
            result["status"] = "budget_exhausted"
        result["terminationOverrunSeconds"] = max(0, result["chargedGpuSeconds"] - seconds)
        result["gpu"] = uuid
    return result


def _train_worker(output, model_id, variant, revision):
    from quality_models import get_model_status
    status = get_model_status(model_id, revision=revision)
    if status["status"] != "ready":
        raise RuntimeError(status["message"])
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from train import training_text_dataset
    from trl import SFTConfig, SFTTrainer
    hp = training_parameters()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Pilot worker requires exactly one visible CUDA GPU")
    set_seed(hp["seed"])
    tokenizer = AutoTokenizer.from_pretrained(status["path"], local_files_only=True, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    # Reuse validated response masks, but never allow the existing helper to truncate.
    data = training_text_dataset(Dataset.from_list(training_rows(variant)), tokenizer, max_length=None)
    if any(length > hp["max_sequence_length"] for length in data["token_count"]):
        raise ValueError("Training would truncate a fixture; refusing the run")
    data = data.remove_columns("token_count")
    model = AutoModelForCausalLM.from_pretrained(status["path"], local_files_only=True, trust_remote_code=False, device_map={"": 0}, torch_dtype=torch.float16)
    model.config.use_cache = False
    adapter = LoraConfig(r=hp["rank"], lora_alpha=hp["alpha"], lora_dropout=0.05, target_modules="all-linear", task_type=TaskType.CAUSAL_LM)
    directory = output / variant
    trainer = SFTTrainer(model=model, peft_config=adapter, processing_class=tokenizer, train_dataset=data,
        args=SFTConfig(output_dir=str(directory), max_steps=hp["max_steps"], max_length=hp["max_sequence_length"], learning_rate=hp["learning_rate"], per_device_train_batch_size=hp["batch_size"], gradient_accumulation_steps=hp["gradient_accumulation_steps"], seed=hp["seed"], data_seed=hp["seed"], completion_only_loss=True, gradient_checkpointing=True, fp16=True, bf16=False, optim="adamw_torch", save_strategy="no", eval_strategy="no", logging_steps=1, dataloader_num_workers=0, report_to=[]))
    trained = trainer.train()
    if trainer.state.global_step != 8:
        raise RuntimeError("Pilot did not complete exactly eight optimizer steps")
    trainer.save_model(str(directory / "adapter"))
    (directory / "training.json").write_text(json.dumps({"parameters": hp, "steps": trainer.state.global_step, "loss": trained.training_loss, "model": status}, ensure_ascii=False, indent=2), encoding="utf-8")


def _worker(output, model_id, phase, revision):
    deadline = os.environ.get("QUALITY_WORKER_DEADLINE")
    if deadline is None:
        raise RuntimeError("Worker must be launched by the bounded pilot harness")
    _arm_worker_deadline(float(deadline))
    if phase == "generate":
        from quality_models import generate_answers
        request = json.loads((output / "request.json").read_text(encoding="utf-8"))
        try:
            answers = generate_answers(request["cases"], request["modelId"], adapter_path=request["adapterPath"], revision=request["revision"])
            result = {"status": "complete", "answers": answers, "message": "Offline local generation completed."}
        except Exception as exc:
            result = {"status": "unavailable", "answers": [], "reason": "generation_failed", "message": f"{type(exc).__name__}: {exc}"}
        (output / "answers.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return
    if phase.startswith("train-"):
        _train_worker(output, model_id, phase.removeprefix("train-"), revision)
        return
    from quality_models import generate_answers
    variant = phase.removeprefix("evaluate-")
    cases = json.loads((output / "cases.json").read_text(encoding="utf-8"))
    adapter = None if variant == "base" else output / variant / "adapter"
    answers = generate_answers(cases, model_id, adapter_path=adapter, revision=revision)
    for answer in answers:
        answer["candidate"] = variant
    (output / (variant + "-answers.json")).write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")


def _supervise_command(command):
    """Separate deadline owner remains alive if the API process disappears."""
    if os.name != "nt" and os.getpgrp() != os.getpid():
        os.setsid()
    deadline = float(os.environ["QUALITY_WORKER_DEADLINE"])
    if not command or time.monotonic() >= deadline:
        return 124
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    child = subprocess.Popen(command, **options)
    try:
        return child.wait(timeout=max(.001, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            _kill_owned_tree(child)
        else:
            # Child shares the supervisor's isolated process group. Stop all
            # descendants; never discover or signal unrelated GPU processes.
            os.killpg(os.getpgrp(), signal.SIGKILL)
        return 124


def evaluate_controlled_outputs(cases, answers, policy):
    from quality_policy import normalize_policy
    from quality_evaluation import evaluate, fingerprint
    policy = normalize_policy(policy)
    attested = [{**answer, 'policyHash': fingerprint(policy)} for answer in answers]
    return evaluate(cases, attested, policy)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--supervise":
        raise SystemExit(_supervise_command(sys.argv[2:]))
    from quality_models import TAIDE_ID, get_model_status
    from quality_evaluation import fingerprint
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=TAIDE_ID)
    parser.add_argument("--revision", help="Immutable cached model SHA; resolved once when omitted")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--runtime-root", type=Path, default=Path(os.environ.get("FORGETUNE_DATA_ROOT", str(Path(__file__).resolve().parent / "data"))))
    parser.add_argument("--seconds", type=float, default=1800)
    parser.add_argument("--benchmark", help="Prepared immutable benchmark SHA, evaluated only")
    parser.add_argument("--policy", type=Path, help="Policy JSON for optional final evaluation")
    parser.add_argument("--worker", choices=("generate", "evaluate-base", "train-original", "evaluate-original", "train-cleaned", "evaluate-cleaned"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not args.revision:
            parser.error("A worker requires its parent's immutable --revision")
        _worker(args.output, args.model, args.worker, args.revision)
        return
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 1800:
        parser.error("--seconds must be > 0 and <= 1800")
    args.output.mkdir(parents=True, exist_ok=False)
    status = get_model_status(args.model, revision=args.revision)
    if status["status"] != "ready":
        summary = {"status": "blocked", "reason": status["status"], "model": status, "chargedGpuSeconds": 0, "phases": []}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return
    partitions = fixtures()
    validate_partitions(partitions)
    from quality_models import load_benchmark
    cases = load_benchmark(args.benchmark, runtime_root=args.runtime_root)["cases"] if args.benchmark else partitions["test"]
    (args.output / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"source": "authored-synthetic-v1", "model": status, "partitions": partitions, "partitionsHash": fingerprint(partitions), "casesHash": fingerprint(cases), "trainingParameters": training_parameters(), "limitation": "Tiny plumbing pilot, writing requires human review, no leaderboard comparability or demonstrated quality gain."}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    phases = ("evaluate-base", "train-original", "evaluate-original", "train-cleaned", "evaluate-cleaned")
    commands = [[sys.executable, "-B", str(Path(__file__).resolve()), "--output", str(args.output.resolve()), "--model", args.model, "--revision", status["revision"], "--worker", phase] for phase in phases]
    summary = run_with_pilot_budget(args.runtime_root, commands, max_seconds=args.seconds, gpu=args.gpu, log_dir=args.output)
    for phase in summary["phases"]:
        phase["name"] = phases[phase["index"]]
    answers = []
    for variant in ("base", "original", "cleaned"):
        path = args.output / (variant + "-answers.json")
        if path.is_file():
            answers.extend(json.loads(path.read_text(encoding="utf-8")))
    policy = json.loads(args.policy.read_text(encoding="utf-8")) if args.policy else {}
    report = evaluate_controlled_outputs(cases, answers, policy)
    (args.output / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
