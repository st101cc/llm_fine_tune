"""Local-first dataset profiling and optional OpenAI-compatible cloud clarification."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

MODELS = [
    {"id": "Qwen/Qwen2.5-3B-Instruct", "name": "Qwen 2.5", "sizeB": 3, "vram": 8, "license": "Qwen Research", "task": "general"},
    {"id": "Qwen/Qwen2.5-7B-Instruct", "name": "Qwen 2.5", "sizeB": 7, "vram": 13, "license": "Apache 2.0", "task": "general"},
    {"id": "Qwen/Qwen2.5-Coder-7B-Instruct", "name": "Qwen 2.5 Coder", "sizeB": 7, "vram": 13, "license": "Apache 2.0", "task": "code"},
    {"id": "mistralai/Mistral-7B-Instruct-v0.3", "name": "Mistral", "sizeB": 7, "vram": 13, "license": "Apache 2.0", "task": "english-writing"},
    {"id": "Qwen/Qwen2.5-14B-Instruct", "name": "Qwen 2.5", "sizeB": 14, "vram": 22, "license": "Apache 2.0", "task": "general"},
]
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")
SECRET = re.compile(r"\b(?:sk|api|key|token|secret)[_-]?[A-Za-z0-9]{12,}\b", re.I)
CREDENTIAL_URL = re.compile(r"https?://[^\s/@:]+:[^\s/@]+@", re.I)
CODE = re.compile(r"\b(?:def|class|function|const|let|import|SELECT|FROM|return)\b|[{};]{2,}", re.I)


def load_source(source: str, name: str, extension: str | None = None) -> Dataset:
    if source == "huggingface":
        return load_dataset(name, split="train")
    suffix = (extension or Path(name).suffix.lstrip(".")).lower()
    loader = {"csv": "csv", "jsonl": "json", "parquet": "parquet"}.get(suffix)
    if not loader:
        raise ValueError("Supported file types are CSV, JSONL, and Parquet.")
    return load_dataset(loader, data_files=name, split="train")


def content_for_row(row: dict[str, Any]) -> str:
    if isinstance(row.get("messages"), list):
        return "\n".join(str(item.get("content", "")) if isinstance(item, dict) else str(item) for item in row["messages"])
    if "prompt" in row or "completion" in row:
        return str(row.get("prompt", "")) + "\n" + str(row.get("completion", ""))
    return str(row.get("text", ""))


def redact(text: str) -> tuple[str, int]:
    before = text
    text = EMAIL.sub("[EMAIL]", text)
    text = PHONE.sub("[PHONE]", text)
    text = SECRET.sub("[SECRET]", text)
    text = CREDENTIAL_URL.sub("https://[CREDENTIALS]@", text)
    return text, int(text != before)


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    return values[min(len(values) - 1, max(0, round((len(values) - 1) * fraction)))]


def detect_schema(columns: list[str]) -> str | None:
    if "text" in columns and any(name in columns for name in ("label", "class", "target")):
        return "classification"
    if "messages" in columns:
        return "chat"
    if {"prompt", "completion"}.issubset(columns):
        return "prompt_completion"
    if "text" in columns:
        return "text"
    return None


def classification_label_field(columns: list[str]) -> str | None:
    return next((name for name in ("label", "class", "target") if name in columns), None)


def choose_positive_label(labels: list[str]) -> str | None:
    """Choose a transparent default positive class when the user has not set one."""
    normalized = {label.lower(): label for label in labels}
    for preferred in ("1", "true", "yes", "positive", "spam", "fraud", "unsafe"):
        if preferred in normalized:
            return normalized[preferred]
    return labels[-1] if len(labels) == 2 else None


def classification_report(rows: list[dict[str, Any]], label_field: str, predictions: list[str]) -> dict[str, Any]:
    """Create reproducible class metrics and an error queue without an LLM judge."""
    labels = sorted({str(row.get(label_field, "")).strip() for row in rows if str(row.get(label_field, "")).strip()})
    positive = choose_positive_label(labels)
    tp = fp = fn = tn = 0
    errors: list[dict[str, Any]] = []
    correct = 0
    for row, predicted in zip(rows, predictions):
        expected = str(row.get(label_field, "")).strip()
        predicted = str(predicted).strip()
        if predicted == expected:
            correct += 1
        if positive:
            if expected == positive and predicted == positive:
                tp += 1
            elif expected != positive and predicted == positive:
                fp += 1
            elif expected == positive and predicted != positive:
                fn += 1
            else:
                tn += 1
        if predicted != expected:
            category = "false_negative" if positive and expected == positive else "false_positive" if positive and predicted == positive else "wrong_class"
            errors.append({
                "rowId": row.get("_row_id"), "input": str(row.get("text", ""))[:600],
                "expected": expected, "predicted": predicted or "(no recognized label)", "category": category,
            })
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {
        "supported": True, "labelField": label_field, "labels": labels, "positiveLabel": positive,
        "total": len(rows), "correct": correct, "accuracy": round(correct / len(rows), 4) if rows else None,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn} if positive else None,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "errors": errors[:100], "errorCount": len(errors),
    }


def local_profile(dataset: Dataset) -> dict[str, Any]:
    columns = list(dataset.column_names)
    schema = detect_schema(columns)
    sample_size = min(len(dataset), 600)
    sample = dataset.select(range(sample_size))
    valid, invalid, texts, language = [], [], [], Counter()
    duplicates, code_count, redactions = 0, 0, 0
    seen: set[str] = set()
    samples: list[dict[str, Any]] = []
    for index, row in enumerate(sample):
        text = content_for_row(row).strip()
        if not text or not schema:
            invalid.append(index)
            continue
        digest = hashlib.sha256(re.sub(r"\s+", " ", text.lower()).encode()).hexdigest()
        duplicates += digest in seen
        seen.add(digest)
        cjk = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
        alpha = sum(ch.isalpha() for ch in text)
        language["Chinese / CJK" if cjk > 3 else "English / Latin" if alpha > 3 else "Unknown"] += 1
        code_count += bool(CODE.search(text))
        clean, count = redact(text)
        redactions += count
        valid.append(index)
        texts.append(text)
        if len(samples) < 20:
            samples.append({"rowId": index, "text": clean[:1200]})

    lengths = [len(value) for value in texts]
    duplicate_rate = duplicates / max(1, len(valid))
    invalid_rate = len(invalid) / max(1, sample_size)
    code_ratio = code_count / max(1, len(valid))
    task = "classification" if schema == "classification" else "code" if code_ratio > .22 else "assistant" if schema == "chat" else "structured" if any("{" in value and "}" in value for value in texts[:30]) else "writing"
    confidence = .95 if schema == "classification" else .92 if schema in {"chat", "prompt_completion"} else .74 if schema == "text" else .25
    findings = []
    if not schema:
        findings.append({"severity": "blocker", "code": "unsupported_schema", "message": "Need messages, prompt/completion, or text columns."})
    if len(valid) < 30:
        findings.append({"severity": "blocker", "code": "too_small", "message": "Fewer than 30 valid examples. Use prompting/RAG or collect more examples."})
    elif len(valid) < 500:
        findings.append({"severity": "warning", "code": "small_dataset", "message": "Fewer than 500 examples; use low rank and review overfitting carefully."})
    if invalid_rate > .2:
        findings.append({"severity": "blocker", "code": "invalid_rows", "message": "More than 20% of sampled rows are empty or incompatible."})
    if duplicate_rate > .25:
        findings.append({"severity": "blocker", "code": "duplicates", "message": "More than 25% of sampled rows are duplicates; deduplicate before training."})
    if task == "writing" and len(valid) < 100:
        findings.append({"severity": "warning", "code": "uncertain_task", "message": "Task classification is uncertain; cloud clarification is available if enabled."})
    return {
        "facts": {
            "schema": schema or "unknown", "columns": columns, "labelField": classification_label_field(columns), "rows": len(dataset), "sampledRows": sample_size,
            "validRows": len(valid), "rejectedRows": len(invalid), "rejectedRowIds": invalid[:50],
            "languages": dict(language), "codeRatio": round(code_ratio, 3), "duplicateRate": round(duplicate_rate, 3),
            "lengthCharacters": {"p50": percentile(lengths, .5), "p95": percentile(lengths, .95), "max": max(lengths, default=0)},
            "estimatedTokenLength": {"p50": math.ceil(percentile(lengths, .5) / 3.5), "p95": math.ceil(percentile(lengths, .95) / 3.5)},
        },
        "classification": {"task": task, "confidence": confidence, "source": "local"},
        "findings": findings,
        "redaction": {"sampleRows": len(samples), "fieldsRedacted": redactions, "sentToCloud": False},
        "_samples": samples,
    }


def rank_candidates(profile: dict[str, Any], goal: str, memory_gb: int = 24) -> list[dict[str, Any]]:
    task = profile["classification"]["task"]
    candidates = []
    for model in MODELS:
        method = "qlora" if model["sizeB"] >= 7 else ("lora" if goal == "quality" else "qlora")
        reason, rejected = [], None
        if model["vram"] > memory_gb:
            rejected = "Estimated VRAM exceeds available GPU memory."
        elif model["sizeB"] == 14 and method != "qlora":
            rejected = "14B requires QLoRA."
        elif task == "code" and model["task"] != "code":
            reason.append("General model; code-specialized candidate ranks higher.")
        elif task == "code":
            reason.append("Code-specialized instruction model.")
        elif task == "assistant" and model["id"].startswith("Qwen"):
            reason.append("Strong multilingual instruction fit.")
        elif goal == "speed" and model["sizeB"] == 3:
            reason.append("Smallest eligible model for fast iteration.")
        elif model["sizeB"] == 14:
            reason.append("Higher capacity but slower; use only if baseline improves materially.")
        score = (100 - model["sizeB"] * (9 if goal == "speed" else 2)) + (30 if task == "code" and model["task"] == "code" else 0) + (16 if task == "assistant" and model["id"].startswith("Qwen") else 0)
        candidates.append({"model": model, "method": method, "estimatedVramGb": model["vram"], "estimatedDuration": "longer" if model["sizeB"] == 14 else "moderate" if model["sizeB"] == 7 else "short", "score": score, "rejected": rejected, "reasons": reason})
    return sorted(candidates, key=lambda item: (item["rejected"] is not None, -item["score"]))


def cloud_status() -> dict[str, Any]:
    configured = all(os.getenv(key) for key in ("ADVISOR_BASE_URL", "ADVISOR_API_KEY", "ADVISOR_MODEL"))
    return {"configured": configured, "baseUrl": os.getenv("ADVISOR_BASE_URL", "") if configured else None, "model": os.getenv("ADVISOR_MODEL", "") if configured else None}


def cloud_clarify(samples: list[dict[str, Any]], local: dict[str, Any]) -> dict[str, Any] | None:
    if not cloud_status()["configured"]:
        return None
    endpoint = os.environ["ADVISOR_BASE_URL"].rstrip("/") + "/chat/completions"
    prompt = {"localClassification": local["classification"], "facts": local["facts"], "rows": samples[:20], "instruction": "Return JSON only with task, languageSummary, confidence, and explanation. Do not infer personal data."}
    request = urllib.request.Request(endpoint, data=json.dumps({"model": os.environ["ADVISOR_MODEL"], "messages": [{"role": "user", "content": json.dumps(prompt)}], "temperature": 0}).encode(), headers={"Authorization": "Bearer " + os.environ["ADVISOR_API_KEY"], "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            content = json.loads(response.read())["choices"][0]["message"]["content"]
        return json.loads(content.strip())
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError, json.JSONDecodeError):
        return None


def cloud_assess_blockers(samples: list[dict[str, Any]], profile: dict[str, Any], blockers: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Ask the optional cloud assistant for advice only; it cannot mutate data."""
    if not cloud_status()["configured"]:
        return None
    endpoint = os.environ["ADVISOR_BASE_URL"].rstrip("/") + "/chat/completions"
    prompt = {
        "profile": {"facts": profile.get("facts", {}), "classification": profile.get("classification", {})},
        "blockers": blockers,
        "redactedRows": samples[:20],
        "instruction": "Return JSON only: {canResolve:boolean, requiresHumanApproval:boolean, reason:string, suggestedOperations:string[]}. Give advice only. Never claim to have changed data, and never recommend bypassing a blocker.",
    }
    request = urllib.request.Request(endpoint, data=json.dumps({"model": os.environ["ADVISOR_MODEL"], "messages": [{"role": "user", "content": json.dumps(prompt)}], "temperature": 0}).encode(), headers={"Authorization": "Bearer " + os.environ["ADVISOR_API_KEY"], "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            content = json.loads(response.read())["choices"][0]["message"]["content"]
        result = json.loads(content.strip())
        return result if isinstance(result, dict) else None
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError, json.JSONDecodeError):
        return None


def connection_test() -> dict[str, Any]:
    if not cloud_status()["configured"]:
        return {"ok": False, "message": "Cloud assist is not configured."}
    result = cloud_clarify([{"rowId": 0, "text": "Classify this harmless sample."}], {"classification": {"task": "unknown", "confidence": 0}, "facts": {}})
    return {"ok": result is not None, "message": "Connection succeeded." if result else "Connection failed or returned an unsupported response."}


def baseline_compare(
    dataset: Dataset,
    candidates: list[dict[str, Any]],
    limit: int = 20,
    eval_indices: list[int] | None = None,
    task: str | None = None,
) -> list[dict[str, Any]]:
    """Run small local base-model comparisons. This intentionally loads one model at a time."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    selected_indices = list(eval_indices)[:limit] if eval_indices is not None else list(range(len(dataset)))[:limit]
    rows = dataset.select(selected_indices)
    results = []
    for candidate in candidates:
        started = time.perf_counter()
        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate["model"]["id"], use_fast=True)
            tokenizer.pad_token = tokenizer.eos_token
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4") if candidate["method"] == "qlora" else None
            model = AutoModelForCausalLM.from_pretrained(candidate["model"]["id"], quantization_config=quant, device_map="auto")
            samples, scores, format_scores, losses = [], [], [], []
            for index, row in enumerate(rows):
                if isinstance(row.get("messages"), list):
                    prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)[:4000]
                else:
                    prompt = str(row.get("prompt", row.get("text", "")))[:4000]
                reference = str(row.get("completion", ""))
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    loss_output = model(**inputs, labels=inputs["input_ids"])
                    if loss_output.loss is not None and math.isfinite(float(loss_output.loss.item())):
                        losses.append(float(loss_output.loss.item()))
                output = model.generate(**inputs, max_new_tokens=96, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                if reference:
                    expected = set(re.findall(r"\w+", reference.lower()))
                    actual = set(re.findall(r"\w+", generated.lower()))
                    scores.append(len(expected & actual) / max(1, len(expected | actual)))
                if task == "structured":
                    try:
                        json.loads(generated)
                        format_scores.append(1.0)
                    except (TypeError, json.JSONDecodeError):
                        format_scores.append(0.0)
                if index < 3:
                    samples.append({"rowId": selected_indices[index], "baseOutput": generated[:600], "referenceAvailable": bool(reference)})
            eval_loss = round(sum(losses) / len(losses), 4) if losses else None
            results.append({"modelId": candidate["model"]["id"], "status": "complete", "examples": len(rows), "evaluationIds": selected_indices, "eval_loss": eval_loss, "perplexity": round(math.exp(eval_loss), 3) if eval_loss is not None and eval_loss < 700 else None, "meanTokenOverlap": round(sum(scores) / len(scores), 3) if scores else None, "formatValidRate": round(sum(format_scores) / len(format_scores), 3) if format_scores else None, "latencySeconds": round(time.perf_counter() - started, 1), "samples": samples})
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as error:
            results.append({"modelId": candidate["model"]["id"], "status": "unavailable", "message": str(error)[:300]})
    return results


def evaluate_tuned_run(
    dataset: Dataset,
    candidate: dict[str, Any],
    run_dir: Path,
    eval_indices: list[int],
    test_dataset: Dataset | None = None,
    task: str | None = None,
) -> dict[str, Any]:
    """Evaluate a saved adapter on the same deterministic examples as its baseline."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    model_id = candidate["model"]["id"]
    method = candidate.get("method", "qlora")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4") if method == "qlora" else None
    base = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization, device_map="auto")
    model = PeftModel.from_pretrained(base, str(run_dir / "adapter"))
    rows_dataset = test_dataset if test_dataset is not None else dataset
    rows = rows_dataset.select(list(eval_indices))
    scores: list[float] = []
    format_scores: list[float] = []
    losses: list[float] = []
    samples: list[dict[str, Any]] = []
    started = time.perf_counter()
    for position, row in enumerate(rows):
        if isinstance(row.get("messages"), list):
            prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)[:4000]
        else:
            prompt = str(row.get("prompt", row.get("text", "")))[:4000]
        reference = str(row.get("completion", ""))
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            loss_output = model(**inputs, labels=inputs["input_ids"])
            if loss_output.loss is not None and math.isfinite(float(loss_output.loss.item())):
                losses.append(float(loss_output.loss.item()))
        output = model.generate(**inputs, max_new_tokens=96, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if reference:
            expected = set(re.findall(r"\w+", reference.lower()))
            actual = set(re.findall(r"\w+", generated.lower()))
            scores.append(len(expected & actual) / max(1, len(expected | actual)))
        if task == "structured":
            try:
                json.loads(generated)
                format_scores.append(1.0)
            except (TypeError, json.JSONDecodeError):
                format_scores.append(0.0)
        if position < 3:
            samples.append({"rowId": eval_indices[position], "tunedOutput": generated[:600], "referenceAvailable": bool(reference)})
    eval_loss = round(sum(losses) / len(losses), 4) if losses else None
    result = {
        "status": "complete",
        "modelId": model_id,
        "examples": len(rows),
        "evaluationIds": list(eval_indices),
        "eval_loss": eval_loss,
        "perplexity": round(math.exp(eval_loss), 3) if eval_loss is not None and eval_loss < 700 else None,
        "meanTokenOverlap": round(sum(scores) / len(scores), 3) if scores else None,
        "formatValidRate": round(sum(format_scores) / len(format_scores), 3) if format_scores else None,
        "latencySeconds": round(time.perf_counter() - started, 1),
        "samples": samples,
    }
    del model, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
