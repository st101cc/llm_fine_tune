"""Offline local TAIDE inference and explicitly invoked preparation commands.

Only prepare_model/prepare_benchmark use the network. Import/status/generation
never download, call hosted inference, or accept license terms.
"""
from __future__ import annotations

import argparse
import csv
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import threading

from quality_evaluation import fingerprint


TAIDE_ID = "taide/Llama-3.1-TAIDE-LX-8B-Chat"
TMMLU_ID = "ikala/tmmluplus"
DEFAULT_SUBJECTS = ("computer_science", "finance_banking", "official_document_management")
BENCHMARKS = Path(os.environ.get("FORGETUNE_DATA_ROOT", str(Path(__file__).resolve().parent / "data"))) / "quality" / "benchmarks"
MCQ_INSTRUCTION = "請回答以下單選題。只輸出一個大寫英文字母 A、B、C 或 D，不要解釋。"
TEXT_INSTRUCTION = "請依照題目要求，使用臺灣繁體中文回答。"
_GENERATION_LOCK = threading.Lock()


def _immutable(revision):
    if not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("An immutable 40-character cache commit SHA is required")
    return revision


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_model_status(model_id=TAIDE_ID, *, revision=None):
    """Inspect cache and license metadata only; ready does not certify HF access."""
    from huggingface_hub import ModelCard, scan_cache_dir, snapshot_download

    status = {"modelId": model_id, "status": "missing_weights", "revision": None, "localOnly": True, "license": {"acceptance": "not_checked_offline"}}
    try:
        try:
            path = Path(snapshot_download(repo_id=model_id, revision=revision or "main", local_files_only=True))
        except (OSError, ValueError):
            if revision:
                raise
            # Explicit SHA downloads need not create a mutable refs/main entry.
            snapshots = [r for repo in scan_cache_dir().repos if repo.repo_id == model_id and repo.repo_type == "model" for r in repo.revisions]
            if not snapshots:
                raise FileNotFoundError("No cached immutable snapshots")
            path = Path(max(snapshots, key=lambda r: (r.last_modified, r.commit_hash)).snapshot_path)
        commit = _immutable(path.name)
        if revision and commit != _immutable(revision):
            raise ValueError("Cached snapshot does not match requested revision")
    except (OSError, ValueError) as exc:
        return {**status, "message": f"Cached immutable weights unavailable. Use explicit quality_models.py prepare model after HF gate authorization. {type(exc).__name__}"}
    status.update({"path": str(path), "revision": commit})
    required = ["config.json", "tokenizer_config.json"]
    if not any((path / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
        required.append("tokenizer.json")
    index = path / "model.safetensors.index.json"
    if index.is_file():
        try:
            shards = list(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
            if not shards or any(Path(name).name != name for name in shards):
                raise ValueError("Invalid shard names")
            required.extend(shards)
        except (OSError, KeyError, ValueError, TypeError):
            return {**status, "message": "Cached model shard index is invalid; prepare the model again."}
    else:
        required.append("model.safetensors")
    missing = [name for name in required if not (path / name).is_file() or (path / name).stat().st_size == 0]
    if missing:
        return {**status, "missingFiles": missing, "message": "Incomplete cached weights; use explicit prepare model after HF gate authorization."}
    card = path / "README.md"
    license_id = None
    if card.is_file():
        try:
            license_id = ModelCard.load(str(card)).data.to_dict().get("license")
        except (ValueError, KeyError):
            pass
    license_files = {p.name: file_hash(p) for p in sorted(path.glob("LICENSE*")) if p.is_file()}
    status["license"].update({"id": license_id, "files": license_files, "cardHash": file_hash(card) if card.is_file() else None, "source": f"https://huggingface.co/{model_id}/tree/{commit}"})
    if not license_id and not license_files:
        return {**status, "status": "missing_license", "message": "Cached license metadata missing; explicitly prepare after reviewing the model license and obtaining required HF access."}
    return {**status, "status": "ready", "message": "Immutable local snapshot available; license/access acceptance is not inferred from cache presence."}


def _adapter_metadata(adapter_path, model_id, revision):
    if adapter_path is None:
        return None
    path = Path(adapter_path).resolve(strict=True)
    config_path = path / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = config.get("base_model_name_or_path")
    if base != model_id and Path(str(base)).name != revision:
        raise ValueError("Adapter base model does not match the immutable model snapshot")
    if config.get("revision") not in (None, revision):
        raise ValueError("Adapter base revision differs")
    weights = path / "adapter_model.safetensors"
    if not weights.is_file():
        raise ValueError("Local safetensors adapter weights are required")
    return {"path": str(path), "hash": fingerprint({"config": file_hash(config_path), "weights": file_hash(weights)})}


def generate_answers(cases, model_id=TAIDE_ID, adapter_path=None, cancelled=lambda: False, *, revision=None):
    """Sequential single-device greedy inference. Refuse input/output truncation.

    Call from quality_pilot for a process-enforced time limit. Cooperative
    cancellation also interrupts generation between decoding steps.
    """
    cases = list(cases)
    if cancelled() or not cases:
        return []
    ids = [c.get("id") for c in cases]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Generation requires unique nonempty case IDs")
    if any(c.get("task") not in ("classification", "json", "writing", "mcq") or not isinstance(c.get("input"), str) for c in cases):
        raise ValueError("Generation requires supported tasks and text inputs")
    status = get_model_status(model_id, revision=revision)
    if status["status"] != "ready":
        raise RuntimeError(status["message"])
    adapter = _adapter_metadata(adapter_path, model_id, status["revision"])
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

    class Cancelled(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return cancelled()

    if not torch.cuda.is_available():
        raise RuntimeError("A local CUDA GPU is required; no hosted API fallback is enabled.")
    with _GENERATION_LOCK:
        if cancelled():
            return []
        model = None
        answers = []
        try:
            tokenizer = AutoTokenizer.from_pretrained(status["path"], local_files_only=True, trust_remote_code=False)
            model = AutoModelForCausalLM.from_pretrained(status["path"], local_files_only=True, trust_remote_code=False, device_map={"": 0}, torch_dtype=torch.float16)
            if adapter:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, adapter["path"], local_files_only=True, is_trainable=False)
            model.eval()
            for case in cases:
                if cancelled():
                    break
                instruction = MCQ_INSTRUCTION if case["task"] == "mcq" else TEXT_INSTRUCTION
                limit = 8 if case["task"] == "mcq" else 256
                messages = [{"role": "system", "content": instruction}, {"role": "user", "content": case["input"]}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=False).to(model.device)
                input_length = inputs["input_ids"].shape[1]
                context = min(getattr(tokenizer, "model_max_length", 10**12), getattr(model.config, "max_position_embeddings", 0))
                if not context or input_length + limit > min(context, 4096):
                    raise ValueError("Input exceeds context budget; truncation is forbidden")
                eos = getattr(model.generation_config, "eos_token_id", None) or tokenizer.eos_token_id
                with torch.inference_mode():
                    output = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=limit, eos_token_id=eos, pad_token_id=tokenizer.eos_token_id, stopping_criteria=StoppingCriteriaList([Cancelled()]))
                if cancelled():
                    break
                tokens = output[0][input_length:]
                eos_ids = eos if isinstance(eos, (tuple, list)) else [eos]
                if len(tokens) >= limit and int(tokens[-1]) not in eos_ids:
                    raise ValueError("Output hit its token limit; truncated answers are not accepted")
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                protocol = {"version": "quality-generation-v1", "instruction": instruction, "zeroShot": True, "doSample": False, "numBeams": 1, "maxNewTokens": limit, "parser": "whole-answer-A-D" if case["task"] == "mcq" else "task-specific", "chatTemplateHash": fingerprint(getattr(tokenizer, "chat_template", None))}
                answers.append({"caseId": case["id"], "candidate": model_id + ("@adapter:" + adapter["hash"] if adapter else "@base"), "output": text, "protocol": protocol, "provenance": {"modelId": model_id, "revision": status["revision"], "license": status["license"], "adapter": adapter, "promptHash": fingerprint(prompt)}})
            return answers
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()


def prepare_model(model_id=TAIDE_ID):
    """Explicit network action; verify existing gate permission, never accept terms."""
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    try:
        info = api.model_info(model_id)
        revision = _immutable(info.sha)
        if info.gated:
            api.auth_check(repo_id=model_id, repo_type="model")
        path = snapshot_download(repo_id=model_id, revision=revision, local_files_only=False)
    except Exception as exc:
        raise RuntimeError("Model preparation failed. Obtain already-authorized HF gate access and configure your HF token; this command never accepts license terms.") from exc
    return {"modelId": model_id, "revision": revision, "path": path, "acceptedTerms": False}


def prepare_benchmark(root=BENCHMARKS, *, all_subjects=False):
    """Pin public TMMLU+ v1.1; default first 10 test rows per selected subject."""
    from huggingface_hub import HfApi, hf_hub_download
    root = Path(root).resolve()
    if root.name != "benchmarks" or "datasets" in {part.lower() for part in root.parts}:
        raise ValueError("Use a dedicated benchmarks directory, never datasets")
    info = HfApi().dataset_info(TMMLU_ID, revision="v1.1")
    revision = _immutable(info.sha)
    available = sorted({match[1] for item in info.siblings if (match := re.fullmatch(r"data/([a-z0-9_]+)_test\.csv", item.rfilename))})
    subjects = available if all_subjects else list(DEFAULT_SUBJECTS)
    if not subjects or not set(subjects) <= set(available):
        raise ValueError("TMMLU+ v1.1 is missing expected subject test files")
    directory = root / "tmmluplus" / revision
    cases, files = [], {}
    for subject in subjects:
        filename = f"data/{subject}_test.csv"
        path = Path(hf_hub_download(repo_id=TMMLU_ID, repo_type="dataset", revision=revision, filename=filename, local_dir=str(directory)))
        files[filename] = file_hash(path)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) < 10:
            raise ValueError("Expected at least 10 questions per subject")
        for index, row in enumerate(rows[:10]):
            if any(not row.get(k) for k in ("question", "A", "B", "C", "D", "answer")) or row["answer"] not in "ABCD" or len(row["answer"]) != 1:
                raise ValueError("Invalid TMMLU+ question schema")
            cases.append({"id": f"tmmluplus:{revision}:{subject}:test:{index}", "input": row["question"] + "\n" + "\n".join(f"{key}. {row[key]}" for key in "ABCD"), "reference": row["answer"], "task": "mcq", "subject": subject})
    manifest = {"datasetId": TMMLU_ID, "version": "v1.1", "revision": revision, "split": "test", "selection": "first-10-per-subject", "allSubjects": all_subjects, "subjects": subjects, "files": files, "cases": cases, "casesHash": fingerprint(cases), "protocol": {"instruction": MCQ_INSTRUCTION, "zeroShot": True, "doSample": False, "maxNewTokens": 8, "parser": "whole-answer-A-D"}, "limitation": "Bounded zero-shot sample; not comparable to leaderboard scores."}
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("all-subjects.json" if all_subjects else "default-30.json")
    content = json.dumps(manifest, ensure_ascii=False, indent=2)
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise ValueError("An immutable benchmark manifest already exists with different content")
    path.write_text(content, encoding="utf-8")
    checksum = file_hash(path)
    path.with_suffix(".sha256").write_text(checksum, encoding="ascii")
    return {**manifest, "manifestChecksum": checksum}


def load_benchmark(revision, *, runtime_root, all_subjects=False, expected_checksum=None):
    """Read only: accept an immutable SHA, never a caller-supplied file path."""
    revision = _immutable(revision)
    root = Path(runtime_root).resolve()
    directory = (root / "quality" / "benchmarks" / "tmmluplus" / revision).resolve()
    if not directory.is_relative_to(root):
        raise ValueError("Benchmark path escapes runtime area")
    path = directory / ("all-subjects.json" if all_subjects else "default-30.json")
    checksum_path = path.with_suffix(".sha256")
    if not path.resolve().is_relative_to(directory) or not checksum_path.resolve().is_relative_to(directory):
        raise ValueError("Benchmark checksum path escapes runtime area")
    if not path.is_file() or not checksum_path.is_file():
        raise ValueError("Prepared benchmark or checksum missing; explicitly prepare benchmark first")
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Benchmark manifest is too large")
    checksum = file_hash(path)
    if checksum_path.read_text(encoding="ascii").strip() != checksum or (expected_checksum is not None and expected_checksum != checksum):
        raise ValueError("Benchmark manifest checksum mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("revision") != revision or manifest.get("datasetId") != TMMLU_ID or manifest.get("version") != "v1.1" or manifest.get("split") != "test" or manifest.get("allSubjects") is not all_subjects:
        raise ValueError("Benchmark provenance does not match the requested immutable dataset")
    if fingerprint(manifest.get("cases")) != manifest.get("casesHash"):
        raise ValueError("Benchmark cases checksum mismatch")
    files = manifest.get("files", {})
    if not files or any(not re.fullmatch(r"data/[a-z0-9_]+_test\.csv", name) for name in files):
        raise ValueError("Invalid benchmark source filenames")
    for name, digest in files.items():
        source = (directory / name).resolve()
        if not source.is_relative_to(directory) or not source.is_file() or file_hash(source) != digest:
            raise ValueError("Benchmark source checksum mismatch or path escape")
    cases = manifest["cases"]
    if len({c["id"] for c in cases}) != len(cases) or any(c.get("task") != "mcq" or c.get("reference") not in ("A", "B", "C", "D") or not c["id"].startswith(f"tmmluplus:{revision}:") for c in cases):
        raise ValueError("Invalid immutable benchmark cases")
    return {**manifest, "manifestChecksum": checksum}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status")
    status.add_argument("--model", default=TAIDE_ID)
    prepare = sub.add_parser("prepare", help="Explicit download; never accepts gate/license terms")
    prepare.add_argument("target", choices=("model", "benchmark"))
    prepare.add_argument("--model", default=TAIDE_ID)
    prepare.add_argument("--root", type=Path, default=BENCHMARKS)
    prepare.add_argument("--all-subjects", action="store_true", help="10 test questions from every subject")
    args = parser.parse_args()
    result = get_model_status(args.model) if args.command == "status" else (prepare_model(args.model) if args.target == "model" else prepare_benchmark(args.root, all_subjects=args.all_subjects))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
