"""LangGraph orchestration for the ForgeTune multi-candidate workflow.

The graph deliberately keeps large datasets and model outputs on disk.  State
contains only references, summaries, and the small amount of information
needed to resume a workflow after a human approval or a process restart.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from advisor import (
    baseline_compare,
    cloud_assess_blockers,
    cloud_clarify,
    evaluate_tuned_run,
    load_source,
    local_profile,
    rank_candidates,
)


class WorkflowState(TypedDict, total=False):
    workflow_id: str
    dataset: dict[str, Any]
    test_dataset: dict[str, Any] | None
    goal: Literal["speed", "balanced", "quality"]
    cloud_assist: bool
    split_seed: int
    max_candidates: int
    max_retries: int
    profile: dict[str, Any]
    split_manifest_id: str
    split_manifest_path: str
    findings: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    baseline_results: list[dict[str, Any]]
    approved_plan: dict[str, Any]
    candidate_index: int
    attempts: dict[str, int]
    training_runs: dict[str, dict[str, Any]]
    evaluation_results: list[dict[str, Any]]
    comparison: dict[str, Any]
    status: str
    pending_action: dict[str, Any] | None
    errors: list[str]
    candidate_status: str
    retry_reason: str | None
    blocker_assessment: dict[str, Any]
    blocker_resolution_action: str | None


@dataclass
class WorkflowServices:
    """Side effects supplied by the FastAPI application.

    Keeping these callbacks outside the graph makes routing unit-testable and
    prevents the graph from importing the FastAPI module back into itself.
    """

    data_root: Path
    gpu_memory_gb: Callable[[], int]
    submit_training: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
    get_training: Callable[[str], Awaitable[dict[str, Any]]]
    training_dir: Callable[[str], Path]


def _dataset_key(dataset: dict[str, Any]) -> str:
    return json.dumps(
        {
            "source": dataset.get("source"),
            "name": dataset.get("name"),
            "extension": dataset.get("extension"),
            "split": dataset.get("split", "train"),
        },
        sort_keys=True,
    )


def _content(row: dict[str, Any]) -> str:
    if isinstance(row.get("messages"), list):
        return "\n".join(str(item.get("content", "")) if isinstance(item, dict) else str(item) for item in row["messages"])
    if "prompt" in row or "completion" in row:
        return str(row.get("prompt", "")) + "\n" + str(row.get("completion", ""))
    return str(row.get("text", ""))


def _valid_indices(dataset: Any, schema: str | None) -> list[int]:
    if not schema:
        return []
    valid: list[int] = []
    for index, row in enumerate(dataset):
        if _content(row).strip():
            valid.append(index)
    return valid


def _blocker_assessment(state: WorkflowState, cloud_advice: dict[str, Any] | None = None) -> dict[str, Any]:
    """Conservative remediation policy. The agent may advise, but never bypasses it."""
    blockers = [item for item in state.get("findings", []) if item.get("severity") == "blocker"]
    codes = {item.get("code") for item in blockers}
    upload = state.get("dataset", {}).get("source") == "upload"
    cleanup_codes = {"invalid_rows", "duplicates"}
    can_materialize = bool(codes) and codes.issubset(cleanup_codes) and upload
    if can_materialize:
        decision = "approval_required"
        operations = (["drop_invalid_rows"] if "invalid_rows" in codes else []) + (["deduplicate_exact_rows"] if "duplicates" in codes else [])
        reason = "The uploaded dataset can be copied to a cleaned derivative, but this removes rows and requires approval."
    else:
        decision = "human_required"
        operations = []
        reason = "This blocker requires a replacement dataset or data collection; no safe automatic remediation is available."
    if cloud_advice and isinstance(cloud_advice.get("reason"), str):
        reason += " Cloud advisor: " + cloud_advice["reason"][:300]
    return {
        "decision": decision,
        "requiresHumanApproval": True,
        "canMaterializeAfterApproval": can_materialize,
        "operations": operations,
        "blockers": blockers,
        "reason": reason,
        "advisor": {"used": cloud_advice is not None, "suggestedOperations": cloud_advice.get("suggestedOperations", []) if cloud_advice else []},
    }


def _clean_uploaded_dataset(dataset: Any, profile: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Create an auditable JSONL derivative after the user approves cleanup."""
    schema = profile.get("facts", {}).get("schema")
    kept, seen = [], set()
    for index in _valid_indices(dataset, schema):
        row = dataset[index]
        digest = hashlib.sha256(" ".join(_content(row).lower().split()).encode()).hexdigest()
        if digest not in seen:
            kept.append(index)
            seen.add(digest)
    if len(kept) < 30:
        raise ValueError("Approved cleanup leaves fewer than 30 valid rows; replace or expand the dataset instead.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset.select(kept).to_json(str(destination))
    return {"keptRows": len(kept), "removedRows": len(dataset) - len(kept), "path": str(destination)}


def _persist_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _make_split_manifest(state: WorkflowState, services: WorkflowServices, dataset: Any, profile: dict[str, Any]) -> dict[str, Any]:
    workflow_dir = services.data_root / "workflows" / state["workflow_id"]
    seed = int(state.get("split_seed", 42))
    schema = profile.get("facts", {}).get("schema")
    train_indices = _valid_indices(dataset, schema)
    test_ref = state.get("test_dataset")

    if test_ref:
        eval_dataset = load_source(test_ref["source"], test_ref["name"], test_ref.get("extension"))
        eval_indices = _valid_indices(eval_dataset, schema)
        train_ref = state["dataset"]
        eval_ref = test_ref
    else:
        shuffled = list(train_indices)
        random.Random(seed).shuffle(shuffled)
        eval_count = max(1, round(len(shuffled) * 0.1)) if shuffled else 0
        eval_indices = sorted(shuffled[:eval_count])
        train_indices = sorted(shuffled[eval_count:])
        train_ref = state["dataset"]
        eval_ref = state["dataset"]

    payload = {
        "workflowId": state["workflow_id"],
        "seed": seed,
        "schema": schema,
        "datasetFingerprint": hashlib.sha256(_dataset_key(state["dataset"]).encode()).hexdigest(),
        "trainDataset": train_ref,
        "evalDataset": eval_ref,
        "trainIndices": train_indices,
        "evalIndices": eval_indices,
        "evaluationIds": eval_indices[:100],
    }
    path = workflow_dir / "split.json"
    _persist_json(path, payload)
    payload["path"] = str(path)
    return payload


def _candidate_key(candidate: dict[str, Any]) -> str:
    return str(candidate["model"]["id"])


def _current_candidate(state: WorkflowState) -> dict[str, Any]:
    candidates = state.get("approved_plan", {}).get("candidates") or state.get("candidates", [])
    return candidates[int(state.get("candidate_index", 0))]


def _default_hyperparameters(profile: dict[str, Any]) -> dict[str, Any]:
    valid_rows = int(profile.get("facts", {}).get("validRows", 0))
    p95 = int(profile.get("facts", {}).get("estimatedTokenLength", {}).get("p95", 1024))
    return {
        "rank": 8 if valid_rows < 500 else 16,
        "alpha": 16 if valid_rows < 500 else 32,
        "learning_rate": 2e-4,
        "epochs": 2 if valid_rows < 500 else 1 if valid_rows > 50000 else 3,
        "batch_size": 1,
        "gradient_accumulation": 8,
        "max_sequence_length": min(4096, max(1024, p95 * 2)),
    }


def _base_plan(state: WorkflowState) -> dict[str, Any]:
    plan_candidates = []
    for item in state.get("candidates", [])[: int(state.get("max_candidates", 3))]:
        plan_candidates.append(
            {
                "model": item["model"],
                "method": item["method"],
                "hyperparameters": _default_hyperparameters(state["profile"]),
                "estimatedVramGb": item.get("estimatedVramGb"),
                "estimatedDuration": item.get("estimatedDuration"),
                "score": item.get("score"),
                "reasons": item.get("reasons", []),
            }
        )
    return {
        "candidates": plan_candidates,
        "candidateCount": len(plan_candidates),
        "splitManifestId": state.get("split_manifest_id"),
        "goal": state.get("goal", "balanced"),
        "baselineResults": state.get("baseline_results", []),
        "maxRetries": int(state.get("max_retries", 2)),
    }


def _validate_plan(plan: dict[str, Any], memory_gb: int) -> tuple[bool, str | None]:
    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or not candidates or len(candidates) > 3:
        return False, "Choose between one and three candidates."
    seen: set[str] = set()
    for candidate in candidates:
        model = candidate.get("model", {})
        model_id = model.get("id")
        if not model_id or model_id in seen:
            return False, "Each candidate must have a unique model ID."
        seen.add(model_id)
        if float(candidate.get("estimatedVramGb", model.get("vram", 0)) or 0) > memory_gb:
            return False, f"{model_id} exceeds the available GPU memory."
        if "14B" in model_id and candidate.get("method") != "qlora":
            return False, "14B models require QLoRA on the configured GPU."
        if candidate.get("method") not in {"lora", "qlora"}:
            return False, "Training method must be LoRA or QLoRA."
    return True, None


def _metric_value(result: dict[str, Any], name: str, default: float | None = None) -> float | None:
    value = result.get(name)
    try:
        return float(value) if value is not None and math.isfinite(float(value)) else default
    except (TypeError, ValueError):
        return default


def _quality_value(result: dict[str, Any]) -> float | None:
    overlap = _metric_value(result, "meanTokenOverlap")
    if overlap is not None:
        return overlap
    format_rate = _metric_value(result, "formatValidRate")
    if format_rate is not None:
        return format_rate
    loss = _metric_value(result, "eval_loss")
    return -loss if loss is not None else None


def _compare_results(state: WorkflowState) -> dict[str, Any]:
    baselines = {item.get("modelId"): item for item in state.get("baseline_results", [])}
    tuned = {item.get("modelId"): item for item in state.get("evaluation_results", [])}
    pairs: list[dict[str, Any]] = []
    for candidate in state.get("approved_plan", {}).get("candidates", []):
        model_id = candidate["model"]["id"]
        base = baselines.get(model_id, {"status": "unavailable"})
        result = tuned.get(model_id, {"status": "unavailable"})
        base_overlap = _metric_value(base, "meanTokenOverlap")
        tuned_overlap = _metric_value(result, "meanTokenOverlap")
        base_format = _metric_value(base, "formatValidRate")
        tuned_format = _metric_value(result, "formatValidRate")
        base_loss = _metric_value(base, "eval_loss")
        tuned_loss = _metric_value(result, "eval_loss")
        base_latency = _metric_value(base, "latencySeconds")
        tuned_latency = _metric_value(result, "latencySeconds")
        overlap_delta = tuned_overlap - base_overlap if tuned_overlap is not None and base_overlap is not None else None
        pairs.append(
            {
                "modelId": model_id,
                "base": base,
                "tuned": result,
                "meanTokenOverlapDelta": round(overlap_delta, 4) if overlap_delta is not None else None,
                "formatValidRateDelta": round(tuned_format - base_format, 4) if tuned_format is not None and base_format is not None else None,
                "evalLossDelta": round(tuned_loss - base_loss, 4) if tuned_loss is not None and base_loss is not None else None,
                "latencyDeltaSeconds": round(tuned_latency - base_latency, 3) if tuned_latency is not None and base_latency is not None else None,
                "passed": result.get("status") == "complete" and result.get("artifactValid", False),
            }
        )

    goal = state.get("goal", "balanced")
    valid = [item for item in pairs if item["passed"]]

    def rank_key(item: dict[str, Any]) -> tuple[float, float, float]:
        tuned_result = item["tuned"]
        quality = _quality_value(tuned_result)
        quality = quality if quality is not None else -1.0
        latency = _metric_value(tuned_result, "latencySeconds", 1e9) or 1e9
        vram = _metric_value(tuned_result, "estimatedVramGb", 1e9) or 1e9
        if goal == "speed":
            return (latency, -quality, vram)
        if goal == "quality":
            return (-quality, vram, latency)
        return (-quality + (latency / 1000), latency, vram)

    leaderboard = sorted(valid, key=rank_key)
    return {
        "goal": goal,
        "pairs": pairs,
        "leaderboard": leaderboard,
        "winner": leaderboard[0]["modelId"] if leaderboard else None,
        "evaluationIds": json.loads(Path(state["split_manifest_path"]).read_text(encoding="utf-8")).get("evaluationIds", []),
        "completedAt": time.time(),
    }


def build_workflow_graph(services: WorkflowServices, checkpointer: Any) -> Any:
    """Build and compile the resumable ForgeTune graph."""

    async def initialize(state: WorkflowState) -> dict[str, Any]:
        return {
            "status": "profiling",
            "candidate_index": 0,
            "attempts": state.get("attempts", {}),
            "training_runs": state.get("training_runs", {}),
            "evaluation_results": state.get("evaluation_results", []),
            "errors": state.get("errors", []),
            "max_candidates": min(3, int(state.get("max_candidates", 3))),
            "max_retries": min(2, int(state.get("max_retries", 2))),
            "split_seed": int(state.get("split_seed", 42)),
        }

    async def load_and_normalize(state: WorkflowState) -> dict[str, Any]:
        dataset_ref = state["dataset"]
        dataset = load_source(dataset_ref["source"], dataset_ref["name"], dataset_ref.get("extension"))
        columns = list(dataset.column_names)
        sample = [_content(dataset[index])[:1000] for index in range(min(100, len(dataset)))]
        fingerprint = hashlib.sha256((_dataset_key(dataset_ref) + json.dumps(sample, ensure_ascii=False)).encode()).hexdigest()
        workflow_dir = services.data_root / "workflows" / state["workflow_id"]
        _persist_json(workflow_dir / "dataset.json", {"reference": dataset_ref, "columns": columns, "rows": len(dataset), "fingerprint": fingerprint})
        return {"status": "profiling", "dataset": {**dataset_ref, "rows": len(dataset), "columns": columns, "fingerprint": fingerprint}}

    async def profile_and_split(state: WorkflowState) -> dict[str, Any]:
        dataset_ref = state["dataset"]
        dataset = load_source(dataset_ref["source"], dataset_ref["name"], dataset_ref.get("extension"))
        profile = local_profile(dataset)
        manifest = _make_split_manifest(state, services, dataset, profile)
        return {
            "profile": profile,
            "findings": profile.get("findings", []),
            "split_manifest_id": hashlib.sha256(str(manifest["path"]).encode()).hexdigest()[:16],
            "split_manifest_path": manifest["path"],
            "status": "planning",
        }

    def route_after_profile(state: WorkflowState) -> str:
        blockers = [item for item in state.get("findings", []) if item.get("severity") == "blocker"]
        confidence = float(state.get("profile", {}).get("classification", {}).get("confidence", 0))
        if blockers:
            return "assess_blockers"
        if confidence < 0.8 and state.get("cloud_assist"):
            return "clarify_data"
        if confidence < 0.8:
            return "human_data_review"
        return "build_candidates"

    async def assess_blockers(state: WorkflowState) -> dict[str, Any]:
        blockers = [item for item in state.get("findings", []) if item.get("severity") == "blocker"]
        advice = cloud_assess_blockers(state.get("profile", {}).get("_samples", []), state.get("profile", {}), blockers) if state.get("cloud_assist") else None
        assessment = _blocker_assessment(state, advice)
        return {"blocker_assessment": assessment, "status": "awaiting_blocker_review"}

    async def blocker_resolution_review(state: WorkflowState) -> dict[str, Any]:
        assessment = state.get("blocker_assessment", {})
        actions = ["replace_dataset", "abort"]
        if assessment.get("canMaterializeAfterApproval"):
            actions.insert(0, "approve_remediation")
        response = interrupt(
            {
                "type": "blocker_resolution",
                "message": assessment.get("reason", "A dataset blocker needs review."),
                "assessment": assessment,
                "actions": actions,
            }
        )
        if not isinstance(response, dict):
            raise ValueError("Blocker resolution response must be an object.")
        action = response.get("action")
        if action == "abort":
            return {"status": "blocked", "pending_action": None, "errors": ["Workflow aborted during blocker resolution."]}
        if action == "replace_dataset":
            dataset = response.get("dataset")
            if not isinstance(dataset, dict) or not dataset.get("source") or not dataset.get("name"):
                raise ValueError("replace_dataset requires a dataset source and name.")
            return {"dataset": dataset, "pending_action": None, "status": "profiling", "blocker_resolution_action": "replace_dataset"}
        if action == "approve_remediation" and assessment.get("canMaterializeAfterApproval"):
            return {"pending_action": None, "status": "remediating", "blocker_resolution_action": "approve_remediation"}
        raise ValueError("Unsupported blocker resolution action.")

    async def apply_blocker_remediation(state: WorkflowState) -> dict[str, Any]:
        if state.get("dataset", {}).get("source") != "upload":
            raise ValueError("Only uploaded datasets can be remediated in place.")
        dataset_ref = state["dataset"]
        dataset = load_source(dataset_ref["source"], dataset_ref["name"], dataset_ref.get("extension"))
        destination = services.data_root / "workflows" / state["workflow_id"] / "remediated.jsonl"
        result = _clean_uploaded_dataset(dataset, state["profile"], destination)
        return {
            "dataset": {"source": "upload", "name": result["path"], "extension": "jsonl", "displayName": "remediated.jsonl"},
            "status": "profiling",
            "errors": state.get("errors", []) + [f"Approved remediation removed {result['removedRows']} rows and kept {result['keptRows']}."],
            "blocker_resolution_action": "remediated",
        }

    def route_after_blocker_review(state: WorkflowState) -> str:
        if state.get("status") == "blocked":
            return "finish_blocked"
        if state.get("blocker_resolution_action") == "replace_dataset":
            return "load_and_normalize"
        return "apply_blocker_remediation"

    async def human_data_review(state: WorkflowState) -> dict[str, Any]:
        response = interrupt(
            {
                "type": "data_review",
                "message": "Review the dataset findings before model selection.",
                "profile": state.get("profile", {}),
                "findings": state.get("findings", []),
                "actions": ["replace_dataset", "accept_classification", "abort"],
            }
        )
        if not isinstance(response, dict):
            raise ValueError("Data review response must be an object.")
        action = response.get("action")
        if action == "abort":
            return {"status": "blocked", "pending_action": None, "errors": ["Workflow aborted during data review."]}
        if action == "replace_dataset":
            dataset = response.get("dataset")
            if not isinstance(dataset, dict) or not dataset.get("source") or not dataset.get("name"):
                raise ValueError("replace_dataset requires a dataset source and name.")
            return {"dataset": dataset, "pending_action": None, "status": "profiling", "data_review_action": "replace_dataset"}
        if action == "accept_classification" and state.get("profile", {}).get("classification"):
            if any(item.get("severity") == "blocker" for item in state.get("findings", [])):
                raise ValueError("Blocker findings must be resolved before accepting the classification.")
            return {"pending_action": None, "status": "planning", "data_review_action": "accept_classification"}
        raise ValueError("Unsupported data review action.")

    def route_after_data_review(state: WorkflowState) -> str:
        if state.get("status") == "blocked":
            return "finish_blocked"
        if state.get("data_review_action") == "replace_dataset":
            return "load_and_normalize"
        return "build_candidates"

    async def clarify_data(state: WorkflowState) -> dict[str, Any]:
        profile = state["profile"]
        cloud = cloud_clarify(profile.get("_samples", [])[:20], profile)
        if not cloud or cloud.get("task") not in {"assistant", "writing", "code", "structured", "classification"}:
            return {
                "status": "planning",
                "errors": ["Cloud clarification was unavailable or abstained."],
                "pending_action": None,
                "cloud_clarification_failed": True,
            }
        updated = dict(profile)
        updated["classification"] = {
            "task": cloud["task"],
            "confidence": min(0.9, float(cloud.get("confidence", 0.6))),
            "source": "cloud-assisted",
            "explanation": cloud.get("explanation", ""),
        }
        updated["redaction"] = {**updated.get("redaction", {}), "sentToCloud": True}
        return {"profile": updated, "status": "planning"}

    def route_after_clarification(state: WorkflowState) -> str:
        confidence = float(state.get("profile", {}).get("classification", {}).get("confidence", 0))
        return "build_candidates" if confidence >= 0.8 else "human_data_review"

    async def build_candidates(state: WorkflowState) -> dict[str, Any]:
        memory = services.gpu_memory_gb() or 24
        ranked = rank_candidates(state["profile"], state.get("goal", "balanced"), memory)
        eligible = [item for item in ranked if not item.get("rejected")]
        if not eligible:
            return {"candidates": [], "status": "blocked", "errors": ["No model is feasible for the available GPU memory."]}
        return {"candidates": eligible[: int(state.get("max_candidates", 3))], "status": "baseline"}

    async def baseline_evaluate(state: WorkflowState) -> dict[str, Any]:
        dataset_ref = state.get("test_dataset") or state["dataset"]
        dataset = load_source(dataset_ref["source"], dataset_ref["name"], dataset_ref.get("extension"))
        manifest = json.loads(Path(state["split_manifest_path"]).read_text(encoding="utf-8"))
        candidates = state.get("candidates", [])[: int(state.get("max_candidates", 3))]
        limit = min(100, max(20, len(manifest.get("evalIndices", []))))
        try:
            comparisons = baseline_compare(dataset, candidates, limit, eval_indices=manifest.get("evalIndices", []), task=state.get("profile", {}).get("classification", {}).get("task"))
        except Exception as error:
            comparisons = [{"status": "unavailable", "message": str(error)[:300]}]
        return {"baseline_results": comparisons, "status": "awaiting_plan"}

    async def select_and_approve_plan(state: WorkflowState) -> dict[str, Any]:
        proposed = _base_plan(state)
        validation_error = None
        while True:
            response = interrupt(
                {
                    "type": "plan_approval",
                    "message": validation_error or "Approve or edit the model and training plan before GPU work begins.",
                    "plan": proposed,
                    "actions": ["approve", "edit", "abort"],
                    "validationError": validation_error,
                }
            )
            if not isinstance(response, dict):
                validation_error = "Plan approval response must be an object."
                continue
            if response.get("action") == "abort":
                return {"status": "cancelled", "pending_action": None}
            plan = response.get("plan") if response.get("action") == "edit" else proposed
            valid, reason = _validate_plan(plan, services.gpu_memory_gb() or 24)
            if not valid:
                validation_error = reason or "Invalid training plan."
                proposed = plan if isinstance(plan, dict) else proposed
                continue
            return {"approved_plan": plan, "candidate_index": 0, "status": "training", "pending_action": None}

    def route_after_plan(state: WorkflowState) -> str:
        return "finish_cancelled" if state.get("status") == "cancelled" else "prepare_candidate"

    async def prepare_candidate(state: WorkflowState) -> dict[str, Any]:
        candidate = _current_candidate(state)
        key = _candidate_key(candidate)
        attempts = dict(state.get("attempts", {}))
        attempts.setdefault(key, 0)
        return {"attempts": attempts, "candidate_status": "prepared", "retry_reason": None}

    async def submit_training(state: WorkflowState) -> dict[str, Any]:
        candidate = _current_candidate(state)
        key = _candidate_key(candidate)
        attempt = int(state.get("attempts", {}).get(key, 0))
        job_key = f"{state['workflow_id']}:{key}:{attempt}"
        existing = state.get("training_runs", {}).get(job_key)
        if existing and existing.get("runId"):
            return {"candidate_status": "submitted"}
        hp = dict(candidate.get("hyperparameters") or _default_hyperparameters(state["profile"]))
        record = await services.submit_training(
            {
                "workflowId": state["workflow_id"],
                "candidateId": key,
                "attempt": attempt,
                "idempotencyKey": job_key,
                "dataset": state["dataset"],
                "testDataset": state.get("test_dataset"),
                "baseModel": key,
                "parameterMethod": candidate["method"],
                "hyperparameters": hp,
                "splitManifestId": state["split_manifest_id"],
                "splitManifestPath": state["split_manifest_path"],
                "splitSeed": state.get("split_seed", 42),
            }
        )
        training_runs = dict(state.get("training_runs", {}))
        training_runs[job_key] = {"runId": record["id"], "candidateId": key, "attempt": attempt, "status": record.get("status", "queued")}
        return {"training_runs": training_runs, "candidate_status": "submitted"}

    async def wait_for_training(state: WorkflowState) -> dict[str, Any]:
        candidate = _current_candidate(state)
        key = _candidate_key(candidate)
        attempt = int(state.get("attempts", {}).get(key, 0))
        job_key = f"{state['workflow_id']}:{key}:{attempt}"
        run_id = state.get("training_runs", {}).get(job_key, {}).get("runId")
        if not run_id:
            return {"candidate_status": "failed", "retry_reason": "Training job was not recorded."}
        started = time.monotonic()
        while True:
            record = await services.get_training(run_id)
            status = record.get("status")
            if status in {"complete", "failed", "cancelled"}:
                training_runs = dict(state.get("training_runs", {}))
                training_runs[job_key] = {**training_runs.get(job_key, {}), "status": status}
                return {"training_runs": training_runs, "candidate_status": "complete" if status == "complete" else "failed", "retry_reason": None if status == "complete" else f"Training ended with status {status}."}
            if time.monotonic() - started > 24 * 60 * 60:
                return {"candidate_status": "failed", "retry_reason": "Training exceeded the 24-hour workflow timeout."}
            await asyncio.sleep(2)

    def route_after_training(state: WorkflowState) -> str:
        return "verify_artifacts" if state.get("candidate_status") == "complete" else "assess_retry"

    async def verify_artifacts(state: WorkflowState) -> dict[str, Any]:
        candidate = _current_candidate(state)
        key = _candidate_key(candidate)
        attempt = int(state.get("attempts", {}).get(key, 0))
        job_key = f"{state['workflow_id']}:{key}:{attempt}"
        run_id = state.get("training_runs", {}).get(job_key, {}).get("runId")
        record = await services.get_training(run_id) if run_id else {}
        run_dir = services.training_dir(run_id) if run_id else Path()
        metrics_path = run_dir / "metrics.json"
        adapter_dir = run_dir / "adapter"
        valid = bool(record.get("status") == "complete" and metrics_path.exists() and adapter_dir.exists())
        if valid:
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                valid = all(value is None or math.isfinite(float(value)) for value in metrics.values() if isinstance(value, (int, float)))
            except (OSError, ValueError, TypeError):
                valid = False
        return {"candidate_status": "verified" if valid else "failed", "retry_reason": None if valid else "Training artifacts are missing or invalid."}

    def route_after_verification(state: WorkflowState) -> str:
        return "evaluate_tuned" if state.get("candidate_status") == "verified" else "assess_retry"

    async def evaluate_tuned(state: WorkflowState) -> dict[str, Any]:
        candidate = _current_candidate(state)
        key = _candidate_key(candidate)
        attempt = int(state.get("attempts", {}).get(key, 0))
        job_key = f"{state['workflow_id']}:{key}:{attempt}"
        run_id = state.get("training_runs", {}).get(job_key, {}).get("runId")
        record = await services.get_training(run_id) if run_id else {}
        metrics_path = services.training_dir(run_id) / "metrics.json" if run_id else Path()
        metrics: dict[str, Any] = {}
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        baseline = next((item for item in state.get("baseline_results", []) if item.get("modelId") == key), {})
        manifest = json.loads(Path(state["split_manifest_path"]).read_text(encoding="utf-8"))
        eval_indices = manifest.get("evalIndices", [])[:100]
        try:
            eval_dataset = None
            if state.get("test_dataset"):
                test_ref = state["test_dataset"]
                eval_dataset = load_source(test_ref["source"], test_ref["name"], test_ref.get("extension"))
            generated = evaluate_tuned_run(
                load_source(state["dataset"]["source"], state["dataset"]["name"], state["dataset"].get("extension")),
                candidate,
                services.training_dir(run_id),
                eval_indices,
                test_dataset=eval_dataset,
                task=state.get("profile", {}).get("classification", {}).get("task"),
            )
        except Exception as error:
            generated = {"status": "unavailable", "message": str(error)[:300], "modelId": key}
        tuned_overlap = _metric_value(generated, "meanTokenOverlap")
        base_overlap = _metric_value(baseline, "meanTokenOverlap")
        tuned_format = _metric_value(generated, "formatValidRate")
        base_format = _metric_value(baseline, "formatValidRate")
        tuned_loss = _metric_value(generated, "eval_loss") or _metric_value(metrics, "eval_loss")
        base_loss = _metric_value(baseline, "eval_loss")
        quality_regression = bool(
            (tuned_overlap is not None and base_overlap is not None and tuned_overlap < base_overlap * 0.9)
            or (tuned_format is not None and base_format is not None and tuned_format < base_format * 0.9)
            or (tuned_overlap is None and tuned_format is None and tuned_loss is not None and base_loss is not None and tuned_loss > base_loss * 1.1)
        )
        eval_loss = tuned_loss
        evaluation = {
            **generated,
            "modelId": key,
            "runId": run_id,
            "artifactValid": generated.get("status") == "complete",
            "eval_loss": eval_loss,
            "perplexity": round(math.exp(eval_loss), 3) if eval_loss is not None and eval_loss < 700 else None,
            "estimatedVramGb": candidate.get("estimatedVramGb"),
            "baseStatus": baseline.get("status", "unavailable"),
            "qualityRegression": quality_regression,
            "trainingRecord": {"status": record.get("status"), "framework": record.get("framework")},
        }
        results = [item for item in state.get("evaluation_results", []) if item.get("modelId") != key]
        results.append(evaluation)
        return {
            "evaluation_results": results,
            "candidate_status": "evaluated" if generated.get("status") == "complete" and not quality_regression else "failed",
            "retry_reason": "Tuned quality regressed by more than 10% against the base model." if quality_regression else None if generated.get("status") == "complete" else generated.get("message", "Tuned evaluation was unavailable."),
        }

    async def assess_retry(state: WorkflowState) -> dict[str, Any]:
        candidate = _current_candidate(state)
        key = _candidate_key(candidate)
        attempt = int(state.get("attempts", {}).get(key, 0))
        reason = state.get("retry_reason") or "Candidate evaluation failed."
        if attempt < int(state.get("max_retries", 2)):
            attempts = dict(state.get("attempts", {}))
            attempts[key] = attempt + 1
            updated = dict(candidate)
            hp = dict(updated.get("hyperparameters") or _default_hyperparameters(state["profile"]))
            hp["learning_rate"] = max(1e-6, float(hp.get("learning_rate", 2e-4)) / 2)
            hp["epochs"] = max(1, int(hp.get("epochs", 1)) - 1)
            updated["hyperparameters"] = hp
            candidates = list(state.get("approved_plan", {}).get("candidates", []))
            candidates[int(state.get("candidate_index", 0))] = updated
            plan = {**state["approved_plan"], "candidates": candidates}
            return {"attempts": attempts, "approved_plan": plan, "candidate_status": "retry", "retry_reason": reason}
        response = interrupt(
            {
                "type": "retry_review",
                "message": f"Candidate {key} did not produce a valid result after the retry budget.",
                "reason": reason,
                "modelId": key,
                "actions": ["retry", "skip_candidate", "abort"],
            }
        )
        action = response.get("action") if isinstance(response, dict) else None
        if action == "retry":
            attempts = dict(state.get("attempts", {}))
            attempts[key] = attempt + 1
            return {"attempts": attempts, "candidate_status": "retry", "retry_reason": reason}
        if action == "skip_candidate":
            return {"candidate_status": "skipped", "retry_reason": None}
        return {"status": "failed", "candidate_status": "failed", "errors": [reason]}

    def route_after_retry(state: WorkflowState) -> str:
        if state.get("status") == "failed":
            return "finish_failed"
        return "submit_training" if state.get("candidate_status") == "retry" else "advance_candidate"

    async def advance_candidate(state: WorkflowState) -> dict[str, Any]:
        next_index = int(state.get("candidate_index", 0)) + 1
        total = len(state.get("approved_plan", {}).get("candidates", []))
        return {"candidate_index": next_index, "status": "training" if next_index < total else "comparing"}

    def route_after_advance(state: WorkflowState) -> str:
        return "prepare_candidate" if int(state.get("candidate_index", 0)) < len(state.get("approved_plan", {}).get("candidates", [])) else "compare_final"

    async def compare_final(state: WorkflowState) -> dict[str, Any]:
        return {"comparison": _compare_results(state), "status": "complete", "pending_action": None}

    async def finish(state: WorkflowState) -> dict[str, Any]:
        return {"pending_action": None}

    builder = StateGraph(WorkflowState)
    nodes = {
        "initialize": initialize,
        "load_and_normalize": load_and_normalize,
        "profile_and_split": profile_and_split,
        "assess_blockers": assess_blockers,
        "blocker_resolution_review": blocker_resolution_review,
        "apply_blocker_remediation": apply_blocker_remediation,
        "human_data_review": human_data_review,
        "clarify_data": clarify_data,
        "build_candidates": build_candidates,
        "baseline_evaluate": baseline_evaluate,
        "select_and_approve_plan": select_and_approve_plan,
        "prepare_candidate": prepare_candidate,
        "submit_training": submit_training,
        "wait_for_training": wait_for_training,
        "verify_artifacts": verify_artifacts,
        "evaluate_tuned": evaluate_tuned,
        "assess_retry": assess_retry,
        "advance_candidate": advance_candidate,
        "compare_final": compare_final,
        "finish_blocked": finish,
        "finish_cancelled": finish,
        "finish_failed": finish,
    }
    for name, node in nodes.items():
        builder.add_node(name, node)

    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "load_and_normalize")
    builder.add_edge("load_and_normalize", "profile_and_split")
    builder.add_conditional_edges("profile_and_split", route_after_profile, {"assess_blockers": "assess_blockers", "human_data_review": "human_data_review", "clarify_data": "clarify_data", "build_candidates": "build_candidates"})
    builder.add_edge("assess_blockers", "blocker_resolution_review")
    builder.add_conditional_edges("blocker_resolution_review", route_after_blocker_review, {"load_and_normalize": "load_and_normalize", "apply_blocker_remediation": "apply_blocker_remediation", "finish_blocked": "finish_blocked"})
    builder.add_edge("apply_blocker_remediation", "load_and_normalize")
    builder.add_conditional_edges("human_data_review", route_after_data_review, {"load_and_normalize": "load_and_normalize", "build_candidates": "build_candidates", "finish_blocked": "finish_blocked"})
    builder.add_conditional_edges("clarify_data", route_after_clarification, {"human_data_review": "human_data_review", "build_candidates": "build_candidates"})
    builder.add_conditional_edges("build_candidates", lambda state: "finish_blocked" if not state.get("candidates") else "baseline_evaluate", {"finish_blocked": "finish_blocked", "baseline_evaluate": "baseline_evaluate"})
    builder.add_edge("baseline_evaluate", "select_and_approve_plan")
    builder.add_conditional_edges("select_and_approve_plan", route_after_plan, {"prepare_candidate": "prepare_candidate", "finish_cancelled": "finish_cancelled"})
    builder.add_edge("prepare_candidate", "submit_training")
    builder.add_edge("submit_training", "wait_for_training")
    builder.add_conditional_edges("wait_for_training", route_after_training, {"verify_artifacts": "verify_artifacts", "assess_retry": "assess_retry"})
    builder.add_conditional_edges("verify_artifacts", route_after_verification, {"evaluate_tuned": "evaluate_tuned", "assess_retry": "assess_retry"})
    builder.add_conditional_edges("evaluate_tuned", lambda state: "advance_candidate" if state.get("candidate_status") == "evaluated" else "assess_retry", {"advance_candidate": "advance_candidate", "assess_retry": "assess_retry"})
    builder.add_conditional_edges("assess_retry", route_after_retry, {"submit_training": "submit_training", "advance_candidate": "advance_candidate", "finish_failed": "finish_failed"})
    builder.add_conditional_edges("advance_candidate", route_after_advance, {"prepare_candidate": "prepare_candidate", "compare_final": "compare_final"})
    builder.add_edge("compare_final", END)
    builder.add_edge("finish_blocked", END)
    builder.add_edge("finish_cancelled", END)
    builder.add_edge("finish_failed", END)
    return builder.compile(checkpointer=checkpointer)
