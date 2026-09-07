"""WSL-local API for ForgeTune. Run with ./start.sh from this folder."""
from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from advisor import baseline_compare, cloud_clarify, cloud_status, connection_test, load_source, local_profile, rank_candidates
from graph import WorkflowServices, build_workflow_graph
from langgraph.types import Command

try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
except ImportError:  # pragma: no cover - requirements install the SQLite saver in production.
    AsyncSqliteSaver = None
from langgraph.checkpoint.memory import MemorySaver

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
DATA_ROOT = ROOT / "data"
RUNS_ROOT = DATA_ROOT / "runs"
DATASETS_ROOT = DATA_ROOT / "datasets"
WORKFLOWS_ROOT = DATA_ROOT / "workflows"
WORKFLOW_DB = DATA_ROOT / "workflows.sqlite"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
WORKFLOWS_ROOT.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="ForgeTune Local Trainer", version="0.1.0")
queue_lock = asyncio.Lock()
active_processes: dict[str, asyncio.subprocess.Process] = {}
active_workflows: dict[str, asyncio.Task] = {}


class DatasetRequest(BaseModel):
    source: Literal["upload", "huggingface"]
    name: str = Field(min_length=1)
    extension: str | None = None
    split: str = "train"


class Hyperparameters(BaseModel):
    rank: int = Field(default=16, ge=1, le=128)
    alpha: int = Field(default=32, ge=1, le=256)
    learning_rate: float = Field(default=2e-4, gt=0, le=1e-2)
    epochs: int = Field(default=3, ge=1, le=10)
    batch_size: int = Field(default=1, ge=1, le=16)
    gradient_accumulation: int = Field(default=8, ge=1, le=128)
    max_sequence_length: int = Field(default=2048, ge=128, le=8192)


class RunRequest(BaseModel):
    dataset: DatasetRequest
    base_model: str
    parameter_method: Literal["lora", "qlora"]
    hyperparameters: Hyperparameters = Field(default_factory=Hyperparameters)
    test_dataset: DatasetRequest | None = None
    split_seed: int = 42
    workflow_id: str | None = None
    candidate_id: str | None = None
    attempt: int = 0
    idempotency_key: str | None = None
    split_manifest_id: str | None = None
    split_manifest_path: str | None = None


class WorkflowRequest(BaseModel):
    dataset: DatasetRequest
    goal: Literal["speed", "balanced", "quality"] = "balanced"
    cloud_assist: bool = False
    test_dataset: DatasetRequest | None = None
    split_seed: int = 42
    candidate_limit: int = Field(default=3, ge=1, le=3)
    max_retries: int = Field(default=2, ge=0, le=2)


class WorkflowResumeRequest(BaseModel):
    response: dict


class AnalysisRequest(BaseModel):
    dataset: DatasetRequest
    goal: Literal["speed", "balanced", "quality"] = "balanced"
    cloud_assist: bool = False


class BaselineRequest(BaseModel):
    dataset: DatasetRequest
    goal: Literal["speed", "balanced", "quality"] = "balanced"
    candidate_ids: list[str] = Field(default_factory=list, max_length=3)


def run_path(run_id: str) -> Path:
    return RUNS_ROOT / run_id / "run.json"


def read_run(run_id: str) -> dict:
    path = run_path(run_id)
    if not path.exists():
        raise HTTPException(404, "Training run not found")
    return json.loads(path.read_text(encoding="utf-8"))


def write_run(record: dict) -> None:
    path = run_path(record["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def gpu_status() -> dict:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
            timeout=5,
        ).strip().splitlines()[0]
        name, memory = [part.strip() for part in output.split(",", 1)]
        memory_gb = round(float(memory.split()[0]) / 1024)
        return {"name": name, "memoryGb": memory_gb, "trainerOnline": True}
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        return {"name": "NVIDIA GPU unavailable", "memoryGb": 0, "trainerOnline": True}


@app.get("/hardware")
async def hardware() -> dict:
    return gpu_status()


@app.get("/models")
async def model_catalog() -> list[dict]:
    return [
        {"id": "Qwen/Qwen2.5-3B-Instruct", "sizeB": 3, "recommendedVramGb": 8, "methods": ["lora", "qlora"]},
        {"id": "Qwen/Qwen2.5-7B-Instruct", "sizeB": 7, "recommendedVramGb": 13, "methods": ["lora", "qlora"]},
        {"id": "Qwen/Qwen2.5-Coder-7B-Instruct", "sizeB": 7, "recommendedVramGb": 13, "methods": ["lora", "qlora"]},
        {"id": "Qwen/Qwen2.5-14B-Instruct", "sizeB": 14, "recommendedVramGb": 22, "methods": ["qlora"]},
    ]


@app.post("/datasets/validate")
async def validate_dataset(request: DatasetRequest) -> dict:
    if request.source == "upload":
        extension = (request.extension or Path(request.name).suffix.lstrip(".")).lower()
        if extension not in {"csv", "jsonl", "parquet"}:
            raise HTTPException(422, "Supported file types are CSV, JSONL, and Parquet.")
        return {"valid": True, "format": extension, "schema": "pending file upload", "suggestedSplit": "90/10"}
    if "/" not in request.name:
        raise HTTPException(422, "Use a Hugging Face dataset ID such as owner/dataset.")
    return {"valid": True, "format": "huggingface", "schema": "inspect on import", "suggestedSplit": "90/10"}


@app.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict:
    extension = Path(file.filename or "").suffix.lower().lstrip(".")
    if extension not in {"csv", "jsonl", "parquet"}:
        raise HTTPException(422, "Supported file types are CSV, JSONL, and Parquet.")
    destination = DATASETS_ROOT / (str(uuid.uuid4()) + "." + extension)
    with destination.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    return {
        "source": "upload",
        "name": str(destination),
        "displayName": file.filename,
        "extension": extension,
        "format": extension,
        "suggestedSplit": "90/10",
    }


@app.get("/advisor/status")
async def advisor_status() -> dict:
    return cloud_status()


@app.post("/advisor/connection-test")
async def advisor_connection_test() -> dict:
    return connection_test()


@app.post("/datasets/analyze")
async def analyze_dataset(request: AnalysisRequest) -> dict:
    try:
        dataset = load_source(request.dataset.source, request.dataset.name, request.dataset.extension)
        analysis = local_profile(dataset)
    except Exception as error:
        # Dataset libraries raise several provider-specific exception classes
        # (for example legacy dataset-script errors). They are invalid inputs,
        # not trainer crashes, so return the actionable message to step 1.
        raise HTTPException(422, str(error)) from error
    cloud = None
    if request.cloud_assist and analysis["classification"]["confidence"] < .8:
        cloud = cloud_clarify(analysis["_samples"], analysis)
        analysis["redaction"]["sentToCloud"] = cloud is not None
        if cloud and cloud.get("task") in {"assistant", "writing", "code", "structured", "classification"}:
            analysis["classification"] = {
                "task": cloud["task"],
                "confidence": min(.9, float(cloud.get("confidence", .6))),
                "source": "cloud-assisted",
                "explanation": cloud.get("explanation", ""),
            }
    candidates = rank_candidates(analysis, request.goal, gpu_status()["memoryGb"] or 24)
    blockers = [item for item in analysis["findings"] if item["severity"] == "blocker"]
    recommended = next((item for item in candidates if not item["rejected"]), None)
    return {
        "facts": analysis["facts"],
        "classification": analysis["classification"],
        "findings": analysis["findings"],
        "redaction": analysis["redaction"],
        "cloud": {"requested": request.cloud_assist, "used": cloud is not None, "status": "not checked" if not request.cloud_assist else "used" if cloud else "unavailable_or_abstained"},
        "candidates": candidates,
        "blockers": blockers,
        "recommendation": None if blockers else recommended,
    }


@app.post("/datasets/analyze/baseline")
async def analyze_baseline(request: BaselineRequest) -> dict:
    """Run a small local zero-shot base-model comparison on a fixed representative slice."""
    analysis = await analyze_dataset(AnalysisRequest(dataset=request.dataset, goal=request.goal))
    eligible = [item for item in analysis["candidates"] if not item["rejected"]]
    if request.candidate_ids:
        eligible = [item for item in eligible if item["model"]["id"] in request.candidate_ids]
    shortlisted = eligible[: min(3, len(eligible))]
    if not shortlisted:
        raise HTTPException(422, "No model is feasible for this dataset and GPU.")
    try:
        dataset = load_source(request.dataset.source, request.dataset.name, request.dataset.extension)
        comparisons = baseline_compare(dataset, shortlisted, min(100, max(20, analysis["facts"]["validRows"] // 10)))
    except Exception as error:
        comparisons = [{"status": "unavailable", "message": str(error)[:300]}]
    for index, candidate in enumerate(shortlisted):
        candidate["baseline"] = next((item for item in comparisons if item.get("modelId") == candidate["model"]["id"]), {"status": "unavailable"})
        candidate["baseline"]["rank"] = index + 1
    completed = [item for item in shortlisted if item["baseline"]["status"] == "complete"]
    selected = max(completed, key=lambda item: (item["baseline"].get("meanTokenOverlap") or -1, -item["model"]["sizeB"])) if completed else shortlisted[0]
    return {"shortlist": shortlisted, "finalSelection": selected, "selectionReason": "Same held-out prompt slice compared locally. Token overlap is available only for completion datasets; review output samples before training."}


async def run_training(run_id: str) -> None:
    async with queue_lock:
        record = read_run(run_id)
        record.update({"status": "running", "startedAt": datetime.now(timezone.utc).isoformat(), "framework": "Transformers + TRL + PEFT"})
        write_run(record)
        command = [sys.executable, str(ROOT / "train.py"), "--run-dir", str(run_path(run_id).parent)]
        process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        active_processes[run_id] = process
        log_path = run_path(run_id).parent / "training.log"
        with log_path.open("w", encoding="utf-8") as log:
            while line := await process.stdout.readline():
                log.write(line.decode(errors="replace"))
        code = await process.wait()
        active_processes.pop(run_id, None)
        record = read_run(run_id)
        if record["status"] == "cancelled":
            return
        record.update({"status": "complete" if code == 0 else "failed", "completedAt": datetime.now(timezone.utc).isoformat()})
        if code == 0:
            metrics_path = run_path(run_id).parent / "metrics.json"
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                record["metrics"] = {
                    "eval_loss": metrics.get("eval_loss"),
                    "perplexity": round(float(__import__("math").exp(metrics["eval_loss"])), 3) if metrics.get("eval_loss") is not None else None,
                    "classification": metrics.get("classification"),
                }
                record["framework"] = metrics.get("framework", record["framework"])
            record["artifacts"] = {"adapter": "adapter/", "mergedModel": "merged/", "ollamaModelfile": "exports/Modelfile"}
        write_run(record)


@app.post("/runs", status_code=202)
async def create_run(request: RunRequest) -> dict:
    if "14B" in request.base_model and request.parameter_method != "qlora":
        raise HTTPException(422, "14B models require QLoRA on the configured 24 GB GPU.")
    if request.idempotency_key:
        for path in RUNS_ROOT.glob("*/run.json"):
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if existing.get("idempotencyKey") == request.idempotency_key:
                return existing
    run_id = str(uuid.uuid4())
    record = {
        "id": run_id,
        "status": "queued",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "config": request.model_dump(),
        "hardware": gpu_status(),
        "metrics": {},
        "artifacts": {},
        "workflowId": request.workflow_id,
        "candidateId": request.candidate_id,
        "attempt": request.attempt,
        "idempotencyKey": request.idempotency_key,
        "splitManifestId": request.split_manifest_id,
        "splitManifestPath": request.split_manifest_path,
    }
    write_run(record)
    asyncio.create_task(run_training(run_id))
    return record


def workflow_path(workflow_id: str) -> Path:
    return WORKFLOWS_ROOT / workflow_id / "workflow.json"


def read_workflow(workflow_id: str) -> dict:
    path = workflow_path(workflow_id)
    if not path.exists():
        raise HTTPException(404, "Workflow not found")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise HTTPException(500, "Workflow metadata is invalid") from error


def write_workflow(record: dict) -> None:
    path = workflow_path(record["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


async def submit_workflow_training(spec: dict) -> dict:
    test_dataset = spec.get("testDataset")
    request = RunRequest(
        dataset=DatasetRequest(**spec["dataset"]),
        base_model=spec["baseModel"],
        parameter_method=spec["parameterMethod"],
        hyperparameters=Hyperparameters(**spec["hyperparameters"]),
        test_dataset=DatasetRequest(**test_dataset) if test_dataset else None,
        split_seed=int(spec.get("splitSeed", 42)),
        workflow_id=spec.get("workflowId"),
        candidate_id=spec.get("candidateId"),
        attempt=int(spec.get("attempt", 0)),
        idempotency_key=spec.get("idempotencyKey"),
        split_manifest_id=spec.get("splitManifestId"),
        split_manifest_path=spec.get("splitManifestPath"),
    )
    return await create_run(request)


async def get_workflow_training(run_id: str) -> dict:
    return read_run(run_id)


def workflow_training_dir(run_id: str) -> Path:
    return run_path(run_id).parent


workflow_services = WorkflowServices(
    data_root=DATA_ROOT,
    gpu_memory_gb=lambda: gpu_status()["memoryGb"],
    submit_training=submit_workflow_training,
    get_training=get_workflow_training,
    training_dir=workflow_training_dir,
)
workflow_graph = build_workflow_graph(workflow_services, MemorySaver())
workflow_checkpointer_context = None


@app.on_event("startup")
async def initialize_workflow_checkpointer() -> None:
    """Use durable async SQLite persistence when the trainer is fully installed."""
    global workflow_graph, workflow_checkpointer_context
    if AsyncSqliteSaver is None:
        return
    try:
        workflow_checkpointer_context = AsyncSqliteSaver.from_conn_string(str(WORKFLOW_DB))
        saver = await workflow_checkpointer_context.__aenter__()
        workflow_graph = build_workflow_graph(workflow_services, saver)
    except Exception:
        workflow_checkpointer_context = None


@app.on_event("shutdown")
async def close_workflow_checkpointer() -> None:
    if workflow_checkpointer_context is not None:
        await workflow_checkpointer_context.__aexit__(None, None, None)


def _interrupt_value(result: dict) -> dict | None:
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    value = getattr(interrupts[0], "value", interrupts[0])
    return value if isinstance(value, dict) else {"type": "human_review", "message": str(value)}


async def execute_workflow(workflow_id: str, resume_response: dict | None = None) -> None:
    record = read_workflow(workflow_id)
    record["status"] = "running"
    record["pendingAction"] = None
    write_workflow(record)
    config = {"configurable": {"thread_id": workflow_id}}
    try:
        if resume_response is None:
            result = await workflow_graph.ainvoke(record["state"], config=config)
        else:
            result = await workflow_graph.ainvoke(Command(resume=resume_response), config=config)
        state_update = {key: value for key, value in result.items() if key != "__interrupt__"}
        state = {**record.get("state", {}), **state_update}
        try:
            snapshot = await workflow_graph.aget_state(config)
            checkpoint_values = getattr(snapshot, "values", None)
            if isinstance(checkpoint_values, dict):
                state = {**state, **checkpoint_values}
        except (AttributeError, TypeError):
            # The result still contains enough state for in-memory/test savers.
            pass
        pending = _interrupt_value(result)
        record["state"] = state
        if pending:
            record["status"] = "waiting"
            record["pendingAction"] = pending
            state["pending_action"] = pending
        else:
            record["status"] = state.get("status", "complete")
            record["pendingAction"] = None
        write_workflow(record)
    except Exception as error:
        record["status"] = "failed"
        record["pendingAction"] = None
        record["error"] = str(error)[:500]
        record.setdefault("state", {}).setdefault("errors", []).append(str(error)[:500])
        write_workflow(record)
    finally:
        active_workflows.pop(workflow_id, None)


@app.post("/workflow/runs", status_code=202)
async def create_workflow(request: WorkflowRequest) -> dict:
    workflow_id = str(uuid.uuid4())
    initial_state = {
        "workflow_id": workflow_id,
        "dataset": request.dataset.model_dump(),
        "test_dataset": request.test_dataset.model_dump() if request.test_dataset else None,
        "goal": request.goal,
        "cloud_assist": request.cloud_assist,
        "split_seed": request.split_seed,
        "max_candidates": request.candidate_limit,
        "max_retries": request.max_retries,
        "status": "queued",
        "attempts": {},
        "training_runs": {},
        "evaluation_results": [],
        "errors": [],
    }
    record = {"id": workflow_id, "status": "queued", "createdAt": datetime.now(timezone.utc).isoformat(), "state": initial_state, "pendingAction": None}
    write_workflow(record)
    active_workflows[workflow_id] = asyncio.create_task(execute_workflow(workflow_id))
    return record


@app.get("/workflow/runs")
async def list_workflows() -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(WORKFLOWS_ROOT.glob("*/workflow.json"), reverse=True)]


@app.get("/workflow/runs/{workflow_id}")
async def get_workflow(workflow_id: str) -> dict:
    record = read_workflow(workflow_id)
    # Record status is authoritative; state may still contain the initial status
    # when a graph invocation fails before it can checkpoint an update.
    return {**record.get("state", {}), **record}


@app.post("/workflow/runs/{workflow_id}/resume", status_code=202)
async def resume_workflow(workflow_id: str, request: WorkflowResumeRequest) -> dict:
    record = read_workflow(workflow_id)
    if record.get("status") != "waiting":
        raise HTTPException(409, "Workflow is not waiting for human input.")
    if workflow_id in active_workflows:
        raise HTTPException(409, "Workflow is already running.")
    record["status"] = "queued"
    write_workflow(record)
    active_workflows[workflow_id] = asyncio.create_task(execute_workflow(workflow_id, request.response))
    return record


@app.get("/runs")
async def list_runs() -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(RUNS_ROOT.glob("*/run.json"), reverse=True)]


@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    return read_run(run_id)


@app.get("/runs/{run_id}/logs")
async def get_logs(run_id: str) -> dict:
    log = run_path(run_id).parent / "training.log"
    return {"lines": log.read_text(encoding="utf-8").splitlines()[-200:] if log.exists() else []}


@app.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> dict:
    record = read_run(run_id)
    process = active_processes.get(run_id)
    if process:
        process.terminate()
    record["status"] = "cancelled"
    write_run(record)
    return record
