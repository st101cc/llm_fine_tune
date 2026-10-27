"""Resumable, cancellable three-judge panel job: chunked, GPU-budgeted grading of a frozen item set."""
import json
import math
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from judge_panel import (AGGREGATION_RULE, CORRECTNESS_RUBRIC, JUDGES, LANGUAGE_RUBRIC, PANEL_VERSION, SYSTEM, TRUNCATION_RULE, build_report, chunk_seconds,
                         chunk_size, estimate, evaluator_id, load_source, sample_items, self_grading_warnings)
from quality_models import judge_status
from quality_pilot import budget_status, idle_gpus, judge_bounded
from quality_store import digest, identifier

NAMES = tuple(j["name"] for j in JUDGES)
STEPS = ("prepare", "judge-breeze", "judge-qwen", "judge-yi", "report")
PHASES = (("prepare",), ("judge-breeze", "judge-qwen"), ("judge-yi",), ("report",))
GPU_BUSY = "gpu_busy: no idle GPU is available for this step; wait for an idle GPU and start a new panel."


def now():
    return datetime.now(timezone.utc).isoformat()


def new_record(source, prompts, display_name):
    step = {"status": "pending", "startedAt": None, "finishedAt": None, "gpuSeconds": 0, "graded": 0, "total": None, "message": None}
    return {"id": str(uuid.uuid4()), "source": source, "prompts": prompts, "displayName": display_name, "status": "queued", "createdAt": now(),
            "steps": [{"name": name, **step} for name in STEPS], "estimate": None, "report": None}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_grades(path):
    """Votes by itemId from complete lines only; a torn trailing fragment from a killed worker is ignored."""
    if not path.is_file():
        return {}
    data = path.read_bytes()
    return {row["itemId"]: row["vote"] for row in (json.loads(line) for line in data[:data.rfind(b"\n") + 1].decode("utf-8").splitlines() if line)}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_panel(store, root, panel_id, stop, *, gpus=idle_gpus, judge=judge_bounded):
    root = Path(root).resolve()
    directory = root / "quality" / "panels" / identifier(panel_id)
    record = store.get("panel", panel_id)
    if record["status"] in ("complete", "failed", "cancelled"):
        return
    steps = {s["name"]: s for s in record["steps"]}
    grades_path = lambda name: directory / f"grades-{name}.jsonl"

    lock = threading.Lock()

    def save():
        # The cancel API writes `cancelling` to the store concurrently; never overwrite it with `running`.
        with lock, store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM records WHERE kind='panel' AND id=?", (record["id"],)).fetchone()
            if row and json.loads(row[0]).get("status") == "cancelling" and record["status"] == "running":
                record["status"] = "cancelling"
            store.put("panel", record, db)

    def report(complete):
        items = read_jsonl(directory / "items.jsonl")
        human = {(i["caseId"], i["candidate"]): i["human"] for i in items if i["human"] is not None}
        record["report"] = build_report(items, {name: read_grades(grades_path(name)) for name in complete}, complete_judges=set(complete), human=human)
        write_json(directory / "report.json", record["report"])

    def prepare(step, gpu):
        source = load_source(store, record["source"])
        human = {(a["caseId"], a["candidate"]): a["human"] for a in source["answers"]}
        items = [{**i, "human": human.get((i["caseId"], i["candidate"]))} for i in sample_items(source, record["prompts"])]
        status = judge_status(root)
        plan = estimate(len(items), budget_status(root)["remainingGpuSeconds"], {n: s["status"] == "ready" for n, s in status.items()})
        if not plan["feasible"]:
            raise ValueError("; ".join(plan["reasons"]))
        revisions = {name: status[name]["revision"] for name in NAMES}
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "items.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
        record.update(estimate=plan, sourceFingerprint=digest(items), judgeRevisions=revisions, referenceMode=source["referenceMode"],
                      protocol={"panelVersion": PANEL_VERSION, "aggregationRule": AGGREGATION_RULE, "system": SYSTEM, "correctnessRubric": CORRECTNESS_RUBRIC(source["referenceMode"]),
                                "languageRubric": LANGUAGE_RUBRIC, "truncationRule": TRUNCATION_RULE},
                      evaluatorId=evaluator_id(revisions, source["referenceMode"]), selfGrading=self_grading_warnings(sorted({i["candidate"] for i in items})),
                      truncatedItems=sum(bool(i["truncated"]) for i in items), truncatedUnknown=source["truncatedUnknown"])
        for name in NAMES:
            steps[f"judge-{name}"]["total"] = len(items)

    def judge_step(step, gpu):
        name = step["name"].removeprefix("judge-")
        path = grades_path(name)
        items = read_jsonl(directory / "items.jsonl")
        failures = 0
        while True:
            done = read_grades(path)
            todo = [i for i in items if i["itemId"] not in done]
            step["graded"] = len(items) - len(todo)
            save()
            if not todo:
                return
            if stop.is_set():
                raise RuntimeError("Stopped.")
            if judge_status(root)[name]["revision"] != record["judgeRevisions"][name]:
                raise RuntimeError(f"{name} revision changed since this panel was prepared; start a new panel.")
            chunk = todo[:chunk_size()]
            need = chunk_seconds(len(chunk))
            left = budget_status(root)["remainingGpuSeconds"]
            if left < need:
                raise RuntimeError(f"GPU ledger has {math.floor(left)} s left; this chunk needs {need} s")
            result = judge(chunk, runtime_root=root, judge=name, grades_path=path, reference_mode=record["referenceMode"],
                           cancelled=stop.is_set, max_seconds=need, gpu=gpu, revision=record["judgeRevisions"][name])
            step["gpuSeconds"] += (result.get("run") or {}).get("chargedGpuSeconds") or 0
            # Only a chunk that graded nothing counts toward the one-retry limit; any progress resets it.
            if len(read_grades(path)) > len(done):
                failures = 0
                continue
            failures += 1
            if stop.is_set() or failures == 2:
                raise RuntimeError(result.get("message") or f"{name} chunk ended with status {result['status']}.")

    handlers = {"prepare": prepare, "judge": judge_step, "report": lambda step, gpu: report(NAMES)}

    def execute(step, gpu=None):
        step.update(status="running", startedAt=now(), finishedAt=None, message=None)
        save()
        try:
            handlers[step["name"].split("-")[0]](step, gpu)
            step.update(status="complete", finishedAt=now())
        except Exception as error:
            # A stopped step stays `running`: on shutdown the next start resumes it; on cancel it is closed below.
            if not stop.is_set():
                step.update(status="failed", finishedAt=now(), message=str(error))
        save()

    record["status"] = "running"
    save()

    for phase in PHASES:
        if stop.is_set() or record["status"] == "cancelling" or any(s["status"] == "failed" for s in record["steps"]):
            break
        pending = [steps[name] for name in phase if steps[name]["status"] != "complete"]
        if not pending:
            continue
        if not pending[0]["name"].startswith("judge-"):
            execute(pending[0])
            continue
        free = gpus()
        if not free:
            pending[0].update(status="failed", finishedAt=now(), message=GPU_BUSY)
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
        if failed["name"].startswith("judge-"):
            report([name for name in NAMES if steps[f"judge-{name}"]["status"] == "complete"])
        record.update(status="failed", message=f"{failed['name']}: {failed['message']}")
    elif record["status"] == "cancelling":
        record["status"] = "cancelled"
        for step in record["steps"]:
            if step["status"] == "running":
                step.update(status="cancelled", finishedAt=now())
    elif all(s["status"] == "complete" for s in record["steps"]):
        record.update(status="complete", message=None)
    save()
