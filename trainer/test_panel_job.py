import json
import threading
import time

import pytest

import panel_job
from panel_job import STEPS, new_record, read_grades, run_panel
from quality_store import QualityStore
from test_judge_panel import make_comparison, make_evaluation

REVISIONS = {"breeze": "b" * 40, "qwen": "q" * 40, "yi": "y" * 40}
VOTE = {"correctness": {"verdict": "pass", "reason": "ok"}, "language": {"verdict": "pass", "reason": "ok"}}


@pytest.fixture(autouse=True)
def ready(monkeypatch):
    monkeypatch.setattr(panel_job, "judge_status", lambda root: {n: {"modelId": n, "revision": r, "status": "ready", "path": "/m"} for n, r in REVISIONS.items()})


def setup(tmp_path, prompts=None, source=None):
    store = QualityStore(tmp_path)
    source = source(store) if source else {"comparisonId": make_comparison(store, cases=25)}
    record = store.put("panel", new_record(source, prompts, "fixture"))
    return store, record["id"]


class Judge:
    def __init__(self, gpus=("GPU-a", "GPU-b"), hook=None):
        self.gpus, self.hook, self.calls, self.lock = list(gpus), hook, [], threading.Lock()

    def __call__(self, items, *, runtime_root, judge, grades_path, reference_mode, cancelled, max_seconds, gpu, revision):
        start = time.monotonic()
        result = self.hook(judge, items, grades_path) if self.hook else None
        if result is None:
            with open(grades_path, "a", encoding="utf-8") as out:
                out.writelines(json.dumps({"itemId": i["itemId"], "vote": VOTE}) + "\n" for i in items)
            result = {"status": "complete", "graded": 0, "run": {"chargedGpuSeconds": 5}, "budget": {}}
        with self.lock:
            self.calls.append({"judge": judge, "ids": [i["itemId"] for i in items], "gpu": gpu, "max_seconds": max_seconds,
                               "start": start, "end": time.monotonic(), "reference_mode": reference_mode, "revision": revision})
        return result

    def run(self, store, root, panel_id, stop=None):
        run_panel(store, root, panel_id, stop or threading.Event(), gpus=lambda: self.gpus, judge=self)
        return store.get("panel", panel_id)

    def judged(self, name):
        return [c for c in self.calls if c["judge"] == name]


def write_lines(path, items, count):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as out:
        out.writelines(json.dumps({"itemId": i["itemId"], "vote": VOTE}) + "\n" for i in items[:count])


def folder(root, panel_id):
    return root / "quality" / "panels" / panel_id


def test_panel_runs_all_judges_and_reports(tmp_path):
    store, pid = setup(tmp_path)
    fake = Judge()
    record = fake.run(store, tmp_path, pid)
    assert record["status"] == "complete", record
    assert [s["name"] for s in record["steps"]] == list(STEPS) and all(s["status"] == "complete" and s["finishedAt"] for s in record["steps"])
    for name in ("breeze", "qwen", "yi"):
        lines = (folder(tmp_path, pid) / f"grades-{name}.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 60
        step = next(s for s in record["steps"] if s["name"] == f"judge-{name}")
        assert step["graded"] == step["total"] == 60 and step["gpuSeconds"] == 5
    items = [json.loads(line) for line in (folder(tmp_path, pid) / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(items) == 60 and all("human" in i and i["human"] is None for i in items)
    assert record["sourceFingerprint"] == panel_job.digest(items) and record["referenceMode"] == "none"
    assert record["judgeRevisions"] == REVISIONS and record["evaluatorId"] == panel_job.evaluator_id(REVISIONS, "none")
    assert record["selfGrading"] == {} and {c["reference_mode"] for c in fake.calls} == {"none"}
    assert {c["judge"]: c["revision"] for c in fake.calls} == REVISIONS
    assert record["truncatedItems"] == 20 and record["truncatedUnknown"] is False
    assert {c["max_seconds"] for c in fake.calls} == {panel_job.chunk_seconds(60)} == {1215}
    protocol = record["protocol"]
    assert set(protocol) == {"panelVersion", "aggregationRule", "system", "correctnessRubric", "languageRubric", "truncationRule"}
    assert protocol["panelVersion"] == "judge-panel-v1" and protocol["system"] == panel_job.SYSTEM and "reference" not in protocol["correctnessRubric"].lower()
    report = json.loads((folder(tmp_path, pid) / "report.json").read_text(encoding="utf-8"))
    assert report == record["report"] and set(report["candidates"]) == {"base", "original", "reviewed"}
    assert report["incomplete"] is False and report["candidates"]["base"]["correctness"]["pass"] == 20


def test_evaluation_source_freezes_human_and_reports_agreement(tmp_path):
    store, pid = setup(tmp_path, source=lambda store: {"evaluationId": make_evaluation(store)})
    record = Judge().run(store, tmp_path, pid)
    assert record["status"] == "complete" and record["referenceMode"] == "case"
    assert record["truncatedItems"] == 0 and record["truncatedUnknown"] is True
    assert record["selfGrading"] == {"Qwen2.5-7B@base": "qwen"}
    assert record["protocol"]["correctnessRubric"] == panel_job.CORRECTNESS_RUBRIC("case") and "reference" in record["protocol"]["correctnessRubric"]
    assert record["report"]["humanAgreement"] == {"compared": 2, "agreed": 1}


def test_breeze_and_qwen_run_in_parallel_on_two_gpus(tmp_path):
    store, pid = setup(tmp_path)
    barrier = threading.Barrier(2, timeout=10)

    def meet(judge, items, path):
        if judge != "yi":
            barrier.wait()

    fake = Judge(hook=meet)
    record = fake.run(store, tmp_path, pid)
    assert record["status"] == "complete", record
    breeze, qwen, yi = (fake.judged(n)[0] for n in ("breeze", "qwen", "yi"))
    assert {breeze["gpu"], qwen["gpu"]} == {"GPU-a", "GPU-b"} and yi["gpu"] == "GPU-a"
    assert breeze["start"] < qwen["end"] and qwen["start"] < breeze["end"]
    assert yi["start"] >= max(breeze["end"], qwen["end"])


def test_one_gpu_runs_judges_sequentially(tmp_path):
    store, pid = setup(tmp_path)
    fake = Judge(gpus=["GPU-a"])
    assert fake.run(store, tmp_path, pid)["status"] == "complete"
    assert [c["judge"] for c in fake.calls] == ["breeze", "qwen", "yi"] and {c["gpu"] for c in fake.calls} == {"GPU-a"}


def test_large_selection_is_chunked(tmp_path, monkeypatch):
    monkeypatch.setattr(panel_job, "chunk_size", lambda: 30)
    store, pid = setup(tmp_path, prompts=25)
    fake = Judge()
    record = fake.run(store, tmp_path, pid)
    assert record["status"] == "complete", record
    for name in ("breeze", "qwen", "yi"):
        calls = fake.judged(name)
        assert [len(c["ids"]) for c in calls] == [30, 30, 15]
        assert len({i for c in calls for i in c["ids"]}) == 75
        assert [c["max_seconds"] for c in calls] == [panel_job.chunk_seconds(n) for n in (30, 30, 15)] == [675, 675, 405]
        assert next(s for s in record["steps"] if s["name"] == f"judge-{name}")["gpuSeconds"] == 15


def test_resume_skips_graded_items_and_finished_steps(tmp_path):
    store, pid = setup(tmp_path)
    stop = threading.Event()

    def qwen_stops(judge, items, path):
        if judge == "qwen":
            write_lines(path, items, 10)
            stop.set()
            return {"status": "cancelled", "graded": 10, "run": {"chargedGpuSeconds": 2}, "budget": {}}

    first = Judge(gpus=["GPU-a"], hook=qwen_stops)
    record = first.run(store, tmp_path, pid, stop)
    assert record["status"] == "running" and [c["judge"] for c in first.calls] == ["breeze", "qwen"]
    steps = {s["name"]: s for s in record["steps"]}
    assert steps["judge-breeze"]["status"] == "complete" and steps["judge-qwen"]["status"] == "running"
    graded = set(read_grades(folder(tmp_path, pid) / "grades-qwen.jsonl"))
    assert len(graded) == 10

    record["status"] = "running"
    store.put("panel", record)
    second = Judge(gpus=["GPU-a"])
    record = second.run(store, tmp_path, pid)
    assert record["status"] == "complete", record
    assert second.judged("breeze") == []
    (qwen,) = second.judged("qwen")
    assert len(qwen["ids"]) == 50 and not graded & set(qwen["ids"])
    assert len((folder(tmp_path, pid) / "grades-qwen.jsonl").read_text(encoding="utf-8").splitlines()) == 60
    assert {s["name"]: s["gpuSeconds"] for s in record["steps"]}["judge-qwen"] == 7


def test_failed_judge_keeps_partial_report(tmp_path):
    store, pid = setup(tmp_path)
    fake = Judge(hook=lambda judge, items, path: {"status": "unavailable", "reason": "missing_weights", "message": "Yi is not ready.",
                                                  "graded": 0, "run": None, "budget": {}} if judge == "yi" else None)
    record = fake.run(store, tmp_path, pid)
    assert record["status"] == "failed" and record["message"].startswith("judge-yi: ") and "Yi is not ready." in record["message"]
    assert len(fake.judged("yi")) == 2
    steps = {s["name"]: s["status"] for s in record["steps"]}
    assert steps == {"prepare": "complete", "judge-breeze": "complete", "judge-qwen": "complete", "judge-yi": "failed", "report": "pending"}
    report = record["report"]
    assert report["incomplete"] is True and report == json.loads((folder(tmp_path, pid) / "report.json").read_text(encoding="utf-8"))
    assert len(report["items"]) == 60
    assert all(i["votes"]["breeze"] == VOTE and i["votes"]["qwen"] == VOTE and i["votes"]["yi"] == {"error": "missing"} for i in report["items"])
    assert report["candidates"]["base"]["correctness"]["rate"] is None and report["candidates"]["base"]["language"]["pass"] == 20


def test_retry_recovers_after_one_failed_chunk(tmp_path):
    store, pid = setup(tmp_path)
    failures = iter([{"status": "failed", "message": "Worker crashed.", "graded": 0, "run": {"chargedGpuSeconds": 3}, "budget": {}}])
    fake = Judge(gpus=["GPU-a"], hook=lambda judge, items, path: next(failures, None) if judge == "qwen" else None)
    record = fake.run(store, tmp_path, pid)
    assert record["status"] == "complete" and len(fake.judged("qwen")) == 2
    assert {s["name"]: s["gpuSeconds"] for s in record["steps"]}["judge-qwen"] == 8


def test_chunks_that_make_progress_do_not_use_up_the_retry(tmp_path):
    results = iter(["budget_exhausted", "failed", "budget_exhausted"])

    def partial(judge, items, path):
        status = next(results, None) if judge == "qwen" else None
        if status:
            write_lines(path, items, 10)
            return {"status": status, "message": "Deadline.", "graded": 10, "run": {"chargedGpuSeconds": 1}, "budget": {}}

    fake = Judge(gpus=["GPU-a"], hook=partial)
    store, pid = setup(tmp_path)
    record = fake.run(store, tmp_path, pid)
    assert record["status"] == "complete", record
    assert [len(c["ids"]) for c in fake.judged("qwen")] == [60, 50, 40, 30]
    assert len(read_grades(folder(tmp_path, pid) / "grades-qwen.jsonl")) == 60


def test_two_no_progress_chunks_in_a_row_fail_the_step(tmp_path):
    results = iter([10, 0, 0])

    def stalls(judge, items, path):
        if judge != "qwen":
            return None
        write_lines(path, items, next(results))
        return {"status": "failed", "message": "Worker crashed.", "graded": 0, "run": {"chargedGpuSeconds": 1}, "budget": {}}

    fake = Judge(gpus=["GPU-a"], hook=stalls)
    store, pid = setup(tmp_path)
    record = fake.run(store, tmp_path, pid)
    assert record["status"] == "failed" and record["message"] == "judge-qwen: Worker crashed."
    assert len(fake.judged("qwen")) == 3 and fake.judged("yi") == []


def test_ledger_short_before_chunk_fails_step(tmp_path, monkeypatch):
    store, pid = setup(tmp_path)
    remaining = iter([15000])
    monkeypatch.setattr(panel_job, "budget_status", lambda root: {"remainingGpuSeconds": next(remaining, 10.7)})
    fake = Judge(gpus=["GPU-a"])
    record = fake.run(store, tmp_path, pid)
    assert record["status"] == "failed" and fake.calls == []
    assert "GPU ledger has 10 s left; this chunk needs 1215 s" in record["message"]
    assert record["report"]["incomplete"] is True


def test_judge_revision_change_fails_step(tmp_path, monkeypatch):
    store, pid = setup(tmp_path)
    fake = Judge(gpus=["GPU-a"])

    def repinned(judge, items, path):
        monkeypatch.setattr(panel_job, "judge_status", lambda root: {n: {"revision": "z" * 40, "status": "ready"} for n in REVISIONS})

    fake.hook = repinned
    record = fake.run(store, tmp_path, pid)
    assert record["status"] == "failed" and "revision changed" in record["message"]
    assert [c["judge"] for c in fake.calls] == ["breeze"]


def test_cancel_marks_cancelled_and_keeps_grades(tmp_path):
    store, pid = setup(tmp_path)
    stop = threading.Event()

    def cancel(judge, items, path):
        write_lines(path, items, 5)
        store.cancel("panel", pid)
        stop.set()
        return {"status": "cancelled", "message": "Cancelled.", "graded": 5, "run": {"chargedGpuSeconds": 1}, "budget": {}}

    fake = Judge(gpus=["GPU-a"], hook=cancel)
    record = fake.run(store, tmp_path, pid, stop)
    assert record["status"] == "cancelled" and len(fake.calls) == 1
    assert [s["status"] for s in record["steps"]] == ["complete", "cancelled", "pending", "pending", "pending"]
    assert len(read_grades(folder(tmp_path, pid) / "grades-breeze.jsonl")) == 5


def test_no_idle_gpu_fails_without_spinning(tmp_path):
    store, pid = setup(tmp_path)
    fake = Judge(gpus=[])
    record = fake.run(store, tmp_path, pid)
    assert record["status"] == "failed" and "gpu_busy" in record["message"] and fake.calls == []
    assert "start a new panel" in record["message"] and "start the panel again" not in record["message"]


def test_read_grades_ignores_torn_trailing_line(tmp_path):
    path = tmp_path / "grades.jsonl"
    path.write_text(json.dumps({"itemId": "a", "vote": VOTE}) + "\n" + '{"itemId": "b", "vo', encoding="utf-8")
    assert read_grades(path) == {"a": VOTE} and read_grades(tmp_path / "missing.jsonl") == {}
