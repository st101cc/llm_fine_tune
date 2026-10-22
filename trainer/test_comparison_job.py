import json
import threading
import time

import pytest

import comparison_job
from comparison_job import STEPS, new_record, run_comparison
from quality_store import QualityStore


@pytest.fixture(autouse=True)
def assets(monkeypatch):
    monkeypatch.setattr(comparison_job, "get_model_status", lambda *a, **k: {"status": "ready", "revision": "a" * 40, "path": "/model"})
    monkeypatch.setattr(comparison_job, "benchmark_revision", lambda root: "b" * 40)
    monkeypatch.setattr(comparison_job, "load_benchmark", lambda *a, **k: {"cases": [{"id": "m1", "task": "mcq", "input": "Q", "reference": "A"}]})


def setup(tmp_path):
    rows = [{"prompt": f"問題{i}", "completion": "這是數據" if i < 120 else "好"} for i in range(320)]
    (tmp_path / "datasets").mkdir(parents=True)
    path = tmp_path / "datasets" / "fixture.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    store = QualityStore(tmp_path)
    audit = store.create_audit({"source": "upload", "name": str(path), "extension": "jsonl"}, {"terms": {"數據": "資料"}}, "full")
    store.run_audit(audit["id"])
    audit = store.get("audit", audit["id"])
    store.accept_customer_terms(audit["id"], audit["fingerprint"])
    version = store.materialize(audit["id"], audit["fingerprint"])
    record = store.put("comparison", new_record(version["id"], {"epochs": 1, "evalPrompts": 20}))
    return store, record["id"]


class Fakes:
    def __init__(self, gpus=("GPU-a", "GPU-b"), train_status="complete", on_train=None):
        self.intervals, self.generated, self.gpus = [], [], list(gpus)
        self.train_status, self.on_train = train_status, on_train

    def train(self, root, commands, *, gpu, cancelled, max_seconds, log_dir, load_mode):
        assert load_mode == "bnb4" and max_seconds <= 1800 and commands[0][-2:] == ["--run-dir", str(log_dir)]
        start = time.monotonic()
        if self.on_train:
            return self.on_train()
        time.sleep(0.2)
        (log_dir / "adapter").mkdir()
        (log_dir / "adapter" / "adapter_model.safetensors").write_bytes(b"x")
        (log_dir / "metrics.json").write_text(json.dumps({"eval_loss": 1.0, "train_loss": 1.5}), encoding="utf-8")
        self.intervals.append((gpu, start, time.monotonic()))
        return {"status": self.train_status, "message": "gpu_busy: all GPUs busy" if self.train_status != "complete" else "ok", "chargedGpuSeconds": 7}

    def generate(self, cases, *, runtime_root, adapter_path, cancelled, max_seconds, revision, load_mode, writing_sample_tokens, gpu):
        assert load_mode == "bnb4" and writing_sample_tokens == 128 and revision == "a" * 40
        self.generated.append((adapter_path, gpu, [c["id"] for c in cases], max_seconds))
        return {"status": "complete", "answers": [{"caseId": c["id"], "output": "A 這是數據"} for c in cases], "run": {"chargedGpuSeconds": 3}}

    def run(self, store, root, comparison_id, stop=None):
        run_comparison(store, root, comparison_id, stop or threading.Event(), gpus=lambda: self.gpus, train=self.train, generate=self.generate, token_count=len)
        return store.get("comparison", comparison_id)


def overlap(a, b):
    return a[1] < b[2] and b[1] < a[2]


def test_happy_path_parallel_training_on_distinct_gpus(tmp_path):
    store, cid = setup(tmp_path)
    fakes = Fakes()
    record = fakes.run(store, tmp_path, cid)
    assert record["status"] == "complete", record
    assert [s["name"] for s in record["steps"]] == list(STEPS) and all(s["status"] == "complete" for s in record["steps"])
    a, b = fakes.intervals
    assert overlap(a, b) and a[0] != b[0]
    assert set(record["report"]["models"]) == {"base", "original", "reviewed"}
    assert record["report"]["models"]["original"]["trainLoss"] == 1.5
    assert set(record["report"]["runs"]) == {"original", "reviewed"}
    assert [g[0] is None for g in fakes.generated].count(True) == 1 and len(fakes.generated) == 3
    assert (tmp_path / "quality" / "comparisons" / cid / "eval" / "reviewed.json").is_file()
    assert record["steps"][1]["gpuSeconds"] == 7
    reserve = record["estimate"]["stepMaxSeconds"]
    assert reserve["evaluate-adapter"] > reserve["evaluate-base"]
    assert sorted(g[3] for g in fakes.generated) == sorted([reserve["evaluate-base"]] + [reserve["evaluate-adapter"]] * 2)
    assert [g[3] for g in fakes.generated if g[0] is None] == [reserve["evaluate-base"]]


def test_single_gpu_runs_sequentially(tmp_path):
    store, cid = setup(tmp_path)
    fakes = Fakes(gpus=["GPU-a"])
    assert fakes.run(store, tmp_path, cid)["status"] == "complete"
    a, b = fakes.intervals
    assert not overlap(a, b) and a[0] == b[0] == "GPU-a"
    assert {g[1] for g in fakes.generated} == {"GPU-a"}


def test_managed_runs_are_normal_run_records(tmp_path):
    store, cid = setup(tmp_path)
    record = Fakes().run(store, tmp_path, cid)
    for variant, step in (("original", record["steps"][1]), ("reviewed", record["steps"][2])):
        run = json.loads((tmp_path / "runs" / step["runId"] / "run.json").read_text(encoding="utf-8"))
        assert run["managedBy"] == cid and run["variant"] == variant and run["status"] == "complete"
        assert run["metrics"]["eval_loss"] == 1.0 and run["metrics"]["perplexity"] == 2.718
        assert run["qualityProvenance"] == {"comparisonId": cid, "versionId": record["versionId"], "variant": variant}
        assert run["config"]["dataset"]["name"].endswith(f"{variant}-train.jsonl") and run["config"]["dataset"]["displayName"].endswith(f"· {variant}")
        assert run["config"]["test_dataset"]["name"].endswith(f"dev-{variant}.jsonl")
        assert run["config"]["parameter_method"] == "qlora" and run["config"]["hyperparameters"]["rank"] == 16
        assert run["qualityModel"]["revision"] == "a" * 40 and run["artifacts"]["adapter"] == "adapter/"


def test_resume_skips_completed_and_retries_interrupted_once(tmp_path):
    store, cid = setup(tmp_path)
    Fakes().run(store, tmp_path, cid)
    record = store.get("comparison", cid)
    steps = {s["name"]: s for s in record["steps"]}
    steps["train-reviewed"].update(status="running", attempts=1)
    steps["evaluate-reviewed"].update(status="running", attempts=1)
    steps["report"].update(status="pending")
    record.update(status="running", report=None)
    store.put("comparison", record)
    fakes = Fakes()
    record = fakes.run(store, tmp_path, cid)
    assert record["status"] == "complete" and record["report"]
    assert fakes.intervals == [] and len(fakes.generated) == 1 and fakes.generated[0][0].name == "adapter"
    assert {s["name"]: s["attempts"] for s in record["steps"]}["evaluate-reviewed"] == 2

    steps = {s["name"]: s for s in record["steps"]}
    steps["evaluate-reviewed"].update(status="running", attempts=2)
    record.update(status="running")
    store.put("comparison", record)
    fakes = Fakes()
    record = fakes.run(store, tmp_path, cid)
    assert record["status"] == "failed" and fakes.generated == []
    assert {s["name"]: s["status"] for s in record["steps"]}["evaluate-reviewed"] == "failed"


def test_cancel_vs_shutdown(tmp_path):
    store, cid = setup(tmp_path)
    stop = threading.Event()

    def cancel():
        store.cancel("comparison", cid)
        stop.set()
        return {"status": "cancelled", "message": "Cancelled.", "chargedGpuSeconds": 1}

    record = Fakes(gpus=["GPU-a"], on_train=cancel).run(store, tmp_path, cid, stop)
    assert record["status"] == "cancelled"
    assert [s["status"] for s in record["steps"]][3:] == ["pending"] * 4

    store, cid = setup(tmp_path / "second")
    stop = threading.Event()

    def shutdown():
        stop.set()
        return {"status": "cancelled", "message": "Cancelled.", "chargedGpuSeconds": 1}

    record = Fakes(gpus=["GPU-a"], on_train=shutdown).run(store, tmp_path / "second", cid, stop)
    assert record["status"] == "running"
    assert record["steps"][1]["status"] == "running" and record["steps"][1]["attempts"] == 1


def test_step_failure_stops_job(tmp_path):
    store, cid = setup(tmp_path)
    fakes = Fakes(train_status="unavailable")
    record = fakes.run(store, tmp_path, cid)
    assert record["status"] == "failed" and "gpu_busy" in record["message"]
    assert [s["status"] for s in record["steps"]][3:] == ["pending"] * 4 and fakes.generated == []


def test_no_idle_gpu_fails_without_spinning(tmp_path):
    store, cid = setup(tmp_path)
    fakes = Fakes(gpus=[])
    record = fakes.run(store, tmp_path, cid)
    assert record["status"] == "failed" and "gpu_busy" in record["message"] and fakes.intervals == []


def test_ledger_shortfall_fails_step_before_launch(tmp_path, monkeypatch):
    store, cid = setup(tmp_path)
    remaining = iter([15000])
    monkeypatch.setattr(comparison_job, "budget_status", lambda root: {"remainingGpuSeconds": next(remaining, 10)})
    fakes = Fakes()
    record = fakes.run(store, tmp_path, cid)
    need = record["estimate"]["stepMaxSeconds"]["train"]
    assert record["status"] == "failed" and fakes.intervals == []
    assert f"GPU ledger has 10 s left; this step needs {need} s" in record["message"]


def test_token_limit_overrun_skips_case(tmp_path):
    store, cid = setup(tmp_path)
    fakes = Fakes()
    plain = fakes.generate

    def generate(cases, **kwargs):
        if cases[0]["id"] == "m1" and kwargs["adapter_path"] is None:
            plain(cases, **kwargs)
            return {"status": "unavailable", "message": "Output reached the token limit.", "answers": []}
        return plain(cases, **kwargs)

    fakes.generate = generate
    record = fakes.run(store, tmp_path, cid)
    assert record["status"] == "complete"
    saved = json.loads((tmp_path / "quality" / "comparisons" / cid / "eval" / "base.json").read_text(encoding="utf-8"))
    assert saved["overLimit"] == ["m1"] and "m1" not in [a["caseId"] for a in saved["answers"]]
