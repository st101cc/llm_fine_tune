import pytest
from hf_import import dataset_id, import_huggingface


def test_hf_ids_and_url_validation():
    assert dataset_id("https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data") == "nvidia/Nemotron-Cascade-2-SFT-Data"
    assert dataset_id("owner/data") == "owner/data"
    for value in ["../data", "https://evil.com/datasets/a/b", "https://huggingface.co/a/b", "owner/data/extra"]:
        with pytest.raises(ValueError): dataset_id(value)


def test_import_keeps_chat_and_selected_subset(tmp_path, monkeypatch):
    import hf_import as module
    from types import SimpleNamespace
    monkeypatch.setattr(module, "dataset_info", lambda *args, **kwargs: SimpleNamespace(sha="abc123"))
    monkeypatch.setattr(module, "get_dataset_config_names", lambda *args, **kwargs: ["chat", "math"])
    calls = []
    class Stream:
        def take(self, count): return iter([{"messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]}] * count)
    def load(*args, **kwargs): calls.append(kwargs); return Stream()
    monkeypatch.setattr(module, "load_dataset", load)
    pending = import_huggingface("owner/data", None, "train", 30, tmp_path)
    assert pending["requiresConfiguration"] and not calls
    result = import_huggingface("owner/data", "chat", "train", 30, tmp_path)
    assert calls[0]["streaming"] and calls[0]["name"] == "chat" and calls[0]["revision"] == "abc123"
    assert result["importedRows"] == 30
    assert '"messages"' in module.Path(result["name"]).read_text()


def mock_hub(monkeypatch, rows):
    import hf_import as module
    from itertools import islice
    from types import SimpleNamespace
    monkeypatch.setattr(module, "dataset_info", lambda *a, **k: SimpleNamespace(sha="fixed-revision"))
    monkeypatch.setattr(module, "get_dataset_config_names", lambda *a, **k: ["chat"])
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=10**12))
    class Stream:
        info = SimpleNamespace(splits={"train": SimpleNamespace(num_examples=len(rows))})
        def __iter__(self): return iter(rows)
        def take(self, count): return islice(self, count)
    monkeypatch.setattr(module, "load_dataset", lambda *a, **k: Stream())
    return module


def test_full_split_exceeds_sample_limits_and_analysis_stays_sampled(tmp_path, monkeypatch):
    from advisor import load_source, local_profile
    from app import DatasetRequest
    from graph import _make_split_manifest, WorkflowServices
    import json
    rows = [{"prompt": f"Question {i}", "completion": f"Answer {i}"} for i in range(10005)]
    module = mock_hub(monkeypatch, rows)
    monkeypatch.setattr(module, "SAMPLE_BYTES", 50)
    progress = []
    result = import_huggingface("owner/data", None, "train", None, tmp_path, progress=progress.append)
    assert result["importedRows"] == 10005 and result["sampleOnly"] is False
    assert result["hubRevision"] == "fixed-revision"
    assert progress[-1]["importedRows"] == progress[-1]["totalRows"] == 10005
    assert progress[-1]["importedBytes"] == module.Path(result["name"]).stat().st_size
    dataset = load_source("upload", result["name"], "jsonl")
    profile = local_profile(dataset)
    assert profile["facts"]["rows"] == 10005
    assert profile["facts"]["sampledRows"] == 600
    ref = DatasetRequest(**result).model_dump()
    manifest = _make_split_manifest({"workflow_id":"full", "dataset":ref}, WorkflowServices(tmp_path, lambda:0, None, None, lambda key:tmp_path/key), dataset, profile)
    assert len(manifest["trainIndices"]) + len(manifest["developmentIndices"]) + len(manifest["evalIndices"]) == 10005
    assert json.loads(module.Path(manifest["path"]).read_text())["trainDataset"]["sampleOnly"] is False
    with pytest.raises(ValueError, match="50 MB"):
        import_huggingface("owner/data", None, "train", 30, tmp_path)
    assert list(tmp_path.glob("*.partial")) == []
    assert len(list(tmp_path.glob("*.jsonl"))) == 1


@pytest.mark.parametrize("failure", ["cancel", "disk", "empty", "network"])
def test_incomplete_import_never_leaves_a_dataset(tmp_path, monkeypatch, failure):
    from threading import Event
    from types import SimpleNamespace
    rows = [] if failure == "empty" else [{"text": "content"}] * 100
    module = mock_hub(monkeypatch, rows)
    stop = Event()
    def progress(values): stop.set()
    if failure == "disk":
        free = iter([10**12, module.DISK_RESERVE + 1])
        monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=next(free)))
    if failure == "network":
        def broken():
            yield {"text":"first row"}
            raise ConnectionError("Network lost")
        monkeypatch.setattr(module, "load_dataset", lambda *a, **k: broken())
    with pytest.raises((ValueError, InterruptedError, ConnectionError)):
        import_huggingface("owner/data", None, "train", None, tmp_path, cancelled=stop.is_set,
                           progress=progress if failure == "cancel" else None)
    assert list(tmp_path.iterdir()) == []


def test_background_import_status_cancel_validation_and_restart(tmp_path, monkeypatch):
    import app as api
    import hf_import as module
    import time
    from threading import Event
    from uuid import uuid4
    from fastapi.testclient import TestClient
    monkeypatch.setattr(api, "DATASETS_ROOT", tmp_path)
    monkeypatch.setattr(api, "WORKFLOW_DB", tmp_path / "checkpoints.sqlite")
    monkeypatch.setattr(api, "active_imports", {})
    started, finish = Event(), Event()
    def importer(*args, progress, cancelled, import_id):
        progress({"importedRows":12345, "importedBytes":60000000, "totalRows":20000})
        started.set()
        while not finish.wait(.01):
            if cancelled(): raise InterruptedError("Import cancelled")
        return {"source":"upload", "name":str(tmp_path / (import_id + ".jsonl")), "importedRows":20000, "sampleOnly":False}
    monkeypatch.setattr(module, "import_huggingface", importer)
    def wait(client, key, status):
        for _ in range(100):
            result = client.get("/datasets/imports/" + key).json()
            if result["status"] == status: return result
            time.sleep(.02)
        raise AssertionError(result)
    with TestClient(api.app) as client:
        assert client.post("/datasets/huggingface", json={"url":"../bad", "mode":"full"}).status_code == 422
        assert client.post("/datasets/huggingface", json={"url":"owner/data", "mode":"unknown"}).status_code == 422
        assert client.get("/datasets/imports/not-a-uuid").status_code == 422
        assert client.get("/datasets/imports/" + str(uuid4())).status_code == 404
        response = client.post("/datasets/huggingface", json={"url":"owner/data", "mode":"full"})
        assert response.status_code == 202
        key = response.json()["id"]
        assert started.wait(2)
        assert client.get("/datasets/imports/" + key).json()["importedRows"] == 12345
        assert client.get("/hardware").status_code == 200
        assert client.post("/datasets/huggingface", json={"url":"owner/data", "mode":"full"}).status_code == 409
        assert client.post(f"/datasets/imports/{key}/cancel").status_code == 202
        wait(client, key, "cancelled")
        finish.set()
        response = client.post("/datasets/huggingface", json={"url":"owner/data", "mode":"full"})
        complete_key = response.json()["id"]
        assert wait(client, complete_key, "complete")["result"]["sampleOnly"] is False
    interrupted = str(uuid4())
    api.write_import({"id":interrupted, "status":"running"})
    partial = tmp_path / (interrupted + ".jsonl.partial")
    partial.write_text("incomplete")
    group = tmp_path / interrupted
    group.mkdir()
    (group / "unfinished.jsonl.partial").write_text("incomplete")
    (group / "finished.jsonl").write_text('{"text":"complete"}')
    with TestClient(api.app) as client:
        assert client.get("/datasets/imports/" + complete_key).json()["status"] == "complete"
        assert client.get("/datasets/imports/" + interrupted).json()["status"] == "interrupted"
        assert not partial.exists()
        assert not (group / "unfinished.jsonl.partial").exists()
        assert (group / "finished.jsonl").exists()


def test_all_subsets_are_pinned_audited_and_kept_separate(tmp_path, monkeypatch):
    import hf_import as module
    from uuid import uuid4
    module = mock_hub(monkeypatch, [])
    monkeypatch.setattr(module, "get_dataset_config_names", lambda *a, **k: ["chat", "math", "missing"])
    calls = []
    def load(repo, **kwargs):
        calls.append(kwargs)
        if kwargs["name"] == "missing": raise ValueError("No train split in this subset")
        return iter([{"prompt":f'{kwargs["name"]} question {i}', "completion":f"Answer {i}"} for i in range(650)])
    monkeypatch.setattr(module, "load_dataset", load)
    updates = []
    result = module.import_all_subsets("owner/data", "train", tmp_path, progress=updates.append, cancelled=lambda:False, import_id=str(uuid4()))
    assert result["allSubsets"] and len(result["subsets"]) == 3
    assert result["failedSubsets"] == 1 and result["importedRows"] == 1300, result
    assert [c["name"] for c in calls] == ["chat", "math", "missing"]
    assert all(c["revision"] == "fixed-revision" and c["streaming"] for c in calls)
    first, second, failed = result["subsets"]
    assert first["dataset"]["name"] != second["dataset"]["name"]
    assert all(x["profile"]["facts"]["sampledRows"] == 600 and x["profile"]["facts"]["rows"] == 650 for x in [first,second])
    assert failed["status"] == "failed" and "No train split" in failed["error"]
    assert updates[-1]["completedSubsets"] == 3


def test_selection_waits_for_all_audits_and_reuses_workflows(tmp_path, monkeypatch):
    import asyncio
    import app as api
    from uuid import uuid4
    from fastapi import HTTPException
    monkeypatch.setattr(api, "DATASETS_ROOT", tmp_path)
    key = uuid4()
    ref = {"source":"upload", "name":"/tmp/subset.jsonl", "sampleOnly":False}
    record = {"id":str(key), "status":"running", "result":{"allSubsets":True, "subsets":[
        {"configuration":"chat", "status":"analyzed", "dataset":ref},
        {"configuration":"math", "status":"analyzed", "dataset":{**ref,"name":"/tmp/math.jsonl"}},
        {"configuration":"bad", "status":"failed"}]}}
    api.write_import(record)
    created = []
    async def create(request):
        assert request.analysis_only and not request.auto_analysis
        created.append(request.dataset.name)
        return {"id":str(len(created))}
    monkeypatch.setattr(api, "create_workflow", create)
    async def run():
        selection = api.SubsetSelectionRequest(configurations=["chat","math"])
        with pytest.raises(HTTPException) as pending:
            await api.select_import_subsets(key, selection)
        assert pending.value.status_code == 409 and not created
        record["status"] = "complete"; api.write_import(record)
        for names in [["bad"], ["unknown"], ["chat","chat"]]:
            with pytest.raises(HTTPException) as invalid:
                await api.select_import_subsets(key, api.SubsetSelectionRequest(configurations=names))
            assert invalid.value.status_code == 422
        first = await api.select_import_subsets(key, selection)
        assert len(first["workflows"]) == 2 and len(created) == 2
        assert await api.select_import_subsets(key, selection) == first
        assert len(created) == 2
    asyncio.run(run())


def test_cancel_all_subsets_keeps_completed_data(tmp_path, monkeypatch):
    import hf_import as module
    from uuid import uuid4
    from threading import Event
    module = mock_hub(monkeypatch, [{"text":f"Row {i}"} for i in range(30)])
    monkeypatch.setattr(module, "get_dataset_config_names", lambda *a, **k: ["chat","math"])
    stop = Event()
    def update(progress):
        if progress["completedSubsets"] == 1: stop.set()
    with pytest.raises(InterruptedError):
        module.import_all_subsets("owner/data", "train", tmp_path, progress=update, cancelled=stop.is_set, import_id=str(uuid4()))
    assert len(list(tmp_path.glob("*/*.jsonl"))) == 1
    assert not list(tmp_path.glob("*/*.partial"))


def test_all_mode_job_runs_audits_without_upfront_subset_choice(tmp_path, monkeypatch):
    import app as api
    import hf_import as module
    import time
    from fastapi.testclient import TestClient
    monkeypatch.setattr(api, "DATASETS_ROOT", tmp_path)
    monkeypatch.setattr(api, "WORKFLOW_DB", tmp_path / "checkpoints.sqlite")
    monkeypatch.setattr(api, "active_imports", {})
    def all_subsets(value, split, directory, **kwargs):
        assert value == "owner/data" and split == "train"
        kwargs["progress"]({"completedSubsets":2, "totalSubsets":2})
        return {"allSubsets":True, "subsets":[{"configuration":name,"status":"analyzed"} for name in ["chat","math"]]}
    monkeypatch.setattr(module, "import_all_subsets", all_subsets)
    with TestClient(api.app) as client:
        response = client.post("/datasets/huggingface", json={"url":"owner/data", "mode":"all"})
        assert response.status_code == 202
        key = response.json()["id"]
        for _ in range(100):
            result = client.get("/datasets/imports/" + key).json()
            if result["status"] == "complete": break
            time.sleep(.02)
        assert result["status"] == "complete"
        assert len(result["result"]["subsets"]) == 2
        assert not result.get("selectedWorkflows")
