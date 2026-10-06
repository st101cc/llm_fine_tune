import asyncio
import pytest
from datasets import Dataset
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
import graph as workflow


@pytest.mark.parametrize("ambiguous", [True, False])
def test_dataset_analysis_pauses_without_gpu_and_records_user_answer(tmp_path, monkeypatch, ambiguous):
    rows = [{"text": f"A unique support article number {i}"} for i in range(120)] if ambiguous else [{"prompt": f"Question {i}", "completion": f"Answer {i}"} for i in range(120)]
    monkeypatch.setattr(workflow, "load_source", lambda *args: Dataset.from_list(rows))
    def no_gpu(*args, **kwargs):
        raise AssertionError("Dataset analysis must not run inference")
    monkeypatch.setattr(workflow, "baseline_compare", no_gpu)
    services = workflow.WorkflowServices(tmp_path, lambda: 0, no_gpu, no_gpu, lambda value: tmp_path / value)
    graph = workflow.build_workflow_graph(services, MemorySaver())
    config = {"configurable": {"thread_id": "dataset-1"}}
    async def run():
        result = await graph.ainvoke({"workflow_id":"dataset-1", "dataset":{"source":"upload", "name":"sample.jsonl"}, "analysis_only":True, "auto_analysis":False}, config)
        pending = result["__interrupt__"][0].value
        assert pending["type"] == ("data_review" if ambiguous else "dataset_ready")
        if ambiguous:
            result = await graph.ainvoke(Command(resume={"action":"accept_classification", "task":"assistant", "notes":"Answer support questions"}), config)
            assert result["__interrupt__"][0].value["type"] == "dataset_ready"
            assert result["profile"]["classification"]["task"] == "assistant"
            assert result["profile"]["classification"]["source"] == "user-confirmed"
            assert result["clarification"]["notes"] == "Answer support questions"
        assert not result.get("training_runs")
    asyncio.run(run())


def test_uploaded_versions_and_clarification_survive_restart(tmp_path, monkeypatch):
    import json
    import time
    import app as api
    from fastapi.testclient import TestClient
    from langgraph.checkpoint.memory import MemorySaver
    root = tmp_path
    for name in ("datasets", "workflows", "runs"):
        (root / name).mkdir()
    for key, value in {"DATA_ROOT":root, "DATASETS_ROOT":root / "datasets", "WORKFLOWS_ROOT":root / "workflows", "WORKFLOW_DB":root / "checkpoints.sqlite"}.items():
        monkeypatch.setattr(api, key, value)
    services = workflow.WorkflowServices(root, lambda: 0, None, None, lambda key: root / "runs" / key)
    monkeypatch.setattr(api, "workflow_services", services)
    monkeypatch.setattr(api, "workflow_graph", workflow.build_workflow_graph(services, MemorySaver()))
    monkeypatch.setattr(api, "workflow_checkpointer_context", None)
    def wait(client, key):
        for _ in range(100):
            result = client.get("/workflow/runs/" + key).json()
            if result["status"] in {"waiting", "failed"}:
                assert result["status"] != "failed", result
                return result
            time.sleep(.02)
        raise AssertionError("Analysis did not reach review")
    content = "\n".join(json.dumps({"text":f"A unique support article number {i}"}) for i in range(120))
    with TestClient(api.app) as client:
        uploaded = client.post("/datasets/upload", files={"file":("sample.jsonl", content, "application/json")}).json()
        response = client.post("/workflow/runs", json={"dataset":uploaded, "analysis_only":True, "auto_analysis":False})
        assert response.status_code == 202
        first_id = response.json()["id"]
        first = wait(client, first_id)
        assert first["pendingAction"]["type"] == "data_review"
    with TestClient(api.app) as client:
        resumed = client.post(f"/workflow/runs/{first_id}/resume", json={"response":{"action":"accept_classification", "task":"assistant", "notes":"Answer support questions"}})
        assert resumed.status_code == 202
        first = wait(client, first_id)
        assert first["pendingAction"]["type"] == "dataset_ready"
        assert first["profile"]["classification"]["source"] == "user-confirmed"
        updated = client.post("/datasets/upload", files={"file":("sample-v2.jsonl", content, "application/json")}).json()
        second = client.post("/workflow/runs", json={"dataset":updated, "analysis_only":True, "auto_analysis":False, "previous_workflow_id":first_id}).json()
        second = wait(client, second["id"])
        assert second["previous_workflow_id"] == first_id
        assert second["pendingAction"]["type"] == "data_review"
        assert uploaded["name"] != updated["name"]
        assert api.Path(uploaded["name"]).exists()
        assert client.get(f"/workflow/runs/{first_id}").json()["pendingAction"]["type"] == "dataset_ready"
        assert not second.get("training_runs")


def test_cloud_suggestion_still_requires_user_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "load_source", lambda *args: Dataset.from_list([{"text":f"A support article number {i}"} for i in range(120)]))
    monkeypatch.setattr(workflow, "cloud_clarify", lambda *args: {"task":"assistant", "confidence":.99})
    services = workflow.WorkflowServices(tmp_path, lambda:0, None, None, lambda key:tmp_path/key)
    graph = workflow.build_workflow_graph(services, MemorySaver())
    config = {"configurable":{"thread_id":"cloud-review"}}
    async def run():
        result = await graph.ainvoke({"workflow_id":"cloud-review", "dataset":{"source":"upload","name":"file.jsonl"},"analysis_only":True,"cloud_assist":True}, config)
        assert result["__interrupt__"][0].value["type"] == "data_review"
        result = await graph.ainvoke(Command(resume={"action":"accept_classification","task":{},"notes":"Invalid task"}), config)
        assert result["__interrupt__"][0].value["type"] == "data_review"
    asyncio.run(run())
