import json
from uuid import UUID
import pytest
from datasets import Dataset
from fastapi.testclient import TestClient
import app as api


def test_comparison_inputs_exclude_answers_and_guard_final_split(tmp_path, monkeypatch):
    workflow_id = str(UUID(int=1))
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"developmentIndices":[1],"evalIndices":[2],"evalDataset":{"source":"upload","name":"same"}}))
    state = {"dataset":{"source":"upload","name":"same"},"split_manifest_path":str(manifest),"split_manifest_id":"split-v1"}
    state.update({"baseline_results":[{"answer":"SEALED"}],"evaluation_results":[{"answer":"SEALED"}],"comparison":{"answer":"SEALED"}})
    record = {"id":workflow_id,"status":"waiting","state":state}
    monkeypatch.setattr(api,"read_workflow",lambda key:record)
    monkeypatch.setattr(api,"RUNS_ROOT",tmp_path)
    monkeypatch.setattr(api,"WORKFLOW_DB",tmp_path / "checkpoints.sqlite")
    rows = [{"messages":[{"role":"user","content":f"Question {i}"},{"role":"assistant","content":f"SECRET GOLD {i}"}]} for i in range(3)]
    monkeypatch.setattr(api,"load_source",lambda *args:Dataset.from_list(rows))
    with TestClient(api.app, raise_server_exceptions=True) as client:
        assert "SEALED" not in client.get(f"/workflow/runs/{workflow_id}").text
        response=client.get(f"/workflow/runs/{workflow_id}/comparison-inputs?phase=development&limit=3")
        assert response.status_code==200
        data=response.json()
        assert data["cases"][0]["rowId"]==1
        assert "SECRET GOLD" not in response.text
        assert client.get(f"/workflow/runs/{workflow_id}/comparison-inputs?phase=heldout").status_code==409
        assert "SEALED" in state["baseline_results"][0]["answer"]
        record["status"]="complete"
        assert "SEALED" in client.get(f"/workflow/runs/{workflow_id}").text
        response=client.get(f"/workflow/runs/{workflow_id}/comparison-inputs?phase=heldout")
        assert response.json()["cases"][0]["rowId"]==2
        assert "SECRET GOLD" not in response.text
