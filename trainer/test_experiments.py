from advisor import evaluation_prompt
from graph import _compare_results


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return str(messages)


def test_chat_reference_is_not_sent_to_model():
    row = {"messages": [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "SECRET ANSWER"}]}
    prompt, reference = evaluation_prompt(row, Tokenizer(), "Be concise")
    assert "SECRET ANSWER" not in prompt
    assert reference == "SECRET ANSWER"
    assert "Be concise" in prompt
    assert len(row["messages"]) == 2


def test_unmeasured_correctness_requires_review(tmp_path):
    path = tmp_path / "split.json"
    path.write_text('{"evaluationIds": [1]}')
    state = {"split_manifest_path": str(path), "approved_plan": {"candidates": [{"model": {"id": "model"}}]},
             "baseline_results": [{"modelId": "model", "status": "complete", "meanTokenOverlap": .6}],
             "evaluation_results": [{"modelId": "model", "status": "complete", "artifactValid": True, "meanTokenOverlap": .6}]}
    result = _compare_results(state)
    assert result["winner"] is None
    assert result["decision"] == "review_required"
    state["baseline_results"] = []
    assert _compare_results(state)["decision"] == "review_required"


def test_experiment_api_validation_and_recording(monkeypatch, tmp_path):
    import app as api
    from fastapi.testclient import TestClient
    monkeypatch.setattr(api, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(api, "gpu_status", lambda: {"memoryGb": 24})
    calls = []
    def infer(dataset, candidates, limit, **kwargs):
        calls.append((list(dataset), kwargs))
        return [{"status": "complete", "samples": [{"baseOutput": "Recorded answer"}]}]
    monkeypatch.setattr(api, "baseline_compare", infer)
    client = TestClient(api.app)
    payload = {"use_case": "Support", "success_criteria": "Correct policy", "model_id": "Qwen/Qwen2.5-3B-Instruct", "instruction": "Be brief", "context": "Policy", "examples": [{"prompt": "Question", "completion": "Ideal"}]}
    response = client.post("/experiments", json=payload)
    assert response.status_code == 200
    assert response.json()["result"]["samples"][0]["baseOutput"] == "Recorded answer"
    assert len(list((tmp_path / "experiments").glob("*.json"))) == 1
    assert "Policy" in calls[0][1]["instruction"]
    assert client.post("/experiments", json={**payload, "model_id": "unknown"}).status_code == 422
    assert client.post("/experiments", json={**payload, "examples": []}).status_code == 422
    monkeypatch.setattr(api, "baseline_compare", lambda *args, **kwargs: [{"status": "unavailable", "message": "Inference failed"}])
    assert client.post("/experiments", json=payload).status_code == 503
    assert len(list((tmp_path / "experiments").glob("*.json"))) == 1
