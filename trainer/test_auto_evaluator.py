import json

from datasets import Dataset


def test_automatic_grades_preserve_exact_checks_and_abstain():
    from evaluator import evaluate_results
    rows = Dataset.from_list([
        {"prompt": "Return JSON", "completion": '{"ok": true}'},
        {"prompt": "Explain gravity", "completion": "Masses attract."},
        {"prompt": "Explain gravity", "completion": "Masses attract."},
    ])
    result = {"modelId": "candidate", "status": "complete", "evaluationIds": [0, 1, 2],
              "samples": [{"rowId": 0, "baseOutput": '{"ok": 1}'},
                          {"rowId": 1, "baseOutput": "Masses attract each other."},
                          {"rowId": 2, "baseOutput": "Uncertain response"}]}
    calls = []
    def judge(payload):
        calls.append(payload)
        return json.dumps({"verdict": "pass" if len(calls) == 1 else "abstain", "reason": "Evidence checked"})
    scored = evaluate_results(rows, [result], judge=judge)[0]
    assert len(calls) == 2
    assert "modelId" not in calls[0] and "candidate_model" not in calls[0]
    assert scored["samples"][0]["grade"]["verdict"] == "fail"
    assert scored["qualityEvidence"]["score"] is None
    assert scored["qualityEvidence"]["coverage"] == 2 / 3
    assert scored["qualityEvidence"]["reviewExamples"] == 1
    assert "grade" not in result["samples"][0]


def test_task_rubrics_and_failure_do_not_fabricate_accuracy():
    from evaluator import evaluate_results, rubric_for
    assert rubric_for({"domain": "math"}, "Question", "code")[0] == "math"
    assert rubric_for({}, "Summarize the supplied text", "code")[0] == "summarization"
    assert rubric_for({}, "Explain gravity", "code")[0] == "assistant"
    rows = Dataset.from_list([{"prompt": "Solve this", "completion": "An answer"}])
    result = {"modelId": "m", "status": "complete", "evaluationIds": [0], "samples": [{"rowId": 0, "baseOutput": "Answer"}]}
    for raw in ['{"verdict":"pass"}', 'not JSON', '{"verdict":true,"reason":"ok"}']:
        scored = evaluate_results(rows, [result], judge=lambda payload: raw)[0]
        assert scored["qualityEvidence"]["score"] is None
    result["samples"][0]["inputTruncated"] = True
    scored = evaluate_results(rows, [result], judge=lambda payload: '{"verdict":"pass","reason":"ok"}')[0]
    assert scored["qualityEvidence"]["score"] is None
    result["samples"][0].pop("inputTruncated")
    result["samples"][0]["input"] = "Actual shortened retrieved context"
    payloads = []
    evaluate_results(rows, [result], judge=lambda payload: payloads.append(json.loads(payload)) or '{"verdict":"pass","reason":"ok"}')
    assert payloads[0]["conversation"][0]["content"] == "Actual shortened retrieved context"
    result["provider"] = "compass"
    payloads.clear()
    evaluate_results(rows, [result], judge=lambda payload: payloads.append(json.loads(payload)) or '{"verdict":"pass","reason":"ok"}')
    assert payloads[0]["conversation"][0]["content"] == "Solve this"
    result["samples"].append(result["samples"][0])
    scored = evaluate_results(rows, [result], judge=lambda payload: (_ for _ in ()).throw(AssertionError("Must not judge duplicates")))[0]
    assert scored["qualityEvidence"]["score"] is None


def test_comparison_requires_the_same_judge_and_complete_coverage(tmp_path):
    from evaluator import evaluate_results
    from graph import _compare_results, accuracy_agent_review
    rows = Dataset.from_list([{"prompt": "Explain gravity", "completion": "Masses attract."}])
    records = [{"modelId": "m", "status": "complete", "artifactValid": True, "evaluationIds": [0],
                "samples": [{"rowId": 0, key: "Answer"}]} for key in ("baseOutput", "tunedOutput")]
    marks = iter(["fail", "pass"])
    base, tuned = evaluate_results(rows, records, judge=lambda p: json.dumps({"verdict": next(marks), "reason": "Checked"}))
    manifest = tmp_path / "split.json"
    manifest.write_text('{"evalIndices":[0]}')
    state = {"approved_plan": {"candidates": [{"model": {"id": "m"}}]}, "baseline_results": [base], "evaluation_results": [tuned], "split_manifest_path": str(manifest)}
    assert _compare_results(state)["pairs"][0]["qualityScoreDelta"] == 1
    review = accuracy_agent_review([base, tuned])
    assert review["status"] == "estimated"
    assert len(review["models"]) == 2
    tuned["qualityEvidence"]["evaluatorId"] = "different"
    assert _compare_results(state)["decision"] == "review_required"


def test_langgraph_grades_prompt_rag_and_final_answers(tmp_path, monkeypatch):
    import asyncio
    import graph as module
    import evaluator
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command
    dataset = Dataset.from_list([{"prompt": f"Explain gravity in example {i}", "completion": "Masses attract."} for i in range(60)])
    monkeypatch.setattr(module, "load_source", lambda *args: dataset)
    calls = []
    def grade(dataset, results, **kwargs):
        calls.append(([r.get("evaluationIds") for r in results], kwargs))
        return evaluator.evaluate_results(dataset, results, **kwargs, judge=lambda payload: json.dumps({"verdict": "pass" if json.loads(payload)["candidate_answer"] == "correct answer" else "fail", "reason": "Checked the reference"}))
    monkeypatch.setattr(module, "evaluate_results", grade)
    def result(ids, key):
        return {"status": "complete", "modelId": "Qwen/Qwen2.5-7B-Instruct", "evaluationIds": list(ids),
                "samples": [{"rowId": i, "input": "Explain gravity", key: "correct answer" if key == "tunedOutput" else "wrong answer"} for i in ids]}
    monkeypatch.setattr(module, "baseline_compare", lambda dataset, candidates, limit, **kw: [{**result(kw["eval_indices"], "baseOutput"), "modelId": candidates[0]["model"]["id"]}])
    monkeypatch.setattr(module, "evaluate_tuned_run", lambda dataset, candidate, directory, ids, **kw: {**result(ids, "tunedOutput"), "modelId": candidate["model"]["id"]})
    async def submit(spec):
        run = tmp_path / "run"
        (run / "adapter").mkdir(parents=True, exist_ok=True)
        (run / "metrics.json").write_text('{"eval_loss": 1.0}')
        return {"id": "run", "status": "complete"}
    async def get_run(key):
        return {"status": "complete"}
    services = module.WorkflowServices(tmp_path, lambda: 15, submit, get_run, lambda key: tmp_path / key)
    graph = module.build_workflow_graph(services, MemorySaver())
    config = {"configurable": {"thread_id": "autograde"}}
    async def run():
        state = await graph.ainvoke({"workflow_id": "autograde", "dataset": {"source": "upload", "name": "data"}, "analysis_limit": 2, "auto_analysis": True, "max_candidates": 1,
            "knowledge_documents": [{"id": "source", "title": "Gravity", "text": "Masses attract."}]}, config)
        if state.get("__interrupt__") and state["__interrupt__"][0].value["type"] == "data_review":
            state = await graph.ainvoke(Command(resume={"action": "accept_classification", "task": "assistant", "notes": "Explain factual concepts accurately."}), config)
        assert len(state.get("prompt_trials", [])) == 4, state.get("__interrupt__")
        assert state["accuracy_review"]["status"] == "estimated"
        assert state["prompt_trials"][-1]["results"][0]["qualityEvidence"]["coverage"] == 1
        state = await graph.ainvoke(Command(resume={"action": "fine_tune", "trial_index": 0}), config)
        state = await graph.ainvoke(Command(resume={"action": "approve"}), config)
        assert state["status"] == "complete"
        assert state["comparison"]["pairs"][0]["qualityScoreDelta"] == 1
        assert len(state["accuracy_review"]["models"]) == 2
        dev_ids = calls[0][0][0]
        final_ids = calls[-1][0][0]
        assert set(dev_ids).isdisjoint(final_ids)
        assert calls[-1][0][0] == calls[-1][0][1]
        assert calls[3][1]["knowledge_documents"]
    asyncio.run(run())


def test_regrade_saved_workflow_preserves_answers_and_does_not_train(tmp_path, monkeypatch):
    import asyncio
    import app as api
    from uuid import UUID
    from fastapi import HTTPException
    dataset = Dataset.from_list([{"prompt": "Explain gravity", "completion": "Masses attract."}])
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"evalDataset": {"source": "upload", "name": "data"}, "evalIndices": [0]}))
    key = str(UUID(int=99))
    results = [{"status": "complete", "modelId": "m", "artifactValid": True, "evaluationIds": [0], "samples": [{"rowId": 0, name: "Answer"}]} for name in ["baseOutput", "tunedOutput"]]
    record = {"id": key, "status": "complete", "state": {"baseline_results": [results[0]], "evaluation_results": [results[1]], "split_manifest_path": str(manifest), "approved_plan": {"candidates": [{"model": {"id": "m"}}]}}}
    monkeypatch.setattr(api, "WORKFLOWS_ROOT", tmp_path)
    monkeypatch.setattr(api, "load_source", lambda *args: dataset)
    monkeypatch.setattr(api, "queue_lock", asyncio.Lock())
    monkeypatch.setattr(api, "active_workflows", {})
    monkeypatch.setattr(api, "active_evaluations", {})
    monkeypatch.setattr(api, "isolated_inference", lambda fn, *args, **kw: fn(*args, **kw, judge=lambda payload: '{"verdict":"pass","reason":"Checked"}'))
    api.write_workflow(record)
    async def run():
        response = await api.evaluate_workflow(UUID(key))
        assert response["status"] == "evaluating_accuracy"
        assert (await api.evaluate_workflow(UUID(key)))["status"] == "evaluating_accuracy"
        await api.active_evaluations[key]
        saved = api.read_workflow(key)
        assert saved["status"] == "complete"
        assert saved["state"]["accuracy_review"]["status"] == "estimated"
        assert saved["state"]["baseline_results"][0]["samples"][0]["baseOutput"] == "Answer"
        assert (tmp_path / key / "before-automatic-evaluation.json").exists()
        saved["status"] = "waiting"
        api.write_workflow(saved)
        try:
            await api.evaluate_workflow(UUID(key))
        except HTTPException as error:
            assert error.status_code == 409
        else:
            raise AssertionError("Must not grade an unfinished workflow")
    asyncio.run(run())
