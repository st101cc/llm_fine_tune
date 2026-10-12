import asyncio
import json
import pytest
from datasets import Dataset
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
import graph as module


@pytest.mark.parametrize("action", ["keep_baseline", "fine_tune"])
def test_prompt_experiments_stay_in_workflow_and_can_keep_baseline(tmp_path, monkeypatch, action):
    dataset = Dataset.from_list([{"prompt":f"Question {i}","completion":f"Answer {i}"} for i in range(120)])
    monkeypatch.setattr(module,"load_source",lambda *args:dataset)
    calls = []
    def baseline(dataset, candidates, limit, **kwargs):
        calls.append(kwargs)
        return [{"modelId":candidates[0]["model"]["id"],"status":"complete","samples":[{"rowId":kwargs["eval_indices"][0],"baseOutput":"Actual test output"}],"meanTokenOverlap":.6}]
    monkeypatch.setattr(module,"baseline_compare",baseline)
    services = module.WorkflowServices(tmp_path,lambda:24,None,None,lambda key:tmp_path/key)
    graph = module.build_workflow_graph(services,MemorySaver())
    config = {"configurable":{"thread_id":"integrated"}}
    async def run():
        result = await graph.ainvoke({"workflow_id":"integrated","dataset":{"source":"upload","name":"data.jsonl"},"analysis_only":True,"max_candidates":1},config)
        result = await graph.ainvoke(Command(resume={"action":"continue_training"}),config)
        assert result["__interrupt__"][0].value["type"] == "prompt_review"
        manifest = json.loads((tmp_path/"workflows/integrated/split.json").read_text())
        assert set(calls[0]["eval_indices"]).isdisjoint(manifest["evalIndices"])
        assert set(calls[0]["eval_indices"]).isdisjoint(manifest["trainIndices"])
        result = await graph.ainvoke(Command(resume={"action":"try_prompt","instruction":"Be concise","context":"Policy text"}),config)
        assert len(result["prompt_trials"]) == 4
        assert "Policy text" in calls[-1]["instruction"]
        result = await graph.ainvoke(Command(resume={"action":action,"trial_index":3}),config)
        if action == "keep_baseline":
            assert result["status"] == "complete"
            assert result["comparison"]["decision"] == "keep_baseline"
        else:
            assert result["__interrupt__"][0].value["type"] == "plan_approval"
            assert "Be concise" in result["selected_instruction"]
        assert not result.get("training_runs")
    asyncio.run(run())


@pytest.mark.parametrize("available", [True, False])
def test_automatic_analysis_never_requires_prompt_writing(tmp_path, monkeypatch, available):
    monkeypatch.setattr(module, "load_source", lambda *args: Dataset.from_list([{"prompt":f"Question {i}", "completion":f"Answer {i}"} for i in range(120)]))
    calls = []
    def evaluate(dataset, candidates, limit, **kwargs):
        calls.append(kwargs)
        return [{"modelId": candidates[0]["model"]["id"], "status":"complete" if available else "unavailable", "meanTokenOverlap":.8 if kwargs["instruction"] else .3, "formatValidRate":1, "samples":[]}]
    monkeypatch.setattr(module, "baseline_compare", evaluate)
    services = module.WorkflowServices(tmp_path, lambda:24, None, None, lambda key:tmp_path/key)
    graph = module.build_workflow_graph(services, MemorySaver())
    config = {"configurable":{"thread_id":"automatic"}}
    async def run():
        result = await graph.ainvoke({"workflow_id":"automatic", "dataset":{"source":"upload","name":"data.jsonl"}, "analysis_only":True, "auto_analysis":True, "max_candidates":1}, config)
        assert result["__interrupt__"][0].value["type"] == "prompt_review"
        assert len(calls) == (3 if available else 1)
        assert not result.get("training_runs")
        recommendation = result["recommendation"]
        assert "Not tested" in recommendation["rag"]
        if available:
            assert recommendation["selectedTrial"] == 1
            assert "automatically tested prompt" in recommendation["title"]
            assert all(call["eval_indices"] == calls[0]["eval_indices"] for call in calls)
            result = await graph.ainvoke(Command(resume={"action":"fine_tune"}), config)
            assert result["__interrupt__"][0].value["type"] == "plan_approval"
            assert not result.get("training_runs")
        else:
            assert recommendation["selectedTrial"] is None
            assert "unavailable" in recommendation["title"]
    asyncio.run(run())


def test_diagnosis_rejects_format_regression_and_does_not_claim_rag():
    trials = [{"developmentIds":[1,2,3], "results":[{"modelId":"m", "status":"complete", "meanTokenOverlap":overlap, "formatValidRate":fmt, "samples":[]}]} for overlap, fmt in [(.3,1),(.9,.5)]]
    result = module.diagnose_trials(trials)
    assert result["selectedTrial"] == 0
    assert "More evidence" in result["title"]
    assert "Not tested" in result["rag"]


def test_workflow_automatically_tests_connected_references(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "load_source", lambda *args: Dataset.from_list([{"prompt":f"Question {i}","completion":f"Answer {i}"} for i in range(120)]))
    calls=[]
    def evaluate(dataset,candidates,limit,**kwargs):
        calls.append(kwargs)
        return [{"modelId":candidates[0]["model"]["id"],"status":"complete","samples":[],"meanTokenOverlap":.3}]
    monkeypatch.setattr(module,"baseline_compare",evaluate)
    services=module.WorkflowServices(tmp_path,lambda:24,None,None,lambda key:tmp_path/key)
    graph=module.build_workflow_graph(services,MemorySaver())
    docs=[{"id":"ref1","title":"Independent reference","text":"Reference material about friction."}]
    async def run():
        result=await graph.ainvoke({"workflow_id":"rag-auto","analysis_limit":3,"dataset":{"source":"upload","name":"data.jsonl"},"auto_analysis":True,"max_candidates":1,"knowledge_documents":docs},{"configurable":{"thread_id":"rag-auto"}})
        assert len(calls)==4
        assert all(len(call["eval_indices"])==3 for call in calls)
        assert calls[-1]["knowledge_documents"]==docs
        assert all(not c.get("knowledge_documents") for c in calls[:-1])
        assert result["prompt_trials"][-1]["usesRag"] is True
        assert "Tested" in result["recommendation"]["rag"]
        assert result["prompt_trials"][3]["knowledgeDocuments"]==docs
        config={"configurable":{"thread_id":"rag-auto"}}
        await graph.ainvoke(Command(resume={"action":"test_rag","documents":[{"id":"other","title":"Other","text":"Different references"}]}),config)
        selected=await graph.ainvoke(Command(resume={"action":"fine_tune","trial_index":3}),config)
        assert selected["knowledge_documents"]==docs
    asyncio.run(run())
