import asyncio
import pytest
from datasets import Dataset
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from benchmark import select_model, apply_review, compass_results
import graph as module


def result(model, score, cost, seconds=1):
    return {"modelId":model,"status":"complete","evaluationIds":[1,2,3],"qualityEvidence":{"metric":"label_accuracy","score":score,"coverage":1,"evaluatedExamples":3},"meanLatencySeconds":seconds,"costUsd":cost,"samples":[{"rowId":i,"baseOutput":"answer"} for i in [1,2,3]],"provider":"local"}


def test_selection_uses_correctness_then_cost_and_latency():
    assert select_model([result("small",.5,.001),result("large",1,.01)],[1,2,3])["winner"]=="large"
    assert select_model([result("a",1,.01),result("b",1,.001,3)],[1,2,3])["winner"]=="b"
    assert select_model([result("a",1,.001,3),result("b",1,.001,1)],[1,2,3])["winner"]=="b"
    missing=result("x",1,None)
    assert select_model([missing,result("b",1,.01)],[1,2,3])["winner"] is None
    assert select_model([missing,result("b",1,.01)],[1,2,3],allow_unknown_cost=True)["winner"]=="b"


def test_no_winner_for_unscored_partial_or_different_questions():
    rows=[result("a",1,.01),result("b",.7,.01)]
    rows[0]["qualityEvidence"]["coverage"]=.5
    assert select_model(rows,[1,2,3])["winner"] is None
    rows[0]["qualityEvidence"]["coverage"]=1
    rows[0]["evaluationIds"]=[7,8,9]
    assert select_model(rows,[1,2,3])["winner"] is None
    assert select_model([rows[1]],[1,2,3])["winner"] is None
    with pytest.raises(ValueError):
        apply_review(rows,{"hourlyCostUsd":-1},{})


def test_new_workflow_benchmarks_all_eligible_models_before_prompt_trials(tmp_path,monkeypatch):
    monkeypatch.setattr(module,"load_source",lambda *args:Dataset.from_list([{"prompt":f"Q{i}","completion":f"A{i}"} for i in range(120)]))
    calls=[]
    def evaluate(dataset,candidates,limit,**kwargs):
        calls.append((candidates,kwargs))
        ids=kwargs["eval_indices"]
        return [{**result(c["model"]["id"],None,None),"evaluationIds":ids,"qualityEvidence":{},"latencySeconds":3,"inferenceSeconds":3,"meanLatencySeconds":1,"samples":[{"rowId":i,"input":"Question","baseOutput":"Answer","reference":"Reference"} for i in ids]} for c in candidates]
    monkeypatch.setattr(module,"baseline_compare",evaluate)
    services=module.WorkflowServices(tmp_path,lambda:15,None,None,lambda key:tmp_path/key)
    graph=module.build_workflow_graph(services,MemorySaver());config={"configurable":{"thread_id":"benchmark"}}
    async def run():
        state=await graph.ainvoke({"workflow_id":"benchmark","benchmark_selection":True,"dataset":{"source":"upload","name":"data"},"analysis_limit":3,"auto_analysis":True,"max_candidates":1},config)
        assert state["__interrupt__"][0].value["type"]=="model_selection"
        assert len(calls)==4 and len({c[0][0]["model"]["id"] for c in calls})==4
        assert all(c[1]["eval_indices"]==calls[0][1]["eval_indices"] for c in calls)
        assert not state.get("prompt_trials") and not state.get("training_runs")
        models=state["model_benchmark"]["results"]
        grades={r["modelId"]:{str(s["rowId"]):int(n==1) for s in r["samples"]} for n,r in enumerate(models)}
        state=await graph.ainvoke(Command(resume={"action":"select_best","grades":grades,"rubric":"Correct, complete, follows requirements","hourlyCostUsd":1}),config)
        assert state["model_benchmark"]["winner"]==models[1]["modelId"]
        assert len(state["candidates"])==1
        assert state["__interrupt__"][0].value["type"]=="prompt_review"
        assert all(len(c[0])==1 for c in calls[4:])
    asyncio.run(run())


def test_compass_validates_same_inputs_and_scores_ground_truth():
    from benchmark import INSTRUCTION
    from advisor import evaluation_messages
    rows=[{"text":"great","label":"positive"},{"text":"awful","label":"negative"}]
    benchmark={"developmentIds":[4,8],"splitManifestId":"split"}
    job={"phase":"development","status":"complete","instruction":INSTRUCTION,"maxOutputTokens":256,"splitManifestId":"split","models":["gpt","gemini"],"cases":[{"rowId":i,"messages":evaluation_messages(row,INSTRUCTION)[0]} for i,row in zip([4,8],rows)],"results":[{"rowId":i,"model":model,"output":row["label"],"seconds":.5,"usage":{"prompt_tokens":100,"completion_tokens":2}} for model in ["gpt","gemini"] for i,row in zip([4,8],rows)]}
    results=compass_results(job,benchmark,rows)
    assert all(r["qualityEvidence"]["score"]==1 for r in results)
    priced,_=apply_review(results,{"tokenPrices":{"gpt":{"input":1,"output":2},"gemini":{"input":2,"output":4}}},{})
    assert select_model(priced,[4,8])["winner"]=="gpt"
    cleared,_=apply_review(priced,{"tokenPrices":{}},{})
    assert all(r["costUsd"] is None for r in cleared)
    job["results"][0]["usage"]=[1]
    with pytest.raises(ValueError): compass_results(job,benchmark,rows)
    job["results"][0]["usage"]={"prompt_tokens":100,"completion_tokens":2}
    job["phase"]="heldout"
    with pytest.raises(ValueError): compass_results(job,benchmark,rows)
    job["phase"]="development";job["cases"][0]["messages"][0]["content"]="Different question"
    with pytest.raises(ValueError): compass_results(job,benchmark,rows)


def test_hosted_winner_never_enters_local_training(tmp_path,monkeypatch):
    from benchmark import INSTRUCTION
    from advisor import evaluation_messages
    dataset=Dataset.from_list([{"prompt":f"Q{i}","completion":f"A{i}"} for i in range(120)])
    monkeypatch.setattr(module,"load_source",lambda *args:dataset)
    calls=[]
    def evaluate(dataset,candidates,limit,**kwargs):
        calls.append(candidates)
        ids=kwargs["eval_indices"]
        return [{"modelId":candidates[0]["model"]["id"],"status":"complete","evaluationIds":ids,"qualityEvidence":{},"meanLatencySeconds":1,"inferenceSeconds":len(ids),"samples":[{"rowId":i,"baseOutput":"answer"} for i in ids]}]
    monkeypatch.setattr(module,"baseline_compare",evaluate)
    graph=module.build_workflow_graph(module.WorkflowServices(tmp_path,lambda:15,None,None,lambda key:tmp_path/key),MemorySaver());config={"configurable":{"thread_id":"hosted"}}
    async def run():
        state=await graph.ainvoke({"workflow_id":"hosted","benchmark_selection":True,"dataset":{"source":"upload","name":"data"},"analysis_limit":3,"auto_analysis":True},config)
        b=state["model_benchmark"];ids=b["developmentIds"]
        job={"phase":"development","status":"complete","instruction":INSTRUCTION,"maxOutputTokens":256,"splitManifestId":b["splitManifestId"],"models":["gpt"],"cases":[{"rowId":i,"messages":evaluation_messages(dataset[i],INSTRUCTION)[0]} for i in ids],"results":[{"rowId":i,"model":"gpt","output":"A","seconds":.1,"usage":{"prompt_tokens":10,"completion_tokens":1}} for i in ids]}
        state=await graph.ainvoke(Command(resume={"action":"add_compass","job":job}),config)
        results=state["model_benchmark"]["results"]
        grades={r["modelId"]:{str(i):int(r["provider"]=="compass") for i in ids} for r in results}
        state=await graph.ainvoke(Command(resume={"action":"select_best","hourlyCostUsd":1,"tokenPrices":{"gpt":{"input":1,"output":1}},"rubric":"Correct and complete","grades":grades}),config)
        assert state["status"]=="complete"
        assert state["comparison"]["modelIds"]==["gpt"]
        assert not state.get("training_runs") and not state.get("prompt_trials")
        assert len(calls)==4
    asyncio.run(run())
