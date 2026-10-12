"""Development-only model selection; no model-size or word-overlap quality proxy."""
import math
from advisor import evaluation_messages, task_quality
from evaluator import comparable_evidence

INSTRUCTION = "Answer accurately and concisely. Follow the requested language and output format exactly. Ask a brief clarification if essential information is ambiguous."
MAX_TOKENS = 256


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def select_model(results, ids, allow_unknown_cost=False):
    complete = [r for r in results if r.get("status") == "complete"]
    summary = {"winner":None,"status":"needs_review","results":results,"developmentIds":ids,"costUsed":False,"sampleWarning":"Fewer than 30 development examples: selection is exploratory, not statistically reliable." if len(ids) < 30 else None}
    if len(complete) < 2:
        return {**summary,"reason":"At least two models must complete the same development questions."}
    metrics = set()
    for r in complete:
        q=r.get("qualityEvidence", {})
        if r.get("evaluationIds") != ids or q.get("coverage") != 1 or q.get("evaluatedExamples") != len(ids) or not number(q.get("score")) or not 0 <= q["score"] <= 1 or not q.get("metric"):
            return {**summary,"reason":"Comparable correctness scores are missing. Grade every answer against the same rubric; word overlap is not correctness."}
        if not number(r.get("meanLatencySeconds")) or r["meanLatencySeconds"] < 0:
            return {**summary,"reason":"Measured answer latency is missing."}
        metrics.add(q["metric"])
    if len(metrics) != 1:
        return {**summary,"reason":"Models must use the same correctness metric or shared review rubric."}
    if any(not comparable_evidence(complete[0], r) for r in complete[1:]):
        return {**summary,"reason":"Models must use the same judge version and rubric."}
    known_cost = all(number(r.get("costUsd")) and r["costUsd"] >= 0 for r in complete)
    if not known_cost and not allow_unknown_cost:
        return {**summary,"reason":"Enter compute/token prices, or explicitly continue with cost unmeasured. Missing cost is not zero."}
    best=max(r["qualityEvidence"]["score"] for r in complete)
    # ponytail: fixed 2-point tolerance; add confidence intervals for larger benchmarks.
    contenders=[r for r in complete if r["qualityEvidence"]["score"] >= best - .02]
    winner=min(contenders,key=lambda r: (r["costUsd"] if known_cost else 0,r["meanLatencySeconds"],r["modelId"]))
    return {**summary,"winner":winner["modelId"],"status":"selected","costUsed":known_cost,"reason":"Highest measured correctness, with cost then latency breaking ties within 2 percentage points." + (" Cost excluded for every model because at least one price is unknown." if not known_cost else " Cost uses measured usage and your configured rates, not a billing invoice.")}


def apply_review(results, response, prices):
    hourly=response.get("hourlyCostUsd",prices.get("hourlyCostUsd"))
    rates=response.get("tokenPrices",prices.get("tokenPrices",{}))
    if hourly is not None and (not number(hourly) or hourly < 0):
        raise ValueError("Hourly cost must be a finite non-negative USD amount.")
    if not isinstance(rates,dict):
        raise ValueError("Token prices must be keyed by model.")
    for rate in rates.values():
        if not isinstance(rate,dict) or any(not number(rate.get(k)) or rate[k]<0 for k in ("input","output")):
            raise ValueError("Provide non-negative input and output USD prices per million tokens.")
    grades=response.get("grades",{})
    if not isinstance(grades,dict): raise ValueError("Invalid answer grades.")
    rubric=response.get("rubric","")
    if grades and (not isinstance(rubric,str) or not rubric.strip() or len(rubric)>2000):
        raise ValueError("Provide a shared correctness rubric for manual grades.")
    reviewed=[]
    for original in results:
        r=dict(original)
        if r.get("provider")=="local":
            r["costUsd"]=r["inferenceSeconds"]*hourly/3600 if hourly is not None and number(r.get("inferenceSeconds")) else None
        else:
            r["costUsd"]=None
            if r["modelId"] in rates:
                usage=r.get("usage") or {};rate=rates[r["modelId"]]
                r["costUsd"]=(usage["prompt_tokens"]*rate["input"]+usage["completion_tokens"]*rate["output"])/1_000_000 if all(number(usage.get(k)) for k in ("prompt_tokens","completion_tokens")) else None
        if r["modelId"] in grades:
            marks=grades[r["modelId"]]
            ids=r.get("evaluationIds",[])
            if not isinstance(marks,dict) or set(marks)!={str(i) for i in ids} or not ids or any(type(v) is not int or v not in (0,1) for v in marks.values()):
                raise ValueError("Grade every development answer as correct (1) or incorrect (0).")
            r["qualityEvidence"]={"metric":"human_rubric_accuracy","score":sum(marks.values())/len(ids),"coverage":1,"evaluatedExamples":len(ids),"rubric":rubric.strip(),"grades":marks}
        reviewed.append(r)
    if grades:
        complete=[r for r in reviewed if r.get("status")=="complete"]
        if any(r.get("qualityEvidence",{}).get("rubric")!=rubric.strip() for r in complete):
            raise ValueError("Grade all completed models with the same rubric.")
    return reviewed,{"hourlyCostUsd":hourly,"tokenPrices":rates}


def compass_results(job, benchmark, rows):
    ids=benchmark["developmentIds"]
    if not isinstance(job,dict) or job.get("phase")!="development" or job.get("status")!="complete" or job.get("splitManifestId")!=benchmark["splitManifestId"] or job.get("instruction")!=INSTRUCTION or job.get("maxOutputTokens")!=MAX_TOKENS:
        raise ValueError("Use a completed Compass development benchmark with this split, prompt and 256-token budget.")
    cases=job.get("cases",[])
    expected=[];references=[]
    for row,i in zip(rows,ids):
        messages,reference=evaluation_messages(row,INSTRUCTION)
        expected.append({"rowId":i,"messages":[m for m in messages if not (m["role"]=="system" and not m["content"].strip())]})
        references.append(reference)
    if cases!=expected: raise ValueError("Compass questions differ from the development benchmark.")
    models=job.get("models",[])
    if not isinstance(models,list) or not 1<=len(models)<=2 or any(not isinstance(m,str) or len(m)>200 for m in models) or len(set(models))!=len(models):
        raise ValueError("Choose one or two distinct Compass models.")
    raw=job.get("results")
    if not isinstance(raw,list) or any(not isinstance(r,dict) or (r.get("usage") is not None and not isinstance(r["usage"],dict)) for r in raw): raise ValueError("Invalid Compass results.")
    results=[]
    for model in models:
        outputs=[r for r in job.get("results",[]) if r.get("model")==model]
        if [r.get("rowId") for r in outputs]!=ids or any(not isinstance(r.get("output"),str) or not r["output"].strip() or len(r["output"])>20000 or not number(r.get("seconds")) or r["seconds"]<0 for r in outputs):
            raise ValueError("Each Compass model must finish every benchmark question with timing data.")
        usage={k:sum(r["usage"][k] for r in outputs) if all(number((r.get("usage") or {}).get(k)) and r["usage"][k]>=0 for r in outputs) else None for k in ("prompt_tokens","completion_tokens")}
        results.append({**task_quality(rows,references,[r["output"] for r in outputs],ids),"modelId":model,"provider":"compass","status":"complete","evaluationIds":ids,"meanLatencySeconds":sum(r["seconds"] for r in outputs)/len(ids),"usage":usage,"costUsd":None,"samples":[{"rowId":i,"input":cases[n]["messages"][-1]["content"],"reference":references[n],"referenceAvailable":True,"baseOutput":r["output"],"hitTokenLimit":r.get("finishReason")=="length"} for n,(i,r) in enumerate(zip(ids,outputs))]})
    return results
