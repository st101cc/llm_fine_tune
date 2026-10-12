import json
from types import SimpleNamespace
from datasets import Dataset
import advisor
import graph


def test_ground_truth_metrics_and_label_boundaries():
    rows = [{"text":"x", "label":"safe"}, {"text":"y", "label":"unsafe"}]
    result = advisor.task_quality(rows, ["safe", "unsafe"], ["unsafe", "unsafe"], [10, 11])
    assert result["qualityEvidence"]["score"] == .5
    assert result["classification"]["accuracy"] == .5
    assert result["classification"]["precision"] == .5
    assert result["classification"]["recall"] == 1
    assert result["classification"]["f1"] == .6667
    assert advisor.task_quality(rows, ["safe", "unsafe"], ["unsafe", "safe"], [10,11])["classification"]["f1"] == 0
    assert advisor.classification_prediction("unsafe", ["safe", "unsafe"]) == "unsafe"
    assert advisor.classification_prediction("This might be safe or unsafe", ["safe", "unsafe"]) == ""
    assert advisor.task_quality([{}], ["A free-form answer"], ["A different answer"], [0])["qualityEvidence"]["score"] is None
    assert advisor.task_quality([{}], ['{"x": 1}'], ['{"x": 2}'], [0])["qualityEvidence"]["score"] == 0
    assert advisor.task_quality([{}], ['{"x": 1}'], ['{"x": true}'], [0])["qualityEvidence"]["score"] == 0
    prompt, reference = advisor.evaluation_prompt(rows[0], SimpleNamespace(apply_chat_template=lambda messages, **kwargs: messages[0]["content"]))
    assert reference == "safe" and "Label:" in prompt and "safe" not in prompt


def test_word_overlap_cannot_recommend_a_fine_tune(tmp_path):
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"evaluationIds":[10,11]}))
    state = {"approved_plan":{"candidates":[{"model":{"id":"m"}}]},"baseline_results":[{"modelId":"m","status":"complete","meanTokenOverlap":.1,"evaluationIds":[10,11]}],"evaluation_results":[{"modelId":"m","status":"complete","artifactValid":True,"meanTokenOverlap":.9,"evaluationIds":[10,11]}],"split_manifest_path":str(manifest)}
    result = graph._compare_results(state)
    assert result["winner"] is None and result["decision"] == "review_required"
    for key, score in [("baseline_results",.5),("evaluation_results",1.0)]:
        state[key][0]["qualityEvidence"] = {"metric":"label_accuracy","score":score,"coverage":1.0,"evaluatedExamples":2}
    assert graph._compare_results(state)["decision"] == "review_fine_tune"
    state["evaluation_results"][0]["qualityEvidence"]["score"] = .5
    assert graph._compare_results(state)["decision"] == "keep_baseline"
    state["evaluation_results"][0]["evaluationIds"] = [12,13]
    assert graph._compare_results(state)["decision"] == "review_required"


def test_ordinary_english_does_not_select_a_code_model():
    rows = [{"messages":[{"role":"user","content":f"Please return advice from an English class, example {i}."},{"role":"assistant","content":"Use clear sentences."}]} for i in range(30)]
    assert advisor.local_profile(Dataset.from_list(rows))["classification"]["task"] == "assistant"


    code_rows = [{"messages":[{"role":"user", "content":"Write Python"}, {"role":"assistant", "content":"def add(a, b):\n    return a + b"}]} for _ in range(30)]
    assert advisor.local_profile(Dataset.from_list(code_rows))["classification"]["task"] == "code"
