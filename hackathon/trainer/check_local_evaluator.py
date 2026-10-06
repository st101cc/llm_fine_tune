"""Run with the backend's CUDA environment and cached EVALUATOR_MODEL."""
import json
from datasets import Dataset
from evaluator import evaluate_results


if __name__ == "__main__":
    rows = Dataset.from_list([
        {"prompt": "Calculate 6 times 7 and explain briefly.", "completion": "Six groups of seven make 42.", "domain": "math"},
        {"prompt": "Calculate 6 times 7 and explain briefly.", "completion": "Six groups of seven make 42.", "domain": "math"},
        {"prompt": "Explain the result of an unspecified experiment.", "completion": "", "domain": "factual"},
    ])
    result = evaluate_results(rows, [{"modelId": "hidden-candidate", "status": "complete", "evaluationIds": [0, 1, 2], "samples": [
        {"rowId": 0, "baseOutput": "6 times 7 is 42, since adding seven six times gives 42."},
        {"rowId": 1, "baseOutput": "6 times 7 is 41, since adding seven six times gives 41."},
        {"rowId": 2, "baseOutput": "It succeeded."},
    ]}])[0]
    print(json.dumps({"qualityEvidence": result["qualityEvidence"], "grades": [s.get("grade") for s in result["samples"]]}, indent=2))
    assert [s["grade"]["verdict"] for s in result["samples"]] == ["pass", "fail", "abstain"]
    assert result["qualityEvidence"]["score"] is None
    print("Real local judge: correct answer passed, incorrect answer failed, missing reference abstained.")
