from __future__ import annotations

from pathlib import Path

from graph import WorkflowServices, _blocker_assessment, _compare_results, _make_split_manifest, _validate_plan


class FakeDataset:
    column_names = ["prompt", "completion"]

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def __iter__(self):
        return iter(self.rows)


def test_plan_rejects_14b_lora():
    valid, reason = _validate_plan(
        {
            "candidates": [
                {"model": {"id": "Qwen/Qwen2.5-14B-Instruct"}, "method": "lora", "estimatedVramGb": 22}
            ]
        },
        24,
    )
    assert not valid
    assert "QLoRA" in reason


def test_blocker_agent_requires_approval_for_upload_cleanup():
    state = {
        "dataset": {"source": "upload", "name": "data.jsonl"},
        "findings": [{"severity": "blocker", "code": "duplicates", "message": "Duplicates exceed the limit."}],
    }
    assessment = _blocker_assessment(state)
    assert assessment["decision"] == "approval_required"
    assert assessment["canMaterializeAfterApproval"] is True
    assert assessment["operations"] == ["deduplicate_exact_rows"]


def test_blocker_agent_requires_human_for_small_dataset():
    state = {
        "dataset": {"source": "upload", "name": "data.jsonl"},
        "findings": [{"severity": "blocker", "code": "too_small", "message": "Need more examples."}],
    }
    assessment = _blocker_assessment(state)
    assert assessment["decision"] == "human_required"
    assert assessment["canMaterializeAfterApproval"] is False


def test_split_manifest_is_reproducible(tmp_path: Path):
    rows = [{"prompt": f"question {index}", "completion": f"answer {index}"} for index in range(50)]
    dataset = FakeDataset(rows)
    services = WorkflowServices(
        data_root=tmp_path,
        gpu_memory_gb=lambda: 24,
        submit_training=None,
        get_training=None,
        training_dir=lambda run_id: tmp_path / run_id,
    )
    state = {"workflow_id": "wf-1", "dataset": {"source": "upload", "name": "data.jsonl"}, "split_seed": 42}
    profile = {"facts": {"schema": "prompt_completion"}}
    first = _make_split_manifest(state, services, dataset, profile)
    second = _make_split_manifest(state, services, dataset, profile)
    assert first["trainIndices"] == second["trainIndices"]
    assert first["evalIndices"] == second["evalIndices"]
    assert set(first["trainIndices"]).isdisjoint(first["evalIndices"])


def test_comparison_ranks_valid_tuned_candidates(tmp_path: Path):
    split_path = tmp_path / "split.json"
    split_path.write_text('{"evaluationIds": [3, 7]}', encoding="utf-8")
    state = {
        "goal": "quality",
        "split_manifest_path": str(split_path),
        "approved_plan": {"candidates": [{"model": {"id": "small"}}, {"model": {"id": "large"}}]},
        "baseline_results": [
            {"modelId": "small", "status": "complete", "meanTokenOverlap": 0.4},
            {"modelId": "large", "status": "complete", "meanTokenOverlap": 0.4},
        ],
        "evaluation_results": [
            {"modelId": "small", "status": "complete", "artifactValid": True, "meanTokenOverlap": 0.5},
            {"modelId": "large", "status": "complete", "artifactValid": True, "meanTokenOverlap": 0.7},
        ],
    }
    for group in ("baseline_results", "evaluation_results"):
        for result in state[group]:
            result["evaluationIds"] = [3, 7]
            result["qualityEvidence"] = {"metric":"label_accuracy", "score":result["meanTokenOverlap"], "coverage":1, "evaluatedExamples":2}
    comparison = _compare_results(state)
    assert comparison["winner"] == "large"
    assert comparison["evaluationIds"] == [3, 7]
    assert len(comparison["pairs"]) == 2
