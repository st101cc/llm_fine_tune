import asyncio
import json
from types import SimpleNamespace

import pytest
from datasets import Dataset
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
import advisor
import graph
from train import training_text_dataset


def test_pilot_fixtures_keep_inputs_disjoint():
    from pipeline_pilot import fixtures
    from advisor import evaluation_messages
    for task in ("classification", "extraction", "summarization"):
        rows = fixtures(task)
        inputs = [json.dumps(evaluation_messages(row)[0], sort_keys=True) for row in rows]
        assert len(inputs) == len(set(inputs)) == 156
        assert all(evaluation_messages(row)[1] for row in rows)


def test_audit_samples_late_rows_reproducibly():
    data = Dataset.from_list([{"text": f"Example {i}" if i < 600 else ""} for i in range(1200)])
    first = advisor.local_profile(data)
    assert first == advisor.local_profile(data)
    assert first["facts"]["rejectedRows"] > 200
    assert all(i >= 600 for i in first["facts"]["rejectedRowIds"])


def test_training_rejects_legacy_manifest_with_duplicate_inputs():
    from train import manifest_training_split
    data = Dataset.from_list([{"prompt": "Same question", "completion": "A"}, {"prompt": "same question", "completion": "B"}, {"prompt": "Other", "completion": "C"}])
    manifest = {"trainIndices": [0], "developmentIndices": [1], "evalIndices": [2], "trainDataset": {"name": "same"}, "evalDataset": {"name": "same"}}
    with pytest.raises(ValueError, match="overlap"):
        manifest_training_split(data, manifest)


def test_privacy_sample_preserves_original_row_ids(tmp_path, monkeypatch):
    data = Dataset.from_list([{"prompt": f"Question {i}", "completion": "Contact person@example.com" if i >= 600 else "Public answer"} for i in range(1200)])
    monkeypatch.setattr(graph, "load_source", lambda *args: data)
    workflow = graph.build_workflow_graph(graph.WorkflowServices(tmp_path, lambda: 15, None, None, lambda key: tmp_path / key), MemorySaver())
    result = asyncio.run(workflow.ainvoke({"workflow_id": "privacy", "dataset": {"source": "upload", "name": "fixture"}, "analysis_only": True, "auto_analysis": False}, {"configurable": {"thread_id": "privacy"}}))
    assert result["privacy_review"]["flaggedRows"]
    assert all(i >= 600 for i in result["privacy_review"]["flaggedRows"])


def test_duplicate_inputs_stay_in_one_split_and_dev_is_not_capped_at_ten(tmp_path):
    data = Dataset.from_list([{"prompt": f"Question {i % 600}", "completion": str(i)} for i in range(1200)])
    services = graph.WorkflowServices(tmp_path, lambda: 15, None, None, lambda key: tmp_path / key)
    state = {"workflow_id": "grouped", "dataset": {"source": "upload", "name": "fixture"}, "analysis_limit": 50}
    manifest = graph._make_split_manifest(state, services, data, advisor.local_profile(data))
    groups = [{data[i]["prompt"] for i in manifest[key]} for key in ("trainIndices", "developmentIndices", "evalIndices")]
    assert all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i))
    assert len(manifest["developmentIndices"]) >= 50
    assert sum(len(manifest[k]) for k in ("trainIndices", "developmentIndices", "evalIndices")) == 1200


def test_external_test_overlap_is_rejected(tmp_path, monkeypatch):
    data = Dataset.from_list([{"prompt": f"Q{i}", "completion": "A"} for i in range(100)])
    monkeypatch.setattr(graph, "load_source", lambda *args: Dataset.from_list([{"prompt": "Q1", "completion": "Different answer"}]))
    services = graph.WorkflowServices(tmp_path, lambda: 15, None, None, lambda key: tmp_path / key)
    with pytest.raises(ValueError, match="overlap"):
        graph._make_split_manifest({"workflow_id": "external", "dataset": {"name": "train"}, "test_dataset": {"source": "upload", "name": "test"}}, services, data, advisor.local_profile(data))


class Tokenizer:
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "".join(m["role"] + ":" + m["content"] + "\n" for m in messages) + ("assistant:" if add_generation_prompt else "")


def test_response_masks_exclude_context_and_reject_lost_targets():
    tokenizer = Tokenizer()
    data = Dataset.from_list([{"messages": [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}], "prompt": "metadata"}])
    row = training_text_dataset(data, tokenizer)[0]
    target = "".join(chr(t) for t, mask in zip(row["input_ids"], row["completion_mask"]) if mask and t)
    assert target == "Answer\n"
    assert not any(row["completion_mask"][:len("user:Question\nassistant:")])
    with pytest.raises(ValueError, match="target"):
        training_text_dataset(data, tokenizer, max_length=5)


def test_new_workflow_requires_task_before_gpu_work(tmp_path, monkeypatch):
    data = Dataset.from_list([{"prompt": f"Q{i}", "completion": "A"} for i in range(100)])
    monkeypatch.setattr(graph, "load_source", lambda *args: data)
    def no_gpu():
        pytest.fail("GPU queried before task confirmation")
    workflow = graph.build_workflow_graph(graph.WorkflowServices(tmp_path, no_gpu, None, None, lambda key: tmp_path / key), MemorySaver())
    result = asyncio.run(workflow.ainvoke({"workflow_id": "intent", "dataset": {"source": "upload", "name": "fixture"}, "task_description": "", "success_metric": ""}, {"configurable": {"thread_id": "intent"}}))
    assert result["__interrupt__"][0].value["type"] == "task_review"
    cancelled = asyncio.run(workflow.ainvoke(Command(resume={"action": "abort"}), {"configurable": {"thread_id": "intent"}}))
    assert cancelled["status"] == "cancelled"


def test_final_regression_cannot_retrain(tmp_path, monkeypatch):
    data = Dataset.from_list([{"prompt": "Q", "completion": '{"x":1}'}])
    monkeypatch.setattr(graph, "load_source", lambda *args: data)
    def record(score):
        return {"modelId": "m", "status": "complete", "evaluationIds": [0], "formatValidRate": score, "qualityEvidence": {"metric": "json_exact_match", "score": score, "coverage": 1, "evaluatedExamples": 1}, "samples": []}
    monkeypatch.setattr(graph, "baseline_compare", lambda *args, **kwargs: [record(1)])
    monkeypatch.setattr(graph, "evaluate_tuned_run", lambda *args, **kwargs: record(0))
    monkeypatch.setattr(graph, "evaluate_results", lambda dataset, results, **kwargs: results)
    async def no_training(config):
        pytest.fail("Final-test results triggered retraining")
    async def training_record(key):
        return {"status": "complete"}
    services = graph.WorkflowServices(tmp_path, lambda: 15, no_training, training_record, lambda key: tmp_path)
    workflow = graph.build_workflow_graph(services, MemorySaver())
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"evalIndices": [0], "evaluationIds": [0], "evalDataset": {"source": "upload", "name": "fixture"}}))
    state = {"workflow_id": "final", "dataset": {"source": "upload", "name": "fixture"}, "approved_plan": {"candidates": [{"model": {"id": "m"}, "method": "qlora"}]}, "candidate_index": 0, "candidate_status": "verified", "training_runs": {"final:m:0": {"runId": "run"}}, "split_manifest_path": str(manifest), "profile": {"classification": {"task": "structured"}}, "max_retries": 2}
    state["split_manifest_id"] = "final-split"
    async def run():
        config = {"configurable": {"thread_id": "final"}}
        await workflow.aupdate_state(config, state, as_node="verify_artifacts")
        return await workflow.ainvoke(None, config)
    assert asyncio.run(run())["status"] == "complete"
