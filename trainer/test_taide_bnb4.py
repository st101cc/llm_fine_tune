"""Offline regressions for the T4 TAIDE inference-only baseline."""
import json
from collections import Counter
from types import SimpleNamespace

import pytest

import quality_models as models
import quality_pilot as pilot


def test_t4_requires_explicit_quantized_mode(monkeypatch):
    def inventory(command, **kwargs):
        return "" if "--query-compute-apps=pid" in command else "GPU-test, 14918\n"
    monkeypatch.setattr(pilot.subprocess, "check_output", inventory)
    assert pilot._select_idle_gpu()[1] == "insufficient_vram"
    assert pilot._select_idle_gpu(load_mode="bnb4")[0] == "GPU-test"
    with pytest.raises(ValueError, match="mode"):
        models.generate_answers([], load_mode="other")


@pytest.mark.parametrize("load_mode", ["fp16", "bnb4"])
def test_load_mode_context_and_telemetry(monkeypatch, load_mode):
    import torch
    import transformers
    status = {"status": "ready", "path": "cached", "revision": "a" * 40, "license": {}}
    monkeypatch.setattr(models, "get_model_status", lambda *a, **k: status)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    for name in ("reset_peak_memory_stats", "synchronize", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 123456)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 234567)
    monkeypatch.setattr(transformers, "BitsAndBytesConfig", lambda **k: SimpleNamespace(**k))
    class Inputs(dict):
        def to(self, device):
            return self
    class Tokenizer:
        model_max_length = 8192
        eos_token_id = 2
        chat_template = "fixed"
        def apply_chat_template(self, messages, **kwargs):
            return messages[-1]["content"]
        def __call__(self, text, **kwargs):
            assert kwargs["truncation"] is False
            return Inputs(input_ids=torch.ones((1, 2041 if text == "long" else 3), dtype=torch.long))
        def decode(self, tokens, **kwargs):
            return "A"
    class Model:
        config = SimpleNamespace(max_position_embeddings=8192)
        generation_config = SimpleNamespace(eos_token_id=2)
        device = "cuda:0"
        def eval(self):
            return self
        def generate(self, input_ids, **kwargs):
            assert kwargs["do_sample"] is False
            return torch.cat((input_ids, torch.tensor([[65, 2]])), dim=1)
    loaded = []
    def load(path, **kwargs):
        assert kwargs["local_files_only"] and kwargs["device_map"] == {"": 0}
        loaded.append(kwargs)
        return Model()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load)
    case = {"id": "one", "task": "mcq", "input": "short"}
    answer = models.generate_answers([case], load_mode=load_mode)[0]
    assert answer["protocol"]["loadMode"] == load_mode
    assert answer["provenance"]["revision"] == "a" * 40
    assert answer["telemetry"]["peakAllocatedBytes"] == 123456
    assert answer["telemetry"]["generationSeconds"] >= 0
    if load_mode == "bnb4":
        config = loaded[0]["quantization_config"]
        assert config.load_in_4bit and config.bnb_4bit_use_double_quant
        assert config.bnb_4bit_quant_type == "nf4" and config.bnb_4bit_compute_dtype == torch.float16
        assert answer["protocol"]["contextTokenBudget"] == 2048
        with pytest.raises(ValueError, match="context"):
            models.generate_answers([{**case, "input": "long"}], load_mode=load_mode)
    else:
        assert "quantization_config" not in loaded[0]
        assert answer["protocol"]["contextTokenBudget"] == 4096
        assert models.generate_answers([{**case, "input": "long"}], load_mode=load_mode)


def test_frozen_support_cases_and_binary_metrics():
    from quality_smoke import support_cases, summarize
    cases = support_cases()
    assert cases == support_cases() and len({c["id"] for c in cases}) == 20
    assert Counter(c["task"] for c in cases) == {"classification": 10, "writing": 10}
    classes = [c for c in cases if c["task"] == "classification"]
    answers = [{"candidate": "taide", "caseId": c["id"], "output": c["reference"]} for c in classes]
    answers[0]["output"] = "無法辨識"
    report = summarize(cases, answers, {})
    metrics = report["classification"]
    assert metrics["positiveLabel"] == "需升級處理" and metrics["accuracy"] == .9
    assert metrics["recall"] == .8 and metrics["precision"] == 1
    assert metrics["invalidPredictions"] == 1
    assert report["writing"]["needsReview"] == 10


def test_api_modes_are_validated():
    from quality_api import EvaluationRequest
    from pydantic import ValidationError
    assert EvaluationRequest().loadMode == "fp16"
    assert EvaluationRequest(loadMode="bnb4").loadMode == "bnb4"
    with pytest.raises(ValidationError):
        EvaluationRequest(loadMode="unknown")


def test_review_decisions_require_reasons_and_preserve_original():
    from quality_smoke import support_cases, summarize, review_template
    from quality_policy import normalize_policy
    cases = support_cases()[10:11]
    answers = [{"caseId": cases[0]["id"], "output": "软件有問題。", "candidate": "taide"}]
    policy = normalize_policy({"terms": {"软件": "軟體"}})
    reviews = review_template(cases, answers, policy)
    review = reviews[cases[0]["id"]]
    decision = next(item for item in review["languageDecisions"] if item["finding"]["original"] == "软件")
    decision["decision"] = "accepted"
    with pytest.raises(ValueError, match="reason"):
        summarize(cases, answers, policy, reviews)
    decision["reason"] = "符合本店用語"
    review.update(reviewer="owner", reason="已檢查題目", judgments={"facts": "fail", "instructions": "pass", "taiwanUsage": "fail"})
    report = summarize(cases, answers, policy, reviews)
    assert report["writing"]["failed"] == 1
    assert report["results"][0]["answer"]["output"] == "软件有問題。"
    review["outputHash"] = "changed"
    with pytest.raises(ValueError, match="hash"):
        summarize(cases, answers, policy, reviews)


def test_quantization_modes_cannot_be_directly_paired():
    from quality_evaluation import evaluate, fingerprint
    from quality_policy import normalize_policy
    policy = normalize_policy({})
    cases = [{"id": "one", "task": "mcq", "input": "Q", "reference": "A"}]
    answers = [{"caseId": "one", "candidate": mode, "output": "A", "protocol": models.load_settings(mode), "policyHash": fingerprint(policy)} for mode in ("fp16", "bnb4")]
    result = evaluate(cases, answers, policy)
    assert not result["comparisons"] and result["unpaired"]


def test_bnb4_rejects_small_and_busy_gpus(monkeypatch):
    monkeypatch.setattr(pilot.subprocess, "check_output", lambda command, **kwargs: "" if "--query-compute-apps=pid" in command else "GPU-test, 11999\n")
    assert pilot._select_idle_gpu(load_mode="bnb4")[1] == "insufficient_vram"
    monkeypatch.setattr(pilot.subprocess, "check_output", lambda command, **kwargs: "22" if "--query-compute-apps=pid" in command else "GPU-test, 14918\n")
    assert pilot._select_idle_gpu(load_mode="bnb4")[1] == "gpu_busy"


@pytest.mark.parametrize("state", ["cancelled", "budget_exhausted"])
def test_bounded_bnb4_preserves_checkpoints_and_mode(tmp_path, monkeypatch, state):
    monkeypatch.setattr(models, "get_model_status", lambda *a, **k: {"status": "ready", "revision": "a" * 40})
    answer = {"caseId": "first", "output": "A"}
    def run(root, commands, **kwargs):
        assert kwargs["load_mode"] == "bnb4"
        request = json.loads((kwargs["log_dir"] / "request.json").read_text(encoding="utf-8"))
        assert request["loadMode"] == "bnb4"
        (kwargs["log_dir"] / "answers.json").write_text(json.dumps({"status": "running", "answers": [answer]}), encoding="utf-8")
        return {"status": state, "message": "Stopped"}
    monkeypatch.setattr(pilot, "run_with_pilot_budget", run)
    result = pilot.generate_bounded([{"id": "first", "task": "mcq", "input": "Q"}], runtime_root=tmp_path, load_mode="bnb4")
    assert result["status"] == state and result["answers"] == [answer] and result["loadMode"] == "bnb4"


def test_worker_final_result_uses_atomic_checkpoint(tmp_path, monkeypatch):
    import time
    from pathlib import Path
    request = {"cases": [], "modelId": models.TAIDE_ID, "revision": "a" * 40, "adapterPath": None, "loadMode": "bnb4"}
    (tmp_path / "request.json").write_text(json.dumps(request), encoding="utf-8")
    monkeypatch.setenv("QUALITY_WORKER_DEADLINE", str(time.monotonic() + 30))
    monkeypatch.setattr(pilot, "_arm_worker_deadline", lambda deadline: None)
    original = Path.write_text
    def checked_write(path, data, **kwargs):
        assert path.name != "answers.json", "Final result must not truncate a durable checkpoint"
        return original(path, data, **kwargs)
    monkeypatch.setattr(Path, "write_text", checked_write)
    def generate(*args, **kwargs):
        answers = [{"caseId": "one", "output": "A"}]
        kwargs["on_answer"](answers)
        return answers
    monkeypatch.setattr(models, "generate_answers", generate)
    pilot._worker(tmp_path, models.TAIDE_ID, "generate", "a" * 40)
    assert json.loads((tmp_path / "answers.json").read_text(encoding="utf-8"))["status"] == "complete"


def test_empty_baseline_can_retry_and_preserves_attempts(tmp_path, monkeypatch):
    import quality_smoke as smoke
    import sys
    mcq = [{"id": f"mcq-{i}", "task": "mcq", "input": "Q", "reference": "A"} for i in range(30)]
    monkeypatch.setattr(smoke, "load_benchmark", lambda *a, **k: {"cases": mcq, "manifestChecksum": "frozen"})
    monkeypatch.setattr(smoke, "get_model_status", lambda *a, **k: {"status": "missing_weights"})
    monkeypatch.setattr(smoke, "generate_bounded", lambda *a, **k: {"status": "unavailable", "answers": [], "reason": "gpu_busy"})
    monkeypatch.setattr(sys, "argv", ["quality_smoke", "--runtime-root", str(tmp_path), "--benchmark-revision", "b" * 40, "--model-revision", "a" * 40, "--load-mode", "bnb4"])
    smoke.main()
    smoke.main()
    attempts = list(tmp_path.glob("quality/baselines/*/attempts/*.json"))
    assert len(attempts) == 2


def test_model_download_honors_explicit_immutable_revision(monkeypatch):
    import huggingface_hub
    calls = []
    revision = "a" * 40
    class Api:
        def model_info(self, model_id, **kwargs):
            assert kwargs["revision"] == revision
            return SimpleNamespace(sha=revision, gated=True)
        def auth_check(self, **kwargs):
            calls.append("auth")
    def download(**kwargs):
        assert calls == ["auth"] and kwargs["revision"] == revision
        return "cached"
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert models.prepare_model(revision=revision)["revision"] == revision
