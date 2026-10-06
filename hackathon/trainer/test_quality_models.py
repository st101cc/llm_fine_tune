"""Offline model, benchmark and process-budget tests. No model downloads."""
import importlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


class Sidecar:
    def __init__(self, module):
        self.module = module

    def __getattr__(self, name):
        assert importlib.util.find_spec(self.module), f"{self.module} sidecar is not implemented"
        return getattr(importlib.import_module(self.module), name)


@pytest.fixture
def models():
    return Sidecar("quality_models")


@pytest.fixture
def pilot():
    return Sidecar("quality_pilot")


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    import huggingface_hub
    path = tmp_path / "snapshots" / ("a" * 40)
    path.mkdir(parents=True)
    for name, content in {"config.json": '{"max_position_embeddings": 512}', "tokenizer_config.json": '{}', "tokenizer.json": '{}', "model.safetensors": "fixture weights", "README.md": "---\nlicense: llama3.1\n---\nFixture", "LICENSE": "Fixture license"}.items():
        (path / name).write_text(content, encoding="utf-8")
    def cached(**kwargs):
        assert kwargs["local_files_only"] is True
        return str(path)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", cached)
    return path


def test_status_offline_revision_license_and_missing_shard(models, snapshot):
    status = models.get_model_status()
    assert status["status"] == "ready"
    assert status["revision"] == "a" * 40
    assert status["license"]["id"] == "llama3.1"
    assert status["license"]["acceptance"] == "not_checked_offline"
    (snapshot / "model.safetensors").unlink()
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "missing.safetensors"}}))
    assert models.get_model_status()["status"] == "missing_weights"


def test_missing_cache_is_actionable_and_never_downloads(models, monkeypatch):
    import huggingface_hub
    def missing(**kwargs):
        assert kwargs["local_files_only"] is True
        raise FileNotFoundError("empty cache")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing)
    status = models.get_model_status()
    assert status["status"] == "missing_weights"
    assert "prepare" in status["message"]
    with pytest.raises(RuntimeError, match="prepare"):
        models.generate_answers([{"id": "a", "input": "Q", "task": "mcq"}])


def test_missing_license_blocks_generation(models, snapshot):
    (snapshot / "LICENSE").unlink()
    (snapshot / "README.md").write_text("No license metadata")
    assert models.get_model_status()["status"] == "missing_license"


def test_generation_greedy_no_truncation_and_provenance(models, snapshot, monkeypatch):
    import torch
    import transformers
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    class Inputs(dict):
        def to(self, device):
            return self
    class Tokenizer:
        model_max_length = 512
        eos_token_id = 2
        def apply_chat_template(self, messages, **kwargs):
            assert "A" in messages[0]["content"]
            return messages[-1]["content"]
        def __call__(self, prompt, **kwargs):
            assert kwargs.get("truncation") is False
            return Inputs(input_ids=torch.tensor([[1] * (510 if "long" in prompt else 3)]))
        def decode(self, ids, **kwargs):
            return "A"
    class Model:
        device = "cuda:0"
        config = SimpleNamespace(max_position_embeddings=512)
        generation_config = SimpleNamespace(eos_token_id=2)
        def eval(self):
            return self
        def generate(self, input_ids, **kwargs):
            assert kwargs["do_sample"] is False and kwargs["max_new_tokens"] == 8
            return torch.tensor([input_ids[0].tolist() + [65, 2]])
    def load_tokenizer(path, **kwargs):
        assert Path(path) == snapshot and kwargs["local_files_only"] is True
        return Tokenizer()
    def load_model(path, **kwargs):
        assert Path(path) == snapshot and kwargs["local_files_only"] is True
        assert kwargs["device_map"] == {"": 0}
        return Model()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load_model)
    cases = [{"id": "x", "input": "Q", "task": "mcq"}]
    rows = models.generate_answers(cases, revision="a" * 40)
    assert rows[0]["output"] == "A" and rows[0]["caseId"] == "x"
    assert rows[0]["provenance"]["revision"] == "a" * 40
    with pytest.raises(ValueError, match="truncat|context"):
        models.generate_answers([{**cases[0], "input": "long"}])
    assert models.generate_answers(cases, cancelled=lambda: True) == []


def test_eight_step_training_uses_safe_local_boundary(pilot, snapshot, tmp_path, monkeypatch):
    import torch
    import transformers
    from datasets import Dataset
    trl = SimpleNamespace()
    monkeypatch.setitem(transformers.__dict__, "set_seed", lambda seed: None)
    monkeypatch.setitem(sys.modules, "trl", trl)
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(LoraConfig=lambda **kw: SimpleNamespace(**kw), TaskType=SimpleNamespace(CAUSAL_LM="CAUSAL_LM")))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    class Tokenizer:
        eos_token_id = 2
        eos_token = "<eos>"
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return "".join(m["role"] + ":" + m["content"] + "\n" for m in messages) + ("assistant:" if add_generation_prompt else "")
        def encode(self, text, add_special_tokens=False):
            return [ord(c) + 10 for c in text]
    def load_tokenizer(path, **kwargs):
        assert Path(path) == snapshot and kwargs["local_files_only"] is True
        return Tokenizer()
    def load_model(path, **kwargs):
        assert Path(path) == snapshot and kwargs["local_files_only"] is True
        return SimpleNamespace(config=SimpleNamespace(use_cache=True))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load_model)
    class Trainer:
        def __init__(self, **kwargs):
            args, config, data = kwargs["args"], kwargs["peft_config"], kwargs["train_dataset"]
            assert args.max_steps == 8 and args.max_length == 512 and args.learning_rate == .0001
            assert args.dataloader_num_workers == 0 and args.report_to == []
            assert config.r == 8 and config.lora_alpha == 16
            assert isinstance(data, Dataset) and len(data) == 8
            assert all(any(row["completion_mask"]) and 0 in row["completion_mask"] for row in data)
            self.state = SimpleNamespace(global_step=0)
        def train(self):
            self.state.global_step = 8
            return SimpleNamespace(training_loss=.25)
        def save_model(self, path):
            Path(path).mkdir(parents=True)
    monkeypatch.setattr(trl, "SFTTrainer", Trainer, raising=False)
    # SFTConfig validates the actual host CUDA backend; keep its boundary data only.
    monkeypatch.setattr(trl, "SFTConfig", lambda **kw: SimpleNamespace(**kw), raising=False)
    for variant in ("original", "cleaned"):
        pilot._train_worker(tmp_path, "taide/Llama-3.1-TAIDE-LX-8B-Chat", variant, "a" * 40)
    original = json.loads((tmp_path / "original" / "training.json").read_text(encoding="utf-8"))
    cleaned = json.loads((tmp_path / "cleaned" / "training.json").read_text(encoding="utf-8"))
    assert original["parameters"] == cleaned["parameters"]
    assert original["steps"] == cleaned["steps"] == 8


def test_prepare_benchmark_pins_sha_and_keeps_it_out_of_datasets(models, tmp_path, monkeypatch):
    import huggingface_hub
    sha = "b" * 40
    subjects = ["computer_science", "finance_banking", "official_document_management", "accounting"]
    class Api:
        def dataset_info(self, repo_id, revision):
            assert repo_id == "ikala/tmmluplus" and revision == "v1.1"
            return SimpleNamespace(sha=sha, siblings=[SimpleNamespace(rfilename=f"data/{s}_test.csv") for s in subjects])
    def download(repo_id, filename, **kwargs):
        assert kwargs["revision"] == sha and kwargs["repo_type"] == "dataset"
        path = Path(kwargs["local_dir"]) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("question,A,B,C,D,answer\n" + "\n".join(f"{filename} Q{i},one,two,three,four,A" for i in range(12)), encoding="utf-8")
        return str(path)
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    manifest = models.prepare_benchmark(tmp_path / "quality" / "benchmarks")
    assert len(manifest["cases"]) == 30
    assert manifest["revision"] == sha
    assert {c["subject"] for c in manifest["cases"]} == set(subjects[:3])
    loaded = models.load_benchmark(sha, runtime_root=tmp_path, expected_checksum=manifest["manifestChecksum"])
    assert loaded["cases"] == manifest["cases"]
    source = tmp_path / "quality" / "benchmarks" / "tmmluplus" / sha / "data" / "computer_science_test.csv"
    source.write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        models.load_benchmark(sha, runtime_root=tmp_path)
    assert len(models.prepare_benchmark(tmp_path / "benchmarks", all_subjects=True)["cases"]) == 40
    with pytest.raises(ValueError, match="benchmarks"):
        models.prepare_benchmark(tmp_path / "datasets")


def test_model_prepare_checks_existing_gate_before_download(models, monkeypatch):
    import huggingface_hub
    events = []
    class Api:
        def model_info(self, repo_id):
            return SimpleNamespace(sha="c" * 40, gated="manual")
        def auth_check(self, **kwargs):
            events.append("authorized")
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    def download(**kwargs):
        assert events == ["authorized"]
        assert kwargs["revision"] == "c" * 40 and kwargs["local_files_only"] is False
        return "cached"
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert models.prepare_model()["revision"] == "c" * 40
    def denied(**kwargs):
        raise PermissionError("gate pending")
    monkeypatch.setattr(Api, "auth_check", lambda self, **kwargs: denied(**kwargs))
    with pytest.raises(RuntimeError, match="authorized|gate"):
        models.prepare_model()


def test_fixture_partitions_are_fixed_and_independent(pilot):
    f = pilot.fixtures()
    assert f == pilot.fixtures()
    inputs = [{r["input"] for r in f[split]} for split in ("train", "development", "test")]
    assert all(inputs[i].isdisjoint(inputs[j]) for i in range(3) for j in range(i))
    original = pilot.training_rows("original")
    cleaned = pilot.training_rows("cleaned")
    assert len(original) == len(cleaned) == 8
    assert [r["messages"][0] for r in original] == [r["messages"][0] for r in cleaned]
    assert original != cleaned
    assert pilot.training_parameters()["max_steps"] == 8
    assert pilot.training_parameters()["rank"] == 8
    assert pilot.training_parameters()["alpha"] == 16
    assert pilot.training_parameters()["learning_rate"] == 0.0001
    assert pilot.training_parameters()["max_sequence_length"] == 512
    assert {row["task"] for rows in f.values() for row in rows} == {"json"}
    assert all(row["reviewDecisions"] and all(d["decision"] == "accept" for d in row["reviewDecisions"]) for row in f["train"])
    pilot.validate_partitions(f)
    f["development"][0]["reference"] = f["train"][0]["reference"]
    with pytest.raises(ValueError, match="overlap"):
        pilot.validate_partitions(f)


def test_worker_has_independent_deadline_if_parent_disappears(pilot):
    command = [sys.executable, "-B", "-c", "import sys,time; sys.path.insert(0, 'hackathon/trainer'); from quality_pilot import _arm_worker_deadline; _arm_worker_deadline(time.monotonic()+.15); time.sleep(30)"]
    completed = subprocess.run(command, timeout=5)
    assert completed.returncode == 124


@pytest.fixture
def free_gpu(monkeypatch):
    def query(command, **kwargs):
        if "--query-gpu=uuid" in command:
            return "GPU-fixture\n"
        return ""
    monkeypatch.setattr(subprocess, "check_output", query)


def test_budget_kills_loading_process_and_does_not_start_next(pilot, free_gpu, tmp_path):
    marker = tmp_path / "unexpected"
    commands = [[sys.executable, "-c", "import time; time.sleep(30)"], [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"]]
    before = time.monotonic()
    result = pilot.run_budgeted(commands, seconds=0.3)
    assert result["status"] == "budget_exhausted"
    assert len(result["phases"]) == 1 and not marker.exists()
    assert time.monotonic() - before < 5
    assert result["chargedGpuSeconds"] >= 0.3


def test_budget_charges_failures_and_cancellation(pilot, free_gpu):
    result = pilot.run_budgeted([[sys.executable, "-c", "import time; time.sleep(.05); raise SystemExit(3)"]], seconds=5)
    assert result["status"] == "failed" and result["phases"][0]["exitCode"] == 3
    assert result["chargedGpuSeconds"] > .05
    result = pilot.run_budgeted([[sys.executable, "-c", "raise SystemExit(99)"]], seconds=5, cancelled=lambda: True)
    assert result["status"] == "cancelled" and result["phases"] == []


@pytest.mark.parametrize("seconds,gpu", [(1801, "0"), (0, "0"), (float("nan"), "0"), (10, "0,1")])
def test_budget_refuses_invalid_limits_and_multiple_gpus(pilot, seconds, gpu):
    with pytest.raises(ValueError):
        pilot.run_budgeted([], seconds=seconds, gpu=gpu)


def test_budget_refuses_unrelated_gpu_processes(pilot, monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", lambda command, **kw: "GPU-fixture\n" if "--query-gpu=uuid" in command else "9999\n")
    with pytest.raises(RuntimeError, match="busy|process"):
        pilot.run_budgeted([], seconds=1)


def test_budget_is_shared_across_successful_phases(pilot, free_gpu):
    commands = [[sys.executable, "-c", "import time; time.sleep(.25)"] for _ in range(6)]
    result = pilot.run_budgeted(commands, seconds=.7)
    assert result["status"] == "budget_exhausted"
    assert 1 <= len(result["phases"]) < 6
    assert result["chargedGpuSeconds"] >= .7


def test_prepared_sha_only_snapshot_is_discoverable(models, tmp_path, monkeypatch):
    import huggingface_hub
    path = tmp_path / "snapshots" / ("d" * 40)
    path.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors", "LICENSE"):
        (path / name).write_text("fixture")
    def no_ref(**kwargs):
        assert kwargs["local_files_only"]
        raise FileNotFoundError("No mutable main ref for explicitly downloaded SHA")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", no_ref)
    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", lambda: SimpleNamespace(repos=[SimpleNamespace(repo_id=models.TAIDE_ID, repo_type="model", revisions=[SimpleNamespace(commit_hash="d" * 40, snapshot_path=path, last_modified=1)])]))
    status = models.get_model_status()
    assert status["status"] == "ready" and status["revision"] == "d" * 40


def test_benchmark_loader_verifies_runtime_containment_and_checksums(models, tmp_path):
    root = tmp_path / "runtime"
    directory = root / "quality" / "benchmarks" / "tmmluplus" / ("a" * 40)
    directory.mkdir(parents=True)
    manifest = {"datasetId": "ikala/tmmluplus", "version": "v1.1", "revision": "a" * 40, "cases": [], "casesHash": "invalid", "files": {}}
    path = directory / "default-30.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        models.load_benchmark("a" * 40, runtime_root=root)
    with pytest.raises(ValueError):
        models.load_benchmark("../outside", runtime_root=root)


def test_persistent_budget_charges_failed_processes_and_cannot_reset(pilot, free_gpu, tmp_path, monkeypatch, models, snapshot):
    import subprocess
    import huggingface_hub
    def query(command, **kwargs):
        if "--query-gpu=uuid,memory.free" in command:
            return "GPU-fixture, 30000\n"
        if "--query-gpu=uuid" in command:
            return "GPU-fixture\n"
        return ""
    monkeypatch.setattr(subprocess, "check_output", query)
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    class FailedProcess:
        pid = 123
        returncode = 3
        def __init__(self, *args, **kwargs):
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-fixture"
            assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
            clock[0] += 10
        def poll(self):
            return 3
    monkeypatch.setattr(subprocess, "Popen", FailedProcess)
    cases = [{"id": "x", "input": "Q", "task": "mcq"}]
    first = pilot.generate_bounded(cases, runtime_root=tmp_path, max_seconds=100)
    assert first["status"] == "unavailable"
    assert first["budget"]["chargedGpuSeconds"] == 10
    second = pilot.generate_bounded(cases, runtime_root=tmp_path, max_seconds=100)
    assert second["budget"]["chargedGpuSeconds"] == 20
    assert pilot.budget_status(tmp_path)["remainingGpuSeconds"] == 1780


@pytest.mark.parametrize("inventory,pids,reason", [("GPU-busy, 40000\n", "999\n", "gpu_busy"), ("GPU-small, 1000\n", "", "insufficient_vram")])
def test_bounded_generation_refuses_busy_or_small_gpu(pilot, models, snapshot, tmp_path, monkeypatch, inventory, pids, reason):
    monkeypatch.setattr(subprocess, "check_output", lambda command, **kw: inventory if "--query-gpu=uuid,memory.free" in command else pids)
    result = pilot.generate_bounded([{"id": "x", "input": "Q", "task": "mcq"}], runtime_root=tmp_path)
    assert result["status"] == "unavailable" and result["reason"] == reason
    assert result["budget"]["chargedGpuSeconds"] == 0


def test_persistent_reservation_survives_process_restart(pilot, tmp_path):
    reservation = pilot._reserve_budget(tmp_path, 1800)
    assert reservation["reservedGpuSeconds"] == 1800
    assert pilot.budget_status(tmp_path)["remainingGpuSeconds"] == 0
    assert pilot._reserve_budget(tmp_path, 1)["reservedGpuSeconds"] == 0


def test_training_command_wrapper_uses_shared_budget_and_offline_env(pilot, tmp_path, monkeypatch):
    def query(command, **kwargs):
        if "--query-gpu=uuid,memory.free" in command:
            return "GPU-fixture, 30000\n"
        return "GPU-fixture\n" if "--query-gpu=uuid" in command else ""
    monkeypatch.setattr(subprocess, "check_output", query)
    command = [sys.executable, "-c", "import os; assert os.environ['HF_HUB_OFFLINE']=='1'; assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-fixture'"]
    first = pilot.run_with_pilot_budget(tmp_path, [command], max_seconds=5)
    assert first["status"] == "complete"
    assert 0 < first["budget"]["chargedGpuSeconds"] < 5
    second = pilot.run_with_pilot_budget(tmp_path, [[sys.executable, "-c", "raise SystemExit(9)"]], max_seconds=5)
    assert second["status"] == "failed"
    assert second["budget"]["chargedGpuSeconds"] > first["budget"]["chargedGpuSeconds"]


def test_supervisor_enforces_deadline_without_api_parent(pilot):
    import os
    command = [sys.executable, "-B", "hackathon/trainer/quality_pilot.py", "--supervise", sys.executable, "-c", "import time; time.sleep(30)"]
    completed = subprocess.run(command, env={**os.environ, "QUALITY_WORKER_DEADLINE": str(time.monotonic() + .3)}, timeout=5, start_new_session=os.name != "nt")
    assert completed.returncode == (124 if os.name == "nt" else -9)


def test_busy_between_phases_does_not_refund_previous_gpu_use(pilot, tmp_path, monkeypatch):
    probes = [0]
    def query(command, **kwargs):
        if "--query-gpu=uuid,memory.free" in command:
            return "GPU-fixture, 30000\n"
        if "--query-gpu=uuid" in command:
            return "GPU-fixture\n"
        probes[0] += 1
        return "999\n" if probes[0] >= 3 else ""
    monkeypatch.setattr(subprocess, "check_output", query)
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    class SuccessfulProcess:
        pid = 123
        returncode = 0
        def __init__(self, *args, **kwargs):
            clock[0] += 10
        def poll(self):
            return 0
    monkeypatch.setattr(subprocess, "Popen", SuccessfulProcess)
    result = pilot.run_with_pilot_budget(tmp_path, [["fixture"], ["fixture"]], max_seconds=100)
    assert result["status"] == "unavailable"
    assert result["budget"]["chargedGpuSeconds"] == 10
