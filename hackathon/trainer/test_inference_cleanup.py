from contextlib import nullcontext
from types import SimpleNamespace
import sys
import weakref
import pytest
import advisor


@pytest.mark.parametrize("fail_generation", [False, True])
def test_evaluation_releases_model_cycles_before_emptying_cuda_cache(monkeypatch, fail_generation):
    models, cleanup_calls = [], []

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        eos_token = "eos"
        eos_token_id = 0
        def __call__(self, *args, **kwargs):
            return Batch(input_ids=SimpleNamespace(shape=(1, 2)))
        def decode(self, *args, **kwargs):
            return "answer"
        def encode(self, *args, **kwargs):
            return [0, 0]

    class Tokens(list):
        @property
        def shape(self):
            return (len(self),)

    class Model:
        device = "cpu"
        def __init__(self):
            self.hook_cycle = self
            models.append(weakref.ref(self))
        def __call__(self, **kwargs):
            return SimpleNamespace(loss=None)
        def generate(self, **kwargs):
            if fail_generation:
                raise RuntimeError("Simulated inference failure")
            return [Tokens([0, 0, 1])]

    def empty_cache():
        assert all(reference() is None for reference in models)
        cleanup_calls.append(True)

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **k: Model()),
        BitsAndBytesConfig=lambda **kwargs: None))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        no_grad=nullcontext, cuda=SimpleNamespace(is_available=lambda: True, empty_cache=empty_cache)))
    monkeypatch.setattr(advisor, "evaluation_prompt", lambda *args: ("question", "answer"))
    dataset = SimpleNamespace(select=lambda indices: [{}], __len__=lambda: 1)
    result = advisor.baseline_compare(dataset, [{"model":{"id":"test"},"method":"lora"}], 1, eval_indices=[0])
    assert result[0]["status"] == ("unavailable" if fail_generation else "complete")
    assert cleanup_calls == [True]


def test_workflow_inference_uses_a_separate_process():
    import os
    from graph import isolated_inference
    assert isolated_inference(os.getpid) != os.getpid()
