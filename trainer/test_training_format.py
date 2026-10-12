from types import SimpleNamespace
from datasets import Dataset
from train import training_text_dataset
from test_pipeline_safeguards import Tokenizer


def test_chat_metadata_does_not_select_prompt_completion_format():
    dataset = Dataset.from_list([{"messages": [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}], "prompt": "Metadata copy", "category": "test"}])
    tokenizer = Tokenizer()
    result = training_text_dataset(dataset, tokenizer)
    assert set(result.column_names) == {"input_ids", "completion_mask", "token_count"}
    decoded = "".join(chr(i) for i in result[0]["input_ids"] if i)
    assert decoded == "user:Question\nassistant:Answer\n"
    assert "Metadata" not in decoded
    assert dataset.column_names == ["messages", "prompt", "category"]


def test_training_validation_keeps_final_test_unseen():
    from train import manifest_training_split
    import pytest
    data = Dataset.from_list([{"text":str(i)} for i in range(6)])
    manifest = {"trainIndices":[0,1,2], "developmentIndices":[3], "evalIndices":[4,5], "trainDataset":{"name":"same"}, "evalDataset":{"name":"same"}}
    split = manifest_training_split(data, manifest)
    assert list(split["train"]["text"]) == ["0","1","2"]
    assert list(split["test"]["text"]) == ["3"]
    with pytest.raises(ValueError, match="overlap"):
        manifest_training_split(data, {**manifest, "developmentIndices":[2]})
    with pytest.raises(ValueError, match="development"):
        manifest_training_split(data, {**manifest, "developmentIndices":[]})
