"""Single-run SFT entry point. The API invokes this in WSL after queuing a run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from advisor import classification_label_field, classification_report, detect_schema


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    config = run["config"]
    method = config["parameter_method"]
    if method not in {"lora", "qlora"}:
        raise ValueError("Unsupported parameter method")

    # Imports are intentionally inside the entry point so the API can start before
    # large ML dependencies have completed installation.
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from trl import SFTTrainer

    hp = config["hyperparameters"]
    def load_ref(dataset_info):
        if dataset_info["source"] == "huggingface":
            return load_dataset(dataset_info["name"], split=dataset_info.get("split", "train"))
        path = dataset_info["name"]
        extension = Path(path).suffix.lower()
        loader = {".csv": "csv", ".jsonl": "json", ".parquet": "parquet"}.get(extension)
        if not loader:
            raise RuntimeError("Uploaded dataset must be CSV, JSONL, or Parquet.")
        return load_dataset(loader, data_files=path, split="train")

    dataset_info = config["dataset"]
    dataset = load_ref(dataset_info)
    manifest_path = config.get("split_manifest_path")
    if manifest_path and Path(manifest_path).exists():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        train_dataset = dataset.select(manifest.get("trainIndices", []))
        if config.get("test_dataset"):
            eval_dataset = load_ref(config["test_dataset"]).select(manifest.get("evalIndices", []))
        else:
            eval_dataset = dataset.select(manifest.get("evalIndices", []))
        split = {"train": train_dataset, "test": eval_dataset}
    elif config.get("test_dataset"):
        split = {"train": dataset, "test": load_ref(config["test_dataset"])}
    else:
        split = dataset.train_test_split(test_size=0.1, seed=config["split_seed"])
    adapter = LoraConfig(r=hp["rank"], lora_alpha=hp["alpha"], lora_dropout=0.05, task_type=TaskType.CAUSAL_LM, target_modules="all-linear")
    framework = "Transformers + TRL + PEFT"
    try:
        if method != "qlora":
            raise ImportError("Unsloth fast path is only used for QLoRA.")
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config["base_model"], max_seq_length=hp["max_sequence_length"], load_in_4bit=True
        )
        model = FastLanguageModel.get_peft_model(
            model, r=hp["rank"], lora_alpha=hp["alpha"], lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        framework = "Unsloth + TRL + PEFT"
    except (ImportError, ModuleNotFoundError):
        tokenizer = AutoTokenizer.from_pretrained(config["base_model"], use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4") if method == "qlora" else None
        base = AutoModelForCausalLM.from_pretrained(config["base_model"], quantization_config=quantization, device_map="auto")
        model = get_peft_model(base, adapter)
    output = run_dir / "adapter"
    def format_example(example):
        if "messages" in example:
            return tokenizer.apply_chat_template(example["messages"], tokenize=False)
        if "prompt" in example and "completion" in example:
            return str(example["prompt"]) + str(example["completion"])
        label_field = classification_label_field(list(example))
        if "text" in example and label_field:
            return "Text: " + str(example["text"]) + "\nLabel: " + str(example[label_field])
        if "text" in example:
            return str(example["text"])
        raise ValueError("Dataset needs messages, prompt/completion, or text columns.")

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=split["train"], eval_dataset=split["test"], formatting_func=format_example,
        args=TrainingArguments(output_dir=str(output), num_train_epochs=hp["epochs"], learning_rate=hp["learning_rate"], per_device_train_batch_size=hp["batch_size"], gradient_accumulation_steps=hp["gradient_accumulation"], evaluation_strategy="epoch", logging_steps=5, save_strategy="epoch", report_to=[]),
    )
    trainer.train()
    metrics = trainer.evaluate()
    label_field = classification_label_field(list(split["test"].column_names))
    if detect_schema(list(split["test"].column_names)) == "classification" and label_field:
        labels = sorted({str(value).strip() for value in split["train"][label_field] + split["test"][label_field] if str(value).strip()})
        predictions, report_rows = [], []
        for index, row in enumerate(split["test"]):
            prompt = "Text: " + str(row["text"])[:4000] + "\nLabel:"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            generated_tokens = model.generate(**inputs, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            generated = tokenizer.decode(generated_tokens[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            normalized = generated.lower()
            predicted = next((label for label in labels if label.lower() in normalized), "")
            predictions.append(predicted)
            report_rows.append({**row, "_row_id": index})
        metrics["classification"] = classification_report(report_rows, label_field, predictions)
    model.save_pretrained(output, safe_serialization=True)
    metrics["framework"] = framework
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    exports = run_dir / "exports"
    exports.mkdir(exist_ok=True)
    (exports / "Modelfile").write_text("# Convert merged weights to GGUF first\\nFROM ./model.gguf\\n", encoding="utf-8")


if __name__ == "__main__":
    main()
