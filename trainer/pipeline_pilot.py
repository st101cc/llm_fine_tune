"""Bounded synthetic pipeline smoke tests, not a production quality benchmark.

Run on the VM with --output pointing to a new runtime directory and --model to
a cached local model snapshot. No downloads, hosted API calls, or deployment.
"""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def fixtures(task):
    rows = []
    for i in range(156):
        if task == "classification":
            label = ("billing", "delivery", "account")[i % 3]
            issue = {"billing": "I was charged twice for my order", "delivery": "My package has not arrived", "account": "I cannot sign into my account"}[label]
            rows.append({"text": f"Classify this ticket as billing, delivery, or account. Return only the label. Ticket {1000+i}: {issue}.", "label": label})
        else:
            if task == "extraction":
                prompt = f'Extract order_id and quantity as a JSON object, no extra text. Order ORD-{1000+i} contains {i % 7 + 1} items.'
                answer = json.dumps({"order_id": f"ORD-{1000+i}", "quantity": i % 7 + 1})
            else:
                prompt = f"Summarize in one sentence, preserving the order ID, delay and action: Order ORD-{1000+i} is delayed by {i % 5 + 1} days because of bad weather. The seller will notify the customer when it ships. Do not invent a delivery date."
                answer = f"Order ORD-{1000+i} is delayed {i % 5 + 1} days due to bad weather, and the seller will notify the customer when it ships."
            rows.append({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]})
    return rows


def evaluate(directory, model_path):
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from advisor import evaluation_prompt, task_quality
    from train import training_text_dataset
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    record = json.loads((directory / "run.json").read_text())
    data = Dataset.from_json(record["config"]["dataset"]["name"])
    manifest = json.loads((directory / "split.json").read_text())
    ids = manifest["evalIndices"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    # Verify actual TRL labels, not only our intermediate mask.
    masked = training_text_dataset(data.select([0]), tokenizer, 256)[0]
    labels = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id, completion_only_loss=True)([masked])["labels"][0].tolist()
    assert any(x != -100 for x in labels)
    assert all(label == -100 for label, mask in zip(labels, masked["completion_mask"]) if not mask)
    base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, device_map={"": 0}, torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16))
    model = PeftModel.from_pretrained(base, str(directory / "adapter")).eval()
    outputs = {}
    for phase in ("base", "tuned"):
        samples, predictions, references = [], [], []
        started = time.monotonic()
        with model.disable_adapter() if phase == "base" else nullcontext():
            for i in ids:
                prompt, reference = evaluation_prompt(data[i], tokenizer)
                inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
                with torch.inference_mode():
                    answer = model.generate(**inputs, max_new_tokens=96, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                generated = tokenizer.decode(answer[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                predictions.append(generated)
                references.append(reference)
                samples.append({"rowId": i, "input": prompt, "reference": reference, "output": generated})
        quality = task_quality([data[i] for i in ids], references, predictions, ids)["qualityEvidence"]
        outputs[phase] = {"qualityEvidence": quality, "seconds": time.monotonic() - started, "samples": samples}
    outputs["limitation"] = "Synthetic template-based smoke test; summarization requires human review. No claim of real-world accuracy."
    outputs["trlMaskVerified"] = True
    (directory / "comparison.json").write_text(json.dumps(outputs, indent=2))
    print(json.dumps({phase: outputs[phase]["qualityEvidence"] for phase in ("base", "tuned")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--tasks", nargs="+", choices=["classification", "extraction", "summarization"], default=["classification", "extraction", "summarization"])
    args = parser.parse_args()
    if args.evaluate:
        evaluate(args.output, args.model)
        return
    if not 1 <= args.seconds <= 1800:
        parser.error("--seconds must be between 1 and 1800")
    args.output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.seconds
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "USE_TF": "0", "USE_FLAX": "0", "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"), "WANDB_DISABLED": "true"}
    summary = []
    for task in args.tasks:
        directory = args.output / task
        directory.mkdir()
        data_path = directory / "data.jsonl"
        data_path.write_text("\n".join(json.dumps(row) for row in fixtures(task)), encoding="utf-8")
        ref = {"source": "upload", "name": str(data_path), "extension": "jsonl"}
        manifest = {"trainIndices": list(range(96)), "developmentIndices": list(range(96,126)), "evalIndices": list(range(126,156)), "trainDataset": ref, "evalDataset": ref}
        (directory / "split.json").write_text(json.dumps(manifest))
        config = {"dataset": ref, "base_model": args.model, "parameter_method": "qlora", "split_seed": 42,
            "split_manifest_path": str(directory / "split.json"),
            "hyperparameters": {"rank": 8, "alpha": 16, "learning_rate": 0.0001, "epochs": 1, "batch_size": 1, "gradient_accumulation": 8, "max_sequence_length": 256}}
        (directory / "run.json").write_text(json.dumps({"config": config}))
        item = {"task": task, "trainRows": 96, "developmentRows": 30, "finalRows": 30}
        started = time.monotonic()
        for phase, command in (
            ("train", [sys.executable, str(Path(__file__).with_name("train.py")), "--run-dir", str(directory)]),
            ("evaluate", [sys.executable, str(Path(__file__).resolve()), "--output", str(directory), "--model", args.model, "--evaluate"]),
        ):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                item["status"] = "budget_exhausted"
                break
            print(f"{task}: {phase}", flush=True)
            with (directory / (phase + ".log")).open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                try:
                    code = process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    item["status"] = "budget_exhausted"
                    break
            if code:
                item["status"] = phase + "_failed"
                break
        else:
            item["status"] = "complete"
            comparison = json.loads((directory / "comparison.json").read_text())
            item["scores"] = {p: comparison[p]["qualityEvidence"] for p in ("base", "tuned")}
        item["elapsedSeconds"] = round(time.monotonic() - started, 1)
        summary.append(item)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(item), flush=True)
        if item["status"] == "budget_exhausted":
            break


if __name__ == "__main__":
    main()
