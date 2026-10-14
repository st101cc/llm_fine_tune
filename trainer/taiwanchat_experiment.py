"""TaiwanChat original-vs-reviewed QLoRA comparison on locally cached TAIDE (bnb4, one T4).

prepare   explicit network read of the pinned dataset; samples train/dev/eval and applies
          only the owner-approved terminology list to final assistant answers
train     budgeted QLoRA through train.py for one variant
evaluate  budgeted bnb4 generation on the frozen eval prompts plus the TMMLU+ smoke subset
report    offline comparison: approved mainland-term rate and TMMLU+ accuracy per model
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from advisor import input_key
from quality_models import TAIDE_ID, fingerprint, get_model_status, load_benchmark
from quality_pilot import _runtime_directory, budget_status, generate_bounded, run_with_pilot_budget
from quality_policy import CUSTOMER_TERM_REASON, inspect_text, row_problem

DATASET_ID = "yentinglin/TaiwanChat"
DATASET_REVISION = "d766aa2cf5743aebfe3d91bbf141b3724de21a9b"
MODEL_REVISION = "6738923786667dfe00363b1e2d225536817b429e"
BENCHMARK_REVISION = "466ccd9752ce6889f35c47f875ac7a65012e2269"
EVAL_SUFFIX = "\n請用100字以內回答。"
# Equal-length writing samples: TAIDE ignores the length request often, and term density per
# 1k characters only needs comparable text, not finished answers.
WRITING_SAMPLE_TOKENS = 128
HYPERPARAMETERS = {"rank": 16, "alpha": 32, "learning_rate": 0.0002, "epochs": 2, "batch_size": 1, "gradient_accumulation": 8, "max_sequence_length": 512}
TRAIN = Path(__file__).resolve().parent / "train.py"


def experiment_dir(root):
    return _runtime_directory(root) / "experiments" / "taiwanchat"


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def approved_policy(terms_file):
    # A replacement containing its own source term (算法→演算法) must be protected, or correct text
    # is flagged and rewritten again (演算法→演演算法).
    self_containing = [r for t, r in terms_file["terms"].items() if t in r]
    return {"terms": terms_file["terms"], "exclusions": sorted(set(terms_file.get("exclusions", []) + self_containing))}


def apply_terms(text, policy):
    findings = [f for f in inspect_text(text, policy) if f["reason"] == CUSTOMER_TERM_REASON]
    for f in sorted(findings, key=lambda f: -f["start"]):
        text = text[:f["start"]] + f["replacement"] + text[f["end"]:]
    return text, [f["original"] for f in findings]


def mainland_hits(text, policy):
    return [f["original"] for f in inspect_text(text, policy) if f["reason"] == CUSTOMER_TERM_REASON]


def prepare(args):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    terms_file = json.loads(args.terms.read_text(encoding="utf-8"))
    if not args.dry_run and not (terms_file.get("reviewer", "").strip() and terms_file.get("approvedAt", "").strip()):
        raise SystemExit("The term list needs a reviewer and approvedAt before it can change training data.")
    policy = approved_policy(terms_file)
    status = get_model_status(TAIDE_ID, revision=MODEL_REVISION)
    if status["status"] != "ready":
        raise SystemExit(status["message"])
    tokenizer = AutoTokenizer.from_pretrained(status["path"], local_files_only=True)
    dataset = load_dataset(DATASET_ID, revision=DATASET_REVISION, split="train").shuffle(seed=args.seed)
    # Owner-approved mix: term-bearing rows are a fixed share of training so the variants differ measurably.
    term_quota = round(args.train * args.term_share)
    plain_quota = args.train - term_quota
    seen, train, dev, evaluation = set(), [], [], []
    term_rows = plain_rows = 0
    for index, row in enumerate(dataset):
        if len(train) >= args.train and len(dev) >= args.dev and len(evaluation) >= args.eval:
            break
        messages = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
        # ponytail: single-turn rows only; multi-turn needs per-turn masking and eval prompts.
        if [m["role"] for m in messages] != ["user", "assistant"] or row_problem({"messages": messages}):
            continue
        key = input_key({"messages": messages})
        if key in seen:
            continue
        record = {"source": row["id"], "sourceIndex": index, "messages": messages}
        hits = bool(mainland_hits(messages[-1]["content"], policy))
        fits = lambda: len(tokenizer.apply_chat_template(messages, tokenize=True)) <= args.max_tokens
        wants_train = (hits and term_rows < term_quota) or (not hits and plain_rows < plain_quota)
        if len(dev) < args.dev and fits():
            dev.append(record)  # Random mix, drawn first.
        elif wants_train and fits():
            train.append(record)
            term_rows += hits
            plain_rows += not hits
        elif hits and len(evaluation) < args.eval and len(tokenizer.apply_chat_template(messages[:1], tokenize=True, add_generation_prompt=True)) <= 400:
            # Eval prompts are chosen because their reference answers used mainland terms,
            # so they probe the same vocabulary; every model answers the identical prompts.
            evaluation.append(record)
        else:
            continue
        seen.add(key)
    if len(train) < args.train or len(dev) < args.dev or len(evaluation) < args.eval:
        raise SystemExit(f"Dataset exhausted: train {len(train)}, dev {len(dev)}, eval {len(evaluation)}")

    def reviewed(rows):
        out, counts, edited = [], {}, 0
        for r in rows:
            content, hits = apply_terms(r["messages"][-1]["content"], policy)
            edited += bool(hits)
            for h in hits:
                counts[h] = counts.get(h, 0) + 1
            out.append({**r, "messages": [r["messages"][0], {"role": "assistant", "content": content}]})
        return out, counts, edited

    train_reviewed, term_counts, train_edited = reviewed(train)
    dev_reviewed, _, dev_edited = reviewed(dev)
    summary = {"trainRows": len(train), "trainRowsEdited": train_edited, "devRows": len(dev), "devRowsEdited": dev_edited,
               "evalPrompts": len(evaluation), "editsByTerm": dict(sorted(term_counts.items(), key=lambda kv: -kv[1]))}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    directory = experiment_dir(args.runtime_root)
    if (directory / "manifest.json").exists():
        raise SystemExit(f"{directory} already holds a prepared experiment; inspect or move it before preparing again.")
    directory.mkdir(parents=True)
    files = {"original.jsonl": train, "reviewed.jsonl": train_reviewed, "dev-original.jsonl": dev, "dev-reviewed.jsonl": dev_reviewed,
             "eval-cases.jsonl": [{"id": f"taiwanchat:{r['sourceIndex']}", "task": "writing", "input": r["messages"][0]["content"] + EVAL_SUFFIX, "reference": r["messages"][1]["content"]} for r in evaluation]}
    for name, rows in files.items():
        write_jsonl(directory / name, rows)
    manifest = {"dataset": DATASET_ID, "datasetRevision": DATASET_REVISION, "modelRevision": MODEL_REVISION, "benchmarkRevision": BENCHMARK_REVISION,
                "seed": args.seed, "maxTokens": args.max_tokens, "termShare": args.term_share, "hyperparameters": HYPERPARAMETERS, "evalSuffix": EVAL_SUFFIX,
                "terms": terms_file, "policyHash": fingerprint(policy), "summary": summary,
                "files": {name: sha256(directory / name) for name in files}}
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {directory}")


def load_manifest(root):
    directory = experiment_dir(root)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        if sha256(directory / name) != digest:
            raise SystemExit(f"{name} changed after preparation")
    return directory, manifest


def train(args):
    directory, manifest = load_manifest(args.runtime_root)
    run_dir = directory / "runs" / args.variant
    if (run_dir / "adapter" / "adapter_model.safetensors").exists():
        raise SystemExit(f"{args.variant} adapter already trained at {run_dir / 'adapter'}")
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {"dataset": {"source": "upload", "name": str(directory / f"{args.variant}.jsonl"), "extension": "jsonl"},
              "test_dataset": {"source": "upload", "name": str(directory / f"dev-{args.variant}.jsonl"), "extension": "jsonl"},
              "base_model": TAIDE_ID, "parameter_method": "qlora", "split_seed": manifest["seed"], "hyperparameters": manifest["hyperparameters"]}
    run = {"config": config, "qualityModel": {"revision": manifest["modelRevision"]},
           "qualityProvenance": {"experiment": "taiwanchat", "manifestHash": sha256(directory / "manifest.json"), "variant": args.variant}}
    (run_dir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    result = run_with_pilot_budget(args.runtime_root, [[sys.executable, "-B", str(TRAIN), "--run-dir", str(run_dir)]],
                                   max_seconds=args.seconds, log_dir=run_dir, load_mode="bnb4")
    (run_dir / "budget-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result.get(k) for k in ("status", "message", "chargedGpuSeconds", "budget")}, ensure_ascii=False, indent=2))


def evaluate(args):
    directory, manifest = load_manifest(args.runtime_root)
    out = directory / "eval" / f"{args.model}.json"
    if out.exists():
        raise SystemExit(f"{out} already exists")
    adapter = None if args.model == "base" else directory / "runs" / args.model / "adapter"
    if adapter and not (adapter / "adapter_model.safetensors").is_file():
        raise SystemExit(f"Train the {args.model} adapter first")
    cases = load_benchmark(manifest["benchmarkRevision"], runtime_root=args.runtime_root)["cases"] + read_jsonl(directory / "eval-cases.jsonl")
    answers, over_limit, attempts = [], [], []
    remaining = cases
    while remaining and len(attempts) < 6:
        result = generate_bounded(remaining, runtime_root=args.runtime_root, adapter_path=adapter, revision=manifest["modelRevision"],
                                  load_mode="bnb4", max_seconds=args.seconds, writing_sample_tokens=WRITING_SAMPLE_TOKENS)
        attempts.append({k: result.get(k) for k in ("status", "message", "reason", "budget")})
        answers += result["answers"]
        if result["status"] == "complete":
            remaining = []
        elif "token limit" in (result.get("message") or ""):
            # The strict generator refuses truncated text; record the overrun and resume after it.
            over_limit.append(remaining[len(result["answers"])]["id"])
            remaining = remaining[len(result["answers"]) + 1:]
        else:
            break
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "adapter": str(adapter) if adapter else None, "complete": not remaining,
                               "answers": answers, "overLimit": over_limit, "unanswered": [c["id"] for c in remaining], "attempts": attempts},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"model": args.model, "answered": len(answers), "overLimit": len(over_limit), "unanswered": len(remaining), "budget": budget_status(args.runtime_root)}, ensure_ascii=False, indent=2))


def leading_letter(text):
    text = text.strip()
    return text[:1] if text and text[0] in "ABCD" and (len(text) == 1 or not text[1].isalnum()) else None


def report(args):
    directory, manifest = load_manifest(args.runtime_root)
    policy = approved_policy(manifest["terms"])
    references = {c["id"]: c for c in load_benchmark(manifest["benchmarkRevision"], runtime_root=args.runtime_root)["cases"]}
    results = {}
    for model in ("base", "original", "reviewed"):
        path = directory / "eval" / f"{model}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        mcq = [a for a in data["answers"] if a["caseId"] in references]
        writing = [a for a in data["answers"] if a["caseId"].startswith("taiwanchat:")]
        hits = [mainland_hits(a["output"], policy) for a in writing]
        chars = sum(len(a["output"]) for a in writing)
        results[model] = {
            "tmmluplusAccuracy": sum(leading_letter(a["output"]) == references[a["caseId"]]["reference"] for a in mcq) / len(mcq) if mcq else None,
            "tmmluplusAnswered": len(mcq),
            "writingAnswered": len(writing), "writingTruncatedSamples": sum(bool(a.get("truncated")) for a in writing), "writingChars": chars,
            "answersWithMainlandTerm": sum(bool(h) for h in hits),
            "mainlandTermsPer1kChars": 1000 * sum(map(len, hits)) / chars if chars else None,
            "mainlandTermCounts": dict(sorted({t: sum(h.count(t) for h in hits) for t in {t for h in hits for t in h}}.items(), key=lambda kv: -kv[1])),
            "complete": data["complete"]}
    (directory / "report.json").write_text(json.dumps({"manifestHash": sha256(directory / "manifest.json"), "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runtime-root", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--terms", type=Path, required=True)
    p.add_argument("--train", type=int, default=500)
    p.add_argument("--dev", type=int, default=50)
    p.add_argument("--eval", type=int, default=60)
    p.add_argument("--max-tokens", type=int, default=480)
    p.add_argument("--term-share", type=float, default=0.5, help="Share of training rows whose answers contain approved terms")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    t = sub.add_parser("train")
    t.add_argument("--variant", choices=("original", "reviewed"), required=True)
    t.add_argument("--seconds", type=float, default=1800)
    e = sub.add_parser("evaluate")
    e.add_argument("--model", choices=("base", "original", "reviewed"), required=True)
    e.add_argument("--seconds", type=float, default=1800)
    sub.add_parser("report")
    args = parser.parse_args()
    {"prepare": prepare, "train": train, "evaluate": evaluate, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
