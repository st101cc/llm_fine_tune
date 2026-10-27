"""Split, cost estimate, files, metric and report for original-vs-reviewed comparisons."""
import hashlib
import json
import math
import random
from pathlib import Path

from advisor import input_key
from quality_policy import CUSTOMER_TERM_REASON, inspect_text
from quality_store import fingerprint, iter_rows

MIN_TRAIN = 100
MIN_EVAL = 20
DEV_ROWS = 50
MAX_ROW_TOKENS = 480
MAX_PROMPT_TOKENS = 400
DEFAULT_TRAIN_CAP = 500
TRAIN_LOAD_SECONDS = 90
SECONDS_PER_TOKEN_EPOCH = 0.0065
EVAL_LOAD_SECONDS = 40
MCQ_SECONDS = 1.5
WRITING_SECONDS = 8
ADAPTER_WRITING_SECONDS = 13
RESERVATION_FACTOR = 1.5
PER_RUN_CAP = 1800
MCQ_CASES = 30
PLAIN_TEXT_REASON = "plain text rows have no prompt to evaluate; comparison needs messages or prompt/completion rows"
STALE_HINT = "start a new audit"


def load_pair(store, version_id):
    version = store.get("version", version_id)
    if version.get("status") != "complete":
        raise ValueError("Version publication is not complete.")
    audit = store.get("audit", version["provenance"]["auditId"])
    source = Path(audit["dataset"].get("name", ""))
    if not source.is_file():
        raise ValueError(f"Source dataset is missing; {STALE_HINT}.")
    source = store.source_path(audit["dataset"])
    if fingerprint(source) != version["provenance"]["sourceFingerprint"]:
        raise ValueError(f"Source dataset changed since the review; {STALE_HINT}.")
    try:
        reviewed_path = store.export_path(version_id)
    except ValueError as error:
        raise ValueError(f"Reviewed version file is missing or changed; {STALE_HINT}.") from error
    if fingerprint(reviewed_path) != version["provenance"]["outputFingerprint"]:
        raise ValueError(f"Reviewed version file changed; {STALE_HINT}.")
    original, reviewed = list(iter_rows(source)), list(iter_rows(reviewed_path))
    if len(original) != len(reviewed):
        raise ValueError("Original and reviewed row counts differ.")
    return original, reviewed, audit, version


def _messages(row):
    messages = row.get("messages") if isinstance(row, dict) else None
    return messages if isinstance(messages, list) and all(isinstance(m, dict) for m in messages) else None


def answer_text(row):
    messages = _messages(row)
    if messages is not None:
        last = messages[-1] if messages else {}
        return last.get("content") if last.get("role") == "assistant" and isinstance(last.get("content"), str) else None
    completion = row.get("completion") if isinstance(row, dict) else None
    return completion if isinstance(completion, str) else None


def single_prompt(row):
    messages = _messages(row)
    if messages is not None:
        head = messages[:-1]
        if len(head) == 1 and head[0].get("role") == "user" and isinstance(head[0].get("content"), str):
            return head[0]["content"]
        return None
    prompt = row.get("prompt") if isinstance(row, dict) else None
    return prompt if isinstance(prompt, str) and "completion" in row else None


def plan_split(original, reviewed, token_count, *, train_rows=None, edited_share=0.5, eval_prompts=60, seed=42):
    n = len(original)
    answers = [(answer_text(original[i]), answer_text(reviewed[i])) for i in range(n)]
    edited = [a is not None and b is not None and a != b for a, b in answers]
    order = list(range(n))
    random.Random(seed).shuffle(order)
    position = {index: p for p, index in enumerate(order)}
    groups = {}
    for index in order:
        groups.setdefault(input_key(original[index]), []).append(index)

    eval_rows, dev_rows, eligible, rest = [], [], 0, []
    for rows in groups.values():
        first = rows[0]
        prompt = single_prompt(original[first])
        if edited[first] and prompt is not None and token_count({"prompt_only": prompt}) <= MAX_PROMPT_TOKENS:
            eligible += 1
            if len(eval_rows) < eval_prompts:
                eval_rows.append(first)
                continue
        rest.append(rows)
    remaining = []
    for rows in rest:
        if len(dev_rows) < DEV_ROWS and token_count(original[rows[0]]) <= MAX_ROW_TOKENS:
            dev_rows.append(rows[0])
        else:
            remaining.append(rows)

    pool = [i for rows in remaining for i in rows]
    fit = [i for i in pool if token_count(original[i]) <= MAX_ROW_TOKENS]
    edited_pool = [i for i in fit if edited[i]]
    plain_pool = [i for i in fit if not edited[i]]
    cap = min(train_rows or min(len(fit), DEFAULT_TRAIN_CAP), len(fit))
    take_edited = min(round(cap * edited_share), len(edited_pool))
    take_plain = min(cap - take_edited, len(plain_pool))
    take_edited = min(cap - take_plain, len(edited_pool))
    train = sorted(edited_pool[:take_edited] + plain_pool[:take_plain], key=position.get)
    mean_tokens = sum(token_count(original[i]) for i in train) / len(train) if train else 0

    edited_total = sum(edited)
    reasons = []
    if all(a is None for a, _ in answers):
        reasons = [PLAIN_TEXT_REASON]
    else:
        if len(train) < MIN_TRAIN:
            reasons.append(f"only {len(train)} usable training rows; at least {MIN_TRAIN} are needed")
        if len(eval_rows) < MIN_EVAL:
            reasons.append(f"only {len(eval_rows)} evaluation prompts; at least {MIN_EVAL} are needed")
        if edited_total == 0:
            reasons.append("the review changed no answers")
        elif not any(edited[i] and single_prompt(original[i]) is not None for i in range(n)):
            reasons.append("no edited row has a single-turn prompt to evaluate")
    return {
        "eval": eval_rows, "dev": dev_rows, "train": train,
        "counts": {"rows": n, "edited": edited_total, "eligibleEval": eligible, "skippedLong": len(pool) - len(fit),
                   "trainEdited": sum(edited[i] for i in train), "meanTrainTokens": mean_tokens},
        "reasons": reasons,
        "settings": {"seed": seed, "trainRows": cap, "editedShare": edited_share, "evalPrompts": eval_prompts,
                     "devRows": DEV_ROWS, "maxRowTokens": MAX_ROW_TOKENS, "maxPromptTokens": MAX_PROMPT_TOKENS},
    }


def estimate(split, mean_train_tokens, epochs, ledger_remaining):
    train = TRAIN_LOAD_SECONDS + SECONDS_PER_TOKEN_EPOCH * mean_train_tokens * len(split["train"]) * epochs
    fixed = EVAL_LOAD_SECONDS + MCQ_SECONDS * MCQ_CASES
    base = fixed + WRITING_SECONDS * len(split["eval"])
    adapter = fixed + ADAPTER_WRITING_SECONDS * len(split["eval"])
    uncapped = {"evaluate-base": math.ceil(base * RESERVATION_FACTOR), "evaluate-adapter": math.ceil(adapter * RESERVATION_FACTOR)}
    reserve = {"train": min(PER_RUN_CAP, math.ceil(train * RESERVATION_FACTOR)), **{k: min(PER_RUN_CAP, v) for k, v in uncapped.items()}}
    reasons = list(split["reasons"])
    if train > PER_RUN_CAP:
        reasons.append(f"one training run would take about {math.ceil(train)} s, over the {PER_RUN_CAP} s per-run cap; use fewer rows or epochs")
    for kind, label in (("evaluate-base", "base"), ("evaluate-adapter", "adapter")):
        if uncapped[kind] > PER_RUN_CAP:
            reasons.append(f"the {label} evaluation step would reserve {uncapped[kind]} s, over the {PER_RUN_CAP} s per-run cap; use fewer evaluation prompts")
    needed = 2 * reserve["train"] + reserve["evaluate-base"] + 2 * reserve["evaluate-adapter"]
    if needed > ledger_remaining:
        reasons.append(f"GPU ledger has {math.ceil(ledger_remaining)} s left; this comparison reserves {needed} s")
    return {"trainSecondsPerRun": math.ceil(train), "evalSecondsBase": math.ceil(base), "evalSecondsAdapter": math.ceil(adapter),
            "totalGpuSeconds": math.ceil(2 * train + base + 2 * adapter), "wallClockSeconds": math.ceil(train + 2 * adapter),
            "stepMaxSeconds": reserve, "feasible": not reasons, "reasons": reasons,
            "ledgerRemaining": ledger_remaining, "perRunCap": PER_RUN_CAP}


def write_files(directory, original, reviewed, split):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pick = lambda rows, indexes: [rows[i] for i in indexes]
    files = {
        "original-train.jsonl": pick(original, split["train"]),
        "reviewed-train.jsonl": pick(reviewed, split["train"]),
        "dev-original.jsonl": pick(original, split["dev"]),
        "dev-reviewed.jsonl": pick(reviewed, split["dev"]),
        "eval-cases.jsonl": [{"id": f"row:{i}", "task": "writing", "input": single_prompt(original[i]), "reference": answer_text(reviewed[i])}
                             for i in split["eval"]],
    }
    contents = {name: "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows) for name, rows in files.items()}
    contents["split.json"] = json.dumps(split, ensure_ascii=False, indent=2)
    for name, text in contents.items():
        (directory / name).write_bytes(text.encode("utf-8"))
    return {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in contents.items()}


def scoring_policy(policy):
    terms = policy.get("terms") or {}
    # A replacement containing its own source term (算法→演算法) must be protected, or correct text
    # is flagged and rewritten again (演算法→演演算法).
    self_containing = [r for t, r in terms.items() if t in r]
    return {**policy, "exclusions": sorted(set(policy.get("exclusions", []) + self_containing))}


def policy_hits(text, policy):
    findings = [f for f in inspect_text(text, scoring_policy(policy)) if f.get("editable")]
    return {"terms": [f["original"] for f in findings if f["reason"] == CUSTOMER_TERM_REASON],
            "opencc": [f["original"] for f in findings if f["reason"] != CUSTOMER_TERM_REASON]}


def leading_letter(text):
    text = text.strip()
    return text[:1] if text and text[0] in "ABCD" and (len(text) == 1 or not text[1].isalnum()) else None


def build_report(answers_by_model, eval_cases, mcq_cases, policy, losses):
    prompts = {c["id"]: c["input"] for c in eval_cases}
    references = {c["id"]: c["reference"] for c in mcq_cases}
    models, hits = {}, {}
    for name, answers in answers_by_model.items():
        writing = [a for a in answers if a["caseId"] in prompts]
        mcq = [a for a in answers if a["caseId"] in references]
        found = {a["caseId"]: policy_hits(a["output"], policy) for a in writing}
        hits[name] = found
        chars = sum(len(a["output"]) for a in writing)
        per_1k = lambda key: 1000 * sum(len(h[key]) for h in found.values()) / chars if chars else None
        models[name] = {
            "termsPer1k": per_1k("terms"), "openccPer1k": per_1k("opencc"),
            "answersWithTerm": sum(bool(h["terms"]) for h in found.values()),
            "answersWithOpencc": sum(bool(h["opencc"]) for h in found.values()),
            "writingAnswered": len(writing), "truncated": sum(bool(a.get("truncated")) for a in writing),
            "tmmluplusAccuracy": sum(leading_letter(a["output"]) == references[a["caseId"]] for a in mcq) / len(mcq) if mcq else None,
            "tmmluplusAnswered": len(mcq), "trainLoss": losses.get(name)}

    paired = {"originalOnly": 0, "reviewedOnly": 0, "both": 0, "neither": 0}
    examples = []
    if "original" in hits and "reviewed" in hits:
        for case_id in prompts:
            if case_id not in hits["original"] or case_id not in hits["reviewed"]:
                continue
            o, r = bool(hits["original"][case_id]["terms"]), bool(hits["reviewed"][case_id]["terms"])
            paired["both" if o and r else "originalOnly" if o else "reviewedOnly" if r else "neither"] += 1
            if o != r and len(examples) < 5:
                present = [m for m in hits if case_id in hits[m]]
                examples.append({"caseId": case_id, "prompt": prompts[case_id],
                                 "answers": {m: next(a["output"] for a in answers_by_model[m] if a["caseId"] == case_id) for m in present},
                                 "hits": {m: hits[m][case_id]["terms"] for m in present}})
    rate = lambda m: "n/a" if models.get(m, {}).get("termsPer1k") is None else f"{models[m]['termsPer1k']:.1f}"
    headline = f"Listed terms per 1,000 characters: original-trained {rate('original')}, reviewed-trained {rate('reviewed')}"
    if "base" in models:
        headline += f" (base {rate('base')})"
    return {"models": models, "paired": paired, "examples": examples, "headline": headline + "."}
