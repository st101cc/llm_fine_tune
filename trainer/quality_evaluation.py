"""Deterministic answer evaluation. Language policy is supplied by quality_policy.

No I/O, inference, answer rewriting, or policy implementation occurs here.
Optional answer protocol/policyHash fields attest the supplied-answer protocol;
without protocol metadata paired comparisons are unavailable.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
from itertools import combinations
import json
import math
import re


VERSION = "quality-evaluation-v1"
TASKS = {"classification", "json", "writing", "mcq"}
MAX_OUTPUT_CHARS = 65536


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def finite(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Non-finite JSON number")
        return number

    def invalid(value):
        raise ValueError("Nonstandard JSON constant: " + value)

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid, parse_float=finite)


def _structural_equal(left, right):
    # bool is a subclass of int in Python; JSON true must not equal JSON 1.
    if type(left) is not type(right):
        return type(left) in (int, float) and type(right) in (int, float) and left == right
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_structural_equal(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_structural_equal(a, b) for a, b in zip(left, right))
    return left == right


def _score(case, answer):
    task, output = case["task"], answer["output"]
    reference = case.get("reference")
    result = {"caseId": case["id"], "task": task, "subject": case.get("subject"), "output": output, "correct": None}
    if task == "writing":
        result["correct"] = {"pass": True, "fail": False}.get(answer.get("judgment"))
    elif task == "json":
        try:
            parsed = _strict_json(output)
            result["validJSON"] = True
        except (ValueError, RecursionError):
            result["validJSON"] = False
        if reference is not None:
            expected = _strict_json(reference) if isinstance(reference, str) else _strict_json(json.dumps(reference, allow_nan=False))
            result["correct"] = result["validJSON"] and _structural_equal(parsed, expected)
    elif reference is not None:
        if task == "mcq":
            result["validMCQ"] = re.fullmatch("[A-D]", output.strip()) is not None
            result["correct"] = result["validMCQ"] and output.strip() == reference
        else:
            result["correct"] = output == reference
    result["judgment"] = "needs_review" if result["correct"] is None else ("pass" if result["correct"] else "fail")
    return result


def _metrics(rows):
    reviewed = [row for row in rows if row["correct"] is not None]
    result = {"answered": len(rows), "scored": len(reviewed), "correct": sum(row["correct"] for row in reviewed), "needsReview": len(rows) - len(reviewed)}
    result["accuracy"] = result["correct"] / len(reviewed) if reviewed else None
    if rows and all(row["task"] == "json" for row in rows):
        result["validJSON"] = sum(row["validJSON"] for row in rows)
        result["validJSONRate"] = result["validJSON"] / len(rows)
        result["structuralExact"] = result["accuracy"]
    return result


def evaluate(cases, answers, policy, *, cancelled=lambda: False) -> dict:
    """Score supplied answers; return explicit coverage, task metrics and provenance.

    Cancellation returns partial results and missing IDs. Empty-answer candidates
    cannot be inferred from this API; requested IDs remain in expectedCaseIds.
    Human judgments affect writing only. Missing references remain unscored.
    Pair deltas use only identically scored IDs; no locality correctness proxy.
    """
    cases, answers, policy = deepcopy(list(cases)), deepcopy(list(answers)), deepcopy(policy)
    from quality_policy import normalize_policy
    policy = normalize_policy(policy)
    indexed = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"]:
            raise ValueError("Cases require nonempty string IDs")
        if case["id"] in indexed:
            raise ValueError("Duplicate case ID: " + case["id"])
        if case.get("task") not in TASKS or not isinstance(case.get("input"), str):
            raise ValueError("Cases require input text and a supported task")
        if case["task"] == "mcq" and case.get("reference") is not None and case["reference"] not in ("A", "B", "C", "D"):
            raise ValueError("MCQ reference must be A-D")
        if case["task"] == "classification" and case.get("reference") is not None and not isinstance(case["reference"], str):
            raise ValueError("Classification reference must be text")
        if case["task"] == "json" and case.get("reference") is not None:
            ref = case["reference"]
            _strict_json(ref if isinstance(ref, str) else json.dumps(ref, allow_nan=False))
        indexed[case["id"]] = case
    grouped = defaultdict(list)
    seen = set()
    for answer in answers:
        if not isinstance(answer, dict) or answer.get("caseId") not in indexed:
            raise ValueError("Unknown answer case ID")
        candidate = answer.get("candidate")
        if not isinstance(candidate, str) or not candidate or not isinstance(answer.get("output"), str):
            raise ValueError("Answers require candidate and output text")
        if len(answer["output"]) > MAX_OUTPUT_CHARS:
            raise ValueError("Evaluation output exceeds 65536 characters; truncated inspection is not permitted")
        key = (candidate, answer["caseId"])
        if key in seen:
            raise ValueError("Duplicate candidate/case ID")
        if "judgment" in answer and answer["judgment"] not in ("pass", "fail", "needs_review"):
            raise ValueError("Invalid judgment")
        seen.add(key)
        grouped[candidate].append(answer)
    ordered_cases = sorted(cases, key=lambda c: c["id"])
    ordered_answers = sorted(answers, key=lambda a: (a["candidate"], a["caseId"]))
    policy_hash = fingerprint(policy)
    provenance = {"evaluatorVersion": VERSION, "casesHash": fingerprint(ordered_cases), "answersHash": fingerprint(ordered_answers), "policyHash": policy_hash, "policyId": policy.get("id"), "policyVersion": policy.get("version"), "openccVersion": policy.get("openccVersion")}
    result = {"status": "complete", "expectedCaseIds": sorted(indexed), "candidates": {}, "comparisons": [], "unpaired": [], "provenance": provenance}
    for candidate in sorted(grouped):
        supplied = sorted(grouped[candidate], key=lambda a: a["caseId"])
        rows = []
        for answer in supplied:
            if cancelled():
                result["status"] = "cancelled"
                break
            # Lazy boundary: primary agent owns the only policy implementation.
            from quality_policy import inspect_text
            row = _score(indexed[answer["caseId"]], answer)
            row["localityFindings"] = inspect_text(answer["output"], policy)
            rows.append(row)
        ids = [row["caseId"] for row in rows]
        protocol_by_case = {a["caseId"]: fingerprint(a["protocol"]) if a.get("protocol") is not None else None for a in supplied}
        protocols = sorted({p for p in protocol_by_case.values() if p is not None})
        policies = sorted({a.get("policyHash") for a in supplied}, key=lambda value: str(value))
        result["candidates"][candidate] = {
            "coverage": {"expected": len(cases), "answered": len(rows), "fraction": len(rows) / len(cases) if cases else None, "caseIds": ids, "missingCaseIds": sorted(set(indexed) - set(ids))},
            "metrics": {task: _metrics([r for r in rows if r["task"] == task]) for task in sorted({c["task"] for c in cases})},
            "subjects": {subject: _metrics([r for r in rows if r["subject"] == subject]) for subject in sorted({r["subject"] for r in rows if r["subject"] is not None})},
            "results": rows, "provenance": {**provenance, "answersHash": fingerprint(supplied), "coverageHash": fingerprint(ids), "protocolHashes": protocols, "protocolByCase": protocol_by_case, "sourcePolicyHashes": policies},
        }
    for left, right in combinations(result["candidates"], 2):
        a, b = result["candidates"][left], result["candidates"][right]
        reason = None
        if result["status"] == "cancelled":
            reason = "evaluation cancelled"
        elif not a["coverage"]["caseIds"] or a["coverage"]["caseIds"] != b["coverage"]["caseIds"]:
            reason = "coverage differs"
        elif None in a["provenance"]["protocolByCase"].values() or None in b["provenance"]["protocolByCase"].values():
            reason = "protocol missing"
        elif a["provenance"]["protocolByCase"] != b["provenance"]["protocolByCase"]:
            reason = "protocol differs"
        elif None in a["provenance"]["sourcePolicyHashes"] or None in b["provenance"]["sourcePolicyHashes"]:
            reason = "policy provenance missing"
        elif a["provenance"]["sourcePolicyHashes"] != [policy_hash] or b["provenance"]["sourcePolicyHashes"] != [policy_hash]:
            reason = "policy differs"
        elif [r["caseId"] for r in a["results"] if r["correct"] is not None] != [r["caseId"] for r in b["results"] if r["correct"] is not None]:
            reason = "judgment coverage differs"
        if reason:
            result["unpaired"].append({"left": left, "right": right, "reason": reason})
            continue
        pairs = [(x, y) for x, y in zip(a["results"], b["results"]) if x["correct"] is not None]
        result["comparisons"].append({"left": left, "right": right, "pairedCaseIds": a["coverage"]["caseIds"], "scoredPairs": len(pairs), "rightWins": sum(not x["correct"] and y["correct"] for x, y in pairs), "leftWins": sum(x["correct"] and not y["correct"] for x, y in pairs), "rightMinusLeftAccuracy": sum(int(y["correct"]) - int(x["correct"]) for x, y in pairs) / len(pairs) if pairs else None, "provenance": provenance, "scope": "Supplied answers only; generation equivalence is caller-attested."})
    return result
