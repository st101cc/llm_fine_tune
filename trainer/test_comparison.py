import json

import pytest

import comparison
from comparison import (answer_text, build_report, estimate, load_pair, plan_split, policy_hits, scoring_policy,
                        single_prompt, write_files)
from quality_store import QualityStore


def tokens(row):
    if "prompt_only" in row:
        return len(row["prompt_only"])
    return len(json.dumps(row, ensure_ascii=False))


def chat(i, answer, long=0):
    return {"messages": [{"role": "user", "content": f"問題{i}" + "字" * long}, {"role": "assistant", "content": answer}]}


def pair(edited, plain):
    original = [chat(i, "舊") for i in range(edited + plain)]
    reviewed = [chat(i, "新" if i < edited else "舊") for i in range(edited + plain)]
    return original, reviewed


def test_split_is_deterministic_and_disjoint():
    original, reviewed = pair(100, 400)
    original += [original[0]]
    reviewed += [reviewed[0]]
    a = plan_split(original, reviewed, tokens)
    assert a == plan_split(original, reviewed, tokens)
    sets = [set(a["eval"]), set(a["dev"]), set(a["train"])]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    keys = [{comparison.input_key(original[i]) for i in s} for s in sets]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])


def test_variants_share_indices_and_differ_only_in_answers(tmp_path):
    original, reviewed = pair(100, 300)
    split = plan_split(original, reviewed, tokens)
    hashes = write_files(tmp_path, original, reviewed, split)
    assert set(hashes) == {"original-train.jsonl", "reviewed-train.jsonl", "dev-original.jsonl", "dev-reviewed.jsonl", "eval-cases.jsonl", "split.json"}
    load = lambda name: [json.loads(line) for line in (tmp_path / name).read_text(encoding="utf-8").splitlines()]
    o, r = load("original-train.jsonl"), load("reviewed-train.jsonl")
    assert [x["messages"][0] for x in o] == [x["messages"][0] for x in r]
    assert [a["messages"][1] != b["messages"][1] for a, b in zip(o, r)] == [i < 100 for i in split["train"]]
    case = load("eval-cases.jsonl")[0]
    assert case == {"id": f"row:{split['eval'][0]}", "task": "writing", "input": original[split["eval"][0]]["messages"][0]["content"], "reference": "新"}


def test_edited_share_is_honoured():
    original, reviewed = pair(300, 700)
    split = plan_split(original, reviewed, tokens, train_rows=200, edited_share=0.5)
    assert len(split["train"]) == 200 and split["counts"]["trainEdited"] == 100


def test_eval_rows_are_edited_single_turn_and_never_trained():
    original, reviewed = pair(100, 300)
    split = plan_split(original, reviewed, tokens)
    assert split["eval"] and all(i < 100 and single_prompt(original[i]) for i in split["eval"])
    assert not set(split["eval"]) & (set(split["train"]) | set(split["dev"]))


def test_minimums_and_reasons():
    original, reviewed = pair(8, 400)
    result = estimate(plan_split(original, reviewed, tokens), 100, 2, 15000)
    assert not result["feasible"] and any("8" in r for r in result["reasons"])
    multi = [{"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": "a"}, {"role": "user", "content": "b"}, {"role": "assistant", "content": "舊"}]} for i in range(300)]
    edited = [{"messages": r["messages"][:-1] + [{"role": "assistant", "content": "新"}]} for r in multi]
    assert "no edited row has a single-turn prompt to evaluate" in plan_split(multi, edited, tokens)["reasons"]


def test_prompt_completion_rows_split_like_messages(tmp_path):
    original = [{"prompt": f"p{i}", "completion": "舊"} for i in range(300)]
    reviewed = [{"prompt": f"p{i}", "completion": "新" if i < 100 else "舊"} for i in range(300)]
    split = plan_split(original, reviewed, tokens)
    write_files(tmp_path, original, reviewed, split)
    cases = [json.loads(line) for line in (tmp_path / "eval-cases.jsonl").read_text(encoding="utf-8").splitlines()]
    assert cases and all(c["input"] == original[int(c["id"][4:])]["prompt"] for c in cases)
    assert answer_text(original[0]) == "舊" and single_prompt(original[0]) == "p0"


def test_plain_text_rows_are_refused_with_reason():
    rows = [{"text": f"t{i}"} for i in range(300)]
    assert plan_split(rows, rows, tokens)["reasons"] == ["plain text rows have no prompt to evaluate; comparison needs messages or prompt/completion rows"]


def test_long_rows_skipped_not_truncated():
    original = [chat(i, "舊", long=2000 if i % 5 == 0 else 0) for i in range(500)]
    reviewed = [chat(i, "新" if i < 200 else "舊", long=2000 if i % 5 == 0 else 0) for i in range(500)]
    split = plan_split(original, reviewed, lambda row: 9999 if "prompt_only" not in row and len(json.dumps(row)) > 2000 else 100)
    long_rows = {i for i in range(500) if i % 5 == 0}
    assert not long_rows & (set(split["train"]) | set(split["dev"]))
    assert split["counts"]["skippedLong"] > 0


def test_estimate_refuses_per_run_cap_and_ledger():
    original, reviewed = pair(100, 400)
    split = plan_split(original, reviewed, tokens)
    heavy = estimate(split, 480, 3, 15000)
    assert not heavy["feasible"] and any("1800" in r for r in heavy["reasons"])
    light = estimate(split, 100, 1, 15000)
    assert light["feasible"] and light["stepMaxSeconds"]["train"] <= 1800
    tight = estimate(split, 100, 1, light["totalGpuSeconds"] - 1)
    assert not tight["feasible"] and any("ledger" in r for r in tight["reasons"])


def test_policy_hits_separates_terms_and_opencc():
    assert policy_hits("數據與软件", {"terms": {"數據": "資料"}}) == {"terms": ["數據"], "opencc": ["软件"]}


def test_scoring_policy_protects_self_containing_replacements():
    policy = {"算法": "演算法"}
    assert policy_hits("演算法", {"terms": policy})["terms"] == []
    assert scoring_policy({"terms": policy})["exclusions"] == ["演算法"]


def test_load_pair_rejects_missing_or_changed_source(tmp_path):
    rows = [{"prompt": "a", "completion": "這是數據"}, {"prompt": "b", "completion": "好"}]
    directory = tmp_path / "datasets"
    directory.mkdir()
    path = directory / "fixture.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    store = QualityStore(tmp_path)
    audit = store.create_audit({"source": "upload", "name": str(path), "extension": "jsonl"}, {"terms": {"數據": "資料"}}, "full")
    store.run_audit(audit["id"])
    audit = store.get("audit", audit["id"])
    store.accept_customer_terms(audit["id"], audit["fingerprint"])
    version = store.materialize(audit["id"], audit["fingerprint"])
    original, reviewed, _, _ = load_pair(store, version["id"])
    assert [r["completion"] for r in original] == ["這是數據", "好"] and [r["completion"] for r in reviewed] == ["這是資料", "好"]
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="start a new audit"):
        load_pair(store, version["id"])
    path.unlink()
    with pytest.raises(ValueError, match="Source dataset is missing; start a new audit."):
        load_pair(store, version["id"])


def test_report_counts_and_paired():
    cases = [{"id": f"row:{i}", "input": f"q{i}"} for i in range(4)]
    mcq = [{"id": "m1", "reference": "A"}, {"id": "m2", "reference": "B"}]
    policy = {"terms": {"數據": "資料"}}
    out = lambda case, text: {"caseId": case, "output": text}
    mcq_answers = [out("m1", "A"), out("m2", "C")]
    answers = {
        "base": [out("row:0", "數據" + "字" * 98), out("row:1", "好" * 100), out("row:2", "好" * 100), out("row:3", "好" * 100)] + mcq_answers,
        "original": [out("row:0", "數據" + "字" * 98), out("row:1", "數據" + "字" * 98), out("row:2", "好" * 100), out("row:3", "數據" + "字" * 98)] + mcq_answers,
        "reviewed": [out("row:0", "數據" + "字" * 98), out("row:1", "好" * 100), out("row:2", "數據" + "字" * 98), out("row:3", "好" * 100)] + mcq_answers,
    }
    report = build_report(answers, cases, mcq, policy, {"original": 1.5, "reviewed": None})
    assert report["models"]["original"]["answersWithTerm"] == 3
    assert report["models"]["original"]["termsPer1k"] == pytest.approx(1000 * 3 / 400)
    assert report["models"]["original"]["tmmluplusAccuracy"] == 0.5 and report["models"]["original"]["trainLoss"] == 1.5
    assert report["paired"] == {"originalOnly": 2, "reviewedOnly": 1, "both": 1, "neither": 0}
    assert "7.5" in report["headline"] and "5.0" in report["headline"]
    assert {e["caseId"] for e in report["examples"]} == {"row:1", "row:2", "row:3"}
