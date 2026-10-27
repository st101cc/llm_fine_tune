import itertools
import json
import math
import uuid

import pytest

import judge_panel as jp
from quality_store import QualityStore

READY = {"breeze": True, "qwen": True, "yi": True}
LABELS = ["pass", "fail", "abstain", "error"]


def vote(correctness, language="pass"):
    if correctness == "error":
        return {"error": "invalid_output"}
    return {"correctness": {"verdict": correctness, "reason": "r"}, "language": {"verdict": language, "reason": "r"}}


def expected_correctness(labels):
    passes, fails = labels.count("pass"), labels.count("fail")
    return "pass" if passes >= 2 else "fail" if fails >= 2 else "needs_review"


@pytest.mark.parametrize("labels", list(itertools.product(LABELS, repeat=3)))
def test_aggregate_correctness_truth_table(labels):
    result = jp.aggregate({name: vote(label) for name, label in zip(("breeze", "qwen", "yi"), labels)})
    assert result["correctness"] == expected_correctness(list(labels))
    assert result["flagged"] == (len(set(labels)) > 1)


def test_language_is_breezes_verdict():
    result = jp.aggregate({"breeze": vote("pass", "fail"), "qwen": vote("pass", "pass"), "yi": vote("pass", "pass")})
    assert result["language"] == "fail" and result["flagged"]
    result = jp.aggregate({"breeze": vote("pass", "abstain"), "qwen": vote("pass"), "yi": vote("pass")})
    assert result["language"] == "needs_review"
    result = jp.aggregate({"breeze": {"error": "boom"}, "qwen": vote("pass"), "yi": vote("pass")})
    assert result["language"] == "needs_review"
    result = jp.aggregate({"breeze": vote("pass", "pass"), "qwen": vote("pass", "fail"), "yi": vote("pass", "fail")})
    assert result["language"] == "pass" and result["flagged"]
    result = jp.aggregate({"breeze": vote("pass", "pass"), "qwen": vote("pass", "fail"), "yi": vote("pass", "pass")})
    assert result["language"] == "pass" and not result["flagged"]


def test_flagged_when_correctness_not_unanimous():
    unanimous = {n: vote("pass") for n in ("breeze", "qwen", "yi")}
    assert not jp.aggregate(unanimous)["flagged"]
    assert jp.aggregate({**unanimous, "yi": vote("fail")})["flagged"]
    assert jp.aggregate(unanimous)["votes"] == unanimous


def test_parse_panel_grade_accepts_fenced_json_and_caps_reason():
    raw = json.dumps({"correctness": {"verdict": "pass", "reason": "字" * 700}, "language": {"verdict": "fail", "reason": " 簡體 "}}, ensure_ascii=False)
    grade = jp.parse_panel_grade(f"```json\n{raw}\n```")
    assert len(grade["correctness"]["reason"]) == 600
    assert grade["language"] == {"verdict": "fail", "reason": "簡體"}


@pytest.mark.parametrize("raw", [
    "這個回答很好",
    json.dumps({"correctness": {"verdict": "pass", "reason": "x"}}),
    json.dumps({"correctness": {"verdict": "maybe", "reason": "x"}, "language": {"verdict": "pass", "reason": "x"}}),
    json.dumps({"correctness": {"verdict": "pass", "reason": ""}, "language": {"verdict": "pass", "reason": "x"}}),
    json.dumps({"verdict": "pass", "reason": "x"}),
    "[]",
])
def test_parse_panel_grade_rejects_prose_and_partial(raw):
    with pytest.raises(ValueError):
        jp.parse_panel_grade(raw)


def test_payload_hides_candidate_and_reference_for_comparison():
    item = {"itemId": "c|base", "caseId": "c", "candidate": "base", "question": "寫信", "reference": "REFTEXT", "answer": "你好", "truncated": True}
    none = json.loads(jp.payload(item, "none"))
    assert "candidate" not in none and "base" not in json.dumps(none) and "reference" not in none and "REFTEXT" not in json.dumps(none)
    assert none["truncated"] is True and none["question"] == "寫信" and none["answer"] == "你好"
    assert "visible text" in none["instruction"]
    case = json.loads(jp.payload({**item, "truncated": False}, "case"))
    assert case["reference"] == "REFTEXT" and case["truncated"] is False and "candidate" not in case
    assert "visible text" not in case["instruction"]


def test_rubrics_follow_reference_mode():
    assert "reference" in jp.CORRECTNESS_RUBRIC("case")
    assert "reference" not in jp.CORRECTNESS_RUBRIC("none")
    assert "Taiwan" in jp.LANGUAGE_RUBRIC


def new_id():
    return str(uuid.uuid4())


def make_comparison(store, status="complete", cases=3, with_eval=True):
    cid = new_id()
    store.put("comparison", {"id": cid, "status": status})
    folder = store.directory / "comparisons" / cid
    folder.mkdir(parents=True)
    rows = [{"id": f"row:{i}", "task": "writing", "input": f"問題{i}", "reference": f"參考{i}"} for i in range(cases)]
    (folder / "eval-cases.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    if with_eval:
        (folder / "eval").mkdir()
        for model in ("base", "original", "reviewed"):
            answers = [{"caseId": r["id"], "output": f"{model}答{r['id']}", **({"truncated": True} if model == "base" else {})} for r in rows]
            answers.append({"caseId": "tmmluplus:1", "output": "A"})
            (folder / "eval" / f"{model}.json").write_text(json.dumps({"answers": answers}, ensure_ascii=False), encoding="utf-8")
    return cid


def test_comparison_source_drops_reference_and_keeps_truncated(tmp_path):
    store = QualityStore(tmp_path)
    cid = make_comparison(store)
    source = jp.load_source(store, {"comparisonId": cid})
    assert source["kind"] == "comparison" and source["id"] == cid and source["referenceMode"] == "none"
    assert source["cases"] == [{"id": f"row:{i}", "input": f"問題{i}", "reference": None} for i in range(3)]
    assert len(source["answers"]) == 9
    assert {a["candidate"] for a in source["answers"]} == {"base", "original", "reviewed"}
    assert all(a["caseId"].startswith("row:") and a["human"] is None for a in source["answers"])
    assert {a["candidate"]: a["truncated"] for a in source["answers"]} == {"base": True, "original": False, "reviewed": False}


@pytest.mark.parametrize("status,with_eval", [("running", True), ("complete", False)])
def test_incomplete_comparison_is_refused(tmp_path, status, with_eval):
    store = QualityStore(tmp_path)
    cid = make_comparison(store, status=status, with_eval=with_eval)
    with pytest.raises(ValueError, match="comparison has no saved evaluation answers"):
        jp.load_source(store, {"comparisonId": cid})
    with pytest.raises(ValueError, match="comparison has no saved evaluation answers"):
        jp.load_source(store, {"comparisonId": new_id()})


def make_evaluation(store, writing=True, generated_truncated=None):
    eid = new_id()
    cases = [{"id": "w1", "task": "writing", "input": "寫一封信", "reference": "範例"}, {"id": "w2", "task": "writing", "input": "寫詩"},
             {"id": "m1", "task": "mcq", "input": "Q", "reference": "A"}]
    rows = lambda: [{"caseId": "w1", "task": "writing", "output": "信", "correct": True, "judgment": "pass"},
                    {"caseId": "w2", "task": "writing", "output": "詩", "correct": None, "judgment": "needs_review"},
                    {"caseId": "m1", "task": "mcq", "output": "A", "correct": True, "judgment": "pass"}]
    record = {"id": eid, "status": "complete", "cases": cases if writing else cases[2:],
              "result": {"candidates": {"mine": {"results": rows() if writing else rows()[2:]},
                                        "Qwen2.5-7B@base": {"results": [{**r, "output": r["output"] + "!", "correct": False if r["correct"] else r["correct"]} for r in (rows() if writing else rows()[2:])]}}}}
    if generated_truncated is not None:
        record["generation"] = {"answers": [{"caseId": "w1", "candidate": "mine", "truncated": generated_truncated}]}
    store.put("evaluation", record)
    return eid


def test_evaluation_source_uses_writing_only_and_defaults_truncated_false(tmp_path):
    store = QualityStore(tmp_path)
    source = jp.load_source(store, {"evaluationId": make_evaluation(store)})
    assert source["kind"] == "evaluation" and source["referenceMode"] == "case"
    assert source["cases"] == [{"id": "w1", "input": "寫一封信", "reference": "範例"}, {"id": "w2", "input": "寫詩", "reference": None}]
    assert sorted((a["caseId"], a["candidate"]) for a in source["answers"]) == [("w1", "Qwen2.5-7B@base"), ("w1", "mine"), ("w2", "Qwen2.5-7B@base"), ("w2", "mine")]
    assert not any(a["truncated"] for a in source["answers"])
    human = {(a["caseId"], a["candidate"]): a["human"] for a in source["answers"]}
    assert human[("w1", "mine")] is True and human[("w2", "mine")] is None and human[("w1", "Qwen2.5-7B@base")] is False
    flagged = jp.load_source(store, {"evaluationId": make_evaluation(store, generated_truncated=True)})
    assert {(a["caseId"], a["candidate"]): a["truncated"] for a in flagged["answers"]}[("w1", "mine")] is True
    assert {(a["caseId"], a["candidate"]): a["truncated"] for a in flagged["answers"]}[("w1", "Qwen2.5-7B@base")] is False


def test_source_without_writing_answers_is_refused(tmp_path):
    store = QualityStore(tmp_path)
    with pytest.raises(ValueError, match="this source has no writing answers to grade"):
        jp.load_source(store, {"evaluationId": make_evaluation(store, writing=False)})
    with pytest.raises(ValueError):
        jp.load_source(store, {"evaluationId": new_id()})
    with pytest.raises(ValueError):
        jp.load_source(store, {})


def synthetic_source(cases=60, candidates=("base", "original", "reviewed")):
    return {"kind": "comparison", "id": "x", "referenceMode": "none",
            "cases": [{"id": f"row:{i}", "input": f"q{i}", "reference": None} for i in range(cases)],
            "answers": [{"caseId": f"row:{i}", "candidate": c, "output": f"{c}{i}", "truncated": c == "base", "human": None}
                        for i in range(cases) for c in candidates]}


def test_sample_items_is_deterministic_and_paired():
    source = synthetic_source()
    items = jp.sample_items(source, 20)
    assert len(items) == 60 and len({i["caseId"] for i in items}) == 20
    assert items == jp.sample_items(source, 20) and items == jp.sample_items(source, None)
    for case_id in {i["caseId"] for i in items}:
        assert sorted(i["candidate"] for i in items if i["caseId"] == case_id) == ["base", "original", "reviewed"]
    first = items[0]
    assert first["itemId"] == f"{first['caseId']}|{first['candidate']}" and first["question"] == "q" + first["caseId"].split(":")[1]
    assert {i["truncated"] for i in items if i["candidate"] == "base"} == {True}
    assert len(jp.sample_items(source, 500)) == 180 and len(jp.sample_items(synthetic_source(5), None)) == 15
    orders = {tuple(i["candidate"] for i in items[n:n + 3]) for n in range(0, 60, 3)}
    assert len(orders) > 1
    with pytest.raises(ValueError):
        jp.sample_items(source, 0)


def test_estimate_default_comparison():
    result = jp.estimate(60, 4274, READY)
    assert result["feasible"] and result["reasons"] == []
    assert result["chunksPerJudge"] == 1 and result["gradesPerJudge"] == 60 and result["ledgerRemaining"] == 4274
    assert result["gpuSeconds"] == 3 * (60 * 12 + 90)


def test_estimate_refuses_missing_judge_and_short_ledger():
    result = jp.estimate(60, 100, {**READY, "qwen": False})
    assert not result["feasible"]
    assert "qwen: missing_weights" in result["reasons"]
    assert any(r.startswith("GPU ledger has 100 s left; this panel needs") for r in result["reasons"])
    assert "yi: missing_weights" in jp.estimate(60, 99999, {"breeze": True, "qwen": True})["reasons"]


def test_estimate_chunks_large_selection():
    result = jp.estimate(180, 10 ** 6, READY)
    chunks = math.ceil(180 / jp.chunk_size())
    assert jp.chunk_size() == (jp.MAX_RUN_SECONDS - jp.LOAD_SECONDS) // jp.SECONDS_PER_GRADE
    assert result["chunksPerJudge"] == chunks == 2
    assert result["gpuSeconds"] == 3 * (180 * jp.SECONDS_PER_GRADE + chunks * jp.LOAD_SECONDS)
    assert jp.LOAD_SECONDS + jp.chunk_size() * jp.SECONDS_PER_GRADE <= jp.MAX_RUN_SECONDS


def items_for(candidates, cases):
    return [{"itemId": f"c{i}|{c}", "caseId": f"c{i}", "candidate": c, "question": "q", "reference": None, "answer": "a", "truncated": False}
            for i in range(cases) for c in candidates]


def grade(correctness, language="pass"):
    return vote(correctness, language)


def test_report_rates_coverage_and_incomplete():
    items = items_for(["a", "b"], 4)
    panel = {
        "a": ["pass", "pass", "fail", "abstain"],
        "b": ["fail", "fail", "pass", "pass"],
    }
    grades = {j: {} for j in ("breeze", "qwen", "yi")}
    for candidate, verdicts in panel.items():
        for i, verdict in enumerate(verdicts):
            language = "fail" if (candidate == "a" and i == 0) else "pass"
            for judge in grades:
                grades[judge][f"c{i}|{candidate}"] = grade(verdict, language)
    grades["yi"]["c3|a"] = grade("pass")
    grades["qwen"]["c0|b"] = grade("pass")
    report = jp.build_report(items, grades, complete_judges={"breeze", "qwen", "yi"}, human=None)
    assert report["incomplete"] is False and report["humanAgreement"] is None
    a, b = report["candidates"]["a"], report["candidates"]["b"]
    assert a["correctness"] == {"pass": 2, "fail": 1, "needsReview": 1, "rate": 2 / 3, "coverage": 3 / 4}
    assert a["language"] == {"pass": 3, "fail": 1, "needsReview": 0, "rate": 3 / 4, "coverage": 1.0}
    assert b["correctness"] == {"pass": 2, "fail": 2, "needsReview": 0, "rate": 0.5, "coverage": 1.0}
    assert {f["itemId"] for f in report["flagged"]} == {"c3|a", "c0|b"}
    assert all("votes" in f for f in report["flagged"])
    assert report["selfGrading"] == {}

    partial = jp.build_report(items, grades, complete_judges={"breeze", "qwen"}, human=None)
    assert partial["incomplete"] is True
    assert partial["candidates"]["a"]["correctness"]["rate"] is None and partial["candidates"]["a"]["correctness"]["coverage"] is None


def test_report_missing_vote_is_needs_review_and_human_agreement():
    items = items_for(["a"], 3)
    grades = {j: {"c0|a": grade("pass"), "c1|a": grade("fail")} for j in ("breeze", "qwen", "yi")}
    human = {("c0", "a"): True, ("c1", "a"): True, ("c2", "a"): True}
    report = jp.build_report(items, grades, complete_judges={"breeze", "qwen", "yi"}, human=human)
    assert report["candidates"]["a"]["correctness"] == {"pass": 1, "fail": 1, "needsReview": 1, "rate": 0.5, "coverage": 2 / 3}
    assert report["humanAgreement"] == {"compared": 2, "agreed": 1}
    assert jp.build_report(items, grades, complete_judges=set(), human=human)["humanAgreement"] is None


def test_self_grading_warning():
    assert jp.self_grading_warnings(["Qwen2.5-7B@base"]) == {"Qwen2.5-7B@base": "qwen"}
    assert jp.self_grading_warnings(["taide@base", "Yi-1.5-9B@x", "my-Breeze"]) == {"Yi-1.5-9B@x": "yi", "my-Breeze": "breeze"}
    assert jp.self_grading_warnings(["taide@base"]) == {}


def test_evaluator_id_changes_with_reference_mode_and_revision():
    revisions = {"breeze": "r1", "qwen": "r2", "yi": "r3"}
    base = jp.evaluator_id(revisions, "none")
    assert base == jp.evaluator_id(dict(revisions), "none") and len(base) == 16
    assert base != jp.evaluator_id(revisions, "case")
    assert base != jp.evaluator_id({**revisions, "yi": "r4"}, "none")


def test_judges_are_the_three_pinned_models():
    assert [(j["name"], j["modelId"]) for j in jp.JUDGES] == [
        ("breeze", "MediaTek-Research/Breeze-7B-Instruct-v1_0"), ("qwen", "Qwen/Qwen2.5-7B-Instruct"), ("yi", "01-ai/Yi-1.5-9B-Chat")]
