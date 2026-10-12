"""Offline contract tests; quality_policy is an externally owned boundary."""
import importlib
import importlib.util
import sys
from types import SimpleNamespace

import pytest


class Sidecar:
    def __getattr__(self, name):
        assert importlib.util.find_spec("quality_evaluation"), "Evaluation sidecar is not implemented"
        return getattr(importlib.import_module("quality_evaluation"), name)


@pytest.fixture
def evaluation(monkeypatch):
    monkeypatch.setitem(sys.modules, "quality_policy", SimpleNamespace(normalize_policy=lambda policy: policy, inspect_text=lambda text, policy: [
        {"code": "term", "start": 0, "end": 2, "original": "软件", "replacement": "軟體", "reason": "fixture", "editable": True}
    ] if "软件" in text else []))
    return Sidecar()


def case(id="c", task="classification", reference="yes"):
    return {"id": id, "input": "question " + id, "task": task, "reference": reference}


def answer(output="yes", candidate="base", case_id="c", **extra):
    from quality_evaluation import fingerprint
    return {"caseId": case_id, "candidate": candidate, "output": output, "protocol": {"fixture": 1}, "policyHash": fingerprint({}), **extra}


def test_locality_never_becomes_correctness(evaluation):
    result = evaluation.evaluate([case(reference="软件")], [answer("软件")], {})
    row = result["candidates"]["base"]["results"][0]
    assert row["correct"] is True
    assert len(row["localityFindings"]) == 1
    assert evaluation.evaluate([case()], [answer("軟體")], {})["candidates"]["base"]["metrics"]["classification"]["correct"] == 0


@pytest.mark.parametrize("cases,answers", [
    ([case(), case()], []),
    ([case()], [answer(), answer()]),
    ([case()], [answer(case_id="unknown")]),
    ([case(task="unsupported")], []),
    ([case()], [answer(judgment="maybe")]),
])
def test_rejects_ambiguous_inputs(evaluation, cases, answers):
    with pytest.raises(ValueError):
        evaluation.evaluate(cases, answers, {})


@pytest.mark.parametrize("output,valid,correct", [
    ('{"b":[2],"a":1}', True, True),
    ('{"a":true,"b":[2]}', True, False),
    ('{"a":1,"a":1,"b":[2]}', False, False),
    ('{"a":NaN}', False, False),
    ('{"a":Infinity}', False, False),
    ('{"a":1e999}', False, False),
    ('```json\n{"a":1}\n```', False, False),
])
def test_json_is_strict_and_structural(evaluation, output, valid, correct):
    row = evaluation.evaluate([case(task="json", reference='{"a":1,"b":[2]}')], [answer(output)], {})["candidates"]["base"]["results"][0]
    assert row["validJSON"] is valid
    assert row["correct"] is correct


@pytest.mark.parametrize("output,correct", [("A", True), (" A\n", True), ("A.", False), ("答案是 A", False), ("a", False), ("AB", False)])
def test_mcq_requires_whole_answer(evaluation, output, correct):
    row = evaluation.evaluate([case(task="mcq", reference="A")], [answer(output)], {})["candidates"]["base"]["results"][0]
    assert row["correct"] is correct


def test_writing_and_missing_reference_require_review(evaluation):
    result = evaluation.evaluate([case(task="writing")], [answer(), answer(candidate="reviewed", judgment="pass")], {})
    assert result["candidates"]["base"]["results"][0]["correct"] is None
    assert result["candidates"]["base"]["metrics"]["writing"]["needsReview"] == 1
    assert result["candidates"]["reviewed"]["results"][0]["correct"] is True
    c = case()
    del c["reference"]
    assert evaluation.evaluate([c], [answer()], {})["candidates"]["base"]["results"][0]["correct"] is None


def test_coverage_pairing_and_provenance(evaluation):
    cases = [case(), case("d")]
    answers = [answer(), answer(candidate="tuned"), answer(case_id="d")]
    result = evaluation.evaluate(cases, answers, {})
    assert result["candidates"]["tuned"]["coverage"]["missingCaseIds"] == ["d"]
    assert result["comparisons"] == []
    assert result["unpaired"][0]["reason"] == "coverage differs"
    answers.append(answer(case_id="d", candidate="tuned", output="no"))
    result = evaluation.evaluate(cases, answers, {})
    assert result["comparisons"][0]["rightMinusLeftAccuracy"] == -0.5
    assert len(result["provenance"]["casesHash"]) == 64
    assert result["provenance"] == evaluation.evaluate(list(reversed(cases)), list(reversed(answers)), {})["provenance"]
    assert result["provenance"]["policyHash"] != evaluation.evaluate(cases, answers, {"version": 2})["provenance"]["policyHash"]


@pytest.mark.parametrize("extra", [{"protocol": {"temperature": 1}}, {"policyHash": "different"}])
def test_different_protocol_or_policy_not_paired(evaluation, extra):
    result = evaluation.evaluate([case()], [answer(), answer(candidate="tuned", **extra)], {})
    assert result["comparisons"] == []
    assert result["unpaired"]


def test_cancellation_exposes_unprocessed_coverage(evaluation):
    result = evaluation.evaluate([case()], [answer()], {}, cancelled=lambda: True)
    assert result["status"] == "cancelled"
    assert result["candidates"]["base"]["coverage"]["missingCaseIds"] == ["c"]
    assert result["comparisons"] == []


def test_mixed_task_protocols_pair_by_case_not_by_unique_protocol(evaluation):
    cases = [case("a", "json", '{"x":1}'), {**case("b", "mcq", "A"), "subject": "mixed"}]
    cases[0]["subject"] = "mixed"
    answers = [answer('{"x":1}', name, "a", protocol={"tokens": 256}) for name in ("base", "tuned")]
    answers += [answer("A", name, "b", protocol={"tokens": 8}) for name in ("base", "tuned")]
    result = evaluation.evaluate(cases, answers, {})
    assert len(result["comparisons"]) == 1
    assert result["comparisons"][0]["rightMinusLeftAccuracy"] == 0
    assert result["candidates"]["base"]["subjects"]["mixed"]["accuracy"] == 1
    answers[-1]["protocol"] = {"tokens": 256}
    assert evaluation.evaluate(cases, answers, {})["comparisons"] == []


def test_protocol_swap_between_cases_is_not_equivalent(evaluation):
    cases = [case("a"), case("b")]
    answers = [answer("yes", "base", "a", protocol="p"), answer("yes", "base", "b", protocol="q"), answer("yes", "tuned", "a", protocol="q"), answer("yes", "tuned", "b", protocol="p")]
    assert evaluation.evaluate(cases, answers, {})["comparisons"] == []


def test_unknown_generation_protocol_is_not_comparable(evaluation):
    answers = [answer(candidate="a"), answer(candidate="b")]
    for row in answers:
        del row["protocol"]
    result = evaluation.evaluate([case()], answers, {})
    assert result["comparisons"] == []
    assert result["unpaired"][0]["reason"] == "protocol missing"


def test_real_policy_normalizes_provenance_and_preserves_informational_findings():
    import quality_evaluation
    from quality_policy import normalize_policy
    text = "软件與電腦"
    cases = [case(reference=text)]
    answers = [answer(text)]
    result = quality_evaluation.evaluate(cases, answers, {})
    normalized = quality_evaluation.evaluate(cases, answers, normalize_policy({}))
    assert result["provenance"]["policyHash"] == normalized["provenance"]["policyHash"]
    assert result["provenance"]["policyId"] == normalize_policy({})["id"]
    row = result["candidates"]["base"]["results"][0]
    assert row["correct"] is True
    assert any(f["code"] == "mixed_script" and f["replacement"] is None and f["editable"] is False for f in row["localityFindings"])


def test_oversized_answer_is_rejected_before_inspection(evaluation):
    with pytest.raises(ValueError, match="65536"):
        evaluation.evaluate([case()], [answer("a" * 65537)], {})
