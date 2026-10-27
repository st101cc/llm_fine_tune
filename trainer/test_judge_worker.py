import json

import pytest

import evaluator
import judge_panel as jp
import quality_pilot as qp

GOOD = json.dumps({"correctness": {"verdict": "pass", "reason": "ok"}, "language": {"verdict": "pass", "reason": "ok"}})


def make_items(count):
    return [{"itemId": f"c{i}|base", "caseId": f"c{i}", "candidate": "base", "question": "q", "reference": None, "answer": "a", "truncated": False} for i in range(count)]


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def bounded(tmp_path, items=None, **overrides):
    options = {"runtime_root": tmp_path, "judge": "qwen", "grades_path": tmp_path / "quality" / "g.jsonl", "reference_mode": "none", "cancelled": lambda: False, "max_seconds": 60}
    return qp.judge_bounded(make_items(1) if items is None else items, **{**options, **overrides})


def test_grade_items_appends_and_resumes(tmp_path):
    out, calls = tmp_path / "grades.jsonl", []
    def call(text):
        calls.append(text)
        return GOOD
    assert jp.grade_items(make_items(3), call, out, "none") == 3
    assert len(lines(out)) == 3
    assert jp.grade_items(make_items(5), call, out, "none") == 2
    assert [row["itemId"] for row in lines(out)] == [f"c{i}|base" for i in range(5)]
    assert len(calls) == 5


def test_grade_items_records_invalid_output_and_continues(tmp_path):
    out, answers = tmp_path / "grades.jsonl", iter([GOOD, "not json at all", '{"verdict": "pass", "reason": "x"}', GOOD])
    assert jp.grade_items(make_items(4), lambda text: next(answers), out, "none") == 4
    votes = [row["vote"] for row in lines(out)]
    assert votes[1] == votes[2] == {"error": "invalid_output"}
    assert "correctness" in votes[3]


def test_grade_items_propagates_call_failures_and_resumes(tmp_path):
    out, seen = tmp_path / "grades.jsonl", []
    def broken(text):
        seen.append(text)
        if len(seen) == 2:
            raise RuntimeError("CUDA out of memory")
        return GOOD
    with pytest.raises(RuntimeError):
        jp.grade_items(make_items(3), broken, out, "none")
    assert len(lines(out)) == 1
    assert jp.grade_items(make_items(3), lambda text: GOOD, out, "none") == 2
    assert [row["itemId"] for row in lines(out)] == ["c0|base", "c1|base", "c2|base"]


def test_grade_items_repairs_torn_trailing_line(tmp_path):
    out, calls = tmp_path / "grades.jsonl", []
    valid = json.dumps({"itemId": "c0|base", "vote": {"error": "invalid_output"}}) + "\n"
    out.write_bytes((valid + '{"itemId": "c1|ba').encode("utf-8"))
    def call(text):
        calls.append(text)
        return GOOD
    assert jp.grade_items(make_items(3), call, out, "none") == 2
    rows = lines(out)
    assert [row["itemId"] for row in rows] == ["c0|base", "c1|base", "c2|base"]
    assert rows[0]["vote"] == {"error": "invalid_output"} and len(calls) == 2
    assert out.read_bytes().endswith(b"\n")


def test_grade_items_drops_torn_line_without_any_newline(tmp_path):
    out = tmp_path / "grades.jsonl"
    out.write_bytes(b'{"itemId": "c0|b')
    assert jp.grade_items(make_items(1), lambda text: GOOD, out, "none") == 1
    assert [row["itemId"] for row in lines(out)] == ["c0|base"]


def test_judge_bounded_counts_only_complete_lines(tmp_path, monkeypatch):
    fake_status(monkeypatch)
    grades = tmp_path / "quality" / "g.jsonl"
    def run(root, commands, **kwargs):
        grades.write_bytes(b'{"itemId": "a", "vote": {}}\n{"itemId": "b", "vo')
        return {"status": "cancelled", "message": "m", "budget": {}}
    monkeypatch.setattr(qp, "run_with_pilot_budget", run)
    assert bounded(tmp_path, grades_path=grades)["graded"] == 1


def test_grade_items_stops_when_cancelled(tmp_path):
    out, flags = tmp_path / "grades.jsonl", iter([False, False, True])
    assert jp.grade_items(make_items(5), lambda text: GOOD, out, "none", cancelled=lambda: next(flags)) == 2
    assert len(lines(out)) == 2


def test_overlong_abstains_on_both_grades():
    parsed = jp.parse_panel_grade(jp.OVERLONG)
    assert parsed["correctness"]["verdict"] == parsed["language"]["verdict"] == "abstain"


def test_local_judge_accepts_system_revision_and_overlong():
    default = evaluator.LocalJudge("m")
    assert default.system == evaluator.SYSTEM and default.revision is None and json.loads(default.overlong)["verdict"] == "abstain"
    judge = evaluator.LocalJudge("m", system="S", revision="abc", overlong="O")
    assert (judge.model_id, judge.system, judge.revision, judge.overlong) == ("m", "S", "abc", "O")


def fake_status(monkeypatch, status="ready"):
    import quality_models
    monkeypatch.setattr(quality_models, "judge_status", lambda root: {"qwen": {"modelId": "Qwen/x", "revision": "r1", "status": status, "path": "p"}})


def no_run(*args, **kwargs):
    raise AssertionError("run_with_pilot_budget must not be called")


def test_judge_bounded_refuses_unprepared_judge(tmp_path, monkeypatch):
    fake_status(monkeypatch, "missing_weights")
    monkeypatch.setattr(qp, "run_with_pilot_budget", no_run)
    result = bounded(tmp_path)
    assert result["status"] == "unavailable" and result["reason"] == "missing_weights"


@pytest.mark.parametrize("seconds", [0, 1801, True])
def test_judge_bounded_rejects_bad_budget(tmp_path, monkeypatch, seconds):
    fake_status(monkeypatch)
    monkeypatch.setattr(qp, "run_with_pilot_budget", no_run)
    with pytest.raises(ValueError):
        bounded(tmp_path, max_seconds=seconds)


def test_judge_bounded_rejects_grades_path_outside_runtime(tmp_path, monkeypatch):
    fake_status(monkeypatch)
    monkeypatch.setattr(qp, "run_with_pilot_budget", no_run)
    with pytest.raises(ValueError):
        bounded(tmp_path, runtime_root=tmp_path / "root", grades_path=tmp_path / "elsewhere.jsonl")


def test_judge_bounded_runs_worker_and_counts_grades(tmp_path, monkeypatch):
    fake_status(monkeypatch)
    grades = tmp_path / "quality" / "panels" / "p1" / "qwen.jsonl"
    seen = {}
    def run(root, commands, **kwargs):
        (command,) = commands
        request = qp.Path(command[command.index("--output") + 1]) / "request.json"
        seen.update(command=command, kwargs=kwargs, request=json.loads(request.read_text(encoding="utf-8")))
        grades.parent.mkdir(parents=True)
        grades.write_text('{"itemId": "a", "vote": {}}\n{"itemId": "b", "vote": {}}\n', encoding="utf-8")
        return {"status": "complete", "phases": [], "chargedGpuSeconds": 5, "budget": {"remainingGpuSeconds": 9}}
    monkeypatch.setattr(qp, "run_with_pilot_budget", run)
    result = bounded(tmp_path, make_items(2), grades_path=grades)
    assert result["status"] == "complete" and result["graded"] == 2 and result["budget"] == {"remainingGpuSeconds": 9}
    command = seen["command"]
    assert command[command.index("--worker") + 1] == "judge"
    assert command[command.index("--model") + 1] == "Qwen/x" and command[command.index("--revision") + 1] == "r1"
    assert seen["kwargs"]["load_mode"] == "bnb4" and seen["kwargs"]["max_seconds"] == 60
    assert seen["request"]["gradesPath"] == str(grades.resolve()) and seen["request"]["referenceMode"] == "none" and len(seen["request"]["items"]) == 2


def test_judge_bounded_passes_through_non_complete_runs(tmp_path, monkeypatch):
    fake_status(monkeypatch)
    for status in ("cancelled", "budget_exhausted", "unavailable"):
        monkeypatch.setattr(qp, "run_with_pilot_budget", lambda *a, _s=status, **k: {"status": _s, "message": "m", "reason": "gpu_busy", "budget": {}})
        result = bounded(tmp_path)
        assert (result["status"], result["message"], result["reason"], result["graded"]) == (status, "m", "gpu_busy", 0)
    monkeypatch.setattr(qp, "run_with_pilot_budget", lambda *a, **k: {"status": "failed", "budget": {}})
    assert bounded(tmp_path)["status"] == "failed"


def test_worker_judge_phase_grades_and_closes(tmp_path, monkeypatch):
    built = []
    class FakeJudge:
        def __init__(self, model_id, **kwargs):
            built.append((model_id, kwargs))
        def __call__(self, text):
            return GOOD
        def close(self):
            built.append("closed")
    monkeypatch.setattr(evaluator, "LocalJudge", FakeJudge)
    monkeypatch.setattr(qp, "_arm_worker_deadline", lambda deadline: None)
    monkeypatch.setenv("QUALITY_WORKER_DEADLINE", "1")
    grades = tmp_path / "g.jsonl"
    request = {"judge": "qwen", "modelId": "Qwen/x", "revision": "r1", "items": make_items(2), "referenceMode": "none", "gradesPath": str(grades)}
    (tmp_path / "request.json").write_text(json.dumps(request), encoding="utf-8")
    qp._worker(tmp_path, "Qwen/x", "judge", "r1")
    assert built == [("Qwen/x", {"system": jp.SYSTEM, "revision": "r1", "overlong": jp.OVERLONG}), "closed"]
    assert len(lines(grades)) == 2
