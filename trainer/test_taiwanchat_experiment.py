import json

from comparison import leading_letter, scoring_policy
from taiwanchat_experiment import apply_terms


def test_self_containing_replacement_does_not_rewrite_correct_text():
    policy = scoring_policy({"terms": {"算法": "演算法", "數據": "資料"}, "exclusions": []})
    assert apply_terms("這個演算法用數據", policy)[0] == "這個演算法用資料"
    assert apply_terms("這個算法", policy)[0] == "這個演算法"


def test_leading_letter_accepts_echoed_option_only():
    assert leading_letter("D. 約 70 萬次") == "D"
    assert leading_letter("B") == "B"
    assert leading_letter("Another answer") is None


def test_report_runs_on_fixture_answers(tmp_path, monkeypatch):
    import argparse
    import taiwanchat_experiment
    directory = taiwanchat_experiment.experiment_dir(tmp_path)
    (directory / "eval").mkdir(parents=True)
    manifest = {"benchmarkRevision": "b" * 40, "terms": {"terms": {"數據": "資料"}}, "files": {}}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(taiwanchat_experiment, "load_benchmark", lambda *a, **k: {"cases": [{"id": "m1", "reference": "A"}]})
    for model in ("base", "original", "reviewed"):
        answers = [{"caseId": "m1", "output": "A"}, {"caseId": "taiwanchat:1", "output": "這是數據"}]
        (directory / "eval" / f"{model}.json").write_text(json.dumps({"answers": answers, "complete": True}), encoding="utf-8")
    taiwanchat_experiment.report(argparse.Namespace(runtime_root=tmp_path))
    results = json.loads((directory / "report.json").read_text(encoding="utf-8"))["results"]
    assert set(results) == {"base", "original", "reviewed"}
    assert results["base"]["tmmluplusAccuracy"] == 1 and results["base"]["answersWithMainlandTerm"] == 1
