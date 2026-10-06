"""Frozen, inference-only TAIDE baseline; writes local evidence, never trains."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from quality_models import TAIDE_ID, fingerprint, get_model_status, load_benchmark, load_settings
from quality_pilot import budget_status, generate_bounded
from quality_policy import inspect_text, normalize_policy

POSITIVE = "需升級處理"
NEGATIVE = "一般處理"
DIMENSIONS = ("facts", "instructions", "taiwanUsage")
JUDGMENTS = ("pass", "fail", "needs_review")


def support_cases():
    situations = [
        ("付款被重複扣款兩次，需要查帳。", POSITIVE),
        ("包裹內商品破裂且造成割傷，需要調查。", POSITIVE),
        ("帳號出現非本人登入及訂單，疑似遭盜用。", POSITIVE),
        ("個人資料被寄给其他客戶，疑似資料外洩。", POSITIVE),
        ("退款超過承諾期限十天仍未入帳，需要查核。", POSITIVE),
        ("想查詢訂單目前配送進度。", NEGATIVE),
        ("想了解店家的營業時間。", NEGATIVE),
        ("想知道如何下載已開立的電子發票。", NEGATIVE),
        ("想了解尚未下單商品的尺寸規格。", NEGATIVE),
        ("想了解如何使用網站上的優惠券。", NEGATIVE),
    ]
    rule = "分類規則：涉及金流異常、人身傷害、帳號安全、資料外洩或退款逾期需升級處理；一般資訊詢問為一般處理。只輸出需升級處理或一般處理。\n"
    cases = [{"id": f"support-class-{i+1:02}", "task": "classification", "input": rule + text, "reference": label, "synthetic": True} for i, (text, label) in enumerate(situations)]
    for i, (text, label) in enumerate(situations):
        expected = "表示理解，說明轉交專人查核；不承諾結果、退款日期或索取密碼。" if label == POSITIVE else "表示理解，引導客戶至訂單頁或網站說明查詢；不虛構配送狀態、營業時間或商品規格。"
        cases.append({"id": f"support-reply-{i+1:02}", "task": "writing", "synthetic": True,
                      "input": f"你是臺灣網路商店客服。客戶說：{text}\n僅知上述資訊。請使用臺灣繁體中文，以兩句話回覆。{expected}",
                      "reference": expected, "rubric": {"facts": "不捏造未知資訊或承諾", "instructions": "兩句話且遵守題目處理方式", "taiwanUsage": "繁體中文與符合臺灣語境的用語"}})
    return cases


def review_template(cases, answers, policy):
    by_id = {case["id"]: case for case in cases}
    reviews = {}
    for answer in answers:
        findings = inspect_text(answer["output"], policy)
        reviews[answer["caseId"]] = {"outputHash": fingerprint(answer["output"]), "reviewer": "", "reason": "",
            "judgments": {key: "needs_review" for key in DIMENSIONS} if by_id[answer["caseId"]]["task"] == "writing" else {},
            "languageDecisions": [{"findingHash": fingerprint(finding), "finding": finding, "decision": "pending", "reason": ""} for finding in findings]}
    return reviews


def summarize(cases, answers, policy, reviews=None):
    policy = normalize_policy(policy)
    ids = {case["id"] for case in cases}
    by_id = {answer["caseId"]: answer for answer in answers}
    if len(by_id) != len(answers) or len(ids) != len(cases) or not set(by_id) <= ids:
        raise ValueError("Duplicate or unknown baseline case IDs")
    templates = review_template(cases, answers, policy)
    reviews = reviews or {}
    if not set(reviews) <= set(templates):
        raise ValueError("Reviews require a saved answer")
    rows = []
    for case in cases:
        answer = by_id.get(case["id"])
        template = templates.get(case["id"])
        review = reviews.get(case["id"], template)
        if review is not None:
            if review.get("outputHash") != template["outputHash"]:
                raise ValueError("Review output hash differs")
            judgments = review.get("judgments", {})
            if set(judgments) != set(template["judgments"]) or any(value not in JUDGMENTS for value in judgments.values()):
                raise ValueError("Invalid human judgment dimensions")
            decisions = review.get("languageDecisions", [])
            expected = {item["findingHash"]: item["finding"] for item in template["languageDecisions"]}
            if len(decisions) != len(expected) or {item.get("findingHash") for item in decisions} != set(expected):
                raise ValueError("Language findings changed")
            for item in decisions:
                if item.get("finding") != expected[item["findingHash"]] or item.get("decision") not in ("accepted", "ignored", "pending"):
                    raise ValueError("Invalid language decision")
                if item["decision"] != "pending" and not str(item.get("reason", "")).strip():
                    raise ValueError("Accepted or ignored decisions require a reason")
            if any(value != "needs_review" for value in judgments.values()) or any(item["decision"] != "pending" for item in decisions):
                if not str(review.get("reviewer", "")).strip() or not str(review.get("reason", "")).strip():
                    raise ValueError("Human decisions require reviewer and reason")
        output = answer["output"].strip() if answer else None
        valid = output in (POSITIVE, NEGATIVE) if case["task"] == "classification" else output in ("A", "B", "C", "D") if case["task"] == "mcq" else None
        correct = output == case["reference"] if case["task"] != "writing" and answer else None
        if case["task"] == "writing":
            judgments = review["judgments"] if review else {}
            correct = False if "fail" in judgments.values() else True if len(judgments) == 3 and all(value == "pass" for value in judgments.values()) else None
        rows.append({"case": case, "answer": answer, "valid": valid, "correct": correct, "review": review})
    def coverage(task):
        subset = [row for row in rows if row["case"]["task"] == task]
        answered = sum(row["answer"] is not None for row in subset)
        return subset, {"total": len(subset), "answered": answered, "missing": len(subset) - answered}
    mcq, mcqm = coverage("mcq")
    mcqm.update(correct=sum(row["correct"] is True for row in mcq), invalidAnswers=sum(row["answer"] is not None and not row["valid"] for row in mcq))
    mcqm["accuracy"] = mcqm["correct"] / mcqm["answered"] if mcqm["answered"] else None
    classes, metrics = coverage("classification")
    tp = sum(row["answer"] is not None and row["case"]["reference"] == POSITIVE and row["answer"]["output"].strip() == POSITIVE for row in classes)
    fp = sum(row["answer"] is not None and row["case"]["reference"] == NEGATIVE and row["answer"]["output"].strip() == POSITIVE for row in classes)
    fn = sum(row["answer"] is not None and row["case"]["reference"] == POSITIVE and row["answer"]["output"].strip() != POSITIVE for row in classes)
    metrics.update(positiveLabel=POSITIVE, truePositive=tp, falsePositive=fp, falseNegative=fn,
                   invalidPredictions=sum(row["answer"] is not None and not row["valid"] for row in classes),
                   accuracy=sum(row["correct"] is True for row in classes) / metrics["answered"] if metrics["answered"] else None,
                   precision=tp / (tp + fp) if tp + fp else None, recall=tp / (tp + fn) if tp + fn else None,
                   F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)
    writing, writingm = coverage("writing")
    writingm.update(passed=sum(row["correct"] is True for row in writing), failed=sum(row["correct"] is False for row in writing), needsReview=sum(row["correct"] is None for row in writing))
    return {"tmmluplus": mcqm, "classification": metrics, "writing": writingm, "policy": policy, "results": rows,
            "metricDenominator": "Answered items only; missing coverage is reported separately. Invalid positive-label answers count as false negatives.",
            "limitation": "Exploratory small sample, not a leaderboard score or proof of quality improvement. Language findings are suggestions requiring human review."}


def save(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True, type=Path, help="Existing shared runtime root; never a new budget ledger")
    parser.add_argument("--benchmark-revision", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--load-mode", choices=("fp16", "bnb4"), default="fp16")
    parser.add_argument("--seconds", type=float, default=1800)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--review", type=Path, help="Rebuild report from edited review JSON; no inference")
    args = parser.parse_args()
    if not 0 < args.seconds <= 1800:
        parser.error("--seconds must be > 0 and <= 1800")
    settings = load_settings(args.load_mode)
    if len(args.model_revision) != 40 or any(ch not in "0123456789abcdef" for ch in args.model_revision):
        parser.error("An immutable model SHA is required")
    benchmark = load_benchmark(args.benchmark_revision, runtime_root=args.runtime_root)
    if len(benchmark["cases"]) != 30:
        parser.error("The baseline requires the frozen 30-question TMMLU+ selection")
    manifest = {"version": "taide-baseline-v1", "modelId": TAIDE_ID, "modelRevision": args.model_revision, "settings": settings,
                "benchmarkChecksum": benchmark["manifestChecksum"], "cases": benchmark["cases"] + support_cases(), "policy": normalize_policy({}), "scoringVersion": "support-v1"}
    directory = args.runtime_root.resolve() / "quality" / "baselines" / fingerprint(manifest)
    if not directory.resolve().is_relative_to(args.runtime_root.resolve()):
        raise ValueError("Baseline path escapes runtime root")
    directory.mkdir(parents=True, exist_ok=True)
    frozen = directory / "manifest.json"
    if frozen.exists() and json.loads(frozen.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Frozen manifest differs")
    save(frozen, manifest)
    answers_path = directory / "answers.json"
    if args.prepare_only and (directory / "run.json").exists():
        print(json.dumps({"directory": str(directory), "status": "existing_evidence_preserved"}))
        return
    if args.review:
        answers = json.loads(answers_path.read_text(encoding="utf-8"))
        reviews = json.loads(args.review.read_text(encoding="utf-8"))
        report = summarize(manifest["cases"], answers, manifest["policy"], reviews)
        save(directory / ("review-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json"), reviews)
    else:
        if answers_path.exists() and json.loads(answers_path.read_text(encoding="utf-8")) and not args.prepare_only:
            parser.error("Baseline already has answers; use --review or inspect existing evidence")
        status = get_model_status(TAIDE_ID, revision=args.model_revision)
        result = {"status": "prepared", "model": status, "budget": budget_status(args.runtime_root), "answers": []}
        if not args.prepare_only:
            # One question must successfully load and complete before the remaining batch.
            result = generate_bounded(manifest["cases"][:1], runtime_root=args.runtime_root, revision=args.model_revision, load_mode=args.load_mode, max_seconds=min(args.seconds, 1800))
            if result["status"] == "complete" and len(result["answers"]) == 1:
                smoke = result
                remaining = args.seconds - smoke.get("run", {}).get("chargedGpuSeconds", 0)
                if remaining > 0:
                    result = generate_bounded(manifest["cases"][1:], runtime_root=args.runtime_root, revision=args.model_revision, load_mode=args.load_mode, max_seconds=remaining)
                else:
                    result = {"status": "budget_exhausted", "answers": [], "message": "Requested allowance consumed by smoke test"}
                result["answers"] = smoke["answers"] + result["answers"]
                result["smoke"] = {key: value for key, value in smoke.items() if key != "answers"}
            save(answers_path, result["answers"])
            save(directory / "reviews.json", review_template(manifest["cases"], result["answers"], manifest["policy"]))
        attempts = directory / "attempts"
        attempts.mkdir(exist_ok=True)
        save(attempts / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json"), result)
        save(directory / "run.json", result)
        report = summarize(manifest["cases"], result["answers"], manifest["policy"])
        report["runStatus"] = result["status"]
    save(directory / "report.json", report)
    print(json.dumps({"directory": str(directory), "status": report.get("runStatus", "reviewed"), "tmmluplus": report["tmmluplus"], "classification": report["classification"], "writing": report["writing"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
