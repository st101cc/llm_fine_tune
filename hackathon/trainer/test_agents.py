from advisor import privacy_scan
from graph import accuracy_agent_review


def test_accuracy_agent_reports_measured_ground_truth():
    report = accuracy_agent_review([
        {
            "modelId": "local-model",
            "status": "complete",
            "qualityEvidence": {
                "metric": "label_accuracy",
                "score": 0.8,
                "coverage": 1,
                "evaluatedExamples": 5,
            },
        }
    ])

    assert report["status"] == "measured"
    assert report["metric"] == "label_accuracy"
    assert report["models"]["local-model"]["score"] == 0.8


def test_accuracy_agent_requests_rubric_for_open_ended_answers():
    report = accuracy_agent_review([
        {"modelId": "local-model", "status": "complete", "qualityEvidence": {"coverage": 0, "score": None}}
    ])

    assert report["status"] == "needs_rubric"
    assert report["metric"] is None
    assert "rubric" in report["message"].lower()


def test_privacy_agent_flags_sensitive_rows_without_returning_values():
    report = privacy_scan([
        {"text": "Contact alice@example.com or +1 555 123 4567"},
        {"text": "token sk_live_123456789012345"},
    ])

    assert report["status"] == "review"
    assert report["safeToSendToHostedModel"] is False
    assert report["matches"]["email"] == 1
    assert report["matches"]["phone"] == 1
    assert report["matches"]["secret"] == 1
    assert report["flaggedRows"] == [0, 1]
    assert "alice@example.com" not in str(report)


def test_privacy_agent_marks_clean_rows_safe():
    report = privacy_scan([{"text": "Explain gradient descent simply."}])

    assert report["status"] == "clear"
    assert report["safeToSendToHostedModel"] is True
    assert report["flaggedRows"] == []
