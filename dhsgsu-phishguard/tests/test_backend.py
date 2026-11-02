"""Backend tests for feature extraction, prediction, and the CLI."""

from __future__ import annotations

import json
from unittest.mock import mock_open, patch

import pytest

from src.cli import main as cli_main
from src.explain import (
    ModelNotTrainedError,
    generate_shap_explanation,
    load_metrics,
    predict_with_explanation,
)
from src.features import extract_features


HARD_PHISH = {
    "sender_name": "Exam Portal",
    "sender_email": "exams@dhsgsu-edu.in",
    "subject": "Urgent: Verify OTP for Exam Portal",
    "body": (
        "Please reply with your password and OTP to verify your account "
        "for the exam portal."
    ),
    "url": "",
    "spf_pass": False,
    "dkim_pass": False,
    "dmarc_pass": False,
}

LEGITIMATE = {
    "sender_name": "Examinations Office",
    "sender_email": "exams@dhsgsu.edu.in",
    "subject": "B.Tech 3rd Sem Exam Timetable",
    "body": (
        "Dear students, the standard examination timetable is attached. "
        "Please review the schedule and contact the examination office "
        "with questions."
    ),
    "url": "",
    "spf_pass": True,
    "dkim_pass": True,
    "dmarc_pass": True,
}

PASSWORD_POLICY = {
    "sender_name": "IT Administration",
    "sender_email": "it-admin@dhsgsu.edu.in",
    "subject": "New IT Password Policy",
    "body": (
        "The new password policy explains minimum length, rotation, and "
        "account security requirements. Please review the policy on the "
        "official IT portal."
    ),
    "url": "",
    "spf_pass": True,
    "dkim_pass": True,
    "dmarc_pass": True,
}


def test_extract_features_detects_official_sender_and_credentials() -> None:
    features = extract_features(HARD_PHISH)

    assert features["sender_domain_is_official"] == 0.0
    assert features["asks_for_password_or_otp"] == 1.0

    official_features = extract_features(LEGITIMATE)
    assert official_features["sender_domain_is_official"] == 1.0
    assert official_features["asks_for_password_or_otp"] == 0.0


def test_security_features_and_critical_reasons() -> None:
    payload = {
        **HARD_PHISH,
        "url": "https://dhsgsu-login.com/verify",
    }
    features = extract_features(payload)
    result = predict_with_explanation(payload)

    assert features["has_blocklisted_url"] == 1.0
    assert any(
        reason.startswith("CRITICAL: Email failed SPF/DKIM/DMARC authentication")
        for reason in result["reasons"]
    )
    assert any("Threat Intelligence blocklist" in reason for reason in result["reasons"])


def test_shap_explanation_maps_transformed_features() -> None:
    explanation = generate_shap_explanation(HARD_PHISH)

    assert len(explanation.values) == len(explanation.feature_names)
    assert "heuristic__spf_pass" in explanation.feature_names
    assert "heuristic__has_blocklisted_url" in explanation.feature_names


@pytest.mark.parametrize("value", ["", None, "   "])
def test_empty_text_values_are_safe(value) -> None:
    features = extract_features(
        {"subject": value, "body": value, "sender_email": None, "url": value}
    )

    expected_length = 0 if value is None else len(value)
    assert features["body_length"] == expected_length


def test_none_email_is_safe() -> None:
    assert extract_features(None)["body_length"] == 0.0


def test_hard_phish_is_high_risk() -> None:
    result = predict_with_explanation(HARD_PHISH)

    assert result["label_name"] == "PHISHING"
    assert result["risk_score"] > 60
    assert result["reasons"]


def test_standard_exam_notice_is_legitimate() -> None:
    result = predict_with_explanation(LEGITIMATE)

    assert result["label_name"] == "LEGITIMATE"
    assert result["risk_score"] < 30


def test_password_policy_is_not_blindly_high_risk() -> None:
    result = predict_with_explanation(PASSWORD_POLICY)

    assert result["label_name"] == "LEGITIMATE"
    assert result["risk_score"] < 30


def test_cli_returns_zero_for_legitimate_email(capsys) -> None:
    exit_code = cli_main(
        [
            "--subject",
            LEGITIMATE["subject"],
            "--sender",
            LEGITIMATE["sender_email"],
            "--body",
            LEGITIMATE["body"],
        ]
    )

    assert exit_code == 0
    assert "LEGITIMATE" in capsys.readouterr().out


def test_cli_returns_one_for_phishing_email(capsys) -> None:
    exit_code = cli_main(
        [
            "--subject",
            HARD_PHISH["subject"],
            "--sender",
            HARD_PHISH["sender_email"],
            "--body",
            HARD_PHISH["body"],
        ]
    )

    assert exit_code == 1
    assert "PHISHING" in capsys.readouterr().out


def test_cli_no_spf_reports_critical_reason(capsys) -> None:
    exit_code = cli_main(
        [
            "--subject", LEGITIMATE["subject"],
            "--sender", LEGITIMATE["sender_email"],
            "--body", LEGITIMATE["body"],
            "--no-spf",
        ]
    )

    assert exit_code in (0, 1)
    assert "CRITICAL: Email failed SPF authentication" in capsys.readouterr().out


def test_corrupt_model_artifact_is_reported() -> None:
    with pytest.raises(ModelNotTrainedError, match="missing or corrupted"):
        from src.explain import _load_pipeline

        with patch("src.explain.load", side_effect=ValueError("corrupt")):
            _load_pipeline()


def test_corrupt_metrics_are_ignored() -> None:
    with patch("src.explain.Path.exists", return_value=True), patch(
        "src.explain.Path.open",
        mock_open(read_data=json.dumps({"accuracy": "not-a-number"})),
    ):
        assert load_metrics() == {}
