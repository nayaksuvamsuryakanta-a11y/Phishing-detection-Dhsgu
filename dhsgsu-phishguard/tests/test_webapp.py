"""Streamlit AppTest coverage for the email analysis workflow."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).parents[1] / "app" / "streamlit_app.py"


def test_app_initializes_without_exceptions() -> None:
    app = AppTest.from_file(str(APP_PATH)).run()

    assert not app.exception


def test_hard_phish_submission_renders_verdict() -> None:
    app = AppTest.from_file(str(APP_PATH)).run()

    app.text_input[0].set_value("Urgent: Verify OTP for Exam Portal")
    app.text_input[1].set_value("Exam Portal")
    app.text_input[2].set_value("exams@dhsgsu-edu.in")
    app.text_area[0].set_value(
        "Please reply with your password and OTP to verify your account "
        "for the exam portal."
    )
    app.text_input[3].set_value("")
    app.checkbox[0].set_value(False)
    app.form_submit_button[0].click().run()

    assert not app.exception
    assert any(metric.label == "Risk score" for metric in app.metric)
    assert any("password" in item.value.lower() for item in app.markdown)
    assert any("What is the ACTUAL status" in item.label for item in app.radio)
