"""Run realistic end-to-end backend demos without starting Streamlit."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explain import generate_shap_explanation, predict_with_explanation


GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"


DEMOS: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "DEMO 1 - Legitimate",
        {
            "subject": "End Semester Examination Timetable - Winter 2026",
            "sender_name": "DHSGSU Examinations Office",
            "sender_email": "exams@dhsgsu.edu.in",
            "body": (
                "Dear Student, the winter 2026 examination timetable has been "
                "published. Please download it from the official student portal "
                "at https://dhsgsu.edu.in/student-login."
            ),
            "url": "https://dhsgsu.edu.in/exam-timetable",
            "spf_pass": True,
            "dkim_pass": True,
            "dmarc_pass": True,
        },
    ),
    (
        "DEMO 2 - Phishing",
        {
            "subject": "URGENT: DHSGSU Student Portal Account Suspension",
            "sender_name": "DHSGSU Student Administration",
            "sender_email": "admin@dhsgsu-edu.in",
            "body": (
                "Your student account will be suspended in 24 hours due to "
                "unusual activity. Verify your identity immediately by replying "
                "with your Roll Number and OTP, or click here: "
                "http://192.168.1.55/verify-dhsgsu"
            ),
            "url": "http://192.168.1.55/verify-dhsgsu",
            "spf_pass": False,
            "dkim_pass": False,
            "dmarc_pass": False,
        },
    ),
)


def _verdict_color(label_name: str) -> str:
    return RED if label_name == "PHISHING" else GREEN


def _print_result(title: str, result: dict[str, Any]) -> None:
    color = _verdict_color(result["label_name"])
    print(f"\n{CYAN}{title}{RESET}")
    print(f"Verdict: {color}{result['label_name']}{RESET}")
    print(f"Risk Score: {result['risk_score']:.2f}/100")
    print(f"Risk Band: {result['risk_band']}")
    print("Top 3 Reasons:")
    for reason in result["reasons"][:3]:
        print(f"  - {reason}")


def main() -> None:
    for title, email in DEMOS:
        result = predict_with_explanation(email)
        _print_result(title, result)

        if result["label_name"] == "PHISHING":
            explanation = generate_shap_explanation(email)
            feature_count = len(explanation.feature_names)
            if len(explanation.values) != feature_count:
                raise RuntimeError(
                    "SHAP explanation feature values and names are misaligned."
                )
            print(
                f"{GREEN}SHAP explanation generated successfully "
                f"({feature_count} features).{RESET}"
            )


if __name__ == "__main__":
    main()
