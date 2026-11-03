"""Exercise the PhishGuard backend with realistic edge-case messages."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explain import ModelNotTrainedError, predict_with_explanation


TEST_CASES: list[dict[str, Any]] = [
    {
        "name": "STANDARD LEGITIMATE",
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
    {
        "name": "STANDARD PHISHING",
        "subject": "URGENT: Verify OTP for DHSGSU Student Portal",
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
    {
        "name": "HARD CASE - LEGITIMATE",
        "subject": "Annual Password Expiration and Security Policy Update",
        "sender_name": "DHSGSU Information Technology Services",
        "sender_email": "it-services@dhsgsu.edu.in",
        "body": (
            "Dear Faculty and Students, this notice explains the annual "
            "password expiration schedule and updated security policy. "
            "Please review the policy information at https://dhsgsu.edu.in. "
            "The email does not request passwords, OTPs, or other credentials."
        ),
        "url": "https://dhsgsu.edu.in",
        "spf_pass": True,
        "dkim_pass": True,
        "dmarc_pass": True,
    },
    {
        "name": "HARD CASE - SUBTLE PHISHING",
        "subject": "Semester Fee Receipt and Invoice Verification",
        "sender_name": "DHSGSU Accounts Department",
        "sender_email": "accounts@dhsgsu-results.online",
        "body": (
            "Dear Student, your semester fee receipt for the Winter 2026 "
            "session is now available. Please review the attached invoice "
            "details and confirm your payment status via the student portal "
            "to avoid any administrative holds."
        ),
        "url": "http://dhsgsu-results.online/invoice/verify",
        "spf_pass": False,
        "dkim_pass": False,
        "dmarc_pass": False,
    },
]


def main() -> None:
    print("DHSGSU PhishGuard Extensive Backend Stress Test")
    print("=" * 60)
    for case in TEST_CASES:
        scenario = case["name"]
        payload = {key: value for key, value in case.items() if key != "name"}
        print(f"\n--- {scenario} ---")
        try:
            result = predict_with_explanation(payload)
        except ModelNotTrainedError as exc:
            print(f"ERROR: Model file may be locked or inaccessible: {exc}")
            continue
        except (KeyError, TypeError, ValueError) as exc:
            print(f"ERROR: Backend prediction failed: {exc}")
            continue

        print(f"Scenario Name: {scenario}")
        print(f"Verdict: {result['label_name']}")
        print(f"Risk Score: {result['risk_score']:.2f}/100")
        print(f"Risk Band: {result['risk_band']}")
        print("Top 2 Heuristic Reasons:")
        for reason in result["reasons"][1:3]:
            print(f"  - {reason}")


if __name__ == "__main__":
    main()
