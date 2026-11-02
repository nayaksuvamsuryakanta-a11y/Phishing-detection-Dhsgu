"""
University constants and security policy for DHSGSU PhishGuard.

Every rule that decides whether a message looks official lives here so the
detection policy can be audited in one place for the BCA project report.
"""

from __future__ import annotations

# Reproducibility seed: synthetic data generation, train/test split, and any
# stochastic model hyper-parameter all use this so results are re-runnable.
SEED = 42

UNIVERSITY_NAME = "Dr. Harisingh Gour Vishwavidyalaya, Sagar (M.P.)"
UNIVERSITY_SHORT = "DHSGSU"
OFFICIAL_DOMAINS = ["dhsgsu.edu.in"]
OFFICIAL_URL = "https://dhsgsu.edu.in/"
THREAT_INTEL_BLOCKLIST = [
    "dhsgsu-login.com",
    "verify-student.com",
    "bit.ly",
    "student-portal-update.com",
    "192.168.",
]

# Verified official senders based on university infrastructure.
# The heuristic branch rewards mail whose address matches one of these exactly;
# a message claiming to be from the "Registrar" but sent from a Gmail address
# is a classic display-name spoof and scores as suspicious.
OFFICIAL_SENDERS = {
    "registrar@dhsgsu.edu.in": "Registrar",
    "admission@dhsgsu.edu.in": "Admission Cell",
    "vc@dhsgsu.edu.in": "Vice-Chancellor",
    "exams@dhsgsu.edu.in": "Controller of Examinations",
}

# Phishing keywords grouped by the social-engineering tactic they signal.
PHISHING_KEYWORDS = {
    "urgency": [
        "urgent", "immediately", "asap", "within 24 hours", "right now",
        "without delay", "emergency", "immediate action", "final notice",
        "last chance", "expire", "suspended", "deactivated", "blocked",
        "restricted", "deadline today", "act now", "critical alert",
    ],
    "threat": [
        "warning", "penalty", "legal action", "suspension", "blacklisted",
        "arrest", "fined", "compulsory", "strict action", "threat",
        "terminat", "banned", "prosecut", "liable", "disconnection",
        "expelled", "confiscat", "forfeit", "court notice",
    ],
    "credential": [
        "verify your account", "verify your password", "reset your password",
        "login credentials", "user id and password", "otp", "one time password",
        "bank details", "account number", "ifsc", "debit card", "credit card",
        "cvv", "click here to verify", "update your credentials",
        "authenticate", "secure your account", "claim your refund",
        "click the link", "provide credentials", "verify identity",
        "enter your password", "portal login", "reactivate your account",
    ],
}

# Look-alike domains attackers register to imitate dhsgsu.edu.in.
# These are *never* legitimate sender domains and are used to seed the phish
# corpus, giving the model explicit negative examples of domain squatting.
SQUAT_DOMAINS = [
    "dhsgsu-edu.in",
    "dhsgsu-results.com",
    "dhsgsu.ac-india.in",
    "dhsgsuportal.in",
    "dhsgsu-edu.com",
    "dhsgsu-official.in",
    "dhsgsuexam.in",
    "dhsgsu-results.in",
    "dhsgsu-notice.in",
    "dhsgsu-portal.org",
]

# Free-mail domains the university never uses as a sender domain.
FREE_MAIL_DOMAINS = [
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
    "zoho.com",
    "protonmail.com",
    "mail.com",
    "inbox.com",
    "rediffmail.com",
    "icloud.com",
]

# TF-IDF limits for the text branch.
TFIDF_MAX_FEATURES = 3000
TFIDF_NGRAM_RANGE = (1, 2)

# Risk-score bands returned by src/explain.py.
RISK_BANDS = [(30, "Low"), (60, "Medium"), (100, "High")]

# Report artefacts.
MODEL_PATH = "models/phishguard_model.joblib"
# Override this value in deployment when the model artifact is regenerated.
MODEL_SHA256 = "851974C5CC03E4DD005A0DAF0DCF0B7134A3624F079654C75C54A2B6607B621D"
METRICS_PATH = "metrics.json"
REPORTS_DIR = "reports"