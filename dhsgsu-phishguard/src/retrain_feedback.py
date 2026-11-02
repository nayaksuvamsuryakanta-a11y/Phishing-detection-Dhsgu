"""Merge human feedback into the corpus, retrain, and refresh model integrity."""

from __future__ import annotations

import csv
import hashlib
import re
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[1]
if not (_PROJECT_ROOT / "src").is_dir() or not (_PROJECT_ROOT / "config.py").exists():
    raise RuntimeError(f"Cannot resolve project root from {_HERE}")
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

from config import MODEL_PATH
from features import extract_features

FEEDBACK_PATH = _PROJECT_ROOT / "data" / "feedback.csv"
CORPUS_PATH = _PROJECT_ROOT / "data" / "emails.csv"
CONFIG_PATH = _PROJECT_ROOT / "config.py"


def merge_feedback() -> int:
    if not FEEDBACK_PATH.exists():
        return 0
    feedback = pd.read_csv(FEEDBACK_PATH)
    if feedback.empty:
        return 0
    required = {
        "subject", "sender_email", "body", "url", "user_true_label",
    }
    missing = required - set(feedback.columns)
    if missing:
        raise ValueError(f"Feedback is missing required columns: {sorted(missing)}")

    rows = []
    for row in feedback.to_dict("records"):
        email = {
            "subject": row.get("subject", ""),
            "sender_email": row.get("sender_email", ""),
            "body": row.get("body", ""),
            "url": row.get("url", ""),
            "spf_pass": True,
            "dkim_pass": True,
            "dmarc_pass": True,
        }
        rows.append({
            **email,
            **{
                key: value
                for key, value in extract_features(email).items()
                if key == "has_blocklisted_url"
            },
            "sender_name": "",
            "label": int(row["user_true_label"]),
        })

    existing = pd.read_csv(CORPUS_PATH)
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    combined.to_csv(CORPUS_PATH, index=False)
    FEEDBACK_PATH.replace(
        FEEDBACK_PATH.with_name("feedback.archive.csv")
    )
    return len(rows)


def update_model_hash() -> str:
    model_path = _PROJECT_ROOT / MODEL_PATH
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest().upper()
    text = CONFIG_PATH.read_text(encoding="utf-8")
    updated, count = re.subn(
        r'MODEL_SHA256 = "[A-Fa-f0-9]+"',
        f'MODEL_SHA256 = "{digest}"',
        text,
        count=1,
    )
    if count != 1:
        raise ValueError("Could not locate MODEL_SHA256 in config.py")
    CONFIG_PATH.write_text(updated, encoding="utf-8")
    return digest


def main() -> None:
    count = merge_feedback()
    if count == 0:
        print("No feedback rows found; nothing to retrain.")
        return
    subprocess.run(
        [sys.executable, "-m", "src.train"],
        cwd=_PROJECT_ROOT,
        check=True,
    )
    print(f"Merged {count} feedback rows.")
    print(f"Updated model SHA-256: {update_model_hash()}")


if __name__ == "__main__":
    main()
