"""
Feature extraction for DHSGSU PhishGuard.

Two parallel feature branches are fused by a single scikit-learn
`ColumnTransformer`:

  * Text branch  -- TF-IDF over the concatenated subject + body. This catches
    the social-engineering language ("urgent", "verify your account", "OTP")
    that phishers reuse verbatim across messages.
  * Heuristic branch -- hand-crafted numeric signals about the sender domain,
    URL structure and urgency. These are the rules a security-aware student
    would check by eye; encoding them as features lets the model learn when to
    trust text evidence over a plausible-looking display name.

`ColumnTransformer` is used rather than `FeatureUnion` because the two branches
consume different input columns (one text column, several numeric columns) and
must each receive only the data they understand. `FeatureUnion` applies every
transformer to the *same* whole input, which would force the vectorizer to try
tokenising numeric flags and the numeric transformer to consume raw text.

Security rationale for every heuristic is given inline so it can be quoted
directly in the BCA project report.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

# --- PATH BOOTSTRAP (idempotent) ---
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[1]
if not (_PROJECT_ROOT / "src").is_dir() or not (_PROJECT_ROOT / "config.py").exists():
    raise RuntimeError(f"Cannot resolve project root from {_HERE}")
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# --- END PATH BOOTSTRAP ---

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer

from config import (
    OFFICIAL_DOMAINS,
    PHISHING_KEYWORDS,
    THREAT_INTEL_BLOCKLIST,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
)

# Canonical names of the heuristic columns. They are referenced by the
# pipeline, the explanation engine and the report charts, so they must stay
# stable across versions.
HEURISTIC_FEATURES = [
    "sender_domain_is_official",
    "levenshtein_distance_to_official_domain",
    "url_count",
    "has_ip_url",
    "has_http_only_url",
    "urgency_keyword_count",
    "asks_for_password_or_otp",
    "body_length",
    "spf_pass",
    "dkim_pass",
    "dmarc_pass",
    "has_blocklisted_url",
]

# The two heuristics that are purely about the sender domain. They are the
# whitelisting rule, and ablation separates their contribution from the
# classifier's own learning.
DOMAIN_FEATURES = [
    "sender_domain_is_official",
    "levenshtein_distance_to_official_domain",
]

TEXT_COLUMN = "text_combined"

# Matches a bare IPv4 address, optionally followed by a port.
IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b")
URL_PATTERN = re.compile(r"https?://[^\s<>\"]+|www\.[^\s<>\"]+")


def _domain_of(email: str) -> str:
    """Return the host part of an address, lower-cased, or '' if malformed."""
    email = (email or "").strip().lower()
    if "@" not in email:
        return ""
    return email.rsplit("@", 1)[1]


def _min_levenshtein(domain: str) -> int:
    """Best (smallest) edit distance from `domain` to any official domain.

    Squatters register names one edit away from the real domain
    (dhsgsu-edu.in vs dhsgsu.edu.in); the smallest distance is the most
    dangerous signal, so we report that rather than an average.
    """
    if not domain:
        return 99
    return min(_levenshtein(domain, official) for official in OFFICIAL_DOMAINS)


def _levenshtein(a: str, b: str) -> int:
    """Classic Wagner-Fischer edit distance."""
    a, b = a.lower(), b.lower()
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,               # deletion
                    current[j - 1] + 1,            # insertion
                    previous[j - 1] + (char_a != char_b),  # substitution
                )
            )
        previous = current
    return previous[-1]


def _count_urls(text: str) -> int:
    return len(URL_PATTERN.findall(text or ""))


def _has_ip_url(text: str) -> float:
    """An IP address in a URL means no registered domain can be vetted.

    Legitimate university mail never links to a raw IP; this is one of the
    strongest single indicators of a phishing landing page.
    """
    return 1.0 if IP_PATTERN.search(text or "") else 0.0


def _has_http_only_url(text: str) -> float:
    """Plain http:// (no TLS) on a credential page is a deliberate downgrade.

    Official university portals are served over HTTPS; a link using only HTTP
    is consistent with a cheaply hosted fake page.
    """
    urls = URL_PATTERN.findall(text or "")
    if not urls:
        return 0.0
    return 1.0 if any(u.startswith("http://") for u in urls) else 0.0


def _has_blocklisted_url(text: str) -> float:
    lowered = (text or "").lower()
    return 1.0 if any(indicator in lowered for indicator in THREAT_INTEL_BLOCKLIST) else 0.0


def _keyword_hits(text: str, keywords: list[str]) -> int:
    lowered = (text or "").lower()
    return sum(1 for kw in keywords if kw in lowered)


def _count_urgency_keywords(text: str) -> int:
    """Urgency compresses the victim's deliberation window.

    Phishers flood messages with time-pressure words so the recipient acts
    before inspecting the sender address or the link destination.
    """
    return _keyword_hits(text, PHISHING_KEYWORDS["urgency"])


def _asks_for_password_or_otp(text: str) -> float:
    """Direct requests for secrets are outside normal university workflow.

    The university never asks for a password or OTP over email, so any such
    request is by definition suspicious. We count both single-word markers
    ("otp", "password") and multi-word phrases ("verify your password").
    """
    lowered = (text or "").lower()
    groups = PHISHING_KEYWORDS["credential"]
    for phrase in groups:
        if phrase in lowered:
            return 1.0
    # Fall back to single-word markers that the phrase scan may have missed.
    single = {"otp", "password", "passwd", "credential", "cvv", "ifsc",
              "secret", "pin"}
    for word in lowered.split():
        if word.strip(".,;:!?\"'") in single:
            return 1.0
    return 0.0


def _text_value(value: Any) -> str:
    return value if isinstance(value, str) else str(value or "")


def _bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def extract_features(email: dict[str, Any] | None) -> dict[str, float]:
    """Return the heuristic feature vector for one email as a plain dict.

    Used by the explanation engine to turn each active signal into a
    human-readable reason, and by tests that want to inspect a single row.
    """
    values = email if isinstance(email, dict) else {}
    sender_email = _text_value(values.get("sender_email"))
    body = _text_value(values.get("body"))
    url = _text_value(values.get("url"))
    text = f"{body}\n{url}"

    domain = _domain_of(sender_email)
    return {
        "sender_domain_is_official": 1.0 if domain in OFFICIAL_DOMAINS else 0.0,
        "levenshtein_distance_to_official_domain": float(_min_levenshtein(domain)),
        "url_count": float(_count_urls(text)),
        "has_ip_url": _has_ip_url(text),
        "has_http_only_url": _has_http_only_url(text),
        "urgency_keyword_count": float(_count_urgency_keywords(text)),
        "asks_for_password_or_otp": _asks_for_password_or_otp(text),
        "body_length": float(len(body)),
        "spf_pass": float(_bool_value(values.get("spf_pass", False))),
        "dkim_pass": float(_bool_value(values.get("dkim_pass", False))),
        "dmarc_pass": float(_bool_value(values.get("dmarc_pass", False))),
        "has_blocklisted_url": _has_blocklisted_url(text),
    }


def build_feature_frame_for_email(email: dict[str, Any] | None) -> pd.DataFrame:
    """Lift one email dict into the 9-column frame the preprocessor expects.

    The preprocessor consumes the fused frame, not the raw mail fields, so a
    single message must be expanded the same way the training corpus was.
    """
    values = email if isinstance(email, dict) else {}
    heuristics = extract_features(values)
    combined = (
        _text_value(values.get("subject"))
        + " "
        + _text_value(values.get("body"))
    )
    return pd.DataFrame([{TEXT_COLUMN: combined, **heuristics}])


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the full input matrix for the ColumnTransformer.

    Produces one text column (subject + body, which the vectorizer consumes)
    plus the eight heuristic columns. Keeping this in one function guarantees
    the training frame and any single-email frame share identical column names
    and ordering.
    """
    out = pd.DataFrame(index=df.index)
    out[TEXT_COLUMN] = (df["subject"].fillna("") + " " + df["body"].fillna(""))

    for row in df.itertuples():
        values = extract_features({
            "sender_email": getattr(row, "sender_email", ""),
            "body": getattr(row, "body", ""),
            "url": getattr(row, "url", ""),
            "spf_pass": getattr(row, "spf_pass", False),
            "dkim_pass": getattr(row, "dkim_pass", False),
            "dmarc_pass": getattr(row, "dmarc_pass", False),
            "has_blocklisted_url": getattr(row, "has_blocklisted_url", False),
        })
        for feature in HEURISTIC_FEATURES:
            out.loc[row.Index, feature] = values[feature]
    return out


def get_preprocessor(df: pd.DataFrame | None = None) -> ColumnTransformer:
    """Return the ColumnTransformer fusing the text and heuristic branches.

    When a DataFrame is supplied the preprocessor is fitted on it and returned
    in the fitted state; otherwise the unfitted transformer is returned so a
    `Pipeline` can fit it during training. Calling it twice on the same data is
    safe because fitting is deterministic (no stochastic estimator is used).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            # Text branch: TF-IDF n-grams over subject + body.
            # stop_words="english" removes high-frequency noise words that
            # carry no discriminative signal and would only dilute the
            # vocabulary budget of max_features.
            (
                "text",
                TfidfVectorizer(
                    max_features=TFIDF_MAX_FEATURES,
                    ngram_range=TFIDF_NGRAM_RANGE,
                    stop_words="english",
                    min_df=2,          # ignore tokens seen in only one mail
                    sublinear_tf=True, # damp repeated-keyword spamming
                ),
                # Pass the column as a single string, not a one-element list.
                # With a list, ColumnTransformer hands the vectorizer a
                # single-column DataFrame, and iterating a DataFrame yields
                # column names rather than documents -- so the vectorizer
                # would see exactly one "document" and fail.
                TEXT_COLUMN,
            ),
            # Heuristic branch: numeric flags pass through un-scaled.
            # HistGradientBoostingClassifier is tree-based and invariant to
            # monotone rescaling, so no StandardScaler is needed.
            ("heuristic", "passthrough", HEURISTIC_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
        # The text branch is sparse and the heuristic branch is dense; without
        # this, ColumnTransformer emits a sparse matrix.
        # HistGradientBoostingClassifier (and most explanation code) requires
        # dense input, so force a dense concatenation.
        sparse_threshold=0.0,
    )
    if df is not None:
        preprocessor.fit(build_feature_frame(df))
    return preprocessor


def _feature_transformer() -> FunctionTransformer:
    """Wrap `build_feature_frame` for use inside a sklearn Pipeline."""
    return FunctionTransformer(
        build_feature_frame, accept_sparse=False, validate=False
    )


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Human-readable names of every fused feature column."""
    return list(preprocessor.get_feature_names_out())


if __name__ == "__main__":
    # Self-check: the heuristic dict must be complete and float-typed.
    sample = {
        "sender_email": "registrar@dhsgsu.edu.in",
        "body": "URGENT: verify your account immediately. Click http://bad.in",
        "url": "http://bad.in",
    }
    print(extract_features(sample))
    print(feature_names(get_preprocessor()))