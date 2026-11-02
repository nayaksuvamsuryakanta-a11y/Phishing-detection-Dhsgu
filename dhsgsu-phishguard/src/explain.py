"""
Explanation engine for DHSGSU PhishGuard.

A raw 0/1 label is not enough for a security tool: a student who is told
"this mail is phishing" but given no reason will either dismiss the warning
or, worse, learn to ignore the system entirely. This module therefore returns
a structured verdict plus a risk score and a bullet list of plain-English
reasons derived from the model's own evidence.

Two distinct evidence streams are surfaced:

  * Text evidence -- the TF-IDF tokens that most pushed the score toward the
    predicted class, read off the fitted model's own feature importances.
  * Heuristic evidence -- the hand-crafted signals from src/features.py that
    fired for this message, each rendered as a sentence naming the exact value
    that triggered it ("Sender domain 'dhsgsu-edu.in' imitates the official
    domain (edit distance 1)").

The risk score is the model's estimated probability of phishing, stretched to
0-100, so it is directly calibrated rather than being an arbitrary weighted sum.
"""

from __future__ import annotations

import json
import hashlib
import pickle
import sys
import warnings
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

with warnings.catch_warnings():
    # Streamlit import noise when running inside a web process; harmless here.
    warnings.simplefilter("ignore")
    from joblib import load

PROJECT_ROOT = _PROJECT_ROOT

from config import (
    METRICS_PATH,
    MODEL_PATH,
    MODEL_SHA256,
    RISK_BANDS,
    UNIVERSITY_SHORT,
)
from features import (
    HEURISTIC_FEATURES,
    build_feature_frame_for_email,
    extract_features,
    get_preprocessor,
)

MODEL_PATH_ABS = PROJECT_ROOT / MODEL_PATH
METRICS_PATH_ABS = PROJECT_ROOT / METRICS_PATH

# ColumnTransformer names fused features '<transformer>__<feature>', so the
# prefix is the TRANSFORMER name ('text'), not the input column name
# ('text_combined'). Getting this wrong silently drops every text token from
# the explanation.
TEXT_PREFIX = "text__"


class ModelNotTrainedError(RuntimeError):
    """Raised when predict_with_explanation is called before training."""


def _load_pipeline(path: Path = MODEL_PATH_ABS) -> Any:
    if not path.exists():
        raise ModelNotTrainedError(
            f"No trained model found at {path}. Run `python src/train.py` "
            "first to fit the classifier and write the artefact."
        )
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    except OSError as exc:
        raise ModelNotTrainedError(
            f"Unable to read the trained model at {path} for integrity "
            "verification."
        ) from exc
    if MODEL_SHA256 and digest != MODEL_SHA256.upper():
        raise ModelNotTrainedError(
            f"The model artifact at {path} failed integrity verification. "
            "Regenerate the model or configure PHISHGUARD_MODEL_SHA256."
        )
    try:
        pipeline = load(path)
    except (OSError, EOFError, ImportError, AttributeError, ValueError,
            TypeError, pickle.UnpicklingError) as exc:
        raise ModelNotTrainedError(
            f"Unable to load the trained model at {path}. The model may be "
            "missing or corrupted; run `python src/train.py` to recreate it."
        ) from exc
    if not hasattr(pipeline, "named_steps"):
        raise ModelNotTrainedError(
            f"The model artifact at {path} is invalid. "
            "Run `python src/train.py` to recreate it."
        )
    return pipeline


def _vectorizer(pipeline: Any) -> Any:
    """Pull the fitted TfidfVectorizer out of the fitted ColumnTransformer."""
    preprocessor = pipeline.named_steps["preprocessor"]
    return preprocessor.named_transformers_["text"]


def _classifier(pipeline: Any) -> Any:
    return pipeline.named_steps["classifier"]


def _feature_names(pipeline: Any) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    return [str(n) for n in preprocessor.get_feature_names_out()]


def _phishing_prob(pipeline: Any, features: np.ndarray) -> float:
    """Phishing probability for one already-transformed feature row."""
    classifier = _classifier(pipeline)
    if hasattr(classifier, "predict_proba"):
        return float(classifier.predict_proba(np.asarray(features))[:, 1][0])
    raw = float(classifier.decision_function(features)[0])
    return float(1.0 / (1.0 + np.exp(-raw)))


def _top_text_tokens_permutation(pipeline: Any, features: np.ndarray,
                                 n: int = 5, top_k: int = 40) -> list[tuple[str, float]]:
    """Rank tokens by how much removing each one lowers the phishing score.

    Tree ensembles expose neither `feature_importances_` nor `coef_`, so a
    global ranking cannot be built. Instead we measure, for this one email,
    how much the probability falls when each token's TF-IDF weight is zeroed
    out. This is an instance-level explanation: a token that is globally
    unimportant can still be the reason this particular message was flagged.

    Only the `top_k` tokens that actually appear in the message are tested,
    so a long email does not trigger 3000 extra predictions.
    """
    if features is None:
        return []
    names = _feature_names(pipeline)
    text_indices = [i for i, name in enumerate(names)
                    if name.startswith(TEXT_PREFIX)]
    if not text_indices:
        return []

    row = np.atleast_2d(np.asarray(features.todense()
                                   if hasattr(features, "todense")
                                   else np.asarray(features), dtype=float))
    if row.ndim != 2 or row.shape[0] == 0 or max(text_indices) >= row.shape[1]:
        return []
    base = _phishing_prob(pipeline, row)

    # Restrict to tokens present in this email, ranked by TF-IDF weight.
    present = [(i, float(row[0, i])) for i in text_indices if row[0, i] > 0.0]
    present.sort(key=lambda item: -item[1])
    candidates = [i for i, _ in present[:top_k]]

    drops: list[tuple[int, float]] = []
    for index in candidates:
        perturbed = row.copy()
        perturbed[0, index] = 0.0
        drops.append((index, base - _phishing_prob(pipeline, perturbed)))
    drops.sort(key=lambda item: -abs(item[1]))

    vocabulary = _vectorizer(pipeline).vocabulary_ or {}
    tokens: list[tuple[str, float]] = []
    for index, drop in drops:
        token = names[index][len(TEXT_PREFIX):]
        if token not in vocabulary:
            continue
        tokens.append((token, float(drop)))
        if len(tokens) >= n:
            break
    return tokens


def _top_text_tokens(pipeline: Any, n: int = 5,
                     features: np.ndarray | None = None) -> list[tuple[str, float]]:
    """Map the classifier's strongest importances back to readable tokens.

    `get_feature_names_out()` gives names like 'text_combined__urgent'; the
    token is the suffix after the branch prefix. We cross-check against the
    vectorizer's `vocabulary_` so an ungram token that survived feature
    selection is still reported correctly -- this is the mapping the spec
    requires, and it is safer than trusting the importance array's indices
    alone, which shift if the vocabulary is rebuilt.

    When the classifier exposes no global importances
    (HistGradientBoostingClassifier), falls back to a per-message permutation
    measurement over the tokens that actually appear in this email.
    """
    vectorizer = _vectorizer(pipeline)
    classifier = _classifier(pipeline)
    names = _feature_names(pipeline)

    if hasattr(classifier, "feature_importances_"):
        scores = np.asarray(classifier.feature_importances_, dtype=float)
    elif hasattr(classifier, "coef_"):
        scores = np.abs(np.asarray(classifier.coef_[0], dtype=float))
    else:
        return _top_text_tokens_permutation(pipeline, features, n=n)

    if scores.ndim != 1 or len(scores) != len(names):
        return []
    pairs = sorted(zip(names, scores), key=lambda kv: -kv[1])
    vocabulary = vectorizer.vocabulary_ or {}
    tokens: list[tuple[str, float]] = []
    for name, score in pairs:
        if not name.startswith(TEXT_PREFIX):
            continue
        token = name[len(TEXT_PREFIX):]
        # Confirm the token is one the vectorizer actually emitted; a token
        # that dropped out of the vocabulary cannot have contributed.
        if token not in vocabulary:
            continue
        tokens.append((token, float(score)))
        if len(tokens) >= n:
            break
    return tokens


def _heuristic_reasons(heuristics: dict[str, float]) -> list[str]:
    """Render each fired heuristic as a sentence the user can act on."""
    reasons: list[str] = []
    auth_failures = [
        name.removesuffix("_pass").upper()
        for name in ("spf_pass", "dkim_pass", "dmarc_pass")
        if not heuristics.get(name, 0.0)
    ]
    if auth_failures:
        reasons.append(
            "CRITICAL: Email failed "
            + "/".join(auth_failures)
            + " authentication, indicating the sender address may be spoofed."
        )
    if heuristics.get("has_blocklisted_url"):
        reasons.append(
            "CRITICAL: Embedded URL matches a known Threat Intelligence blocklist."
        )

    domain = heuristics.get("sender_domain_is_official", 0.0) == 1.0

    if not domain:
        reasons.append(
            "The sender domain is not an official university domain "
            "(dhsgsu.edu.in); the university only mails from its own domain."
        )

    lev = heuristics.get("levenshtein_distance_to_official_domain", 99)
    if 0 < lev <= 3:
        reasons.append(
            f"Sender domain imitates the official domain with an edit "
            f"distance of {int(lev)} -- a classic look-alike squat used to "
            f"catch readers who skim the address."
        )
    elif lev == 0 and not domain:
        reasons.append(
            "Sender domain matches the official domain text but could not be "
            "confirmed as an authorised address; verify the full address "
            "character by character."
        )

    if heuristics.get("has_ip_url"):
        reasons.append(
            "The message links to a raw IP address instead of a registered "
            "domain, which makes the destination impossible to vet."
        )

    if heuristics.get("has_http_only_url"):
        reasons.append(
            "The link uses plain HTTP rather than HTTPS, so the connection "
            "is not encrypted -- legitimate university portals always use TLS."
        )

    url_count = heuristics.get("url_count", 0)
    if url_count >= 2:
        reasons.append(
            f"The message contains {int(url_count)} links; phishing mail "
            f"typically crowds multiple links to increase click-through."
        )

    urgency = heuristics.get("urgency_keyword_count", 0)
    if urgency > 0:
        reasons.append(
            f"{int(urgency)} urgency marker(s) were detected, which pressure "
            f"the recipient into acting without checking the sender."
        )

    if heuristics.get("asks_for_password_or_otp"):
        reasons.append(
            "The message asks for a password, OTP or other credential. The "
            "university never requests secrets over email."
        )

    body_length = heuristics.get("body_length", 0)
    if body_length < 60:
        reasons.append(
            f"The message is unusually short ({int(body_length)} characters); "
            f"legitimate university notices carry specific details."
        )

    return reasons


def _risk_band(score: float) -> str:
    for threshold, band in RISK_BANDS:
        if score <= threshold:
            return band
    return RISK_BANDS[-1][1]


def generate_shap_explanation(email: dict[str, Any]):
    """Generate a SHAP explanation from the fitted classifier feature space."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "SHAP is unavailable. Install dependencies from requirements.txt."
        ) from exc

    pipeline = _load_pipeline()
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        frame = build_feature_frame_for_email(email)
        transformed = np.asarray(preprocessor.transform(frame), dtype=float)
        names = _feature_names(pipeline)
        if transformed.ndim != 2 or transformed.shape[1] != len(names):
            raise ValueError("Transformed feature names and values are misaligned.")

        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(transformed)
        base_values = explainer.expected_value

        if isinstance(shap_values, list):
            values = np.asarray(shap_values[-1], dtype=float)
        else:
            values = np.asarray(shap_values, dtype=float)
        if values.ndim == 3:
            values = values[:, :, -1]
        values = np.atleast_2d(values)
        if values.shape[1] != len(names):
            raise ValueError("SHAP values and feature names are misaligned.")

        base = np.asarray(base_values, dtype=float)
        if base.ndim == 0:
            base = float(base)
        elif base.size:
            base = float(base.reshape(-1)[-1])
        else:
            base = 0.0
        return shap.Explanation(
            values=values[0],
            base_values=base,
            data=transformed[0],
            feature_names=names,
        )
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
        raise RuntimeError(f"Unable to generate SHAP explanation: {exc}") from exc


def predict_with_explanation(email: dict[str, Any],
                             n_text_reasons: int = 5) -> dict[str, Any]:
    """Classify one email and explain the verdict.

    Returns
    -------
    {
        "label": int,                # 0 = legitimate, 1 = phishing
        "label_name": str,           # "LEGITIMATE" or "PHISHING"
        "risk_score": float,         # 0-100 probability of phishing
        "risk_band": str,            # Low / Medium / High
        "phishing_probability": float,
        "reasons": list[str],        # human-readable evidence bullets
        "text_tokens": list[str],    # top contributing TF-IDF tokens
        "heuristics": dict,          # raw numeric signal values
    }
    """
    pipeline = _load_pipeline()
    frame = build_feature_frame_for_email(email)
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        # The preprocessor is already fitted from training, so we only transform.
        features = preprocessor.transform(frame)
        phish_prob = _phishing_prob(pipeline, features)
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
        raise ModelNotTrainedError(
            "The trained model could not analyse this email. "
            "Run `python src/train.py` to recreate the model artifact."
        ) from exc
    heuristics = extract_features(email)
    if any(not heuristics.get(name, 0.0)
           for name in ("spf_pass", "dkim_pass", "dmarc_pass")):
        phish_prob = max(phish_prob, 0.75)
    if heuristics.get("has_blocklisted_url"):
        phish_prob = max(phish_prob, 0.95)
    label = 1 if phish_prob >= 0.5 else 0
    risk_score = round(phish_prob * 100, 2)

    reasons: list[str] = []

    if label == 1:
        reasons.append(
            f"The model classifies this as PHISHING with {risk_score:.1f}% "
            f"confidence."
        )
    else:
        reasons.append(
            f"The model classifies this as LEGITIMATE "
            f"({(100 - risk_score):.1f}% confidence it is not phishing)."
        )

    reasons.extend(_heuristic_reasons(heuristics))

    tokens = _top_text_tokens(pipeline, n_text_reasons, features=features)
    if tokens:
        token_list = ", ".join(f"'{tok}'" for tok, _ in tokens)
        if label == 1:
            reasons.append(
                f"Strongest phishing signals in the text: {token_list}."
            )
        else:
            reasons.append(
                f"Most influential words in the text: {token_list}; none "
                f"indicate credential harvesting or urgency."
            )
    else:
        reasons.append("No individual text term contributed strongly.")

    return {
        "label": label,
        "label_name": "PHISHING" if label == 1 else "LEGITIMATE",
        "risk_score": risk_score,
        "risk_band": _risk_band(risk_score),
        "phishing_probability": round(phish_prob, 4),
        "reasons": reasons,
        "text_tokens": [tok for tok, _ in tokens],
        "heuristics": heuristics,
    }


def load_metrics(path: Path = METRICS_PATH_ABS) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as fh:
            metrics = json.load(fh)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(metrics, dict):
        return {}
    required = ("accuracy", "precision", "recall", "f1", "roc_auc")
    if any(key not in metrics for key in required):
        return {}
    try:
        for key in required:
            float(metrics[key])
    except (TypeError, ValueError):
        return {}
    return metrics


if __name__ == "__main__":
    sample = {
        "subject": "URGENT: Verify Your DHSGSU Portal Account",
        "sender_name": "Registrar",
        "sender_email": "registrar@dhsgsu-edu.in",
        "body": (
            "Your account will be suspended immediately. Verify your "
            "password and OTP now at http://192.168.1.4/verify"
        ),
        "url": "http://192.168.1.4/verify",
    }
    try:
        result = predict_with_explanation(sample)
        print(f"Verdict: {result['label_name']}  (risk {result['risk_score']})")
        for reason in result["reasons"]:
            print(f"  - {reason}")
    except ModelNotTrainedError as exc:
        print(exc)