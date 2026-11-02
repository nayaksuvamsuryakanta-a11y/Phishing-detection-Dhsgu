"""
Branded web interface for DHSGSU PhishGuard.

Run with:  streamlit run app/streamlit_app.py

The sidebar displays the model metrics recorded at training time so every
verdict is presented alongside an auditable measure of how well the detector
is known to perform. Colour coding follows traffic-light convention
(green = safe, amber = caution, red = phishing) so the result is readable at
a glance by students whose first language may not be English.
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- PATH BOOTSTRAP (idempotent) ---
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[1]
if not (_PROJECT_ROOT / "src").is_dir() or not (_PROJECT_ROOT / "config.py").exists():
    raise RuntimeError(f"Cannot resolve project root from {_HERE}")
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# --- END PATH BOOTSTRAP ---

import matplotlib
import streamlit as st

PROJECT_ROOT = _PROJECT_ROOT

from config import (
    OFFICIAL_SENDERS,
    OFFICIAL_URL,
    UNIVERSITY_NAME,
    UNIVERSITY_SHORT,
)
from explain import (
    ModelNotTrainedError,
    generate_shap_explanation,
    load_metrics,
    predict_with_explanation,
)

matplotlib.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#E0E0E0",
    "text.color": "#1F2937",
})

st.set_page_config(
    page_title="DHSGSU PhishGuard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    header[data-testid="stHeader"] {
        background-color: transparent;
    }
    div[data-testid="stAppViewContainer"] {
        background: #FFFFFF;
        color: #1F2937;
        padding-top: 0.5rem;
    }
    section[data-testid="stSidebar"] {
        background: #E8F1FB;
        border-right: 1px solid #D3E3F5;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }
    div.stButton > button:first-child,
    button[kind="primary"] {
        background: #0D47A1;
        border: 1px solid #0D47A1;
        color: #FFFFFF;
        font-weight: 600;
        transition: background 0.2s ease, border-color 0.2s ease;
    }
    div.stButton > button:first-child:hover,
    button[kind="primary"]:hover {
        background: #1565C0;
        border-color: #1565C0;
        color: #FFFFFF;
    }
    .dhsgsu-banner {
        background: linear-gradient(90deg, #0D47A1 0%, #1E88E5 100%);
        border-radius: 0.75rem;
        color: #FFFFFF;
        margin: 0 0 1.5rem;
        padding: 1.35rem 1.6rem;
        box-shadow: 0 4px 12px rgba(13, 71, 161, 0.18);
    }
    .dhsgsu-banner h1 {
        font-size: 2rem;
        margin: 0;
    }
    .dhsgsu-banner p {
        margin: 0.35rem 0 0;
        opacity: 0.94;
    }
    .verdict-card {
        border-radius: 0.65rem;
        border: 1px solid #D3E3F5;
        border-left-width: 5px;
        margin: 0.5rem 0 1rem;
        padding: 0.85rem 1.1rem;
    }
    .verdict-card h2 {
        margin: 0;
    }
    .verdict-legit {
        background: #E8F5E9;
        border-left-color: #1B5E20;
        color: #1B5E20;
    }
    .verdict-suspicious {
        background: #FFF8E1;
        border-left-color: #B26A00;
        color: #B26A00;
    }
    .verdict-phish {
        background: #FFEBEE;
        border-left-color: #B71C1C;
        color: #B71C1C;
    }
    .dhsgsu-footer {
        color: #1F2937;
        font-size: 0.85rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

FEEDBACK_PATH = PROJECT_ROOT / "data" / "feedback.csv"


def render_verdict(result: dict) -> None:
    band = result["risk_band"]
    verdict_class = {
        "Low": "verdict-legit",
        "Medium": "verdict-suspicious",
        "High": "verdict-phish",
    }.get(band, "verdict-suspicious")
    icon = {"Low": "✅", "Medium": "⚠️", "High": "🚫"}.get(band, "•")
    st.markdown(
        f"<div class='verdict-card {verdict_class}'>"
        f"<h2>{icon} {result['label_name']}</h2>"
        f"</div>",
        unsafe_allow_html=True,
    )
    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_b:
        st.metric("Risk score", f"{result['risk_score']:.1f} / 100")
    with col_c:
        st.metric("Risk band", band)

    st.progress(result["risk_score"] / 100.0)

    with st.expander("Reasons", expanded=True):
        for reason in result["reasons"]:
            marker = "•"
            st.markdown(f"{marker} {reason}")

    if result["text_tokens"]:
        with st.expander("Text evidence"):
            st.caption("Words the model weighed most heavily:")
            st.markdown(
                ", ".join(f"`{t}`" for t in result["text_tokens"])
            )

    with st.expander("Raw signal values"):
        st.json(result["heuristics"])


def render_footer() -> None:
    st.divider()
    st.markdown(
        "<p class='dhsgsu-footer'>"
        f"<b>{UNIVERSITY_SHORT} PhishGuard</b> &mdash; "
        f"an experimental phishing detector for students and staff of "
        f"{UNIVERSITY_SHORT}. "
        f"Official university communications are sent only from "
        f"addresses ending in <code>@dhsgsu.edu.in</code>. "
        f"When in doubt, contact the university directly at "
        f"<a href='{OFFICIAL_URL}' target='_blank'>{OFFICIAL_URL}</a> "
        f"rather than replying to the message.</p>",
        unsafe_allow_html=True,
    )


def render_shap_and_feedback(email: dict, result: dict) -> None:
    st.subheader("Visual Explainability (SHAP)")
    try:
        import matplotlib.pyplot as plt
        import shap

        explanation = generate_shap_explanation(email)
        figure = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=10, show=False)
        st.pyplot(figure, clear_figure=True)
        plt.close(figure)
    except (ImportError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        st.warning(f"SHAP visualization is unavailable: {exc}")

    st.subheader("Report Misclassification (Active Learning)")
    actual_status = st.radio(
        "What is the ACTUAL status of this email?",
        ("Legitimate", "Phishing"),
        key="feedback_status",
    )
    if st.button("Submit Feedback", key="submit_feedback"):
        try:
            FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
            write_header = not FEEDBACK_PATH.exists()
            import csv
            from datetime import datetime, timezone

            with FEEDBACK_PATH.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "timestamp", "subject", "sender_email", "body", "url",
                        "model_prediction", "user_true_label", "risk_score",
                    ],
                )
                if write_header:
                    writer.writeheader()
                writer.writerow({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "subject": email.get("subject", ""),
                    "sender_email": email.get("sender_email", ""),
                    "body": email.get("body", ""),
                    "url": email.get("url", ""),
                    "model_prediction": result["label_name"],
                    "user_true_label": 1 if actual_status == "Phishing" else 0,
                    "risk_score": result["risk_score"],
                })
            st.success(
                "Thank you! Your feedback has been logged for the next "
                "model retraining cycle."
            )
        except OSError as exc:
            st.error(f"Unable to log feedback: {exc}")


def main() -> None:
    st.markdown(
        f"""
        <div class="dhsgsu-banner">
            <h1>🛡️ DHSGSU PhishGuard</h1>
            <p>Institutional email security for {UNIVERSITY_NAME}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"Classifies email as **LEGITIMATE** or **PHISHING** for "
        f"{UNIVERSITY_NAME}.",
    )

    with st.sidebar:
        st.sidebar.title("About")
        st.sidebar.markdown(
            f"PhishGuard analyses the language, sender address and links in a "
            f"message and returns a risk score with reasons.\n\n"
            f"**Trained on** synthetic email data generated to resemble "
            f"DHSGSU administrative mail and common university-themed "
            f"phishing attacks.\n\n"
            f"**Remember:** results are advisory. The university never asks "
            f"for a password or OTP over email."
        )

        st.sidebar.divider()
        st.sidebar.subheader("Model metrics")
        metrics = load_metrics()
        if metrics:
            st.sidebar.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            c1, c2 = st.sidebar.columns(2)
            c1.metric("Precision", f"{metrics['precision']:.2%}")
            c2.metric("Recall", f"{metrics['recall']:.2%}")
            st.sidebar.metric("F1-score", f"{metrics['f1']:.2%}")
            st.sidebar.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
            st.sidebar.caption(
                f"Test set: {metrics.get('n_test', '?')} emails "
                f"({metrics.get('model', '?')})."
            )
        else:
            st.sidebar.warning(
                "No metrics found. Run `python src/train.py` first to train "
                "the model and write metrics.json."
            )

        st.sidebar.divider()
        st.sidebar.subheader("Verified senders")
        for address, role in OFFICIAL_SENDERS.items():
            st.sidebar.caption(f"**{role}** — {address}")

    with st.form("email_form"):
        subject = st.text_input("Subject",
                                value="URGENT: Verify Your DHSGSU Portal Account")
        sender_name = st.text_input("Sender Name", value="Registrar")
        sender_email = st.text_input(
            "Sender Email", value="registrar@dhsgsu-edu.in"
        )
        body = st.text_area(
            "Body",
            value=(
                "Dear Student,\n\n"
                "Our security system has detected unusual login activity on "
                "your student portal account. To prevent permanent "
                "deactivation, you must verify your account within the next "
                "24 hours.\n\n"
                "Click the link below to verify your credentials:\n"
                "https://dhsgsu-edu.in/verify?id=4821\n\n"
                "Failure to verify will result in suspension of your portal "
                "access and examination services.\n\n"
                "Regards,\n"
                "IT Support Desk"
            ),
            height=180,
        )
        url = st.text_input(
            "URL", value="https://dhsgsu-edu.in/verify?id=4821"
        )
        spf_pass = st.checkbox("SPF Pass", value=True)
        dkim_pass = st.checkbox("DKIM Pass", value=True)
        dmarc_pass = st.checkbox("DMARC Pass", value=True)
        submitted = st.form_submit_button("🔍 Analyse", type="primary")

    if submitted:
        email = {
            "subject": subject,
            "sender_name": sender_name,
            "sender_email": sender_email,
            "body": body,
            "url": url,
            "spf_pass": spf_pass,
            "dkim_pass": dkim_pass,
            "dmarc_pass": dmarc_pass,
        }
        if not any(email.get(key) for key in
                   ("subject", "sender_name", "sender_email", "body", "url")):
            st.error("Please enter at least a subject or a body.")
            return
        try:
            result = predict_with_explanation(email)
        except (ModelNotTrainedError, ValueError, TypeError, KeyError) as exc:
            st.error(f"Unable to analyse this email: {exc}")
            st.info("Train the model first: `python src/train.py`")
            return

        st.session_state["analysis_email"] = email
        st.session_state["analysis_result"] = result

    email = st.session_state.get("analysis_email")
    result = st.session_state.get("analysis_result")
    if email and result:
        render_verdict(result)
        render_shap_and_feedback(email, result)

    render_footer()


if __name__ == "__main__":
    main()