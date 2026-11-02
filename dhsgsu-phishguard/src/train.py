"""
Training and evaluation pipeline for DHSGSU PhishGuard.

Reads the synthetic corpus, fits a single sklearn Pipeline whose first stage
is the ColumnTransformer from src/features.py, and writes three artefacts:

  * models/phishguard_model.joblib  -- the fitted pipeline, loadable by
    src/explain.py and app/streamlit_app.py.
  * metrics.json                    -- accuracy, precision, recall, F1 and
    ROC-AUC, consumed by the Streamlit sidebar.
  * reports/confusion_matrix.png / reports/roc_curve.png -- figures for the
    BCA project report.

The model is a HistGradientBoostingClassifier: it handles the mixed
text-and-numeric feature matrix natively, needs no feature scaling, and
captures interactions such as "urgent language AND a non-official sender
domain" that a linear model would only see if an interaction term were added
by hand.
"""

from __future__ import annotations

import json
import sys
import warnings
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

PROJECT_ROOT = _PROJECT_ROOT

import matplotlib

matplotlib.use("Agg")  # headless: no display server on the build machine

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)

from config import (
    METRICS_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    SEED,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
)
from explain import load_metrics
from features import (
    DOMAIN_FEATURES,
    HEURISTIC_FEATURES,
    TEXT_COLUMN,
    build_feature_frame,
    get_preprocessor,
)

DATA_PATH = PROJECT_ROOT / "data" / "emails.csv"
MODEL_PATH_ABS = PROJECT_ROOT / MODEL_PATH
METRICS_PATH_ABS = PROJECT_ROOT / METRICS_PATH
REPORTS_DIR_ABS = PROJECT_ROOT / REPORTS_DIR


def load_corpus(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Generate it first with "
            "`python data/generate_dataset.py`."
        )
    df = pd.read_csv(path)
    required = {
        "subject", "sender_name", "sender_email", "body", "url",
        "spf_pass", "dkim_pass", "dmarc_pass", "has_blocklisted_url", "label",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {sorted(missing)}")
    return df


def build_pipeline() -> Pipeline:
    """Assemble the fitted pipeline.

    The text + heuristic frame is built once, up front, from the raw columns;
    the pipeline then holds only the ColumnTransformer and the classifier.
    Step names 'preprocessor' and 'classifier' are referenced by
    src/explain.py, so renaming them must be mirrored there.
    """
    return Pipeline(
        steps=[
            ("preprocessor", get_preprocessor()),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.08,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    random_state=SEED,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=15,
                ),
            ),
        ]
    )


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          out: Path) -> None:
    labels = ["Legitimate", "Phishing"]
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix -- DHSGSU PhishGuard")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, out: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2,
            label=f"PhishGuard (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--",
            label="Random classifier")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC Curve -- DHSGSU PhishGuard")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_feature_importance(pipeline: Pipeline, out: Path,
                            top_n: int = 15,
                            importances: np.ndarray | None = None,
                            names: list[str] | None = None) -> None:
    """Bar chart of the strongest features, for the project report.

    `importances`/`names` are supplied by main(), which resolves them with
    permutation importance -- the only form that works for
    HistGradientBoostingClassifier, which exposes neither feature_importances_
    nor coef_. When omitted the function falls back to those attributes for
    classifiers that do provide them.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    if names is None:
        names = [str(n) for n in preprocessor.get_feature_names_out()]
    if importances is None:
        if hasattr(classifier, "feature_importances_"):
            importances = np.asarray(classifier.feature_importances_,
                                     dtype=float)
        elif hasattr(classifier, "coef_"):
            importances = np.abs(np.asarray(classifier.coef_[0], dtype=float))
        else:
            return

    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x=importances[order], y=[names[i] for i in order],
        color="#4c72b0", ax=ax,
    )
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def run_ablation(pipeline: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_train: np.ndarray,
                 y_test: np.ndarray) -> dict[str, dict[str, float]]:
    """Measure what the whitelist rule and the ML layer each contribute.

    On this corpus the sender domain is a perfect separator by construction,
    so headline accuracy would overstate the classifier. This study reports
    (a) the domain whitelist alone, (b) the classifier with the domain signal
    removed so it must judge on wording and URL structure, and (c) the full
    pipeline for reference.
    """
    results: dict[str, dict[str, float]] = {}

    # (a) Domain whitelist alone: anything not from dhsgsu.edu.in is phishing.
    domain_only = (1.0 - X_test["sender_domain_is_official"].to_numpy())
    results["domain_whitelist_only"] = {
        "accuracy": float(accuracy_score(y_test, domain_only)),
        "precision": float(precision_score(y_test, domain_only, zero_division=0)),
        "recall": float(recall_score(y_test, domain_only, zero_division=0)),
        "f1": float(f1_score(y_test, domain_only, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, domain_only)),
    }

    # (b) Classifier with the domain signal stripped out. It must now rely on
    #     urgency, credential requests, IP links and text -- which is the
    #     genuinely hard part of the problem.
    kept_heuristics = [h for h in HEURISTIC_FEATURES
                       if h not in DOMAIN_FEATURES]
    ablation_preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=TFIDF_MAX_FEATURES,
                                     ngram_range=TFIDF_NGRAM_RANGE,
                                     stop_words="english", min_df=2,
                                     sublinear_tf=True), TEXT_COLUMN),
            ("heuristic", "passthrough", kept_heuristics),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    no_domain_pipeline = Pipeline([
        ("preprocessor", ablation_preprocessor),
        ("classifier", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=SEED,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=15)),
    ])
    no_domain_pipeline.fit(X_train, y_train)
    y_pred = no_domain_pipeline.predict(X_test)
    y_proba = no_domain_pipeline.predict_proba(X_test)[:, 1]
    results["classifier_without_domain_signal"] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    # (c) Full pipeline, for reference (repeats metrics.json for a direct
    # side-by-side comparison in the ablation table).
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    results["full_pipeline"] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    return results


def main() -> None:
    df = load_corpus()
    print(f"Loaded {len(df)} emails from {DATA_PATH}")
    print(f"  legitimate: {(df['label'] == 0).sum()}   "
          f"phishing: {(df['label'] == 1).sum()}")

    y = df["label"].to_numpy()

    # Build the fused text + heuristic frame once, from the raw columns, then
    # split it. The frame is what the ColumnTransformer consumes, so the split
    # must happen *after* frame construction to keep train and test aligned.
    X_frame = build_feature_frame(df)

    # Stratified split preserves the 50/50 class balance in both halves,
    # which matters because accuracy on an imbalanced split would be
    # misleading for a security detector.
    X_train, X_test, y_train, y_test = train_test_split(
        X_frame, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")
    print(f"Features: {len(X_frame.columns)} "
          f"({sum(1 for c in X_frame.columns if c != TEXT_COLUMN)} heuristic, "
          f"1 text)")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "random_state": SEED,
        "model": "HistGradientBoostingClassifier",
    }

    REPORTS_DIR_ABS.mkdir(parents=True, exist_ok=True)
    MODEL_PATH_ABS.parent.mkdir(parents=True, exist_ok=True)

    # HistGradientBoostingClassifier does not expose feature_importances_,
    # so measure importance by permutation instead: each column is shuffled
    # and the drop in the chosen score records how much the model relied on it.
    # Negative log-loss is used rather than accuracy or ROC-AUC because on this
    # synthetic corpus both saturate at their ceiling, which would collapse
    # every permutation score to zero and make the chart meaningless; log-loss
    # measures probability degradation and stays sensitive even when the
    # ranking of predictions is already perfect.
    from sklearn.inspection import permutation_importance

    preprocessor = pipeline.named_steps["preprocessor"]
    names = [str(n) for n in preprocessor.get_feature_names_out()]
    permutation = permutation_importance(
        pipeline, X_test, y_test, n_repeats=10,
        random_state=SEED, scoring="neg_log_loss",
    )
    importances = np.asarray(permutation.importances_mean, dtype=float)

    plot_confusion_matrix(y_test, y_pred,
                          REPORTS_DIR_ABS / "confusion_matrix.png")
    plot_roc_curve(y_test, y_proba, REPORTS_DIR_ABS / "roc_curve.png")
    plot_feature_importance(
        pipeline, REPORTS_DIR_ABS / "feature_importance.png",
        importances=importances, names=names,
    )

    dump(pipeline, MODEL_PATH_ABS)
    with METRICS_PATH_ABS.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print("\n=== Evaluation on held-out test set ===")
    for key, value in metrics.items():
        print(f"  {key:<14}: {value}")

    order = np.argsort(importances)[::-1][:8]
    print("\nTop features (resolved via preprocessor.get_feature_names_out()):")
    for idx in order:
        print(f"  {names[idx]:<42} {importances[idx]:.4f}")

    print(f"\nSaved model      -> {MODEL_PATH_ABS}")
    print(f"Saved metrics    -> {METRICS_PATH_ABS}")
    print(f"Saved charts     -> {REPORTS_DIR_ABS}")

    # Ablation study: separate the whitelist rule from the ML layer.
    # On this corpus the sender domain is a perfect separator by construction,
    # so the headline metrics would overstate the classifier's contribution.
    ablation = run_ablation(pipeline, X_train, X_test, y_train, y_test)
    ablation_path = REPORTS_DIR_ABS / "ablation.json"
    with ablation_path.open("w", encoding="utf-8") as fh:
        json.dump(ablation, fh, indent=2)
    print("\n=== Ablation (what each component contributes) ===")
    for label, result in ablation.items():
        print(f"  {label:<34} accuracy={result['accuracy']:.3f} "
              f"f1={result['f1']:.3f} auc={result['roc_auc']:.3f}")
    print(f"Saved ablation -> {ablation_path}")


if __name__ == "__main__":
    main()