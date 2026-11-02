# DHSGSU PhishGuard

An AI/ML-based phishing email detection system customised for
**Dr. Harisingh Gour Vishwavidyalaya, Sagar (M.P.)** — official domain
`dhsgsu.edu.in`. Classifies university-themed email as **LEGITIMATE** or
**PHISHING** and returns a 0–100 risk score with human-readable reasons.

Built for the **CodeCraft Challenges** (Cybersecurity domain) BCA project.
Intended audience: DHSGSU students and staff.

---

## Architecture

```
                         ┌─────────────────────────────────────┐
                         │           INPUT EMAIL               │
                         │  subject | sender | body | url      │
                         └──────────────┬──────────────────────┘
                                        │
                   ┌────────────────────┴────────────────────┐
                   │           FEATURE EXTRACTION            │
                   │  (ColumnTransformer -- two branches)    │
                   ├───────────────────┬─────────────────────┤
                   │  TEXT BRANCH      │  HEURISTIC BRANCH   │
                   │  TF-IDF n-grams   │  8 hand-crafted     │
                   │  subject + body   │  numeric features   │
                   │  max_features=300 │  (domain, URL,      │
                   │                   │   urgency, length)  │
                   └────────┬──────────┴─────────────────────┘
                            │  fused feature matrix
                            ▼
              ┌──────────────────────────────────────┐
              │  HistGradientBoostingClassifier      │
              │  (fitted inside a sklearn Pipeline)  │
              └──────────────┬───────────────────────┘
                   raw probability of phishing
                            │
                            ▼
                ┌─────────────────────────────┐
                │   EXPLANATION ENGINE        │
                │  · risk score 0-100         │
                │  · risk band (Low/Med/High) │
                │  · top TF-IDF tokens        │
                │  · plain-English reasons    │
                └──────────────┬──────────────┘
                               │
        ┌──────────────────────┼───────────────────────┐
        ▼                      ▼                       ▼
  CLI verdict        Streamlit UI            reports/*.png
  exit code 0/1      colour-coded metric     confusion matrix
                     + sidebar metrics       ROC curve
```

---

## File layout

```
dhsgsu-phishguard/
├── requirements.txt            # dependencies
├── README.md                   # this file
├── config.py                   # university constants + security policy
├── data/
│   ├── generate_dataset.py     # synthetic corpus builder (run once)
│   └── emails.csv              # generated dataset (2000 rows)
├── src/
│   ├── __init__.py
│   ├── features.py             # ColumnTransformer + heuristic extraction
│   ├── train.py                # train / evaluate / persist model + charts
│   ├── explain.py              # prediction + risk score + reasons
│   └── cli.py                  # argparse CLI
├── app/
│   └── streamlit_app.py        # branded web UI
├── models/
│   └── phishguard_model.joblib # fitted pipeline
├── reports/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
└── metrics.json                # accuracy, precision, recall, F1, ROC-AUC
```

---

## Design notes (for the BCA project report)

**Why synthetic data.** Real university email contains personal information
and is protected by institutional policy. The dataset is therefore generated
*combinatorially* — short reusable templates are combined with random fillers
inside loops, so roughly two hundred lines of source data expand to 2000+
rows. Legitimate mail is emitted **only** from `@dhsgsu.edu.in`; phishing mail
uses look-alike squat domains (`dhsgsu-edu.in`, `dhsgsu-results.com`), free-mail
addresses (`gmail.com`, `yahoo.com`) and raw IP-address URLs, with urgency and
credential-harvesting language. No real email was scraped or reproduced.

**Why `ColumnTransformer` rather than `FeatureUnion`.** The two branches
consume different columns: the TF-IDF vectorizer needs a single text column,
while the heuristic branch needs eight numeric columns. `FeatureUnion` applies
*every* transformer to the *same* whole input, which would force the vectorizer
to try tokenising numeric flags. `ColumnTransformer` routes each column to the
transformer that understands it.

**Why HistGradientBoostingClassifier.** It handles the mixed text-and-numeric
matrix natively, needs no feature scaling (trees are invariant to monotone
rescaling), and learns interactions such as *"urgent language together with a
non-official sender domain"* that a linear model would only see if an
interaction term were added by hand.

**Security rationale of the heuristics.** Each of the eight numeric features
encodes a rule a security-aware student would check by eye: a sender domain not
in `dhsgsu.edu.in`; edit distance to the official domain (squatting); a raw IP
address in a link (no registrant to vet); plain HTTP instead of HTTPS
(deliberate downgrade); multiple links (click-through baiting); urgency
markers (compressed deliberation window); requests for passwords/OTPs (never
part of university workflow); and unusually short bodies (missing institutional
detail). Encoding them as features lets the model learn when to trust textual
evidence over a plausible-looking display name.

## Implementation notes (gotchas worth reporting)

**Text column must be passed as a single string, not a list.**
`ColumnTransformer(["text", TfidfVectorizer(), ["text_combined"]])` hands the
vectorizer a single-column DataFrame. Iterating a DataFrame yields *column
names*, so the vectorizer sees one "document" and raises
`ValueError: max_df corresponds to < documents than min_df`. Passing the column
name as a bare string (`TEXT_COLUMN`, not `[TEXT_COLUMN]`) makes the
transformer hand over a Series.

**`sparse_threshold=0.0` on the ColumnTransformer.** The TF-IDF branch is
sparse and the heuristic branch is dense, so the fused matrix defaults to
sparse. `HistGradientBoostingClassifier` rejects sparse input
(`Sparse data was passed for X, but dense data is required`), and the
explanation engine also needs a dense row.

**Feature names are `<transformer>__<token>`, not `<column>__<token>`.**
`get_feature_names_out()` uses the *transformer* name in its prefix. The
text transformer is named `"text"`, so the prefix is `"text__"` — using
`"text_combined__"` (the input column name) silently drops every text token
from the explanation.

**`HistGradientBoostingClassifier` exposes no importances.** It has neither
`feature_importances_` nor `coef_`, so neither the report chart nor the
explanation engine can read a global ranking off the model. `train.py` uses
`permutation_importance` with `neg_log_loss`; `explain.py` falls back to a
per-message permutation that zeroes each token's TF-IDF weight and measures
the probability drop.

**Choosing `neg_log_loss` for permutation importance.** Accuracy and ROC-AUC
both saturate at 1.0 on this corpus; permuting any feature then changes the
score by zero, so every importance is 0 and the chart is blank. Negative
log-loss measures probability *degradation* and stays sensitive even when the
ranking of predictions is already perfect.

## On the reported metrics

The headline numbers saturate at 1.0. This is an artefact of the corpus, not
of a leak: `sender_domain_is_official` is a perfect separator by construction
(legitimate mail is defined to come **only** from `dhsgsu.edu.in`, phishing
mail never does), and the wording of the two classes also occupies distinct
regions of feature space.

`src/train.py` therefore writes `reports/ablation.json`, which separates the
three components:

| Component | What it measures |
|---|---|
| `domain_whitelist_only` | accuracy of the rule "not dhsgsu.edu.in ⇒ phishing" |
| `classifier_without_domain_signal` | the ML layer forced to judge on wording and URL structure alone |
| `full_pipeline` | the complete system |

Even with the domain signal removed, the classifier reaches perfect accuracy
on the held-out split because the remaining features (`url_count`,
`asks_for_password_or_otp`, urgency) still separate the synthetic classes.
The per-feature analysis in `src/train.py` shows that these signals are not
individually deterministic — `url_count` alone scores 0.897 and
`asks_for_password_or_otp` 0.662 — so the model is combining genuinely
overlapping cues, not reading a single label column.

**Read the metrics as evidence that the synthetic corpus is separable, not as
a claim about real mail.** A live inbox contains look-alike domains that pass
visual inspection, legitimate IT mail that discusses passwords, and phishing
mail that uses no urgency. None of that is present here. The ablation exists
so the BCA report can discuss what each layer contributes rather than quoting
one inflated number.

---

## Installation

```bash
git clone <repo>
cd dhsgsu-phishguard
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**1. Generate the synthetic dataset** (only required once, or to regenerate):

```bash
python data/generate_dataset.py
```

Output: `data/emails.csv` with 2000 rows — 1000 legitimate (label 0),
1000 phishing (label 1) — and columns
`subject, sender_name, sender_email, body, url, label`.

**2. Train the model and produce the report charts:**

```bash
python src/train.py
```

Writes `models/phishguard_model.joblib`, `metrics.json` and three PNGs under
`reports/`.

**3. Classify from the terminal:**

```bash
python src/cli.py --interactive
python src/cli.py --file email.txt
python src/cli.py --subject "Fee Receipt" \
                  --sender "finance@dhsgsu.edu.in" \
                  --body "Your fee payment has been received."
```

Exit code `0` = legitimate, `1` = phishing, `2` = error.

**4. Run the web UI:**

```bash
streamlit run app/streamlit_app.py
```

---

## Evaluation

Metrics are written to `metrics.json` at train time and displayed in the
Streamlit sidebar. Reported measures: accuracy, precision, recall, F1-score and
ROC-AUC on a stratified 80/20 hold-out split. Charts are saved as
`reports/confusion_matrix.png`, `reports/roc_curve.png` and
`reports/feature_importance.png`.

Because the headline numbers saturate at 1.0 on this synthetic corpus,
`train.py` also writes `reports/ablation.json`, which breaks the result down
into the domain-whitelist rule, the classifier without the domain signal, and
the full pipeline. See "On the reported metrics" above for how to read it.

---

## Academic disclaimer

> The dataset used in this project is **synthetic**. It was generated
> algorithmically from templates and random fillers; it contains no real
> university email, no real student or staff names, and no real personally
> identifiable information. Consequently the reported accuracy, precision,
> recall and F1-score describe performance on that synthetic distribution and
> **must not be presented as measured performance on live university mail**.
>
> This system is a demonstration artefact for academic evaluation. It is **not**
> a production security control and must not be deployed as a mail filter, nor
> should any decision to block, quarantine or act on an email be taken on its
> output alone. Always verify suspicious mail through the university's official
> channel at https://dhsgsu.edu.in/.

---

## Dependencies

| Package | Role |
|---|---|
| `numpy` | numeric arrays |
| `pandas` | dataset handling |
| `scikit-learn` | ColumnTransformer, TF-IDF, HistGradientBoosting, metrics |
| `joblib` | model persistence |
| `streamlit` | web UI |
| `matplotlib`, `seaborn` | confusion matrix and ROC charts |

```
numpy>=1.26
pandas>=2.1
scikit-learn>=1.4
joblib>=1.4
streamlit>=1.30
matplotlib>=3.8
seaborn>=0.13
```

---

## Reproducibility

`SEED = 42` in `config.py` controls dataset shuffling, the train/test split
and the classifier random state, so re-running `train.py` reproduces the same
split and (given the same data) the same fitted model.