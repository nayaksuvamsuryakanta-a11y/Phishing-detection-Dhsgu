 🛡️ DHSGSU PhishGuard

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Explainability](https://img.shields.io/badge/XAI-SHAP-lightgrey.svg)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-Academic_Use_Only-green.svg)](LICENSE)

> **An Enterprise-Grade, Explainable AI (XAI) Phishing Detection System**  
> Custom-built for **Dr. Harisingh Gour Vishwavidyalaya, Sagar (M.P.)**  
> *Submitted for the "CodeCraft Challenges" (Cybersecurity Domain)*

---

## 📌 Executive Summary

**DHSGSU PhishGuard** is a hybrid AI/ML system designed to classify university-related emails as **LEGITIMATE** or **PHISHING**. Unlike traditional "black-box" models, PhishGuard combines Natural Language Processing (TF-IDF) with domain-specific cybersecurity heuristics (domain spoofing detection, URL analysis, credential-request scanning). 

It goes beyond basic classification by providing **SHAP-based visual explainability**, a **Human-in-the-Loop Active Learning** feedback mechanism, and **SHA-256 cryptographic model integrity checks** to prevent supply-chain tampering.

---

## 🚀 Key Features

- 🧠 **Hybrid ML Pipeline**: Combines text analysis (`TfidfVectorizer`) with 15+ numeric heuristics using `ColumnTransformer` and `HistGradientBoostingClassifier`.
- 🔍 **Explainable AI (XAI)**: Integrates `SHAP` (SHapley Additive exPlanations) to generate visual waterfall charts, showing exactly *why* an email was flagged.
- 🔄 **Active Learning (MLOps)**: A built-in feedback loop allows admins to correct misclassifications, which are logged and used to automatically retrain and improve the model.
- 🔒 **Cryptographic Integrity**: Every trained model is hashed (SHA-256). The system verifies this hash at runtime to ensure the model has not been tampered with.
- 🎨 **Institutional UI**: A clean, accessible, white-and-blue Streamlit interface matching official DHSGSU branding.
- 🛡️ **Security Posture**: Simulates SPF/DKIM/DMARC authentication checks and cross-references URLs against a synthetic threat intelligence blocklist.

---

## 🏗️ System Architecture

```text
                               [ Raw Email Input ]
                                       │
                                       ▼
                      ┌────────────────────────────────┐
                      │   Feature Extraction Engine    │
                      │  • TF-IDF (Subject + Body)     │
                      │  • Heuristics (Domain, URL,    │
                      │    Urgency, Auth Checks)       │
                      └────────────────┬───────────────┘
                                       │
                                       ▼
                      ┌────────────────────────────────┐
                      │   Hybrid ML Pipeline           │
                      │  (ColumnTransformer + HGB)     │
                      └────────┬───────────────┬───────┘
                               │               │
               ┌───────────────┘               └───────────────┐
               ▼                                               ▼
    ┌─────────────────────┐                         ┌─────────────────────┐
    │  Prediction Engine  │                         │  SHAP Explainer     │
    │  • Risk Score (0-100)                        │  • Feature Attribution
    │  • Risk Band (L/M/H) │                         │  • Waterfall Chart  │
    └──────────┬──────────┘                         └─────────────────────┘
               │
               ▼
    ┌─────────────────────┐
    │ Active Learning Loop│ ◄── [ Admin Feedback: "Actually Legitimate" ]
    │  • Log to CSV       │
    │  • Retrain Model    │
    │  • Update SHA-256   │
    └─────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Core Language** | Python 3.10+ |
| **Machine Learning** | `scikit-learn`, `numpy`, `pandas`, `joblib` |
| **Explainability (XAI)** | `shap`, `matplotlib` |
| **Web Interface** | `streamlit` |
| **Testing & Quality** | `pytest` |

---

## ⚡ Getting Started

### 1. Prerequisites
Ensure you have Python 3.10+ installed. Navigate to the project root directory:
```bash
cd dhsgsu-phishguard
```

### 2. Environment Setup
Create and activate a virtual environment, then install dependencies:
```powershell
# Windows
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Generate Synthetic Dataset
*(No real emails are scraped. This generates a balanced, 2000+ row academic dataset.)*
```bash
python data/generate_dataset.py
```

### 4. Train the Model
This will evaluate multiple models, select the best by F1-score, save the pipeline, and generate the SHA-256 hash.
```bash
python -m src.train
```

---

## 🧪 How to Evaluate (Demo)

We have provided an automated demo script to prove the backend logic works flawlessly without needing the UI.

### Automated Backend Test
Run the demo script to see real-time predictions and SHAP feature mapping:
```bash
.venv\Scripts\python.exe scripts\demo_test.py
```
*Expected Output: A legitimate email scores ~0.01/100 (Low Risk), while a phishing email scores ~99.99/100 (High Risk) with a 2,418-feature SHAP explanation.*

### Manual UI Testing
Launch the Streamlit application:
```bash
.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py --server.headless true --server.port 8502
```
Open `http://localhost:8502` in your browser. *(Note: If the theme appears dark, click `⋮` → Settings → Theme → "Use custom theme", then `Ctrl+F5`)*.

**Try this Phishing Payload:**
- **Subject:** `URGENT: DHSGSU Student Portal Account Suspension`
- **Sender Email:** `admin@dhsgsu-edu.in` *(Notice the typo-squatting domain)*
- **Body:** `Your student account will be suspended in 24 hours. Verify your identity immediately by replying with your Roll Number and OTP, or click here: http://192.168.1.55/verify-dhsgsu`
- **URL:** `http://192.168.1.55/verify-dhsgsu`

---

## 📂 Project Structure

```text
dhsgsu-phishguard/
├── .streamlit/
│   └── config.toml             # Institutional white/blue theme config
├── app/
│   └── streamlit_app.py        # Main web interface with SHAP & Feedback UI
├── data/
│   ├── generate_dataset.py     # Combinatorial synthetic data generator
│   └── emails.csv              # Generated training data (gitignored)
├── models/
│   └── phishguard_model.joblib # Trained, hashed ML pipeline
├── scripts/
│   └── demo_test.py            # Automated backend verification script
├── src/
│   ├── config.py               # University constants & security rules
│   ├── features.py             # ColumnTransformer & heuristic extraction
│   ├── train.py                # Training, evaluation, and hash generation
│   ├── explain.py              # Prediction + SHAP explanation engine
│   ├── cli.py                  # Command-line interface
│   └── retrain_feedback.py     # Active learning ingestion & retraining
├── tests/
│   └── test_backend.py         # Pytest suite for regression testing
├── requirements.txt
└── README.md
```

---

## 🔒 Security Posture & Academic Disclaimer

- **Supply Chain Security**: The system computes a SHA-256 hash of `phishguard_model.joblib` after every training run. At runtime, `explain.py` verifies this hash. If the model file is altered or corrupted, the system halts and alerts the user.
- **Authentication Simulation**: The UI includes toggles for SPF, DKIM, and DMARC pass/fail states, heavily penalizing emails that claim to be from `@dhsgsu.edu.in` but fail authentication.
- **Academic Disclaimer**: This project is developed solely for academic evaluation and the CodeCraft Challenge. The dataset is **100% synthetically generated** using combinatorial templates. No real university emails were scraped, stored, or analyzed. 

---

## 👨‍💻 Author & Acknowledgments

Developed by **Suvam** for the BCA Program, Dr. Harisingh Gour Vishwavidyalaya, Sagar (M.P.).  
*Special thanks to the CodeCraft Evaluation Committee for the problem statement.*




Your GitHub repository is now presentation-ready. We have two major deliverables left to secure top marks:

**Option 1:** The Formal BCA Project Report (Chapters 1 to 5)  
**Option 2:** The Viva Voce Presentation Deck & Defense Script  

Which one would you like to generate next?
