# 🛡️ DHSGSU PhishGuard

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
