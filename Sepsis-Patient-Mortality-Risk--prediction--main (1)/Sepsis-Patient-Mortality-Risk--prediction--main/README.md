# 🧬 Sepsis Patient Mortality Risk Prediction

> Early-warning ML system predicting in-hospital mortality for ICU sepsis patients using clinical data from MIMIC-IV.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Best_Model-FF6600?style=flat)](https://xgboost.readthedocs.io)
[![AUC](https://img.shields.io/badge/Test_AUC-0.8588-10b981?style=flat)](#-model-performance)
[![Flask](https://img.shields.io/badge/Flask-Web_App-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![MIMIC-IV](https://img.shields.io/badge/Dataset-MIMIC--IV-0066CC?style=flat)](https://physionet.org/content/mimiciv/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

---

## 📌 Overview

Sepsis is a life-threatening condition responsible for one of the highest ICU mortality rates globally. This project delivers an **end-to-end ML pipeline** — from raw MIMIC-IV data to a deployed Flask web application — that predicts mortality risk in real time and surfaces actionable risk levels for clinical teams.

**Pipeline:** Data Ingestion → Transformation → XGBoost Training → F2-Threshold Tuning → Flask Web App

<!-- 🎬 **[Watch Demo](https://drive.google.com/file/d/1vfAtL3qrUr5U6ToNtJ9Jf6OEGZf4CABv/view?usp=drive_link)** -->

---

## 📊 Dataset — MIMIC-IV

| Property | Value |
|---|---|
| Raw records | 58,653 rows × 53 features |
| After deduplication | 8,704 unique ICU stays |
| Target | `hospital_expire_flag` (27% mortality) |
| Train / Test | 6,971 / 1,733 (subject-ID split, **0 overlap**) |
| After SMOTE | 10,226 balanced train samples |
| Features used | **47 clinical features** |

**Features:** Vitals (HR, BP, SpO2, RR, Temp) · Labs (Glucose, Sodium) · SOFA Score · ICU LOS · Urine Output · Comorbidities · Antibiotic classes · Demographics

---

## 🧪 Model Performance

### Comparison — After Hyperparameter Tuning

| Model | CV AUC | Test AUC | Accuracy | Precision | Recall | F2 | MCC |
|---|---|---|---|---|---|---|---|
| **XGBoost** | 0.9494 | **0.8345** | **0.7904** | 0.6303 | 0.5119 | 0.5319 | **0.4328** |
| LightGBM | 0.9518 | 0.8342 | 0.7898 | **0.6351** | 0.4924 | 0.5156 | 0.4258 |
| Random Forest | **0.9541** | 0.8297 | 0.7800 | 0.5957 | **0.5378** | **0.5485** | 0.4195 |

### ⭐ Production Model — XGBoost + F2-Optimised Threshold

| CV AUC | Test AUC | Threshold | Recall | Precision | F2 |
|---|---|---|---|---|---|
| 0.9492 | **0.8588** | **0.12** | **0.9172** | 0.4532 | **0.7613** |

> Threshold tuned to maximise F2 (`min_precision = 0.35`). In ICU mortality prediction, **missing a high-risk patient is far more costly than a false alarm** — F2 weights recall 2× over precision.

### Risk Stratification

| Level | Probability | Action |
|---|---|---|
| 🟢 Low | < 25% | Standard sepsis bundle |
| 🟡 Moderate | 25% – threshold | Enhanced monitoring |
| 🟠 High | threshold – 70% | Immediate escalation |
| 🔴 Critical | ≥ 70% | Urgent intervention |

---

## ⚙️ Tech Stack

**ML:** Python · XGBoost · LightGBM · Scikit-learn · imbalanced-learn (SMOTE)  
**App:** Flask · Jinja2 · HTML5 / CSS3 / Vanilla JS

---

## 🚀 Quick Start

```bash
git clone https://github.com/FAZIL-SIDDIQUI/Sepsis-Patient-Mortality-Risk--prediction-.git
cd Sepsis-Patient-Mortality-Risk-prediction

python -m venv sepsis-risk
sepsis-risk\Scripts\activate        # Windows
# source sepsis-risk/bin/activate   # Mac / Linux

pip install -r requirements.txt

# Train pipeline (~31 seconds)
python src/pipeline/train_pipeline.py

# Launch web app
python app.py
# → http://localhost:5000
```

### Routes

| Route | Method | Description |
|---|---|---|
| `/` | GET | Landing page |
| `/predictdata` | GET / POST | Patient form + prediction result |
| `/api/predict` | POST | JSON API endpoint |
| `/health` | GET | Artifact health check |
| `/retrain` | GET | Trigger retraining |

---

## 📁 Project Structure

```
├── app.py
├── requirements.txt
├── artifacts/               ← model.pkl · preprocessor.pkl · threshold.txt
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   └── pipeline/
│       ├── train_pipeline.py
│       └── predict_pipeline.py
└── templates/
    ├── index.html
    └── home.html
```

---

## 🔍 Key Design Decisions

| Decision | Why |
|---|---|
| Subject-ID split (not row-level) | Zero patient leakage between train / test |
| SMOTE on train only | Prevents synthetic data inflating test metrics |
| F2 threshold + precision floor | Prioritises recall; floor prevents degenerate threshold |
| IQR capping fit on train only | Prevents test distribution leaking into feature bounds |

---

## ⚠️ Disclaimer

For clinical decision support only. Not a substitute for clinical judgement. MIMIC-IV data requires [PhysioNet credentialed access](https://physionet.org/content/mimiciv/view-required-training/3.1/).
