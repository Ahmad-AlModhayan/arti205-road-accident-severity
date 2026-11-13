# Predicting the Severity of Road Traffic Accidents Using Machine Learning

**ARTI 205 – Machine Learning** · Imam Abdulrahman Bin Faisal University (CCIS)  
**Group 3** · Instructor: Dr. Irfan Ullah

---

## Team (Group 3)

| S. No | Student ID | Name | Role |
|---:|---|---|---|
| 1 | `2250030223` | Ahmad AlModhayan | Leader |
| 2 | `2250030275` | Sultan Al-Duhaim | Member |
| 3 | `2250030188` | Badr Al-Ali | Member |
| 4 | `2250030186` | Abdulelah Al-Anqoudi | Member |
| 5 | `2250030134` | Abdullah Al-Yousef | Member |
| 6 | `2250030132` | Mohammed Al-Mawlid | Member |

---

## Project Overview

This project predicts whether a police-reported road traffic collision is **Slight** (75.16%), **Serious** (23.35%), or **Fatal** (1.49%) using only conditions known at the time of the crash (time, road class/type, speed limit, urban/rural area, junction details, light, weather, road surface, carriageway hazards, and number of vehicles).

Because fatal collisions represent only **1.49%** of the 100,927 collisions (a **51 : 1** imbalance ratio), a dummy baseline that always predicts *Slight* reaches 75.2% accuracy while finding **0%** of fatal collisions. We compare four models (**Logistic Regression**, **Random Forest**, **XGBoost**, and **MLP Neural Network**) under identical preprocessing, 5-fold stratified cross-validation, and four imbalance strategies (**None**, **Class weights**, **Random under-sampling**, and **SMOTE-NC**), evaluated with **macro F1**, **balanced accuracy**, **per-class recall**, **confusion matrices**, and **SHAP**.

### Final Test Set Results (20,186 collisions; 300 fatal)

| Model | Balancing | Accuracy | Balanced Acc. | Macro F1 | Recall Slight | Recall Serious | Recall Fatal | Precision Fatal | AUC (OvR) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Always "Slight" (dummy) | — | 75.2% | 33.3% | 0.286 | 100.0% | 0.0% | 0.0% | 0.0% | — |
| Logistic Regression | SMOTE-NC | 51.4% | 43.8% | 0.341 | 58.0% | 30.8% | 42.7% | 3.5% | 0.606 |
| **Random Forest (Proposed)** | **Class weights** | **55.5%** | **45.3%** | **0.371** | **60.6%** | **40.4%** | **35.0%** | **4.3%** | **0.636** |
| XGBoost | Class weights | 55.5% | 42.3% | 0.365 | 60.3% | 41.9% | 24.7% | 3.8% | 0.625 |
| Neural Network (MLP) | SMOTE-NC | 58.6% | 38.9% | 0.363 | 66.3% | 36.6% | 13.7% | 3.3% | 0.590 |

Key SHAP drivers pushing predictions toward **Fatal**: higher speed limits, rural roads, single-vehicle collisions, collisions away from junctions, A roads, and darkness without street lights.

---

## Repository Contents

| Folder or file | Content |
|---|---|
| `Reports/` | Preliminary and final project reports (Word and PDF) |
| `Presentation/` | Final project presentation (`ARTI 205 - Group 3 - Project Presentation IAU.pptx`) |
| `Analysis/` | Three executed Jupyter notebooks, `figures/` (`01`–`12` PNGs), and `results/` (CSV/JSON summaries) |
| `Data/` | Raw DfT 2024 collision CSV, `prepare_dft2024.py`, `UK_collisions_final.csv` (100,927 × 17), and [Data/README.md](Data/README.md) |
| `Literature/` | Literature review (Word and PDF) and Mendeley references (`References — Mendeley import.ris`) |
| `ARTI 308 - Milestone 01 - Project Topic Selection and Brainstorming.*` | Milestone 01 topic selection (Word and PDF, submitted Sep 2026 prior to switching the final topic to Accident Severity) |
| `requirements.txt` | Python package dependencies |

---

## Running the Notebooks

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebooks from inside `Analysis/` in order:

1. `01_EDA_and_Data_Preparation.ipynb` — EDA charts (`figures/01`–`07`), data quality fixes, and builds `../Data/UK_collisions_final.csv` (~1 minute).
2. `02_Baseline_Models.ipynb` — Untuned baseline comparison and `figures/08_confusion_matrix.png` (~2 minutes).
3. `03_Model_Tuning_and_Evaluation.ipynb` — 5-fold CV grid search, balancing comparison, feature selection test, final test evaluation, and SHAP explanations (~1 to 1.5 hours).

---

## Dataset & Attribution

- **Dataset:** Road Safety Data – Collisions 2024 (UK Department for Transport, police STATS19 records, Open Government Licence)
- **Obtained from:** [Kaggle — Road Safety Data – Collisions 2024](https://www.kaggle.com/datasets/juharimat/road-safety-data-collisions-2024)
