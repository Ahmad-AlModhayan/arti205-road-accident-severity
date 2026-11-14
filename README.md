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

This project predicts whether a police-reported road traffic collision is **Slight** (75.16%), **Serious** (23.35%), or **Fatal** (1.49%) using only pre-crash conditions known at the time of the crash: collision conditions (`month`, `day_of_week`, `hour`, road class/type, speed limit, urban/rural area, junction details, light, weather, road surface, carriageway hazards, and number of vehicles), 9 joined pre-crash vehicle & driver features (`vehicle_type`, `driver_age`, `sex_of_driver`, `vehicle_manoeuvre`, `junction_location`, `journey_purpose`, `towing_and_articulation`, `propulsion_type`, `vehicle_age`), and 6 engineered pre-crash interaction features (`speed_x_vehicles`, `speed_per_vehicle`, `speed_x_area`, `light_x_speed`, `vehicle_x_speed`, `vehicle_x_manoeuvre`).

Because fatal collisions represent only **1.49%** of the 100,927 collisions (a **51 : 1** imbalance ratio), a dummy baseline that always predicts *Slight* reaches 75.2% accuracy while finding **0%** of fatal collisions. We compare four models (**Logistic Regression**, **Decision Tree**, **Random Forest**, and **XGBoost**) under identical preprocessing, 5-fold stratified cross-validation, and four imbalance strategies (**None**, **Class weights**, **Random under-sampling**, and **SMOTE-NC**), evaluated with **macro F1**, **balanced accuracy**, **per-class recall**, **confusion matrices**, and **SHAP**.

### Final Test Set Results (20,186 collisions; 300 fatal)

| Model | Balancing | Accuracy | Balanced Acc. | Macro F1 | Recall Slight | Recall Serious | Recall Fatal | Precision Fatal | AUC (OvR) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Always "Slight" (dummy) | — | 75.2% | 33.3% | 0.286 | 100.0% | 0.0% | 0.0% | 0.0% | — |
| Logistic Regression | SMOTE-NC | 59.9% | 47.2% | 0.389 | 67.8% | 35.9% | 38.0% | 5.1% | 0.672 |
| Decision Tree | SMOTE-NC | 62.2% | 39.4% | 0.380 | 71.2% | 36.4% | 10.7% | 5.0% | 0.581 |
| **Random Forest (Proposed)** | **Class weights** | **66.9%** | **45.8%** | **0.435** | **75.1%** | **43.6%** | **18.7%** | **9.4%** | **0.714** |
| XGBoost | Class weights | 60.4% | 47.2% | 0.406 | 64.6% | 49.2% | 27.7% | 5.3% | 0.706 |

Key SHAP drivers pushing predictions toward **Fatal**: `sex_of_driver` (Male), rural roads (`urban_or_rural_area`), `vehicle_type` (especially Heavy Goods Vehicles 7.5t+ and large motorcycles), `speed_x_area` (Rural 60–70 mph), `speed_limit`, `vehicle_x_manoeuvre`, `speed_per_vehicle`, `vehicle_manoeuvre` (going ahead), `light_x_speed`, and collisions away from junctions.

### Live Prediction CLI (`predict.py`)

```bash
python predict.py --demo
python predict.py --interactive
python predict.py --json '{"speed_limit": 60, "urban_or_rural_area": "Rural", "vehicle_type": "Heavy goods (7.5t+)"}'
python predict.py --input collisions.csv --output predictions.csv
```

`predict.py` loads the trained Random Forest pipeline saved at `Analysis/results/final_model.joblib` and automatically computes the 6 engineered interaction features from the 25 base collision/vehicle/driver inputs.

---

## Repository Contents

| Folder or file | Content |
|---|---|
| `Reports/` | Preliminary and final project reports (Word and PDF) |
| `Presentation/` | Final project presentation (`ARTI 205 - Group 3 - Project Presentation IAU.pptx`) |
| `Analysis/` | Three executed Jupyter notebooks, `figures/` (`01`–`12` PNGs), and `results/` (CSV/JSON summaries + `final_model.joblib`) |
| `Data/` | Raw DfT 2024 collision & vehicle CSVs, `prepare_dft2024.py`, `UK_collisions_final.csv` (100,927 × 32), and [Data/README.md](Data/README.md) |
| `Literature/` | Literature review (Word and PDF) and Mendeley references (`References — Mendeley import.ris`) |
| `ARTI 308 - Milestone 01 - Project Topic Selection and Brainstorming.*` | Milestone 01 topic selection (Word and PDF, submitted Sep 2026 prior to switching the final topic to Accident Severity) |
| `predict.py` | CLI prediction script using `Analysis/results/final_model.joblib` |
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
