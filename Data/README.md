# ARTI 205 Project — Dataset

**Project:** Predicting the Severity of Road Traffic Accidents Using Machine Learning
**Dataset:** Road Safety Data – Collisions 2024 (UK Department for Transport), obtained from Kaggle
**Prepared:** 28 Sep 2026

## Source and citation

| | |
|---|---|
| Originator | UK Department for Transport — police STATS19 collision & vehicle returns, published under the Open Government Licence |
| Obtained from | Kaggle (`dft-road-casualty-statistics-collision-2024.csv`) and UK DfT Open Data (`dft-road-casualty-statistics-vehicle-2024.csv`) |
| Files | `dft-road-casualty-statistics-collision-2024.csv` (19.6 MB) and `dft-road-casualty-statistics-vehicle-2024.csv` (20.3 MB) |
| Years | 2024 only (`collision_year` = 2024 in every row) |

Cite the Department for Transport as the author and Kaggle as where it was obtained. This is reference [10] in the literature review:

> UK Department for Transport, "Road Safety Data – Collisions 2024," *Kaggle*, 2024. [Online]. Available: https://www.kaggle.com/datasets/juharimat/road-safety-data-collisions-2024 (accessed Sep. 28, 2026).

## What is in it

- **100,927 collisions × 44 columns** (joined with **183,514 vehicle records × 32 columns** on `collision_index`), no duplicate collision rows, one row per collision.
- **Target:** `collision_severity` — 3 classes

| Class | Collisions | Share |
|---|---:|---:|
| Slight | 75,858 | 75.16% |
| Serious | 23,567 | 23.35% |
| Fatal | 1,502 | 1.49% |

Imbalance ratio **51 : 1** (Slight : Fatal).

## Files in this folder

| File | What it is |
|---|---|
| `dft-road-casualty-statistics-collision-2024.csv` | Raw collision file as downloaded. Do not edit. |
| `dft-road-casualty-statistics-vehicle-2024.csv` | Raw 2024 vehicle file from UK DfT Open Data (`collision_index` join for 9 pre-crash vehicle & driver features). Do not edit. |
| `prepare_dft2024.py` | Keeps the conditions known at the time of the crash, merges collision-level pre-crash vehicle/driver features (`vehicle_type`, `driver_age`, `sex_of_driver`, `vehicle_manoeuvre`, `junction_location`, `journey_purpose`, `towing_and_articulation`, `propulsion_type`, `vehicle_age`), maps severity to labels, writes the two files below. Run: `python prepare_dft2024.py` |
| `UK_collisions_model_ready.csv` | 100,927 rows × 30 columns — intermediate dataset with 9 pre-crash vehicle & driver features |
| `UK_collisions_final.csv` | 100,927 rows × 32 columns (25 base pre-crash features + 6 engineered interaction features + `severity`) — **the final modelling file**, written by `../Analysis/01_EDA_and_Data_Preparation.ipynb` |
| `UK_profile.txt` | Class counts, missing-value rates per column, column list |

## What the script removes, and why

Everything recorded as part of the **outcome** is dropped, so the models cannot see the answer:
`number_of_casualties`, `enhanced_severity_collision`, `collision_injury_based`, `collision_adjusted_severity_serious`, `collision_adjusted_severity_slight`.
Location coordinates, police force, reference numbers and the `_historic` columns are also dropped.

## Before modelling

- **Values are numeric codes**, not labels (e.g. `light_conditions = 4`). Map them with the DfT's *Road Safety Open Dataset Data Guide* before plotting or running SHAP.
- **Missing codes differ by column.** `-1` is always missing. `9` or `99` means unknown in most columns, but `number_of_vehicles` is a real count, and the 2024 re-coded columns (`junction_detail`, `pedestrian_crossing`, `carriageway_hazards`) use `99`. The script handles this; see `UK_profile.txt` for the rates.
- `junction_control = -1` mostly means "not at a junction", not missing.
- `special_conditions_at_site` is missing for about 61% of rows. Consider dropping it.
- `date` and `time` are text. Convert to month and hour before use.

## Limits to state in the report

- UK data, not Saudi. No open Saudi crash-level dataset exists, so the review's Saudi limitation stays as future work.
- One year (2024) of police-reported collisions; slight collisions are likely under-reported.
- The serious share (23.35%) is higher than in older STATS19 years. The file flags collisions recorded with the injury-based reporting system and gives adjusted severities; read the DfT guide before comparing with older studies.

## Analysis notebooks (`../Analysis/`)

- `01_EDA_and_Data_Preparation.ipynb` — EDA charts, data issues, builds `UK_collisions_final.csv`. Figures `01`–`07` are saved in `../Analysis/figures/`.
- `02_Baseline_Models.ipynb` — first untuned comparison of Logistic Regression, Decision Tree, Random Forest, and XGBoost (`08_confusion_matrix.png`).
- `03_Model_Tuning_and_Evaluation.ipynb` — hyperparameter tuning (5-fold CV), balancing comparison (none, class weights, random under-sampling, SMOTE-NC), feature selection check, held-out test evaluation, and SHAP explanations (`09`–`12` figures and `../Analysis/results/`).
