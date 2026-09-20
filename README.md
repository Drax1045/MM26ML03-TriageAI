<div align="center">

# 🚑 Healthcare Triage Classification

### AI/ML-Assisted, Cost-Aware Decision Support for START Triage

**Problem Statement ID:** `MM26ML03` • **Team ID:** `MM2626` • **Modelling Minds 2.0 (ML Track)**

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Final%20Model-189AB4)
![Task](https://img.shields.io/badge/Task-Multiclass%20Classification-8A2BE2)

</div>

---



## 📌 Overview

This project addresses the **MM26ML03 Healthcare Triage Classification** problem using supervised machine learning.

The objective is to predict one of four triage categories — **RED, YELLOW, GREEN, or BLACK** — from patient-level tabular data containing vital signs, consciousness information, demographics, arrival details, chief complaint, and historical patient measurements.

A key part of the problem is that **different misclassification errors have different costs**. Therefore, we evaluate the models using both standard classification metrics and the **official competition misclassification cost matrix**.

| Item | Details |
|---|---|
| **Task** | Supervised multiclass classification |
| **Target** | `start_category` |
| **Classes** | RED · YELLOW · GREEN · BLACK |
| **Train / Test** | 664 rows / 112 rows |
| **Features used** | 37 (31 numeric, 6 categorical), after removing `subject_id` |

**Data includes:** demographics, arrival transport, vital signs, GCS, AVPU/consciousness, pain information, chief complaint, historical vital-sign aggregates, and diagnosis/medication/vital-sign counts.

**Training class distribution:** YELLOW 284 · GREEN 176 · RED 135 · BLACK 69.

---

## 🧭 Pipeline

```text
Patient Data
     ↓
Preprocessing
     ↓
Multiple ML Models
     ↓
Model Evaluation
     ↓
XGBoost Final Model
     ↓
Class Probabilities
     ↓
Decision Rule
     ↓
Final Triage Prediction
     ↓
Submission CSV
```

---

## ⚙️ Approach

### Preprocessing

Preprocessing was implemented inside a scikit-learn pipeline to avoid data leakage.

* `subject_id` was excluded from predictive features and retained only for patient identification/submission.
* Numerical missing values were handled using **median imputation**.
* Categorical missing values were handled using **most-frequent imputation**.
* Categorical variables were converted using **one-hot encoding**.
* The dataset already provides useful derived features such as GCS total and minimum/maximum/mean historical vital signs.
* Additional missingness indicators were tested but did not provide sufficient improvement to become part of the main approach.

### Models Tested

We experimented with:

* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost
* LightGBM
* CatBoost
* XGBoost + CatBoost ensemble
* Optuna-tuned XGBoost
* Cost-aware decision rule

### Final Model

The final model is **XGBoost** with:

```text
n_estimators = 400
max_depth = 5
learning_rate = 0.05
subsample = 0.85
colsample_bytree = 0.85
```

The model was retrained on all 664 training rows before generating the final submission predictions.

### Validation

Model comparison used a **stratified 80/20 hold-out split (133 validation samples)**.

We also performed **5-fold stratified cross-validation** for XGBoost, obtaining a mean accuracy of **0.9141 ± 0.0163**.

---

## 💰 Cost-Sensitive Decision

The competition provides an official misclassification cost matrix. **We used this matrix exactly as provided and did not modify it.**

| Actual ↓ / Predicted → | RED | YELLOW | GREEN | BLACK |
| ---------------------- | --: | -----: | ----: | ----: |
| **RED**                |   0 |      5 |    10 |     5 |
| **YELLOW**             |   2 |      0 |     3 |     2 |
| **GREEN**              |   1 |      1 |     0 |     1 |
| **BLACK**              |   2 |      2 |     2 |     0 |

**Lower total misclassification cost is better.**

### How We Use the Probabilities

The model first produces probabilities for all four classes:

```text
RED     → probability
YELLOW  → probability
GREEN   → probability
BLACK   → probability
```

We also tested a cost-aware decision rule that calculates the expected cost of each possible prediction:

```python
expected_costs = probabilities @ COST_MATRIX
prediction = np.argmin(expected_costs)
```

In simple terms:

> **The cost-aware rule considers both how likely each class is and how costly each possible mistake would be.**

### Decision Rule Selection

We compared two decision strategies on the validation split:

| Decision Rule                    | Total Cost |
| -------------------------------- | ---------: |
| **Highest-probability (argmax)** |     **33** |
| Cost-aware expected-cost rule    |         35 |

Since the **highest-probability rule achieved the lower validation cost**, it was used for the final submitted labels.

The cost-aware rule was still an important part of the project because it allowed us to explicitly evaluate the model against the competition's asymmetric error costs.

---

## 📊 Evaluation

We evaluate the models using:

* Accuracy
* Per-class Precision
* Per-class Recall
* Per-class F1-score
* Macro-F1
* Weighted-F1
* Confusion Matrix
* Total Misclassification Cost

### Validation Results — 133 Samples

| Model               |   Accuracy |   Macro-F1 | Total Cost ↓ |
| ------------------- | ---------: | ---------: | -----------: |
| **XGBoost**         | **0.9323** | **0.9325** |       **33** |
| CatBoost            |     0.9248 |     0.9268 |           36 |
| Gradient Boosting   |     0.9173 |     0.9188 |           39 |
| LightGBM            |     0.9248 |     0.9161 |           42 |
| Random Forest       |     0.9248 |     0.9161 |           42 |
| Logistic Regression |     0.9098 |     0.9195 |           48 |

The XGBoost + CatBoost ensemble matched XGBoost with a cost of 33, while Optuna-tuned XGBoost achieved a cost of 36. Therefore, the simpler single XGBoost model was selected.

### XGBoost Per-Class Results

| Class  | Precision | Recall |   F1 |
| ------ | --------: | -----: | ---: |
| RED    |      1.00 |   0.78 | 0.88 |
| YELLOW |      0.88 |   1.00 | 0.93 |
| GREEN  |      1.00 |   0.91 | 0.96 |
| BLACK  |      0.93 |   1.00 | 0.97 |

### Confusion Matrix

Rows represent **actual** classes and columns represent **predicted** classes.

| Actual ↓ / Predicted → | RED | YELLOW | GREEN | BLACK |
| ---------------------- | --: | -----: | ----: | ----: |
| **RED**                |  21 |      5 |     0 |     1 |
| **YELLOW**             |   0 |     57 |     0 |     0 |
| **GREEN**              |   0 |      3 |    32 |     0 |
| **BLACK**              |   0 |      0 |     0 |    14 |

On this validation split, RED had the lowest recall (0.78), with 6 of 27 actual RED patients classified as another category.

Importantly, there were **0 RED → GREEN errors** in this validation split. The official cost matrix assigns a cost of **10** to an actual RED patient being predicted as GREEN.

### Feature Importance

Important XGBoost features included:

* `gcs_motor`
* Arrival by ambulance
* `vs_resprate_max`
* Chief-complaint categories
* Respiratory and oxygen-saturation related aggregate features

---

## 📁 Repository Contents

```text
.
├── MM26ML03.ipynb
├── MM2626_MM26ML03.csv
├── train_ml03.csv
├── test_pred_ml03.csv
└── README.md
```

* `MM26ML03.ipynb` — complete preprocessing, model training, evaluation and submission pipeline.
* `MM2626_MM26ML03.csv` — final competition submission.
* `train_ml03.csv` — provided training dataset.
* `test_pred_ml03.csv` — provided test dataset.
* `README.md` — project documentation.

---

## 📤 Submission Format

`MM2626_MM26ML03.csv` contains **112 rows** in the original test order with:

| Column               | Description                       |
| -------------------- | --------------------------------- |
| `Patient ID`         | Patient identifier (`subject_id`) |
| `Predicted Triage`   | RED / YELLOW / GREEN / BLACK      |
| `RED Probability`    | Model probability for RED         |
| `YELLOW Probability` | Model probability for YELLOW      |
| `GREEN Probability`  | Model probability for GREEN       |
| `BLACK Probability`  | Model probability for BLACK       |

The four probabilities sum to approximately 1 for each patient.

All 112 test rows are retained in the original order, including repeated patient IDs.

**Final prediction distribution:**

```text
YELLOW → 48
GREEN  → 33
RED    → 20
BLACK  → 11
```

---

## ▶️ How to Run

1. Open `MM26ML03.ipynb` in Google Colab or Jupyter.
2. Upload `train_ml03.csv` and `test_pred_ml03.csv`.
3. Install required packages if needed:

```bash
pip install xgboost lightgbm catboost optuna
```

4. Run the notebook cells to reproduce preprocessing, model training, evaluation and submission generation.

---

## 🔭 Future Scope

Since the competition dataset is fixed, future improvements would focus on making better use of the existing data:

* Better probability calibration
* Further model tuning
* Improved cost-sensitive decision optimization
* Patient-grouped validation to reduce potential patient-level leakage
* Frontend integration for easier patient-data entry and result visualization
* Agentic AI layer for explanation and workflow support

The future Agentic AI layer would support the ML model by explaining predictions and organizing workflow information rather than replacing the underlying classification model.

---

<div align="center">

Built for **Modelling Minds 2.0** · Problem Statement **MM26ML03** · Team **MM2626**

</div>
