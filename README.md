# Diabetes Mellitus Prognostic Model

Exploratory data analysis and a Random Forest classifier that predicts diabetes status
(non-diabetic / pre-diabetic / diabetic) from routine clinical and laboratory measurements.

## The problem

Pre-diabetes is often missed, even though the blood tests that could flag it (HbA1c, lipids,
kidney markers, BMI) are already in most patients' records. The question is whether those
routine results are enough to tell non-diabetic, pre-diabetic and diabetic patients apart.

## Who it's for

Health data analysts and clinical researchers who want to screen patient records for diabetes
risk. It is also a worked example of an end-to-end classification project on imbalanced
medical data.

## What it does

Cleans the dataset, explores how each measurement differs between classes and trains a tuned
Random Forest that predicts diabetes status from 11 routine features.

## Key results

- **98% accuracy** on a held-out test set of 200 patients (1,000 records in total).
- **F1 = 0.93** for non-diabetic, **0.99** for diabetic and **1.00** for pre-diabetic patients.
- Found and fixed label errors in the raw data (`f` vs `F`, whitespace and lowercase values in the
  target) that would otherwise have created false classes.
- **Limitation:** 84% of patients are diabetic, and the pre-diabetic class has only 6 test
  cases. The scores need confirming on a larger, balanced dataset before anyone relies on them.

**Stack:** Python, pandas, scikit-learn (Random Forest, GridSearchCV), matplotlib, seaborn

---

## Dataset

**Diabetes Dataset** — Rashid, A. (2020), Mendeley Data, V1.
DOI: [10.17632/wj9rwkp9c2.1](https://doi.org/10.17632/wj9rwkp9c2.1)

- 1,000 anonymised patient records, 14 columns
- Features: gender, age, urea, creatinine (Cr), HbA1c, cholesterol, triglycerides (TG),
  HDL, LDL, VLDL, BMI
- Target `CLASS`: `N` — non-diabetic (103), `P` — pre-diabetic (53), `Y` — diabetic (844)

The CSV is not included in this repository. Download `dataset_of_diabetes.csv` from the link
above and place it next to the notebook.

## Workflow

1. **Data cleaning** — fixed inconsistent labels (`f` → `F` in `Gender`; stray whitespace and
   lowercase values in `CLASS`); no missing values found.
2. **Encoding** — `Gender` and `CLASS` label-encoded; `ID` and `No_Pation` excluded from features.
3. **EDA** — distributions, class / gender / age / BMI / HbA1c breakdowns, summary table by
   class, correlation heatmap.
4. **Modelling** — `RandomForestClassifier` tuned with `GridSearchCV` (5-fold CV, 80/20
   train-test split). Best parameters: `max_depth=10`, `n_estimators=100`,
   `min_samples_split=2`, `min_samples_leaf=1`.
5. **Evaluation** — classification report, confusion matrix, feature importances,
   visualisation of a single tree.

## Results (test set, n = 200)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|------|
| N     | 0.95      | 0.90   | 0.93 |
| P     | 1.00      | 1.00   | 1.00 |
| Y     | 0.99      | 0.99   | 0.99 |

**Accuracy: 0.98**

The dataset is strongly imbalanced (84% diabetic) and the pre-diabetic class has only
6 test samples, so per-class scores for `P` should be read with caution.

## Running

The notebook was written in Google Colab. To run locally, remove the first two cells
(Google Drive mount and `%cd`) and install the dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook diabetes_analysis.ipynb
```

## Author

**Anna Hlinska** — [github.com/anna-hlinska-data](https://github.com/anna-hlinska-data)
