# Smartphone Activity Recognition from IMU Signals

Machine-learning pipeline for classifying human activities from smartphone accelerometer and gyroscope signals.

## Problem

Smartphone IMU data can be used to infer activities such as sitting, standing, walking, running, and stair climbing. The original project studied user-independent generalization using feature engineering, supervised learning, clustering, and participant-level validation.

## Methods

- Windowed accelerometer and gyroscope features
- Time-domain statistics and frequency-domain energy features
- Feature selection with mutual information and recursive feature elimination
- Supervised classifiers including logistic regression, decision trees, naive Bayes, and k-nearest neighbors
- Leave-one-participant-out validation and external blind validation in the original project

## Public Dataset

The original raw recordings are not included. This repository contains an anonymized feature table derived from the processed IMU recordings, with participant names, timestamps, and local paths removed.

Run:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python src\har_pipeline.py
```

Open the end-to-end notebook:

```powershell
jupyter notebook notebooks\har_activity_recognition.ipynb
```

Current benchmark summary on the included feature table:

| Evaluation | Best model | Balanced accuracy |
| --- | --- | ---: |
| Stratified holdout | Random forest | 0.962 |
| Leave-one-participant-out | Random forest | 0.628 mean |

## Repository Structure

```text
src/              Feature extraction and model evaluation code
notebooks/        End-to-end portfolio notebook
data/             Anonymized derived feature table and data notes
figures/          Generated notebook figures
```

## Technologies

Python, NumPy, pandas, scikit-learn, signal features, classification, activity recognition.

## Data Availability

The raw smartphone recordings are omitted because they contain participant sensor traces and raw collection artifacts. The included CSV is a derived feature table with anonymized participant IDs.

## Status and Limitations

Curated public portfolio version. The notebook and command-line pipeline run on the included anonymized feature dataset.
