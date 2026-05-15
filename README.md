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

## Public Demo

The original raw recordings are not included. This repository contains a small synthetic demo pipeline that mirrors the feature-engineering and classifier workflow without exposing participant data.

Run:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python src\har_pipeline.py --synthetic
```

## Repository Structure

```text
src/              Public-safe HAR feature extraction and synthetic demo pipeline
data/sample/      Notes for synthetic/sample data
figures/          Reserved for selected public-safe figures
```

## Technologies

Python, NumPy, pandas, scikit-learn, signal features, classification, activity recognition.

## Data Availability

The real smartphone recordings are omitted because they contain private participant sensor traces and raw collection artifacts. The public repo uses synthetic data for demonstration only.

## Status and Limitations

Curated public portfolio version. The synthetic demo validates the code path but is not a substitute for reporting real-world model performance.

