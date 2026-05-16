"""Feature engineering and evaluation utilities for smartphone HAR.

The public dataset in this repository contains derived features only. Raw sensor
recordings, original participant names, timestamps, and local paths are not
stored in the repository.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


SENSOR_COLUMNS = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "grav_x",
    "grav_y",
    "grav_z",
]

MAGNITUDE_GROUPS = {
    "acc": ["acc_x", "acc_y", "acc_z"],
    "gyro": ["gyro_x", "gyro_y", "gyro_z"],
    "grav": ["grav_x", "grav_y", "grav_z"],
}

META_COLUMNS = ["participant", "activity", "trial_id"]

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")


def trial_sort_key(path: Path) -> tuple[int, str]:
    """Sort t1, t2, ..., t10 in natural order."""

    stem = path.name.lower().lstrip("t")
    return (int(stem), path.name) if stem.isdigit() else (10_000, path.name)


def participant_sort_key(name: str) -> tuple[int, str]:
    suffix = name.lower().lstrip("p")
    return (int(suffix), name) if suffix.isdigit() else (10_000, name)


def extract_features(frame: pd.DataFrame) -> dict[str, float]:
    """Convert one trial-level sensor recording into numeric features."""

    features: dict[str, float] = {}
    numeric = frame[SENSOR_COLUMNS].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.interpolate(limit_direction="both").dropna()
    if numeric.empty:
        raise ValueError("No usable numeric sensor rows found.")

    features["n_samples"] = float(len(numeric))
    if "seconds_elapsed" in frame.columns:
        seconds = pd.to_numeric(frame["seconds_elapsed"], errors="coerce").dropna()
        features["duration_s"] = float(seconds.max() - seconds.min()) if len(seconds) > 1 else 0.0

    for column in SENSOR_COLUMNS:
        values = numeric[column].to_numpy(dtype=float)
        centered = values - values.mean()
        spectrum = np.fft.rfft(centered)

        features[f"{column}_mean"] = float(values.mean())
        features[f"{column}_std"] = float(values.std(ddof=0))
        features[f"{column}_min"] = float(values.min())
        features[f"{column}_max"] = float(values.max())
        features[f"{column}_median"] = float(np.median(values))
        features[f"{column}_iqr"] = float(np.percentile(values, 75) - np.percentile(values, 25))
        features[f"{column}_rms"] = float(np.sqrt(np.mean(values**2)))
        features[f"{column}_energy"] = float(np.mean(centered**2))
        features[f"{column}_fft_energy"] = float(np.mean(np.abs(spectrum) ** 2))

    for name, columns in MAGNITUDE_GROUPS.items():
        magnitude = np.sqrt((numeric[columns] ** 2).sum(axis=1)).to_numpy(dtype=float)
        centered = magnitude - magnitude.mean()
        spectrum = np.fft.rfft(centered)

        features[f"{name}_mag_mean"] = float(magnitude.mean())
        features[f"{name}_mag_std"] = float(magnitude.std(ddof=0))
        features[f"{name}_mag_min"] = float(magnitude.min())
        features[f"{name}_mag_max"] = float(magnitude.max())
        features[f"{name}_mag_iqr"] = float(np.percentile(magnitude, 75) - np.percentile(magnitude, 25))
        features[f"{name}_mag_energy"] = float(np.mean(centered**2))
        features[f"{name}_mag_fft_energy"] = float(np.mean(np.abs(spectrum) ** 2))

    for prefix, columns in MAGNITUDE_GROUPS.items():
        corr = numeric[columns].corr().to_numpy(dtype=float)
        features[f"{prefix}_corr_xy"] = float(corr[0, 1])
        features[f"{prefix}_corr_xz"] = float(corr[0, 2])
        features[f"{prefix}_corr_yz"] = float(corr[1, 2])

    return features


def build_feature_dataset(processed_root: Path) -> pd.DataFrame:
    """Build an anonymized trial-level feature dataset from processed CSV files."""

    processed_root = processed_root.resolve()
    if not processed_root.exists():
        raise FileNotFoundError(f"Processed data folder not found: {processed_root}")

    participant_dirs = sorted(
        [path for path in processed_root.iterdir() if path.is_dir()],
        key=lambda p: participant_sort_key(p.name),
    )
    participant_map = {
        path.name: f"participant_{idx:02d}" for idx, path in enumerate(participant_dirs, start=1)
    }

    rows: list[dict[str, float | str]] = []
    for participant_dir in participant_dirs:
        anon_participant = participant_map[participant_dir.name]
        for activity_dir in sorted([path for path in participant_dir.iterdir() if path.is_dir()]):
            trial_dirs = sorted(
                [path for path in activity_dir.iterdir() if path.is_dir()],
                key=trial_sort_key,
            )
            for trial_number, trial_dir in enumerate(trial_dirs, start=1):
                csv_path = trial_dir / "processed_merged.csv"
                if not csv_path.exists():
                    continue

                frame = pd.read_csv(csv_path)
                if not set(SENSOR_COLUMNS).issubset(frame.columns):
                    continue

                try:
                    features = extract_features(frame)
                except ValueError:
                    continue

                row: dict[str, float | str] = {
                    "participant": anon_participant,
                    "activity": activity_dir.name,
                    "trial_id": f"{anon_participant}_{activity_dir.name}_{trial_number:03d}",
                }
                row.update(features)
                rows.append(row)

    dataset = pd.DataFrame(rows)
    if dataset.empty:
        raise ValueError("No feature rows were generated from the processed data folder.")

    feature_columns = sorted([column for column in dataset.columns if column not in META_COLUMNS])
    return dataset[META_COLUMNS + feature_columns]


def load_feature_dataset(path: Path) -> pd.DataFrame:
    dataset = pd.read_csv(path)
    missing = set(META_COLUMNS) - set(dataset.columns)
    if missing:
        raise ValueError(f"Missing required metadata columns: {sorted(missing)}")
    return dataset


def split_features(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    x = dataset.drop(columns=META_COLUMNS).replace([np.inf, -np.inf], np.nan)
    y = dataset["activity"]
    groups = dataset["participant"]
    return x, y, groups


def make_models() -> dict[str, object]:
    return {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1500, class_weight="balanced"),
        ),
        "knn": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5),
        ),
        "gaussian_nb": make_pipeline(SimpleImputer(strategy="median"), GaussianNB()),
        "decision_tree": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(max_depth=8, random_state=42, class_weight="balanced"),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=250,
                random_state=42,
                class_weight="balanced",
                n_jobs=1,
            ),
        ),
    }


def evaluate_holdout(dataset: pd.DataFrame, test_size: float = 0.25) -> pd.DataFrame:
    x, y, _ = split_features(dataset)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42, stratify=y
    )

    rows = []
    for name, model in make_models().items():
        fitted = clone(model)
        fitted.fit(x_train, y_train)
        pred = fitted.predict(x_test)
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, pred),
                "macro_f1": f1_score(y_test, pred, average="macro"),
            }
        )

    return pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)


def evaluate_leave_one_participant_out(dataset: pd.DataFrame) -> pd.DataFrame:
    x, y, groups = split_features(dataset)
    logo = LeaveOneGroupOut()

    rows = []
    for name, model in make_models().items():
        for fold, (train_idx, test_idx) in enumerate(logo.split(x, y, groups), start=1):
            fitted = clone(model)
            fitted.fit(x.iloc[train_idx], y.iloc[train_idx])
            pred = fitted.predict(x.iloc[test_idx])
            held_out = groups.iloc[test_idx].iloc[0]
            rows.append(
                {
                    "model": name,
                    "fold": fold,
                    "held_out_participant": held_out,
                    "accuracy": accuracy_score(y.iloc[test_idx], pred),
                    "balanced_accuracy": balanced_accuracy_score(y.iloc[test_idx], pred),
                    "macro_f1": f1_score(y.iloc[test_idx], pred, average="macro"),
                }
            )

    return pd.DataFrame(rows)


def summarize_lopo(lopo_results: pd.DataFrame) -> pd.DataFrame:
    return (
        lopo_results.groupby("model", as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
        )
        .sort_values("balanced_accuracy_mean", ascending=False)
    )
