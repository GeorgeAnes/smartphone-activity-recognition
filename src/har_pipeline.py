"""Public-safe smartphone activity recognition demo.

The original project used private smartphone IMU recordings. This script keeps
the same high-level workflow with synthetic windows so the repository remains
safe to publish and easy to run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
ACTIVITIES = ("sitting", "standing", "walking", "running", "stairs")


@dataclass(frozen=True)
class SyntheticConfig:
    windows_per_activity: int = 40
    samples_per_window: int = 100
    sample_rate_hz: float = 100.0
    seed: int = 7


def _activity_waveform(activity: str, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, 0.04, size=(t.size, len(CHANNELS)))

    if activity == "sitting":
        base = np.column_stack(
            [
                0.02 * np.sin(2 * np.pi * 0.2 * t),
                0.02 * np.cos(2 * np.pi * 0.2 * t),
                1.0 + 0.01 * np.sin(2 * np.pi * 0.1 * t),
                0.01 * np.sin(2 * np.pi * 0.2 * t),
                0.01 * np.cos(2 * np.pi * 0.2 * t),
                0.01 * np.sin(2 * np.pi * 0.1 * t),
            ]
        )
    elif activity == "standing":
        base = np.column_stack(
            [
                0.04 * np.sin(2 * np.pi * 0.4 * t),
                0.03 * np.cos(2 * np.pi * 0.4 * t),
                1.0 + 0.03 * np.sin(2 * np.pi * 0.3 * t),
                0.02 * np.sin(2 * np.pi * 0.4 * t),
                0.02 * np.cos(2 * np.pi * 0.4 * t),
                0.02 * np.sin(2 * np.pi * 0.3 * t),
            ]
        )
    elif activity == "walking":
        base = _periodic_motion(t, accel_amp=0.35, gyro_amp=0.22, freq=1.8)
    elif activity == "running":
        base = _periodic_motion(t, accel_amp=0.75, gyro_amp=0.45, freq=3.0)
    elif activity == "stairs":
        base = _periodic_motion(t, accel_amp=0.55, gyro_amp=0.32, freq=2.2)
        base[:, 2] += 0.18 * np.sin(2 * np.pi * 0.7 * t)
    else:
        raise ValueError(f"Unknown activity: {activity}")

    return base + noise


def _periodic_motion(t: np.ndarray, accel_amp: float, gyro_amp: float, freq: float) -> np.ndarray:
    return np.column_stack(
        [
            accel_amp * np.sin(2 * np.pi * freq * t),
            0.6 * accel_amp * np.cos(2 * np.pi * freq * t),
            1.0 + 0.4 * accel_amp * np.sin(4 * np.pi * freq * t),
            gyro_amp * np.sin(2 * np.pi * freq * t + 0.4),
            0.7 * gyro_amp * np.cos(2 * np.pi * freq * t),
            0.5 * gyro_amp * np.sin(4 * np.pi * freq * t),
        ]
    )


def generate_synthetic_windows(config: SyntheticConfig) -> list[tuple[str, pd.DataFrame]]:
    rng = np.random.default_rng(config.seed)
    t = np.arange(config.samples_per_window) / config.sample_rate_hz
    windows: list[tuple[str, pd.DataFrame]] = []

    for activity in ACTIVITIES:
        for _ in range(config.windows_per_activity):
            values = _activity_waveform(activity, t, rng)
            windows.append((activity, pd.DataFrame(values, columns=CHANNELS)))

    rng.shuffle(windows)
    return windows


def extract_features(window: pd.DataFrame) -> dict[str, float]:
    features: dict[str, float] = {}

    for channel in CHANNELS:
        signal = window[channel].to_numpy()
        centered = signal - signal.mean()
        spectrum = np.fft.rfft(centered)

        features[f"{channel}_mean"] = float(signal.mean())
        features[f"{channel}_std"] = float(signal.std())
        features[f"{channel}_min"] = float(signal.min())
        features[f"{channel}_max"] = float(signal.max())
        features[f"{channel}_energy"] = float(np.mean(centered**2))
        features[f"{channel}_fft_energy"] = float(np.mean(np.abs(spectrum) ** 2))

    acc_mag = np.sqrt((window[["acc_x", "acc_y", "acc_z"]] ** 2).sum(axis=1))
    gyro_mag = np.sqrt((window[["gyro_x", "gyro_y", "gyro_z"]] ** 2).sum(axis=1))
    features["acc_mag_mean"] = float(acc_mag.mean())
    features["acc_mag_std"] = float(acc_mag.std())
    features["gyro_mag_mean"] = float(gyro_mag.mean())
    features["gyro_mag_std"] = float(gyro_mag.std())
    return features


def build_feature_table(windows: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for label, window in windows:
        row = extract_features(window)
        row["activity"] = label
        rows.append(row)
    return pd.DataFrame(rows)


def train_and_evaluate(features: pd.DataFrame) -> None:
    x = features.drop(columns=["activity"])
    y = features["activity"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print(f"Synthetic demo accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions, zero_division=0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the public-safe HAR demo.")
    parser.add_argument("--synthetic", action="store_true", help="Run the synthetic demo pipeline.")
    parser.add_argument("--windows-per-activity", type=int, default=40)
    parser.add_argument("--samples-per-window", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.synthetic:
        raise SystemExit("Only the synthetic public demo is available in this repository.")

    config = SyntheticConfig(
        windows_per_activity=args.windows_per_activity,
        samples_per_window=args.samples_per_window,
    )
    windows = generate_synthetic_windows(config)
    features = build_feature_table(windows)
    train_and_evaluate(features)


if __name__ == "__main__":
    main()
