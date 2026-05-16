"""Build the anonymized HAR feature dataset from local processed recordings."""

from __future__ import annotations

import argparse
from pathlib import Path

from har_features import build_feature_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build anonymized HAR features.")
    parser.add_argument(
        "--processed-root",
        type=Path,
        required=True,
        help="Path to the local processed HAR folder containing participant/activity/trial folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/anonymized_har_features.csv"),
        help="Output CSV path for anonymized features.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_feature_dataset(args.processed_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    print(f"Wrote {len(dataset):,} rows and {len(dataset.columns):,} columns to {args.output}")
    print("Participants:", ", ".join(sorted(dataset["participant"].unique())))
    print("Activities:", ", ".join(sorted(dataset["activity"].unique())))


if __name__ == "__main__":
    main()

