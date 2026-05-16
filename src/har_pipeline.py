"""Run the public HAR model comparison on anonymized feature data."""

from __future__ import annotations

import argparse
from pathlib import Path

from har_features import (
    evaluate_holdout,
    evaluate_leave_one_participant_out,
    load_feature_dataset,
    summarize_lopo,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HAR classifiers.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/anonymized_har_features.csv"),
        help="Anonymized feature CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_feature_dataset(args.features)

    print(f"Dataset shape: {dataset.shape[0]} rows x {dataset.shape[1]} columns")
    print("Participants:", ", ".join(sorted(dataset["participant"].unique())))
    print("Activities:", ", ".join(sorted(dataset["activity"].unique())))

    print("\nStratified holdout results:")
    print(evaluate_holdout(dataset).round(3).to_string(index=False))

    print("\nLeave-one-participant-out summary:")
    lopo = evaluate_leave_one_participant_out(dataset)
    print(summarize_lopo(lopo).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

