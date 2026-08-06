"""Strict JSON and CSV artifacts for CEC2017 method comparisons."""

import csv
import json
from pathlib import Path


PER_RUN_COLUMNS = (
    "problem_id",
    "source_function_id",
    "category",
    "method",
    "training_seed",
    "evaluation_seed",
    "final_best",
    "gap",
    "steps",
    "final_fe",
)

SUMMARY_COLUMNS = (
    "problem_id",
    "source_function_id",
    "category",
    "method",
    "training_seed",
    "runs",
    "mean_gap",
    "median_gap",
    "std_gap",
    "min_gap",
    "max_gap",
)


def write_strict_json(path, value):
    """Write one UTF-8 JSON document without non-standard numbers."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            value,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        file.write("\n")
    return str(output_path.resolve())


def _write_csv(path, columns, records):
    output_path = Path(path)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)
    return str(output_path.resolve())


def save_comparison_artifacts(
    run_dir,
    *,
    manifest,
    per_run_records,
    summary_records,
):
    """Persist the aggregate comparison manifest and result tables."""
    run_path = Path(run_dir).resolve()
    if not run_path.is_dir():
        raise ValueError(
            f"run_dir must be an existing directory, got {run_path}"
        )

    paths = {
        "manifest": write_strict_json(
            run_path / "manifest.json",
            manifest,
        ),
        "per_run": _write_csv(
            run_path / "per_run.csv",
            PER_RUN_COLUMNS,
            per_run_records,
        ),
        "summary_csv": _write_csv(
            run_path / "summary.csv",
            SUMMARY_COLUMNS,
            summary_records,
        ),
        "summary_json": write_strict_json(
            run_path / "summary.json",
            {
                "suite": "cec2017",
                "group_by": [
                    "problem_id",
                    "method",
                    "training_seed",
                ],
                "records": summary_records,
            },
        ),
    }
    return paths


__all__ = [
    "PER_RUN_COLUMNS",
    "SUMMARY_COLUMNS",
    "save_comparison_artifacts",
    "write_strict_json",
]
