import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

from learning_ddpg.evaluation.evaluate_actor import (
    DEFAULT_VALIDATION_SEEDS,
    evaluate_checkpoint,
)


REPORT_FIELDS = [
    "function_id",
    "rank",
    "selected",
    "training_seed",
    "training_episode",
    "model_path",
    "gap_mean",
    "gap_std",
    "gbest_mean",
    "gbest_std",
    "reward_mean",
    "reward_std",
]


def find_candidate_models(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    candidates = sorted(checkpoint_dir.rglob("actor_episode_*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No actor_episode_*.pt candidates found under: {checkpoint_dir}"
        )
    return candidates


def evaluate_candidates(candidate_paths, seeds, device):
    results = []

    for index, model_path in enumerate(candidate_paths, start=1):
        print(
            f"evaluating candidate {index}/{len(candidate_paths)}: "
            f"{model_path}"
        )
        result = evaluate_checkpoint(model_path, seeds=seeds, device=device)
        results.append(result)
        print(
            f"  F{result['function_id']:02d} "
            f"train_seed={result['training_seed']} "
            f"episode={result['training_episode']} "
            f"gap={result['gap_mean']:.6e} +/- {result['gap_std']:.6e}"
        )

    return results


def rank_candidates_per_function(results):
    grouped = defaultdict(list)
    for result in results:
        grouped[int(result["function_id"])].append(result)

    for function_id, function_results in grouped.items():
        function_results.sort(
            key=lambda item: (
                item["gap_mean"],
                item["gap_std"],
                -1 if item["training_seed"] is None else int(item["training_seed"]),
                item["training_episode"],
                item["model_path"],
            )
        )
        for rank, result in enumerate(function_results, start=1):
            result["rank"] = rank
            result["selected"] = rank == 1

    return dict(sorted(grouped.items()))


def write_csv(path, results):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result[field] for field in REPORT_FIELDS})


def save_selection_reports(grouped_results, seeds, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_ranked_results = []
    selected_models = []

    for function_id, function_results in grouped_results.items():
        function_dir = output_dir / f"F{function_id:02d}"
        function_dir.mkdir(parents=True, exist_ok=True)

        best_result = function_results[0]
        best_source_path = Path(best_result["model_path"])
        best_actor_path = function_dir / "best_actor.pt"
        shutil.copy2(best_source_path, best_actor_path)

        function_csv_path = function_dir / "selected_models.csv"
        write_csv(function_csv_path, function_results)

        function_summary = {
            "function_id": function_id,
            "function_optimum": best_result["function_optimum"],
            "selection_metric": "mean_final_optimality_gap",
            "validation_seeds": [int(seed) for seed in seeds],
            "best_source_model": str(best_source_path),
            "best_actor_path": str(best_actor_path.resolve()),
            "best_training_seed": best_result["training_seed"],
            "best_training_episode": best_result["training_episode"],
            "best_gap_mean": best_result["gap_mean"],
            "best_gap_std": best_result["gap_std"],
            "candidates": function_results,
        }
        with (function_dir / "model_selection.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(function_summary, file, ensure_ascii=False, indent=2)

        selected_models.append(function_summary)
        all_ranked_results.extend(function_results)

    root_csv_path = output_dir / "selected_models.csv"
    write_csv(root_csv_path, all_ranked_results)

    root_json_path = output_dir / "model_selection.json"
    with root_json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "selection_metric": "mean_final_optimality_gap_per_function",
                "validation_seeds": [int(seed) for seed in seeds],
                "selected_models": selected_models,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "selected_models": selected_models,
        "csv_path": root_csv_path,
        "json_path": root_json_path,
    }


def select_best_models(
    checkpoint_dir,
    seeds=DEFAULT_VALIDATION_SEEDS,
    device="cpu",
    output_dir=None,
):
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir) if output_dir else checkpoint_dir

    candidates = find_candidate_models(checkpoint_dir)
    results = evaluate_candidates(candidates, seeds, device)
    grouped_results = rank_candidates_per_function(results)
    reports = save_selection_reports(grouped_results, seeds, output_dir)

    return {
        "grouped_results": grouped_results,
        **reports,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select one best Actor per CEC2013 function."
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_VALIDATION_SEEDS),
        help="Validation seeds; final test seeds must remain independent.",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    selection = select_best_models(
        checkpoint_dir=args.checkpoint_dir,
        seeds=args.seeds,
        device=args.device,
        output_dir=args.output_dir,
    )

    for selected in selection["selected_models"]:
        print(
            f"selected F{selected['function_id']:02d} Actor | "
            f"train_seed={selected['best_training_seed']} | "
            f"episode={selected['best_training_episode']} | "
            f"gap={selected['best_gap_mean']:.6e} "
            f"+/- {selected['best_gap_std']:.6e}"
        )

    print(f"combined ranking CSV: {selection['csv_path']}")
    print(f"selection summary JSON: {selection['json_path']}")


if __name__ == "__main__":
    main()
