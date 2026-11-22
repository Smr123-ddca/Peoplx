"""
evaluate.py
-----------
CLI tool to evaluate crowd counting accuracy and benchmark YOLO performance.
Compares YOLO detections with ground-truth manual counts.
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import (
    evaluate_accuracy,
    format_accuracy_summary,
    generate_ground_truth_template,
    save_accuracy_results,
)


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate crowd counting accuracy against manual ground truth."
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default=str(PROJECT_ROOT / "output" / "crowd_estimation_results.csv"),
        help="Path to YOLO predictions CSV.",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "ground_truth.csv"),
        help="Path to ground truth annotations CSV.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(PROJECT_ROOT / "dataset"),
        help="Path to dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "output"),
        help="Path to output directory.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactively prompt for ground-truth counts in terminal.",
    )
    return parser.parse_args()


def interactive_ground_truth_input(gt_path: Path) -> None:
    """Interactive CLI prompt to enter ground-truth counts image-by-image."""
    if not gt_path.exists():
        print(f"[Error] Ground-truth file not found: {gt_path}")
        return

    df = pd.read_csv(gt_path)
    total = len(df)
    print("\n" + "=" * 65)
    print("      INTERACTIVE GROUND-TRUTH MANUAL ENTRY")
    print("=" * 65)
    print("Enter the actual person count for each image.")
    print("Press [Enter] to leave existing/blank count unchanged.")
    print("Type 'q' and press [Enter] to save and exit anytime.\n")

    modified = False
    for idx, row in df.iterrows():
        fname = row["image_filename"]
        category = row["dataset_category"]
        current_val = row["actual_person_count"]
        curr_str = f" [Current: {int(current_val)}]" if pd.notna(current_val) and str(current_val).strip() != "" else " [Current: Blank]"

        try:
            user_input = input(f"[{idx+1}/{total}] {category}/{fname}{curr_str} -> Count: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive input...")
            break

        if user_input.lower() == "q":
            break
        if user_input == "":
            continue

        try:
            count = int(user_input)
            df.at[idx, "actual_person_count"] = count
            modified = True
        except ValueError:
            print("  [Warning] Invalid number. Skipping.")

    if modified:
        df.to_csv(gt_path, index=False)
        print(f"\n[Success] Ground-truth updates saved to: {gt_path}\n")
    else:
        print("\nNo changes made.")


def run_evaluation() -> None:
    """Execute evaluation workflow."""
    args = parse_arguments()

    pred_path = Path(args.predictions)
    gt_path = Path(args.ground_truth)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    print("=" * 65)
    print("        CROWD COUNTING ACCURACY EVALUATION")
    print("=" * 65)
    print(f"Predictions File  : {pred_path}")
    print(f"Ground Truth File : {gt_path}")
    print(f"Output Directory  : {output_dir}")
    print("=" * 65)

    # Step 1: Ensure ground truth template exists with all dataset images
    if not gt_path.exists():
        print(f"\n[Info] Generating ground-truth template with all dataset images...")
        generate_ground_truth_template(dataset_dir, gt_path)
        print(f"[Created] {gt_path}")

    # Optional interactive entry mode
    if args.interactive:
        interactive_ground_truth_input(gt_path)

    # Step 2: Check predictions file
    if not pred_path.exists():
        print(f"\n[Error] Predictions file '{pred_path}' not found!")
        print("Please run the detector first using: python src/main.py\n")
        sys.exit(1)

    pred_df = pd.read_csv(pred_path)
    gt_df = pd.read_csv(gt_path)

    # Step 3: Load benchmark metrics if available
    benchmark_file = output_dir / "performance_benchmark.json"
    benchmark_metrics = None
    if benchmark_file.exists():
        try:
            with open(benchmark_file, "r", encoding="utf-8") as f:
                benchmark_metrics = json.load(f)
        except Exception:
            pass

    # Step 4: Evaluate accuracy
    results = evaluate_accuracy(pred_df, gt_df)

    # Step 5: Format summary
    summary_text = format_accuracy_summary(results, benchmark_metrics)

    # Step 6: Save results
    csv_file, txt_file = save_accuracy_results(results["detailed_records"], summary_text, output_dir)

    # Step 7: Display summary in terminal
    print("\n" + summary_text + "\n")
    print(f"[Results Saved]")
    print(f"  - Detailed CSV : {csv_file}")
    print(f"  - Summary Text : {txt_file}")
    print("=" * 65)


if __name__ == "__main__":
    run_evaluation()
