"""
evaluator.py
------------
Evaluation and benchmarking module for crowd counting accuracy.
Calculates MAE, MAPE, per-category accuracy breakdown, and performance benchmarks.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.utils import CATEGORIES, collect_dataset_images


def generate_ground_truth_template(dataset_dir: Path, output_file: Path) -> Path:
    """
    Generate or update the ground_truth.csv template containing all dataset images.
    Preserves any already entered ground-truth counts.

    Args:
        dataset_dir: Path to dataset directory containing category subfolders.
        output_file: Path where ground_truth.csv will be saved.

    Returns:
        Path to the saved ground_truth.csv file.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Collect all images currently in dataset
    images = collect_dataset_images(dataset_dir)

    existing_counts = {}
    if output_file.exists():
        try:
            existing_df = pd.read_csv(output_file)
            for _, row in existing_df.iterrows():
                fname = str(row.get("image_filename", "")).strip()
                val = row.get("actual_person_count", None)
                if pd.notna(val) and str(val).strip() != "":
                    try:
                        existing_counts[fname] = int(round(float(val)))
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass

    records = []
    for img_path, category in images:
        filename = img_path.name
        actual = existing_counts.get(filename, "")
        records.append({
            "image_filename": filename,
            "dataset_category": category,
            "actual_person_count": actual,
        })

    df = pd.DataFrame(records, columns=["image_filename", "dataset_category", "actual_person_count"])
    df.to_csv(output_file, index=False)
    return output_file


def calculate_mae(predictions: List[float], actuals: List[float]) -> Optional[float]:
    """
    Calculate Mean Absolute Error (MAE).
    Formula: MAE = Sum of absolute counting errors / Number of evaluated images.

    Args:
        predictions: List of predicted counts.
        actuals: List of ground-truth counts.

    Returns:
        MAE float, or None if no samples.
    """
    if not predictions or not actuals or len(predictions) != len(actuals):
        return None
    abs_errors = [abs(p - a) for p, a in zip(predictions, actuals)]
    return float(np.mean(abs_errors))


def calculate_mape(predictions: List[float], actuals: List[float]) -> Optional[float]:
    """
    Calculate Mean Absolute Percentage Error (MAPE), strictly excluding zero actual counts.

    Args:
        predictions: List of predicted counts.
        actuals: List of ground-truth counts.

    Returns:
        MAPE percentage float (e.g. 15.2 for 15.2%), or None if no non-zero actuals.
    """
    if not predictions or not actuals or len(predictions) != len(actuals):
        return None

    pct_errors = []
    for p, a in zip(predictions, actuals):
        if a > 0:
            pct_errors.append(abs(p - a) / a * 100.0)

    if not pct_errors:
        return None
    return float(np.mean(pct_errors))


def evaluate_accuracy(
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Evaluate predicted crowd counts against ground-truth annotations.

    Args:
        predictions_df: DataFrame containing at least 'image_filename' and 'detected_person_count'.
        ground_truth_df: DataFrame containing 'image_filename' and 'actual_person_count'.

    Returns:
        Dictionary containing overall metrics, category breakdown, and detailed records.
    """
    # Clean whitespace in filenames for safe merging
    pred_df = predictions_df.copy()
    gt_df = ground_truth_df.copy()

    pred_df["image_filename"] = pred_df["image_filename"].astype(str).str.strip()
    gt_df["image_filename"] = gt_df["image_filename"].astype(str).str.strip()

    # Determine predicted count column name
    pred_col = "detected_person_count"
    if pred_col not in pred_df.columns and "Detected Person Count" in pred_df.columns:
        pred_col = "Detected Person Count"

    # Merge predictions with ground truth
    merged = pd.merge(
        pred_df,
        gt_df[["image_filename", "actual_person_count"]],
        on="image_filename",
        how="left",
    )

    detailed_records = []
    eval_predictions = []
    eval_actuals = []
    eval_categories = []

    for _, row in merged.iterrows():
        fname = row["image_filename"]
        category = row.get("dataset_category", "unknown")
        pred_count = int(row.get(pred_col, 0))

        actual_raw = row.get("actual_person_count", None)

        is_valid_gt = False
        actual_count = None
        if pd.notna(actual_raw) and str(actual_raw).strip() != "":
            try:
                actual_count = int(round(float(actual_raw)))
                is_valid_gt = True
            except (ValueError, TypeError):
                is_valid_gt = False

        if is_valid_gt and actual_count is not None:
            abs_err = abs(pred_count - actual_count)
            pct_err = round((abs_err / actual_count * 100.0), 2) if actual_count > 0 else None
            status = "EVALUATED"

            eval_predictions.append(pred_count)
            eval_actuals.append(actual_count)
            eval_categories.append(category)
        else:
            abs_err = None
            pct_err = None
            status = "MISSING_GROUND_TRUTH"

        detailed_records.append({
            "image_filename": fname,
            "dataset_category": category,
            "predicted_person_count": pred_count,
            "actual_person_count": actual_count if actual_count is not None else "",
            "absolute_error": abs_err if abs_err is not None else "",
            "percentage_error": pct_err if pct_err is not None else "",
            "evaluation_status": status,
        })

    total_images = len(detailed_records)
    evaluated_count = len(eval_predictions)
    missing_count = total_images - evaluated_count

    # Calculate overall metrics
    overall_mae = calculate_mae(eval_predictions, eval_actuals)
    overall_mape = calculate_mape(eval_predictions, eval_actuals)
    avg_pred = float(np.mean(eval_predictions)) if eval_predictions else None
    avg_actual = float(np.mean(eval_actuals)) if eval_actuals else None

    # Calculate category breakdown
    category_metrics = {}
    for cat in CATEGORIES:
        cat_preds = [p for p, c in zip(eval_predictions, eval_categories) if c == cat]
        cat_acts = [a for a, c in zip(eval_actuals, eval_categories) if c == cat]
        cat_total = sum(1 for r in detailed_records if r["dataset_category"] == cat)
        cat_eval_count = len(cat_preds)

        category_metrics[cat] = {
            "total_images": cat_total,
            "evaluated_images": cat_eval_count,
            "missing_images": cat_total - cat_eval_count,
            "mae": calculate_mae(cat_preds, cat_acts),
            "mape": calculate_mape(cat_preds, cat_acts),
            "avg_predicted": float(np.mean(cat_preds)) if cat_preds else None,
            "avg_actual": float(np.mean(cat_acts)) if cat_acts else None,
        }

    return {
        "total_images": total_images,
        "evaluated_images": evaluated_count,
        "missing_images": missing_count,
        "mae": overall_mae,
        "mape": overall_mape,
        "avg_predicted": avg_pred,
        "avg_actual": avg_actual,
        "category_metrics": category_metrics,
        "detailed_records": detailed_records,
    }


def format_accuracy_summary(
    eval_metrics: Dict[str, Any],
    benchmark_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Format a comprehensive human-readable summary report.

    Args:
        eval_metrics: Output from evaluate_accuracy().
        benchmark_metrics: Optional performance benchmarking metrics.

    Returns:
        Formatted summary string.
    """
    lines = []
    lines.append("=" * 72)
    lines.append("              CROWD COUNTING ACCURACY EVALUATION REPORT")
    lines.append("=" * 72)
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total Dataset Images      : {eval_metrics['total_images']}")
    lines.append(f"Evaluated Images (with GT): {eval_metrics['evaluated_images']}")
    lines.append(f"Skipped Images (missing GT): {eval_metrics['missing_images']}")

    lines.append("\n" + "-" * 72)
    lines.append("1. OVERALL ACCURACY METRICS")
    lines.append("-" * 72)

    if eval_metrics["evaluated_images"] > 0:
        mae_str = f"{eval_metrics['mae']:.2f} people" if eval_metrics['mae'] is not None else "N/A"
        mape_str = f"{eval_metrics['mape']:.2f}%" if eval_metrics['mape'] is not None else "N/A"
        avg_pred_str = f"{eval_metrics['avg_predicted']:.2f}" if eval_metrics['avg_predicted'] is not None else "N/A"
        avg_act_str = f"{eval_metrics['avg_actual']:.2f}" if eval_metrics['avg_actual'] is not None else "N/A"

        lines.append(f"Mean Absolute Error (MAE)       : {mae_str}")
        lines.append(f"Mean Absolute Pct Error (MAPE)  : {mape_str}")
        lines.append(f"Average Predicted Person Count  : {avg_pred_str}")
        lines.append(f"Average Actual Person Count     : {avg_act_str}")
        lines.append("\nFormula: MAE = Sum of absolute counting errors / Number of evaluated images")
        lines.append("Note: MAPE excludes images with 0 actual count to prevent division by zero.")
    else:
        lines.append("STATUS: NO GROUND TRUTH ENTERED YET.")
        lines.append("All 38 images currently have blank ground-truth counts.")
        lines.append("Please enter actual counts in 'dataset/ground_truth.csv' and run evaluation.")

    lines.append("\n" + "-" * 72)
    lines.append("2. ACCURACY BREAKDOWN BY DENSITY CATEGORY")
    lines.append("-" * 72)

    for cat in ["low_density", "medium_density", "high_density"]:
        cdata = eval_metrics["category_metrics"].get(cat, {})
        lines.append(f"[{cat.upper()}]")
        lines.append(f"  Total Images In Category   : {cdata.get('total_images', 0)}")
        lines.append(f"  Evaluated with Ground Truth: {cdata.get('evaluated_images', 0)}")
        lines.append(f"  Missing Ground Truth       : {cdata.get('missing_images', 0)}")

        if cdata.get("evaluated_images", 0) > 0:
            c_mae = f"{cdata['mae']:.2f}" if cdata['mae'] is not None else "N/A"
            c_mape = f"{cdata['mape']:.2f}%" if cdata['mape'] is not None else "N/A"
            c_pred = f"{cdata['avg_predicted']:.2f}" if cdata['avg_predicted'] is not None else "N/A"
            c_act = f"{cdata['avg_actual']:.2f}" if cdata['avg_actual'] is not None else "N/A"
            lines.append(f"  Mean Absolute Error (MAE)  : {c_mae} people")
            lines.append(f"  Mean Absolute Pct (MAPE)   : {c_mape}")
            lines.append(f"  Average Predicted Count    : {c_pred}")
            lines.append(f"  Average Actual Count       : {c_act}")
        else:
            lines.append("  Metrics: Pending ground-truth entry.")
        lines.append("")

    if benchmark_metrics:
        lines.append("-" * 72)
        lines.append("3. PERFORMANCE BENCHMARKING (CPU)")
        lines.append("-" * 72)
        lines.append(f"Model Loading Time              : {benchmark_metrics.get('model_loading_seconds', 0.0):.4f} s")
        lines.append(f"First-Image Warm-Up Latency     : {benchmark_metrics.get('warmup_inference_ms', 0.0):.1f} ms")
        lines.append(f"Steady-State Average Latency    : {benchmark_metrics.get('steady_state_inference_ms', 0.0):.1f} ms")
        lines.append(f"Average OpenCV Annotation Time  : {benchmark_metrics.get('avg_annotation_ms', 0.0):.1f} ms")
        lines.append(f"Average Image Saving Time       : {benchmark_metrics.get('avg_save_ms', 0.0):.1f} ms")
        lines.append(f"Total Processing Time           : {benchmark_metrics.get('total_pipeline_seconds', 0.0):.2f} s")
        lines.append("\nNote: First-image PyTorch/MKL warm-up is excluded from the steady-state")
        lines.append("average inference time and reported separately above.")

    lines.append("=" * 72)
    return "\n".join(lines)


def save_accuracy_results(
    detailed_records: List[Dict[str, Any]],
    summary_text: str,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """
    Save output/accuracy_results.csv and output/accuracy_summary.txt.

    Args:
        detailed_records: List of record dictionaries.
        summary_text: Preformatted summary string.
        output_dir: Destination output directory.

    Returns:
        Tuple of (csv_path, summary_txt_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "accuracy_results.csv"
    summary_path = output_dir / "accuracy_summary.txt"

    # Export CSV
    df = pd.DataFrame(detailed_records)
    cols = [
        "image_filename",
        "dataset_category",
        "predicted_person_count",
        "actual_person_count",
        "absolute_error",
        "percentage_error",
        "evaluation_status",
    ]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(csv_path, index=False)

    # Export Summary Text
    summary_path.write_text(summary_text, encoding="utf-8")

    return csv_path, summary_path
