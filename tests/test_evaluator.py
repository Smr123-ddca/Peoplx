"""
test_evaluator.py
-----------------
Unit tests for accuracy evaluation functions, error metrics (MAE, MAPE),
ground-truth template generation, and reporting.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.evaluator import (
    calculate_mae,
    calculate_mape,
    evaluate_accuracy,
    format_accuracy_summary,
    generate_ground_truth_template,
    save_accuracy_results,
)


class TestEvaluatorMetrics:
    """Unit tests for MAE, MAPE, and edge-case handling."""

    def test_calculate_mae_known_values(self):
        """
        Verify MAE calculation on known numbers:
        Preds: [10, 5, 15]
        Acts:  [8,  5, 20]
        Abs Errors: [2, 0, 5] -> Sum = 7 -> MAE = 7/3 = 2.3333...
        """
        preds = [10.0, 5.0, 15.0]
        acts = [8.0, 5.0, 20.0]
        mae = calculate_mae(preds, acts)
        assert mae is not None
        assert mae == pytest.approx(7.0 / 3.0, rel=1e-4)

    def test_calculate_mae_perfect_accuracy(self):
        """Verify MAE is 0.0 when predictions perfectly match ground truth."""
        preds = [3.0, 12.0, 25.0]
        acts = [3.0, 12.0, 25.0]
        assert calculate_mae(preds, acts) == 0.0

    def test_calculate_mae_empty(self):
        """Verify calculate_mae handles empty lists gracefully."""
        assert calculate_mae([], []) is None

    def test_calculate_mape_known_values(self):
        """
        Verify MAPE calculation on known numbers:
        Pred: 10, Act: 8 -> |10-8|/8 = 2/8 = 25.0%
        Pred: 5,  Act: 5 -> |5-5|/5 = 0.0%
        Pred: 15, Act: 20 -> |15-20|/20 = 5/20 = 25.0%
        Mean MAPE = (25 + 0 + 25) / 3 = 16.6667%
        """
        preds = [10.0, 5.0, 15.0]
        acts = [8.0, 5.0, 20.0]
        mape = calculate_mape(preds, acts)
        assert mape is not None
        assert mape == pytest.approx(50.0 / 3.0, rel=1e-4)

    def test_calculate_mape_zero_actual_handling(self):
        """
        Verify that images with 0 actual count are excluded from MAPE
        to prevent division by zero, without breaking the calculation.
        """
        preds = [2.0, 10.0]
        acts = [0.0, 8.0]  # First image has 0 actual count
        mape = calculate_mape(preds, acts)
        assert mape is not None
        # Only the second image should be evaluated: |10 - 8| / 8 * 100 = 25.0%
        assert mape == pytest.approx(25.0, rel=1e-4)

    def test_calculate_mape_all_zero_actuals(self):
        """Verify calculate_mape returns None if all actual counts are 0."""
        preds = [1.0, 2.0]
        acts = [0.0, 0.0]
        assert calculate_mape(preds, acts) is None


class TestEvaluatorWorkflow:
    """Tests verifying dataset merging, missing-entry handling, and report generation."""

    def test_evaluate_accuracy_all_missing_ground_truth(self):
        """Verify behavior when all ground-truth entries are blank/missing."""
        pred_df = pd.DataFrame([
            {"image_filename": "img1.png", "dataset_category": "low_density", "detected_person_count": 3},
            {"image_filename": "img2.png", "dataset_category": "medium_density", "detected_person_count": 8},
        ])
        gt_df = pd.DataFrame([
            {"image_filename": "img1.png", "dataset_category": "low_density", "actual_person_count": ""},
            {"image_filename": "img2.png", "dataset_category": "medium_density", "actual_person_count": None},
        ])

        results = evaluate_accuracy(pred_df, gt_df)

        assert results["total_images"] == 2
        assert results["evaluated_images"] == 0
        assert results["missing_images"] == 2
        assert results["mae"] is None
        assert results["mape"] is None

        for rec in results["detailed_records"]:
            assert rec["evaluation_status"] == "MISSING_GROUND_TRUTH"

    def test_evaluate_accuracy_partial_ground_truth(self):
        """
        Verify that images with ground truth are evaluated,
        and missing ones are skipped cleanly without crashing.
        """
        pred_df = pd.DataFrame([
            {"image_filename": "img1.png", "dataset_category": "low_density", "detected_person_count": 4},
            {"image_filename": "img2.png", "dataset_category": "low_density", "detected_person_count": 7},
            {"image_filename": "img3.png", "dataset_category": "high_density", "detected_person_count": 22},
        ])
        gt_df = pd.DataFrame([
            {"image_filename": "img1.png", "dataset_category": "low_density", "actual_person_count": 5},
            {"image_filename": "img2.png", "dataset_category": "low_density", "actual_person_count": ""},  # missing
            {"image_filename": "img3.png", "dataset_category": "high_density", "actual_person_count": 20},
        ])

        results = evaluate_accuracy(pred_df, gt_df)

        assert results["total_images"] == 3
        assert results["evaluated_images"] == 2
        assert results["missing_images"] == 1

        # Evaluated: img1 (abs err |4-5|=1), img3 (abs err |22-20|=2)
        # Overall MAE = (1 + 2) / 2 = 1.5
        assert results["mae"] == pytest.approx(1.5)
        # Avg Predicted = (4 + 22) / 2 = 13.0
        assert results["avg_predicted"] == pytest.approx(13.0)
        # Avg Actual = (5 + 20) / 2 = 12.5
        assert results["avg_actual"] == pytest.approx(12.5)

    def test_generate_ground_truth_template(self, tmp_path):
        """Verify template generation creates required columns and preserves existing values."""
        dataset_dir = tmp_path / "dataset"
        (dataset_dir / "low_density").mkdir(parents=True)
        (dataset_dir / "low_density" / "test1.png").write_text("fake")
        (dataset_dir / "low_density" / "test2.png").write_text("fake")

        gt_file = tmp_path / "ground_truth.csv"
        generate_ground_truth_template(dataset_dir, gt_file)

        assert gt_file.is_file()
        df = pd.read_csv(gt_file)
        assert list(df.columns) == ["image_filename", "dataset_category", "actual_person_count"]
        assert len(df) == 2
        # Counts should be blank initially
        assert pd.isna(df.iloc[0]["actual_person_count"])

        # Enter a value for test1.png
        df.at[0, "actual_person_count"] = 4
        df.to_csv(gt_file, index=False)

        # Re-generate: should PRESERVE 4 for test1.png
        generate_ground_truth_template(dataset_dir, gt_file)
        df_re = pd.read_csv(gt_file)
        assert int(df_re.loc[df_re["image_filename"] == "test1.png", "actual_person_count"].values[0]) == 4

    def test_save_accuracy_results(self, tmp_path):
        """Verify saving accuracy_results.csv and accuracy_summary.txt."""
        records = [
            {
                "image_filename": "sample.png",
                "dataset_category": "low_density",
                "predicted_person_count": 3,
                "actual_person_count": 4,
                "absolute_error": 1,
                "percentage_error": 25.0,
                "evaluation_status": "EVALUATED",
            }
        ]
        summary_text = "EVALUATION SUMMARY DUMMY TEXT"

        csv_path, txt_path = save_accuracy_results(records, summary_text, tmp_path)

        assert csv_path.is_file()
        assert txt_path.is_file()

        df = pd.read_csv(csv_path)
        expected_cols = [
            "image_filename",
            "dataset_category",
            "predicted_person_count",
            "actual_person_count",
            "absolute_error",
            "percentage_error",
            "evaluation_status",
        ]
        assert list(df.columns) == expected_cols
        assert txt_path.read_text(encoding="utf-8") == summary_text
