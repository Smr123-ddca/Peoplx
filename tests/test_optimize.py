"""
test_optimize.py
----------------
Unit tests for the model parameter optimization module (src/optimize.py).
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.optimize import (
    CONF_THRESHOLDS,
    IMAGE_SIZES,
    parse_arguments,
    print_comparison_and_recommendation,
)


def test_parameter_grid_definitions():
    """Verify the grid contains all required thresholds and image sizes."""
    assert CONF_THRESHOLDS == [0.15, 0.25, 0.35]
    assert IMAGE_SIZES == [320, 416, 640]
    assert len(CONF_THRESHOLDS) * len(IMAGE_SIZES) == 9


def test_parse_arguments_defaults():
    """Verify default CLI arguments."""
    sys_argv_backup = sys.argv
    try:
        sys.argv = ["optimize.py"]
        args = parse_arguments()
        assert Path(args.dataset_dir) == PROJECT_ROOT / "dataset"
        assert Path(args.ground_truth) == PROJECT_ROOT / "dataset" / "ground_truth.csv"
        assert Path(args.output_csv) == PROJECT_ROOT / "output" / "optimization_results.csv"
        assert args.device == "cpu"
    finally:
        sys.argv = sys_argv_backup


def test_print_comparison_and_recommendation():
    """Verify that best MAE, best MAPE, and fastest configs are correctly identified."""
    mock_results = [
        {
            "confidence_threshold": 0.15,
            "image_size": 320,
            "mae": 7.50,
            "mape": 35.00,
            "avg_inference_time_ms": 120.0,
            "avg_predicted_count": 12.0,
            "total_detections": 450,
        },
        {
            "confidence_threshold": 0.25,
            "image_size": 416,
            "mae": 5.20,  # Best MAE
            "mape": 28.50,
            "avg_inference_time_ms": 180.0,
            "avg_predicted_count": 15.0,
            "total_detections": 570,
        },
        {
            "confidence_threshold": 0.35,
            "image_size": 320,
            "mae": 8.10,
            "mape": 40.00,
            "avg_inference_time_ms": 110.0,  # Fastest
            "avg_predicted_count": 10.0,
            "total_detections": 380,
        },
        {
            "confidence_threshold": 0.15,
            "image_size": 640,
            "mae": 5.40,
            "mape": 24.10,  # Best MAPE
            "avg_inference_time_ms": 320.0,
            "avg_predicted_count": 16.5,
            "total_detections": 627,
        },
    ]

    analysis = print_comparison_and_recommendation(mock_results)
    assert analysis["best_mae"]["mae"] == 5.20
    assert analysis["best_mae"]["image_size"] == 416

    assert analysis["best_mape"]["mape"] == 24.10
    assert analysis["best_mape"]["confidence_threshold"] == 0.15

    assert analysis["fastest"]["avg_inference_time_ms"] == 110.0
    assert analysis["fastest"]["confidence_threshold"] == 0.35
