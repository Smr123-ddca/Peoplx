"""
test_pipeline.py
----------------
Integration tests verifying pipeline utilities and CSV results export.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.utils import (
    CATEGORIES,
    SUPPORTED_EXTENSIONS,
    collect_dataset_images,
    ensure_directories,
    save_results_to_csv,
)


class TestPipelineIntegration:
    """Tests verifying directory setup, image discovery, and CSV results formatting."""

    def test_ensure_directories(self, tmp_path):
        """Verify all category subdirectories are created in dataset and output."""
        paths = ensure_directories(tmp_path)

        assert paths["dataset"].is_dir()
        assert paths["output"].is_dir()
        assert paths["models"].is_dir()

        for category in CATEGORIES:
            assert (paths["dataset"] / category).is_dir()
            assert (paths["output"] / category).is_dir()

    def test_collect_dataset_images(self, tmp_path):
        """Verify supported image formats (.jpg, .jpeg, .png) are discovered across categories."""
        dataset_dir = tmp_path / "dataset"
        for cat in CATEGORIES:
            (dataset_dir / cat).mkdir(parents=True)

        # Create supported files
        (dataset_dir / "low_density" / "img1.jpg").write_text("fake_jpg")
        (dataset_dir / "medium_density" / "img2.PNG").write_text("fake_png")
        (dataset_dir / "high_density" / "img3.jpeg").write_text("fake_jpeg")
        # Unsupported file
        (dataset_dir / "low_density" / "ignore.txt").write_text("ignore")

        images = collect_dataset_images(dataset_dir)
        discovered_names = [img_path.name for img_path, cat in images]

        assert "img1.jpg" in discovered_names
        assert "img2.PNG" in discovered_names
        assert "img3.jpeg" in discovered_names
        assert "ignore.txt" not in discovered_names
        assert len(images) == 3

    def test_save_results_to_csv(self, tmp_path):
        """Verify CSV export contains the required 4 columns and correct data."""
        csv_path = tmp_path / "output" / "results.csv"

        records = [
            {
                "Image Filename": "test1.jpg",
                "Dataset Category": "low_density",
                "Detected Person Count": 3,
                "Processing Time (s)": 0.0452,
            },
            {
                "Image Filename": "test2.png",
                "Dataset Category": "medium_density",
                "Detected Person Count": 12,
                "Processing Time (s)": 0.0811,
            },
        ]

        save_results_to_csv(records, csv_path)

        assert csv_path.is_file()
        df = pd.read_csv(csv_path)

        expected_columns = [
            "Image Filename",
            "Dataset Category",
            "Detected Person Count",
            "Processing Time (s)",
        ]
        assert list(df.columns) == expected_columns
        assert len(df) == 2
        assert df.iloc[0]["Detected Person Count"] == 3
        assert df.iloc[1]["Dataset Category"] == "medium_density"

    def test_save_results_to_csv_prompt3_format(self, tmp_path):
        """Verify CSV export contains the Prompt 3 7-column schema."""
        csv_path = tmp_path / "output" / "crowd_estimation_results.csv"

        records = [
            {
                "image_filename": "screen1.png",
                "dataset_category": "low_density",
                "detected_person_count": 2,
                "inference_time_seconds": 0.0521,
                "image_width": 1080,
                "image_height": 620,
                "processing_status": "SUCCESS",
            },
            {
                "image_filename": "screen2.png",
                "dataset_category": "high_density",
                "detected_person_count": 28,
                "inference_time_seconds": 0.0784,
                "image_width": 1200,
                "image_height": 750,
                "processing_status": "SUCCESS",
            },
        ]

        save_results_to_csv(records, csv_path)

        assert csv_path.is_file()
        df = pd.read_csv(csv_path)

        expected_columns = [
            "image_filename",
            "dataset_category",
            "detected_person_count",
            "inference_time_seconds",
            "image_width",
            "image_height",
            "processing_status",
        ]
        assert list(df.columns) == expected_columns
        assert len(df) == 2
        assert df.iloc[0]["detected_person_count"] == 2
        assert df.iloc[1]["image_width"] == 1200
        assert df.iloc[1]["processing_status"] == "SUCCESS"
