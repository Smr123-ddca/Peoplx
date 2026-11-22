"""
utils.py
--------
Utility functions for file management, dataset scanning, and CSV export.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# Supported image extensions (case-insensitive)
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Standard dataset categories
CATEGORIES = ("low_density", "medium_density", "high_density")


def ensure_directories(project_root: Path) -> Dict[str, Path]:
    """
    Ensure all required dataset, model, and output directories exist.

    Args:
        project_root: The root directory of the project.

    Returns:
        Dict mapping directory identifiers to their Path objects.
    """
    paths = {
        "dataset": project_root / "dataset",
        "output": project_root / "output",
        "models": project_root / "models",
    }

    # Ensure main directories exist
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    # Ensure category subdirectories exist in both dataset and output
    for category in CATEGORIES:
        (paths["dataset"] / category).mkdir(parents=True, exist_ok=True)
        (paths["output"] / category).mkdir(parents=True, exist_ok=True)

    return paths


def collect_dataset_images(dataset_dir: Path) -> List[Tuple[Path, str]]:
    """
    Scan category folders for supported images (JPG, JPEG, PNG).

    Args:
        dataset_dir: Path to the dataset root folder.

    Returns:
        A list of tuples: (image_path, category_name)
    """
    collected = []

    for category in CATEGORIES:
        category_dir = dataset_dir / category
        if not category_dir.exists():
            continue

        # Sort files for deterministic processing order
        for file_path in sorted(category_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                collected.append((file_path, category))

    return collected


# Standard CSV columns
PROMPT3_COLUMNS = [
    "image_filename",
    "dataset_category",
    "detected_person_count",
    "inference_time_seconds",
    "image_width",
    "image_height",
    "processing_status",
]

LEGACY_COLUMNS = [
    "Image Filename",
    "Dataset Category",
    "Detected Person Count",
    "Processing Time (s)",
]


def save_results_to_csv(records: List[Dict], output_file: Path) -> None:
    """
    Export numerical detection results to a CSV file.

    Args:
        records: List of result dictionaries containing image metrics and counts.
        output_file: Destination Path for the CSV file.
    """
    # Ensure parent output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)

    if df.empty:
        df = pd.DataFrame(columns=PROMPT3_COLUMNS)
    else:
        # Determine column order based on present fields
        if "image_filename" in df.columns:
            ordered = [col for col in PROMPT3_COLUMNS if col in df.columns]
        elif "Image Filename" in df.columns:
            ordered = [col for col in LEGACY_COLUMNS if col in df.columns]
        else:
            ordered = list(df.columns)
        df = df[ordered]

    df.to_csv(output_file, index=False)
