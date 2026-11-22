"""
main.py
-------
Crowd Estimation Pipeline Orchestrator.
Processes images one by one across dataset density folders using YOLO and OpenCV.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path to allow running both from project root and inside src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import PersonDetector
from src.image_processor import draw_detections, load_image, save_annotated_image
from src.utils import collect_dataset_images, ensure_directories, save_results_to_csv


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the crowd estimation pipeline."""
    parser = argparse.ArgumentParser(
        description="Beginner-friendly Crowd Estimation using YOLO and OpenCV."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(PROJECT_ROOT / "dataset"),
        help="Path to the dataset directory containing category subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "output"),
        help="Path to the output directory where annotations and CSV will be saved.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Model name or path to weights (defaults to lightweight 'yolov8n.pt').",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for person detection (default: 0.25).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device for inference ('cpu' by default).",
    )
    return parser.parse_args()


def run_pipeline() -> None:
    """Execute the crowd estimation pipeline over the local dataset."""
    args = parse_arguments()

    dataset_path = Path(args.dataset_dir)
    output_path = Path(args.output_dir)

    print("=" * 65)
    print("      CROWD ESTIMATION PIPELINE (YOLO + OpenCV)")
    print("=" * 65)
    print(f"Dataset Directory : {dataset_path}")
    print(f"Output Directory  : {output_path}")
    print(f"Model             : {args.model}")
    print(f"Confidence Level  : {args.conf}")
    print(f"Inference Device  : {args.device}")
    print("=" * 65)

    # Step 1: Ensure directory structure exists
    ensure_directories(PROJECT_ROOT)

    # Step 2: Initialize person detector (loads model weights onto CPU)
    try:
        # Check if weights exist in models/ folder
        local_model = PROJECT_ROOT / "models" / args.model
        model_target = str(local_model) if local_model.exists() else args.model
        detector = PersonDetector(
            model_name_or_path=model_target,
            conf_threshold=args.conf,
            device=args.device,
        )
    except Exception as e:
        print(f"[Error] Failed to initialize detector: {e}")
        sys.exit(1)

    # Step 3: Discover images from all density folders
    image_items = collect_dataset_images(dataset_path)

    if not image_items:
        print("\n[Notice] No supported images (JPG, JPEG, PNG) found in dataset folders:")
        print(f"  - {dataset_path / 'low_density'}")
        print(f"  - {dataset_path / 'medium_density'}")
        print(f"  - {dataset_path / 'high_density'}")
        print("\nPlease add test images into these folders and run the script again.\n")
        return

    print(f"\nFound {len(image_items)} image(s) to process.\n")

    results_records = []
    total_start = time.perf_counter()

    # Step 4: Process images one at a time to minimize memory usage
    for idx, (image_path, category) in enumerate(image_items, start=1):
        filename = image_path.name
        img_start = time.perf_counter()

        # Step 4.1: Load image safely
        image = load_image(image_path)
        if image is None:
            print(f"[{idx}/{len(image_items)}] [WARN] Failed to read image: {category}/{filename}. Skipping.")
            continue

        # Step 4.2: Run detection on CPU
        try:
            detections, person_count, _ = detector.detect(image)
        except Exception as e:
            print(f"[{idx}/{len(image_items)}] [ERROR] Inference failed on {category}/{filename}: {e}. Skipping.")
            continue

        # Step 4.3: Annotate image with boxes and crowd count
        annotated_image = draw_detections(image, detections, person_count)

        # Step 4.4: Save annotated image to matching output category folder
        output_file = output_path / category / filename
        saved = save_annotated_image(annotated_image, output_file)

        # Measure total processing time for this individual image
        img_elapsed = time.perf_counter() - img_start

        if saved:
            status_tag = "OK"
        else:
            status_tag = "SAVE_ERR"

        print(
            f"[{idx}/{len(image_items)}] [{category:<14}] {filename:<25} "
            f"-> Count: {person_count:3d} | Time: {img_elapsed * 1000.0:6.1f} ms [{status_tag}]"
        )

        # Record metrics for CSV export
        results_records.append({
            "Image Filename": filename,
            "Dataset Category": category,
            "Detected Person Count": person_count,
            "Processing Time (s)": round(img_elapsed, 4),
        })

        # Explicit cleanup to ensure minimal memory retention
        del image
        del annotated_image

    total_elapsed = time.perf_counter() - total_start

    # Step 5: Export results to CSV
    csv_file = output_path / "crowd_estimation_results.csv"
    save_results_to_csv(results_records, csv_file)

    # Step 6: Print terminal summary
    print("\n" + "=" * 65)
    print("                      PROCESSING SUMMARY")
    print("=" * 65)
    print(f"Total Images Processed : {len(results_records)} / {len(image_items)}")
    print(f"Total Execution Time   : {total_elapsed:.2f} s")
    if results_records:
        avg_time = sum(r["Processing Time (s)"] for r in results_records) / len(results_records)
        print(f"Average Time / Image   : {avg_time * 1000.0:.1f} ms")
        print(f"Total People Detected  : {sum(r['Detected Person Count'] for r in results_records)}")

        # Category breakdown
        print("\nBreakdown by Category:")
        categories_seen = sorted(set(r["Dataset Category"] for r in results_records))
        for cat in categories_seen:
            cat_records = [r for r in results_records if r["Dataset Category"] == cat]
            cat_count = sum(r["Detected Person Count"] for r in cat_records)
            print(f"  - {cat:<14}: {len(cat_records):2d} images, {cat_count:4d} people counted")

    print(f"\nAnnotated images saved to : {output_path}")
    print(f"Results CSV saved to      : {csv_file}")
    print("=" * 65)


if __name__ == "__main__":
    run_pipeline()
