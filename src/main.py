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
    t_load_start = time.perf_counter()
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
    model_load_time = time.perf_counter() - t_load_start
    print(f"[Detector] Model loaded in {model_load_time:.4f} s.")

    # Step 3: Discover images from all density folders
    image_items = collect_dataset_images(dataset_path)

    if not image_items:
        print("\n[Notice] No supported images (JPG, JPEG, PNG) found in dataset folders:")
        print(f"  - {dataset_path / 'low_density'}")
        print(f"  - {dataset_path / 'medium_density'}")
        print(f"  - {dataset_path / 'high_density'}")
        print("\nPlease add test images into these folders and run the script again.\n")
        return

    total_discovered = len(image_items)
    print(f"\nFound {total_discovered} image(s) to process across density folders.\n")

    results_records = []
    annotation_times = []
    saving_times = []
    total_start = time.perf_counter()

    # Step 4: Process images one at a time to minimize memory usage
    for idx, (image_path, category) in enumerate(image_items, start=1):
        filename = image_path.name
        img_w, img_h = 0, 0

        # Step 4.1: Load image safely
        image = load_image(image_path)
        if image is None:
            print(f"[{idx}/{total_discovered}] [{category:<14}] {filename:<32} [WARN] Unreadable/corrupt image. Skipping.")
            results_records.append({
                "image_filename": filename,
                "dataset_category": category,
                "detected_person_count": 0,
                "inference_time_seconds": 0.0,
                "image_width": 0,
                "image_height": 0,
                "processing_status": "FAILED_LOAD",
            })
            continue

        img_h, img_w = image.shape[:2]

        # Step 4.2: Run detection on CPU
        try:
            detections, person_count, inference_time = detector.detect(image)
        except Exception as e:
            print(f"[{idx}/{total_discovered}] [{category:<14}] {filename:<32} [ERROR] Inference failed: {e}. Skipping.")
            results_records.append({
                "image_filename": filename,
                "dataset_category": category,
                "detected_person_count": 0,
                "inference_time_seconds": 0.0,
                "image_width": img_w,
                "image_height": img_h,
                "processing_status": "FAILED_INFERENCE",
            })
            del image
            continue

        # Step 4.3: Annotate image with boxes and crowd count
        t_annot_start = time.perf_counter()
        annotated_image = draw_detections(image, detections, person_count)
        annot_time = time.perf_counter() - t_annot_start
        annotation_times.append(annot_time)

        # Step 4.4: Save annotated image to matching output category folder
        output_file = output_path / category / filename
        t_save_start = time.perf_counter()
        saved = save_annotated_image(annotated_image, output_file)
        save_time = time.perf_counter() - t_save_start
        saving_times.append(save_time)

        status_tag = "SUCCESS" if saved else "SAVE_ERROR"

        print(
            f"[{idx:2d}/{total_discovered}] [{category:<14}] {filename:<32} "
            f"-> Count: {person_count:3d} | Infer: {inference_time * 1000.0:6.1f} ms | "
            f"Size: {img_w}x{img_h} [{status_tag}]"
        )

        # Record metrics for CSV export matching required specifications
        results_records.append({
            "image_filename": filename,
            "dataset_category": category,
            "detected_person_count": person_count,
            "inference_time_seconds": round(inference_time, 4),
            "image_width": img_w,
            "image_height": img_h,
            "processing_status": status_tag,
        })

        # Explicit cleanup to ensure minimal memory retention
        del image
        del annotated_image

    total_elapsed = time.perf_counter() - total_start

    # Step 5: Export results to CSV
    csv_file = output_path / "crowd_estimation_results.csv"
    save_results_to_csv(results_records, csv_file)

    # Step 6: Compute metrics and print category summary
    successful_records = [r for r in results_records if r["processing_status"] == "SUCCESS"]
    failed_records = [r for r in results_records if r["processing_status"] != "SUCCESS"]

    successful_count = len(successful_records)
    failed_count = len(failed_records)
    total_detections = sum(r["detected_person_count"] for r in successful_records)

    avg_people_per_image = (
        total_detections / successful_count if successful_count > 0 else 0.0
    )
    all_inference_times = [r["inference_time_seconds"] for r in successful_records]
    avg_inference_time = (
        sum(all_inference_times) / successful_count if successful_count > 0 else 0.0
    )

    # Performance benchmarking: separate 1st image warm-up from steady-state
    if all_inference_times:
        warmup_inference_ms = all_inference_times[0] * 1000.0
        steady_state_times = all_inference_times[1:] if len(all_inference_times) > 1 else [all_inference_times[0]]
        steady_state_inference_ms = (sum(steady_state_times) / len(steady_state_times)) * 1000.0
    else:
        warmup_inference_ms = 0.0
        steady_state_inference_ms = 0.0

    avg_annotation_ms = (sum(annotation_times) / len(annotation_times) * 1000.0) if annotation_times else 0.0
    avg_save_ms = (sum(saving_times) / len(saving_times) * 1000.0) if saving_times else 0.0

    # Save benchmark metrics to JSON for evaluation tool
    benchmark_data = {
        "model_loading_seconds": round(model_load_time, 4),
        "warmup_inference_ms": round(warmup_inference_ms, 1),
        "steady_state_inference_ms": round(steady_state_inference_ms, 1),
        "avg_annotation_ms": round(avg_annotation_ms, 1),
        "avg_save_ms": round(avg_save_ms, 1),
        "total_pipeline_seconds": round(total_elapsed, 2),
    }
    import json
    benchmark_file = output_path / "performance_benchmark.json"
    benchmark_file.write_text(json.dumps(benchmark_data, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("                     DATASET PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total Images Discovered          : {total_discovered}")
    print(f"Successfully Processed Images    : {successful_count}")
    print(f"Failed Images                    : {failed_count}")
    print(f"Total Detected People (Sum)      : {total_detections}")
    print(f"Average Detected People / Image  : {avg_people_per_image:.2f}")
    print(f"Average Inference Time (All)     : {avg_inference_time * 1000.0:.1f} ms ({avg_inference_time:.4f} s)")
    print(f"Total Pipeline Elapsed Time      : {total_elapsed:.2f} s")
    print("-" * 70)
    print("PERFORMANCE BENCHMARKING (CPU):")
    print(f"  - Model Loading Time           : {model_load_time:.4f} s")
    print(f"  - First-Image Warm-Up Latency  : {warmup_inference_ms:.1f} ms")
    print(f"  - Steady-State Average Latency : {steady_state_inference_ms:.1f} ms")
    print(f"  - OpenCV Annotation Time (Avg) : {avg_annotation_ms:.1f} ms")
    print(f"  - Image Saving Time (Avg)      : {avg_save_ms:.1f} ms")
    print("-" * 70)
    print("Note: 'Total Detected People' is the cumulative sum of detections across")
    print("all images, not the count of unique individuals. The same person may")
    print("appear in multiple images.")
    print("-" * 70)

    print("Breakdown by Density Category:")
    categories_order = ["low_density", "medium_density", "high_density"]
    for cat in categories_order:
        cat_records = [r for r in results_records if r["dataset_category"] == cat]
        cat_success = [r for r in cat_records if r["processing_status"] == "SUCCESS"]
        cat_people = sum(r["detected_person_count"] for r in cat_success)
        cat_avg = (cat_people / len(cat_success)) if cat_success else 0.0
        print(
            f"  - {cat:<15}: {len(cat_records):2d} processed "
            f"({len(cat_success)} success), {cat_people:4d} people counted "
            f"(avg {cat_avg:.1f}/img)"
        )

    print("=" * 70)
    print(f"Annotated images saved to : {output_path}")
    print(f"Results CSV saved to      : {csv_file}")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
