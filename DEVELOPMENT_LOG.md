# Crowd Estimation Project — Development History Log

A chronological record of project milestones, architectural decisions, technical hurdles, and implementation details for the **Crowd Estimation Using YOLO and OpenCV** repository.

---

## Stage 1 — Initial Project Setup

- **Timeline**: 2026-11-22 / 2026-11-23 (Git commit `7c795de`)
- **Objective**: Establish the initial project repository, directory architecture, core dependencies, and foundational pipeline modules for beginner-friendly crowd estimation on CPU.
- **Files Created / Modified**:
  - `requirements.txt`
  - `.gitignore`
  - `src/__init__.py`, `src/detector.py`, `src/image_processor.py`, `src/utils.py`, `src/main.py`
  - `dataset/low_density/.gitkeep`, `dataset/medium_density/.gitkeep`, `dataset/high_density/.gitkeep`
  - `output/low_density/.gitkeep`, `output/medium_density/.gitkeep`, `output/high_density/.gitkeep`
  - `models/.gitkeep`
  - Initial `README.md`
- **Work Completed**:
  - Defined the modular repository layout separating detection (`src/detector.py`), OpenCV drawing (`src/image_processor.py`), directory/export utilities (`src/utils.py`), and orchestration (`src/main.py`).
  - Integrated Ultralytics YOLOv8n with explicit CPU device target (`device='cpu'`).
  - Configured Git exclusions to prevent committing model binaries (`.pt`), raw dataset images, and generated output files to source control.
- **Tests Performed**:
  - Initial basic unit tests in `tests/test_detector.py`, `tests/test_image_processor.py`, and `tests/test_pipeline.py`.
- **Results**:
  - Verified modular separation; pipeline could successfully load `yolov8n.pt` and process mock synthetic images.
- **Problems Encountered & Solutions**:
  - *Dependency availability*: Initial environment lacked `ultralytics` and `cv2`. Installed via pip with CPU-only PyTorch.
  - *Pytest class fixture deprecation*: Replaced deprecated `@pytest.fixture(scope="class")` methods with module-scoped standalone fixtures.
- **Remaining Tasks**:
  - Verify real person-only detection filtering and boundary clamping on actual images.

---

## Stage 2 — Detection Verification & Strengthening

- **Timeline**: 2026-11-23 (Pre-commit / Commit `7c795de`)
- **Objective**: Ensure the YOLO person detection module correctly isolates COCO class ID `0`, prevents model reloads, clamps coordinate bounds safely, and handles missing or corrupted image files gracefully.
- **Files Created / Modified**:
  - `src/detector.py`
  - `src/image_processor.py`
  - `tests/test_detector.py`
  - `tests/test_image_processor.py`
- **Work Completed**:
  - Implemented `PersonDetector` with explicit class verification: only boxes where `int(box.cls) == 0` are accepted.
  - Added configurable confidence threshold $\tau$ (default `0.25`).
  - Designed `DetectionResult` named tuple returning bounding box coordinates, confidences, class IDs, total count, and pure inference time.
  - Implemented boundary validation in OpenCV drawing: sorted coordinates ($x_{\min} < x_{\max}, y_{\min} < y_{\max}$) and clamped values within $[0, W-1]$ and $[0, H-1]$.
  - Verified that model weights are loaded once in memory and reused across all subsequent image inferences.
- **Tests Performed**:
  - Unit tests using synthetic black images and mocked YOLO prediction objects.
  - Verified exception handling for missing model weights (`FileNotFoundError`) and invalid image arrays (`TypeError`).
  - Total tests passing: 20 tests.
- **Results**:
  - Person-only filtering verified; non-person COCO classes (e.g., cars, dogs) successfully ignored.
- **Problems Encountered & Solutions**:
  - *MagicMock reserved keyword*: `MagicMock(cls=...)` raised `TypeError` in unit tests because `cls` is reserved in `MagicMock`. Resolved by instantiating `box = MagicMock()` and setting `box.cls = ...` separately.
  - *Mock CPU check failure*: PyTorch device comparison failed on mock objects. Resolved by adding `isinstance(model_core, torch.nn.Module)` validation before querying parameter devices.

---

## Stage 3 — Full Dataset Processing & Batch Pipeline

- **Timeline**: 2026-11-23 (Prompt 3 execution)
- **Objective**: Execute sequential batch inference across all images in the local dataset across three density categories and generate annotated images and a standardized CSV report.
- **Files Created / Modified**:
  - `src/main.py`
  - `src/utils.py`
  - `dataset/` (populated with 38 image files)
  - `output/` (generated annotated images)
  - `output/crowd_estimation_results.csv`
- **Work Completed**:
  - Discovered 38 images in the dataset: 14 in `low_density`, 11 in `medium_density`, 13 in `high_density`.
  - Implemented sequential single-image pipeline with explicit reference deletion (`del image, annotated_image`) to maintain memory usage below 600 MB RAM.
  - Annotated each image with anti-aliased green bounding boxes, individual confidence labels, and a top-left headcount banner.
  - Exported structured CSV results with columns: `image_filename`, `dataset_category`, `detected_person_count`, `inference_time_seconds`, `image_width`, `image_height`, `processing_status`.
- **Tests Performed**:
  - Verified file existence in `output/` subfolders.
  - Verified CSV schema and non-empty status.
- **Results**:
  - Total images processed: 38 / 38 (100% success rate, 0 failures).
  - Total person detections: 407 instances.
  - Category breakdown:
    - `low_density`: 14 images, 103 detections (avg 7.36).
    - `medium_density`: 11 images, 110 detections (avg 10.00).
    - `high_density`: 13 images, 194 detections (avg 14.92).
  - Reported historical execution time: 26.17 seconds.
- **Problems Encountered & Solutions**:
  - *Schema alignment*: Reconciled Prompt 1/2 legacy CSV format with Prompt 3 7-column schema in `src/utils.py`.

---

## Stage 4 — Accuracy Evaluation & Ground-Truth Verification

- **Timeline**: 2026-11-23 (Git commit `858b7ee`)
- **Objective**: Quantify counting accuracy by comparing YOLO predictions directly against human-annotated ground-truth counts using Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE).
- **Files Created / Modified**:
  - `dataset/ground_truth.csv`
  - `src/evaluator.py`
  - `src/evaluate.py`
  - `output/accuracy_results.csv`
  - `output/accuracy_summary.txt`
  - `output/performance_benchmark.json`
  - `tests/test_evaluator.py`
- **Work Completed**:
  - Generated ground-truth template for all 38 images; user manually counted and recorded the actual headcount for every image.
  - Implemented mathematical evaluation functions:
    $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |\hat{y}_i - y_i|, \quad \text{MAPE} = \frac{100\%}{M} \sum_{i=1, y_i > 0}^M \frac{|\hat{y}_i - y_i|}{y_i}$$
  - Implemented granular performance benchmarking separating model load time, first-image warm-up latency, steady-state latency, OpenCV drawing time, and image saving duration.
- **Tests Performed**:
  - Automated tests covering known MAE/MAPE inputs, zero actual count handling, missing ground truth, and report formatting (`tests/test_evaluator.py`). Total tests passing: 31 tests.
- **Results**:
  - Evaluated images: 38 / 38.
  - Baseline MAE: **6.66 people**.
  - Baseline MAPE: **31.88%**.
  - Average predicted count: 10.71 vs. Average actual count: 17.16.
  - High density demonstrated substantial undercounting (predicted 14.92 vs. actual 25.54, MAE 10.92) due to inter-person occlusions.
  - Steady-state CPU latency: ~151–318 ms; warm-up latency: ~4.2–5.3 s.

---

## Stage 5 — Model Parameter Optimization

- **Timeline**: 2026-11-23 (Prompt 5 execution)
- **Objective**: Execute a controlled grid search across confidence thresholds ($0.15, 0.25, 0.35$) and image resolutions ($320, 416, 640$) to identify optimal trade-offs between counting error and inference throughput.
- **Files Created / Modified**:
  - `src/detector.py` (added dynamic `imgsz` parameter)
  - `src/optimize.py`
  - `output/optimization_results.csv`
  - `tests/test_optimize.py`
- **Work Completed**:
  - Built `src/optimize.py` to iterate through the $3 \times 3 = 9$ parameter configurations on all 38 pre-cached images.
  - Recorded confidence threshold, image size, MAE, MAPE, average inference latency, average predicted count, and total detections.
  - Formulated quantitative improvement comparison relative to baseline ($6.66$ MAE).
- **Tests Performed**:
  - Verified grid search constants, CLI arguments, and best-configuration metric selection logic in `tests/test_optimize.py`. Total test suite expanded to 34 passing tests.
- **Results**:
  - **Best Overall Accuracy**: `Confidence = 0.15, Image Size = 640` achieved an MAE of **5.05 people** (MAPE **24.13%**), representing a **24.17% relative MAE improvement** over baseline.
  - **Fastest Speed**: `Confidence = 0.35, Image Size = 416` executed in **143.5 ms/image** (measured on Windows x86_64 CPU), but undercounted dense crowds (MAE 10.24).
  - **Proposed Raspberry Pi 4 Configuration (Theoretical Recommendation)**: `Confidence = 0.15, Image Size = 416` achieved an MAE of **7.58 people** while reducing computational FLOPs by ~58% relative to 640 (proposed based on computational modeling; pending physical hardware verification).

---

## Stage 6 — Raspberry Pi Deployment Preparation

- **Timeline**: 2026-11-23 (Prompt 5 & Final Documentation)
- **Objective**: Prepare the codebase for deployment on the ARM Cortex-A72 architecture of the Raspberry Pi 4 under 64-bit Raspberry Pi OS.
- **Files Created / Modified**:
  - `requirements-pi.txt`
  - `README.md`
  - `PROJECT_DOCUMENTATION.md`
  - `DEVELOPMENT_LOG.md`
  - `CHANGELOG.md`
- **Work Completed**:
  - Specified `opencv-python-headless` to eliminate missing X11/Qt library errors on headless OS installs.
  - Documented Debian Linux OS prerequisite packages (`libglib2.0-0`, `libgomp1`, `libatlas-base-dev`).
  - Evaluated Tencent NCNN and ONNX Runtime model export paths for ARM NEON SIMD vector acceleration.
  - Designed thermal guidelines to prevent CPU throttling at 80°C.
- **Deployment Status**:
  - **Prepared and Verified**: Dependency specifications, headless scripts, and CLI flags are verified.
  - **Pending Physical Run**: Benchmarking on a physical Raspberry Pi 4 board remains to be executed once physical hardware is connected.
- **Remaining Tasks**:
  - Flash Raspberry Pi OS 64-bit, execute live physical bench tests, and record actual hardware thermals and frame latencies.
