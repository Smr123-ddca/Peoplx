# Changelog

All notable changes to the **Crowd Estimation Using YOLO and OpenCV on Raspberry Pi** project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] - 2026-11-23

### Added
- **Parameter Optimization Grid Search (`src/optimize.py`)**:
  - Automated evaluation across 9 configurations: confidence thresholds (`0.15`, `0.25`, `0.35`) and image resolutions (`320`, `416`, `640`).
  - Automatic calculation of MAE, MAPE, latency, average predicted counts, and total detections.
  - Export of optimization data to `output/optimization_results.csv`.
- **Dynamic Resolution Support**: Added `imgsz` parameter to `PersonDetector.detect()` in `src/detector.py`.
- **Raspberry Pi 4 Headless Configuration**:
  - Created `requirements-pi.txt` using `opencv-python-headless` to eliminate desktop GUI dependencies.
  - Prepared step-by-step installation instructions for 64-bit Raspberry Pi OS (Debian Bookworm/Bullseye).
- **Unit Tests for Optimization (`tests/test_optimize.py`)**:
  - Added tests for grid definitions, CLI argument parser defaults, and best-configuration metric selection.
  - Expanded automated test suite from 31 to 34 passing tests.
- **Technical and Academic Documentation**:
  - Created `PROJECT_DOCUMENTATION.md` covering all 23 technical sections with mathematical formulas and Mermaid diagrams.
  - Created `DEVELOPMENT_LOG.md` documenting Stages 1 through 6.
  - Rewrote `README.md` into a 15-section guide.

### Changed
- Improved best-accuracy configuration to `conf = 0.15, imgsz = 640`, achieving MAE of **5.05 people** (a **24.17% improvement** over the 6.66 baseline).
- Formulated proposed Raspberry Pi 4 configuration: `conf = 0.15, imgsz = 416` (MAE **7.58 people**, ~58% fewer FLOPs; proposed based on computational modeling, pending physical hardware verification).

---

## [0.2.0] - 2026-11-23

### Added
- **Accuracy Evaluation Pipeline (`src/evaluator.py`, `src/evaluate.py`)**:
  - Ground-truth comparison engine calculating Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE).
  - Exclusion of zero-actual counts in MAPE to prevent division-by-zero errors.
  - Breakdown of accuracy across `low_density`, `medium_density`, and `high_density` categories.
  - Export to `output/accuracy_results.csv` and `output/accuracy_summary.txt`.
- **Ground-Truth Dataset (`dataset/ground_truth.csv`)**:
  - Template populated with all 38 dataset image filenames and verified human ground-truth counts.
- **Performance Benchmarking**:
  - Fine-grained timing instrumentation in `src/main.py`.
  - Export to `output/performance_benchmark.json` separating model load time, first-image warm-up latency, steady-state latency, OpenCV annotation time, and image saving duration.
- **Evaluator Test Suite (`tests/test_evaluator.py`)**:
  - Added 11 automated unit tests verifying MAE, MAPE, and summary generation.

### Changed
- Standardized `output/crowd_estimation_results.csv` to Prompt 3 7-column schema.

---

## [0.1.0] - 2026-11-23

### Added
- **Initial Modular Architecture**:
  - `src/detector.py`: `PersonDetector` class wrapping Ultralytics YOLOv8n with CPU enforcement and COCO class ID `0` filtering.
  - `src/image_processor.py`: OpenCV loading, bounding-box coordinate clamping, anti-aliased drawing, and count banner rendering.
  - `src/utils.py`: Category discovery, supported format filtering (`.jpg`, `.jpeg`, `.png`), and CSV export.
  - `src/main.py`: Sequential single-image orchestrator maintaining memory footprint below 600 MB.
- **Dataset Hierarchy**:
  - Structured folders: `dataset/low_density/`, `dataset/medium_density/`, `dataset/high_density/`.
- **Automated Test Suite**:
  - 20 unit and integration tests across `tests/test_detector.py`, `tests/test_image_processor.py`, and `tests/test_pipeline.py`.
- **Baseline Ingestion**:
  - Batch processed all 38 dataset images on CPU, detecting 407 person instances across 38 images.
