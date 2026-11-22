# Crowd Estimation Prototype

A lightweight, beginner-friendly crowd estimation prototype built using **YOLOv8** and **OpenCV**. It scans local images across low-, medium-, and high-density folders, detects individuals using a pretrained COCO model (person class ID 0), counts the detected crowd, draws bounding boxes and count overlays, and exports structured results to CSV.

The project is designed strictly for **CPU inference**, making it ideal for development on laptops, desktop PCs, and deployment on edge devices like the **Raspberry Pi 4**.

---

## Folder Structure

```text
crowd-estimation/
│
├── dataset/
│   ├── low_density/         # Low crowd density images (1 - 5 people)
│   ├── medium_density/      # Medium crowd density images (6 - 20 people)
│   └── high_density/        # High crowd density images (20+ people)
│
├── models/                  # Local storage for pretrained YOLO weights
│
├── output/
│   ├── low_density/         # Annotated images corresponding to low_density
│   ├── medium_density/      # Annotated images corresponding to medium_density
│   ├── high_density/        # Annotated images corresponding to high_density
│   └── crowd_estimation_results.csv # Exported numerical results
│
├── src/
│   ├── __init__.py
│   ├── detector.py          # Model loading and inference engine
│   ├── image_processor.py   # Image loading, annotation, and saving
│   ├── utils.py             # File discovery, directories, and CSV export
│   └── main.py              # Pipeline orchestrator
│
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
└── .gitignore               # Git exclusions for models, datasets, and outputs
```

---

## Architectural Separation of Concerns

To keep the codebase maintainable and modular:

- **`src/detector.py` (`PersonDetector`)**:
  - Handles pretrained YOLO model loading (`yolov8n.pt` by default).
  - Explicitly targets CPU (`device='cpu'`).
  - Restricts detection to COCO class index `0` (`person`).
  - Keeps model weights and inference completely isolated from image drawing or file I/O.

- **`src/image_processor.py`**:
  - Reads images safely via OpenCV (`load_image`), returning `None` if an image is corrupted or invalid.
  - Draws bounding boxes, confidence tags, and an overlay count banner (`draw_detections`).
  - Writes annotated images to disk (`save_annotated_image`).

- **`src/utils.py`**:
  - Verifies and creates necessary directory paths (`ensure_directories`).
  - Scans category folders for supported extensions (`.jpg`, `.jpeg`, `.png`).
  - Exports numerical results to CSV using `pandas`.

- **`src/main.py`**:
  - Orchestrates the sequential pipeline.
  - Processes images **one at a time** to maintain a minimal memory footprint.
  - Manages error handling and outputs terminal progress and summary metrics.

---

## Installation

### Prerequisites
- Python 3.9 to 3.13
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Smr123-ddca/Peoplx.git
cd Peoplx
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS / Raspberry Pi
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## How to Add Images to the Dataset

Place your test images (`.jpg`, `.jpeg`, or `.png`) into the appropriate density category inside `dataset/`:

1. **`dataset/low_density/`**: Images containing few people (e.g., 1–5 individuals, sparse walking scenes).
2. **`dataset/medium_density/`**: Images containing moderate crowd sizes (e.g., 6–20 individuals, small groups).
3. **`dataset/high_density/`**: Images containing dense gatherings (e.g., public squares, events, busy streets).

> **Note**: Empty folders contain `.gitkeep` so git tracks the directory structure without committing datasets. Do not remove `.gitkeep`.

---

## How to Run the Project

Run the main pipeline using:

```bash
python src/main.py
```

### Optional Command-Line Arguments

You can customize the detection threshold, directories, or model weights:

```bash
# Custom confidence threshold (e.g. 0.35)
python src/main.py --conf 0.35

# Custom model weights
python src/main.py --model yolov8s.pt

# Explicit custom dataset or output folders
python src/main.py --dataset-dir dataset --output-dir output
```

---

## Expected Terminal Output

```text
=================================================================
      CROWD ESTIMATION PIPELINE (YOLO + OpenCV)
=================================================================
Dataset Directory : C:\Users\HP\OneDrive\Documents\GitHub\Peoplx\dataset
Output Directory  : C:\Users\HP\OneDrive\Documents\GitHub\Peoplx\output
Model             : yolov8n.pt
Confidence Level  : 0.25
Inference Device  : cpu
=================================================================
[Detector] Loading model 'yolov8n.pt' on device 'cpu'...

Found 3 image(s) to process.

[1/3] [low_density   ] park_walk.jpg             -> Count:   2 | Time:   65.3 ms [OK]
[2/3] [medium_density] bus_stop.png              -> Count:  11 | Time:   72.8 ms [OK]
[3/3] [high_density  ] festival.jpeg             -> Count:  34 | Time:   88.5 ms [OK]

=================================================================
                      PROCESSING SUMMARY
=================================================================
Total Images Processed : 3 / 3
Total Execution Time   : 0.35 s
Average Time / Image   : 75.5 ms
Total People Detected  : 47

Breakdown by Category:
  - high_density  :  1 images,   34 people counted
  - low_density   :  1 images,    2 people counted
  - medium_density:  1 images,   11 people counted

Annotated images saved to : C:\Users\HP\OneDrive\Documents\GitHub\Peoplx\output
Results CSV saved to      : C:\Users\HP\OneDrive\Documents\GitHub\Peoplx\output\crowd_estimation_results.csv
=================================================================
```

### Generated CSV Output (`output/crowd_estimation_results.csv`)

| Image Filename | Dataset Category | Detected Person Count | Processing Time (s) |
|---|---|---|---|
| park_walk.jpg | low_density | 2 | 0.0653 |
| bus_stop.png | medium_density | 11 | 0.0728 |
| festival.jpeg | high_density | 34 | 0.0885 |

---

## Raspberry Pi 4 Deployment: Limitations & Compatibility

When targeting a **Raspberry Pi 4 (Broadcom BCM2711, Quad-core Cortex-A72 @ 1.5GHz)**:

1. **CPU & Inference Latency**:
   - On a modern desktop CPU, `yolov8n` runs in ~30–80 ms per image.
   - On a Raspberry Pi 4 (CPU-only), inference typically takes **300 ms to 900 ms per image** (around 1 to 3 FPS).
   - This prototype processes images one-by-one, which prevents RAM spikes and fits comfortably within the Pi 4's 2GB/4GB/8GB RAM.

2. **Headless OpenCV Dependency**:
   - On Raspberry Pi OS Lite (headless, no desktop GUI), install `opencv-python-headless` instead of `opencv-python` to avoid missing X11/Qt shared library errors:
     ```bash
     pip install opencv-python-headless
     ```

3. **Thermal Throttling**:
   - Sustained CPU inference on Raspberry Pi 4 triggers thermal throttling (dropping CPU clock from 1.5GHz to 1.0GHz or 600MHz). A passive heatsink or active 5V cooling fan is recommended.

4. **Future Model Optimization Paths for Raspberry Pi**:
   - **NCNN or ONNX Runtime**: Exporting the PyTorch model to NCNN (`yolo export format=ncnn`) or ONNX with quantized INT8/FP16 yields 2x–3x faster inference on ARM Cortex-A72.
   - **TFLite**: Ultralytics supports export to TensorFlow Lite (`yolo export format=tflite`), which integrates cleanly with Raspberry Pi's Debian packages.
