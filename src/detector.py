"""
detector.py
-----------
Pretrained YOLO model loader and inference engine for person detection.
Decoupled strictly from image processing and visualization.
"""

import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
import numpy as np
import torch
from ultralytics import YOLO


class DetectionResult(NamedTuple):
    """
    Structured result returned by PersonDetector.detect().
    Can be unpacked as (detections, count, inference_time).
    """
    detections: List[Dict]
    count: int
    inference_time: float


class PersonDetector:
    """
    Person detector using Ultralytics YOLO pretrained on COCO dataset.
    Configured for CPU execution with lightweight resource requirements.
    """

    # In the COCO dataset, class index 0 corresponds strictly to 'person'
    COCO_PERSON_CLASS_ID = 0

    # Common official pretrained weights that Ultralytics can auto-download
    OFFICIAL_MODELS = {
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
        "yolov9t.pt", "yolov9s.pt", "yolov9m.pt", "yolov9c.pt", "yolov9e.pt",
        "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
    }

    def __init__(
        self,
        model_name_or_path: Union[str, Path] = "yolov8n.pt",
        conf_threshold: float = 0.25,
        device: str = "cpu",
    ) -> None:
        """
        Initialize the YOLO model for person detection on CPU.

        Args:
            model_name_or_path: Path to weights or model name (e.g. 'yolov8n.pt' or 'models/yolov8n.pt').
            conf_threshold: Minimum confidence score to accept a detection (default: 0.25).
            device: Inference device ('cpu' by default, suitable for standard PCs and Raspberry Pi).
        """
        self.conf_threshold = float(conf_threshold)
        self.device = str(device).lower().strip()
        self.model_path = Path(model_name_or_path)

        # Validate that custom weight files exist before initializing
        if not self.model_path.exists():
            if self.model_path.name not in self.OFFICIAL_MODELS:
                raise FileNotFoundError(
                    f"Model weights file not found at '{self.model_path}'. "
                    f"Please verify the file path or provide a valid pretrained model name."
                )
            target = self.model_path.name
        else:
            target = str(self.model_path)

        print(f"[Detector] Loading model '{target}' on device '{self.device}'...")

        try:
            # Load YOLO model weights
            self.model = YOLO(target)
            # Explicitly enforce CPU execution on the underlying PyTorch module
            if hasattr(self.model, "model") and self.model.model is not None:
                self.model.model.to(self.device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load YOLO model from '{target}' on device '{self.device}': {e}"
            ) from e

    @property
    def is_cpu(self) -> bool:
        """Verify that the model is operating on CPU."""
        model_core = getattr(self.model, "model", None)
        if isinstance(model_core, torch.nn.Module):
            try:
                first_param = next(model_core.parameters())
                if hasattr(first_param, "device") and hasattr(first_param.device, "type"):
                    return first_param.device.type == "cpu"
            except StopIteration:
                pass
        return self.device == "cpu"

    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Detect people in a single image.

        Args:
            image: Image array in BGR format (as loaded by OpenCV).

        Returns:
            DetectionResult tuple with:
            - 'detections': List of dicts, each with:
                - 'bbox': [x1, y1, x2, y2] bounding box coordinates
                - 'confidence': float confidence score
                - 'class_id': COCO class ID (0 for person)
                - 'class_name': 'person'
            - 'count': int, total number of accepted detected people
            - 'inference_time': float, inference time in seconds
        """
        # Validate image input
        if image is None:
            return DetectionResult(detections=[], count=0, inference_time=0.0)

        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"Invalid image type '{type(image).__name__}'. Expected numpy.ndarray."
            )

        if image.size == 0 or len(image.shape) < 2:
            return DetectionResult(detections=[], count=0, inference_time=0.0)

        start_time = time.perf_counter()

        try:
            # Run inference on CPU, restricting NMS candidates strictly to the person class (ID 0)
            results = self.model.predict(
                source=image,
                classes=[self.COCO_PERSON_CLASS_ID],
                conf=self.conf_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            raise RuntimeError(f"YOLO inference error during person detection: {e}") from e

        inference_time = time.perf_counter() - start_time

        detections: List[Dict] = []

        # Parse detections from results
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                # Validate detection strictly matches COCO person class and confidence threshold
                if cls_id == self.COCO_PERSON_CLASS_ID and confidence >= self.conf_threshold:
                    coords = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = [int(round(c)) for c in coords]

                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "class_id": cls_id,
                        "class_name": "person",
                    })

        total_people = len(detections)
        return DetectionResult(
            detections=detections,
            count=total_people,
            inference_time=inference_time,
        )

