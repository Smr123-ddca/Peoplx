"""
test_detector.py
----------------
Unit tests and real inference tests for PersonDetector in src/detector.py.
Clearly distinguishes mocked/synthetic unit tests from real YOLO inference tests.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch

from src.detector import DetectionResult, PersonDetector

# Paths for real assets
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_WEIGHTS = PROJECT_ROOT / "models" / "yolov8n.pt"


# =====================================================================
# 1. UNIT TESTS (Mocked & Synthetic Data)
# =====================================================================

class TestPersonDetectorUnit:
    """Unit tests verifying detector logic, input validation, and filtering."""

    @patch("src.detector.YOLO")
    def test_detector_initialization_cpu(self, mock_yolo_cls):
        """Verify the model loads onto CPU with default confidence threshold 0.25."""
        mock_model_instance = MagicMock()
        mock_yolo_cls.return_value = mock_model_instance

        detector = PersonDetector(model_name_or_path="yolov8n.pt", conf_threshold=0.25, device="cpu")

        assert detector.conf_threshold == 0.25
        assert detector.device == "cpu"
        assert detector.is_cpu is True
        mock_yolo_cls.assert_called_once_with("yolov8n.pt")

    @patch("src.detector.YOLO")
    def test_configurable_confidence_threshold(self, mock_yolo_cls):
        """Verify confidence threshold is configurable."""
        detector = PersonDetector(conf_threshold=0.65, device="cpu")
        assert detector.conf_threshold == 0.65

    def test_missing_model_weights_raises_filenotfound(self):
        """Verify understandable FileNotFoundError on non-existent custom weights."""
        with pytest.raises(FileNotFoundError) as exc_info:
            PersonDetector(model_name_or_path="models/nonexistent_custom_model.pt")

        assert "Model weights file not found" in str(exc_info.value)

    @patch("src.detector.YOLO")
    def test_invalid_image_type_raises_typeerror(self, mock_yolo_cls):
        """Verify TypeError when input is not a numpy array."""
        detector = PersonDetector(conf_threshold=0.25, device="cpu")

        with pytest.raises(TypeError) as exc_info:
            detector.detect("not_an_array_image")  # type: ignore

        assert "Invalid image type" in str(exc_info.value)

    @patch("src.detector.YOLO")
    def test_none_or_empty_image_returns_zero_detections(self, mock_yolo_cls):
        """Verify None or zero-sized arrays return empty DetectionResult safely."""
        detector = PersonDetector(conf_threshold=0.25, device="cpu")

        res_none = detector.detect(None)  # type: ignore
        assert isinstance(res_none, DetectionResult)
        assert res_none.detections == []
        assert res_none.count == 0
        assert res_none.inference_time == 0.0

        empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
        res_empty = detector.detect(empty_img)
        assert res_empty.detections == []
        assert res_empty.count == 0

    @patch("src.detector.YOLO")
    def test_person_only_filtering_with_mock(self, mock_yolo_cls):
        """
        Unit test with mock YOLO results:
        Ensure that detections of other COCO classes (car, dog, bus) are excluded,
        and that only class ID 0 (person) meeting the confidence threshold is returned.
        """
        mock_yolo_inst = MagicMock()
        mock_yolo_cls.return_value = mock_yolo_inst

        # Construct synthetic boxes:
        # Box 1: Person (cls 0), conf 0.85 -> ACCEPT
        # Box 2: Car (cls 2), conf 0.95 -> REJECT (not person)
        # Box 3: Dog (cls 16), conf 0.70 -> REJECT (not person)
        # Box 4: Person (cls 0), conf 0.18 -> REJECT (conf < 0.25 threshold)
        box1 = MagicMock()
        box1.cls = torch.tensor([0.0])
        box1.conf = torch.tensor([0.85])
        box1.xyxy = torch.tensor([[10.0, 20.0, 50.0, 90.0]])

        box2 = MagicMock()
        box2.cls = torch.tensor([2.0])
        box2.conf = torch.tensor([0.95])
        box2.xyxy = torch.tensor([[60.0, 30.0, 120.0, 80.0]])

        box3 = MagicMock()
        box3.cls = torch.tensor([16.0])
        box3.conf = torch.tensor([0.70])
        box3.xyxy = torch.tensor([[20.0, 40.0, 40.0, 60.0]])

        box4 = MagicMock()
        box4.cls = torch.tensor([0.0])
        box4.conf = torch.tensor([0.18])
        box4.xyxy = torch.tensor([[15.0, 25.0, 35.0, 55.0]])

        mock_result = MagicMock()
        mock_result.boxes = [box1, box2, box3, box4]
        mock_yolo_inst.predict.return_value = [mock_result]

        detector = PersonDetector(conf_threshold=0.25, device="cpu")
        synthetic_img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.detect(synthetic_img)

        # Verify only Box 1 was accepted
        assert result.count == 1
        assert len(result.detections) == 1
        det = result.detections[0]
        assert det["class_id"] == 0
        assert det["class_name"] == "person"
        assert det["confidence"] == pytest.approx(0.85)
        assert det["bbox"] == [10, 20, 50, 90]

    @patch("src.detector.YOLO")
    def test_model_loaded_once_and_reused(self, mock_yolo_cls):
        """Verify model weights are loaded only once during init, not on every detect() call."""
        mock_yolo_inst = MagicMock()
        mock_yolo_inst.predict.return_value = []
        mock_yolo_cls.return_value = mock_yolo_inst

        detector = PersonDetector(conf_threshold=0.25, device="cpu")
        # YOLO constructor called once during initialization
        assert mock_yolo_cls.call_count == 1

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Call detect 5 times
        for _ in range(5):
            detector.detect(img)

        # YOLO constructor was NOT called again; predict was called 5 times on the same instance
        assert mock_yolo_cls.call_count == 1
        assert mock_yolo_inst.predict.call_count == 5


# =====================================================================
# 2. REAL YOLO INFERENCE TESTS (Actual Weights & Sample Images)
# =====================================================================

@pytest.fixture(scope="module")
def real_detector():
    """Load real PersonDetector using local or standard weights on CPU."""
    model_target = str(LOCAL_WEIGHTS) if LOCAL_WEIGHTS.exists() else "yolov8n.pt"
    detector = PersonDetector(model_name_or_path=model_target, conf_threshold=0.25, device="cpu")
    return detector


@pytest.fixture(scope="module")
def sample_image():
    """Find a real sample image containing people (from ultralytics assets)."""
    import ultralytics
    assets_dir = Path(ultralytics.__file__).resolve().parent / "assets"
    bus_image = assets_dir / "bus.jpg"
    if not bus_image.exists():
        pytest.skip("Ultralytics sample asset 'bus.jpg' not found.")
    import cv2
    img = cv2.imread(str(bus_image))
    assert img is not None, "Failed to load test asset bus.jpg"
    return img


class TestPersonDetectorRealInference:
    """Real end-to-end inference tests with actual YOLOv8 weights and images."""

    def test_real_model_is_on_cpu(self, real_detector):
        """Verify real model parameters reside on CPU."""
        assert real_detector.is_cpu is True
        assert real_detector.device == "cpu"

    def test_real_inference_person_detection(self, real_detector, sample_image):
        """
        Run real YOLOv8 inference on bus.jpg (which contains 4 people and a bus).
        Verifies:
        - Only people (class ID 0) are accepted (the bus is ignored).
        - Count equals len(detections).
        - Bounding boxes are valid within image bounds.
        - Confidences are in (0.25, 1.0].
        """
        img_h, img_w = sample_image.shape[:2]
        result = real_detector.detect(sample_image)

        assert isinstance(result, DetectionResult)
        assert result.count > 0, "Expected at least 1 person detected in bus.jpg"
        assert result.count == len(result.detections)
        assert result.inference_time > 0.0

        for det in result.detections:
            assert det["class_id"] == 0, f"Non-person class detected: {det['class_id']}"
            assert det["class_name"] == "person"
            assert 0.25 <= det["confidence"] <= 1.0

            x1, y1, x2, y2 = det["bbox"]
            assert 0 <= x1 < x2 <= img_w, f"Invalid bbox X coords: {x1}, {x2} for width {img_w}"
            assert 0 <= y1 < y2 <= img_h, f"Invalid bbox Y coords: {y1}, {y2} for height {img_h}"
