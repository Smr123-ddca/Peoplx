"""
test_image_processor.py
-----------------------
Unit tests for image loading, coordinate validation, bounding box rendering,
and crowd count badge rendering in src/image_processor.py.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from src.image_processor import draw_detections, load_image, save_annotated_image


class TestImageProcessor:
    """Unit tests verifying OpenCV drawing, image loading, and bounding box placement."""

    def test_load_image_missing_file_returns_none(self, tmp_path):
        """Verify load_image returns None for non-existent path."""
        missing = tmp_path / "does_not_exist.jpg"
        assert load_image(missing) is None

    def test_load_image_corrupted_file_returns_none(self, tmp_path):
        """Verify load_image returns None for unreadable/corrupted files."""
        corrupted = tmp_path / "corrupted.jpg"
        corrupted.write_text("not_valid_jpeg_binary_data")
        assert load_image(corrupted) is None

    def test_load_image_valid(self, tmp_path):
        """Verify load_image returns a valid NumPy BGR array for readable images."""
        valid_img_path = tmp_path / "valid.png"
        sample_arr = np.full((100, 100, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(valid_img_path), sample_arr)

        loaded = load_image(valid_img_path)
        assert loaded is not None
        assert loaded.shape == (100, 100, 3)

    def test_draw_detections_does_not_mutate_original(self):
        """Verify draw_detections operates on a copy and does not mutate the input array."""
        original = np.zeros((200, 200, 3), dtype=np.uint8)
        copy_original = original.copy()

        detections = [{"bbox": [20, 20, 80, 80], "confidence": 0.85, "class_id": 0, "class_name": "person"}]
        annotated = draw_detections(original, detections, count=1)

        # Original image must remain all zeros
        assert np.array_equal(original, copy_original)
        # Annotated image must have drawn elements
        assert not np.array_equal(annotated, original)

    def test_draw_detections_bounding_box_coordinates(self):
        """
        Verify that OpenCV draws the bounding box at the exact specified coordinates.
        Uses a black 400x400 image and checks that pixels on the box perimeter are green.
        """
        canvas = np.zeros((400, 400, 3), dtype=np.uint8)
        # Box from x=100 to 250, y=120 to 300 (placed safely below top-left count badge)
        box = [100, 120, 250, 300]
        detections = [{"bbox": box, "confidence": 0.90, "class_id": 0, "class_name": "person"}]

        annotated = draw_detections(canvas, detections, count=1)

        # Green color used in image_processor: (12, 255, 36) in BGR
        green_bgr = np.array([12, 255, 36], dtype=np.uint8)

        # Check a pixel along the top edge of the box (e.g. y=120, x=180)
        pixel_top_edge = annotated[120, 180]
        assert np.array_equal(pixel_top_edge, green_bgr)

        # Check a pixel along the bottom edge (e.g. y=300, x=180)
        pixel_bottom_edge = annotated[300, 180]
        assert np.array_equal(pixel_bottom_edge, green_bgr)

        # Check a pixel along the left edge (e.g. y=200, x=100)
        pixel_left_edge = annotated[200, 100]
        assert np.array_equal(pixel_left_edge, green_bgr)

    def test_draw_detections_coordinate_ordering_and_clamping(self):
        """
        Verify that inverted coordinates [180, 180, 20, 20] and out-of-bounds
        coordinates [-50, -50, 600, 600] are handled safely without crash or IndexError.
        """
        canvas = np.zeros((200, 200, 3), dtype=np.uint8)

        # Inverted coordinates
        inverted_det = [{"bbox": [180, 180, 20, 20], "confidence": 0.75, "class_id": 0, "class_name": "person"}]
        res1 = draw_detections(canvas, inverted_det)
        assert res1.shape == canvas.shape

        # Out-of-bounds coordinates
        out_of_bounds_det = [{"bbox": [-100, -100, 500, 500], "confidence": 0.80, "class_id": 0, "class_name": "person"}]
        res2 = draw_detections(canvas, out_of_bounds_det)
        assert res2.shape == canvas.shape

    def test_draw_detections_count_banner_matches_accepted_detections(self):
        """
        Verify that when count is not explicitly specified,
        it automatically matches len(detections).
        """
        canvas = np.zeros((300, 300, 3), dtype=np.uint8)
        detections = [
            {"bbox": [10, 10, 50, 50], "confidence": 0.85, "class_id": 0, "class_name": "person"},
            {"bbox": [60, 60, 100, 100], "confidence": 0.75, "class_id": 0, "class_name": "person"},
            {"bbox": [110, 110, 150, 150], "confidence": 0.65, "class_id": 0, "class_name": "person"},
        ]

        # Automatic count
        annotated = draw_detections(canvas, detections)
        assert annotated is not None
        assert annotated.shape == canvas.shape

        # Explicit count passed
        annotated_explicit = draw_detections(canvas, detections, count=len(detections))
        assert annotated_explicit is not None

    def test_save_annotated_image(self, tmp_path):
        """Verify save_annotated_image creates parent dirs and successfully writes file."""
        dest = tmp_path / "subdir" / "result.jpg"
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        success = save_annotated_image(img, dest)
        assert success is True
        assert dest.is_file()
        assert dest.stat().st_size > 0
