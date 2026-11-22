"""
image_processor.py
------------------
Handles image I/O, bounding box rendering, and visual count annotation.
Completely separated from model loading and inference logic.
"""

from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np


def load_image(image_path: Path) -> Optional[np.ndarray]:
    """
    Safely load an image from disk using OpenCV.

    Args:
        image_path: Path to the image file.

    Returns:
        NumPy array (BGR image) if readable, otherwise None.
    """
    try:
        path_obj = Path(image_path)
        if not path_obj.is_file():
            return None
        # cv2.imread returns None if the image file is corrupted or cannot be decoded
        img = cv2.imread(str(path_obj))
        if img is None or img.size == 0:
            return None
        return img
    except Exception:
        return None


def draw_detections(
    image: np.ndarray,
    detections: List[Dict],
    count: Optional[int] = None,
) -> np.ndarray:
    """
    Draw bounding boxes, confidence labels, and total count on the image.

    Args:
        image: Original BGR image as a NumPy array.
        detections: List of detection dictionaries (each with 'bbox' and 'confidence').
        count: Total accepted person detections in this image. If None, len(detections) is used.

    Returns:
        Annotated BGR image as a new NumPy array.
    """
    if count is None:
        count = len(detections)

    # Create an independent copy to avoid mutating the original image in memory
    annotated = image.copy()
    img_h, img_w = annotated.shape[:2]

    # Adaptive font scale and thickness based on image dimensions
    scale = max(0.4, min(img_w, img_h) / 1000.0)
    line_thickness = max(1, int(scale * 2.5))
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw each person detection
    for det in detections:
        raw_x1, raw_y1, raw_x2, raw_y2 = det["bbox"]
        conf = det.get("confidence", 0.0)

        # Ensure correct coordinate ordering (x1 <= x2 and y1 <= y2)
        x1, x2 = sorted((int(raw_x1), int(raw_x2)))
        y1, y2 = sorted((int(raw_y1), int(raw_y2)))

        # Clamp coordinates strictly within the image boundaries
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))

        # Green bounding box for detected person (BGR format: (12, 255, 36))
        box_color = (12, 255, 36)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, line_thickness)

        # Label displaying class and confidence percentage
        label = f"Person {conf * 100:.1f}%"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, font, fontScale=scale * 0.7, thickness=max(1, int(line_thickness / 2))
        )

        # Draw filled background rectangle behind text for high legibility
        label_y1 = max(0, y1 - label_h - baseline - 4)
        label_y2 = y1
        label_x2 = min(img_w, x1 + label_w + 6)
        cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), box_color, cv2.FILLED)

        # Draw label text in black on top of the green tag
        text_pos = (x1 + 3, label_y2 - baseline - 2)
        cv2.putText(
            annotated,
            label,
            text_pos,
            font,
            fontScale=scale * 0.7,
            color=(0, 0, 0),
            thickness=max(1, int(line_thickness / 2)),
            lineType=cv2.LINE_AA,
        )

    # Add total person count overlay badge at top-left
    count_text = f"Crowd Count: {count}"
    badge_scale = max(0.6, scale * 1.1)
    badge_thickness = max(2, int(line_thickness * 1.2))

    (badge_w, badge_h), b_baseline = cv2.getTextSize(
        count_text, font, fontScale=badge_scale, thickness=badge_thickness
    )

    padding = 12
    badge_x1 = 15
    badge_y1 = 15
    badge_x2 = badge_x1 + badge_w + (padding * 2)
    badge_y2 = badge_y1 + badge_h + (padding * 2)

    # Dark background banner (BGR: (25, 25, 25))
    cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), (25, 25, 25), cv2.FILLED)
    # Bright border (BGR: (0, 200, 255))
    cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), (0, 200, 255), max(1, line_thickness))

    # White count text with anti-aliasing
    text_origin = (badge_x1 + padding, badge_y1 + badge_h + padding - 2)
    cv2.putText(
        annotated,
        count_text,
        text_origin,
        font,
        fontScale=badge_scale,
        color=(255, 255, 255),
        thickness=badge_thickness,
        lineType=cv2.LINE_AA,
    )

    return annotated


def save_annotated_image(image: np.ndarray, output_path: Path) -> bool:
    """
    Save annotated image to target destination.

    Args:
        image: NumPy array in BGR format.
        output_path: Path where the image should be saved.

    Returns:
        True if successfully written, False otherwise.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(output_path), image))
    except Exception:
        return False
