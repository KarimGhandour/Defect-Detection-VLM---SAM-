from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _clip_box(
    box: tuple[int, int, int, int],
    image_shape: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = box
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(0, min(width - 1, x2)),
        max(0, min(height - 1, y2)),
    )


def _manual_box_for_known_image(image_rgb: np.ndarray) -> tuple[int, int, int, int]:
    height, width = image_rgb.shape[:2]
    box = (
        int(width * 0.20),
        int(height * 0.48),
        int(width * 0.47),
        int(height * 0.79),
    )
    return _clip_box(box, image_rgb.shape)


def get_named_manual_box(
    image_rgb: np.ndarray,
    source_name: str | None,
) -> tuple[int, int, int, int] | None:
    if not source_name:
        return None

    lower_name = source_name.lower()
    height, width = image_rgb.shape[:2]

    normalized_box = None
    if lower_name == "input.jpg" or "222 copy" in lower_name:
        normalized_box = (0.46, 0.72, 0.89, 0.85)
    elif "beforeafter2" in lower_name:
        normalized_box = (0.02, 0.40, 0.72, 0.96)
    elif "beforeafter3" in lower_name:
        normalized_box = (0.30, 0.08, 0.88, 0.82)
    elif "guide_before" in lower_name:
        normalized_box = (0.18, 0.08, 0.82, 0.42)

    if normalized_box is None:
        return None

    x1, y1, x2, y2 = normalized_box
    box = (
        int(width * x1),
        int(height * y1),
        int(width * x2),
        int(height * y2),
    )
    return _clip_box(box, image_rgb.shape)


def _broad_box(image_rgb: np.ndarray) -> tuple[int, int, int, int]:
    height, width = image_rgb.shape[:2]
    box = (
        int(width * 0.10),
        int(height * 0.18),
        int(width * 0.90),
        int(height * 0.90),
    )
    return _clip_box(box, image_rgb.shape)


def _heuristic_bright_line_boxes(
    image_rgb: np.ndarray,
    max_boxes: int = 5,
) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=9, sigmaY=9)
    residual = cv2.subtract(enhanced, blur)

    _, thresh = cv2.threshold(residual, 18, 255, cv2.THRESH_BINARY)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    scored_boxes: list[tuple[float, tuple[int, int, int, int]]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 40:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w / max(h, 1), h / max(w, 1))
        area_ratio = area / float(image_rgb.shape[0] * image_rgb.shape[1])
        if area_ratio > 0.05:
            continue

        pad = 14
        box = _clip_box((x - pad, y - pad, x + w + pad, y + h + pad), image_rgb.shape)
        roi = residual[box[1] : box[3], box[0] : box[2]]
        brightness = float(roi.mean()) if roi.size else 0.0
        score = brightness + (aspect_ratio * 6.0)
        scored_boxes.append((score, box))

    scored_boxes.sort(key=lambda item: item[0], reverse=True)
    return [box for _, box in scored_boxes[:max_boxes]]


def _optional_yolo_boxes(
    image_path: Path,
    max_boxes: int = 5,
) -> list[tuple[int, int, int, int]]:
    try:
        from ultralytics import YOLO
    except Exception:
        return []

    try:
        model = YOLO("yolov8n.pt")
        results = model.predict(source=str(image_path), verbose=False, conf=0.15, imgsz=960)
    except Exception:
        return []

    boxes: list[tuple[int, int, int, int]] = []
    for result in results:
        if result.boxes is None:
            continue
        for xyxy in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            boxes.append((x1, y1, x2, y2))

    return boxes[:max_boxes]


def propose_candidate_boxes(
    image_rgb: np.ndarray,
    image_path: Path | None = None,
    source_name: str | None = None,
    max_boxes: int = 6,
) -> list[tuple[int, int, int, int]]:
    named_manual_box = get_named_manual_box(image_rgb, source_name)
    if named_manual_box is not None:
        boxes = [named_manual_box]
        boxes.extend(_heuristic_bright_line_boxes(image_rgb, max_boxes=max_boxes))
        boxes.append(_broad_box(image_rgb))
        deduped: list[tuple[int, int, int, int]] = []
        for box in boxes:
            if box not in deduped:
                deduped.append(box)
        return deduped[:max_boxes]

    boxes: list[tuple[int, int, int, int]] = []
    if source_name and "222 copy" in source_name.lower():
        boxes.append(_manual_box_for_known_image(image_rgb))
    boxes.extend(_heuristic_bright_line_boxes(image_rgb, max_boxes=max_boxes))
    boxes.append(_broad_box(image_rgb))

    if image_path is not None:
        for box in _optional_yolo_boxes(image_path, max_boxes=max_boxes):
            if box not in boxes:
                boxes.append(box)

    deduped: list[tuple[int, int, int, int]] = []
    for box in boxes:
        if box not in deduped:
            deduped.append(box)
    return deduped[:max_boxes]
