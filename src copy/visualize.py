from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def overlay_mask(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.4,
) -> np.ndarray:
    overlay = image_bgr.copy()
    overlay[mask] = (
        overlay[mask].astype(np.float32) * (1.0 - alpha)
        + np.array(color, dtype=np.float32) * alpha
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
    return overlay


def save_overlay(path: Path, image_bgr: np.ndarray, mask: np.ndarray) -> None:
    overlay = overlay_mask(image_bgr, mask)
    cv2.imwrite(str(path), overlay)


def draw_boxes(
    image_bgr: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    color: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    preview = image_bgr.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
    return preview


def save_box_preview(
    path: Path,
    image_bgr: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
) -> None:
    preview = draw_boxes(image_bgr, boxes)
    cv2.imwrite(str(path), preview)


def save_side_by_side_comparison(
    path: Path,
    original_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
) -> None:
    height = max(original_bgr.shape[0], overlay_bgr.shape[0])
    width = original_bgr.shape[1] + overlay_bgr.shape[1]
    canvas = np.full((height + 40, width + 24, 3), 250, dtype=np.uint8)

    left_y = 32
    left_x = 8
    right_x = left_x + original_bgr.shape[1] + 8

    canvas[
        left_y : left_y + original_bgr.shape[0],
        left_x : left_x + original_bgr.shape[1],
    ] = original_bgr
    canvas[
        left_y : left_y + overlay_bgr.shape[0],
        right_x : right_x + overlay_bgr.shape[1],
    ] = overlay_bgr
    cv2.putText(
        canvas,
        "Original",
        (left_x, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Overlay",
        (right_x, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(path), canvas)
