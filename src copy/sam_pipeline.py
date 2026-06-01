from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from segment_anything import (
    SamAutomaticMaskGenerator,
    SamPredictor,
    sam_model_registry,
)


@dataclass
class MaskCandidate:
    mask: np.ndarray
    score: float
    bbox_xyxy: tuple[int, int, int, int]
    area_ratio: float
    brightness_delta: float
    aspect_ratio: float


def load_sam_model(checkpoint_path: str, device: str):
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam.to(device=device)
    sam.eval()
    return sam


def generate_auto_masks(image_rgb: np.ndarray, sam) -> list[dict]:
    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        min_mask_region_area=200,
    )
    generator.point_grids = [
        point_grid.astype(np.float32) for point_grid in generator.point_grids
    ]
    return generator.generate(image_rgb)


def _compute_feature_maps(image_rgb: np.ndarray) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=9, sigmaY=9)
    positive_contrast = np.clip(gray - blur, 0.0, None)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    return {
        "gray": gray,
        "positive_contrast": positive_contrast,
        "gradient": gradient,
    }


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2, y2


def _ring_from_mask(mask: np.ndarray, radius: int = 10) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1),
    )
    expanded = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return np.logical_and(expanded, ~mask)


def _mask_candidate_from_binary(
    mask: np.ndarray,
    feature_maps: dict[str, np.ndarray],
) -> MaskCandidate | None:
    if mask is None or mask.sum() < 100:
        return None

    image_area = float(mask.shape[0] * mask.shape[1])
    area_ratio = float(mask.sum() / image_area)
    if area_ratio > 0.18:
        return None

    x1, y1, x2, y2 = _bbox_from_mask(mask)
    image_height, image_width = mask.shape[:2]
    width = max(1, x2 - x1 + 1)
    height = max(1, y2 - y1 + 1)
    aspect_ratio = max(width / height, height / width)
    bbox_area = float(width * height)
    fill_ratio = float(mask.sum() / bbox_area)
    min_border_distance = min(x1, y1, image_width - 1 - x2, image_height - 1 - y2)

    ring = _ring_from_mask(mask)
    if ring.sum() == 0:
        return None

    gray = feature_maps["gray"]
    contrast = feature_maps["positive_contrast"]
    gradient = feature_maps["gradient"]

    inside_gray = float(gray[mask].mean())
    ring_gray = float(gray[ring].mean())
    brightness_delta = inside_gray - ring_gray
    local_contrast = float(contrast[mask].mean())
    local_gradient = float(gradient[mask].mean())

    size_center = 0.003
    size_score = np.exp(-((np.log(area_ratio + 1e-6) - np.log(size_center)) ** 2) / 1.6)
    aspect_score = np.tanh((aspect_ratio - 1.2) / 2.2)
    thinness_score = 1.0 - min(fill_ratio, 1.0)
    brightness_score = max(brightness_delta, 0.0) * 6.0
    contrast_score = local_contrast * 8.0
    gradient_score = local_gradient * 2.5
    border_penalty = 0.0
    if min_border_distance <= 4:
        border_penalty = 3.0
    elif min_border_distance <= 16:
        border_penalty = 1.2

    score = (
        1.2 * float(size_score)
        + 1.4 * float(aspect_score)
        + 1.1 * thinness_score
        + brightness_score
        + contrast_score
        + gradient_score
        - border_penalty
    )

    return MaskCandidate(
        mask=mask,
        score=float(score),
        bbox_xyxy=(x1, y1, x2, y2),
        area_ratio=area_ratio,
        brightness_delta=brightness_delta,
        aspect_ratio=aspect_ratio,
    )


def select_scratch_mask(
    masks: list[dict],
    image_rgb: np.ndarray,
    top_k: int = 3,
    minimum_score: float = 1.6,
) -> tuple[MaskCandidate | None, list[MaskCandidate]]:
    if not masks:
        return None, []

    feature_maps = _compute_feature_maps(image_rgb)
    candidates: list[MaskCandidate] = []
    for mask_dict in masks:
        mask = np.asarray(mask_dict["segmentation"], dtype=bool)
        candidate = _mask_candidate_from_binary(mask, feature_maps)
        if candidate is not None:
            candidates.append(candidate)

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    if not candidates:
        return None, []

    best = candidates[0]
    if best.score < minimum_score:
        return None, candidates[:top_k]
    return best, candidates[:top_k]


def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_small)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel_large)
    return refined.astype(bool)


def refine_mask_with_box(
    predictor: SamPredictor,
    image_rgb: np.ndarray,
    box_xyxy: tuple[int, int, int, int],
) -> tuple[np.ndarray, float]:
    predictor.set_image(image_rgb)
    input_box = np.array(box_xyxy, dtype=np.float32)
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box,
        multimask_output=True,
    )
    best_idx = int(np.argmax(scores))
    return masks[best_idx].astype(bool), float(scores[best_idx])
