from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from config import get_settings
from io_utils import (
    crop_before_panel,
    discover_reference_images,
    ensure_directories,
    load_image,
    save_json,
    sanitize_stem,
    save_mask,
)
from qwen_perception import QwenPerceptionLayer
from sam2_pipeline import SAM2BoxSegmenter
from sam_pipeline import MaskCandidate, postprocess_mask, select_scratch_mask
from visualize import (
    overlay_mask,
    save_box_preview,
    save_overlay,
    save_side_by_side_comparison,
)
from yolo_fallback import get_named_manual_box, propose_candidate_boxes


def _save_candidate_overlays(image_bgr, candidates, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, candidate in enumerate(candidates[:3], start=1):
        save_overlay(
            output_dir / f"overlay_candidate_{index}.png",
            image_bgr,
            postprocess_mask(candidate.mask),
        )


def _candidate_is_suspicious(candidate, image_shape) -> bool:
    image_height, image_width = image_shape[:2]
    x1, y1, x2, y2 = candidate.bbox_xyxy
    min_border_distance = min(x1, y1, image_width - 1 - x2, image_height - 1 - y2)
    if min_border_distance <= 4:
        return True
    if candidate.area_ratio <= 0.0003:
        return True
    return False


def _bbox_from_mask(mask) -> tuple[int, int, int, int]:
    ys, xs = mask.nonzero()
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _fallback_candidate(mask, score: float) -> MaskCandidate | None:
    area = int(mask.sum())
    if area < 100:
        return None

    image_area = float(mask.shape[0] * mask.shape[1])
    area_ratio = area / image_area
    if area_ratio > 0.55:
        return None

    x1, y1, x2, y2 = _bbox_from_mask(mask)
    width = max(1, x2 - x1 + 1)
    height = max(1, y2 - y1 + 1)
    aspect_ratio = max(width / height, height / width)
    return MaskCandidate(
        mask=mask,
        score=float(score),
        bbox_xyxy=(x1, y1, x2, y2),
        area_ratio=area_ratio,
        brightness_delta=0.0,
        aspect_ratio=aspect_ratio,
    )


def _mask_candidate_from_refinement(mask, image_rgb, base_score: float) -> MaskCandidate | None:
    candidate, _ = select_scratch_mask(
        [{"segmentation": mask}],
        image_rgb,
        top_k=1,
        minimum_score=float("-inf"),
    )
    if candidate is not None:
        candidate.score += base_score
        return candidate

    return _fallback_candidate(mask, base_score)


def _dedupe_boxes(boxes_with_scores):
    seen: set[tuple[int, int, int, int]] = set()
    deduped = []
    for source, box, score in boxes_with_scores:
        if box in seen:
            continue
        seen.add(box)
        deduped.append((source, box, score))
    return deduped


def _run_detection_boxes(perception_result, image_rgb, image_path: Path):
    boxes_with_scores = [
        ("qwen", detection.bbox_xyxy, detection.confidence)
        for detection in perception_result.detections
    ]

    named_manual_box = get_named_manual_box(image_rgb, image_path.name)
    if named_manual_box is not None:
        boxes_with_scores.append(("manual", named_manual_box, 0.35))

    for box in propose_candidate_boxes(
        image_rgb,
        image_path=image_path,
        source_name=image_path.name,
        max_boxes=6,
    ):
        boxes_with_scores.append(("fallback", box, 0.0))

    return _dedupe_boxes(boxes_with_scores)


def _run_mask_pipeline(
    segmenter: SAM2BoxSegmenter,
    perception_result,
    image_rgb,
    image_path: Path,
    top_k: int,
):
    box_candidates = _run_detection_boxes(perception_result, image_rgb, image_path)
    refined_candidates = []
    for source, box, box_score in box_candidates:
        refined_mask, sam_score = segmenter.predict_box_masks(image_rgb, box)
        refined_mask = postprocess_mask(refined_mask)
        source_bonus = 1.4 if source == "qwen" else 0.9 if source == "manual" else 0.2
        candidate = _mask_candidate_from_refinement(
            refined_mask,
            image_rgb,
            base_score=sam_score + box_score + source_bonus,
        )
        if candidate is None:
            continue
        refined_candidates.append((source, candidate))

    if not refined_candidates:
        raise RuntimeError("Could not isolate a plausible scratch mask.")

    refined_candidates.sort(key=lambda item: item[1].score, reverse=True)
    best_source, best_candidate = refined_candidates[0]
    top_candidates = [candidate for _, candidate in refined_candidates[:top_k]]

    if _candidate_is_suspicious(best_candidate, image_rgb.shape) and len(top_candidates) > 1:
        best_candidate = top_candidates[1]

    if best_source == "qwen":
        method = "qwen-vlm -> sam2.1"
    elif best_source == "manual":
        method = "manual-box -> sam2.1"
    else:
        method = "fallback-box -> sam2.1"

    return postprocess_mask(best_candidate.mask), method, top_candidates, box_candidates


def _process_and_save(
    segmenter: SAM2BoxSegmenter,
    perception_layer: QwenPerceptionLayer,
    input_path: Path,
    mask_path: Path,
    overlay_path: Path,
    comparison_path: Path,
    perception_dir: Path,
    output_dir: Path,
    top_k: int,
    dry_run: bool = False,
) -> None:
    image_bgr, _ = load_image(input_path)
    working_bgr, _ = crop_before_panel(image_bgr, input_path.name)
    working_rgb = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2RGB)
    image_height, image_width = working_rgb.shape[:2]
    stem = sanitize_stem(input_path)

    perception_input_path = perception_dir / f"{stem}_input.png"
    perception_input_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(perception_input_path), working_bgr)

    perception_result = perception_layer.detect(
        perception_input_path,
        image_width=image_width,
        image_height=image_height,
    )
    save_json(perception_dir / f"{stem}.json", perception_result.to_dict())

    vlm_boxes = [detection.bbox_xyxy for detection in perception_result.detections]
    save_box_preview(perception_dir / f"{stem}_boxes.png", working_bgr, vlm_boxes)

    if dry_run:
        print(f"{input_path.name}: dry-run perception complete")
        print(f"  detections: {len(perception_result.detections)}")
        print(f"  perception_json: {perception_dir / f'{stem}.json'}")
        return

    final_mask, method, top_candidates, box_candidates = _run_mask_pipeline(
        segmenter=segmenter,
        perception_result=perception_result,
        image_rgb=working_rgb,
        image_path=input_path,
        top_k=top_k,
    )
    save_mask(mask_path, final_mask)
    save_overlay(overlay_path, working_bgr, final_mask)

    overlay_bgr = overlay_mask(working_bgr, final_mask)
    save_side_by_side_comparison(comparison_path, working_bgr, overlay_bgr)

    candidate_dir = output_dir / f"{stem}_candidates"
    _save_candidate_overlays(working_bgr, top_candidates, candidate_dir)
    print(f"{input_path.name}: {method}")
    print(f"  mask: {mask_path}")
    print(f"  overlay: {overlay_path}")
    print(f"  comparison: {comparison_path}")
    print(f"  perception_json: {perception_dir / f'{stem}.json'}")
    print(f"  candidate_boxes: {[box for _, box, _ in box_candidates[:3]]}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the VLM perception layer only on the primary image.",
    )
    return parser


def run() -> None:
    args = _build_parser().parse_args()
    settings = get_settings()
    ensure_directories(
        [
            settings.data_dir,
            settings.outputs_dir,
            settings.models_dir,
            settings.perception_dir,
        ]
    )

    perception_layer = QwenPerceptionLayer(
        settings.perception_model_path,
        max_tokens=settings.qwen_max_tokens,
    )
    segmenter = SAM2BoxSegmenter(settings.sam2_model_id, settings.sam2_device)

    _process_and_save(
        segmenter=segmenter,
        perception_layer=perception_layer,
        input_path=settings.input_path,
        mask_path=settings.mask_path,
        overlay_path=settings.overlay_path,
        comparison_path=settings.comparison_path,
        perception_dir=settings.perception_dir,
        output_dir=settings.outputs_dir,
        top_k=settings.top_k_candidates,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(f"Perception model: {settings.perception_model_path}")
        return

    extra_images = []
    for image_path in discover_reference_images(settings.project_root):
        if image_path.resolve() == settings.input_path.resolve():
            continue
        if image_path.name == "222 copy.jpeg":
            continue
        extra_images.append(image_path)

    for image_path in extra_images:
        stem = sanitize_stem(image_path)
        _process_and_save(
            segmenter=segmenter,
            perception_layer=perception_layer,
            input_path=image_path,
            mask_path=settings.outputs_dir / f"mask_{stem}.png",
            overlay_path=settings.outputs_dir / f"overlay_{stem}.png",
            comparison_path=settings.outputs_dir / f"comparison_{stem}.png",
            perception_dir=settings.perception_dir,
            output_dir=settings.outputs_dir,
            top_k=settings.top_k_candidates,
        )

    print(f"SAM2 device: {settings.sam2_device}")
    print(f"Perception model: {settings.perception_model_path}")
    print(f"Primary overlay saved to: {settings.overlay_path}")


if __name__ == "__main__":
    run()
