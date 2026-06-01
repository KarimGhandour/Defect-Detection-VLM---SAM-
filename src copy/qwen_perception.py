from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

from mlx_vlm import load
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template


@dataclass
class PerceptionDetection:
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    label: str
    source_key: str


@dataclass
class PerceptionResult:
    model_path: str
    raw_text: str
    detections: list[PerceptionDetection]
    prompt_tokens: int
    generation_tokens: int
    prompt_tps: float
    generation_tps: float
    peak_memory: float

    def to_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "raw_text": self.raw_text,
            "detections": [asdict(detection) for detection in self.detections],
            "prompt_tokens": self.prompt_tokens,
            "generation_tokens": self.generation_tokens,
            "prompt_tps": self.prompt_tps,
            "generation_tps": self.generation_tps,
            "peak_memory_gb": self.peak_memory,
        }


def _extract_json_blob(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", stripped)
        stripped = re.sub(r"\n```$", "", stripped)

    start_positions = [idx for idx in (stripped.find("{"), stripped.find("[")) if idx >= 0]
    if not start_positions:
        return stripped

    start = min(start_positions)
    end = max(stripped.rfind("}"), stripped.rfind("]"))
    if end <= start:
        return stripped[start:]
    return stripped[start : end + 1]


def _coerce_bbox(
    value,
    image_width: int,
    image_height: int,
    source_key: str,
) -> tuple[int, int, int, int] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None

    coords = [float(item) for item in value]
    if source_key == "bbox_2d" and max(coords) <= 1000.0:
        x1 = int(round((coords[0] / 1000.0) * image_width))
        y1 = int(round((coords[1] / 1000.0) * image_height))
        x2 = int(round((coords[2] / 1000.0) * image_width))
        y2 = int(round((coords[3] / 1000.0) * image_height))
    elif max(coords) <= 1.5:
        x1 = int(round(coords[0] * image_width))
        y1 = int(round(coords[1] * image_height))
        x2 = int(round(coords[2] * image_width))
        y2 = int(round(coords[3] * image_height))
    else:
        x1, y1, x2, y2 = [int(round(coord)) for coord in coords]

    x1, x2 = sorted((max(0, x1), min(image_width - 1, x2)))
    y1, y2 = sorted((max(0, y1), min(image_height - 1, y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _parse_detections(
    raw_text: str,
    image_width: int,
    image_height: int,
) -> list[PerceptionDetection]:
    json_blob = _extract_json_blob(raw_text)
    if not json_blob:
        return []

    try:
        payload = json.loads(json_blob)
    except json.JSONDecodeError:
        return []

    if isinstance(payload, list):
        detection_items = payload
    else:
        detection_items = payload.get("detections", [])

    detections: list[PerceptionDetection] = []
    for item in detection_items:
        if not isinstance(item, dict):
            continue

        bbox = None
        source_key = ""
        for key in ("bbox_xyxy", "bbox_2d", "bbox", "box"):
            if key in item:
                bbox = _coerce_bbox(
                    item[key],
                    image_width,
                    image_height,
                    source_key=key,
                )
                source_key = key
                break

        if bbox is None:
            continue

        detections.append(
            PerceptionDetection(
                bbox_xyxy=bbox,
                confidence=float(item.get("confidence", item.get("score", 0.5))),
                label=str(item.get("label", "scratch")),
                source_key=source_key,
            )
        )

    return detections


class QwenPerceptionLayer:
    def __init__(
        self,
        model_path: Path,
        max_tokens: int = 160,
    ) -> None:
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.model, self.processor = load(str(model_path))

    def _build_prompt(
        self,
        image_width: int,
        image_height: int,
        relaxed: bool = False,
    ) -> str:
        if relaxed:
            return (
                "You are a leather defect proposal model. "
                f"Image size is {image_width}x{image_height} pixels. "
                "Task: detect visible leather scratch or scuff damage and return JSON only. "
                "Rules: prefer thin irregular damage, not stitching, not seat seams, "
                "not panel borders, and not specular highlights. "
                "Return up to 3 ranked detections. "
                'Format: {"detections":[{"bbox_xyxy":[x1,y1,x2,y2],"confidence":0.0,"label":"scratch"}]}. '
                'If no plausible damage exists, return {"detections":[]}.'
            )

        return (
            "You are a leather defect proposal model. "
            f"Image size is {image_width}x{image_height} pixels. "
            "Return JSON only. No explanation. No reasoning. No markdown. "
            "Detect only narrow localized leather scratches or scuffs. "
            "Bounding boxes must be tight around damaged pixels only. "
            "Exclude fingers, hands, stitching, seat seams, folds, panel borders, "
            "specular highlights, and broad worn regions. "
            "Do not cover intact leather for context. "
            "Return up to 3 ranked detections. "
            'Output exactly {"detections":[{"bbox_xyxy":[x1,y1,x2,y2],"confidence":0.0,"label":"scratch"}]}. '
            'If no plausible damage exists, return {"detections":[]}.'
        )

    def _run_generation(self, image_path: Path, prompt: str):
        formatted_prompt = apply_chat_template(
            self.processor,
            self.model.config,
            prompt,
            num_images=1,
            enable_thinking=False,
        )
        return generate(
            self.model,
            self.processor,
            prompt=formatted_prompt,
            image=str(image_path),
            max_tokens=self.max_tokens,
            temperature=0.0,
            prefill_step_size=256,
            enable_thinking=False,
        )

    def detect(
        self,
        image_path: Path,
        image_width: int,
        image_height: int,
    ) -> PerceptionResult:
        generation = self._run_generation(
            image_path,
            self._build_prompt(image_width, image_height, relaxed=False),
        )
        detections = _parse_detections(generation.text, image_width, image_height)
        if not detections:
            generation = self._run_generation(
                image_path,
                self._build_prompt(image_width, image_height, relaxed=True),
            )
            detections = _parse_detections(generation.text, image_width, image_height)
        return PerceptionResult(
            model_path=str(self.model_path),
            raw_text=generation.text,
            detections=detections,
            prompt_tokens=generation.prompt_tokens,
            generation_tokens=generation.generation_tokens,
            prompt_tps=generation.prompt_tps,
            generation_tps=generation.generation_tps,
            peak_memory=generation.peak_memory,
        )
