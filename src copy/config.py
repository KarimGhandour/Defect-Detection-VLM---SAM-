from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import torch


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    outputs_dir: Path
    perception_dir: Path
    models_dir: Path
    input_path: Path
    mask_path: Path
    overlay_path: Path
    comparison_path: Path
    perception_model_path: Path
    sam2_model_id: str
    sam2_device: str
    top_k_candidates: int = 3
    qwen_max_tokens: int = 160


def _select_sam2_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_perception_model_path() -> Path:
    configured_path = os.environ.get("LEATHER_VLM_MODEL_PATH")
    candidate_paths = [
        Path(configured_path).expanduser() if configured_path else None,
        Path.home() / ".lmstudio/models/mlx-community/Qwen3.5-9B-MLX-4bit",
        Path.home() / ".cache/huggingface/hub/dealignai/Qwen3.5-VL-9B-8bit-MLX-CRACK",
        Path.home() / ".lmstudio/models/dealignai/Qwen3.5-VL-9B-8bit-MLX-CRACK",
    ]

    for candidate_path in candidate_paths:
        if candidate_path is not None and candidate_path.exists():
            return candidate_path

    raise FileNotFoundError(
        "No compatible local Qwen VL model path was found. "
        "Set LEATHER_VLM_MODEL_PATH to a local MLX model directory."
    )


def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parent.parent
    outputs_dir = project_root / "outputs"
    perception_dir = outputs_dir / "perception"

    return Settings(
        project_root=project_root,
        data_dir=project_root / "data",
        outputs_dir=outputs_dir,
        perception_dir=perception_dir,
        models_dir=project_root / "models",
        input_path=project_root / "data" / "input.jpg",
        mask_path=outputs_dir / "mask.png",
        overlay_path=outputs_dir / "overlay.png",
        comparison_path=outputs_dir / "comparison_input.png",
        perception_model_path=_resolve_perception_model_path(),
        sam2_model_id="facebook/sam2.1-hiera-large",
        sam2_device=_select_sam2_device(),
    )
