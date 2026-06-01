from __future__ import annotations

import json
from pathlib import Path
import re
from urllib.request import urlretrieve

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def ensure_directories(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def validate_input_image(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Input image not found at {path}. Place the image at ./data/input.jpg."
        )


def download_file(url: str, destination: Path) -> Path:
    if destination.exists():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, destination)
    return destination


def load_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
    validate_input_image(path)
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not decode image at {path}.")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_bgr, image_rgb


def save_mask(path: Path, mask: np.ndarray) -> None:
    mask_uint8 = (mask.astype(np.uint8) * 255)
    cv2.imwrite(str(path), mask_uint8)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def discover_reference_images(project_root: Path) -> list[Path]:
    image_paths = []
    for path in sorted(project_root.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image_paths.append(path)
    return image_paths


def sanitize_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("_")


def crop_before_panel(
    image_bgr: np.ndarray,
    source_name: str,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    lower_name = source_name.lower()
    height, width = image_bgr.shape[:2]
    if "beforeafter" not in lower_name:
        return image_bgr.copy(), (0, 0, width, height)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    search_start = int(width * 0.35)
    search_end = int(width * 0.65)
    col_mean = gray[:, search_start:search_end].mean(axis=0)
    divider = search_start + int(np.argmin(col_mean))
    divider = max(int(width * 0.45), min(int(width * 0.55), divider))
    divider = max(1, divider)
    top_trim = int(height * 0.18)
    cropped = image_bgr[top_trim:, :divider].copy()
    return cropped, (0, top_trim, divider, height)
