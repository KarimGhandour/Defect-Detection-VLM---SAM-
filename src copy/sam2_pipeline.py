from __future__ import annotations

import numpy as np
from sam2.build_sam import build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2BoxSegmenter:
    def __init__(self, model_id: str, device: str) -> None:
        self.model_id = model_id
        self.device = device
        self.model = build_sam2_hf(model_id, device=device)
        self.predictor = SAM2ImagePredictor(self.model)

    def predict_box_masks(
        self,
        image_rgb: np.ndarray,
        box_xyxy: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, float]:
        self.predictor.set_image(image_rgb)
        box = np.array(box_xyxy, dtype=np.float32)
        masks, scores, _ = self.predictor.predict(box=box, multimask_output=True)
        best_idx = int(np.argmax(scores))
        return masks[best_idx].astype(bool), float(scores[best_idx])
