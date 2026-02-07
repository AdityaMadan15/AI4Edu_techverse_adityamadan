from __future__ import annotations

from typing import Iterable, List

import cv2
import numpy as np
import torch

# Dummy gaze estimator without MediaPipe dependency
class GazeEstimator:
    def __init__(self) -> None:
        pass

    def extract(self, frame_bgr: np.ndarray) -> torch.Tensor:
        # Return dummy gaze features (3 features: left_eye, right_eye, gaze_horizontal)
        return torch.zeros(3, dtype=torch.float32)

    def batch_extract(self, frames_bgr: Iterable[np.ndarray]) -> torch.Tensor:
        # Return dummy gaze features for all frames
        num_frames = len(list(frames_bgr)) if not isinstance(frames_bgr, list) else len(frames_bgr)
        return torch.zeros((num_frames, 3), dtype=torch.float32)

    def close(self) -> None:
        pass
