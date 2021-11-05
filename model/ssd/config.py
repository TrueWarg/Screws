from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    image_size: int
    image_mean: np.ndarray
    image_std: float
    iou_threshold: float
    center_variance: float
    size_variance: float
