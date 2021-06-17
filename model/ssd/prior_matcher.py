import torch
from typing import Tuple

from bbox.metrics import rotated_xcycwh_iou


def _make_offset_transformation(boxes: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
    return torch.cat([
        (boxes[..., :2] - priors[..., :2]) / priors[..., 2:],
        torch.log(boxes[..., 2:4] / priors[..., 2:4]),
        torch.cos(boxes[..., 4] - priors[..., 4])
    ], dim=boxes.dim() - 1)


def _assign_priors(target_boxes: torch.Tensor,
                   target_categories: torch.Tensor,
                   priors: torch.Tensor,
                   riou_threshold: float) -> Tuple:
    # rotated_ious have shape = (prior_count, target_count)
    rotated_ious = rotated_xcycwh_iou(target_boxes.unsqueeze(0), priors.unsqueeze(1))

    # target for priors. shape = (prior_count)
    max_scores, target_indices = rotated_ious.max(1)

    categories = target_categories[target_indices]
    categories[target_indices < riou_threshold] = 0  # mark as bg id = 0
    boxes = target_boxes[target_indices]

    return boxes, categories


class RotatedPriorMatcher(object):
    def __init__(self, priors: torch.Tensor, riou_threshold: float):
        self.priors = priors
        self.riou_threshold = riou_threshold

    def __call__(self, target_boxes: torch.Tensor, target_categories: torch.Tensor):
        boxes, labels = _assign_priors(target_boxes, target_categories, self.priors, self.riou_threshold)
        boxes = _make_offset_transformation(boxes, self.priors)
        return boxes, labels
