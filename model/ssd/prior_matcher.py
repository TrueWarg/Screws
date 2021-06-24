import numpy as np
import torch
from typing import Tuple

from bbox.converters import xcycwha_to_xyxya, xyxy_to_xcycwh, convert_boxes_to_locations
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
    ious = rotated_xcycwh_iou(target_boxes.unsqueeze(0), priors.unsqueeze(1))
    best_target_per_prior, best_target_per_prior_index = ious.max(1)
    best_prior_per_target, best_prior_per_target_index = ious.max(0)

    for target_index, prior_index in enumerate(best_prior_per_target_index):
        best_target_per_prior_index[prior_index] = target_index
    # 2.0 is used to make sure every target has a prior assigned
    best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
    labels = target_categories[best_target_per_prior_index]
    labels[best_target_per_prior < riou_threshold] = 0
    boxes = target_boxes[best_target_per_prior_index]

    return boxes, labels


class RotatedPriorMatcher(object):
    # todo rename xcycy in all places as center_form_priors?

    def __init__(self, center_form_priors, center_variance: float, size_variance: float, iou_threshold: float):
        self._center_form_priors = center_form_priors
        self._corner_form_priors = xcycwha_to_xyxya(center_form_priors)
        self._center_variance = center_variance
        self._size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)

        boxes, labels = _assign_priors(
            gt_boxes,
            gt_labels,
            self._corner_form_priors,
            self.iou_threshold,
        )
        boxes = xyxy_to_xcycwh(boxes)
        locations = convert_boxes_to_locations(
            boxes,
            self._center_form_priors,
            self._center_variance,
            self._size_variance,
        )
        return locations, labels
