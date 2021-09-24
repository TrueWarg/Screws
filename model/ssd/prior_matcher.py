from typing import Tuple

import numpy as np
import torch

from bbox.converters import boxes_to_locations, corner_form_to_center_form, \
    center_form_to_corner_form
from bbox.metrics import rotated_xyxy_iou


def _assign_priors(target_boxes: torch.Tensor,
                   target_categories: torch.Tensor,
                   priors: torch.Tensor,
                   riou_threshold: float) -> Tuple:
    # rotated_ious have shape = (prior_count, target_count)
    ious = rotated_xyxy_iou(target_boxes.unsqueeze(0), priors.unsqueeze(1))
    best_target_per_prior, best_target_per_prior_index = ious.max(1)
    best_prior_per_target, best_prior_per_target_index = ious.max(0)

    for target_index, prior_index in enumerate(best_prior_per_target_index):
        best_target_per_prior_index[prior_index] = target_index
    # 2.0 is used to make sure every target has a prior assigned
    best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
    labels = target_categories[best_target_per_prior_index]
    # the backgournd id
    labels[best_target_per_prior < riou_threshold] = 0
    boxes = target_boxes[best_target_per_prior_index]

    return boxes, labels


class RotatedPriorMatcher(object):
    def __init__(self, center_form_priors, center_variance: float, size_variance: float, iou_threshold: float):
        self._center_form_priors = center_form_priors
        self._corner_form_priors = center_form_to_corner_form(center_form_priors)
        self._center_variance = center_variance
        self._size_variance = size_variance
        self._iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)

        boxes, labels = _assign_priors(gt_boxes, gt_labels, self._corner_form_priors, self._iou_threshold)
        boxes = corner_form_to_center_form(boxes)
        locations = boxes_to_locations(
            boxes,
            self._center_form_priors,
            self._center_variance,
            self._size_variance,
        )
        return locations, labels
