from typing import Tuple

import numpy as np
import torch

from bbox.converters import boxes_to_locations, corner_form_to_center_form, \
    center_form_to_corner_form
from bbox.metrics import rotated_xyxy_iou


def _assign_targets_for_priors(target_boxes: torch.Tensor,
                               target_categories: torch.Tensor,
                               priors: torch.Tensor,
                               riou_threshold: float) -> Tuple:
    """
        For each prior is assign target box, target label if ios >= riou_threshold else background label
    """
    # insert additional dimensions.
    # Let a.shape = (target_count, 5) b.shape = (prior_count, 5);
    # unsqueeze -> a.shape = (1, target_count, 5) b.shape = (prior_count, 1, 5).
    # Calculation by last dimension (boxes) leads to to rotated_ious combination
    # with shape (prior_count, target_count) (np.subtract.outer analog)
    ious = rotated_xyxy_iou(target_boxes.unsqueeze(0), priors.unsqueeze(1))
    # rows - priors, column - targets
    best_prior_per_target_values, best_prior_per_target_indices = ious.max(0)
    best_target_per_prior_values, best_target_per_prior_indices = ious.max(1)

    # hack: 1 is used to make sure correct assign target label:
    # in other case best_target_per_prior_values < riou_threshold will be always true and label will be background
    # best_prior_per_target_indices is used for this goal because it contains indices
    # (row in matrix == indices of best_prior_per_target_values list) that
    # indicate max ious both in best_prior_per_target_values and best_target_per_prior_values
    best_target_per_prior_values.index_fill_(0, best_prior_per_target_indices, 1)
    labels = target_categories[best_target_per_prior_indices]
    # the background id
    labels[best_target_per_prior_values < riou_threshold] = 0
    boxes = target_boxes[best_target_per_prior_indices]

    return boxes, labels


class RotatedPriorMatcher:
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

        boxes, labels = _assign_targets_for_priors(gt_boxes, gt_labels, self._corner_form_priors, self._iou_threshold)
        boxes = corner_form_to_center_form(boxes)
        locations = boxes_to_locations(
            boxes,
            self._center_form_priors,
            self._center_variance,
            self._size_variance,
        )
        return locations, labels
