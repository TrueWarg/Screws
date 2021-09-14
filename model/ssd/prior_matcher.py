from typing import Tuple

import numpy as np
import torch

from bbox.converters import convert_boxes_to_locations, corner_form_to_center_form, \
    center_form_to_corner_form


def _rotated_xyxy_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor, eps=1e-5) -> torch.Tensor:
    """
    Calculate rotated intersection over union in 2D space:
    RIoU = (INTERSECTION / UNION) * cos(angle_a - angle_b)
    :param boxes_a:  bounding boxes with shape [N, 5] and with format xcycwha
    :param boxes_b:  bounding boxes with shape [N, 5] and with format xcycwha
    :param eps: additional value to union
    :return: iou scores in [0.0 - 1.0] with shape [N, 4]
    """
    return _iou(boxes_a[..., :4], boxes_b[..., :4], eps) * torch.cos(boxes_a[..., 4] - boxes_b[..., 4])


def _area(left_top: torch.Tensor, right_bottom: torch.Tensor) -> torch.Tensor:
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def _iou(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:4], boxes1[..., 2:4])

    overlap_area = _area(overlap_left_top, overlap_right_bottom)
    area0 = _area(boxes0[..., :2], boxes0[..., 2:4])
    area1 = _area(boxes1[..., :2], boxes1[..., 2:4])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def _assign_priors(target_boxes: torch.Tensor,
                   target_categories: torch.Tensor,
                   priors: torch.Tensor,
                   riou_threshold: float) -> Tuple:
    # rotated_ious have shape = (prior_count, target_count)
    ious = _rotated_xyxy_iou(target_boxes.unsqueeze(0), priors.unsqueeze(1))
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
        locations = convert_boxes_to_locations(
            boxes,
            self._center_form_priors,
            self._center_variance,
            self._size_variance,
        )
        return locations, labels
