from dataclasses import dataclass
from typing import Tuple

import torch

from bbox.metrics import iou


@dataclass(frozen=True)
class IoUThresholdBound:
    bottom: float
    top: float


def _match_boxes(
        targets: torch.Tensor,
        priors: torch.tensor,
        iou_bounds: IoUThresholdBound,
        batch=8,
) -> Tuple:
    ious = torch.cat([iou(targets[i: i + batch], priors) for i in range(0, targets.size(0), batch)])

    max_ious, box_indices = torch.max(ious, dim=0)
    neg_mask = max_ious < iou_bounds.bottom
    pos_mask = max_ious > iou_bounds.top
    return pos_mask, neg_mask, box_indices


def encode(gt_bbox, det_bbox, anchor, variance):
    xy = (gt_bbox[..., 0: 2] - anchor[..., 0: 2]) / anchor[..., 2: 4] / variance[0]
    wh = torch.log(gt_bbox[..., 2: 4] / anchor[..., 2: 4]) / variance[1]
    a = gt_bbox[..., [4]] / 45 / variance[2]
    gt_bbox = torch.cat([xy, wh, a], dim=-1)
    det_bbox = torch.cat([det_bbox[..., :4], torch.tanh(det_bbox[..., [4]]) / variance[2]], dim=-1)
    return gt_bbox, det_bbox


def decode(det_bbox, anchor, variance):
    xy = det_bbox[..., 0: 2] * variance[0] * anchor[..., 2: 4] + anchor[..., 0: 2]
    wh = torch.exp(det_bbox[..., 2: 4] * variance[1]) * anchor[..., 2: 4]
    a = torch.tanh(det_bbox[..., [4]]) * 45
    return torch.cat([xy, wh, a], dim=-1)


class RotatedBboxLoss:
    def __init__(self,
                 iou_bounds: IoUThresholdBound,
                 variance: float,
                 pos_neg_balance: float,
                 num_classes: int,
                 device=None,
                 ):
        self._iou_bounds = iou_bounds
        self._variance = variance
        self._balance = pos_neg_balance
        self._device = device
        self._num_classes = num_classes

    def __call__(self,
                 predicted_class_scorers: torch.Tensor,
                 predicted_locs: torch.Tensor,
                 target_class_ids: torch.Tensor,
                 target_boxes: torch.Tensor,
                 prior_boxes: torch.Tensor,
                 ):
        weight_pos, weight_neg = 2 * self._balance, 2 * (1 - self._balance)

        score_loss, loc_loss = torch.zeros([2], dtype=torch.float, device=self._device, requires_grad=True)
        num_positives = 0
        for index, class_ids in enumerate(target_class_ids):
            boxes = target_boxes[index]
            pos_mask, neg_mask, box_indices = _match_boxes(boxes[:, :4], prior_boxes, self._iou_bounds)
            positive_indices = box_indices[pos_mask]

            matched_boxes = boxes[positive_indices]
            matched_priors = prior_boxes[pos_mask]
            predicted_boxes = predicted_locs[index][pos_mask]

            class_ids = class_ids[box_indices]
