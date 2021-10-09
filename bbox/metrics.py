from typing import Dict

import numpy as np
import torch


def rotated_xyxy_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor, eps=1e-5) -> torch.Tensor:
    """
        Calculate rotated intersection over union in 2D space:
        RIoU = (INTERSECTION / UNION) * cos(angle_a - angle_b)

    Args:
        boxes_a:  bounding boxes with shape [N, 5] and with format xcycwha
        boxes_b:  bounding boxes with shape [N, 5] and with format xcycwha
        eps: additional value to union

    Return:
         iou scores in [0.0 - 1.0] with shape [N, 4]
    """
    return iou(boxes_a[..., :4], boxes_b[..., :4], eps) * torch.cos(boxes_a[..., 4] - boxes_b[..., 4])


def area(left_top: torch.Tensor, right_bottom: torch.Tensor) -> torch.Tensor:
    """
        Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor, eps=1e-5):
    """
        Calculate rotated intersection over union in 2D space:
        RIoU = (INTERSECTION / UNION)

    Args:
        boxes_a:  bounding boxes with shape [N, 4 and with format xcycwha
        boxes_b:  bounding boxes with shape [N, 4] and with format xcycwha
        eps: additional value to union

    Return:
         iou scores in [0.0 - 1.0] with shape [N, 4]
    """
    overlap_left_top = torch.max(boxes_a[..., :2], boxes_b[..., :2])
    overlap_right_bottom = torch.min(boxes_a[..., 2:4], boxes_b[..., 2:4])

    overlap_area = area(overlap_left_top, overlap_right_bottom)
    area_a = area(boxes_a[..., :2], boxes_a[..., 2:4])
    area_b = area(boxes_b[..., :2], boxes_b[..., 2:4])
    return overlap_area / (area_a + area_b - overlap_area + eps)


def compute_average_precision_per_class(
        num_true_cases: int,
        gt_boxes: Dict,
        difficult_cases: Dict,
        prediction_file: str,
        iou_threshold: float,
):
    """

    :param num_true_cases: count object labeled as current class. Necessary for recall calculation.
    :param gt_boxes:
    :param difficult_cases:
    :param prediction_file:
    :param iou_threshold:
    :return: average precision for class
    """
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            predicted_line = line.rstrip().split(" ")
            image_ids.append(predicted_line[0])
            scores.append(float(predicted_line[1]))
            box = torch.tensor([float(v) for v in predicted_line[2:]]).unsqueeze(0)
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue

            gt_box = gt_boxes[image_id]
            # ious = box_utils.iou_of(box, gt_box) * torch.cos(box[..., 4] - gt_box[..., 4])
            ious = iou(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases
    return compute_voc2007_average_precision(precision, recall)


def compute_voc2007_average_precision(precision: float, recall: float):
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap = ap + p / 11.
    return ap
