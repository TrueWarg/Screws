import torch

from bbox.converters import xcycwh_to_xyxy
from math import cos


def xyxy_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    """
    Calculate intersection over union in 2D space:
    IoU = INTERSECTION / UNION
    :param box_a:  bounding box with format xyxy
    :param box_b:  bounding box with format xyxy
    :return: iou score in [0.0 - 1.0]
    """
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    intersection = max(0.0, (x_b - x_a)) * max(0.0, (y_b - y_a))

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union = area_a + area_b - intersection

    return intersection / union


def xcycwh_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    """
    Calculate intersection over union in 2D space:
    IoU = INTERSECTION / UNION
    :param box_a:  bounding box with format xcycwh
    :param box_b:  bounding box with format xcycwh
    :return: iou score in [0.0 - 1.0]
    """

    return xyxy_iou(xcycwh_to_xyxy(box_a), xcycwh_to_xyxy(box_b))


def rotated_xcycwh_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    """
    Calculate rotated intersection over union in 2D space:
    IoU = (INTERSECTION / UNION) * cos(angle_a - angle_b)
    :param box_a:  bounding box with format xcycwha with angle in radian
    :param box_b:  bounding box with format xcycwha with angle in radian
    :return: iou score in [0.0 - 1.0]
    """
    return xcycwh_iou(box_a[:4], box_b[:4]) * cos(box_a[4] - box_b[4])
