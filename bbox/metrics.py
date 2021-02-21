import torch

from bbox.converters import xcycwh_to_xyxy


def _area(lt: torch.Tensor, rb: torch.Tensor) -> torch.Tensor:
    hw = torch.clamp(rb - lt, min=0.0)
    return hw[..., 0] * hw[..., 1]


def xyxy_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor, eps=1e-5) -> torch.Tensor:
    """
    Calculate intersection over union in 2D space:
    IoU = INTERSECTION / UNION
    :param boxes_a:  bounding boxes with shape [N, 4] and with format xyxy
    :param boxes_b:  bounding boxes with shape [N, 4] and with format xyxy
    :param eps: additional value to union
    :return: iou scores in [0.0 - 1.0] with shape [N, 4]
    """
    left_top = torch.max(boxes_a[..., :2], boxes_b[..., :2])
    right_bottom = torch.min(boxes_a[..., 2:], boxes_b[..., 2:])
    intersection = _area(left_top, right_bottom)

    area_a = _area(boxes_a[..., :2], boxes_a[..., 2:])
    area_b = _area(boxes_b[..., :2], boxes_b[..., 2:])
    union = area_a + area_b - intersection + eps

    return intersection / union


def xcycwh_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor, eps=1e-5) -> torch.Tensor:
    """
    Calculate intersection over union in 2D space:
    IoU = INTERSECTION / UNION
    :param boxes_a:  bounding boxes with shape [N, 4] and with format xcycwh
    :param boxes_b:  bounding boxes with shape [N, 4] and with format xcycwh
    :param eps: additional value to union
    :return: iou scores in [0.0 - 1.0] with shape [N, 4]
    """
    return xyxy_iou(xcycwh_to_xyxy(boxes_a), xcycwh_to_xyxy(boxes_b), eps)


def rotated_xcycwh_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor, eps=1e-5) -> torch.Tensor:
    """
    Calculate rotated intersection over union in 2D space:
    RIoU = (INTERSECTION / UNION) * cos(angle_a - angle_b)
    :param boxes_a:  bounding boxes with shape [N, 5] and with format xcycwha
    :param boxes_b:  bounding boxes with shape [N, 5] and with format xcycwha
    :param eps: additional value to union
    :return: iou scores in [0.0 - 1.0] with shape [N, 4]
    """
    return xcycwh_iou(boxes_a[..., :4], boxes_b[..., :4], eps) * torch.cos(boxes_a[..., 4] - boxes_b[..., 4])
