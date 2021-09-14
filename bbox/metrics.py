import torch


def rotated_xyxy_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor, eps=1e-5) -> torch.Tensor:
    """
    Calculate rotated intersection over union in 2D space:
    RIoU = (INTERSECTION / UNION) * cos(angle_a - angle_b)
    :param boxes_a:  bounding boxes with shape [N, 5] and with format xcycwha
    :param boxes_b:  bounding boxes with shape [N, 5] and with format xcycwha
    :param eps: additional value to union
    :return: iou scores in [0.0 - 1.0] with shape [N, 4]
    """
    return iou(boxes_a[..., :4], boxes_b[..., :4], eps) * torch.cos(boxes_a[..., 4] - boxes_b[..., 4])


def area(left_top: torch.Tensor, right_bottom: torch.Tensor) -> torch.Tensor:
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def iou(boxes0, boxes1, eps=1e-5):
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

    overlap_area = area(overlap_left_top, overlap_right_bottom)
    area0 = area(boxes0[..., :2], boxes0[..., 2:4])
    area1 = area(boxes1[..., :2], boxes1[..., 2:4])
    return overlap_area / (area0 + area1 - overlap_area + eps)
