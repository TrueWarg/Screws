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
