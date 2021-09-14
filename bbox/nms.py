import torch

from bbox.metrics import rotated_xyxy_iou, iou


def hard_nms(
        box_scores: torch.Tensor,
        iou_threshold: float,
        top_k=-1,
        candidate_size=200
):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    _, indexes = scores.sort(descending=True)
    indexes = indexes[:candidate_size]
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        ious = iou(
            rest_boxes,
            current_box.unsqueeze(0),
        )
        indexes = indexes[ious <= iou_threshold]

    return box_scores[picked, :]
