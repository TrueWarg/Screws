import torch

from bbox.metrics import iou


def hard_nms(
        box_and_scores: torch.Tensor,
        iou_threshold: float,
        top_k=-1,
        candidate_size=200,
):
    scores = box_and_scores[:, -1]
    boxes = box_and_scores[:, :-1]
    picked = []
    _, indices = scores.sort(descending=True)
    indices = indices[:candidate_size]
    while len(indices) > 0:
        current = indices[0]
        picked.append(current.item())
        if 0 < top_k == len(picked) or len(indices) == 1:
            break
        current_box = boxes[current, :]
        indices = indices[1:]
        rest_boxes = boxes[indices, :]
        ious = iou(
            rest_boxes,
            current_box.unsqueeze(0),
        )
        indices = indices[ious <= iou_threshold]

    return box_and_scores[picked, :]
