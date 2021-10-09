import torch

from bbox.metrics import iou


def hard_nms(
        boxes_and_scores: torch.Tensor,
        iou_threshold: float,
        top_k=-1,
        candidate_size=200,
):
    scores = boxes_and_scores[:, -1]
    boxes = boxes_and_scores[:, :-1]
    picked = []
    _, indices = scores.sort(descending=True)
    indices = indices[:candidate_size]
    while len(indices) > 0:
        current_index = indices[0]
        picked.append(current_index.item())
        if 0 < top_k == len(picked) or len(indices) == 1:
            break
        current_max_scored_box = boxes[current_index, :]
        indices = indices[1:]
        rest_boxes = boxes[indices, :]
        ious = iou(
            rest_boxes,
            current_max_scored_box.unsqueeze(0),
        )
        indices = indices[ious <= iou_threshold]

    return boxes_and_scores[picked, :]
