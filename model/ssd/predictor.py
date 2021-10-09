import torch
from torch import nn

from bbox.nms import hard_nms
from dataset.voc_dataset import BACKGROUND_CLASS_ID


class Predictor:
    def __init__(self,
                 net: nn.Module,
                 transform=None,
                 iou_threshold=0.45,
                 filter_threshold=0.4,
                 candidate_size=200,
                 device=None
                 ):
        self._net = net
        self._transform = transform
        self._iou_threshold = iou_threshold
        self._filter_threshold = filter_threshold
        self._candidate_size = candidate_size
        self._device = device

        if device:
            self._net.to(device)

        self._net.eval()

    def predict(self, image, top_k=-1):
        height, width, _ = image.shape
        image = self._transform(image)
        image = image.unsqueeze(0)
        image = image.to(self._device)
        with torch.no_grad():
            scores, boxes = self._net.forward(image)

        boxes = boxes.squeeze(0)
        scores = scores.squeeze(0)

        picked_boxes_and_scores = []
        picked_labels = []

        for class_id in range(scores.size(1)):
            # ignore bg class
            if class_id == BACKGROUND_CLASS_ID:
                continue
            subset_scores = scores[:, class_id]
            mask = subset_scores > self._filter_threshold
            print(f" probs {subset_scores.max()}")
            subset_scores = subset_scores[mask]

            if subset_scores.size(0) == 0:
                continue

            subset_boxes = boxes[mask, :]

            boxes_and_scores = torch.cat([subset_boxes, subset_scores.reshape(-1, 1)], dim=1)
            boxes_and_scores = hard_nms(
                boxes_and_scores=boxes_and_scores,
                iou_threshold=self._iou_threshold,
                top_k=top_k,
                candidate_size=self._candidate_size
            )
            picked_boxes_and_scores.append(boxes_and_scores)
            picked_labels.extend([class_id] * boxes_and_scores.size(0))

        if not picked_boxes_and_scores:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])

        picked_boxes_and_scores = torch.cat(picked_boxes_and_scores)
        picked_boxes_and_scores[:, 0] *= width
        picked_boxes_and_scores[:, 1] *= height
        picked_boxes_and_scores[:, 2] *= width
        picked_boxes_and_scores[:, 3] *= height
        return picked_boxes_and_scores[:, :5], torch.tensor(picked_labels), picked_boxes_and_scores[:, 4]
