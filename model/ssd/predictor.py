import torch
from torch import nn

from bbox.nms import hard_nms


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
        cpu_device = torch.device("cpu")
        height, width, _ = image.shape
        image = self._transform(image)
        images = image.unsqueeze(0)
        images = images.to(self._device)
        with torch.no_grad():
            scores, boxes = self._net.forward(images)
        boxes = boxes[0]
        scores = scores[0]
        boxes = boxes.to(cpu_device)
        scores = scores.to(cpu_device)
        picked_box_probs = []
        picked_labels = []

        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > self._filter_threshold
            print(f" probs {probs.max()}")
            probs = probs[mask]

            if probs.size(0) == 0:
                continue

            subset_boxes = boxes[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            box_probs = hard_nms(
                box_and_scores=box_probs,
                iou_threshold=self._iou_threshold,
                top_k=top_k,
                candidate_size=self._candidate_size
            )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))

        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])

        picked_box_probs = torch.cat(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :5], torch.tensor(picked_labels), picked_box_probs[:, 4]
