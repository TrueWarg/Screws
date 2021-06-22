import torch
from torch import nn


class Predictor:
    def __init__(self,
                 net: nn.Module,
                 transform,
                 nms_method=None,
                 iou_threshold=0.45,
                 filter_threshold=0.01,
                 candidate_size=200,
                 sigma=0.5,
                 device=None
    ):
        self.net = net
        self.transform = transform
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.nms_method = nms_method
        self.sigma = sigma
        self.device = device
        # make eval and to device extra?
        self.net.to(self.device)
        self.net.eval()

    def predict(self, image):
        height, width, _ = image.shape
        image = self.transform(image)
        images = image.unsqueeze(0)
        images = images.to(self.device)
        with torch.no_grad():
            scores, boxes = self.net.forward(images)

        scores = scores[0]
        boxes = boxes[0]

        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > self.filter_threshold

            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            # box_probs = todo apply nms method
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
