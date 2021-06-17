import math

import torch
from torch import nn
from typing import Tuple
import torch.nn.functional as F


def calculate_hard_negative_mining_mask(log_softmax_loss,
                                        categories: torch.Tensor,
                                        negative_negative_ratio: float
                                        ) -> torch.Tensor:
    positive_mask = categories > 0
    positive_count = positive_mask.sum(dim=1, keepdim=True)
    negative_count = positive_count * negative_negative_ratio
    log_softmax_loss[positive_mask] = -math.inf
    _, indexes = log_softmax_loss.sort(dim=1, descending=True)


class RotatedMultiBoxLoss(nn.Module):
    def __init__(self, num_classes: int,
                 overlap_threshold: float,
                 positive_negative_ratio: float,
                 ):
        super().__init__()
        self._num_classes = num_classes
        self._overlap_threshold = overlap_threshold
        self._positive_negative_ratio = positive_negative_ratio

    def forward(self,
                predicted_boxes: torch.Tensor,
                confidences: torch.Tensor,
                target_boxes: torch.Tensor,
                target_categories: torch.Tensor):
        with torch.no_grad():
            loss = -F.log_softmax(confidences, dim=2)[:, :, 0]
            mask = calculate_hard_negative_mining_mask(loss, target_categories, self.neg_pos_ratio)
