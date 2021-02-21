import torch
from torch import nn
from typing import Tuple
import torch.nn.functional as F


def match(overlap_threshold: float,
          target_locs: torch.Tensor,
          target_classes: torch.Tensor,
          priors: torch.Tensor,
          ) -> Tuple:
    # todo implement matching method (move in some other py module)
    pass


class RotatedMultiBoxLoss(nn.Module):
    def __init__(self, num_classes: int, overlap_threshold: float):
        super().__init__()
        self.nun_classes = num_classes
        self._overlap_threshold = overlap_threshold

    def forward(self, prediction: Tuple, targets: torch.Tensor):
        localisations, confidences, priors = prediction
        batch_size = localisations.size(0)
        num_priors = priors.size(0)

        matched_locs = torch.Tensor(localisations.size)
        matched_confs = torch.LongTensor(batch_size, num_priors)

        matched_locs.requires_grad = False
        matched_confs.requires_grad = False

        for batch_idx in range(batch_size):
            target_locs = targets[batch_idx][:, :-1]
            target_classes = targets[batch_idx][:, -1]

            matched_loc, matched_conf = match(self._overlap_threshold, target_locs, target_classes, priors)
            matched_locs[batch_idx] = matched_loc
            matched_confs[batch_idx] = matched_conf

        # todo complete full implementation
        # calculate location loss
        # calculate class loss

