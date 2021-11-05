from typing import List, Tuple

import torch
from torch import nn


class PredictHead(nn.Module):
    def __init__(self, plane_square: int, num_priors: int, num_values: int):
        super(PredictHead, self).__init__()
        self._num_classes = num_values
        self.module = nn.Sequential(
            nn.Conv2d(in_channels=plane_square, out_channels=3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_priors * num_values, out_channels=3, kernel_size=1, stride=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.module(x)
        return x.permute(0, 2, 3, 1).reshape(x.size(0), -1, self.num_classes)


class Detector(nn.Module):
    def __init__(self, depth: int, plane_square: int, priors_counts: List[int], num_classes: int):
        super(Detector, self).__init__()
        self.classification_heads = nn.ModuleList()
        self.location_heads = nn.ModuleList()
        for i in range(depth):
            self.classification_heads.append(PredictHead(plane_square, priors_counts[i], num_classes))
            # len([x, y, w, h, angle]) - bbox values
            self.location_heads.append(PredictHead(plane_square, priors_counts[i], num_values=5))

    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        predicted_classes, predicted_locs = [], []
        for i, feature in enumerate(features):
            predicted_classes.append(self.classification_heads[i](feature))
            predicted_locs.append(self.location_heads[i](feature))
        predicted_classes = torch.cat(predicted_classes, dim=1)
        predicted_locs = torch.cat(predicted_locs, dim=1)
        return predicted_classes, predicted_locs


class FeaturePyramid(nn.Module):
    def __init__(self, depth: int, plane_square: int):
        super(FeaturePyramid, self).__init__()
        self.links = nn.ModuleList()
        self.fusions = nn.ModuleList()
        for i in range(depth):
            self.links.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=plane_square, out_channels=1, kernel_size=1, stride=0, bias=False),
                    nn.BatchNorm2d(num_features=1),
                ))

            self.fusions.append(nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=plane_square, out_channels=3, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(num_features=3),
            ))

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        features = [self.links[i](feature) for i, feature in enumerate(features)]
        # inverse loop
        for i in range(len(features))[::-1]:
            if i != len(features) - 1:
                features[i] = self.fusions[i](
                    features[i] + nn.Upsample(scale_factor=2)(features[i + 1])
                )
        features = [nn.ReLU(inplace=True)(feature) for feature in features]
        return features
