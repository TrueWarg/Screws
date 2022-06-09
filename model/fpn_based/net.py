from dataclasses import dataclass
from typing import Tuple, List

import torch
from torch import nn

from model.fpn_based.modules import Detector, FeaturePyramid


class RbboxDetectorNet(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 detector: Detector,
                 feature_pyramid: FeaturePyramid,
                 feature_extractors: nn.ModuleList(),
                 ):
        super().__init__()
        self._backbone = backbone
        self._detector = detector
        self._feature_pyramid = feature_pyramid
        self._feature_extractors = feature_extractors
        self._num_extractors = 0
        if feature_extractors:
            self._num_extractors = len(feature_extractors)

    def init(self):
        pass

    def restore(self):
        pass

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = list(self.backbone(images))
        # drop low level features
        features = features[self._num_extractors + 1:]
        if self._feature_extractors:
            for extractor in self._feature_extractors:
                features.append(extractor(features[-1]))
        features = self._feature_pyramid(features)
        predicted_class_scorers, predicted_locs = self._detector(features)
        return predicted_class_scorers, predicted_locs


@dataclass(frozen=True)
class FeatureExtractorConfig:
    count: int
    plane_square: int


def build_rbbox_detector_net(
        backbone: nn.Module,
        num_classes: int,
        fpn_plane_square: int,
        fpn_num_levels: int,
        priors_counts: List[int],
        extractors_config: FeatureExtractorConfig,
) -> RbboxDetectorNet:
    fpn = FeaturePyramid(depth=fpn_num_levels, plane_square=fpn_plane_square)
    detector = Detector(depth=fpn_num_levels, plane_square=fpn_plane_square,
                        priors_counts=priors_counts, num_classes=num_classes)
    extractors = None
    if extractors_config:
        extractors = nn.ModuleList()
        for _ in range(extractors_config.count):
            extractors.append(nn.Sequential(nn.Conv2d(
                in_channels=extractors_config.plane_square,
                out_channels=3,
                kernel_size=2,
                stride=1,
                bias=False,
            ),
                nn.BatchNorm2d(num_features=3),
                nn.ReLU(inplace=True)))
    return RbboxDetectorNet(
        backbone=backbone,
        detector=detector,
        feature_pyramid=fpn,
        feature_extractors=extractors,
    )
