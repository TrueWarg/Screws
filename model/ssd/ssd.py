from collections import namedtuple
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from bbox import converters

GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])


class SSD(nn.Module):
    def __init__(self,
                 num_classes: int,
                 base_net: nn.Module,
                 base_net_source_layer_indices: List[int],
                 feature_extractors: nn.ModuleList,
                 classification_headers: nn.ModuleList,
                 regression_headers: nn.ModuleList,
                 ):
        super(SSD, self).__init__()

        assert len(base_net_source_layer_indices) \
               + len(feature_extractors) \
               == len(classification_headers) == len(regression_headers), "Wrong ssd config: source layer from " \
                                                                          "base net + feature_extractors, " \
                                                                          "classification_headers and " \
                                                                          "regression_headers" \
                                                                          "must be equal "

        self._num_classes = num_classes
        self.base_net = base_net
        self._source_layer_indices = base_net_source_layer_indices
        self.feature_extractors = feature_extractors
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        end_layer_index = 0
        for end_layer_index in self._source_layer_indices:
            for layer in self.base_net[start_layer_index: end_layer_index]:
                x = layer(x)

            start_layer_index = end_layer_index
            confidence, location = self._compute_header(header_index, x)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        for layer in self.base_net[end_layer_index:]:
            x = layer(x)

        for layer in self.feature_extractors:
            x = layer(x)
            confidence, location = self._compute_header(header_index, x)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)

        return confidences, locations

    def _compute_header(self, index, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidence = self.classification_headers[index](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self._num_classes)

        location = self.regression_headers[index](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 5)

        return confidence, location

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


class SSDTest(nn.Module):
    def __init__(self,
                 ssd: SSD,
                 config=None,
                 priors=None,
                 device=None,
                 ):
        super(SSDTest, self).__init__()
        self._ssd = ssd
        self._config = config
        self._priors = priors

        if device:
            self._ssd.to(device)
            self._priors.to(device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidences, locations = self._ssd(x)
        confidences = F.softmax(confidences, dim=2)
        boxes = converters.locations_to_boxes(
            locations, self._priors, self._config.center_variance, self._config.size_variance
        )
        boxes = converters.center_form_to_corner_form(boxes)
        return confidences, boxes
