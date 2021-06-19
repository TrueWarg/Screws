from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from bbox import converters
from initializers import xavier_init


class SSDNetwork(nn.Module):
    def __init__(self,
                 base_net: nn.ModuleList,
                 feature_extractors: nn.ModuleList,
                 classification_headers: nn.ModuleList,
                 regression_headers: nn.ModuleList,
                 source_layer_indexes: List[int],
                 num_classes: int,
                 device,
                 config=None,
                 is_test=None,
                 ):
        super(SSDNetwork, self).__init__()

        self._num_classes = num_classes
        self._base_net = base_net
        self._feature_extractors = feature_extractors
        self._classification_headers = classification_headers
        self._regression_headers = regression_headers
        self._source_layer_indexes = source_layer_indexes
        self._num_classes = num_classes
        self._device = device
        self._config = config
        self._is_test = is_test

        if is_test:
            self._priors = config.priors.to(device)

    def init_from_base_net(self, model):
        self._base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        self._feature_extractors.apply(xavier_init)
        self._classification_headers.apply(xavier_init)
        self._regression_headers.apply(xavier_init)

    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if
                      not (k.startswith("classification_headers") or k.startswith("regression_headers"))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self._classification_headers.apply(xavier_init)
        self._regression_headers.apply(xavier_init)

    def init_default(self):
        self._base_net.apply(xavier_init)
        self._feature_extractors.apply(xavier_init)
        self._classification_headers.apply(xavier_init)
        self._regression_headers.apply(xavier_init)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0

        for end_layer_index in self._source_layer_indexes:
            if isinstance(end_layer_index, tuple):
                added_layer = end_layer_index[1]
                end_layer_index = end_layer_index[0]
            else:
                added_layer = None

            for layer in self._base_net[start_layer_index: end_layer_index]:
                x = layer(x)

            if added_layer:
                base_net_output = added_layer(x)
            else:
                base_net_output = x

            start_layer_index = end_layer_index
            confidence, location = self._compute_header(header_index, base_net_output)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

            for layer in self._base_net[end_layer_index:]:
                x = layer(x)

            for layer in self._feature_extractors:
                x = layer(x)
                confidence, location = self._compute_header(header_index, x)
                header_index += 1
                confidences.append(confidence)
                locations.append(location)

            confidences = torch.cat(confidences, 1)
            locations = torch.cat(locations, 1)

            if self.is_test:
                confidences = F.softmax(confidences, dim=2)
                boxes = converters.locations_to_boxes(
                    locations, self.priors, self.config.center_variance, self.config.size_variance
                )
                boxes = converters.xcycwha_to_xyxya(boxes)
                return confidences, boxes
            else:
                return confidences, locations

    def _compute_header(self, index, x):
        confidence = self._classification_headers[index](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self._regression_headers[index](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 5)

        return confidence, location
