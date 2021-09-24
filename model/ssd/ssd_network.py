from collections import namedtuple
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from bbox import converters
from model.ssd.initializers import xavier_init

GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])


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
                 priors=None,
                 is_test=None,
                 ):
        super(SSDNetwork, self).__init__()

        self.num_classes = num_classes
        self.base_net = base_net
        self.extras = feature_extractors
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.source_layer_indexes = source_layer_indexes
        self.device = device
        self.config = config
        self.is_test = is_test
        self.priors = priors.to(device)

        # register layers in source_layer_indexes by adding them to a module list
        self.source_layer_add_ons = nn.ModuleList([t[1] for t in source_layer_indexes
                                                   if isinstance(t, tuple) and not isinstance(t, GraphPath)])

    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        self.feature_extractors.apply(xavier_init)
        self.source_layer_add_ons.apply(xavier_init)
        self.classification_headers.apply(xavier_init)
        self.regression_headers.apply(xavier_init)

    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if
                      not (k.startswith("classification_headers") or k.startswith("regression_headers"))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.classification_headers.apply(xavier_init)
        self.regression_headers.apply(xavier_init)

    def init_default(self):
        self.base_net.apply(xavier_init)
        self.feature_extractors.apply(xavier_init)
        self.source_layer_add_ons.apply(xavier_init)
        self.classification_headers.apply(xavier_init)
        self.regression_headers.apply(xavier_init)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        for end_layer_index in self.source_layer_indexes:
            if isinstance(end_layer_index, GraphPath):
                path = end_layer_index
                end_layer_index = end_layer_index.s0
                added_layer = None
            elif isinstance(end_layer_index, tuple):
                added_layer = end_layer_index[1]
                end_layer_index = end_layer_index[0]
                path = None
            else:
                added_layer = None
                path = None
            for layer in self.base_net[start_layer_index: end_layer_index]:
                x = layer(x)
            if added_layer:
                y = added_layer(x)
            else:
                y = x
            if path:
                sub = getattr(self.base_net[end_layer_index], path.name)
                for layer in sub[:path.s1]:
                    x = layer(x)
                y = x
                for layer in sub[path.s1:]:
                    x = layer(x)
                end_layer_index += 1
            start_layer_index = end_layer_index
            confidence, location = self._compute_header(header_index, y)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        for layer in self.base_net[end_layer_index:]:
            x = layer(x)

        for layer in self.extras:
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
                locations, self.priors, self.config._center_variance, self.config._size_variance
            )
            boxes = converters.center_form_to_corner_form(boxes)
            return confidences, boxes
        else:
            return confidences, locations

    def _compute_header(self, index, x):
        confidence = self.classification_headers[index](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[index](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 5)

        return confidence, location

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)
