from collections import OrderedDict
from dataclasses import dataclass
from typing import List

import torch


@dataclass
class GenConfig:
    size: int
    aspects: List[int]
    scales: List[float]


def _generate_widths_and_heights(stride: int, config: GenConfig) -> List:
    widths_and_heights = []
    pairs = [[aspect, scale] for aspect in config.aspects for scale in config.scales]
    for aspect, scale in pairs:
        length = stride * config.size * scale
        if aspect == 1:
            widths_and_heights.append([length, length])
        else:
            widths_and_heights.append([length * aspect ** 0.5, length / aspect ** 0.5])
            widths_and_heights.append([length / aspect ** 0.5, length * aspect ** 0.5])
    return widths_and_heights


def generate_priors(image_size: int, stride_to_configs: OrderedDict) -> torch.Tensor:
    """
    Generate prior bounded boxes according provides configs
    :param image_size: image size
    :param stride_to_configs: stride to GenConfig dict
    :return: tensor of bbox = [x, y, w, h]
    """
    priors = []
    for stride, config in stride_to_configs.items():
        scaled_w, scaled_h = image_size // stride, image_size // stride
        widths_and_heights = torch.tensor(_generate_widths_and_heights(stride, config), dtype=torch.float)
        x, y = torch.meshgrid([torch.arange(scaled_w), torch.arange(scaled_h)])
        center_x = x + 0.5
        center_y = y + 0.5
        centers = torch.stack([center_x, center_y], dim=-1) * stride
        centers = centers[:, :, None, :].repeat(1, 1, widths_and_heights.size(0), 1)
        widths_and_heights = widths_and_heights[None, None, :, :].repeat(scaled_h, scaled_w, 1, 1)
        priors.append(torch.cat([centers, widths_and_heights], dim=-1).reshape(-1, 4))

    return torch.cat(priors)


class PriorBboxesProvider:
    def __init__(self, stride_to_configs: OrderedDict):
        self._stride_to_configs = stride_to_configs
        self._cache = {}

    def get_or_generate(self, image_size: int) -> torch.Tensor:
        cached = self._cache[image_size]
        if cached is not None:
            return cached
        priors = generate_priors(image_size, self._stride_to_configs)
        self._cache[image_size] = priors
        return priors
