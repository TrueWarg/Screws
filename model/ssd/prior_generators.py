import torch
from typing import List
import itertools
import math

from dataclasses import dataclass


@dataclass()
class SsdBoxGenParams:
    feature_map_size: int
    shrinkage: int
    average_box_relates_size: float
    rotation_step: int
    aspect_ratios: List


def generate_with_ratios(origin_rbox: List, aspect_ratios: List) -> List:
    rboxes = []
    x_center, y_center, width, height, angle = origin_rbox
    for ratio in aspect_ratios:
        ratio = math.sqrt(ratio)
        rboxes.append([x_center, y_center, width * ratio, height / ratio, 0.0])
        rboxes.append([x_center, y_center, width / ratio, height * ratio, 0.0])

    return rboxes


def generate_with_rotations(origin_box: List, rotation_step: int) -> List:
    x_center, y_center, width, height = origin_box
    angle = 0
    rboxes = []
    while angle <= 360:
        rboxes.append([x_center, y_center, width, height, (angle * math.pi) / 180])
        angle += rotation_step

    return rboxes


def generate_rotated_prior_boxes(input_size: int, params: SsdBoxGenParams, clamp=True) -> torch.Tensor:
    priors = []
    scale = input_size / params.shrinkage
    for i, j in itertools.product(range(params.feature_map_size), repeat=2):
        x_center = (i + 0.5) / scale
        y_center = (j + 0.5) / scale

        width = height = params.average_box_relates_size
        base_prior = [x_center, y_center, width, height, 0.0]
        priors.append(base_prior)
        priors.extend(generate_with_ratios(base_prior, params.aspect_ratios))

    priors = torch.tensor(priors)
    if clamp:
        return torch.clamp(priors, 0.0, 1.0)
    return priors
