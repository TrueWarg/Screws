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
    x_center, y_center, width, height, vanilla_angle = origin_box
    angle = -math.pi
    rboxes = []
    rotation_step = (rotation_step / 360) * 2 * math.pi
    while angle <= math.pi:
        rboxes.append([x_center, y_center, width, height, angle])
        angle += rotation_step

    return rboxes


def generate_rotated_prior_boxes(input_size: int, params: SsdBoxGenParams, clamp=True):
    priors = []
    scale = input_size / params.shrinkage
    for i, j in itertools.product(range(params.feature_map_size), repeat=2):
        x_center = (i + 0.5) / scale
        y_center = (j + 0.5) / scale

        width = height = params.average_box_relates_size
        ratio = params.aspect_ratios[0]
        ratio = math.sqrt(ratio)
        base_prior = [x_center, y_center, width * ratio, height / ratio, 0.0]
        priors.append(base_prior)
        priors.extend(generate_with_rotations(base_prior, params.rotation_step))
        # priors.extend(generate_with_ratios(base_prior, params.aspect_ratios))

        # base_prior = [x_center, y_center, width / ratio, height * ratio, 0.0]
        # priors.append(base_prior)
        # # priors.extend(generate_with_ratios(base_prior, params.aspect_ratios))
        #
        # ratio = math.sqrt(2)
        # base_prior = [x_center, y_center, width * ratio, height / ratio, 0.0]
        # priors.append(base_prior)
        #
        # ratio = math.sqrt(4)
        # base_prior = [x_center, y_center, width / ratio, height * ratio, 0.0]
        # priors.append(base_prior)
        #
        # ratio = math.sqrt(2)
        # base_prior = [x_center, y_center, width * ratio, height / ratio, 0.0]
        # priors.append(base_prior)

        # ratio = math.sqrt(2)
        # base_prior = [x_center, y_center, width / ratio, height * ratio, 0.0]
        # priors.append(base_prior)

    return priors


def generate_ssd_priors(specs: List[SsdBoxGenParams], image_size, clamp=True) -> torch.Tensor:
    priors = []
    for spec in specs:
        priors.extend(generate_rotated_prior_boxes(image_size, spec))

    priors = torch.tensor(priors)
    # priors = priors[priors.shape[0] - 3000:priors.shape[0]]
    if clamp:
        low = torch.tensor([[0.0, 0.0, 0.0, 0.0, -math.pi]])
        high = torch.tensor([[1.0, 1.0, 1.0, 1.0, math.pi]])
        return torch.max(torch.min(priors, high), low)
    return priors