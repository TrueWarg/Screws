import torch


def xcycwh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    return torch.cat((
        boxes[..., :2] - boxes[..., 2:] / 2,
        boxes[..., :2] + boxes[..., 2:] / 2
    ), boxes.dim() - 1)


def xyxy_to_xcycwh(boxes: torch.Tensor) -> torch.Tensor:
    return torch.cat([
        (boxes[..., :2] + boxes[..., 2:4]) / 2,
        boxes[..., 2:4] - boxes[..., :2],
        boxes[..., 4:],
    ], boxes.dim() - 1)


def xcycwha_to_xyxya(boxes: torch.Tensor) -> torch.Tensor:
    return torch.cat([boxes[..., :2] - boxes[..., 2:4] / 2,
                      boxes[..., :2] + boxes[..., 2:4] / 2,
                      boxes[..., 4:],
                      ], boxes.dim() - 1)


def locations_to_boxes(locations, priors, center_variance, size_variance):
    # priors can have one dimension less.
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)

    locations[..., 4][locations[..., 4] > 1.0] = 1.00
    locations[..., 4][locations[..., 4] < -1.0] = -1.00

    return torch.cat([
        locations[..., :2] * center_variance * priors[..., 2:4] + priors[..., :2],
        torch.exp(locations[..., 2:4] * size_variance) * priors[..., 2:4],
        torch.arctan(locations[..., 4:]) + priors[..., 4:],
    ], dim=locations.dim() - 1)


def convert_boxes_to_locations(center_form_boxes, center_form_priors, center_variance, size_variance):
    # priors can have one dimension less
    if center_form_priors.dim() + 1 == center_form_boxes.dim():
        center_form_priors = center_form_priors.unsqueeze(0)
    return torch.cat([
        (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:4] / center_variance,
        torch.log(center_form_boxes[..., 2:4] / center_form_priors[..., 2:4]) / size_variance,
        torch.tan(center_form_boxes[..., 4:] - center_form_priors[..., 4:]),
    ], dim=center_form_boxes.dim() - 1)


def center_form_to_corner_form(locations):
    return torch.cat([locations[..., :2] - locations[..., 2:4] / 2,
                      locations[..., :2] + locations[..., 2:4] / 2,
                      locations[..., 4:],
                      ], locations.dim() - 1)


def corner_form_to_center_form(boxes):
    return torch.cat([
        (boxes[..., :2] + boxes[..., 2:4]) / 2,
        boxes[..., 2:4] - boxes[..., :2],
        boxes[..., 4:],
    ], boxes.dim() - 1)
