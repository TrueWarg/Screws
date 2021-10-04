import torch


def locations_to_boxes(
        locations: torch.Tensor,
        center_form_priors: torch.Tensor,
        center_variance: float,
        size_variance: float,
):
    # priors can have one dimension less.
    if center_form_priors.dim() + 1 == locations.dim():
        center_form_priors = center_form_priors.unsqueeze(0)

    locations[..., 4][locations[..., 4] > 1.0] = 1.00
    locations[..., 4][locations[..., 4] < -1.0] = -1.00

    return torch.cat([
        locations[..., :2] * center_variance * center_form_priors[..., 2:4] + center_form_priors[..., :2],
        torch.exp(locations[..., 2:4] * size_variance) * center_form_priors[..., 2:4],
        torch.arctan(locations[..., 4:]) + center_form_priors[..., 4:],
    ], dim=locations.dim() - 1)


# the author of the original paper comments about the variances.
# Probably, the naming comes from the idea, that the ground truth bounding boxes are
# not always precise, in other words, they vary from image to image probably
# for the same object in the same position just because human labellers cannot
# ideally repeat themselves. Thus, the encoded values are some random values,
# and we want them to have unit variance that is why we divide by some value.
def boxes_to_locations(
        center_form_boxes: torch.Tensor,
        center_form_priors: torch.Tensor,
        center_variance: float,
        size_variance: float,
) -> torch.Tensor:
    # priors can have one dimension less
    if center_form_priors.dim() < center_form_boxes.dim():
        center_form_priors = center_form_priors.unsqueeze(0)

    return torch.cat([
        (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:4] / center_variance,
        torch.log(center_form_boxes[..., 2:4] / center_form_priors[..., 2:4]) / size_variance,
        torch.tan(center_form_boxes[..., 4:] - center_form_priors[..., 4:]),
    ], dim=center_form_boxes.dim() - 1)


def center_form_to_corner_form(boxes: torch.Tensor) -> torch.Tensor:
    return torch.cat([
        boxes[..., :2] - boxes[..., 2:4] / 2,
        boxes[..., :2] + boxes[..., 2:4] / 2,
        boxes[..., 4:],
    ], boxes.dim() - 1)


def corner_form_to_center_form(boxes: torch.Tensor) -> torch.Tensor:
    return torch.cat([
        (boxes[..., :2] + boxes[..., 2:4]) / 2,
        boxes[..., 2:4] - boxes[..., :2],
        boxes[..., 4:],
    ], boxes.dim() - 1)
