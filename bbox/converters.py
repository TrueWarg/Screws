import numpy as np
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


def vertex_form_to_center_form(boxes: np.ndarray) -> np.ndarray:
    """
    Convert vertex form box in center form box with angle 0 <= a < 180

    :param boxes: [2DPoint, 2DPoint, 2DPoint, 2DPoint], where 2DPoint = [x, y]

    :return: [x_center, y_center, width, height, angle]
    """
    x_center, y_center = np.mean(boxes, axis=0)
    # sqrt( (x_2 - x_1)^2 + (y_2 - y_1)^2 )
    diff01 = boxes[0] - boxes[1]
    diff03 = boxes[0] - boxes[3]
    w = np.sqrt(np.square(diff01).sum())
    h = np.sqrt(np.square(diff03).sum())
    # y = k*x + b, k = tg(a)
    a = np.rad2deg(np.arctan2(diff01[1], diff01[0]))

    # if a < 0 -> 180 - abs(a), if a == 180 or 360 -> 0
    a = a % 180

    return np.stack([x_center, y_center, w, h, a])


def vertex_form_to_center_form_angle_bounded45(boxes: np.ndarray) -> np.ndarray:
    """
    Convert vertex form box in center form box with angle -45 <= a < 45

    :param boxes: [2DPoint, 2DPoint, 2DPoint, 2DPoint], where 2DPoint = [x, y]

    :return: [x_center, y_center, width, height, angle]
    """
    x_center, y_center = np.mean(boxes, axis=0)
    # sqrt( (x_2 - x_1)^2 + (y_2 - y_1)^2 )
    diff01 = boxes[0] - boxes[1]
    diff03 = boxes[0] - boxes[3]
    w = np.sqrt(np.square(diff01).sum())
    h = np.sqrt(np.square(diff03).sum())
    # y = k*x + b, k = tg(a)
    a = np.rad2deg(np.arctan2(diff01[1], diff01[0]))

    # if a < 0 -> 180 - abs(a), if a == 180 or 360 -> 0
    a = a % 180

    if 45 <= a < 135:
        w, h = h, w
        a -= 90
    elif a >= 135:
        a -= 180

    return np.stack([x_center, y_center, w, h, a])
