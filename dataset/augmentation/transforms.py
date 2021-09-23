from enum import Enum

from numpy import random
from typing import List

import cv2
import numpy as np
import torch


class TrainTransform:
    def __init__(self, image_size: int, mean=0.0, std=1.0):
        """
        Args:
            image_size: the size the of final image.
            mean: mean pixel value per channel.
            std: std pixel value per channel
        """

        self._transforms = Compose([
            ConvertToFloat32(),
            PhotometricDistort(),
            ToPercentCoordinates(),
            Resize(image_size),
            SubtractMeans(mean),
            lambda image, boxes=None, labels=None: (image / std, boxes, labels),
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        return self._transforms(image, boxes, labels)


class TestTransform:
    def __init__(self, image_size: int, mean=0.0, std=1.0):
        """
        Args:
            image_size: the size the of final image.
            mean: mean pixel value per channel.
            std: std pixel value per channel
        """
        self._transforms = Compose([
            ToPercentCoordinates(),
            Resize(image_size),
            SubtractMeans(mean),
            lambda image, boxes=None, labels=None: (image / std, boxes, labels),
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        return self._transforms(image, boxes, labels)


class PredictionTransform:

    def __init__(self, image_size: int, mean=0.0, std=1.0):
        """
        Args:
            image_size: the size the of final image.
            mean: mean pixel value per channel.
            std: std pixel value per channel
        """
        self.transform = Compose([
            Resize(image_size),
            SubtractMeans(mean),
            lambda image, boxes=None, labels=None: (image / std, boxes, labels),
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image


class ToTensor:
    def __call__(self, image: np.ndarray, boxes=None, labels=None):
        return torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1), boxes, labels


class Resize:
    def __init__(self, image_size: int):
        self._image_size = image_size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self._image_size, self._image_size))
        return image, boxes, labels


class SubtractMeans:
    def __init__(self, mean: float):
        self._mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self._mean
        return image.astype(np.float32), boxes, labels


class ToPercentCoordinates:
    def __call__(self, image: np.ndarray, boxes=None, labels=None):
        height, width, _ = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class ConvertToFloat32:
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class Compose:
    def __init__(self, transforms: List):
        self._transforms = transforms

    def __call__(self, image, boxes=None, labels=None):
        for t in self._transforms:
            image, boxes, labels = t(image, boxes, labels)
        return image, boxes, labels


class PhotometricDistort:
    def __init__(self):
        self._color_model_transform = [
            RandomContrast(),  # RGB
            ConvertColor(from_model=ColorModel.RGB, to_model=ColorModel.HSV),  # HSV
            RandomSaturation(),  # HSV
            RandomHue(),  # HSV
            ConvertColor(from_model=ColorModel.HSV, to_model=ColorModel.RGB),  # RGB
            RandomContrast()  # RGB
        ]
        self._rand_brightness = RandomBrightness()
        self._rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        img = image.copy()
        img, boxes, labels = self._rand_brightness(img, boxes, labels)
        if random.randint(2):
            distort = Compose(self._color_model_transform[:-1])
        else:
            distort = Compose(self._color_model_transform[1:])
        img, boxes, labels = distort(img, boxes, labels)
        return self._rand_light_noise(img, boxes, labels)


class RandomSaturation:
    def __init__(self, lower=0.5, upper=1.5):
        assert 0 <= lower <= upper, "lower must be in range 0.0 - upper"

        self._lower = lower
        self._upper = upper

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self._lower, self._upper)

        return image, boxes, labels


class RandomHue:
    def __init__(self, delta=18.0):
        assert 0.0 <= delta <= 360.0, "delta must be range 0.0 - 360.0"
        self._delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self._delta, self._delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise:
    def __init__(self):
        self._perms = ((0, 1, 2), (0, 2, 1),
                       (1, 0, 2), (1, 2, 0),
                       (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self._perms[random.randint(len(self._perms))]
            shuffle = SwapChannels(swap)
            image = shuffle(image)
        return image, boxes, labels


class ColorModel(Enum):
    RGB = 1
    BGR = 2
    HSV = 3


class ConvertColor:
    def __init__(self, from_model: ColorModel, to_model: ColorModel):
        self._from_model = from_model
        self._to_model = to_model

    def __call__(self, image, boxes=None, labels=None):
        if self._from_model == ColorModel.BGR and self._to_model == ColorModel.HSV:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self._from_model == ColorModel.RGB and self._to_model == ColorModel.HSV:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self._from_model == ColorModel.BGR and self._to_model == ColorModel.RGB:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self._from_model == ColorModel.HSV and self._to_model == ColorModel.BGR:
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self._from_model == ColorModel.HSV and self._to_model == ColorModel.RGB:
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast:
    def __init__(self, lower=0.5, upper=1.5):
        assert 0 <= lower <= upper, "lower must be in range 0.0 - upper"

        self._lower = lower
        self._upper = upper

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self._lower, self._upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness:
    def __init__(self, delta=32):
        assert 0.0 <= delta <= 255.0, "delta must be in range 0.0 - 255.0"
        self._delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self._delta, self._delta)
            image += delta
        return image, boxes, labels


class SwapChannels:
    def __init__(self, swaps):
        self._swaps = swaps

    def __call__(self, image):
        image = image[:, :, self._swaps]
        return image
