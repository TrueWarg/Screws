from enum import Enum

from numpy import random
from typing import List, Tuple

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
            PhotometricDistortion(),
            ToPercentCoordinates(),
            ResizeImageOnly(image_size),
            SubtractMeans(mean),
            lambda image, boxes=None, class_ids=None: (image / std, boxes, class_ids),
            ToTensor(),
        ])

    def __call__(self, image, boxes, class_ids):
        return self._transforms(image, boxes, class_ids)


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
            ResizeImageOnly(image_size),
            SubtractMeans(mean),
            lambda image, boxes=None, class_ids=None: (image / std, boxes, class_ids),
            ToTensor(),
        ])

    def __call__(self, image, boxes, class_ids):
        return self._transforms(image, boxes, class_ids)


class PredictionTransform:

    def __init__(self, image_size: int, mean=0.0, std=1.0):
        """
        Args:
            image_size: the size the of final image.
            mean: mean pixel value per channel.
            std: std pixel value per channel
        """
        self.transform = Compose([
            ResizeImageOnly(image_size),
            SubtractMeans(mean),
            lambda image, boxes=None, class_ids=None: (image / std, boxes, class_ids),
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image


class ToTensor:
    def __call__(self, image: np.ndarray, boxes=None, class_ids=None):
        return torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1), boxes, class_ids


class ResizeImageOnly:
    def __init__(self, image_size: int):
        self._image_size = image_size

    def __call__(self, image, boxes=None, class_ids=None):
        image = cv2.resize(image, (self._image_size, self._image_size))
        return image, boxes, class_ids


class Resize:
    def __init__(self, size: Tuple):
        self._size = size

    def __call__(self, image, boxes=None, class_ids=None):
        rw, rh = self._size
        if boxes:
            ih, iw = image.shape[:2]
            bboxes = boxes * [rw / iw, rh / ih]
            boxes = np.array([cv2.boxPoints(cv2.minAreaRect(bbox)) for bbox in bboxes.astype(np.float32)])
        image = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
        return image, boxes, class_ids


class ResizeJitter:
    def __init__(self, scale=(0.8, 1.2)):
        self._scale = scale

    def __call__(self, image, boxes=None, class_ids=None):
        ih, iw = image.shape[:2]
        rh, rw = [ih, iw] * np.random.uniform(*self._scale, 2)
        return Resize((int(rw), int(rw)))(image, boxes, class_ids)


class SubtractMeans:
    def __init__(self, mean: float):
        self._mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, class_ids=None):
        image = image.astype(np.float32)
        image -= self._mean
        return image.astype(np.float32), boxes, class_ids


class Normalize:
    def __init__(self, mean: float, std: float):
        self._mean = mean
        self._std = std

    def __call__(self, image, boxes=None, class_ids=None):
        image = (image - self._mean) / self._std
        return image, boxes, class_ids


class RandomHorizontalFlip:
    def __call__(self, image, boxes=None, class_ids=None):
        if np.random.randint(2):
            if boxes:
                ih, iw = image.shape[:2]
                boxes[:, :, 0] = iw - 1 - boxes[:, :, 0]
            image = np.ascontiguousarray(np.fliplr(image))
        return image, boxes, class_ids


class RandomVerticalFlip:
    def __call__(self, image, boxes=None, class_ids=None):
        if np.random.randint(2):
            if boxes:
                ih, iw = image.shape[:2]
                boxes[:, :, 1] = ih - 1 - boxes[:, :, 1]
            image = np.ascontiguousarray(np.flipud(image))
        return image, boxes, class_ids


class RandomRotate90:
    def __call__(self, image, boxes=None, class_ids=None):
        clockwise = np.random.choice((0, 1, 2, 3))
        ih, iw = image.shape[:2]
        if boxes:
            if clockwise == 1:
                boxes[:, :, 1] = ih - 1 - boxes[:, :, 1]
                boxes = boxes[:, :, [1, 0]]
            if clockwise == 2:
                boxes = ([iw - 1, ih - 1] - boxes)
            if clockwise == 3:
                boxes[:, :, 0] = iw - 1 - boxes[:, :, 0]
                boxes = boxes[:, :, [1, 0]]
        if clockwise % 4 != 0:
            image = np.ascontiguousarray(np.rot90(image, -clockwise))
        return image, boxes, class_ids


class ToPercentCoordinates:
    def __call__(self, image: np.ndarray, boxes=None, class_ids=None):
        height, width, _ = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, class_ids


class ConvertToFloat32:
    def __call__(self, image, boxes=None, class_ids=None):
        return image.astype(np.float32), boxes, class_ids


class Compose:
    def __init__(self, transforms: List):
        self._transforms = transforms

    def __call__(self, image, boxes=None, class_ids=None):
        for t in self._transforms:
            image, boxes, class_ids = t(image, boxes, class_ids)
        return image, boxes, class_ids


class PhotometricDistortion:
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

    def __call__(self, image, boxes, class_ids):
        img = image.copy()
        img, boxes, class_ids = self._rand_brightness(img, boxes, class_ids)
        if random.randint(2):
            distort = Compose(self._color_model_transform[:-1])
        else:
            distort = Compose(self._color_model_transform[1:])
        img, boxes, class_ids = distort(img, boxes, class_ids)
        return self._rand_light_noise(img, boxes, class_ids)


class RandomSaturation:
    """
        Apply random saturation. HSV color model is expected.
    """

    def __init__(self, lower=0.5, upper=1.5):
        assert 0 <= lower <= upper, "lower must be in range 0.0 - upper"

        self._lower = lower
        self._upper = upper

    def __call__(self, image, boxes=None, class_ids=None):
        if random.randint(2):
            # S in HSV
            image[:, :, 1] *= random.uniform(self._lower, self._upper)

        return image, boxes, class_ids


class RandomHue:
    """
        Apply random hue. HSV color model is expected.
    """

    def __init__(self, delta=18.0):
        assert 0.0 <= delta <= 360.0, "delta must be range 0.0 - 360.0"
        self._delta = delta

    def __call__(self, image, boxes=None, class_ids=None):
        if random.randint(2):
            # H in HSV
            image[:, :, 0] += random.uniform(-self._delta, self._delta)
            # prevent out of 0.0 - 360.0 range
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, class_ids


class RandomLightingNoise:
    def __init__(self):
        self._perms = ((0, 1, 2), (0, 2, 1),
                       (1, 0, 2), (1, 2, 0),
                       (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, class_ids=None):
        if random.randint(2):
            swap = self._perms[random.randint(len(self._perms))]
            shuffle = SwapChannels(swap)
            image = shuffle(image)
        return image, boxes, class_ids


class ColorModel(Enum):
    RGB = 1
    BGR = 2
    HSV = 3


class ConvertColor:
    def __init__(self, from_model: ColorModel, to_model: ColorModel):
        self._from_model = from_model
        self._to_model = to_model

    def __call__(self, image, boxes=None, class_ids=None):
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
        return image, boxes, class_ids


class RandomContrast:
    def __init__(self, lower=0.5, upper=1.5):
        assert 0 <= lower <= upper, "lower must be in range 0.0 - upper"

        self._lower = lower
        self._upper = upper

    def __call__(self, image, boxes=None, class_ids=None):
        if random.randint(2):
            image *= random.uniform(self._lower, self._upper)
        return image, boxes, class_ids


class RandomBrightness:
    def __init__(self, delta=32):
        assert 0.0 <= delta <= 255.0, "delta must be in range 0.0 - 255.0"
        self._delta = delta

    def __call__(self, image, boxes=None, class_ids=None):
        if random.randint(2):
            image += random.uniform(-self._delta, self._delta)
        return image, boxes, class_ids


class SwapChannels:
    def __init__(self, swaps):
        self._swaps = swaps

    def __call__(self, image):
        image = image[:, :, self._swaps]
        return image
