import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Tuple, List

import cv2
import numpy as np
from torch.utils.data import Dataset

BACKGROUND_CLASS_LABEL = 'BACKGROUND'
BACKGROUND_CLASS_ID = 0


@dataclass(frozen=True)
class Config:
    root_path: str
    annotations_relative_path: str
    annotation_extension: str
    images_relative_path: str
    images_extension: str
    image_ids: List
    class_labels: Tuple
    skip_difficult: bool


class VOCDataset(Dataset):
    def __init__(self, config: Config, transform=None, target_transform=None):
        self._config = config
        self._root_path = config.root_path
        self._transform = transform
        self._target_transform = target_transform
        self._classes = {class_label: class_id for class_id, class_label in enumerate(config.class_labels)}

    def __getitem__(self, index: int):
        image_id = self._config.image_ids[index]
        boxes, class_ids, is_difficult = self._extract_annotations(image_id)
        if not self._config.skip_difficult:
            boxes = boxes[is_difficult == 0]
            class_ids = class_ids[is_difficult == 0]
        image = self._read_image(image_id)
        if self._transform:
            image, boxes, class_ids = self._transform(image, boxes, class_ids)
        if self._target_transform:
            boxes, class_ids = self._target_transform(boxes, class_ids)
        return image, boxes, class_ids

    def _extract_annotations(self, image_id: str):
        annotation_path = os.path.join(
            self._root_path,
            self._config.annotations_relative_path,
            f'{image_id}.{self._config.annotation_extension}',
        )
        elements = ET.parse(annotation_path).findall("object")
        boxes = []
        class_ids = []
        is_difficult = []

        for element in elements:
            class_name = element.find('name').text.lower().strip()
            bbox = self._extract_bbox(element)
            boxes.append(bbox)
            class_ids.append(self._classes[class_name])
            is_difficult_str = element.find('difficult').text
            is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (
            np.array(boxes, dtype=np.float32),
            np.array(class_ids, dtype=np.int64),
            np.array(is_difficult, dtype=np.uint8),
        )

    def _extract_bbox(self, element) -> List:
        bbox = element.find('bndbox')
        x_min = float(bbox.find('xmin').text)
        y_min = float(bbox.find('ymin').text)
        x_max = float(bbox.find('xmax').text)
        y_max = float(bbox.find('ymax').text)
        angle = float(bbox.find('angle').text)
        return [x_min, y_min, x_max, y_max, angle]

    def _read_image(self, image_id: str) -> np.ndarray:
        image_path = os.path.join(
            self._root_path,
            self._config.images_relative_path,
            f'{image_id}.{self._config.images_extension}',
        )
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_image(self, index: int):
        image_id = self._config.image_ids[index]
        image = self._read_image(image_id)
        if self._transform:
            image, _ = self._transform(image)
        return image

    def get_annotation(self, index):
        image_id = self._config.image_ids[index]
        return image_id, self._extract_annotations(image_id)

    def __len__(self):
        return len(self._config.image_ids)
