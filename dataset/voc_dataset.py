import os
from typing import Tuple, List
from dataclasses import dataclass
import cv2
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import numpy as np


@dataclass()
class Config:
    root_path: str
    images_sets_relative_path: str
    image_ids: List
    class_labels: List
    difficult_only: bool


class VOCDataset(Dataset):
    def __init__(self, config: Config, transform=None, target_transform=None):
        self._config = config
        self._root_path = config.root_path
        self._transform = transform
        self._target_transform = target_transform
        self._images_sets_path = os.path.join(config.root_path, config.images_sets_relative_path)
        self._classes = {class_label: i for i, class_label in enumerate(config.class_labels)}

    def __getitem__(self, index: int):
        image_id = self._config.image_ids[index]
        boxes, labels, is_difficult = self._extract_annotations(image_id)
        if self._config.difficult_only:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)

        if self._transform:
            image, boxes, labels = self._transform(image, boxes, labels)
        if self._target_transform:
            boxes, labels = self._target_transform(boxes, labels)
        return image, boxes, labels

    # todo move Annotation relative path to config?
    def _extract_annotations(self, image_id: str):
        annotation_path = os.path.join(self._root_path, f"Annotations/{image_id}.xml")
        objects = ET.parse(annotation_path).findall("object")
        boxes = []
        labels = []
        is_difficult = []

        for object in objects:
            class_name = object.find('name').text.lower().strip()
            bbox = self._extract_bbox(object)
            boxes.append(bbox)
            labels.append(self.class_dict[class_name])
            is_difficult_str = object.find('difficult').text
            is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (
            np.array(boxes, dtype=np.float32),
            np.array(labels, dtype=np.int64),
            np.array(is_difficult, dtype=np.uint8),
        )

    def _extract_bbox(self, object) -> List:
        bbox = object.find('bndbox')
        # Voc from Matlab, which indices start from 0
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1
        angle = float(bbox.find('angle').text)
        return [x1, y1, x2, y2, angle]

    def _read_image(self, image_id: str) -> np.ndarray:
        image_path = os.path.join(self._root_path, f"JPEGImages/{image_id}.jpg")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __len__(self):
        return len(self._config.image_ids)
