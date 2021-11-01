import json
import os
from typing import Tuple, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from bbox.converters import vertex_form_to_center_form_angle_bounded45


class RbboxDataset(Dataset):
    def __init__(self,
                 root_path: str,
                 samples_set_paths: List[str],
                 class_labels: str,
                 annotations_extractor,
                 transform=None):
        self._samples_set_paths = samples_set_paths
        self._root = root_path
        self._labels_to_ids = {class_label: class_id for class_id, class_label in enumerate(class_labels)}
        self.image_paths, self.annotation_paths = self._extract_image_and_annotation_paths()
        self._extractor = annotations_extractor
        self._transform = transform

    def __getitem__(self, index: int) -> Tuple:
        image = self._read_image(index)
        boxes, class_ids = self._extractor(self.annotation_paths[index])

        if self._transform:
            image, boxes = self._transform(image, boxes)

        return image, boxes, class_ids

    def _extract_image_and_annotation_paths(self) -> Tuple:
        images = []
        annotations = []
        for samples_set in self._samples_set_paths:
            for image, annotation in json.load(open(os.path.join(self._root, f'{samples_set}.json'))):
                images.append(os.path.join(self._root, image))
                annotations.append(os.path.join(self._root, annotation) if annotation else None)
        return images, annotations

    def _read_image(self, index: int) -> np.ndarray:
        image_path = self.image_paths(index)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


def collate(batch: torch.Tensor) -> Tuple[torch.Tensor, List, List]:
    images, final_boxes, final_class_ids = [], [], []
    for index, (image, boxes, class_ids) in enumerate(batch):
        images.append(torch.from_numpy(image).reshape(*image.shape[:2], -1).float())
        final_boxes.append(
            torch.from_numpy(np.stack([vertex_form_to_center_form_angle_bounded45(box) for box in boxes])).float()
        )
        final_class_ids.append(torch.from_numpy(class_ids).long())

    return torch.stack(images).permute(0, 3, 1, 2), final_boxes, final_class_ids
