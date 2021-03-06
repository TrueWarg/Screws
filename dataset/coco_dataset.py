import os
from typing import Tuple, List

import cv2
from torch.utils.data import Dataset


class CocoDataset(Dataset):
    def __init__(self, images_path: str, images: List, annotations: List, transform=None):
        self._transform = transform
        self._images = images
        self._annotations = annotations
        self._images_path = images_path

    def __getitem__(self, index: int) -> Tuple:
        item = self._images[index]
        file_name = item['file_name']
        image_id = item['id']

        image = cv2.imread(os.path.join(self._images_path, file_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        selection = []
        for annotation in self._annotations:
            if annotation['image_id'] == image_id:
                selection.append(annotation)

        category_ids = list(map(lambda x: x['category_id'], selection))
        boxes = list(map(lambda x: x['bbox'], selection))

        return image, category_ids, boxes

    def __len__(self) -> int:
        return len(self._images)
