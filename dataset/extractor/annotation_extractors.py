import json
from typing import Tuple, Dict

import numpy as np


class AnnotationExtractor:
    def __init__(self):
        pass


class VertexFormAnnotationExtractor(AnnotationExtractor):
    def __init__(self, class_label_to_ids: Dict):
        super().__init__()
        self._labels_to_ids = class_label_to_ids

    def __call__(self, path: str) -> Tuple:
        objects = json.load(open(path))
        boxes = [obj['bbox'] for obj in objects]
        class_ids = [self._labels_to_ids[obj['name']] for obj in objects]
        boxes = np.array(boxes, dtype=np.float32)
        class_ids = np.array(class_ids)

        return boxes, class_ids
