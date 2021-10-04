from typing import List


def read_image_ids(image_sets_path) -> List:
    ids = []
    with open(image_sets_path) as f:
        for line in f:
            ids.append(line.rstrip())
    return ids


def read_class_label(labels_path) -> List:
    labels = []
    with open(labels_path) as f:
        for line in f:
            labels.append(line.rstrip())
    return labels
