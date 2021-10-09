import json
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from bbox.metrics import iou
from dataset.augmentation.transforms import TestTransform, PredictionTransform
from dataset.voc_dataset import Config, VOCDataset
from file_readers import read_image_ids, read_class_label
from model.ssd.mobilenet import mobileV1_ssd_config
from model.ssd.mobilenet.mobileV1_ssd import create_mobilenetv1_ssd
from model.ssd.mobilenet.mobileV1_ssd_config import CONFIG
from model.ssd.predictor import Predictor
from model.ssd.prior_matcher import RotatedPriorMatcher
from model.ssd.ssd import SSDTest

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BACKGROUND_CLASS = 'BACKGROUND'


@dataclass()
class EvalConfig:
    dataset_path: str
    image_ids_path: str
    labels_path: str
    trained_model_path: str
    results_path: str
    iou_threshold: float


# add default value on error?
def _read_config() -> EvalConfig:
    with open("eval_config.json") as f:
        config_items = json.load(f)

    return EvalConfig(
        dataset_path=config_items['dataset_path'],
        image_ids_path=config_items['image_ids_path'],
        labels_path=config_items['labels_path'],
        trained_model_path=config_items['trained_model_path'],
        results_path=config_items['results_path'],
        iou_threshold=config_items['iou_threshold']
    )


def _group_annotation_by_class(dataset: VOCDataset) -> Tuple:
    """
    Make grouping annotations:

    all_gt_boxes = {
      class_index_0 : { image_id_0 : [box_0, ... box_n], image_id_1 : [box_0, ...], ...},

      class_index_1 : { image_id_0 : [box_0, ... box_n], image_id_1 : [box_0, ...], ...}

    }

    all_difficult_cases = {
      class_index_0 : { image_id_0 : [box_0_is_difficult, ...], image_id_1 : [box_0_is_difficult, ...], ...},

      class_index_1 : { image_id_0 : [box_0_is_difficult, ... ], image_id_1 : [box_0_is_difficult, ...], ...}

    }

    :param dataset: voc dataset
    :return: (not_difficult_cases_count_by_class_id, all_gt_boxes, all_difficult_cases)
    """
    not_difficult_cases_count_by_class_id = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    for annotation_index in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(annotation_index)
        gt_boxes, classes, difficult_cases = annotation
        gt_boxes = torch.from_numpy(gt_boxes)

        for index in range(0, len(classes)):
            class_id = int(classes[index])
            gt_box = gt_boxes[index]
            difficult = difficult_cases[index]
            if not difficult:
                not_difficult_cases_count_by_class_id[class_id] = \
                    not_difficult_cases_count_by_class_id.get(class_id, 0) + 1

            if class_id not in all_gt_boxes:
                all_gt_boxes[class_id] = {}
            if image_id not in all_gt_boxes[class_id]:
                all_gt_boxes[class_id][image_id] = []

            all_gt_boxes[class_id][image_id].append(gt_box)

            if class_id not in all_difficult_cases:
                all_difficult_cases[class_id] = {}
            if image_id not in all_difficult_cases[class_id]:
                all_difficult_cases[class_id][image_id] = []

            all_difficult_cases[class_id][image_id].append(difficult)

    for class_id in all_gt_boxes:
        for image_id in all_gt_boxes[class_id]:
            all_gt_boxes[class_id][image_id] = torch.stack(all_gt_boxes[class_id][image_id])

    return not_difficult_cases_count_by_class_id, all_gt_boxes, all_difficult_cases


def _compute_average_precision_per_class(
        num_true_cases,
        gt_boxes,
        difficult_cases,
        prediction_file,
        iou_threshold,
        use_2007_metric,
):
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue

            gt_box = gt_boxes[image_id]
            # ious = box_utils.iou_of(box, gt_box) * torch.cos(box[..., 4] - gt_box[..., 4])
            ious = iou(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases
    if use_2007_metric:
        return _compute_voc2007_average_precision(precision, recall)
    else:
        return _compute_average_precision(precision, recall)


def _compute_average_precision(precision, recall):
    """
    It computes average precision based on the definition of Pascal Competition. It computes the under curve area
    of precision and recall. Recall follows the normal definition. Precision is a variant.
    pascal_precision[i] = typical_precision[i:].max()
    """
    # identical but faster version of new_precision[i] = old_precision[i:].max()
    precision = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    # find the index where the value changes
    recall = np.concatenate([[0.0], recall, [1.0]])
    changing_points = np.where(recall[1:] != recall[:-1])[0]

    # compute under curve area
    areas = (recall[changing_points + 1] - recall[changing_points]) * precision[changing_points + 1]
    return areas.sum()


def _compute_voc2007_average_precision(precision, recall):
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap = ap + p / 11.
    return ap


if __name__ == '__main__':
    eval_config = _read_config()

    images_ids = read_image_ids(os.path.join(eval_config.dataset_path, eval_config.image_ids_path))

    class_labels = read_class_label(eval_config.labels_path)
    class_labels.insert(0, BACKGROUND_CLASS)

    dataset_config = Config(
        root_path=eval_config.dataset_path,
        annotations_relative_path='Annotations',
        annotation_extension='xml',
        images_relative_path='JPEGImages',
        images_extension='png',
        image_ids=images_ids,
        class_labels=tuple(class_labels),
        skip_difficult=False
    )

    config = mobileV1_ssd_config.CONFIG
    priors = mobileV1_ssd_config.priors

    target_transform = RotatedPriorMatcher(priors, config.center_variance, config.size_variance, iou_threshold=0.5)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    dataset = VOCDataset(dataset_config)

    num_classes = len(dataset_config.class_labels)
    net = create_mobilenetv1_ssd(num_classes)
    net.load(eval_config.trained_model_path)
    net = SSDTest(
        ssd=net,
        config=CONFIG,
        priors=mobileV1_ssd_config.priors,
    )
    net = net.to(DEVICE)

    true_case_stat, all_gb_boxes, all_difficult_cases = _group_annotation_by_class(dataset)

    predictor = Predictor(
        net=net,
        transform=PredictionTransform(CONFIG.image_size, CONFIG.image_mean, CONFIG.image_std),
        iou_threshold=CONFIG.iou_threshold,
        candidate_size=200,
        device=DEVICE,
        filter_threshold=0.01
    )

    results = []
    for i in range(len(dataset)):
        print("process image", i)
        image = dataset.get_image(i)
        boxes, labels, probs = predictor.predict(image)
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        results.append(torch.cat([
            indexes.reshape(-1, 1),
            labels.reshape(-1, 1).float(),
            probs.reshape(-1, 1),
            boxes + 1.0
        ], dim=1))
    results = torch.cat(results)
    for class_index, class_label in enumerate(dataset_config.class_labels):
        if class_index == 0:
            continue  # ignore background
        prediction_path = os.path.join(eval_config.results_path, f'detection_{class_label}.txt')

        with open(prediction_path, "w") as f:
            sub = results[results[:, 1] == class_index, :]
            for i in range(sub.size(0)):
                prob_box = sub[i, 2:].numpy()
                image_id = dataset_config.image_ids[int(sub[i, 0])]
                print(
                    image_id + " " + " ".join([str(v) for v in prob_box]),
                    file=f
                )
    aps = []
    print("\n\nAverage Precision Per-class:")
    for class_index, class_label in enumerate(dataset_config.class_labels):
        if class_index == 0:
            continue
        prediction_path = os.path.join(eval_config.results_path, f'detection_{class_label}.txt')
        ap = _compute_average_precision_per_class(
            true_case_stat[class_index],
            all_gb_boxes[class_index],
            all_difficult_cases[class_index],
            prediction_path,
            eval_config.iou_threshold,
            True
        )
        aps.append(ap)
        print(f"{class_label}: {ap}")

    print(f"\nAverage Precision Across All Classes:{sum(aps) / len(aps)}")
