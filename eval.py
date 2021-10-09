import json
import os
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import torch

from bbox.metrics import iou, compute_average_precision_per_class
from dataset.augmentation.transforms import TestTransform, PredictionTransform
from dataset.voc_dataset import Config, VOCDataset, BACKGROUND_CLASS_LABEL, BACKGROUND_CLASS_ID
from file_readers import read_image_ids, read_class_label
from model.ssd.mobilenet import mobileV1_ssd_config
from model.ssd.mobilenet.mobileV1_ssd import create_mobilenetv1_ssd
from model.ssd.mobilenet.mobileV1_ssd_config import CONFIG
from model.ssd.predictor import Predictor
from model.ssd.prior_matcher import RotatedPriorMatcher
from model.ssd.ssd import SSDTest

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def _grouped_annotation_by_class(dataset: VOCDataset) -> Tuple:
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

        for index in range(len(classes)):
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


if __name__ == '__main__':
    eval_config = _read_config()

    images_ids = read_image_ids(os.path.join(eval_config.dataset_path, eval_config.image_ids_path))

    class_labels = read_class_label(eval_config.labels_path)
    class_labels.insert(0, BACKGROUND_CLASS_LABEL)

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

    predictor = Predictor(
        net=net,
        transform=PredictionTransform(CONFIG.image_size, CONFIG.image_mean, CONFIG.image_std),
        iou_threshold=CONFIG.iou_threshold,
        candidate_size=200,
        device=DEVICE,
        filter_threshold=0.01
    )

    results = []
    # index ~ index of image ids
    for index in range(len(dataset)):
        print('Process image', index)
        image = dataset.get_image(index)
        boxes, class_ids, scores = predictor.predict(image)
        predictions_count = class_ids.size(0)
        indices = torch.ones(predictions_count, 1, dtype=torch.float32) * index

        # shape: (predictions_count, 8), where 8: (sample_index, class_id, score, x_min, y_min, x_max, y_max, angle)
        results.append(torch.cat([
            indices,
            class_ids.reshape(-1, 1).float(),
            scores.reshape(-1, 1),
            boxes
        ], dim=1))

    # shape: (all_predictions_count_for_all_dataset, 8),
    # where 8: (sample_index, class_id, score, x_min, y_min, x_max, y_max, angle)
    results = torch.cat(results)
    for class_id, class_label in enumerate(dataset_config.class_labels):
        # ignore background
        if class_id == BACKGROUND_CLASS_ID:
            continue
        prediction_path = os.path.join(eval_config.results_path, f'detection_{class_label}.txt')

        with open(prediction_path, "w") as f:
            # 1 - index of class id
            sub = results[results[:, 1] == class_id, :]
            # loop over all samples for current class
            for i in range(sub.size(0)):
                score_and_box = sub[i, 2:].numpy()
                # index ~ index of image ids list
                image_id = dataset_config.image_ids[int(sub[i, 0])]
                print(
                    image_id + " " + " ".join([str(v) for v in score_and_box]),
                    file=f
                )

    average_precisions = []
    print("\n\nAverage Precision Per-class:")
    not_difficult_cases_count_by_class_id, all_gb_boxes, all_difficult_cases = _grouped_annotation_by_class(dataset)
    for class_id, class_label in enumerate(dataset_config.class_labels):
        # ignore background
        if class_id == BACKGROUND_CLASS_ID:
            continue
        prediction_path = os.path.join(eval_config.results_path, f'detection_{class_label}.txt')
        average_precision = compute_average_precision_per_class(
            not_difficult_cases_count_by_class_id[class_id],
            all_gb_boxes[class_id],
            all_difficult_cases[class_id],
            prediction_path,
            eval_config.iou_threshold,
        )
        average_precisions.append(average_precision)
        print(f"{class_label}: {average_precision}")

    print(f"\nAverage Precision Across All Classes:{sum(average_precisions) / len(average_precisions)}")
