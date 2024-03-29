import itertools
import json
import os
from dataclasses import dataclass

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset.augmentation.transforms import TestTransform, TrainTransform
from dataset.voc_dataset import VOCDataset, Config, BACKGROUND_CLASS_LABEL
from file_readers import read_image_ids, read_class_label
from fitter import Fitter
from model.ssd.box_losses import RotatedMultiboxLoss
from model.ssd.mobilenet import mobileV1_ssd_config
from model.ssd.mobilenet.mobileV1_ssd import create_mobilenetv1_ssd
from model.ssd.prior_matcher import RotatedPriorMatcher

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass(frozen=True)
class TrainConfig:
    train_dataset_path: str
    train_image_ids_path: str
    validation_dataset_path: str
    validation_image_ids_path: str
    labels_path: str
    base_net_path: str
    checkpoint_folder_path: str
    batch_size: int
    num_epochs: int
    lr: float
    momentum: float
    weight_decay: float
    t_max: int
    validation_step: int


# add default value on error?
def _read_config() -> TrainConfig:
    with open("train_config.json") as f:
        config_items = json.load(f)

    return TrainConfig(
        train_dataset_path=config_items['train_dataset_path'],
        train_image_ids_path=config_items['train_image_ids_path'],
        validation_dataset_path=config_items['validation_dataset_path'],
        validation_image_ids_path=config_items['validation_image_ids_path'],
        labels_path=config_items['labels_path'],
        base_net_path=config_items['base_net_path'],
        checkpoint_folder_path=config_items['checkpoint_folder_path'],
        batch_size=config_items['batch_size'],
        num_epochs=config_items['num_epochs'],
        lr=config_items['lr'],
        momentum=config_items['momentum'],
        weight_decay=config_items['weight_decay'],
        t_max=config_items['t_max'],
        validation_step=config_items['validation_step'],
    )


if __name__ == '__main__':
    train_config = _read_config()

    train_images_ids = read_image_ids(
        os.path.join(train_config.train_dataset_path, train_config.train_image_ids_path))

    validation_images_ids = read_image_ids(
        os.path.join(train_config.validation_dataset_path, train_config.validation_image_ids_path))

    config = mobileV1_ssd_config.CONFIG
    priors = mobileV1_ssd_config.priors

    train_transform = TrainTransform(config.image_size, config.image_mean, config.image_std)
    target_transform = RotatedPriorMatcher(priors, config.center_variance, config.size_variance, iou_threshold=0.5)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    class_labels = read_class_label(train_config.labels_path)
    class_labels.insert(0, BACKGROUND_CLASS_LABEL)
    num_classes = len(class_labels)

    train_dataset_config = Config(
        root_path=train_config.train_dataset_path,
        annotations_relative_path='Annotations',
        annotation_extension='xml',
        images_relative_path='JPEGImages',
        images_extension='png',
        image_ids=train_images_ids,
        class_labels=tuple(class_labels),
        skip_difficult=False
    )

    train_dataset = VOCDataset(train_dataset_config, transform=train_transform, target_transform=target_transform)
    train_loader = DataLoader(train_dataset, train_config.batch_size, num_workers=4, shuffle=True)

    validation_dataset_config = Config(
        root_path=train_config.validation_dataset_path,
        annotations_relative_path='Annotations',
        annotation_extension='xml',
        images_relative_path='JPEGImages',
        images_extension='png',
        image_ids=validation_images_ids,
        class_labels=tuple(class_labels),
        skip_difficult=False
    )

    val_dataset = VOCDataset(validation_dataset_config, transform=test_transform, target_transform=target_transform)
    val_loader = DataLoader(val_dataset, train_config.batch_size, num_workers=4, shuffle=False)

    net = create_mobilenetv1_ssd(num_classes)

    params = [
        {'params': net.base_net.parameters(), 'lr': train_config.lr},
        {'params': itertools.chain(
            net.feature_extractors.parameters()
        ), 'lr': train_config.lr},
        {'params': itertools.chain(
            net.regression_headers.parameters(),
            net.classification_headers.parameters()
        )}
    ]

    last_epoch = -1
    net.to(DEVICE)

    loss_function = RotatedMultiboxLoss(neg_pos_ratio=3)

    optimizer = torch.optim.SGD(params,
                                lr=train_config.lr,
                                momentum=train_config.momentum,
                                weight_decay=train_config.weight_decay,
                                )
    scheduler = CosineAnnealingLR(optimizer, train_config.t_max, last_epoch=last_epoch)

    fitter = Fitter(
        net=net,
        loss_function=loss_function,
        optimizer=optimizer,
        device=DEVICE,
    )

    for epoch in range(last_epoch + 1, train_config.num_epochs):
        scheduler.step()
        fitter.train(train_loader)
        print(f'Epoch: {epoch}')
        if epoch != 0 and epoch % train_config.validation_step == 0:
            # todo move in config
            train_loss, train_regression_loss, train_classification_loss = fitter.validate(train_loader)
            val_loss, val_regression_loss, val_classification_loss = fitter.validate(val_loader)

            print(
                f'Train Loss: {train_loss:.8f}\n' +
                f'Train Regression Loss: {train_regression_loss:.8f}\n' +
                f'Train Classification Loss: {train_classification_loss:.8f}\n'
            )

            print(
                f'Validation Loss: {val_loss:.8f}\n' +
                f'Validation Regression Loss: {val_regression_loss:.8f}\n' +
                f'Validation Classification Loss: {val_classification_loss:.8f}\n'
            )
            model_path = os.path.join(train_config.checkpoint_folder_path,
                                      f"mobilev1-ssd-Epoch-{epoch}-Loss-{val_loss}.pth")
            net.save(model_path)
            print(f'Saved model {model_path}')
