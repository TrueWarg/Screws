import itertools
import logging
import os

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset.augmentation.transforms import TrainTransform, TestTransform
from dataset.voc_dataset import VOCDataset, Config
from model.ssd.box_losses import RotatedMultiboxLoss
from model.ssd.mobilenet import mobileV1_ssd_config
from model.ssd.mobilenet.mobileV1_ssd import create_mobilenetv1_ssd
from model.ssd.prior_matcher import RotatedPriorMatcher

# todo refactor all training flow when working of it will be available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(loader, net, loss_function, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = loss_function(confidence, locations, labels, boxes)
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0


def test(loader, net, loss_function, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = loss_function(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


def _read_image_ids(image_sets_file):
    ids = []
    with open(image_sets_file) as f:
        for line in f:
            ids.append(line.rstrip())
    return ids


if __name__ == '__main__':
    train_dataset_path = '/home/truewarg/data/VOC2007-fake-3/'
    train_images_path = "ImageSets/Main/trainval.txt"
    train_images_ids = _read_image_ids(os.path.join(train_dataset_path, train_images_path))

    validation_dataset_path = '/home/truewarg/data/fake-test-3/VOC2007-fake/'
    validation_images_path = "ImageSets/Main/test.txt"
    validation_images_ids = _read_image_ids(os.path.join(validation_dataset_path, validation_images_path))

    base_net_path = 'models/mobilenet_v1_with_relu_69_5.pth'
    batch_size = 8
    num_epochs = 300
    lr = 0.01
    momentum = 0.9
    weight_decay = 5e-4
    t_max = 120
    checkpoint_path = ''
    checkpoint_folder = '/models'

    config = mobileV1_ssd_config.config
    priors = mobileV1_ssd_config.priors

    train_transform = TrainTransform(config.image_size, config.image_mean, config.image_std)
    target_transform = RotatedPriorMatcher(priors, config.center_variance, config.size_variance, 0.5)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    # todo add label file
    train_dataset_config = Config(
        root_path=train_dataset_path,
        images_sets_relative_path=train_images_path,
        image_ids=train_images_ids,
        class_labels=(),
        difficult_only=False
    )
    train_dataset = VOCDataset(train_dataset_config, transform=train_transform, target_transform=target_transform)
    label_file = os.path.join(checkpoint_path, "voc-model-labels.txt")
    # store_labels(label_file, train_dataset_config.class_labels)
    num_classes = len(train_dataset_config.class_labels)
    train_loader = DataLoader(train_dataset, batch_size, num_workers=4, shuffle=True)

    # todo add label file
    train_dataset_config = Config(
        root_path=validation_dataset_path,
        images_sets_relative_path=validation_images_path,
        image_ids=validation_images_ids,
        class_labels=(),
        difficult_only=False
    )

    val_dataset = VOCDataset(train_dataset_config, transform=test_transform, target_transform=target_transform)
    val_loader = DataLoader(val_dataset, batch_size, num_workers=4, shuffle=False)

    net = create_mobilenetv1_ssd(num_classes)

    params = [
        {'params': net.base_net.parameters(), 'lr': lr},
        {'params': itertools.chain(
            net.source_layer_add_ons.parameters(),
            net.extras.parameters()
        ), 'lr': lr},
        {'params': itertools.chain(
            net.regression_headers.parameters(),
            net.classification_headers.parameters()
        )}
    ]

    last_epoch = -1
    net.to(device)

    loss_function = RotatedMultiboxLoss(
        priors, iou_threshold=0.5, neg_pos_ratio=3, center_variance=0.1, size_variance=0.2, device=device,
    )
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, t_max, last_epoch=last_epoch)

    for epoch in range(last_epoch + 1, num_epochs):
        scheduler.step()
        train(train_loader, net, loss_function, optimizer, device=device, debug_steps=100, epoch=epoch)
        if epoch % 10 == 0:
            print(f"epoch = {epoch}")
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, loss_function, device)
            logging.info(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.8f}, " +
                f"Validation Regression Loss {val_regression_loss:.8f}, " +
                f"Validation Classification Loss: {val_classification_loss:.8f}"
            )
            model_path = os.path.join(checkpoint_folder, f"mobilev1-ssd-Epoch-{epoch}-Loss-{val_loss}.pth")
            net.save(model_path)
            logging.info(f"Saved model {model_path}")
