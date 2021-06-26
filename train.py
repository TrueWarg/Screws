import torch.optim as optim
import torch
import os

from dataset.augmentation.transforms import TrainTransform
from model.ssd.ssd_network import SSDNetwork
from torchvision import models
from torch import nn
from torch.utils.data import DataLoader
from dataset.coco_dataset import CocoDataset

DEVICE = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model: nn.Module, optimizer: optim.Optimizer, data_loader: DataLoader, loss) -> nn.Module:
    model.train()

    optimizer.zero_grad()

    for i, batch in data_loader:
        images, ground_truth = batch

        images = [image.to(DEVICE) for image in images]
        ground_truth = [g.to(DEVICE) for g in ground_truth]

        predicted = model(images)
        loss_result = loss(predicted, ground_truth)
        loss_result.backward()
        optimizer.step()
        optimizer.zero_grad()

    return model


if __name__ == '__main__':
    train_dataset_path = ''
    validation_dataset_path = ''
    images_path = ''
    base_net_path = 'models/mobilenet_v1_with_relu_69_5.pth'
    batch_size = 8
    num_epochs = 300
    lr = 0.01
    config = ""
    # train_transform = TrainTransform(config.image_size, config.image_mean, config.image_std)
    # target_transform = MatchPrior(config.priors, config.center_variance,
    #                               config.size_variance, 0.5)
    #
    # test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)


def train(args):
    train_dataset_path = ''
    validation_dataset_path = ''

    os.makedirs(f'{args.checkpoint_dir}', exist_ok=True)

    model = SSDNetwork(resnet34=models.resnet34(pretrained=True))
    model.to(DEVICE)

    # todo add scheduler for optimiser later
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # todo add batch sampler
    dataset = CocoDataset(images_path='', images=[], annotations=[])
    data_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)

    for epoch in range(args.epoch_count):
        model = train_epoch(model, optimizer, data_loader)

    torch.save(model, f'{args.checkpoint_dir}/model.pt')


# todo add some arg parser
if __name__ == '__main__':
    resnet34 = models.resnet34(pretrained=False)
    print(resnet34)
    train()
