import torch.nn.functional as F

from torch import nn, Tensor
from torchvision.models import ResNet
from typing import List


class SSDNetwork(nn.Module):
    def __init__(self, resnet34: ResNet):
        super().__init__()
        self._base = self._create_resnet34_based_block()
        self._extra_conv1 = self._create_extra_conv1()
        self._extra_conv2 = self._create_extra_conv2()
        self._extra_conv3 = self._create_extra_conv3()

        self._side_base_conv1 = self._side_conv(in_channels=256)
        self._side_extra_conv1 = self._side_conv(in_channels=512)
        self._side_extra_conv2 = self._side_conv(in_channels=512)
        self._side_extra_conv3 = self._side_conv(in_channels=256)

    def _create_resnet34_based_block(self, resnet34: ResNet) -> nn.Module:
        return nn.Sequential(
            resnet34.conv1,
            resnet34.bn1,
            resnet34.relu,
            resnet34.maxpool,
            resnet34.layer1,
            resnet34.layer2,
            resnet34.layer3,
        )

    def _create_extra_conv1(self) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )

    def _create_extra_conv2(self) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
        )

    def _create_extra_conv3(self) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
        )

    def _side_conv(self, in_channels: int) -> nn.Conv2d:
        return nn.Conv2d(in_channels, out_channels=24, kernel_size=3, padding=1, stride=1)

    def forward(self, x: Tensor) -> List[Tensor]:
        output = list()

        base_output = self._base.forward(x)
        output.append(self.side_base_conv1(F.relu(base_output)))

        extra_conv1_output = F.relu(self._extra_conv1(base_output))
        output.append(self.side_extra_conv1(extra_conv1_output))

        extra_conv2_output = self._extra_conv2(extra_conv1_output)
        output.append(self._side_extra_conv2(extra_conv2_output))

        extra_conv3_output = self._extra_conv3(extra_conv2_output)
        output.append(self._side_extra_conv3(extra_conv3_output))

        return output
