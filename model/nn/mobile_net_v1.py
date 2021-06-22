from torch import nn
import torch.nn.functional as F


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1024):
        super(MobileNetV1, self).__init__()

        self.conv = nn.Sequential(
            _conv_bn(3, 32, 2),
            _conv_dw(32, 64, 1),
            _conv_dw(64, 128, 2),
            _conv_dw(128, 128, 1),
            _conv_dw(128, 256, 2),
            _conv_dw(256, 256, 1),
            _conv_dw(256, 512, 2),
            _conv_dw(512, 512, 1),
            _conv_dw(512, 512, 1),
            _conv_dw(512, 512, 1),
            _conv_dw(512, 512, 1),
            _conv_dw(512, 512, 1),
            _conv_dw(512, 1024, 2),
            _conv_dw(1024, 1024, 1),
        )
        self.fully_connected = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 1024)
        x = self.fully_connected(x)
        return x


def _conv_bn(in_channels: int, out_channels: int, stride: int):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def _conv_dw(in_channels: int, out_channels: int, stride: int):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=stride, padding=1, groups=in_channels,
                  bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
