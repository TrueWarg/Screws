from torch import nn
from torch.nn import Conv2d, Sequential, ModuleList, ReLU

from model.ssd.ssd import SSD


def create_mobilenetv1_ssd(num_classes):
    return SSD(
        num_classes=num_classes,
        base_net=_create_base_net(),
        base_net_source_layer_indices=[12, 14],
        feature_extractors=_create_feature_extraction_layers(),
        classification_headers=_create_classification_headers(num_classes),
        regression_headers=_create_regression_headers(),
    )


def _create_base_net() -> Sequential:
    return Sequential(
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


def _conv_bn(in_channels: int, out_channels: int, stride: int):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def _conv_dw(in_channels: int, out_channels: int, stride: int):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels,
                  bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _create_feature_extraction_layers() -> ModuleList:
    return ModuleList([
        Sequential(
            Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            ReLU()
        ),
        Sequential(
            Conv2d(in_channels=512, out_channels=128, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            ReLU()
        ),
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            ReLU()
        ),
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            ReLU()
        )
    ])


def _create_regression_headers() -> ModuleList:
    return ModuleList([
        Conv2d(in_channels=512, out_channels=14 * 5, kernel_size=3, padding=1),
        Conv2d(in_channels=1024, out_channels=14 * 5, kernel_size=3, padding=1),
        Conv2d(in_channels=512, out_channels=14 * 5, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=14 * 5, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=14 * 5, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=14 * 5, kernel_size=3, padding=1),
    ])


def _create_classification_headers(num_classes: int) -> ModuleList:
    return ModuleList([
        Conv2d(in_channels=512, out_channels=14 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=1024, out_channels=14 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=512, out_channels=14 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=14 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=14 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=14 * num_classes, kernel_size=3, padding=1),
    ])
