from torch.nn import Conv2d, Sequential, ModuleList, ReLU

from model.nn.mobile_net_v1 import MobileNetV1
from model.ssd.config import Config
from model.ssd.ssd_network import SSDNetwork


def create_mobilenetv1_ssd(num_classes: int, is_test=False, device=None) -> SSDNetwork:
    base_net = MobileNetV1().conv

    return SSDNetwork(
        base_net=base_net,
        feature_extractors=_create_feature_extraction_layers(),
        classification_headers=_create_classification_headers(num_classes),
        regression_headers=_create_regression_headers(),
        source_layer_indexes=[12, 14],
        num_classes=num_classes,
        device=device,
        config=_create_config(),
        is_test=is_test
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


def _create_classification_headers(num_classes: int) -> ModuleList:
    return ModuleList([
        Conv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=1024, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
    ])


def _create_regression_headers() -> ModuleList:
    return ModuleList([
        Conv2d(in_channels=512, out_channels=6 * 5, kernel_size=3, padding=1),
        Conv2d(in_channels=1024, out_channels=6 * 5, kernel_size=3, padding=1),
        Conv2d(in_channels=512, out_channels=6 * 5, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=6 * 5, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=6 * 5, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=6 * 5, kernel_size=3, padding=1),
    ])


def _create_config() -> Config:
    return Config(
        center_variance=0.1,
        size_variance=0.2,
    )
