from torch.nn import Conv2d, Sequential, ModuleList, ReLU, Module

from model.nn.mobile_net_v1 import MobileNetV1
from model.ssd import Predictor
from model.ssd.config import Config
from model.ssd.mobilenet.mobileV1_ssd_config import config, priors
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
        config=config,
        priors=priors,
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


#  todo make some predictor config ?
def create_mobilenetv1_ssd_predictor(
        net: Module,
        transform,
        nms_method=None,
        iou_threshold=0.5,
        candidate_size=200,
        sigma=0.5,
        device=None,
):
    predictor = Predictor(net=net,
                          transform=transform,
                          nms_method=nms_method,
                          iou_threshold=iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device,
                          )
    return predictor
