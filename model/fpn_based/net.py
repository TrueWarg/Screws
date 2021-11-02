from torch import nn


class RbboxDetector:
    def __init__(self, backbone: nn.Module):
        self.backbone = backbone

        pass
