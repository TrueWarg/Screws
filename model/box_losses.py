from torch import nn


class RotatedMultiBoxLoss(nn.Module):
    def __init__(self):
        super().__init__()
