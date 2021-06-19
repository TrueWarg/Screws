from torch import nn


def xavier_init(module: nn.Module):
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
