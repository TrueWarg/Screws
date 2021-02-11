from torchvision import models


def train():
    pass


if __name__ == '__main__':
    resnet34 = models.resnet34(pretrained=False)
    print(resnet34)
    train()
