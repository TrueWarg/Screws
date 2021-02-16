from torchvision import transforms


class DefaultTransform(object):
    # use some extra config id needed
    def __init__(self, image_size, mean, std):
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.GaussianBlur(kernel_size=3),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image, boxes, category_ids):
        # todo mage transformations for boxes also! torchvision.transforms has no such functionality
        return self.transform(image)
