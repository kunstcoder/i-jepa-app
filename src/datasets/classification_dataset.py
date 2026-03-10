from pathlib import Path

from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_classification_transform(img_size: int = 224, is_train: bool = True):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class ClassificationDataset(datasets.ImageFolder):
    """ImageFolder dataset with I-JEPA-compatible transforms.

    Expected directory layout:
        root/
          class_a/
            img_001.jpg
            ...
          class_b/
            ...
    """

    def __init__(self, root: str, img_size: int = 224, is_train: bool = True):
        transform = get_classification_transform(img_size, is_train)
        super().__init__(root=root, transform=transform)
