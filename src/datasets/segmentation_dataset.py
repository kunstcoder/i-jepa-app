from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SegmentationDataset(Dataset):
    """Dataset for semantic segmentation with image + mask pairs.

    Expected directory layout:
        images_dir/
          img_001.jpg
          img_002.jpg
        masks_dir/
          img_001.png   (class index per pixel, 0-indexed)
          img_002.png

    Mask files should be single-channel PNGs where each pixel value
    is the class index (0 to num_classes-1). Use 255 for ignore regions.
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        img_size: int = 224,
        is_train: bool = True,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size
        self.is_train = is_train

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        self.images = sorted([
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in image_extensions
        ])

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        mask_name = img_path.stem + ".png"
        mask_path = self.masks_dir / mask_name

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        image = self.img_transform(image)
        mask = mask.resize(
            (self.img_size, self.img_size), resample=Image.NEAREST,
        )
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask
