import random
from pathlib import Path

import torch
from PIL import Image, ImageFilter, ImageOps, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SketchInpaintingDataset(Dataset):
    """Dataset for masked sketch reconstruction from RGB images."""

    def __init__(
        self,
        images_dir: str,
        img_size: int = 224,
        is_train: bool = True,
        min_mask_ratio: float = 0.1,
        max_mask_ratio: float = 0.4,
        sketch_blur_radius: float = 0.5,
    ):
        self.images_dir = Path(images_dir)
        self.img_size = img_size
        self.is_train = is_train
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.sketch_blur_radius = sketch_blur_radius

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        self.images = sorted([
            p for p in self.images_dir.iterdir() if p.suffix.lower() in image_extensions
        ])
        if not self.images:
            raise ValueError(f"No images found in {self.images_dir}")

        if is_train:
            self.image_transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

        self.pre_sketch_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.images)

    def _build_mask(self) -> torch.Tensor:
        mask_img = Image.new("L", (self.img_size, self.img_size), 0)
        draw = ImageDraw.Draw(mask_img)

        target_area = random.uniform(self.min_mask_ratio, self.max_mask_ratio) * (self.img_size ** 2)
        painted = 0
        while painted < target_area:
            if random.random() < 0.7:
                w = random.randint(self.img_size // 8, self.img_size // 2)
                h = random.randint(self.img_size // 8, self.img_size // 2)
                x1 = random.randint(0, self.img_size - w)
                y1 = random.randint(0, self.img_size - h)
                draw.rectangle((x1, y1, x1 + w, y1 + h), fill=255)
                painted += w * h
            else:
                points = [
                    (random.randint(0, self.img_size), random.randint(0, self.img_size))
                    for _ in range(random.randint(3, 6))
                ]
                width = random.randint(8, 24)
                draw.line(points, fill=255, width=width)
                painted += width * len(points) * (self.img_size / 8)

        mask = transforms.ToTensor()(mask_img)
        return (mask > 0.5).float()

    def _create_sketch_target(self, image: Image.Image) -> torch.Tensor:
        gray = ImageOps.grayscale(image)
        edge = gray.filter(ImageFilter.FIND_EDGES)
        if self.sketch_blur_radius > 0:
            edge = edge.filter(ImageFilter.GaussianBlur(radius=self.sketch_blur_radius))
        edge = ImageOps.autocontrast(edge)
        edge = ImageOps.invert(edge)
        return self.pre_sketch_transform(edge)

    def __getitem__(self, idx: int):
        image = Image.open(self.images[idx]).convert("RGB")
        sketch_target = self._create_sketch_target(image)
        image_tensor = self.image_transform(image)
        mask = self._build_mask()
        return image_tensor, mask, sketch_target
