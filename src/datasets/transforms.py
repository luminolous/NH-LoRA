from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance


Transform = Callable[[Image.Image], torch.Tensor | Image.Image]


class Compose:
    def __init__(self, transforms: Sequence[Transform]):
        self.transforms = list(transforms)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        output: Image.Image | torch.Tensor = image
        for transform in self.transforms:
            output = transform(output)  # type: ignore[assignment]
        if isinstance(output, Image.Image):
            raise TypeError("Compose must end with a tensor-producing transform.")
        return output


class Resize:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.resize((self.size, self.size), Image.BILINEAR)


class CenterCrop:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        left = max((width - self.size) // 2, 0)
        top = max((height - self.size) // 2, 0)
        return image.crop((left, top, left + self.size, top + self.size))


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        return image


class RandomCrop:
    def __init__(self, size: int, padding: int = 0):
        self.size = size
        self.padding = padding

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.padding > 0:
            width, height = image.size
            padded = Image.new(image.mode, (width + self.padding * 2, height + self.padding * 2))
            padded.paste(image, (self.padding, self.padding))
            image = padded
        width, height = image.size
        if width == self.size and height == self.size:
            return image
        left = random.randint(0, max(width - self.size, 0))
        top = random.randint(0, max(height - self.size, 0))
        return image.crop((left, top, left + self.size, top + self.size))


class RandomResizedCrop:
    def __init__(self, size: int, scale: tuple[float, float] = (0.8, 1.0)):
        self.size = size
        self.scale = scale

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        area = width * height
        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            crop_size = int(round(math.sqrt(target_area)))
            if crop_size <= min(width, height):
                left = random.randint(0, width - crop_size)
                top = random.randint(0, height - crop_size)
                cropped = image.crop((left, top, left + crop_size, top + crop_size))
                return cropped.resize((self.size, self.size), Image.BILINEAR)
        return Resize(self.size)(image)


class ColorJitter:
    def __init__(self, brightness: float = 0.0):
        self.brightness = brightness

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.brightness <= 0:
            return image
        factor = random.uniform(max(0.0, 1.0 - self.brightness), 1.0 + self.brightness)
        return ImageEnhance.Brightness(image).enhance(factor)


class ToTensor:
    def __call__(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 2:
            array = np.expand_dims(array, axis=-1)
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor / 255.0


class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std


def build_cifar_train_transform(image_size: int) -> Compose:
    transforms: List[Transform] = [
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
    ]
    if image_size != 32:
        transforms.append(Resize(image_size))
    transforms.extend(
        [
            ToTensor(),
            Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761),
            ),
        ]
    )
    return Compose(transforms)


def build_cifar_test_transform(image_size: int) -> Compose:
    transforms: List[Transform] = []
    if image_size != 32:
        transforms.append(Resize(image_size))
    transforms.extend(
        [
            ToTensor(),
            Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761),
            ),
        ]
    )
    return Compose(transforms)


def build_imagenet_train_transform(image_size: int) -> Compose:
    return Compose(
        [
            RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def build_imagenet_test_transform(image_size: int) -> Compose:
    return Compose(
        [
            Resize(int(image_size * 256 / 224)),
            CenterCrop(image_size),
            ToTensor(),
            Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
