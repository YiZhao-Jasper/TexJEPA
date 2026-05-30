from logging import getLogger

import torch
import torchvision.transforms as transforms
from PIL import ImageFilter

logger = getLogger()

IMAGENET_NORM = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CHEST_XRAY_NORM = ((0.499, 0.499, 0.499), (0.291, 0.291, 0.291))


def make_pretrain_transforms(
    crop_size=224,
    crop_scale=(0.3, 1.0),
    horizontal_flip=False,
    color_distortion=False,
    color_jitter=0.0,
    gaussian_blur=False,
    normalization=IMAGENET_NORM,
):
    """Transforms for I-JEPA self-supervised pretraining."""
    logger.info('building pretrain transforms')

    transform_list = [
        transforms.RandomResizedCrop(crop_size, scale=crop_scale),
    ]

    if horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    if color_distortion and color_jitter > 0:
        cj = transforms.ColorJitter(
            0.4 * color_jitter, 0.4 * color_jitter, 0.2 * color_jitter, 0.0)
        transform_list.append(transforms.RandomApply([cj], p=0.5))

    if gaussian_blur:
        transform_list.append(GaussianBlur(p=0.3))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(normalization[0], normalization[1]),
    ])

    return transforms.Compose(transform_list)


def make_eval_transforms(
    crop_size=224,
    normalization=IMAGENET_NORM,
):
    """Deterministic transforms for evaluation."""
    return transforms.Compose([
        transforms.Resize(int(crop_size * 256 / 224)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(normalization[0], normalization[1]),
    ])


class GaussianBlur:
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img
        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
