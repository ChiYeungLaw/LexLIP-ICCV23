import torch
from .utils import (
    inception_normalize,
    imagenet_normalize,
    MinMaxResize,
)
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from .randaug import RandAugment

logit_laplace_eps: float = 0.1
def map_pixels(x: torch.Tensor) -> torch.Tensor:
	if x.dtype != torch.float:
		raise ValueError('expected input to have type float')

	return (1 - 2 * logit_laplace_eps) * x + logit_laplace_eps

def pixelbert_transform(size=800):
    longer = int((1333 / 800) * size)
    return transforms.Compose(
        [
            MinMaxResize(shorter=size, longer=longer),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )

def pixelbert_transform_randaug(size=800):
    longer = int((1333 / 800) * size)
    trs = transforms.Compose(
        [
            MinMaxResize(shorter=size, longer=longer),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )
    trs.transforms.insert(0, RandAugment(2, 9))
    return trs

def imagenet_transform(size=800):
    return transforms.Compose(
        [
            Resize(size, interpolation=Image.BICUBIC),
            CenterCrop(size),
            transforms.ToTensor(),
            imagenet_normalize,
        ]
    )

def imagenet_transform_randaug(size=800):
    trs = transforms.Compose(
        [
            Resize(size, interpolation=Image.BICUBIC),
            CenterCrop(size),
            transforms.ToTensor(),
            imagenet_normalize,
        ]
    )
    trs.transforms.insert(0, RandAugment(2, 9))
    return trs

def vit_transform(size=800):
    return transforms.Compose(
        [
            Resize(size, interpolation=Image.BICUBIC),
            CenterCrop(size),
            lambda image: image.convert("RGB"),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )

def vit_transform_randaug(size=800):
    trs = transforms.Compose(
        [
            Resize(size, interpolation=Image.BICUBIC),
            CenterCrop(size),
            lambda image: image.convert("RGB"),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )
    trs.transforms.insert(0, lambda image: image.convert('RGBA'))
    trs.transforms.insert(0, RandAugment(2, 9))
    trs.transforms.insert(0, lambda image: image.convert('RGB'))
    return trs


def beit_transform(size=800):
    common_transform = transforms.Compose(
        [
            lambda image: image.convert("RGB"),
            Resize(size, interpolation=Image.BICUBIC),
            CenterCrop(size),
        ]
    )

    patch_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            inception_normalize,
        ]
    )

    visual_token_transform = transforms.Compose(
        [
            Resize(size // 2, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            map_pixels,
        ]
    )

    return [common_transform, patch_transform, visual_token_transform]

def beit_transform_randaug(size=224):
    common_transform = transforms.Compose(
        [
            lambda image: image.convert("RGB"),
            Resize(size, interpolation=Image.BICUBIC),
            CenterCrop(size),
        ]
    )
    common_transform.transforms.insert(0, lambda image: image.convert('RGBA'))
    common_transform.transforms.insert(0, RandAugment(2, 9))
    common_transform.transforms.insert(0, lambda image: image.convert('RGB'))

    patch_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            inception_normalize,
        ]
    )

    visual_token_transform = transforms.Compose(
        [
            Resize(size // 2, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            map_pixels,
        ]
    )

    return [common_transform, patch_transform, visual_token_transform]

