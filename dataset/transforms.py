"""
Image transformation pipelines for training, validation, testing, and real-time video frames.
Matches standard PyTorch ImageNet normalization.
"""

from torchvision import transforms
import numpy as np
import torch

# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Data augmentation and normalization pipeline for training.
    Applies random horizontal flip, slight rotation, color jitter, and standard normalization.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Deterministic preprocessing pipeline for validation, testing, and offline evaluation.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_video_frame_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Transform pipeline applied to RGB PIL images or OpenCV frames converted to PIL.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def unnormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverses ImageNet normalization to return a displayable numpy array [H, W, C] in [0, 1].
    """
    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1.0 / s for s in IMAGENET_STD]
    )
    unnorm = inv_normalize(tensor.cpu().clone())
    img = unnorm.permute(1, 2, 0).numpy()
    return np.clip(img, 0.0, 1.0)
