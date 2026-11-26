"""
MobileNetV3-Large Model Definition and Checkpoint Loader.
Fine-tuned for binary classification:
  Class 0: Normal safe driving
  Class 1: Using mobile phone (talking, texting, browsing)
"""

from pathlib import Path
from typing import Optional, Union
import logging
import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger("mobile_detection.model")


def build_model(num_classes: int = 2, pretrained_backbone: bool = True) -> nn.Module:
    """
    Constructs a MobileNetV3-Large neural network architecture
    and replaces the final classification layer for binary classification.

    Args:
        num_classes (int): Number of target classes (default 2: normal vs using_phone).
        pretrained_backbone (bool): Whether to load ImageNet pre-trained weights.

    Returns:
        nn.Module: Configured PyTorch model.
    """
    weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
    model = models.mobilenet_v3_large(weights=weights)

    # MobileNetV3 classifier head structure:
    # classifier[0]: Linear(960 -> 1280)
    # classifier[1]: Hardswish()
    # classifier[2]: Dropout(p=0.2, inplace=True)
    # classifier[3]: Linear(1280 -> 1000)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model


def load_model(
    weights_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    num_classes: int = 2,
    fallback_pretrained: bool = True
) -> nn.Module:
    """
    Loads MobileNetV3-Large and restores trained weights from a checkpoint file.
    If no checkpoint exists, can fall back to ImageNet backbone with a notice.

    Args:
        weights_path: Path to .pth state_dict file.
        device: 'cuda', 'cpu', or torch.device. If None, auto-selects.
        num_classes: Number of output classes (default: 2).
        fallback_pretrained: Fall back to ImageNet backbone if checkpoint not found.

    Returns:
        nn.Module: Model in eval mode moved to target device.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = build_model(num_classes=num_classes, pretrained_backbone=fallback_pretrained)

    if weights_path is not None:
        weights_path = Path(weights_path)
        if weights_path.exists():
            logger.info(f"Loading trained weights checkpoint from: {weights_path}")
            checkpoint = torch.load(weights_path, map_location=device)
            # Handle state_dict vs full checkpoint dict
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
            logger.info("Checkpoint weights loaded successfully.")
        else:
            logger.warning(
                f"Checkpoint not found at '{weights_path}'. "
                f"Using default backbone architecture on device '{device}'. "
                f"For optimal inference, train or place 'best_mobile_detector.pth' into weights/."
            )
    else:
        logger.info("No weights_path provided; initialized with default backbone.")

    model = model.to(device)
    model.eval()
    return model
