"""
Configuration module for Mobile Detection System.
Centralizes all hyperparameters, model paths, video processing rules, and cloud settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
import os
import torch


@dataclass
class ModelConfig:
    """Model architecture and checkpoint settings."""
    model_name: str = "mobilenet_v3_large"
    num_classes: int = 2
    class_names: list = field(default_factory=lambda: ["normal", "using_phone"])
    class_to_idx: dict = field(default_factory=lambda: {"normal": 0, "using_phone": 1})
    weights_path: Path = Path(__file__).resolve().parent.parent / "weights" / "best_mobile_detector.pth"
    image_size: int = 224
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DatasetConfig:
    """Dataset splitting and State Farm Kaggle class mapping."""
    seed: int = 42
    source_dir: Path = Path("data/kaggle_state_farm/train")
    output_dir: Path = Path("data/dataset_binary")
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # State Farm Kaggle competition classes:
    # c0: safe driving
    # c1: texting - right
    # c2: talking on the phone - right
    # c3: texting - left
    # c4: talking on the phone - left
    # c5-c9: other distractions (operating radio, drinking, talking to passenger, etc.)
    class_mapping: dict = field(default_factory=lambda: {
        "c0": "normal",
        "c1": "using_phone",
        "c2": "using_phone",
        "c3": "using_phone",
        "c4": "using_phone",
    })


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 32
    num_epochs: int = 5
    learning_rate: float = 1e-4
    num_workers: int = 2
    pin_memory: bool = True
    save_dir: Path = Path(__file__).resolve().parent.parent / "weights"


@dataclass
class VideoProcessingConfig:
    """
    Settings specifically tuned for long cloud trip videos (up to 1 hour+).
    
    A 1-hour video at 30 fps has ~108,000 frames.
    Sampling at 1.0 or 2.0 FPS reduces frame count to 3,600 - 7,200 frames,
    allowing inference to finish in minutes/seconds with high temporal accuracy.
    """
    sample_fps: float = 1.0          # Process 1 frame per second (configurable)
    batch_size: int = 32             # Mini-batch inference size for GPU/CPU acceleration
    confidence_threshold: float = 0.70  # Minimum softmax probability to classify as using_phone
    min_consecutive_detections: int = 2  # Require at least N detections to confirm an incident
    cooldown_seconds: float = 3.0    # Time gap with no phone detections before closing an incident
    save_annotated_ss: bool = True   # Save screenshot with timestamp & visual banner
    save_raw_ss: bool = False        # Also save an un-annotated raw screenshot
    output_base_dir: Path = Path("runs/trip_detections")


@dataclass
class CloudConfig:
    """Cloud server, API and worker settings."""
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    upload_temp_dir: Path = Path("temp_uploads")
    max_video_size_mb: int = 4096    # 4 GB max upload size for long trips
    worker_concurrency: int = 2


# Global Default Instances
model_cfg = ModelConfig()
dataset_cfg = DatasetConfig()
training_cfg = TrainingConfig()
video_cfg = VideoProcessingConfig()
cloud_cfg = CloudConfig()
