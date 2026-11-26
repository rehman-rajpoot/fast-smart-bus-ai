"""Inference package for single-frame prediction and cloud video processing."""
from .predictor import MobileDetectorPredictor
from .video_processor import TripVideoProcessor, TripReport, MobileIncident

__all__ = [
    "MobileDetectorPredictor",
    "TripVideoProcessor",
    "TripReport",
    "MobileIncident"
]
