"""
Mobile Detector Predictor.
Encapsulates model initialization, preprocessing, and single/batch inference with confidence scoring.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from configs.config import model_cfg
from dataset.transforms import get_video_frame_transforms
from models.mobilenet_v3 import load_model


class MobileDetectorPredictor:
    """
    High-performance inference engine for MobileNetV3 Mobile Detector.
    Supports single images, raw OpenCV frames, and mini-batch tensor inference.
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        num_classes: int = model_cfg.num_classes,
        class_names: Optional[List[str]] = None
    ):
        if weights_path is None:
            weights_path = model_cfg.weights_path
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.class_names = class_names or model_cfg.class_names
        self.model = load_model(
            weights_path=weights_path,
            device=self.device,
            num_classes=num_classes
        )
        self.model.eval()
        self.transform = get_video_frame_transforms(image_size=model_cfg.image_size)

    def _prepare_pil_image(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> Image.Image:
        """Standardizes varied inputs into RGB PIL Image."""
        if isinstance(image_input, (str, Path)):
            return Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            # Assume OpenCV BGR if 3 channels
            if HAS_CV2 and len(image_input.shape) == 3 and image_input.shape[2] == 3:
                rgb_arr = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb_arr)
            return Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def predict_single(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image]
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Runs inference on a single frame.

        Returns:
            Tuple of:
              - predicted_label: 'normal' or 'using_phone'
              - confidence: float score for the top predicted class
              - probabilities: dict mapping class name to softmax probability
        """
        pil_img = self._prepare_pil_image(image_input)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        predicted_idx = int(np.argmax(probs))
        predicted_label = self.class_names[predicted_idx]
        confidence = float(probs[predicted_idx])

        prob_dict = {
            name: float(probs[i]) for i, name in enumerate(self.class_names)
        }
        return predicted_label, confidence, prob_dict

    def predict_batch(
        self,
        images: List[Union[str, Path, np.ndarray, Image.Image]]
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Vectorized batch inference over multiple frames.
        Significantly accelerates video processing pipelines.
        """
        if not images:
            return []

        pil_images = [self._prepare_pil_image(img) for img in images]
        tensors = torch.stack([self.transform(img) for img in pil_images]).to(self.device)

        with torch.no_grad():
            logits = self.model(tensors)
            probs = F.softmax(logits, dim=1).cpu().numpy()

        results = []
        for i in range(len(images)):
            pred_idx = int(np.argmax(probs[i]))
            label = self.class_names[pred_idx]
            conf = float(probs[i][pred_idx])
            prob_dict = {
                name: float(probs[i][j]) for j, name in enumerate(self.class_names)
            }
            results.append((label, conf, prob_dict))

        return results
