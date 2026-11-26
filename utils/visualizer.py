"""
Visualizer module for Mobile Detection.
Draws professional HUD/alert overlays and formats timestamps for driver trip incident screenshots.
Supports both OpenCV (cv2) and PIL fallback.
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from PIL import Image, ImageDraw, ImageFont


def format_seconds_to_timestamp(seconds: float, include_millis: bool = False) -> str:
    """
    Converts seconds (e.g. 872.4) to formatted string: 'HH:MM:SS' or 'HH:MM:SS.mmm'.
    """
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    
    if include_millis:
        millis = int((seconds - total_seconds) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def annotate_screenshot(
    image: Union[np.ndarray, Image.Image],
    timestamp_str: str,
    confidence: float,
    incident_id: Optional[int] = None,
    output_path: Optional[Union[str, Path]] = None
) -> np.ndarray:
    """
    Overlays a high-visibility violation banner on the screenshot:
      - Banner with Alert badge
      - Trip Timestamp (HH:MM:SS)
      - Model Detection Confidence (%)
      - Incident number

    Args:
        image: OpenCV frame (BGR np.ndarray) or PIL Image.
        timestamp_str: Formatted time string (e.g. '00:14:32').
        confidence: Probability score between 0.0 and 1.0.
        incident_id: Sequential incident count (e.g. 1, 2, ...).
        output_path: If specified, writes the image to disk.

    Returns:
        np.ndarray: Annotated BGR image suitable for cv2.imwrite or display.
    """
    # Convert PIL Image to OpenCV BGR if necessary
    if isinstance(image, Image.Image):
        rgb_arr = np.array(image.convert("RGB"))
        frame = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR) if HAS_CV2 else rgb_arr
    else:
        frame = image.copy()

    h, w = frame.shape[:2]

    if HAS_CV2:
        # 1. Overlay Header Banner (Semi-transparent dark red banner at top)
        overlay = frame.copy()
        banner_height = max(70, int(h * 0.12))
        cv2.rectangle(overlay, (0, 0), (w, banner_height), (20, 20, 160), -1)  # BGR Dark Red
        alpha = 0.82
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # 2. Text layout & styling
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.55, min(w / 1000.0, 0.9))
        thickness = 2

        incident_tag = f"INCIDENT #{incident_id} | " if incident_id is not None else ""
        header_text = f"ALERT: MOBILE PHONE USAGE DETECTED"
        meta_text = f"{incident_tag}TIMESTAMP: {timestamp_str} | CONFIDENCE: {confidence * 100:.1f}%"

        # Red warning pill indicator
        pill_radius = int(banner_height * 0.18)
        pill_center = (25, int(banner_height * 0.35))
        cv2.circle(frame, pill_center, pill_radius, (0, 0, 255), -1)

        # Draw Title
        cv2.putText(
            frame,
            header_text,
            (45, int(banner_height * 0.42)),
            font,
            font_scale * 1.05,
            (255, 255, 255),
            thickness + 1,
            cv2.LINE_AA
        )

        # Draw Subtitle metadata
        cv2.putText(
            frame,
            meta_text,
            (45, int(banner_height * 0.82)),
            font,
            font_scale * 0.85,
            (200, 240, 255),
            max(1, thickness - 1),
            cv2.LINE_AA
        )

        # 3. Outer border warning
        border_thickness = 3
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 220), border_thickness)

    else:
        # Fallback using PIL
        pil_img = Image.fromarray(frame if isinstance(image, Image.Image) else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        banner_h = max(60, int(h * 0.12))
        draw.rectangle([(0, 0), (w, banner_h)], fill=(160, 20, 20, 220))
        incident_tag = f"INCIDENT #{incident_id} | " if incident_id is not None else ""
        text = f"ALERT: MOBILE USAGE DETECTED\n{incident_tag}TIME: {timestamp_str} | CONF: {confidence * 100:.1f}%"
        draw.text((20, 10), text, fill=(255, 255, 255))
        frame = np.array(pil_img)

    # 4. Save to disk if requested
    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if HAS_CV2:
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            Image.fromarray(frame).save(str(out_path), quality=95)

    return frame
