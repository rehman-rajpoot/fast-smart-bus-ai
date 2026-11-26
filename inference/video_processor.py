"""
Trip Video Processor for Long Cloud Driver Video Analysis (up to 1 hour+).
Implements:
  - Smart frame downsampling (1.0 - 2.0 FPS) to process 108,000+ frames rapidly.
  - Streaming mini-batch tensor inference to prevent memory overload.
  - Temporal Incident Tracker with cooldown logic and hysteresis.
  - Keyframe extraction saving annotated screenshots with trip timestamps.
  - Automated report generation (JSON and CSV).
"""

import csv
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from tqdm import tqdm

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from configs.config import video_cfg
from inference.predictor import MobileDetectorPredictor
from utils.visualizer import annotate_screenshot, format_seconds_to_timestamp

logger = logging.getLogger("mobile_detection.video_processor")


@dataclass
class MobileIncident:
    """Represents a continuous period where driver was detected using a mobile phone."""
    incident_id: int
    start_time_seconds: float
    end_time_seconds: float
    duration_seconds: float
    start_timestamp: str
    end_timestamp: str
    peak_timestamp: str
    peak_confidence: float
    screenshot_path: str
    raw_screenshot_path: Optional[str] = None


@dataclass
class TripReport:
    """Consolidated analytics report for a full driver trip."""
    video_path: str
    video_duration_seconds: float
    video_duration_formatted: str
    original_fps: float
    total_frames: int
    sampled_frames_processed: int
    processing_time_seconds: float
    total_incidents: int
    total_distraction_seconds: float
    distraction_percentage: float
    safety_risk_level: str
    incidents: List[MobileIncident] = field(default_factory=list)

    def to_dict(self) -> Dict:
        data = asdict(self)
        return data


class TripVideoProcessor:
    """
    Cloud-ready video processing engine tailored for analyzing driver trips.
    """

    def __init__(
        self,
        predictor: Optional[MobileDetectorPredictor] = None,
        sample_fps: float = video_cfg.sample_fps,
        batch_size: int = video_cfg.batch_size,
        confidence_threshold: float = video_cfg.confidence_threshold,
        min_consecutive_detections: int = video_cfg.min_consecutive_detections,
        cooldown_seconds: float = video_cfg.cooldown_seconds,
        save_annotated_ss: bool = video_cfg.save_annotated_ss,
        save_raw_ss: bool = video_cfg.save_raw_ss
    ):
        if not HAS_CV2:
            raise ImportError("OpenCV (cv2) is required for TripVideoProcessor. Please install opencv-python.")

        self.predictor = predictor or MobileDetectorPredictor()
        self.sample_fps = max(0.1, float(sample_fps))
        self.batch_size = max(1, int(batch_size))
        self.confidence_threshold = float(confidence_threshold)
        self.min_consecutive_detections = int(min_consecutive_detections)
        self.cooldown_seconds = float(cooldown_seconds)
        self.save_annotated_ss = save_annotated_ss
        self.save_raw_ss = save_raw_ss

    def process_video(
        self,
        video_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> TripReport:
        """
        Processes a full trip video (supports up to 1-hour or longer).
        Extracts screenshots with timestamp overlays for every mobile usage incident.

        Args:
            video_path: Path to trip video file.
            output_dir: Destination folder for screenshots and reports.
            progress_callback: Optional callback receiving completion fraction (0.0 to 1.0).

        Returns:
            TripReport: Full structured metadata and incident list.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        video_name = video_path.stem
        if output_dir is None:
            output_dir = video_cfg.output_base_dir / video_name
        output_dir = Path(output_dir)
        screenshots_dir = output_dir / "screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initiating trip analysis for '{video_path.name}'...")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / orig_fps if total_frames > 0 else 0.0

        # Compute sampling stride (e.g. 30 fps / 1 fps = sample every 30th frame)
        stride = max(1, int(round(orig_fps / self.sample_fps)))
        estimated_samples = total_frames // stride if total_frames > 0 else 0

        logger.info(
            f"Trip duration: {format_seconds_to_timestamp(duration_sec)} ({duration_sec:.1f}s) | "
            f"Original FPS: {orig_fps:.2f} | Total Frames: {total_frames} | "
            f"Sampling every {stride} frame(s) (~{self.sample_fps} FPS, ~{estimated_samples} total frames to evaluate)"
        )

        incidents: List[MobileIncident] = []
        incident_counter = 0

        # State tracking for active incident
        in_incident = False
        consecutive_positives = 0
        incident_start_time = 0.0
        last_positive_time = 0.0
        peak_confidence = 0.0
        peak_time = 0.0
        peak_frame: Optional[np.ndarray] = None

        frame_batch: List[np.ndarray] = []
        time_batch: List[float] = []

        start_wall_time = time.time()
        current_frame_idx = 0
        samples_processed = 0

        pbar = tqdm(total=estimated_samples, desc="Analyzing Trip Video", unit="frame")

        def flush_batch():
            nonlocal in_incident, consecutive_positives, incident_start_time
            nonlocal last_positive_time, peak_confidence, peak_time, peak_frame
            nonlocal incident_counter, samples_processed

            if not frame_batch:
                return

            batch_results = self.predictor.predict_batch(frame_batch)

            for i, (label, conf, probs) in enumerate(batch_results):
                t_sec = time_batch[i]
                phone_prob = probs.get("using_phone", 0.0)
                is_phone = phone_prob >= self.confidence_threshold

                if is_phone:
                    consecutive_positives += 1
                    last_positive_time = t_sec

                    # If not yet flagged as active incident, check threshold
                    if not in_incident and consecutive_positives >= self.min_consecutive_detections:
                        in_incident = True
                        incident_counter += 1
                        incident_start_time = t_sec - ((consecutive_positives - 1) / self.sample_fps)
                        incident_start_time = max(0.0, incident_start_time)
                        peak_confidence = phone_prob
                        peak_time = t_sec
                        peak_frame = frame_batch[i].copy()
                        logger.info(
                            f"  [!] Incident #{incident_counter} triggered at "
                            f"{format_seconds_to_timestamp(incident_start_time)} (prob={phone_prob:.2f})"
                        )

                    elif in_incident:
                        # Update peak frame if confidence is higher
                        if phone_prob > peak_confidence:
                            peak_confidence = phone_prob
                            peak_time = t_sec
                            peak_frame = frame_batch[i].copy()

                else:
                    consecutive_positives = 0
                    # Check if active incident has cooled down
                    if in_incident and (t_sec - last_positive_time >= self.cooldown_seconds):
                        # Finalize incident
                        _finalize_incident(end_time=last_positive_time)

            samples_processed += len(frame_batch)
            pbar.update(len(frame_batch))
            if progress_callback and total_frames > 0:
                progress_callback(min(1.0, current_frame_idx / total_frames))

            frame_batch.clear()
            time_batch.clear()

        def _finalize_incident(end_time: float):
            nonlocal in_incident, incident_counter, peak_frame, peak_confidence, peak_time

            duration = max(1.0 / self.sample_fps, end_time - incident_start_time)
            start_ts = format_seconds_to_timestamp(incident_start_time)
            end_ts = format_seconds_to_timestamp(end_time)
            peak_ts = format_seconds_to_timestamp(peak_time)

            clean_ts = peak_ts.replace(":", "-")
            ss_filename = f"incident_{incident_counter:03d}_{clean_ts}.jpg"
            ss_path = screenshots_dir / ss_filename

            # Save annotated screenshot
            if peak_frame is not None:
                annotate_screenshot(
                    image=peak_frame,
                    timestamp_str=peak_ts,
                    confidence=peak_confidence,
                    incident_id=incident_counter,
                    output_path=ss_path
                )
            else:
                logger.warning(f"Incident #{incident_counter} has no peak frame captured.")

            raw_path_str = None
            if self.save_raw_ss and peak_frame is not None:
                raw_ss_path = screenshots_dir / f"incident_{incident_counter:03d}_{clean_ts}_raw.jpg"
                cv2.imwrite(str(raw_ss_path), peak_frame)
                raw_path_str = str(raw_ss_path.relative_to(output_dir))

            inc = MobileIncident(
                incident_id=incident_counter,
                start_time_seconds=round(incident_start_time, 2),
                end_time_seconds=round(end_time, 2),
                duration_seconds=round(duration, 2),
                start_timestamp=start_ts,
                end_timestamp=end_ts,
                peak_timestamp=peak_ts,
                peak_confidence=round(peak_confidence, 4),
                screenshot_path=str(ss_path.relative_to(output_dir)),
                raw_screenshot_path=raw_path_str
            )
            incidents.append(inc)
            logger.info(
                f"  [*] Incident #{incident_counter} committed: {start_ts} -> {end_ts} "
                f"(Duration: {duration:.1f}s, Peak Conf: {peak_confidence * 100:.1f}%). Screenshot saved."
            )

            # Reset state
            in_incident = False
            peak_frame = None
            peak_confidence = 0.0

        # Video ingestion loop
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame_idx % stride == 0:
                timestamp_sec = current_frame_idx / orig_fps
                frame_batch.append(frame)
                time_batch.append(timestamp_sec)

                if len(frame_batch) >= self.batch_size:
                    flush_batch()

            current_frame_idx += 1

        # Flush remaining frames
        if frame_batch:
            flush_batch()

        # If incident active at the end of video, commit it
        if in_incident:
            _finalize_incident(end_time=last_positive_time or duration_sec)

        cap.release()
        pbar.close()

        total_wall_time = time.time() - start_wall_time
        total_distraction = sum(inc.duration_seconds for inc in incidents)
        distraction_pct = (total_distraction / duration_sec * 100.0) if duration_sec > 0 else 0.0

        # Risk level determination
        if len(incidents) == 0:
            risk = "SAFE / EXCELLENT"
        elif len(incidents) <= 2 and total_distraction < 15.0:
            risk = "LOW RISK"
        elif len(incidents) <= 5 or total_distraction < 60.0:
            risk = "MODERATE RISK"
        else:
            risk = "HIGH RISK / SEVERE"

        report = TripReport(
            video_path=str(video_path),
            video_duration_seconds=round(duration_sec, 2),
            video_duration_formatted=format_seconds_to_timestamp(duration_sec),
            original_fps=round(orig_fps, 2),
            total_frames=total_frames,
            sampled_frames_processed=samples_processed,
            processing_time_seconds=round(total_wall_time, 2),
            total_incidents=len(incidents),
            total_distraction_seconds=round(total_distraction, 2),
            distraction_percentage=round(distraction_pct, 2),
            safety_risk_level=risk,
            incidents=incidents
        )

        # Export JSON report
        json_report_path = output_dir / "trip_report.json"
        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        # Export CSV report
        csv_report_path = output_dir / "trip_incidents.csv"
        _export_csv(incidents, csv_report_path)

        logger.info("=" * 60)
        logger.info("          TRIP ANALYSIS COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Video File           : {video_path.name}")
        logger.info(f"Trip Duration        : {report.video_duration_formatted} ({duration_sec:.1f}s)")
        logger.info(f"Processing Time      : {total_wall_time:.1f}s (Speed: {duration_sec / max(0.1, total_wall_time):.1f}x real-time)")
        logger.info(f"Mobile Phone Incidents: {len(incidents)}")
        logger.info(f"Total Phone Usage    : {total_distraction:.1f}s ({distraction_pct:.1f}% of trip)")
        logger.info(f"Safety Assessment    : {risk}")
        logger.info(f"Report JSON          : {json_report_path}")
        logger.info(f"Screenshots Saved To : {screenshots_dir}")
        logger.info("=" * 60)

        return report


def _export_csv(incidents: List[MobileIncident], csv_path: Path):
    """Exports tabular list of incidents."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Incident ID",
            "Start Timestamp",
            "End Timestamp",
            "Peak Timestamp",
            "Duration (sec)",
            "Peak Confidence (%)",
            "Screenshot Path"
        ])
        for inc in incidents:
            writer.writerow([
                inc.incident_id,
                inc.start_timestamp,
                inc.end_timestamp,
                inc.peak_timestamp,
                inc.duration_seconds,
                f"{inc.peak_confidence * 100:.1f}",
                inc.screenshot_path
            ])
