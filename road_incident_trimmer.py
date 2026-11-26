# -*- coding: utf-8 -*-
"""
Road Dashcam 2-Minute Incident Video Trimmer
Rules enforced per specification:
  - Routine 'Sudden Braking' (HARSH_BRAKE) and 'Rapid Acceleration' (RAPID_ACCEL)
    DO NOT trigger video trimming (they only deduct points in the driver scorecard).
  - A 2-Minute Video Clip (120 seconds: 60s before to 60s after the event) is ONLY
    trimmed from the road video when one of these 3 Real Incidents occurs:
      1. DANGEROUS_OVERTAKE  (Close cut / Ultrasonic proximity < 80 cm at speed)
      2. ACCIDENT_COLLISION  (Collision impact / High G-Force spike)
      3. SHARP_TURN          (Aggressive cornering / High gyroscope yaw rate gz)
  - Maintains a 7-Clip FIFO Cloud Storage Buffer (oldest clip is deleted when > 7).
"""

import os
import datetime
from pathlib import Path
from typing import Dict, List, Optional
import cv2

TRIM_ELIGIBLE_EVENTS = {
    "DANGEROUS_OVERTAKE",
    "TAILGATING",
    "ACCIDENT_COLLISION",
    "COLLISION",
    "ACCIDENT",
    "SHARP_TURN"
}

NON_TRIM_SCORE_ONLY_EVENTS = {
    "HARSH_BRAKE",
    "HARSH_BRAKING",
    "SUDDEN_BRAKE",
    "RAPID_ACCEL",
    "RAPID_ACCELERATION"
}

MAX_FIFO_CLIPS = 7
CLIP_DURATION_SECONDS = 120.0  # 2-minute incident clip (-60s to +60s)


def is_eligible_for_video_trim(event_type: str) -> bool:
    """Returns True ONLY for Dangerous Overtake, Accident/Collision, or Sharp Turn."""
    ev = (event_type or "").strip().upper()
    if ev in NON_TRIM_SCORE_ONLY_EVENTS:
        return False
    return ev in TRIM_ELIGIBLE_EVENTS


def enforce_7_clip_fifo_buffer(incidents_dir: Path):
    """Keeps at most 7 trimmed incident .mp4 clips on disk (deletes oldest)."""
    incidents_dir.mkdir(parents=True, exist_ok=True)
    clips = sorted(
        incidents_dir.glob("incident_clip_*.mp4"),
        key=lambda p: p.stat().st_mtime
    )
    while len(clips) > MAX_FIFO_CLIPS:
        oldest = clips.pop(0)
        try:
            oldest.unlink()
        except Exception:
            pass


def trim_2min_incident_clip(
    source_video_path: str,
    event_offset_seconds: float,
    event_type: str,
    output_dir: str,
    trip_start_dt: Optional[datetime.datetime] = None,
    incident_id: str = "INC-801"
) -> Optional[Dict]:
    """
    Trims a 2-minute (120s) clip centered around event_offset_seconds (-60s to +60s)
    from source_video_path ONLY if event_type is a real incident
    (Dangerous Overtake, Accident/Collision, or Sharp Turn).
    """
    if not is_eligible_for_video_trim(event_type):
        return None

    src_path = Path(source_video_path)
    if not src_path.exists():
        return None

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if total_frames > 0 else 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    half_window = CLIP_DURATION_SECONDS / 2.0  # 60 seconds before & 60 seconds after
    start_sec = max(0.0, float(event_offset_seconds) - half_window)
    end_sec = min(video_duration, start_sec + CLIP_DURATION_SECONDS)
    if (end_sec - start_sec) < CLIP_DURATION_SECONDS and start_sec > 0:
        start_sec = max(0.0, end_sec - CLIP_DURATION_SECONDS)

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    if trip_start_dt is None:
        trip_start_dt = datetime.datetime.now().replace(hour=8, minute=15, second=0)
    real_time_12hr = (trip_start_dt + datetime.timedelta(seconds=float(event_offset_seconds))).strftime("%I:%M:%S %p")

    clean_type = event_type.strip().lower().replace(" ", "_")
    clip_filename = f"incident_clip_{incident_id.lower()}_{clean_type}.mp4"
    clip_path = out_dir / clip_filename

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    curr_frame = start_frame

    while curr_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        curr_frame += 1

    cap.release()
    writer.release()

    enforce_7_clip_fifo_buffer(out_dir)

    return {
        "id": incident_id,
        "eventType": event_type.upper(),
        "timestamp12Hr": real_time_12hr,
        "clipStartSec": round(start_sec, 1),
        "clipEndSec": round(end_sec, 1),
        "clipDurationSec": round(end_sec - start_sec, 1),
        "clipUrl": f"/uploads/incidents/{clip_filename}"
    }
