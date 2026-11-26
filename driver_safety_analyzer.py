# -*- coding: utf-8 -*-
"""
Unified Driver Cabin Video AI Analyzer
Runs BOTH models in a single pass on one Driver Cabin Video stream (trip_driver.mp4):
  1. Mobile Phone Usage Detector (PyTorch MobileNetV3-Large: weights/best_mobile_detector.pth) @ ~1.5 FPS
  2. Driver Drowsiness & Fatigue Detector @ ~6.0 FPS:
       - YuNet ONNX face detector (weights/drowsiness/yunet.onnx)
       - MediaPipe 478-landmark Face Landmarker (weights/drowsiness/face_landmarker.task)
       - Eye CNN v2 ONNX (weights/drowsiness/eye_cnn_v2.onnx, 78,993 params, horizontal-flip TTA)
       - Geometric EAR + CNN agreement fusion + Sideways-turn & Looking-down Head-Pitch filters
       - 60-second PERCLOS (>15%) + Severe Eye Closure (>=2.0s)
       - Sustained Yawn (MAR > 0.60 for >=2.0s) secondary fatigue warning (filters talking & coughing)
"""

import datetime
import json
import os
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DROWSY_DIR = BASE_DIR / "weights" / "drowsiness"

EYE_IMG_SIZE = 64
EYE_IDX = {
    "left": [33, 160, 158, 133, 153, 144],
    "right": [362, 385, 387, 263, 373, 380],
}
# MediaPipe inner-lip landmarks for Mouth Aspect Ratio (MAR)
MOUTH_TOP_IDX = 13
MOUTH_BOT_IDX = 14
MOUTH_LEFT_IDX = 78
MOUTH_RIGHT_IDX = 308

DETECT_WIDTH = 640
DETECT_CONF = 0.60
BOX_PAD = 0.50
TURN_UNKNOWN_RATIO = 0.55


def compute_real_world_12hr_timestamp(trip_start_dt: datetime.datetime, offset_seconds: float) -> str:
    event_dt = trip_start_dt + datetime.timedelta(seconds=float(offset_seconds))
    return event_dt.strftime("%I:%M:%S %p")


def save_annotated_evidence_image(
    frame: np.ndarray,
    violation_type: str,
    real_time_12hr: str,
    confidence: float,
    output_path: Path,
) -> str:
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    overlay = annotated.copy()
    banner_h = max(68, int(h * 0.13))
    v_up = violation_type.upper()
    if "MOBILE" in v_up:
        banner_color = (20, 20, 175)
    elif "YAWN" in v_up or "FATIGUE" in v_up:
        banner_color = (0, 130, 210)
    else:
        banner_color = (0, 85, 200)

    cv2.rectangle(overlay, (0, 0), (w, banner_h), banner_color, -1)
    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.52, min(w / 960.0, 0.82))

    title = f"EVIDENCE: {v_up}"
    sub = f"TIME: {real_time_12hr} (12-HR) | CONFIDENCE: {confidence * 100:.1f}%"

    cv2.putText(annotated, title, (20, int(banner_h * 0.44)), font, scale, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated, sub, (20, int(banner_h * 0.82)), font, scale * 0.82, (215, 245, 255), 1, cv2.LINE_AA)
    cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), banner_color, 3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 94])
    return str(output_path)


def _ear_from_points(p: np.ndarray) -> float:
    return float(
        (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4]))
        / (2.0 * np.linalg.norm(p[0] - p[3]) + 1e-6)
    )


def _mar_from_landmarks(pts_all: np.ndarray) -> float:
    vert = np.linalg.norm(pts_all[MOUTH_TOP_IDX] - pts_all[MOUTH_BOT_IDX])
    horiz = np.linalg.norm(pts_all[MOUTH_LEFT_IDX] - pts_all[MOUTH_RIGHT_IDX])
    return float(vert / (horiz + 1e-6))


def _eye_crop(gray: np.ndarray, p: np.ndarray, margin: float = 0.5) -> np.ndarray:
    x0, y0, x1, y1 = p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = max(x1 - x0, y1 - y0) * (0.5 + margin)
    h, w = gray.shape[:2]
    a, b = int(max(0, cy - half)), int(min(h, cy + half))
    c, d = int(max(0, cx - half)), int(min(w, cx + half))
    patch = gray[a:b, c:d]
    if patch.size == 0:
        return np.zeros((EYE_IMG_SIZE, EYE_IMG_SIZE), np.uint8)
    resized = cv2.resize(patch, (EYE_IMG_SIZE, EYE_IMG_SIZE), interpolation=cv2.INTER_AREA)
    # Mild CLAHE only when eye patch is dim (e.g., early morning cabin lighting)
    if float(np.mean(resized)) < 75.0:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        resized = clahe.apply(resized)
    return resized


class UnifiedDriverSafetyAnalyzer:
    """
    Single-pass dual-FPS video analyzer running:
      - Mobile Phone Usage Detection (MobileNetV3-Large PyTorch) @ ~1.5 FPS
      - Driver Drowsiness & Yawn Fatigue Detection (YuNet + MediaPipe + Eye CNN v2 ONNX) @ ~6.0 FPS
    """

    def __init__(
        self,
        mobile_weights_path: Optional[str] = None,
        sample_fps: float = 1.5,
        drowsy_sample_fps: float = 6.0,
        mobile_conf_thresh: float = 0.72,
        drowsy_closed_frames_thresh: int = 2,
        cooldown_seconds: float = 15.0,
    ):
        self.sample_fps = float(sample_fps)
        self.drowsy_sample_fps = float(drowsy_sample_fps)
        self.mobile_conf_thresh = float(mobile_conf_thresh)
        self.drowsy_closed_frames_thresh = int(drowsy_closed_frames_thresh)
        self.cooldown_seconds = float(cooldown_seconds)

        # Configurable drowsiness thresholds (env-overridable per camera calibration)
        self.drowsy_rule = os.environ.get("DROWSY_RULE", "agreement").strip().lower()
        self.ear_thresh = float(os.environ.get("DROWSY_EAR_THRESH", "0.20"))
        self.ear_soft = float(os.environ.get("DROWSY_EAR_SOFT", "0.25"))
        self.cnn_open_thresh = float(os.environ.get("DROWSY_CNN_THRESH", "0.50"))
        self.use_tta = True
        self.mar_yawn_thresh = float(os.environ.get("DROWSY_MAR_THRESH", "0.60"))
        self.yawn_min_sec = float(os.environ.get("DROWSY_YAWN_SEC", "2.0"))
        self.severe_closure_sec = float(os.environ.get("DROWSY_SEVERE_SEC", "2.0"))
        self.perclos_thresh_pct = float(os.environ.get("DROWSY_PERCLOS_PCT", "15.0"))

        cfg_path = DROWSY_DIR / "eye_cnn_v2_threshold.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                self.use_tta = bool(cfg.get("tta", True))
            except Exception:
                pass

        # 1. MobileNetV3-Large Mobile Detector (PyTorch)
        self.mobile_predictor = None
        try:
            from inference.predictor import MobileDetectorPredictor
            self.mobile_predictor = MobileDetectorPredictor(weights_path=mobile_weights_path)
        except Exception:
            self.mobile_predictor = None

        # 2. Drowsiness Pipeline: YuNet + MediaPipe Face Landmarker + Eye CNN v2 ONNX
        self.yunet = None
        self.landmarker = None
        self.eye_cnn_sess = None
        self.eye_cnn_input = None
        self.mp_module = None
        self.drowsy_backend = "unavailable"

        yunet_path = DROWSY_DIR / "yunet.onnx"
        landmarker_path = DROWSY_DIR / "face_landmarker.task"
        eye_onnx_path = DROWSY_DIR / "eye_cnn_v2.onnx"

        try:
            import mediapipe as mp
            import onnxruntime as ort
            from mediapipe.tasks.python import BaseOptions, vision

            if yunet_path.exists() and landmarker_path.exists() and eye_onnx_path.exists():
                self.yunet = cv2.FaceDetectorYN.create(
                    str(yunet_path), "", (DETECT_WIDTH, DETECT_WIDTH), DETECT_CONF, 0.3, 5000
                )
                self.landmarker = vision.FaceLandmarker.create_from_options(
                    vision.FaceLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=str(landmarker_path)),
                        running_mode=vision.RunningMode.IMAGE,
                        num_faces=1,
                    )
                )
                self.eye_cnn_sess = ort.InferenceSession(
                    str(eye_onnx_path), providers=["CPUExecutionProvider"]
                )
                self.eye_cnn_input = self.eye_cnn_sess.get_inputs()[0].name
                self.mp_module = mp
                self.drowsy_backend = "yunet+mediapipe+eye_cnn_v2_onnx"
        except Exception as e:
            print(f" [DROWSY INIT NOTE] Full ONNX/MediaPipe drowsiness unavailable ({e}).")

    def _evaluate_drowsiness_frame(self, frame: np.ndarray, pitch_hist: deque) -> Dict:
        """
        Evaluate a single driver frame using YuNet + MediaPipe + Eye CNN v2 ONNX.
        Returns dict with keys: usable, closed, ear, p_open, mar, looking_down, turned, reason
        """
        if self.drowsy_backend != "yunet+mediapipe+eye_cnn_v2_onnx":
            return {"usable": False, "closed": False, "looking_down": False, "turned": False, "mar": 0.0, "reason": "no_model"}

        H, W = frame.shape[:2]
        scale = DETECT_WIDTH / float(max(W, 1))
        small = cv2.resize(frame, (DETECT_WIDTH, max(1, int(H * scale))))
        self.yunet.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.yunet.detect(small)

        if faces is None or len(faces) == 0:
            return {"usable": False, "closed": False, "looking_down": False, "turned": False, "mar": 0.0, "reason": "no_face"}

        # Widest face = nearest (the driver)
        b = max(faces, key=lambda q: q[2])
        x, y, fw, fh = b[:4] / scale
        pad = fw * BOX_PAD
        bx1 = int(max(0, x - pad))
        by1 = int(max(0, y - pad))
        bx2 = int(min(W, x + fw + pad))
        by2 = int(min(H, y + fh + pad))
        crop = frame[by1:by2, bx1:bx2]
        if crop.size == 0:
            return {"usable": False, "closed": False, "looking_down": False, "turned": False, "mar": 0.0, "reason": "empty_crop"}

        mp_img = self.mp_module.Image(
            image_format=self.mp_module.ImageFormat.SRGB,
            data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
        )
        res = self.landmarker.detect(mp_img)
        if not res.face_landmarks:
            return {"usable": False, "closed": False, "looking_down": False, "turned": False, "mar": 0.0, "reason": "no_landmarks"}

        bh, bw = crop.shape[:2]
        pts_all = np.array([[p.x * bw, p.y * bh] for p in res.face_landmarks[0]], dtype=np.float32) + np.array([bx1, by1], dtype=np.float32)

        # Sideways head turn check
        pts = {s: pts_all[idx] for s, idx in EYE_IDX.items()}
        widths = {s: float(np.linalg.norm(p[0] - p[3])) for s, p in pts.items()}
        ratio = min(widths.values()) / max(max(widths.values()), 1e-6)
        if ratio < TURN_UNKNOWN_RATIO:
            return {"usable": False, "closed": False, "looking_down": False, "turned": True, "mar": 0.0, "reason": "head_turned"}

        # Head pitch (looking down at floor / speedometer / phone)
        eye_mid = (pts_all[33] + pts_all[263]) / 2.0
        inter = float(np.linalg.norm(pts_all[33] - pts_all[263]))
        pitch_val = float((pts_all[152][1] - eye_mid[1]) / max(inter, 1e-6))
        pitch_hist.append(pitch_val)
        pitch_med = float(np.median(pitch_hist)) if len(pitch_hist) >= 5 else pitch_val
        pitch_rel = pitch_val / max(pitch_med, 1e-6)
        looking_down = pitch_rel < 0.88

        mar = _mar_from_landmarks(pts_all)

        if looking_down:
            return {
                "usable": False,
                "closed": False,
                "looking_down": True,
                "turned": False,
                "mar": mar,
                "pitch_rel": pitch_rel,
                "reason": "looking_down",
            }

        # Frontal eye EAR + Eye CNN v2 ONNX (with horizontal-flip TTA)
        frontal = "left" if widths["left"] >= widths["right"] else "right"
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crop_l = _eye_crop(gray, pts["left"])
        crop_r = _eye_crop(gray, pts["right"])

        # Dark sunglasses / completely occluded eye patch guard
        frontal_crop = crop_l if frontal == "left" else crop_r
        if float(np.std(frontal_crop)) < 6.0 and float(np.mean(frontal_crop)) < 25.0:
            return {"usable": False, "closed": False, "looking_down": False, "turned": False, "mar": mar, "reason": "occluded_dark"}

        ear = _ear_from_points(pts[frontal])
        X = np.stack([crop_l, crop_r], axis=0)[..., None].astype(np.float32)
        if self.use_tta:
            X = np.concatenate([X, X[:, :, ::-1, :]], axis=0)
        preds = np.asarray(self.eye_cnn_sess.run(None, {self.eye_cnn_input: X})[0]).ravel()
        if self.use_tta:
            p_open = float((preds[0] + preds[2]) / 2.0 if frontal == "left" else (preds[1] + preds[3]) / 2.0)
        else:
            p_open = float(preds[0] if frontal == "left" else preds[1])

        ear_says = ear < self.ear_thresh
        cnn_says = p_open < self.cnn_open_thresh
        if self.drowsy_rule == "ear_leads":
            closed = ear_says or (cnn_says and ear < self.ear_soft)
        elif self.drowsy_rule == "cnn_leads":
            closed = cnn_says or (ear_says and p_open < 0.95)
        else:
            # Default calibrated "agreement" rule: both EAR and Eye CNN must agree
            closed = ear_says and cnn_says

        return {
            "usable": True,
            "closed": bool(closed),
            "ear": ear,
            "p_open": p_open,
            "mar": mar,
            "looking_down": False,
            "turned": False,
            "pitch_rel": pitch_rel,
            "reason": "closed" if closed else "open",
        }

    def analyze_driver_video(
        self,
        video_path: str,
        evidence_output_dir: str,
        trip_start_time_str: Optional[str] = None,
    ) -> Dict:
        """
        Single-pass dual-FPS analysis on driver cabin video:
          - Drowsiness (YuNet + MediaPipe + Eye CNN v2 ONNX + PERCLOS + Sustained Yawn) @ 6.0 FPS
          - Mobile Phone Usage (MobileNetV3-Large PyTorch) @ 1.5 FPS
        """
        video_file = Path(video_path)
        evidence_dir = Path(evidence_output_dir)
        evidence_dir.mkdir(parents=True, exist_ok=True)

        if trip_start_time_str:
            try:
                trip_start_dt = datetime.datetime.strptime(trip_start_time_str, "%Y-%m-%d %I:%M:%S %p")
            except ValueError:
                try:
                    trip_start_dt = datetime.datetime.strptime(trip_start_time_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    trip_start_dt = datetime.datetime.now().replace(hour=8, minute=15, second=0)
        else:
            trip_start_dt = datetime.datetime.now().replace(hour=8, minute=15, second=0)

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            return {"success": False, "error": f"Cannot open driver video: {video_path}", "events": []}

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        drowsy_stride = max(1, int(round(fps / self.drowsy_sample_fps)))
        mobile_stride = max(1, int(round(fps / self.sample_fps)))
        dt_drowsy = drowsy_stride / fps

        events: List[Dict] = []
        last_mobile_time = -999.0
        last_drowsy_time = -999.0
        last_yawn_alert_time = -999.0
        consec_mobile = 0

        # Drowsiness & yawn temporal state
        perclos_window = deque(maxlen=max(10, int(round(60.0 / dt_drowsy))))
        pitch_hist = deque(maxlen=max(20, int(round(20.0 / dt_drowsy))))
        closed_run_sec = 0.0
        yawn_run_sec = 0.0
        yawn_timestamps: List[float] = []
        usable_frames = 0
        sampled_drowsy_frames = 0
        looking_down_frames = 0
        max_closure_sec = 0.0
        peak_perclos = 0.0
        last_looking_down = False

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t_sec = frame_idx / fps

            # 1. Drowsiness & Yawn Pass (~6.0 FPS)
            if frame_idx % drowsy_stride == 0:
                sampled_drowsy_frames += 1
                dr = self._evaluate_drowsiness_frame(frame, pitch_hist)
                last_looking_down = bool(dr.get("looking_down", False))
                if last_looking_down:
                    looking_down_frames += 1

                if dr.get("usable"):
                    usable_frames += 1
                    is_closed = bool(dr["closed"])
                    perclos_window.append(1 if is_closed else 0)
                    if is_closed:
                        closed_run_sec += dt_drowsy
                        if closed_run_sec > max_closure_sec:
                            max_closure_sec = closed_run_sec
                    else:
                        closed_run_sec = 0.0

                    cur_perclos = (100.0 * sum(perclos_window) / len(perclos_window)) if len(perclos_window) >= int(10.0 / dt_drowsy) else 0.0
                    if cur_perclos > peak_perclos:
                        peak_perclos = cur_perclos

                    severe_eye = closed_run_sec >= self.severe_closure_sec
                    high_perclos = cur_perclos > self.perclos_thresh_pct

                    if (severe_eye or high_perclos) and (t_sec - last_drowsy_time) >= self.cooldown_seconds:
                        last_drowsy_time = t_sec
                        ts_12hr = compute_real_world_12hr_timestamp(trip_start_dt, t_sec)
                        ev_id = len(events) + 1
                        img_name = f"evidence_drowsiness_{ev_id:03d}_{int(t_sec)}s.jpg"
                        img_path = evidence_dir / img_name
                        conf_score = max(0.75, min(0.99, 1.0 - float(dr.get("p_open", 0.1))))
                        reason_label = (
                            f"Driver Drowsiness (Eyes Closed {closed_run_sec:.1f}s)"
                            if severe_eye
                            else f"Driver Drowsiness (PERCLOS {cur_perclos:.1f}%)"
                        )
                        save_annotated_evidence_image(
                            frame=frame,
                            violation_type="Driver Drowsiness / Sleeping",
                            real_time_12hr=ts_12hr,
                            confidence=conf_score,
                            output_path=img_path,
                        )
                        events.append({
                            "id": f"DRV-EV-{ev_id:03d}",
                            "time": ts_12hr,
                            "videoOffsetSec": round(t_sec, 1),
                            "type": reason_label,
                            "confidence": round(conf_score * 100, 1),
                            "impact": "-10 pts",
                            "severity": "Critical",
                            "evidenceUrl": f"/uploads/evidence/{img_name}",
                        })

                    # Sustained Yawn (MAR > 0.60 continuously for >= 2.0s; excludes talking & coughing)
                    if float(dr.get("mar", 0.0)) >= self.mar_yawn_thresh:
                        yawn_run_sec += dt_drowsy
                    else:
                        if yawn_run_sec >= self.yawn_min_sec:
                            yawn_timestamps.append(t_sec)
                            # Keep yawns within last 5 minutes
                            yawn_timestamps = [yt for yt in yawn_timestamps if (t_sec - yt) <= 300.0]
                            if len(yawn_timestamps) >= 3 and (t_sec - last_yawn_alert_time) >= 60.0:
                                last_yawn_alert_time = t_sec
                                ts_12hr = compute_real_world_12hr_timestamp(trip_start_dt, t_sec)
                                ev_id = len(events) + 1
                                img_name = f"evidence_fatigue_yawn_{ev_id:03d}_{int(t_sec)}s.jpg"
                                img_path = evidence_dir / img_name
                                save_annotated_evidence_image(
                                    frame=frame,
                                    violation_type="Driver Fatigue (Repeated Yawning)",
                                    real_time_12hr=ts_12hr,
                                    confidence=0.85,
                                    output_path=img_path,
                                )
                                events.append({
                                    "id": f"DRV-EV-{ev_id:03d}",
                                    "time": ts_12hr,
                                    "videoOffsetSec": round(t_sec, 1),
                                    "type": f"Driver Fatigue ({len(yawn_timestamps)} Sustained Yawns)",
                                    "confidence": 85.0,
                                    "impact": "-5 pts",
                                    "severity": "Warning",
                                    "evidenceUrl": f"/uploads/evidence/{img_name}",
                                })
                        yawn_run_sec = 0.0
                else:
                    closed_run_sec = 0.0
                    yawn_run_sec = 0.0

            # 2. Mobile Phone Detection Pass (~1.5 FPS)
            if frame_idx % mobile_stride == 0:
                is_mobile = False
                mobile_conf = 0.0
                if self.mobile_predictor is not None:
                    _, _, probs = self.mobile_predictor.predict_single(frame)
                    mobile_conf = float(probs.get("using_phone", 0.0))
                    # Cross-signal synergy: looking down + phone confidence >= 0.55 confirms texting
                    eff_thresh = 0.55 if last_looking_down else self.mobile_conf_thresh
                    is_mobile = mobile_conf >= eff_thresh

                if is_mobile:
                    consec_mobile += 1
                    if consec_mobile >= 2 and (t_sec - last_mobile_time) >= self.cooldown_seconds:
                        last_mobile_time = t_sec
                        ts_12hr = compute_real_world_12hr_timestamp(trip_start_dt, t_sec)
                        ev_id = len(events) + 1
                        img_name = f"evidence_mobile_{ev_id:03d}_{int(t_sec)}s.jpg"
                        img_path = evidence_dir / img_name
                        save_annotated_evidence_image(
                            frame=frame,
                            violation_type="Mobile Phone Usage Detected",
                            real_time_12hr=ts_12hr,
                            confidence=mobile_conf,
                            output_path=img_path,
                        )
                        events.append({
                            "id": f"DRV-EV-{ev_id:03d}",
                            "time": ts_12hr,
                            "videoOffsetSec": round(t_sec, 1),
                            "type": "Mobile Phone Usage",
                            "confidence": round(mobile_conf * 100, 1),
                            "impact": "-10 pts",
                            "severity": "Violation",
                            "evidenceUrl": f"/uploads/evidence/{img_name}",
                        })
                else:
                    consec_mobile = 0

            frame_idx += 1

        cap.release()
        coverage_pct = round(100.0 * usable_frames / max(sampled_drowsy_frames, 1), 1)
        return {
            "success": True,
            "videoAnalyzed": video_file.name,
            "tripStartTime": trip_start_dt.strftime("%I:%M:%S %p"),
            "drowsyBackend": self.drowsy_backend,
            "drowsyMetrics": {
                "sampledFrames": sampled_drowsy_frames,
                "usableFrames": usable_frames,
                "coveragePct": coverage_pct,
                "lookingDownExcludedFrames": looking_down_frames,
                "maxClosureSec": round(max_closure_sec, 2),
                "peakPerclosPct": round(peak_perclos, 1),
                "sustainedYawns": len(yawn_timestamps),
            },
            "totalEvents": len(events),
            "events": events,
        }
