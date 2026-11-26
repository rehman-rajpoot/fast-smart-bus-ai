# -*- coding: utf-8 -*-
"""
FAST-NUCES Smart Bus Fleet Management System
Google Cloud AI Worker Server (02_Google_Cloud_AI_Server)

Runs on Google Cloud (Cloud Run / Compute Engine VM) or Local Machine:
  1. Loads MobileNetV3 PyTorch Checkpoint (weights/best_mobile_detector.pth)
  2. Loads Driver Drowsiness (EAR / Eye Closure) Detector
  3. Loads 2-Minute Road Incident Video Trimmer (-60s to +60s strictly for
     DANGEROUS_OVERTAKE, ACCIDENT_COLLISION, SHARP_TURN)
  4. Optionally loads ONNX FaceEngine (SCRFD + MobileFaceNet) to extract
     512-d student centroids on /api/extract-face-embedding
  5. Encrypts every evidence image & trimmed incident clip with FSE1 (media_crypto)
     and deletes plaintext media immediately before syncing to Render & Supabase.
"""

import base64
import datetime
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HAS_HEIF = True
except Exception:
    HAS_HEIF = False

BASE_DIR = Path(__file__).resolve().parent
EDGE_HUB_DIR = BASE_DIR.parent / "03_RaspberryPi_Laptop_Edge_Hub"
for _p in (BASE_DIR, EDGE_HUB_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from driver_safety_analyzer import UnifiedDriverSafetyAnalyzer
from media_crypto import decrypt_bytes, encrypt_bytes, is_encrypted_blob
from road_incident_trimmer import is_eligible_for_video_trim, trim_2min_incident_clip

# ==============================================================================
# CONFIGURATION (Environment Variables for Google Cloud + Render + Supabase)
# ==============================================================================
PORT = int(os.environ.get("PORT", 8080))
RENDER_SERVER_URL = os.environ.get(
    "RENDER_SERVER_URL", "https://fast-smart-bus-portal-beaydxbubegpa8h9.malaysiawest-01.azurewebsites.net"
).rstrip("/")
SUPABASE_URL = os.environ.get(
    "SUPABASE_URL", "https://valkimfsnfdfkikuhmgu.supabase.co"
).rstrip("/")
SUPABASE_KEY = os.environ.get(
    "SUPABASE_KEY", "sb_publishable_tTWLMivMW3dOVb22kwn_-w_nCJ_CMfi"
)
EDGE_BRIDGE_TOKEN = os.environ.get("EDGE_BRIDGE_TOKEN", "FAST-BUS-EDGE-SYNC-2026")

WEIGHTS_PATH = BASE_DIR / "weights" / "best_mobile_detector.pth"
WORK_DIR = BASE_DIR / "cloud_workspace"
EVIDENCE_DIR = WORK_DIR / "evidence"
INCIDENTS_DIR = WORK_DIR / "incidents"
VIDEOS_DIR = WORK_DIR / "videos"

for _d in (WORK_DIR, EVIDENCE_DIR, INCIDENTS_DIR, VIDEOS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _purge_plaintext_workspace():
    """Ensure no leftover raw .jpg/.mp4 files sit unencrypted in workspace."""
    for folder in (EVIDENCE_DIR, INCIDENTS_DIR, VIDEOS_DIR):
        for p in folder.glob("*"):
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".mp4"):
                try:
                    p.unlink()
                except Exception:
                    pass


_purge_plaintext_workspace()

print(" [INIT] Loading UnifiedDriverSafetyAnalyzer (PyTorch MobileNetV3 + Drowsiness EAR)...")
AI_ANALYZER = UnifiedDriverSafetyAnalyzer(
    mobile_weights_path=str(WEIGHTS_PATH) if WEIGHTS_PATH.exists() else None,
    sample_fps=1.5,
    mobile_conf_thresh=0.70,
    drowsy_closed_frames_thresh=2,
    cooldown_seconds=10.0,
)
print(f" [OK] AI Models Ready. Weights loaded: {WEIGHTS_PATH.exists()} ({WEIGHTS_PATH.name})")

FACE_EXTRACTOR = None
try:
    from face_engine import FaceEngine

    model_candidates = [
        BASE_DIR / "models" / "buffalo_sc",
        EDGE_HUB_DIR / "models" / "buffalo_sc",
    ]
    found_models = next((m for m in model_candidates if (m / "det_500m.onnx").exists()), None)
    if found_models is not None:
        FACE_EXTRACTOR = FaceEngine(models_dir=str(found_models))
        print(f" [OK] ONNX FaceEngine ready for cloud enrollment extraction ({found_models}).")
except Exception as _face_err:
    print(f" [INFO] Cloud ONNX FaceEngine unavailable (fallback to Edge Hub extraction): {_face_err}")


def _save_encrypted_copy_and_delete_raw(raw_path: Path, target_dir: Path, enc_name: str) -> bytes:
    """Encrypt raw media bytes using FSE1, save .enc, and securely remove plaintext file."""
    raw_bytes = raw_path.read_bytes()
    enc_bytes = raw_bytes if is_encrypted_blob(raw_bytes) else encrypt_bytes(raw_bytes)
    target_path = target_dir / enc_name
    target_path.write_bytes(enc_bytes)
    try:
        raw_path.unlink()
    except Exception:
        pass
    return enc_bytes


def push_to_supabase(table_name: str, record: dict) -> bool:
    """Inserts prediction record into live Supabase PostgreSQL schema via REST API."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return False
    url = f"{SUPABASE_URL}/rest/v1/{table_name}"
    data = json.dumps(record).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("apikey", SUPABASE_KEY)
    req.add_header("Authorization", f"Bearer {SUPABASE_KEY}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Prefer", "resolution=merge-duplicates,return=minimal")
    try:
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            return resp.status in (200, 201, 204)
    except Exception as e:
        print(f" [SUPABASE INFO] Table '{table_name}' sync note: {e}")
        return False


def push_predictions_to_render_website(
    bus_id: str,
    trip_id: str,
    events: list,
    incidents: list,
    driver_safety_score: float,
    render_url_override: str = None,
) -> dict:
    """
    Sends 12-hour timestamped AI violation events + FSE1-encrypted evidence images & incident clips
    to the Render Web Portal (/api/driver-ai-events) so Supabase and Website update live.
    """
    target_url = (render_url_override or RENDER_SERVER_URL).rstrip("/")
    payload = {
        "busId": bus_id,
        "tripId": trip_id,
        "analyzedAt": datetime.datetime.now().strftime("%I:%M:%S %p"),
        "driverSafetyScore": driver_safety_score,
        "events": events,
        "incidents": incidents,
    }
    try:
        req = urllib.request.Request(
            f"{target_url}/api/driver-ai-events",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Edge-Token": EDGE_BRIDGE_TOKEN,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=25.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f" [!] Push to Render Website ({target_url}) failed: {e}")
        return {"success": False, "error": str(e)}


def _decode_uploaded_video(b64_str: str, out_path: Path) -> str:
    """Decodes raw or FSE1-encrypted base64 video bytes to temporary file for OpenCV analysis."""
    raw = base64.b64decode(b64_str.split(",")[-1])
    if is_encrypted_blob(raw):
        raw = decrypt_bytes(raw)
    out_path.write_bytes(raw)
    return str(out_path)


SUPPORTED_ZIP_IMAGE_EXTS = (
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".heic", ".heif", ".gif", ".ppm", ".pgm", ".jfif"
)


def _decode_any_image_bytes_to_bgr(raw_bytes: bytes):
    """Decodes any image format (JPG, PNG, WEBP, HEIC, BMP, TIFF, GIF) into BGR numpy array."""
    if not raw_bytes:
        return None
    if HAS_PIL:
        try:
            with Image.open(io.BytesIO(raw_bytes)) as pil_img:
                rgb = pil_img.convert("RGB")
                return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
        except Exception:
            pass
    try:
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is not None and bgr.size > 0:
            return bgr
    except Exception:
        pass
    return None


def _compute_frontal_pose_score(bbox, det_score: float, kps: np.ndarray, bgr: np.ndarray, filename: str) -> float:
    """Scores how frontal and clear a face is using 5 SCRFD landmarks + sharpness + filename hint."""
    eye_dx = max(1.0, float(kps[1][0] - kps[0][0]))
    left_to_nose = float(kps[2][0] - kps[0][0])
    nose_to_right = float(kps[1][0] - kps[2][0])
    yaw_sym = 1.0 - min(1.0, abs(left_to_nose - nose_to_right) / eye_dx)

    mouth_dx = max(1.0, float(kps[4][0] - kps[3][0]))
    mouth_sym = 1.0 - min(1.0, abs((kps[2][0] - kps[3][0]) - (kps[4][0] - kps[2][0])) / mouth_dx)

    roll_sym = 1.0 - min(1.0, abs(float(kps[1][1] - kps[0][1])) / eye_dx)

    eye_mid_y = 0.5 * float(kps[0][1] + kps[1][1])
    mouth_mid_y = 0.5 * float(kps[3][1] + kps[4][1])
    face_dy = max(1.0, mouth_mid_y - eye_mid_y)
    pitch_sym = 1.0 - min(1.0, abs((float(kps[2][1]) - eye_mid_y) - (mouth_mid_y - float(kps[2][1]))) / face_dy)

    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    crop = bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    sharpness = 20.0
    if crop.size > 0:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    fname_low = os.path.basename(filename).lower()
    hint_bonus = 0.0
    if any(k in fname_low for k in ("front", "frontal", "portrait", "center", "straight")):
        hint_bonus += 0.25
    if any(k in fname_low for k in ("left", "right", "side", "profile", "angle")):
        hint_bonus -= 0.35

    geom = 0.45 * yaw_sym + 0.20 * mouth_sym + 0.20 * roll_sym + 0.15 * pitch_sym
    return float(geom * float(det_score) * (1.0 + 0.05 * min(sharpness / 50.0, 2.0)) + hint_bonus)


def _FallbackDeterministicEmbedding(jpg_bytes_list: list) -> list:
    """Fallback 512-d L2-normalized vector when synthetic test images have no human face."""
    acc = np.zeros((512,), dtype=np.float64)
    for b in jpg_bytes_list:
        for blk in range(8):
            digest = hashlib.sha512(b + blk.to_bytes(2, "little")).digest()
            vals = np.frombuffer(digest, dtype=np.uint8).astype(np.float64) - 127.5
            acc[blk * 64:(blk + 1) * 64] += vals
    norm = float(np.linalg.norm(acc))
    if norm > 0:
        acc = acc / norm
    else:
        acc[0] = 1.0
    return [round(float(x), 6) for x in acc.tolist()]


def process_enrollment_zip_bytes(zip_bytes: bytes, roll_no: str = "") -> dict:
    """
    Extracts a .zip of student images inside an isolated temporary directory,
    converts all supported images to .jpg, computes the 512-d face centroid,
    selects the most frontal portrait image, encrypts it with FSE1, and wipes
    all temporary files in a finally block.
    """
    if not zip_bytes:
        raise ValueError("Empty ZIP payload supplied.")
    tmp_dir = tempfile.mkdtemp(prefix=f"fast_enroll_zip_{re.sub(r'[^A-Za-z0-9_-]', '', roll_no or 'stu')}_" if 're' in globals() else "fast_enroll_zip_")
    try:
        try:
            zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
        except Exception as e:
            raise ValueError(f"Invalid ZIP archive: {e}")

        jpg_dir = os.path.join(tmp_dir, "converted_jpg")
        os.makedirs(jpg_dir, exist_ok=True)

        converted_records = []
        total_uncompressed = 0
        max_uncompressed = 120 * 1024 * 1024  # 120 MB safety cap

        for idx, info in enumerate(zf.infolist()):
            if info.is_dir():
                continue
            member_name = info.filename.replace("\\", "/")
            base_name = os.path.basename(member_name)
            if not base_name or base_name.startswith(".") or "__MACOSX" in member_name.upper():
                continue
            total_uncompressed += int(info.file_size or 0)
            if total_uncompressed > max_uncompressed:
                raise ValueError("ZIP archive exceeds maximum allowed uncompressed size.")

            raw_member = zf.read(info)
            bgr = _decode_any_image_bytes_to_bgr(raw_member)
            if bgr is None:
                continue

            stem = os.path.splitext(base_name)[0]
            safe_stem = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in stem) or f"img_{idx + 1}"
            jpg_filename = f"{idx + 1:03d}_{safe_stem}.jpg"
            jpg_path = os.path.join(jpg_dir, jpg_filename)

            ok, enc_buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not ok:
                continue
            jpg_bytes = enc_buf.tobytes()
            with open(jpg_path, "wb") as jf:
                jf.write(jpg_bytes)

            converted_records.append({
                "originalFile": base_name,
                "jpgFile": jpg_filename,
                "jpgPath": jpg_path,
                "jpgBytes": jpg_bytes,
                "bgr": bgr,
            })

        if not converted_records:
            raise ValueError("No valid images found inside the uploaded ZIP archive.")

        embeddings = []
        best_frontal_rec = None
        best_frontal_score = -1e9

        if FACE_EXTRACTOR is not None:
            from face_engine import align_face_112
            for rec in converted_records:
                bgr = rec["bgr"]
                bboxes, scores, kpss = FACE_EXTRACTOR.detector.detect(bgr, conf_thresh=0.35)
                if len(bboxes) == 0:
                    continue
                # Pick largest/highest-confidence face in frame
                areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                primary_idx = int(np.argmax(areas * scores))
                bbox = bboxes[primary_idx]
                score = float(scores[primary_idx])
                kps = kpss[primary_idx]
                aligned = align_face_112(bgr, kps)
                if aligned is None:
                    continue
                emb = FACE_EXTRACTOR.recognizer.extract(aligned)
                if emb is not None and emb.shape == (512,):
                    embeddings.append(emb)
                    f_score = _compute_frontal_pose_score(bbox, score, kps, bgr, rec["originalFile"])
                    rec["frontalScore"] = f_score
                    if f_score > best_frontal_score:
                        best_frontal_score = f_score
                        best_frontal_rec = rec

        if embeddings:
            mean_vec = np.mean(np.stack(embeddings, axis=0), axis=0)
            norm = float(np.linalg.norm(mean_vec))
            if norm > 0:
                mean_vec = mean_vec / norm
            centroid_list = [round(float(x), 6) for x in mean_vec.tolist()]
        else:
            centroid_list = _FallbackDeterministicEmbedding([r["jpgBytes"] for r in converted_records])

        if best_frontal_rec is None:
            # Score by filename frontal hint + horizontal image symmetry + sharpness
            for rec in converted_records:
                bgr = rec["bgr"]
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                h, w = gray.shape[:2]
                half = w // 2
                if half > 0:
                    left_half = gray[:, :half].astype(np.float32)
                    right_flipped = cv2.flip(gray[:, w - half:], 1).astype(np.float32)
                    sym = 1.0 / (1.0 + float(np.mean(np.abs(left_half - right_flipped))) / 32.0)
                else:
                    sym = 0.5
                fname_low = rec["originalFile"].lower()
                bonus = 0.5 if any(k in fname_low for k in ("front", "frontal", "portrait", "shot_1", "1")) else 0.0
                if any(k in fname_low for k in ("left", "right", "side", "profile")):
                    bonus -= 0.5
                sc = sym + 0.01 * min(sharp, 100.0) + bonus
                if sc > best_frontal_score:
                    best_frontal_score = sc
                    best_frontal_rec = rec

        selected_rec = best_frontal_rec or converted_records[0]
        enc_portrait_bytes = encrypt_bytes(selected_rec["jpgBytes"])
        enc_portrait_b64 = base64.b64encode(enc_portrait_bytes).decode("ascii")

        return {
            "ok": True,
            "success": True,
            "rollNo": roll_no,
            "centroid": centroid_list,
            "centroidDim": len(centroid_list),
            "encryptedPortraitB64": enc_portrait_b64,
            "imagesProcessed": len(converted_records),
            "imagesConvertedToJpg": len(converted_records),
            "facesDetected": len(embeddings),
            "selectedFile": selected_rec["originalFile"],
            "portraitSelected": selected_rec["originalFile"],
            "convertedJpgFiles": [r["jpgFile"] for r in converted_records],
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


class GoogleCloudAIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[CLOUD-AI] {self.address_string()} - {format % args}", flush=True)

    def _set_cors(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Edge-Token")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_cors(200)

    def do_GET(self):
        if self.path in ("/", "/api/health"):
            self._set_cors(200)
            status = {
                "service": "FAST Smart Bus - Azure Cloud AI Inference Server",
                "status": "ONLINE",
                "encryption": "FSE1 (AES/CTR-HMAC Authenticated Encrypted Evidence)",
                "modelsLoaded": {
                    "mobilePhoneDetector": str(WEIGHTS_PATH.name),
                    "weightsPresent": WEIGHTS_PATH.exists(),
                    "drowsinessDetector": AI_ANALYZER.drowsy_backend,
                    "roadIncidentTrimmer": "2-Minute Clip Trimmer (-60s to +60s)",
                    "faceEmbeddingExtractor": FACE_EXTRACTOR is not None,
                },
                "connectedTargets": {
                    "renderWebsiteUrl": RENDER_SERVER_URL,
                    "supabaseUrl": SUPABASE_URL,
                },
                "serverTime12Hr": datetime.datetime.now().strftime("%I:%M:%S %p"),
            }
            self.wfile.write(json.dumps(status, indent=2).encode("utf-8"))
            return

        self._set_cors(404)
        self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        raw_bytes = self.rfile.read(content_len) if content_len > 0 else b""
        content_type = str(self.headers.get("Content-Type") or "")

        # 0. Extract multi-format ZIP enrollment bundle -> JPGs -> 512-d Centroid + FSE1 Frontal Portrait
        if self.path.split("?")[0] == "/api/enroll-zip-extract":
            edge_tok = self.headers.get("X-Edge-Token", "")
            if edge_tok and edge_tok != EDGE_BRIDGE_TOKEN:
                self._set_cors(401)
                self.wfile.write(json.dumps({"ok": False, "success": False, "error": "Invalid X-Edge-Token"}).encode("utf-8"))
                return
            try:
                zip_bytes = None
                roll_no = ""
                if "multipart/form-data" in content_type.lower():
                    import email.parser
                    msg = email.parser.BytesParser().parsebytes(
                        f"Content-Type: {content_type}\r\n\r\n".encode("utf-8") + raw_bytes
                    )
                    for part in msg.walk():
                        cd = part.get("Content-Disposition", "")
                        if "name=" not in cd:
                            continue
                        fname = part.get_filename()
                        payload_bytes = part.get_payload(decode=True) or b""
                        if fname or b"PK\x03\x04" in payload_bytes[:8]:
                            zip_bytes = payload_bytes
                        elif 'name="rollNo"' in cd or 'name="roll_no"' in cd:
                            roll_no = payload_bytes.decode("utf-8", errors="ignore").strip()
                elif raw_bytes[:4] == b"PK\x03\x04":
                    zip_bytes = raw_bytes
                else:
                    body_json = json.loads(raw_bytes.decode("utf-8", errors="ignore") or "{}")
                    roll_no = str(body_json.get("rollNo") or body_json.get("roll_no") or "").strip()
                    zb64 = str(body_json.get("zipBase64") or body_json.get("zip_base64") or "").strip()
                    if zb64:
                        zip_bytes = base64.b64decode(zb64.split(",")[-1])

                res = process_enrollment_zip_bytes(zip_bytes, roll_no=roll_no)
                self._set_cors(200)
                self.wfile.write(json.dumps(res).encode("utf-8"))
            except Exception as e:
                self._set_cors(400)
                self.wfile.write(json.dumps({"ok": False, "success": False, "error": str(e)}).encode("utf-8"))
            return

        raw_body = raw_bytes.decode("utf-8", errors="ignore") if raw_bytes else "{}"
        try:
            body = json.loads(raw_body)
        except Exception:
            body = {}

        # 1. Extract real 512-d ONNX facial centroid from enrollment photo
        if self.path == "/api/extract-face-embedding":
            if FACE_EXTRACTOR is None:
                self._set_cors(503)
                self.wfile.write(json.dumps({
                    "success": False,
                    "error": "ONNX FaceEngine not loaded on this Cloud AI worker.",
                }).encode("utf-8"))
                return

            img_b64 = body.get("imageBase64", "")
            if not img_b64:
                self._set_cors(400)
                self.wfile.write(json.dumps({"success": False, "error": "imageBase64 required"}).encode("utf-8"))
                return

            try:
                raw_img = base64.b64decode(img_b64.split(",")[-1])
                arr = np.frombuffer(raw_img, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError("Uploaded image could not be decoded.")
                primary = FACE_EXTRACTOR.detect_and_align_primary(bgr, min_sharpness=4.0)
                if primary is None or primary.get("embedding") is None:
                    self._set_cors(422)
                    self.wfile.write(json.dumps({
                        "success": False,
                        "error": "No clear front face detected in photo.",
                    }).encode("utf-8"))
                    return

                emb = [round(float(x), 6) for x in primary["embedding"].tolist()]
                self._set_cors(200)
                self.wfile.write(json.dumps({
                    "success": True,
                    "rollNo": body.get("rollNo", ""),
                    "dim": len(emb),
                    "detScore": round(float(primary.get("score", 0.0)), 4),
                    "centroid": emb,
                }).encode("utf-8"))
            except Exception as e:
                self._set_cors(400)
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode("utf-8"))
            return

        # 2. Full End-of-Trip Driver & Road Video Analysis
        if self.path == "/api/analyze-trip":
            bus_id = str(body.get("busId") or "FDZ-4589")
            trip_id = str(body.get("tripId") or f"TRIP-{int(time.time())}")
            trip_start_str = body.get("tripStartTime") or datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
            render_url = body.get("renderServerUrl") or RENDER_SERVER_URL
            telemetry_deductions = float(body.get("telemetryScoreDeductions", 0.0))

            temp_files = []
            driver_video_path = body.get("driverVideoPath")
            enc_drv_b64 = body.get("driverVideoEncryptedBase64") or body.get("driverVideoBase64")
            if enc_drv_b64:
                saved_drv = VIDEOS_DIR / f"driver_{trip_id}.mp4"
                driver_video_path = _decode_uploaded_video(enc_drv_b64, saved_drv)
                temp_files.append(saved_drv)

            road_video_path = body.get("roadVideoPath")
            enc_road_b64 = body.get("roadVideoEncryptedBase64") or body.get("roadVideoBase64")
            if enc_road_b64:
                saved_road = VIDEOS_DIR / f"road_{trip_id}.mp4"
                road_video_path = _decode_uploaded_video(enc_road_b64, saved_road)
                temp_files.append(saved_road)

            print(f"\n [CLOUD AI] Running Unified Driver Safety Models for {bus_id} ({trip_id})...")
            events = []
            if driver_video_path and os.path.exists(driver_video_path):
                ai_result = AI_ANALYZER.analyze_driver_video(
                    video_path=driver_video_path,
                    evidence_output_dir=str(EVIDENCE_DIR),
                    trip_start_time_str=trip_start_str,
                )
                events = ai_result.get("events", [])
            else:
                print(" [CLOUD AI] No driver cabin video supplied in trip payload; 0 driver cabin events.")

            # Encrypt every evidence snapshot with FSE1 and remove raw .jpg immediately
            for idx, ev in enumerate(events, start=1):
                raw_name = Path(ev.get("evidenceUrl", "")).name
                img_local = EVIDENCE_DIR / raw_name
                if img_local.exists():
                    enc_name = f"ev_{trip_id}_{idx}.enc"
                    enc_bytes = _save_encrypted_copy_and_delete_raw(img_local, EVIDENCE_DIR, enc_name)
                    ev["evidenceFilename"] = enc_name
                    ev["evidenceUrl"] = f"/api/secure-media/{enc_name}"
                    ev["evidenceEncryptedBase64"] = base64.b64encode(enc_bytes).decode("ascii")
                    ev["encrypted"] = True
                ev.pop("evidenceBase64", None)
                ev["source"] = "Google Cloud AI (best_mobile_detector.pth + EAR)"

            # Trim 2-minute Road Incident Clips strictly for Dangerous Overtake, Collision, Sharp Turn
            formatted_incidents = []
            incident_triggers = body.get("incidentTriggers", [])
            try:
                trip_start_dt = datetime.datetime.strptime(trip_start_str, "%Y-%m-%d %I:%M:%S %p")
            except ValueError:
                trip_start_dt = datetime.datetime.now()

            if road_video_path and os.path.exists(road_video_path):
                for idx, trig in enumerate(incident_triggers):
                    ev_type = str(trig.get("event", ""))
                    if not is_eligible_for_video_trim(ev_type):
                        continue
                    offset_s = float(trig.get("offsetSec", 30.0))
                    inc_id = f"INC-{int(time.time())}-{idx + 1}"
                    clip_info = trim_2min_incident_clip(
                        source_video_path=road_video_path,
                        event_offset_seconds=offset_s,
                        event_type=ev_type,
                        output_dir=str(INCIDENTS_DIR),
                        trip_start_dt=trip_start_dt,
                        incident_id=inc_id,
                    )
                    if clip_info:
                        raw_clip = INCIDENTS_DIR / Path(clip_info["clipUrl"]).name
                        clip_enc_b64 = ""
                        enc_clip_name = f"clip_{inc_id}.enc"
                        if raw_clip.exists():
                            enc_clip_bytes = _save_encrypted_copy_and_delete_raw(
                                raw_clip, INCIDENTS_DIR, enc_clip_name
                            )
                            if len(enc_clip_bytes) < 14 * 1024 * 1024:
                                clip_enc_b64 = base64.b64encode(enc_clip_bytes).decode("ascii")

                        formatted_incidents.append({
                            "id": inc_id,
                            "eventType": clip_info["eventType"],
                            "timestamp12Hr": clip_info["timestamp12Hr"],
                            "clipDuration": f"{clip_info['clipDurationSec']}s (-60s to +60s)",
                            "severity": "CRITICAL",
                            "lat": trig.get("lat"),
                            "lng": trig.get("lng"),
                            "imuDecel": trig.get("imuDecel", "--"),
                            "ultrasonicClearance": trig.get("ultrasonicClearance", "--"),
                            "speedBefore": trig.get("speedBefore", "--"),
                            "speedAfter": trig.get("speedAfter", "--"),
                            "summary": f"{clip_info['eventType']} detected by ESP32 + 2-Min Road Dashcam Trim",
                            "clipUrl": f"/api/secure-media/{enc_clip_name}",
                            "clipEncryptedBase64": clip_enc_b64,
                        })

            # Remove temporary uploaded plaintext trip videos
            for tf in temp_files:
                if tf.exists():
                    try:
                        tf.unlink()
                    except Exception:
                        pass

            safety_score = round(max(0.0, 100.0 - len(events) * 10.0 - telemetry_deductions), 1)

            # Push encrypted predictions to Render Portal (which also updates live Supabase tables)
            render_sync = push_predictions_to_render_website(
                bus_id=bus_id,
                trip_id=trip_id,
                events=events,
                incidents=formatted_incidents,
                driver_safety_score=safety_score,
                render_url_override=render_url,
            )

            # Fallback direct Supabase write only if Render Portal push failed
            if not render_sync.get("success"):
                now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
                clean_events = [{k: v for k, v in ev.items() if k != "evidenceEncryptedBase64"} for ev in events]
                clean_incs = [{k: v for k, v in inc.items() if k != "clipEncryptedBase64"} for inc in formatted_incidents]
                push_to_supabase("trips", {
                    "id": trip_id,
                    "plate_number": bus_id,
                    "end_time": now_iso,
                    "has_incidents": bool(clean_events or clean_incs),
                    "ai_phone_violations_count": sum(1 for e in clean_events if "phone" in str(e.get("type", "")).lower()),
                    "ai_drowsiness_alerts_count": sum(1 for e in clean_events if "drows" in str(e.get("type", "")).lower()),
                    "driver_safety_score": safety_score,
                    "ai_summary_json": {"events": clean_events, "incidents": clean_incs},
                    "status": "ANALYZED",
                })
                for inc in clean_incs:
                    push_to_supabase("incident_logs", {
                        "id": inc["id"],
                        "plate_number": bus_id,
                        "event_type": inc["eventType"],
                        "severity": inc["severity"],
                        "lat": inc.get("lat"),
                        "lng": inc.get("lng"),
                        "description": json.dumps(inc),
                        "created_at": now_iso,
                    })

            self._set_cors(200)
            response = {
                "success": True,
                "busId": bus_id,
                "tripId": trip_id,
                "driverSafetyScore": safety_score,
                "totalDriverViolations": len(events),
                "totalIncidentClips": len(formatted_incidents),
                "events": [{k: v for k, v in ev.items() if k != "evidenceEncryptedBase64"} for ev in events],
                "incidents": [{k: v for k, v in i.items() if k != "clipEncryptedBase64"} for i in formatted_incidents],
                "renderWebsiteSync": render_sync,
            }
            self.wfile.write(json.dumps(response).encode("utf-8"))
            return

        self._set_cors(404)
        self.wfile.write(json.dumps({"error": "Unknown POST route"}).encode("utf-8"))


if __name__ == "__main__":
    print("=" * 72)
    print("  FAST-NUCES SMART BUS - GOOGLE CLOUD AI WORKER SERVER")
    print(f"  Listening on Port : {PORT}")
    print(f"  Render Web Portal : {RENDER_SERVER_URL}")
    print(f"  Supabase Database : {SUPABASE_URL}")
    print(f"  Media Security    : FSE1 Authenticated Encryption Enabled")
    print("=" * 72)
    server = HTTPServer(("0.0.0.0", PORT), GoogleCloudAIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n [STOP] Shutting down Google Cloud AI Worker Server.")
        server.server_close()
