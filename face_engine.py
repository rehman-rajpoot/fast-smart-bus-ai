"""
Face Engine: Ultra-Lightweight Pure ONNX Runtime Face Recognition Stack
Optimized for Raspberry Pi 5 (8GB) Edge Attendance
Detector: SCRFD-500M (det_500m.onnx)
Recognizer: MobileFaceNet / ArcFace (w600k_mbf.onnx)
FYP: AI-Based Smart Bus Fleet Management System
"""

import os
import sys
import time
import numpy as np
import cv2
import onnxruntime as ort

# Standard ArcFace 112x112 canonical reference landmark positions (5 landmarks: left eye, right eye, nose, left mouth, right mouth)
ARCFACE_REFERENCE_POINTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def align_face_112(image, kps):
    """
    Performs standard 5-point similarity transformation alignment to generate 112x112 normalized face crop.
    kps: (5, 2) array of coordinates [[x0, y0], ..., [x4, y4]]
    """
    assert kps.shape == (5, 2), f"Expected kps shape (5, 2), got {kps.shape}"
    M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), ARCFACE_REFERENCE_POINTS)
    if M is None:
        return None
    warped = cv2.warpAffine(image, M, (112, 112), borderValue=0.0)
    return warped

def nms(bboxes, scores, thresh=0.4):
    """Greedy Non-Maximum Suppression."""
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

class SCRFDDetector:
    """
    Pure ONNX Runtime SCRFD-500M Face Detector.
    Processes dynamic resolution (e.g. 640x640 or camera native).
    Outputs: bboxes (x1, y1, x2, y2), det_score, and 5 facial landmarks.
    """
    def __init__(self, model_path, input_size=(640, 640), conf_thresh=0.5, nms_thresh=0.4):
        self.model_path = model_path
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        # Configure ONNX Runtime CPU execution provider with multi-threading
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.center_cache = {}

    def _get_anchor_centers(self, height, width, stride):
        key = (height, width, stride)
        if key in self.center_cache:
            return self.center_cache[key]
        anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        if self._num_anchors > 1:
            anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
        self.center_cache[key] = anchor_centers
        return anchor_centers

    def detect(self, img, conf_thresh=None):
        if conf_thresh is None:
            conf_thresh = self.conf_thresh

        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(self.input_size[1]) / self.input_size[0]
        if im_ratio > model_ratio:
            new_height = self.input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = self.input_size[0]
            new_height = int(new_width * im_ratio)

        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        # Normalization (img - 127.5) / 128.0
        blob = cv2.dnn.blobFromImage(det_img, 1.0 / 128.0, self.input_size, (127.5, 127.5, 127.5), swapRB=True)

        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        scores_list = []
        bboxes_list = []
        kpss_list = []

        input_height = blob.shape[2]
        input_width = blob.shape[3]

        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + self.fmc] * stride
            kps_preds = net_outs[idx + self.fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            anchor_centers = self._get_anchor_centers(height, width, stride)

            pos_inds = np.where(scores >= conf_thresh)[0]
            if len(pos_inds) == 0:
                continue

            # distance2bbox
            x1 = anchor_centers[:, 0] - bbox_preds[:, 0]
            y1 = anchor_centers[:, 1] - bbox_preds[:, 1]
            x2 = anchor_centers[:, 0] + bbox_preds[:, 2]
            y2 = anchor_centers[:, 1] + bbox_preds[:, 3]
            bboxes = np.stack([x1, y1, x2, y2], axis=-1)

            # distance2kps
            kpss = np.zeros((anchor_centers.shape[0], 5, 2), dtype=np.float32)
            for i in range(5):
                kpss[:, i, 0] = anchor_centers[:, 0] + kps_preds[:, i * 2]
                kpss[:, i, 1] = anchor_centers[:, 1] + kps_preds[:, i * 2 + 1]

            scores_list.append(scores[pos_inds].reshape(-1))
            bboxes_list.append(bboxes[pos_inds])
            kpss_list.append(kpss[pos_inds])

        if len(scores_list) == 0:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0, 5, 2))

        scores = np.concatenate(scores_list, axis=0)
        bboxes = np.concatenate(bboxes_list, axis=0)
        kpss = np.concatenate(kpss_list, axis=0)

        # Scale back to original image coordinates
        bboxes = bboxes / det_scale
        kpss = kpss / det_scale

        # Clamp bounding boxes to image bounds
        bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, img.shape[1] - 1)
        bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, img.shape[0] - 1)

        keep = nms(bboxes, scores, thresh=self.nms_thresh)
        return bboxes[keep], scores[keep], kpss[keep]

class ArcFaceRecognizer:
    """
    Pure ONNX Runtime MobileFaceNet Feature Extractor.
    Input: 112x112 aligned face crop.
    Output: 512-dimensional L2-normalized unit vector.
    """
    def __init__(self, model_path):
        self.model_path = model_path
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def extract(self, face_chip_112):
        """
        Extract 512-d L2 normalized embedding from 112x112 BGR face chip.
        """
        assert face_chip_112.shape == (112, 112, 3), f"Expected (112, 112, 3), got {face_chip_112.shape}"
        # Preprocessing: BGR to RGB, normalize (crop - 127.5) / 127.5 or (crop - 127.5) / 128.0
        blob = cv2.dnn.blobFromImage(
            face_chip_112, 
            scalefactor=1.0 / 127.5, 
            size=(112, 112), 
            mean=(127.5, 127.5, 127.5), 
            swapRB=True
        )
        net_out = self.session.run([self.output_name], {self.input_name: blob})[0]
        vec = net_out[0]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

class FaceEngine:
    """
    Unified High-Level Interface combining SCRFD-500M and MobileFaceNet.
    Runs efficiently on both Laptop and Raspberry Pi 5.
    """
    def __init__(self, models_dir=None, det_size=(640, 640), conf_thresh=0.5):
        if models_dir is None:
            # Check local buffalo_sc first
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            candidate_dirs = [
                os.path.join(cur_dir, "models", "buffalo_sc"),
                os.path.join(cur_dir, "..", "face recognition Models for FYP", "models", "buffalo_sc"),
                os.path.expanduser(r"~\.insightface\models\buffalo_sc")
            ]
            for c in candidate_dirs:
                if os.path.exists(os.path.join(c, "det_500m.onnx")):
                    models_dir = c
                    break

        if models_dir is None or not os.path.exists(models_dir):
            raise FileNotFoundError(f"Model weights not found. Searched {candidate_dirs}")

        det_path = os.path.join(models_dir, "det_500m.onnx")
        rec_path = os.path.join(models_dir, "w600k_mbf.onnx")

        print(f"[FaceEngine] Loading SCRFD detector from: {det_path}")
        self.detector = SCRFDDetector(det_path, input_size=det_size, conf_thresh=conf_thresh)
        print(f"[FaceEngine] Loading MobileFaceNet recognizer from: {rec_path}")
        self.recognizer = ArcFaceRecognizer(rec_path)
        print("[FaceEngine] Engine initialized successfully.")

    def detect_and_align_primary(self, img, doorway_box=None, min_sharpness=15.0):
        """
        Detect faces in image, filter for primary active subject in doorway,
        align to 112x112, and extract embedding.
        Returns dict with bbox, score, kps, aligned_chip, embedding, sharpness.
        """
        t0 = time.time()
        bboxes, scores, kpss = self.detector.detect(img)
        det_time = (time.time() - t0) * 1000.0

        if len(bboxes) == 0:
            return None

        h, w, _ = img.shape
        candidates = []

        for bbox, score, kps in zip(bboxes, scores, kpss):
            x1, y1, x2, y2 = bbox.astype(int)
            bw, bh = x2 - x1, y2 - y1
            area = bw * bh

            # Ignore tiny background faces (<45x45)
            if bw < 45 or bh < 45 or area < 2025:
                continue

            # Doorway ROI filtering if specified (e.g. normalized [xmin, ymin, xmax, ymax])
            if doorway_box is not None:
                dw_x1, dw_y1, dw_x2, dw_y2 = doorway_box
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                if not (dw_x1 <= cx <= dw_x2 and dw_y1 <= cy <= dw_y2):
                    continue

            # Sharpness check on face crop
            crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            if sharpness < min_sharpness:
                continue

            # Prioritize prominent doorway faces
            focus_score = area * score * np.sqrt(max(1.0, sharpness))
            candidates.append({
                "bbox": bbox,
                "score": float(score),
                "kps": kps,
                "sharpness": sharpness,
                "focus_score": focus_score,
                "crop": crop
            })

        if not candidates:
            return None

        # Pick candidate with highest focus score
        candidates.sort(key=lambda c: c["focus_score"], reverse=True)
        primary = candidates[0]

        # ArcFace 112x112 alignment
        aligned_chip = align_face_112(img, primary["kps"])
        if aligned_chip is None:
            aligned_chip = cv2.resize(primary["crop"], (112, 112))

        t1 = time.time()
        embedding = self.recognizer.extract(aligned_chip)
        rec_time = (time.time() - t1) * 1000.0

        primary["aligned_chip"] = aligned_chip
        primary["embedding"] = embedding
        primary["det_latency_ms"] = det_time
        primary["rec_latency_ms"] = rec_time
        primary["total_latency_ms"] = det_time + rec_time

        return primary
