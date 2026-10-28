"""
scored_live.py -- Frame Scoring System for Waste-Collection Camera
=================================================================
Integrates:
  - YOLOv8 person & hand detection (from live_person.py)
  - Background-subtraction motion detection (ported from MotionFull-main/motion.py)

Scoring model (max 100 pts per frame):
  COMPONENT           MAX   LOGIC
  Person presence      30   flat bonus if >= 1 person detected
  Person confidence    10   scaled by max person confidence [0-1]
  Hand proximity       50   inverse-distance to ROI centroid
  Motion activity      10   scaled by contour area in ROI

Controls:
  q         Quit
  w / s     Raise / lower YOLO confidence threshold
  r         Toggle ROI drawing mode (left-click to place, right-click to confirm)
  c         Reset ROI polygon back to default
  1/2/3     Save raw frame to frames/no_person, person_no_dump, dumping
"""

import os
import sys
import time
import csv
import json
import math
import colorsys
import threading
import collections
from datetime import datetime

if sys.platform == "win32":
    for candidate_dir in [
        r"C:\Program Files\Common Files\microsoft shared\ClickToRun",
        r"C:\Program Files\KiCad\10.0\bin",
    ]:
        if os.path.isdir(candidate_dir):
            try:
                os.add_dll_directory(candidate_dir)
            except Exception:
                pass

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Determine compute device
DEVICE = 0 if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    torch.set_num_threads(min(4, os.cpu_count() or 4))

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
MODEL_WEIGHTS      = "yolov8n.pt"
HAND_MODEL_WEIGHTS = "hand_yolov8n.pt"
HAND_MODEL_URL     = "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt"
INFERENCE_SIZE     = 320

DEFAULT_CONF_THRESHOLD = 0.25
MIN_CONF_THRESHOLD     = 0.05
MAX_CONF_THRESHOLD     = 0.90
CONF_STEP              = 0.05

LOG_FILE = "scored_log.csv"

# Motion detection constants (from MotionFull-main/motion.py, unchanged)
DIFF_THRESHOLD   = 22
MIN_CONTOUR_AREA = 350
BG_ALPHA         = 0.04
WARMUP_FRAMES    = 30

# Scoring weights
SCORE_PERSON_FLAT = 30   # flat bonus for person present
SCORE_PERSON_CONF = 10   # up to 10 pts for confidence
SCORE_HAND_PROX   = 50   # up to 50 pts for hand proximity to ROI centroid
SCORE_MOTION      = 10   # up to 10 pts for motion activity inside ROI

# Default ROI polygon (normalised [0-1] x, y) - same format as MotionFull
DEFAULT_ROI_POLYGON = [
    [0.35, 0.20],
    [0.65, 0.20],
    [0.65, 0.70],
    [0.35, 0.70],
]
ROI_CONFIG_FILE = "roi_polygon.json"

# Save categories
SAVE_CATEGORIES = {
    "1": ("no_person",      os.path.join("frames", "no_person")),
    "2": ("person_no_dump", os.path.join("frames", "person_no_dump")),
    "3": ("dumping",        os.path.join("frames", "dumping")),
}

# Colours (BGR)
COLOR_PERSON    = (0,  230,  80)
COLOR_HAND      = (0,  165, 255)
COLOR_ROI_FILL  = (0,   80, 180)
COLOR_ROI_EDGE  = (0,  220, 255)
COLOR_MOTION    = (0,  255, 200)
COLOR_SCORE_HI  = (0,  255, 100)
COLOR_SCORE_MED = (0,  200, 255)
COLOR_SCORE_LO  = (0,  100, 255)

SCORE_HI_THRESH  = 65
SCORE_MED_THRESH = 35

SCORE_HISTORY_LEN = 120


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

def ensure_directories():
    for _, folder_path in SAVE_CATEGORIES.values():
        os.makedirs(folder_path, exist_ok=True)


def ensure_hand_weights():
    if not os.path.isfile(HAND_MODEL_WEIGHTS):
        print(f"[INFO] Downloading '{HAND_MODEL_WEIGHTS}' from Hugging Face...")
        import urllib.request
        try:
            urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_WEIGHTS)
            print(f"[INFO] Downloaded '{HAND_MODEL_WEIGHTS}' successfully.")
        except Exception as e:
            print(f"[WARNING] Download failed: {e}. Trying curl...")
            os.system(f'curl.exe -L -o "{HAND_MODEL_WEIGHTS}" "{HAND_MODEL_URL}"')


def init_logger():
    file_exists = os.path.isfile(LOG_FILE)
    handle = open(LOG_FILE, mode="a", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    if not file_exists:
        writer.writerow([
            "timestamp", "total_score",
            "person_score", "hand_score", "motion_score",
            "n_persons", "n_hands",
            "max_person_conf", "max_hand_conf",
            "motion_area", "threshold",
        ])
        handle.flush()
    return handle, writer


def load_roi_polygon():
    if os.path.exists(ROI_CONFIG_FILE):
        try:
            with open(ROI_CONFIG_FILE, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                if len(data) >= 3:
                    print(f"[ROI] Loaded {len(data)}-point polygon from {ROI_CONFIG_FILE}")
                    return data
                elif len(data) == 0:
                    print(f"[ROI] ROI cleared in {ROI_CONFIG_FILE} (full frame mode)")
                    return []
        except Exception as e:
            print(f"[ROI] Failed to load polygon: {e}")
    return DEFAULT_ROI_POLYGON.copy()


def save_roi_polygon(pts):
    try:
        with open(ROI_CONFIG_FILE, "w") as f:
            json.dump(pts, f, indent=2)
        if pts:
            print(f"[ROI] Saved {len(pts)}-point polygon to {ROI_CONFIG_FILE}")
        else:
            print(f"[ROI] Cleared polygon in {ROI_CONFIG_FILE} (full-frame mode)")
    except Exception as e:
        print(f"[ROI] Save failed: {e}")


def open_webcam(device_index=0):
    """Open webcam reliably on Windows with DirectShow fallback to prevent MSMF hangs."""
    if sys.platform == "win32":
        cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(device_index)


def poly_pixel_pts(polygon_pts, w, h):
    """Convert normalised polygon points to pixel coordinates."""
    return np.array(
        [[int(p[0] * w), int(p[1] * h)] for p in polygon_pts],
        dtype=np.int32
    )


def roi_centroid_px(polygon_pts, w, h):
    """Return pixel centroid of the ROI polygon."""
    pts = poly_pixel_pts(polygon_pts, w, h)
    cx = int(np.mean(pts[:, 0]))
    cy = int(np.mean(pts[:, 1]))
    return cx, cy


def apply_roi_mask(thresh, polygon_pts, frame_w, frame_h):
    """Mask a binary threshold image to only the ROI polygon."""
    if not polygon_pts or len(polygon_pts) < 3:
        return thresh
    pts_array = poly_pixel_pts(polygon_pts, frame_w, frame_h)
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_array], 255)
    return cv2.bitwise_and(thresh, mask)


def draw_box(frame, x1, y1, x2, y2, label, color):
    """Draw a filled-header bounding box."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    top = max(y1, th + 8)
    cv2.rectangle(frame, (x1, top - th - 6), (x1 + tw + 6, top), color, -1)
    cv2.putText(frame, label, (x1 + 3, top - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)


def score_color(score):
    if score >= SCORE_HI_THRESH:
        return COLOR_SCORE_HI
    elif score >= SCORE_MED_THRESH:
        return COLOR_SCORE_MED
    return COLOR_SCORE_LO


# ------------------------------------------------------------------
# MOTION DETECTOR (ported from MotionFull-main/motion.py)
# ------------------------------------------------------------------

class MotionDetector:
    """
    Background-subtraction motion detector.
    Algorithm parameters are IDENTICAL to MotionFull-main/motion.py:
      DIFF_THRESHOLD   = 22
      MIN_CONTOUR_AREA = 350
      BG_ALPHA         = 0.04
      WARMUP_FRAMES    = 30
    """

    def __init__(self):
        self.bg_frame    = None
        self.frame_count = 0

    def _update_bg(self, gray_frame):
        """Update running background model. Returns threshold image or None."""
        if self.bg_frame is None:
            self.bg_frame    = gray_frame.astype("float")
            self.frame_count = 0
            return None

        cv2.accumulateWeighted(gray_frame, self.bg_frame, BG_ALPHA)
        self.frame_count += 1

        if self.frame_count < WARMUP_FRAMES:
            return None

        bg_uint8    = cv2.convertScaleAbs(self.bg_frame)
        frame_delta = cv2.absdiff(bg_uint8, gray_frame)
        _, thresh   = cv2.threshold(frame_delta, DIFF_THRESHOLD, 255,
                                    cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)
        return thresh

    def detect(self, gray_frame, polygon_pts, frame_w, frame_h):
        """
        Full pipeline: update background, apply ROI mask, find contours.
        Returns: (motion_detected: bool, total_area: int, contours: list)
        """
        thresh = self._update_bg(gray_frame)
        if thresh is None:
            return False, 0, []

        roi_thresh = apply_roi_mask(thresh, polygon_pts, frame_w, frame_h)
        cnts, _    = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
        valid      = [c for c in cnts if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
        total_area = int(sum(cv2.contourArea(c) for c in valid))

        return len(valid) > 0, total_area, valid


# ------------------------------------------------------------------
# SCORING ENGINE
# ------------------------------------------------------------------

class FrameScorer:
    """
    Computes a 0-100 frame quality score.

    Person presence  : 30 pts flat + 10 pts scaled by confidence = max 40
    Hand proximity   : 0-50 pts (inverse distance to ROI centroid)
    Motion in ROI    : 0-10 pts (scaled by area, caps at MAX_MOTION_AREA)
    """

    MAX_MOTION_AREA = 60_000  # px^2 at which motion score saturates

    def compute(self, person_boxes, hand_boxes, motion_area,
                polygon_pts, frame_w, frame_h):
        """Returns dict: total, person_score, hand_score, motion_score."""

        # Person score (proximity-weighted to ROI)
        person_score = 0.0
        if person_boxes:
            if polygon_pts and len(polygon_pts) >= 3:
                cx, cy = roi_centroid_px(polygon_pts, frame_w, frame_h)
                diag = math.hypot(frame_w, frame_h)
                max_dist = diag * 0.5
                best_p_score = 0.0
                for pb in person_boxes:
                    px1, py1, px2, py2, pconf = pb[0], pb[1], pb[2], pb[3], pb[4]
                    closest_x = max(px1, min(cx, px2))
                    closest_y = max(py1, min(cy, py2))
                    p_dist = math.hypot(closest_x - cx, closest_y - cy)
                    p_prox = max(0.0, 1.0 - p_dist / max_dist)
                    cand_score = (SCORE_PERSON_FLAT + SCORE_PERSON_CONF * pconf) * p_prox
                    if cand_score > best_p_score:
                        best_p_score = cand_score
                person_score = min(best_p_score, SCORE_PERSON_FLAT + SCORE_PERSON_CONF)
            else:
                max_conf = max(b[4] for b in person_boxes)
                person_score = min(SCORE_PERSON_FLAT + SCORE_PERSON_CONF * max_conf,
                                   SCORE_PERSON_FLAT + SCORE_PERSON_CONF)

        # Hand proximity score
        hand_score = 0.0
        if hand_boxes:
            if polygon_pts and len(polygon_pts) >= 3:
                cx, cy = roi_centroid_px(polygon_pts, frame_w, frame_h)
                diag = math.hypot(frame_w, frame_h)
                max_dist = diag * 0.5
                best_h_prox = 0.0
                for hb in hand_boxes:
                    hcx = (hb[0] + hb[2]) / 2
                    hcy = (hb[1] + hb[3]) / 2
                    dist = math.hypot(hcx - cx, hcy - cy)
                    prox = max(0.0, 1.0 - dist / max_dist)
                    best_h_prox = max(best_h_prox, prox)
                hand_score = SCORE_HAND_PROX * best_h_prox
            else:
                max_h_conf = max(b[4] for b in hand_boxes)
                hand_score = SCORE_HAND_PROX * max_h_conf

        # Motion score
        motion_score = 0.0
        if motion_area > 0:
            motion_score = SCORE_MOTION * min(1.0, motion_area / self.MAX_MOTION_AREA)

        total = min(round(person_score + hand_score + motion_score, 1), 100.0)

        return {
            "total":        total,
            "person_score": round(person_score, 1),
            "hand_score":   round(hand_score,   1),
            "motion_score": round(motion_score, 1),
        }


# ------------------------------------------------------------------
# ASYNC YOLO DETECTOR (from live_person.py)
# ------------------------------------------------------------------

class AsyncDetector:
    def __init__(self, obj_model, hand_model, imgsz=320, device="cpu"):
        self.obj_model  = obj_model
        self.hand_model = hand_model
        self.imgsz      = imgsz
        self.device     = device
        self.lock       = threading.Lock()
        self.next_frame = None
        self.conf_threshold    = DEFAULT_CONF_THRESHOLD
        self.latest_obj_boxes  = []
        self.latest_hand_boxes = []
        self.infer_fps         = 0.0
        self.has_new_result    = False
        self.running           = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def update_frame(self, frame, conf):
        with self.lock:
            self.next_frame     = frame
            self.conf_threshold = conf

    def get_detections(self):
        with self.lock:
            is_new              = self.has_new_result
            self.has_new_result = False
            return (self.latest_obj_boxes, self.latest_hand_boxes,
                    self.infer_fps, is_new)

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.5)

    def _loop(self):
        while self.running:
            frame_to_process = None
            conf = DEFAULT_CONF_THRESHOLD
            with self.lock:
                if self.next_frame is not None:
                    frame_to_process = self.next_frame
                    self.next_frame  = None
                    conf             = self.conf_threshold

            if frame_to_process is None:
                time.sleep(0.005)
                continue

            t0 = time.time()
            try:
                # Detect ONLY persons (class 0) - ignores all other objects
                res_obj = self.obj_model.predict(
                    source=frame_to_process, imgsz=self.imgsz,
                    classes=[0],
                    conf=conf, verbose=False, device=self.device)
                res_hand = self.hand_model.predict(
                    source=frame_to_process, imgsz=self.imgsz,
                    conf=conf, verbose=False, device=self.device)

                parsed_obj = []
                for b in res_obj[0].boxes:
                    cid = int(b.cls[0])
                    if cid == 0:  # Only person
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        c    = float(b.conf[0])
                        name = self.obj_model.names[cid]
                        parsed_obj.append((x1, y1, x2, y2, c, cid, name))

                parsed_hand = []
                for b in res_hand[0].boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    c = float(b.conf[0])
                    parsed_hand.append((x1, y1, x2, y2, c))

                dt     = time.time() - t0
                cur_fp = 1.0 / dt if dt > 0 else 0.0

                with self.lock:
                    self.latest_obj_boxes  = parsed_obj
                    self.latest_hand_boxes = parsed_hand
                    self.infer_fps = (
                        0.8 * self.infer_fps + 0.2 * cur_fp
                        if self.infer_fps > 0 else cur_fp
                    )
                    self.has_new_result = True

            except Exception:
                time.sleep(0.01)


# ------------------------------------------------------------------
# HUD & OVERLAY DRAWING
# ------------------------------------------------------------------

def draw_roi_polygon(frame, polygon_pts, is_drawing=False, drawing_pts=None, mouse_pos=None):
    """Draw the ROI polygon with a semi-transparent fill and centroid crosshair."""
    h, w = frame.shape[:2]

    # Draw saved ROI (dimmed outline if currently in draw mode)
    if polygon_pts and len(polygon_pts) >= 3:
        pts = poly_pixel_pts(polygon_pts, w, h)
        if is_drawing:
            # Faint outline while drawing a new one
            cv2.polylines(frame, [pts], isClosed=True,
                          color=(70, 90, 110), thickness=1)
        else:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], COLOR_ROI_FILL)
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
            cv2.polylines(frame, [pts], isClosed=True,
                          color=COLOR_ROI_EDGE, thickness=2)
            for pt in pts:
                cv2.circle(frame, (pt[0], pt[1]), 5, COLOR_ROI_EDGE, -1)

            cx, cy = roi_centroid_px(polygon_pts, w, h)
            cv2.drawMarker(frame, (cx, cy), COLOR_ROI_EDGE,
                           cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
            cv2.putText(frame, "ROI", (cx + 12, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ROI_EDGE,
                        1, cv2.LINE_AA)

    # In Draw Mode: show current vertices and live preview lines
    if is_drawing:
        if drawing_pts:
            if len(drawing_pts) >= 2:
                cv2.polylines(frame,
                              [np.array(drawing_pts, dtype=np.int32)],
                              isClosed=False, color=(0, 255, 255), thickness=2)
            for i, (px, py) in enumerate(drawing_pts):
                col = (0, 255, 0) if i == 0 else (0, 220, 255)
                cv2.circle(frame, (px, py), 6, col, -1)
                cv2.circle(frame, (px, py), 8, (0, 0, 0), 1)
                cv2.putText(frame, str(i + 1), (px + 8, py - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # Live guide line from last point to current mouse position
            if mouse_pos and 0 <= mouse_pos[0] < w and 0 <= mouse_pos[1] < h:
                cv2.line(frame, drawing_pts[-1], mouse_pos, (0, 255, 255), 1, cv2.LINE_AA)
                if len(drawing_pts) >= 2:
                    cv2.line(frame, mouse_pos, drawing_pts[0], (0, 180, 180), 1, cv2.LINE_AA)


def draw_motion_contours(frame, contours):
    """Draw bounding boxes around motion contours inside the ROI."""
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + cw, y + ch), COLOR_MOTION, 1)


def draw_score_panel(frame, scores, score_history):
    """
    Draw the live scoring panel on the right side of the frame.
      - Big numeric total score
      - Per-component horizontal bars (Person, Hand, Motion)
      - 120-frame spark-line history with colour-coded segments
    """
    h, w = frame.shape[:2]

    panel_w = 200
    panel_x = w - panel_w - 8
    panel_y = 50
    panel_h = 320
    bar_h   = 16

    # Panel background
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (panel_x - 6, panel_y - 6),
                  (panel_x + panel_w + 6, panel_y + panel_h + 6),
                  (12, 12, 18), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame,
                  (panel_x - 6, panel_y - 6),
                  (panel_x + panel_w + 6, panel_y + panel_h + 6),
                  (60, 60, 80), 1)

    total = scores["total"]
    s_col = score_color(total)

    # Title
    cv2.putText(frame, "FRAME SCORE", (panel_x, panel_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 220), 1, cv2.LINE_AA)

    # Big score number
    big_label  = f"{total:.0f}"
    (bw, _), _ = cv2.getTextSize(big_label, cv2.FONT_HERSHEY_DUPLEX, 2.2, 3)
    cv2.putText(frame, big_label,
                (panel_x + (panel_w - bw) // 2, panel_y + 75),
                cv2.FONT_HERSHEY_DUPLEX, 2.2, s_col, 3, cv2.LINE_AA)
    cv2.putText(frame, "/ 100",
                (panel_x + (panel_w - bw) // 2 + bw - 10, panel_y + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 160), 1, cv2.LINE_AA)

    # Full-width score bar
    bar_y  = panel_y + 85
    fill_w = int(panel_w * total / 100.0)
    cv2.rectangle(frame, (panel_x, bar_y),
                  (panel_x + panel_w, bar_y + 8), (40, 40, 50), -1)
    cv2.rectangle(frame, (panel_x, bar_y),
                  (panel_x + fill_w, bar_y + 8), s_col, -1)

    # Component breakdown bars
    components = [
        ("Person", scores["person_score"],
         SCORE_PERSON_FLAT + SCORE_PERSON_CONF, COLOR_PERSON),
        ("Hand",   scores["hand_score"],   SCORE_HAND_PROX, COLOR_HAND),
        ("Motion", scores["motion_score"], SCORE_MOTION,    COLOR_MOTION),
    ]

    cy = bar_y + 22
    for lbl, val, max_val, color in components:
        pct  = val / max_val if max_val > 0 else 0
        fill = int(panel_w * pct)

        cv2.putText(frame, lbl, (panel_x, cy + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{val:.0f}/{max_val}",
                    (panel_x + panel_w - 55, cy + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)
        cy += 16
        cv2.rectangle(frame, (panel_x, cy),
                      (panel_x + panel_w, cy + bar_h), (30, 30, 40), -1)
        cv2.rectangle(frame, (panel_x, cy),
                      (panel_x + fill, cy + bar_h), color, -1)
        cy += bar_h + 8

    # Spark-line history
    spark_y = cy + 8
    spark_h = 60
    cv2.rectangle(frame, (panel_x, spark_y),
                  (panel_x + panel_w, spark_y + spark_h), (25, 25, 35), -1)
    cv2.putText(frame, "Live Feed", (panel_x, spark_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 140), 1, cv2.LINE_AA)

    hist = list(score_history)
    if len(hist) >= 2:
        pts_px = []
        for i, s in enumerate(hist):
            sx = panel_x + int(i * panel_w / (SCORE_HISTORY_LEN - 1))
            sy = spark_y + spark_h - int(s / 100.0 * spark_h)
            pts_px.append((sx, sy))
        for i in range(len(pts_px) - 1):
            mid   = (hist[i] + hist[i + 1]) / 2
            c_seg = score_color(mid)
            cv2.line(frame, pts_px[i], pts_px[i + 1], c_seg, 1, cv2.LINE_AA)

    # Reference lines
    for tv, tc in [(35, (0, 100, 255)), (65, (0, 200, 100))]:
        ty = spark_y + spark_h - int(tv / 100.0 * spark_h)
        cv2.line(frame, (panel_x, ty),
                 (panel_x + panel_w, ty), tc, 1, cv2.LINE_AA)


def draw_hud(frame, cam_fps, infer_fps, threshold,
             person_count, hand_count, motion_area,
             has_roi, is_roi_drawing, alert_text="", device_tag="CPU"):
    """Top and bottom semi-transparent HUD strips designed to fit cleanly inside 640px."""
    fh, fw = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, 46), (20, 20, 20), -1)
    cv2.rectangle(overlay, (0, fh - 30), (fw, fh), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Top Bar: Line 1 (Cam & AI Stats) + Line 2 (Detections & ROI state)
    roi_status = "DRAWING" if is_roi_drawing else ("ACTIVE" if has_roi else "FULL-FRAME")
    stats_l1 = f"Cam: {cam_fps:4.1f}fps | AI: {infer_fps:4.1f}fps [{device_tag}] | Conf: {threshold:.2f} (w/s)"
    stats_l2 = f"Persons: {person_count} | Hands: {hand_count} | Motion: {motion_area:,}px | ROI: {roi_status}"

    cv2.putText(frame, stats_l1, (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, stats_l2, (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 255, 200), 1, cv2.LINE_AA)

    # Bottom Bar: Clean Controls (fits in 640px)
    hint = "[q] Quit | [r] Reset ROI | [d] Draw ROI | [c] Clear ROI | [1-3] Save Frame"
    cv2.putText(frame, hint, (10, fh - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.41, (220, 220, 220), 1, cv2.LINE_AA)

    # Alert Banner: High-contrast floating pill
    if alert_text:
        (tw, th), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        bx, by = 10, 54
        cv2.rectangle(frame, (bx, by), (bx + tw + 16, by + th + 12), (15, 15, 20), -1)
        cv2.rectangle(frame, (bx, by), (bx + tw + 16, by + th + 12), (0, 255, 120), 2)
        cv2.putText(frame, alert_text, (bx + 8, by + th + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 120), 2, cv2.LINE_AA)


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  Frame Scoring System -- Person + Hand Proximity + Motion")
    print("=" * 65)

    device_str = "CPU"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        device_str = f"GPU ({gpu_name})"
    print(f"[INFO] Compute Device : {device_str}")

    ensure_directories()
    log_handle, csv_writer = init_logger()
    print(f"[INFO] Logging scores to '{LOG_FILE}'")

    print(f"[INFO] Loading person model '{MODEL_WEIGHTS}'...")
    try:
        model = YOLO(MODEL_WEIGHTS)
    except Exception as e:
        print(f"[ERROR] {e}")
        log_handle.close()
        sys.exit(1)

    print(f"[INFO] Loading hand model '{HAND_MODEL_WEIGHTS}'...")
    ensure_hand_weights()
    try:
        hand_model = YOLO(HAND_MODEL_WEIGHTS)
    except Exception as e:
        print(f"[ERROR] {e}")
        log_handle.close()
        sys.exit(1)

    polygon_pts   = load_roi_polygon()
    detector      = AsyncDetector(model, hand_model, imgsz=INFERENCE_SIZE, device=DEVICE)
    motion_det    = MotionDetector()
    scorer        = FrameScorer()
    score_history = collections.deque([0.0] * SCORE_HISTORY_LEN,
                                      maxlen=SCORE_HISTORY_LEN)

    print("[INFO] Opening webcam (device 0)...")
    cap = open_webcam(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        detector.stop()
        log_handle.close()
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    conf_threshold    = DEFAULT_CONF_THRESHOLD
    cam_fps           = 0.0
    prev_time         = time.time()
    alert_msg         = ""
    alert_expiry      = 0.0
    is_roi_drawing    = False
    roi_drawing_pts   = []
    current_mouse_pos = (0, 0)
    frame_w           = 640
    frame_h           = 480
    latest_scores     = {"total": 0.0, "person_score": 0.0,
                         "hand_score": 0.0, "motion_score": 0.0}
    latest_motion_area = 0
    latest_motion_cnts = []

    window_name = "Frame Scoring -- Live Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # ROI drawing mouse callback
    def on_mouse(event, x, y, flags, param):
        nonlocal is_roi_drawing, roi_drawing_pts, polygon_pts, current_mouse_pos, alert_msg, alert_expiry
        current_mouse_pos = (x, y)
        if not is_roi_drawing:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            roi_drawing_pts.append((x, y))
            alert_msg = f"Point {len(roi_drawing_pts)} added (Right-Click when finished)"
            alert_expiry = time.time() + 4.0
            print(f"[ROI] Point {len(roi_drawing_pts)}: ({x}, {y})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(roi_drawing_pts) >= 3:
                polygon_pts = [
                    [round(px / frame_w, 4), round(py / frame_h, 4)]
                    for px, py in roi_drawing_pts
                ]
                save_roi_polygon(polygon_pts)
                is_roi_drawing  = False
                roi_drawing_pts = []
                alert_msg = f"ROI Saved: {len(polygon_pts)}-point polygon"
                alert_expiry = time.time() + 3.0
                print(f"[ROI] Confirmed {len(polygon_pts)}-point polygon.")
            else:
                alert_msg = "Need at least 3 points to confirm ROI!"
                alert_expiry = time.time() + 3.0
                print("[ROI] Need at least 3 points before confirming.")

    cv2.setMouseCallback(window_name, on_mouse)
    print("\n[READY] Camera active. Press 'q' to quit.\n")

    try:
        while True:
            ok, raw_frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            now       = time.time()
            dt        = now - prev_time
            prev_time = now
            if dt > 0:
                fps_i   = 1.0 / dt
                cam_fps = (0.90 * cam_fps + 0.10 * fps_i) if cam_fps > 0 else fps_i

            frame_h, frame_w = raw_frame.shape[:2]

            # Motion detection (synchronous on main thread - very fast)
            gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            motion_detected, motion_area, motion_cnts = motion_det.detect(
                gray, polygon_pts, frame_w, frame_h)
            latest_motion_area = motion_area
            latest_motion_cnts = motion_cnts

            # YOLO detection (async background thread)
            detector.update_frame(raw_frame, conf_threshold)
            obj_boxes, hand_boxes, infer_fps, is_new = detector.get_detections()
            person_boxes = [b for b in obj_boxes if b[6] == "person"]

            # Score the current frame
            latest_scores = scorer.compute(
                person_boxes, hand_boxes, motion_area,
                polygon_pts, frame_w, frame_h)
            score_history.append(latest_scores["total"])

            # Log when YOLO has a new result
            if is_new:
                p_confs    = [b[4] for b in person_boxes]
                h_confs    = [b[4] for b in hand_boxes]
                max_p_conf = max(p_confs) if p_confs else 0.0
                max_h_conf = max(h_confs) if h_confs else 0.0
                ts_str     = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                csv_writer.writerow([
                    ts_str,
                    f"{latest_scores['total']:.1f}",
                    f"{latest_scores['person_score']:.1f}",
                    f"{latest_scores['hand_score']:.1f}",
                    f"{latest_scores['motion_score']:.1f}",
                    len(person_boxes),
                    len(hand_boxes),
                    f"{max_p_conf:.3f}",
                    f"{max_h_conf:.3f}",
                    motion_area,
                    f"{conf_threshold:.2f}",
                ])
                log_handle.flush()

            # Build display
            disp = raw_frame.copy()

            draw_roi_polygon(disp, polygon_pts, is_roi_drawing, roi_drawing_pts, current_mouse_pos)
            draw_motion_contours(disp, latest_motion_cnts)

            has_roi = bool(polygon_pts and len(polygon_pts) >= 3)
            if has_roi:
                roi_cx, roi_cy = roi_centroid_px(polygon_pts, frame_w, frame_h)
            else:
                roi_cx, roi_cy = frame_w // 2, frame_h // 2

            # 1. Person boxes with proximity line & distance indicator (ONLY PERSON)
            for x1, y1, x2, y2, conf, cls_id, name in person_boxes:
                if has_roi:
                    closest_x = max(x1, min(roi_cx, x2))
                    closest_y = max(y1, min(roi_cy, y2))
                    p_dist_px = int(math.hypot(closest_x - roi_cx, closest_y - roi_cy))
                    dist_tag = "IN ROI" if p_dist_px == 0 else f"d={p_dist_px}px"
                    draw_box(disp, x1, y1, x2, y2,
                             f"Person {conf:.2f}  {dist_tag}", COLOR_PERSON)
                    if p_dist_px > 0:
                        cv2.line(disp, (closest_x, closest_y), (roi_cx, roi_cy),
                                 COLOR_PERSON, 1, cv2.LINE_AA)
                else:
                    draw_box(disp, x1, y1, x2, y2,
                             f"Person {conf:.2f}", COLOR_PERSON)

            # 2. Hand boxes with proximity line & distance indicator (ONLY HAND)
            for x1, y1, x2, y2, conf in hand_boxes:
                hcx       = (x1 + x2) // 2
                hcy       = (y1 + y2) // 2
                h_dist_px = int(math.hypot(hcx - roi_cx, hcy - roi_cy))
                if has_roi:
                    draw_box(disp, x1, y1, x2, y2,
                             f"Hand {conf:.2f}  d={h_dist_px}px", COLOR_HAND)
                    cv2.line(disp, (hcx, hcy), (roi_cx, roi_cy),
                             COLOR_HAND, 1, cv2.LINE_AA)
                else:
                    draw_box(disp, x1, y1, x2, y2,
                             f"Hand {conf:.2f}", COLOR_HAND)

            draw_score_panel(disp, latest_scores, score_history)

            active_alert = alert_msg if time.time() < alert_expiry else ""
            device_hud = "GPU" if DEVICE != "cpu" else "CPU"
            draw_hud(disp, cam_fps, infer_fps, conf_threshold,
                     len(person_boxes), len(hand_boxes), latest_motion_area,
                     has_roi, is_roi_drawing, active_alert, device_hud)

            cv2.imshow(window_name, disp)

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("\n[INFO] Window closed.")
                break

            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q')):
                print("\n[INFO] Quit.")
                break

            elif key in (ord('w'), ord('W')):
                conf_threshold = min(MAX_CONF_THRESHOLD,
                                     round(conf_threshold + CONF_STEP, 2))
                alert_msg = f"Confidence Threshold: {conf_threshold:.2f}"
                alert_expiry = time.time() + 1.5
                print(f"[CONF] Up -> {conf_threshold:.2f}")

            elif key in (ord('s'), ord('S')):
                conf_threshold = max(MIN_CONF_THRESHOLD,
                                     round(conf_threshold - CONF_STEP, 2))
                alert_msg = f"Confidence Threshold: {conf_threshold:.2f}"
                alert_expiry = time.time() + 1.5
                print(f"[CONF] Down -> {conf_threshold:.2f}")

            elif key in (ord('r'), ord('R')):
                # Reset ROI to default centered box
                polygon_pts     = DEFAULT_ROI_POLYGON.copy()
                is_roi_drawing  = False
                roi_drawing_pts = []
                save_roi_polygon(polygon_pts)
                alert_msg       = "ROI Reset to Default Centered Box"
                alert_expiry    = time.time() + 3.0
                print("[ROI] Reset to default polygon.")

            elif key in (ord('c'), ord('C')):
                # Clear ROI completely (full-frame mode)
                polygon_pts     = []
                is_roi_drawing  = False
                roi_drawing_pts = []
                save_roi_polygon(polygon_pts)
                alert_msg       = "ROI Cleared (Full-Frame Mode Active)"
                alert_expiry    = time.time() + 3.0
                print("[ROI] Cleared ROI (full-frame mode).")

            elif key in (ord('d'), ord('D')):
                # Toggle interactive Draw ROI mode
                if is_roi_drawing:
                    is_roi_drawing  = False
                    roi_drawing_pts = []
                    alert_msg       = "ROI Draw Cancelled"
                    alert_expiry    = time.time() + 2.0
                    print("[ROI] Drawing cancelled.")
                else:
                    is_roi_drawing  = True
                    roi_drawing_pts = []
                    alert_msg       = "DRAW MODE: Click corners, Right-Click to save, Esc to cancel"
                    alert_expiry    = time.time() + 5.0
                    print("[ROI] Draw mode ON -- left-click pts, right-click to confirm (>=3).")

            elif key == 27:  # ESC key cancels drawing
                if is_roi_drawing:
                    is_roi_drawing  = False
                    roi_drawing_pts = []
                    alert_msg       = "ROI Draw Cancelled"
                    alert_expiry    = time.time() + 2.0
                    print("[ROI] Drawing cancelled.")

            elif key in (8, ord('u'), ord('U')):  # Backspace or 'u' to undo last point
                if is_roi_drawing and roi_drawing_pts:
                    popped = roi_drawing_pts.pop()
                    alert_msg = f"Removed point ({len(roi_drawing_pts)} points left)"
                    alert_expiry = time.time() + 2.0
                    print(f"[ROI] Removed point {popped}")

            elif key < 128 and chr(key) in SAVE_CATEGORIES:
                cat_key          = chr(key)
                label_name, fdr  = SAVE_CATEGORIES[cat_key]
                tag              = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                save_path        = os.path.join(fdr, f"frame_{tag}.jpg")
                cv2.imwrite(save_path, raw_frame)
                alert_msg    = f"Saved [{label_name}]: {os.path.basename(save_path)}"
                alert_expiry = time.time() + 2.5
                print(f"[SAVED] {save_path}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        print("[INFO] Releasing resources...")
        detector.stop()
        cap.release()
        cv2.destroyAllWindows()
        log_handle.close()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
