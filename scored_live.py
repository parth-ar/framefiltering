"""
scored_live.py -- Frame Scoring System for Waste-Collection Camera
=================================================================
Cloud-Deployment Model:
  Frames arriving in the cloud are already triggered by edge motion detection.
  Scoring evaluates Person and Hand presence & proximity to the designated ROI (bin).

Scoring Model (max 100 pts per frame):
  COMPONENT           MAX   LOGIC
  Person Proximity     68   Base 48 + 20*conf when box center is near ROI edge (<= 120px).
                            Generous acceptance distance allows a person near or in front
                            of the bin to reach >= 60 pts on their own if hands are obscured.
                            Smooth falloff over 160px beyond 120px.
  Hand Proximity       32   Up to 32 pts when hand center is near ROI edge (<= 70px).
                            Acts as a booster (total score up to 100 pts).

Deployment Decision:
  >= 60 pts  -> ACCEPTED (Dumping / person activity at bin)
  <  60 pts  -> REJECTED (Passerby / empty scene / person far away)

Controls:
  q         Quit
  w / s     Raise / lower YOLO confidence threshold
  r         Reset ROI polygon back to default centered box
  d         Draw custom ROI polygon (left-click points, right-click to save)
  c         Clear ROI polygon (full-frame mode)
  u / bksp  Undo last point during ROI drawing
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
INFERENCE_SIZE     = 320   # Fast processing configuration (320x320)

# Camera Preview Resolution (720p HD for crisp, high-quality visual preview)
PREVIEW_WIDTH      = 1280
PREVIEW_HEIGHT     = 720

DEFAULT_CONF_THRESHOLD = 0.25
MIN_CONF_THRESHOLD     = 0.05
MAX_CONF_THRESHOLD     = 0.90
CONF_STEP              = 0.05

LOG_FILE = "scored_log.csv"

# ------------------------------------------------------------------
# SCORING CONFIGURATION (Person + Hand Distance to ROI EDGE)
# ------------------------------------------------------------------
# Proximity is computed from the CENTER of bounding boxes to the nearest EDGE
# of the ROI polygon (0px if inside). This prevents issues where large ROIs
# cause center-measured distances to be artificially inflated.
# Weights:
#   Person Proximity (Max 68 pts): Base 48 + 20*conf.
#   Hand Proximity   (Max 32 pts): Up to 32 pts when hand is near the ROI edge.
#   Acceptance Thresh: 60 pts.
# Generous acceptance distance: person center within 120px of the bin edge
# scores >= 60 pts on their own (ACCEPTED), even if hands are obscured.
SCORE_PERSON_BASE = 48   # Base presence points for person
SCORE_PERSON_CONF = 20   # Up to 20 pts based on person confidence (Max Person = 68)
SCORE_HAND_PROX   = 32   # Up to 32 pts for hand detection & proximity booster (Max Hand = 32)

SCORE_HI_THRESH   = 60   # Acceptance threshold
SCORE_MED_THRESH  = 35   # Review / medium threshold

# Proximity zones (measured from bounding box CENTER to nearest ROI EDGE, in pixels)
PERSON_NEAR_ZONE_PX = 120  # Generous acceptance distance: within 120px of bin edge is 100% proximity
PERSON_FALLOFF_PX   = 160  # Distance beyond near-zone at which proximity drops to 0 (280px total)

HAND_NEAR_ZONE_PX   = 70   # Hand center within 70px of bin edge is 100% proximity
HAND_FALLOFF_PX     = 180  # Distance beyond near-zone for hand falloff

# Default ROI polygon (normalised [0-1] x, y)
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
COLOR_SCORE_HI  = (0,  255, 100)
COLOR_SCORE_MED = (0,  200, 255)
COLOR_SCORE_LO  = (0,  100, 255)

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
            "person_score", "hand_score",
            "person_prox", "hand_prox",
            "n_persons", "n_hands",
            "max_person_conf", "max_hand_conf",
            "accepted", "threshold",
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


def point_to_polygon_edge_dist(pt, polygon_pts, frame_w, frame_h):
    """
    Computes Euclidean distance from a pixel point (px, py) to the nearest EDGE
    of the ROI polygon. If the point is INSIDE or ON the polygon boundary, distance is 0.0.
    Also returns the closest point on the polygon boundary (cx_edge, cy_edge).
    """
    if not polygon_pts or len(polygon_pts) < 3:
        return 0.0, (int(round(pt[0])), int(round(pt[1])))

    pts = poly_pixel_pts(polygon_pts, frame_w, frame_h)
    px, py = float(pt[0]), float(pt[1])

    # Check if point is inside or on polygon boundary
    test = cv2.pointPolygonTest(pts, (px, py), measureDist=False)
    if test >= 0:
        return 0.0, (int(round(px)), int(round(py)))

    # Point is outside: find minimum distance across all polygon edge segments
    min_dist = float("inf")
    closest_pt = (int(round(px)), int(round(py)))
    n = len(pts)
    for i in range(n):
        p1 = pts[i].astype(float)
        p2 = pts[(i + 1) % n].astype(float)

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq == 0:
            d = math.hypot(px - p1[0], py - p1[1])
            cp = p1
        else:
            t = max(0.0, min(1.0, ((px - p1[0]) * dx + (py - p1[1]) * dy) / seg_len_sq))
            cp = np.array([p1[0] + t * dx, p1[1] + t * dy])
            d = math.hypot(px - cp[0], py - cp[1])

        if d < min_dist:
            min_dist = d
            closest_pt = (int(round(cp[0])), int(round(cp[1])))

    return min_dist, closest_pt


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
# SCORING ENGINE (Distance to ROI Edge)
# ------------------------------------------------------------------

class FrameScorer:
    """
    Computes a 0-100 frame quality score based on Person and Hand distance to the nearest ROI EDGE.

    Cloud deployment context: frames received by the cloud system are already triggered
    by edge motion sensors/detectors, so computing motion here is redundant.

    Distance to ROI Edge Logic:
      - Person proximity (Max 68 pts):
          * Measured from the CENTER of the person bounding box (pcx, pcy) to the nearest ROI polygon EDGE.
          * If center is INSIDE or within PERSON_NEAR_ZONE_PX (120px) of the ROI edge, proximity is 1.0.
          * Score = (SCORE_PERSON_BASE + SCORE_PERSON_CONF * conf) * prox_factor.
          * With Base=48 and Conf=20, a person standing near or in front of the bin (<= 120px from edge)
            scores >= 60 on their own (ACCEPTED) even if hands are obscured!
          * Smooth falloff over PERSON_FALLOFF_PX (160px); drops to 0 beyond 280px.
      - Hand presence & proximity (Max 32 pts):
          * Measured from the CENTER of the hand bounding box (hcx, hcy) to the nearest ROI EDGE.
          * Within HAND_NEAR_ZONE_PX (70px) of edge or inside, proximity is 1.0.
          * Score = SCORE_HAND_PROX * h_prox * (0.5 + 0.5 * hconf).
          * Boosts the total score up to 100 pts.
      - Total Score: min(round(person_score + hand_score, 1), 100.0)
    """

    def compute(self, person_boxes, hand_boxes, polygon_pts, frame_w, frame_h):
        """
        Returns dict:
          total: float (0-100)
          person_score: float (0-68)
          hand_score: float (0-32)
          person_prox: float (0-1)
          hand_prox: float (0-1)
        """
        has_roi = bool(polygon_pts and len(polygon_pts) >= 3)

        # Scale proximity zones proportionally to frame width so distance criteria
        # remains identical across 640x480, 1280x720 (720p), and 1920x1080 (1080p).
        res_scale   = frame_w / 640.0
        near_zone_p = PERSON_NEAR_ZONE_PX * res_scale
        falloff_p   = PERSON_FALLOFF_PX * res_scale
        near_zone_h = HAND_NEAR_ZONE_PX * res_scale
        falloff_h   = HAND_FALLOFF_PX * res_scale

        # 1. Person Scoring (Max 68 pts) - Center to nearest ROI edge
        person_score = 0.0
        best_p_prox = 0.0
        if person_boxes:
            if has_roi:
                best_cand = 0.0
                for pb in person_boxes:
                    px1, py1, px2, py2, pconf = pb[0], pb[1], pb[2], pb[3], pb[4]
                    pcx = (px1 + px2) / 2
                    pcy = (py1 + py2) / 2
                    p_dist, _ = point_to_polygon_edge_dist((pcx, pcy), polygon_pts, frame_w, frame_h)
                    if p_dist <= near_zone_p:
                        prox = 1.0
                    else:
                        prox = max(0.0, 1.0 - (p_dist - near_zone_p) / falloff_p)
                    cand_score = (SCORE_PERSON_BASE + SCORE_PERSON_CONF * pconf) * prox
                    if cand_score > best_cand:
                        best_cand = cand_score
                        best_p_prox = prox
                person_score = min(best_cand, float(SCORE_PERSON_BASE + SCORE_PERSON_CONF))
            else:
                max_conf = max(b[4] for b in person_boxes)
                best_p_prox = 1.0
                person_score = min(float(SCORE_PERSON_BASE + SCORE_PERSON_CONF * max_conf),
                                   float(SCORE_PERSON_BASE + SCORE_PERSON_CONF))

        # 2. Hand Scoring (Max 32 pts) - Center to nearest ROI edge
        hand_score = 0.0
        best_h_prox = 0.0
        if hand_boxes:
            if has_roi:
                best_cand = 0.0
                for hb in hand_boxes:
                    hx1, hy1, hx2, hy2, hconf = hb[0], hb[1], hb[2], hb[3], hb[4]
                    hcx = (hx1 + hx2) / 2
                    hcy = (hy1 + hy2) / 2
                    h_dist, _ = point_to_polygon_edge_dist((hcx, hcy), polygon_pts, frame_w, frame_h)
                    if h_dist <= near_zone_h:
                        prox = 1.0
                    else:
                        prox = max(0.0, 1.0 - (h_dist - near_zone_h) / falloff_h)
                    cand_score = SCORE_HAND_PROX * prox * (0.5 + 0.5 * hconf)
                    if cand_score > best_cand:
                        best_cand = cand_score
                        best_h_prox = prox
                hand_score = min(best_cand, float(SCORE_HAND_PROX))
            else:
                max_h_conf = max(b[4] for b in hand_boxes)
                best_h_prox = 1.0
                hand_score = SCORE_HAND_PROX * (0.5 + 0.5 * max_h_conf)

        total = min(round(person_score + hand_score, 1), 100.0)

        return {
            "total":        total,
            "person_score": round(person_score, 1),
            "hand_score":   round(hand_score, 1),
            "person_prox":  round(best_p_prox, 2),
            "hand_prox":    round(best_h_prox, 2),
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


def draw_deployment_status(frame, scores, person_count, hand_count, has_roi):
    """
    On-screen field in the camera view displaying whether the current frame
    would be ACCEPTED or REJECTED in a deployed cloud monitoring scene.
    """
    total = scores["total"]
    person_score = scores["person_score"]
    hand_score = scores["hand_score"]

    is_accepted = total >= SCORE_HI_THRESH  # 60

    if is_accepted:
        status_text = "DEPLOYED SCENE: ACCEPTED"
        if hand_count > 0 and hand_score >= 10:
            detail_text = f"Person + Hands at Bin ({total:.0f}>={SCORE_HI_THRESH})"
        else:
            detail_text = f"Person Near Bin [No Hands] ({total:.0f}>={SCORE_HI_THRESH})"
        badge_bg    = (15, 42, 22)
        border_col  = (0, 255, 100)
        text_col    = (0, 255, 100)
    else:
        status_text = "DEPLOYED SCENE: REJECTED"
        if person_count == 0:
            reason = "No Person"
        elif person_score < SCORE_HI_THRESH and hand_count == 0:
            reason = f"Person Far from Bin Edge (Score {total:.0f} < {SCORE_HI_THRESH})"
        else:
            reason = f"Score {total:.0f} < {SCORE_HI_THRESH}"
        detail_text = reason
        badge_bg    = (22, 16, 20)
        border_col  = (70, 70, 130)
        text_col    = (150, 150, 230)

    bx, by = 10, 52
    full_label = f"{status_text} | {detail_text}"
    (tw, th), _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)

    bw = tw + 22
    bh = th + 12

    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), badge_bg, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), border_col, 2 if is_accepted else 1)

    # Status indicator bullet dot
    dot_x = bx + 9
    dot_y = by + bh // 2
    cv2.circle(frame, (dot_x, dot_y), 4, border_col, -1)

    cv2.putText(frame, full_label, (bx + 18, by + th + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, text_col, 1, cv2.LINE_AA)


def draw_score_panel(frame, scores, score_history):
    """
    Compact live scoring HUD on the right side of the frame.
    Footprint: 145px width x ~150px height.
    Displays Person proximity (max 68) and Hand proximity (max 32).
    """
    h, w = frame.shape[:2]

    panel_w = 145
    panel_x = w - panel_w - 6
    panel_y = 52
    panel_h = 148

    # Panel background
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (panel_x - 4, panel_y - 4),
                  (panel_x + panel_w + 4, panel_y + panel_h + 4),
                  (12, 12, 18), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame,
                  (panel_x - 4, panel_y - 4),
                  (panel_x + panel_w + 4, panel_y + panel_h + 4),
                  (50, 50, 70), 1)

    total = scores["total"]
    s_col = score_color(total)
    is_accepted = total >= SCORE_HI_THRESH

    # Row 1: Header + Acceptance mini-pill
    cv2.putText(frame, "SCORE", (panel_x, panel_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 200), 1, cv2.LINE_AA)

    tag_str = "ACCEPT" if is_accepted else "REJECT"
    tag_col = (0, 255, 100) if is_accepted else (120, 120, 180)
    (tw, _), _ = cv2.getTextSize(tag_str, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
    cv2.rectangle(frame, (panel_x + panel_w - tw - 8, panel_y + 1),
                  (panel_x + panel_w, panel_y + 15), (25, 25, 35), -1)
    cv2.rectangle(frame, (panel_x + panel_w - tw - 8, panel_y + 1),
                  (panel_x + panel_w, panel_y + 15), tag_col, 1)
    cv2.putText(frame, tag_str, (panel_x + panel_w - tw - 4, panel_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, tag_col, 1, cv2.LINE_AA)

    # Row 2: Score Number
    score_txt = f"{total:.0f}"
    (bw, _), _ = cv2.getTextSize(score_txt, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
    cv2.putText(frame, score_txt, (panel_x, panel_y + 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, s_col, 2, cv2.LINE_AA)
    cv2.putText(frame, "/100", (panel_x + bw + 4, panel_y + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 160), 1, cv2.LINE_AA)

    # Row 3: Slim Total Bar
    bar_y = panel_y + 46
    fill_w = int(panel_w * total / 100.0)
    cv2.rectangle(frame, (panel_x, bar_y),
                  (panel_x + panel_w, bar_y + 4), (35, 35, 45), -1)
    cv2.rectangle(frame, (panel_x, bar_y),
                  (panel_x + fill_w, bar_y + 4), s_col, -1)

    # Row 4: Component breakdown (Person max 68, Hand max 32) - NO MOTION
    components = [
        ("P", scores["person_score"], SCORE_PERSON_BASE + SCORE_PERSON_CONF, COLOR_PERSON),
        ("H", scores["hand_score"],   SCORE_HAND_PROX, COLOR_HAND),
    ]

    cy = bar_y + 14
    for lbl, val, max_val, color in components:
        pct = val / max_val if max_val > 0 else 0
        fill = int((panel_w - 55) * pct)

        cv2.putText(frame, f"{lbl}: {val:.0f}/{max_val}",
                    (panel_x, cy + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (panel_x + 55, cy - 5),
                      (panel_x + panel_w, cy - 1), (30, 30, 40), -1)
        cv2.rectangle(frame, (panel_x + 55, cy - 5),
                      (panel_x + 55 + fill, cy - 1), color, -1)
        cy += 13

    # Row 5: Compact sparkline history
    spark_y = cy + 4
    spark_h = 28
    cv2.rectangle(frame, (panel_x, spark_y),
                  (panel_x + panel_w, spark_y + spark_h), (20, 20, 28), -1)

    hist = list(score_history)
    if len(hist) >= 2:
        pts_px = []
        for i, s in enumerate(hist):
            sx = panel_x + int(i * panel_w / (SCORE_HISTORY_LEN - 1))
            sy = spark_y + spark_h - int(s / 100.0 * spark_h)
            pts_px.append((sx, sy))
        for i in range(len(pts_px) - 1):
            mid = (hist[i] + hist[i + 1]) / 2
            c_seg = score_color(mid)
            cv2.line(frame, pts_px[i], pts_px[i + 1], c_seg, 1, cv2.LINE_AA)

    # Reference line at 60 (Acceptance threshold)
    ty60 = spark_y + spark_h - int(SCORE_HI_THRESH / 100.0 * spark_h)
    cv2.line(frame, (panel_x, ty60), (panel_x + panel_w, ty60), (0, 180, 80), 1, cv2.LINE_AA)


def draw_hud(frame, cam_fps, infer_fps, threshold,
             person_count, hand_count,
             has_roi, is_roi_drawing, alert_text="", device_tag="CPU"):
    """Top and bottom semi-transparent HUD strips designed to fit cleanly inside 640px."""
    fh, fw = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, 46), (20, 20, 20), -1)
    cv2.rectangle(overlay, (0, fh - 30), (fw, fh), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Top Bar: Line 1 (Cam & AI Stats) + Line 2 (Detections & ROI state)
    roi_status = "DRAWING" if is_roi_drawing else ("ACTIVE" if has_roi else "FULL-FRAME")
    stats_l1 = f"Cam: {cam_fps:4.1f}fps [{fw}x{fh}] | AI: {infer_fps:4.1f}fps [{INFERENCE_SIZE}p {device_tag}] | Conf: {threshold:.2f} (w/s)"
    stats_l2 = f"Persons: {person_count} | Hands: {hand_count} | ROI: {roi_status} | Accept Thresh: {SCORE_HI_THRESH}"

    cv2.putText(frame, stats_l1, (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, stats_l2, (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 255, 200), 1, cv2.LINE_AA)

    # Bottom Bar: Clean Controls (fits in 640px)
    hint = "[q] Quit | [r] Reset ROI | [d] Draw ROI | [c] Clear ROI | [1-3] Save Frame"
    cv2.putText(frame, hint, (10, fh - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.41, (220, 220, 220), 1, cv2.LINE_AA)

    # Alert Banner: High-contrast floating pill (positioned below deployment status badge)
    if alert_text:
        (tw, th), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
        bx, by = 10, 84
        cv2.rectangle(frame, (bx, by), (bx + tw + 16, by + th + 12), (15, 15, 20), -1)
        cv2.rectangle(frame, (bx, by), (bx + tw + 16, by + th + 12), (0, 255, 120), 2)
        cv2.putText(frame, alert_text, (bx + 8, by + th + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 120), 2, cv2.LINE_AA)


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  Frame Scoring System -- Person + Hand Proximity (Cloud Model)")
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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  PREVIEW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_HEIGHT)

    conf_threshold    = DEFAULT_CONF_THRESHOLD
    cam_fps           = 0.0
    prev_time         = time.time()
    alert_msg         = ""
    alert_expiry      = 0.0
    is_roi_drawing    = False
    roi_drawing_pts   = []
    current_mouse_pos = (0, 0)
    frame_w           = PREVIEW_WIDTH
    frame_h           = PREVIEW_HEIGHT
    latest_scores     = {"total": 0.0, "person_score": 0.0,
                         "hand_score": 0.0, "person_prox": 0.0, "hand_prox": 0.0}

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

            # YOLO detection (async background thread)
            detector.update_frame(raw_frame, conf_threshold)
            obj_boxes, hand_boxes, infer_fps, is_new = detector.get_detections()
            person_boxes = [b for b in obj_boxes if b[6] == "person"]

            # Score the current frame based strictly on Person & Hand proximity to ROI
            latest_scores = scorer.compute(
                person_boxes, hand_boxes,
                polygon_pts, frame_w, frame_h)
            score_history.append(latest_scores["total"])

            # Log when YOLO has a new result
            if is_new:
                p_confs    = [b[4] for b in person_boxes]
                h_confs    = [b[4] for b in hand_boxes]
                max_p_conf = max(p_confs) if p_confs else 0.0
                max_h_conf = max(h_confs) if h_confs else 0.0
                ts_str     = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                is_acc     = 1 if latest_scores['total'] >= SCORE_HI_THRESH else 0
                csv_writer.writerow([
                    ts_str,
                    f"{latest_scores['total']:.1f}",
                    f"{latest_scores['person_score']:.1f}",
                    f"{latest_scores['hand_score']:.1f}",
                    f"{latest_scores.get('person_prox', 0.0):.2f}",
                    f"{latest_scores.get('hand_prox', 0.0):.2f}",
                    len(person_boxes),
                    len(hand_boxes),
                    f"{max_p_conf:.3f}",
                    f"{max_h_conf:.3f}",
                    is_acc,
                    f"{conf_threshold:.2f}",
                ])
                log_handle.flush()

            # Build display
            disp = raw_frame.copy()

            draw_roi_polygon(disp, polygon_pts, is_roi_drawing, roi_drawing_pts, current_mouse_pos)

            has_roi = bool(polygon_pts and len(polygon_pts) >= 3)
            if has_roi:
                roi_cx, roi_cy = roi_centroid_px(polygon_pts, frame_w, frame_h)
            else:
                roi_cx, roi_cy = frame_w // 2, frame_h // 2

            res_scale   = frame_w / 640.0
            near_zone_p = PERSON_NEAR_ZONE_PX * res_scale
            near_zone_h = HAND_NEAR_ZONE_PX * res_scale

            # 1. Person boxes with center dot, edge-proximity line & distance indicator
            for x1, y1, x2, y2, conf, cls_id, name in person_boxes:
                if has_roi:
                    pcx = int((x1 + x2) / 2)
                    pcy = int((y1 + y2) / 2)
                    p_dist_px, closest_edge_pt = point_to_polygon_edge_dist(
                        (pcx, pcy), polygon_pts, frame_w, frame_h)
                    p_dist_int = int(round(p_dist_px))
                    dist_tag = "IN ROI" if p_dist_int == 0 else ("AT BIN" if p_dist_px <= near_zone_p else f"d={p_dist_int}px")
                    draw_box(disp, x1, y1, x2, y2,
                             f"Person {conf:.2f}  [{dist_tag}]", COLOR_PERSON)
                    cv2.circle(disp, (pcx, pcy), 5, COLOR_PERSON, -1)
                    if p_dist_int > 0:
                        cv2.circle(disp, closest_edge_pt, 4, COLOR_ROI_EDGE, -1)
                        cv2.line(disp, (pcx, pcy), closest_edge_pt,
                                 COLOR_PERSON, 1, cv2.LINE_AA)
                else:
                    draw_box(disp, x1, y1, x2, y2,
                             f"Person {conf:.2f}", COLOR_PERSON)

            # 2. Hand boxes with center dot, edge-proximity line & distance indicator
            for x1, y1, x2, y2, conf in hand_boxes:
                hcx = int((x1 + x2) / 2)
                hcy = int((y1 + y2) / 2)
                if has_roi:
                    h_dist_px, closest_edge_pt = point_to_polygon_edge_dist(
                        (hcx, hcy), polygon_pts, frame_w, frame_h)
                    h_dist_int = int(round(h_dist_px))
                    dist_tag = "IN ROI" if h_dist_int == 0 else ("AT BIN" if h_dist_px <= near_zone_h else f"d={h_dist_int}px")
                    draw_box(disp, x1, y1, x2, y2,
                             f"Hand {conf:.2f}  [{dist_tag}]", COLOR_HAND)
                    cv2.circle(disp, (hcx, hcy), 4, COLOR_HAND, -1)
                    if h_dist_int > 0:
                        cv2.circle(disp, closest_edge_pt, 3, COLOR_ROI_EDGE, -1)
                        cv2.line(disp, (hcx, hcy), closest_edge_pt,
                                 COLOR_HAND, 1, cv2.LINE_AA)
                else:
                    draw_box(disp, x1, y1, x2, y2,
                             f"Hand {conf:.2f}", COLOR_HAND)

            draw_score_panel(disp, latest_scores, score_history)

            active_alert = alert_msg if time.time() < alert_expiry else ""
            device_hud = "GPU" if DEVICE != "cpu" else "CPU"
            draw_hud(disp, cam_fps, infer_fps, conf_threshold,
                     len(person_boxes), len(hand_boxes),
                     has_roi, is_roi_drawing, active_alert, device_hud)
            draw_deployment_status(disp, latest_scores, len(person_boxes),
                                   len(hand_boxes), has_roi)

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
