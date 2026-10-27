"""
Live Person & Hand Detection Tool for Waste-Collection Camera System
====================================================================
This script captures live video from your default webcam and simultaneously
detects people (and other objects) using YOLOv8n and hands using a pre-trained
hand detection model (hand_yolov8n.pt).

Performance Architecture:
  - Decoupled Multi-Threading: Camera capture & viewfinder display run at
    native 30 FPS on the main thread without stutter or lag.
  - Asynchronous AI Worker: Deep learning models run continuously in the
    background at 320x320 resolution, updating bounding boxes in real-time.

Controls:
  - q       : Quit application
  - w / s   : Increase / decrease confidence threshold by 0.05 (0.05 - 0.90)
  - 1       : Save raw frame to frames/no_person/
  - 2       : Save raw frame to frames/person_no_dump/
  - 3       : Save raw frame to frames/dumping/
"""

import os
import sys
import time
import csv
import colorsys
import threading
from datetime import datetime

# On Windows, ensure modern Visual C++ 2015-2022 runtimes are accessible for PyTorch
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
import torch
from ultralytics import YOLO

# Optimize PyTorch CPU threading for laptops / dual-core CPUs
torch.set_num_threads(min(4, os.cpu_count() or 4))


# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
# Use YOLOv8 Nano for 4x faster CPU inference over YOLOv8 Small
MODEL_WEIGHTS = "yolov8n.pt"            # COCO model (person + 79 objects)
HAND_MODEL_WEIGHTS = "hand_yolov8n.pt"  # Pretrained YOLOv8 hand detector
HAND_MODEL_URL = "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt"

# Inference resolution: 320x320 processes 75% fewer pixels than 640x640,
# providing dramatic speedups on CPU while keeping accurate detections.
INFERENCE_SIZE = 320

# Colors for bounding boxes (BGR format)
HAND_BOX_COLOR = (0, 165, 255)       # High-visibility vibrant orange for hands

# Confidence threshold settings
DEFAULT_CONF_THRESHOLD = 0.25
MIN_CONF_THRESHOLD = 0.05
MAX_CONF_THRESHOLD = 0.90
CONF_STEP = 0.05

# Log file path
LOG_FILE = "log.csv"

# Destination directories for saved raw frames (keys: '1', '2', '3')
SAVE_CATEGORIES = {
    "1": ("no_person", os.path.join("frames", "no_person")),
    "2": ("person_no_dump", os.path.join("frames", "person_no_dump")),
    "3": ("dumping", os.path.join("frames", "dumping")),
}


def ensure_directories():
    """Ensure all required folders exist before capturing frames."""
    for _, folder_path in SAVE_CATEGORIES.values():
        os.makedirs(folder_path, exist_ok=True)


def ensure_hand_weights():
    """Ensure hand_yolov8n.pt exists, downloading from Hugging Face if missing."""
    if not os.path.isfile(HAND_MODEL_WEIGHTS):
        print(f"[INFO] '{HAND_MODEL_WEIGHTS}' not found. Downloading from Hugging Face...")
        import urllib.request
        try:
            urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_WEIGHTS)
            print(f"[INFO] Downloaded '{HAND_MODEL_WEIGHTS}' successfully.")
        except Exception as e:
            print(f"[WARNING] Automatic download failed: {e}. Trying curl.exe...")
            os.system(f'curl.exe -L -o "{HAND_MODEL_WEIGHTS}" "{HAND_MODEL_URL}"')


def init_logger():
    """
    Initialize the CSV log file. If the file does not exist,
    create it and write the column header.
    """
    file_exists = os.path.isfile(LOG_FILE)
    log_handle = open(LOG_FILE, mode="a", newline="", encoding="utf-8")
    writer = csv.writer(log_handle)
    
    if not file_exists:
        writer.writerow([
            "timestamp",
            "number_of_persons",
            "number_of_hands",
            "max_person_confidence",
            "max_hand_confidence",
            "current_threshold"
        ])
        log_handle.flush()
        
    return log_handle, writer


def draw_box(frame, x1, y1, x2, y2, label, color):
    """Draw a bounding box with filled header and text label."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (txt_w, txt_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    box_top = max(y1, txt_h + 10)
    cv2.rectangle(
        frame,
        (x1, box_top - txt_h - 8),
        (x1 + txt_w + 6, box_top),
        color,
        -1
    )
    cv2.putText(
        frame,
        label,
        (x1 + 3, box_top - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )


def draw_hud(frame, cam_fps, infer_fps, threshold, person_count, hand_count, alert_text=""):
    """
    Draw a heads-up display (HUD) banner with Viewfinder FPS, AI Inference FPS,
    current threshold, detection counts, and hotkey hints.
    """
    height, width, _ = frame.shape

    # Semi-transparent top bar for stats
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 42), (20, 20, 20), -1)
    
    # Semi-transparent bottom bar for keyboard hints
    cv2.rectangle(overlay, (0, height - 35), (width, height), (20, 20, 20), -1)
    
    # Blend overlay with original frame (alpha = 0.65)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Top bar text: Viewfinder FPS, AI FPS, Threshold, Persons, Hands
    stats_text = (
        f"Cam: {cam_fps:04.1f} FPS  |  AI: {infer_fps:04.1f} FPS  |  "
        f"Thresh: {threshold:.2f} (w/s)  |  Persons: {person_count}  |  Hands: {hand_count}"
    )
    cv2.putText(frame, stats_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)

    # Bottom bar text: Controls
    hint_text = "[q] Quit  |  [w/s] +/- Thresh  |  [1] No Person  |  [2] Person No Dump  |  [3] Dumping"
    cv2.putText(frame, hint_text, (15, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    # If a frame was recently saved, show notification alert
    if alert_text:
        cv2.putText(frame, alert_text, (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


class AsyncDetector:
    """
    Background worker that runs YOLO object and hand detection asynchronously.
    This frees the camera capture and UI loop to run at maximum camera frame rate (30 FPS)
    without stuttering.
    """
    def __init__(self, obj_model, hand_model, imgsz=320):
        self.obj_model = obj_model
        self.hand_model = hand_model
        self.imgsz = imgsz
        
        self.lock = threading.Lock()
        self.next_frame = None
        self.conf_threshold = DEFAULT_CONF_THRESHOLD
        
        # Parsed detection storage
        self.latest_obj_boxes = []   # list of (x1, y1, x2, y2, conf, cls_id, class_name)
        self.latest_hand_boxes = []  # list of (x1, y1, x2, y2, conf)
        self.infer_fps = 0.0
        self.has_new_result = False
        self.running = True

        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def update_frame(self, frame, conf):
        """Submit the freshest raw camera frame to be analyzed."""
        with self.lock:
            self.next_frame = frame
            self.conf_threshold = conf

    def get_detections(self):
        """Retrieve the latest detection boxes and inference rate atomically."""
        with self.lock:
            is_new = self.has_new_result
            self.has_new_result = False
            return self.latest_obj_boxes, self.latest_hand_boxes, self.infer_fps, is_new

    def stop(self):
        """Signal the worker thread to exit."""
        self.running = False
        self.thread.join(timeout=1.0)

    def _worker_loop(self):
        """Worker thread main loop: continually runs detection on incoming frames."""
        while self.running:
            frame_to_process = None
            conf = DEFAULT_CONF_THRESHOLD

            with self.lock:
                if self.next_frame is not None:
                    frame_to_process = self.next_frame
                    self.next_frame = None  # Consume frame
                    conf = self.conf_threshold

            if frame_to_process is None:
                time.sleep(0.005)
                continue

            t_start = time.time()
            try:
                # 1. Run detection for General objects / Persons
                res_obj = self.obj_model.predict(
                    source=frame_to_process,
                    imgsz=self.imgsz,
                    conf=conf,
                    verbose=False
                )

                # 2. Run detection for Hands
                res_hand = self.hand_model.predict(
                    source=frame_to_process,
                    imgsz=self.imgsz,
                    conf=conf,
                    verbose=False
                )

                # Parse object boxes
                parsed_obj = []
                for b in res_obj[0].boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    c = float(b.conf[0])
                    cid = int(b.cls[0])
                    cname = self.obj_model.names[cid]
                    parsed_obj.append((x1, y1, x2, y2, c, cid, cname))

                # Parse hand boxes
                parsed_hand = []
                for b in res_hand[0].boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    c = float(b.conf[0])
                    parsed_hand.append((x1, y1, x2, y2, c))

                duration = time.time() - t_start
                cur_fps = 1.0 / duration if duration > 0 else 0.0

                with self.lock:
                    self.latest_obj_boxes = parsed_obj
                    self.latest_hand_boxes = parsed_hand
                    self.infer_fps = (0.8 * self.infer_fps) + (0.2 * cur_fps) if self.infer_fps > 0 else cur_fps
                    self.has_new_result = True

            except Exception as e:
                time.sleep(0.01)


def main():
    print("=" * 60)
    print("  Waste-Collection Camera: Live Person & Hand Detector")
    print("=" * 60)

    # 1. Ensure required folders exist
    ensure_directories()

    # 2. Initialize CSV logger
    log_handle, csv_writer = init_logger()
    print(f"[INFO] Logging frame detections to '{LOG_FILE}'")

    # 3. Load YOLO models
    print(f"[INFO] Loading Person/Object model '{MODEL_WEIGHTS}' (YOLOv8 Nano)...")
    try:
        model = YOLO(MODEL_WEIGHTS)
    except Exception as e:
        print(f"[ERROR] Failed to load object model: {e}")
        log_handle.close()
        sys.exit(1)

    print(f"[INFO] Loading Hand detection model '{HAND_MODEL_WEIGHTS}'...")
    ensure_hand_weights()
    try:
        hand_model = YOLO(HAND_MODEL_WEIGHTS)
    except Exception as e:
        print(f"[ERROR] Failed to load hand model: {e}")
        log_handle.close()
        sys.exit(1)

    # 4. Start Asynchronous Detection Worker
    print(f"[INFO] Initializing Async AI Detector (Resolution: {INFERENCE_SIZE}x{INFERENCE_SIZE})...")
    detector = AsyncDetector(model, hand_model, imgsz=INFERENCE_SIZE)

    # 5. Open default webcam (device 0)
    print("[INFO] Opening default webcam (device 0)...")
    if sys.platform == "win32":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        print("        Please check that:")
        print("        1. Your webcam is plugged in and recognized by Windows.")
        print("        2. Camera access permissions are enabled in Windows Settings.")
        print("        3. No other application (Zoom, Teams, browser) is locking the webcam.")
        detector.stop()
        log_handle.close()
        sys.exit(1)

    # Set camera resolution for smooth performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # State variables
    conf_threshold = DEFAULT_CONF_THRESHOLD
    cam_fps = 0.0
    prev_time = time.time()
    
    # Notification alert banner state
    alert_message = ""
    alert_expiry_time = 0.0

    window_name = "Live Person & Hand Detection - Waste System Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n[READY] Camera feed active. Press 'q' in the camera window to exit.\n")

    try:
        while True:
            success, raw_frame = cap.read()
            if not success:
                print("[WARNING] Failed to grab frame from webcam. Retrying...")
                time.sleep(0.05)
                continue

            # Calculate Camera Viewfinder FPS
            curr_time = time.time()
            time_diff = curr_time - prev_time
            prev_time = curr_time
            if time_diff > 0:
                current_fps = 1.0 / time_diff
                # Exponential moving average for smooth display
                cam_fps = (0.90 * cam_fps) + (0.10 * current_fps) if cam_fps > 0 else current_fps

            # Submit current frame to AI detector (non-blocking)
            detector.update_frame(raw_frame, conf_threshold)

            # Retrieve latest detections and AI speed (instant, non-blocking)
            obj_boxes, hand_boxes, infer_fps, is_new_result = detector.get_detections()

            # Separate person detections
            person_boxes = [b for b in obj_boxes if b[6] == "person"]
            person_count = len(person_boxes)
            hand_count = len(hand_boxes)

            # Log to CSV whenever a new AI detection completes
            if is_new_result:
                person_confs = [b[4] for b in person_boxes]
                hand_confs = [b[4] for b in hand_boxes]
                max_person_conf = max(person_confs) if person_confs else 0.0
                max_hand_conf = max(hand_confs) if hand_confs else 0.0

                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                csv_writer.writerow([
                    timestamp_str,
                    person_count,
                    hand_count,
                    f"{max_person_conf:.3f}",
                    f"{max_hand_conf:.3f}",
                    f"{conf_threshold:.2f}"
                ])
                log_handle.flush()

            # Create display copy so annotations do NOT alter raw frame
            display_frame = raw_frame.copy()

            # Draw general object bounding boxes (Persons, items)
            for x1, y1, x2, y2, conf, cls_id, class_name in obj_boxes:
                if class_name == "person":
                    color = (0, 255, 0)   # Clean bright green for persons
                else:
                    hue = (cls_id * 37) % 180
                    r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(hue / 180.0, 0.9, 0.95)]
                    color = (b, g, r)

                draw_box(display_frame, x1, y1, x2, y2, f"{class_name.capitalize()} {conf:.2f}", color)

            # Draw hand bounding boxes (high-visibility orange)
            for x1, y1, x2, y2, conf in hand_boxes:
                draw_box(display_frame, x1, y1, x2, y2, f"Hand {conf:.2f}", HAND_BOX_COLOR)

            # Check if active alert message has expired
            active_alert = alert_message if time.time() < alert_expiry_time else ""

            # Draw HUD overlay (Camera FPS, AI FPS, threshold, persons, hands, hints, alerts)
            draw_hud(display_frame, cam_fps, infer_fps, conf_threshold, person_count, hand_count, active_alert)

            # Show the annotated frame
            cv2.imshow(window_name, display_frame)

            # Check if user clicked window close button ('X')
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("\n[INFO] Window closed by user. Exiting...")
                break

            # Check for keyboard inputs (1 millisecond wait)
            key = cv2.waitKey(1) & 0xFF

            # Handle 'q' to quit
            if key == ord('q') or key == ord('Q'):
                print("\n[INFO] 'q' pressed. Exiting...")
                break

            # Handle 'w' to raise threshold
            elif key == ord('w') or key == ord('W'):
                conf_threshold = min(MAX_CONF_THRESHOLD, round(conf_threshold + CONF_STEP, 2))
                print(f"[THRESHOLD] Raised confidence threshold to {conf_threshold:.2f}")

            # Handle 's' to lower threshold
            elif key == ord('s') or key == ord('S'):
                conf_threshold = max(MIN_CONF_THRESHOLD, round(conf_threshold - CONF_STEP, 2))
                print(f"[THRESHOLD] Lowered confidence threshold to {conf_threshold:.2f}")

            # Handle '1', '2', '3' to save raw frame
            elif chr(key) in SAVE_CATEGORIES:
                cat_key = chr(key)
                label_name, folder_path = SAVE_CATEGORIES[cat_key]
                time_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"frame_{time_tag}.jpg"
                save_path = os.path.join(folder_path, filename)

                # CRITICAL: Save unmodified raw_frame (NO drawn boxes or HUD)
                cv2.imwrite(save_path, raw_frame)

                alert_message = f"Saved: {save_path}"
                alert_expiry_time = time.time() + 2.0  # Display alert for 2 seconds
                print(f"[SAVED] Raw frame saved to [{label_name}]: {save_path}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C). Exiting...")

    finally:
        # Cleanup and release resources cleanly
        print("[INFO] Stopping AI worker, releasing webcam, and closing windows...")
        detector.stop()
        cap.release()
        cv2.destroyAllWindows()
        log_handle.close()
        print("[INFO] Done. Log file closed successfully.")


if __name__ == "__main__":
    main()
