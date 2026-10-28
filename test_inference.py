"""
Batch Inference Tool for Test Frames
=====================================
Runs simultaneous Person/Object detection and Hand detection on all images
in the 'test frames' directory, outputs detailed detection statistics, and
saves annotated visual results to 'test frames/annotated'.
"""

import os
import sys
import colorsys
import cv2
from ultralytics import YOLO

# On Windows, ensure modern Visual C++ runtimes are accessible for PyTorch
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

INPUT_DIR = os.path.join(os.path.dirname(__file__), "test frames")
OUTPUT_DIR = os.path.join(INPUT_DIR, "annotated")

OBJ_MODEL_WEIGHTS = "yolov8n.pt"
HAND_MODEL_WEIGHTS = "hand_yolov8n.pt"
CONF_THRESHOLD = 0.25
IMGSZ = 640

HAND_BOX_COLOR = (0, 165, 255)  # Orange for hands (BGR)


def draw_box(frame, x1, y1, x2, y2, label, color):
    """Draw a bounding box with header label."""
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


def run_batch_inference():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  Running Dual Person & Hand Inference on Test Frames")
    print("=" * 65)
    print(f"[INFO] Input Directory : {INPUT_DIR}")
    print(f"[INFO] Output Directory: {OUTPUT_DIR}")
    print(f"[INFO] Object Model    : {OBJ_MODEL_WEIGHTS}")
    print(f"[INFO] Hand Model      : {HAND_MODEL_WEIGHTS}")
    print(f"[INFO] Confidence Thresh: {CONF_THRESHOLD}")
    print("=" * 65)

    obj_model = YOLO(OBJ_MODEL_WEIGHTS)
    hand_model = YOLO(HAND_MODEL_WEIGHTS)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [f for f in sorted(os.listdir(INPUT_DIR)) if os.path.splitext(f.lower())[1] in valid_exts]

    if not files:
        print("[WARNING] No image files found in test frames directory.")
        return

    print(f"[INFO] Found {len(files)} image(s) to process.\n")

    summary_results = []

    for idx, fname in enumerate(files, 1):
        img_path = os.path.join(INPUT_DIR, fname)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[{idx}/{len(files)}] Could not read '{fname}'. Skipping.")
            continue

        annotated_img = img.copy()

        # 1. Run Person Detection ONLY (class 0)
        res_obj = obj_model.predict(
            source=img,
            imgsz=IMGSZ,
            classes=[0],
            conf=CONF_THRESHOLD,
            verbose=False
        )

        # 2. Run Hand Detection
        res_hand = hand_model.predict(
            source=img,
            imgsz=IMGSZ,
            conf=CONF_THRESHOLD,
            verbose=False
        )

        obj_boxes = res_obj[0].boxes
        hand_boxes = res_hand[0].boxes

        detected_items = []
        person_count = 0
        hand_count = len(hand_boxes)

        # Draw persons
        for b in obj_boxes:
            cid = int(b.cls[0])
            if cid == 0:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf = float(b.conf[0])
                person_count += 1
                label = f"Person {conf:.2f}"
                detected_items.append(label)
                draw_box(annotated_img, x1, y1, x2, y2, label, (0, 255, 0))

        # Draw hands
        for b in hand_boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            conf = float(b.conf[0])
            label = f"Hand {conf:.2f}"
            detected_items.append(label)
            draw_box(annotated_img, x1, y1, x2, y2, label, HAND_BOX_COLOR)

        # Save annotated image
        out_path = os.path.join(OUTPUT_DIR, fname)
        cv2.imwrite(out_path, annotated_img)

        print(f"[{idx}/{len(files)}] Processed: {fname}")
        print(f"      Persons Detected: {person_count}")
        print(f"      Hands Detected  : {hand_count}")
        print(f"      All Detections  : {', '.join(detected_items) if detected_items else 'None'}")
        print(f"      Saved to        : {out_path}\n")

        summary_results.append({
            "filename": fname,
            "persons": person_count,
            "hands": hand_count,
            "items": detected_items,
            "out_path": out_path
        })

    print("=" * 65)
    print("  Inference Complete!")
    print(f"  All annotated results saved in '{OUTPUT_DIR}'")
    print("=" * 65)


if __name__ == "__main__":
    run_batch_inference()
