# Live Person Detection Test Tool

A lightweight, beginner-friendly Python tool for testing real-time person detection on a webcam using a pretrained YOLOv8 model (`yolov8s.pt`).

This tool is designed to test detector performance and confidence scores for a waste-collection camera system before deployment to target hardware (such as a Raspberry Pi).

---

## Features

- **Pretrained YOLOv8s Detection**: Uses Ultralytics YOLOv8 with weights trained on COCO to detect **only persons** (Class ID `0`). No training or fine-tuning required.
- **Real-Time Display**: Live OpenCV window with bounding boxes, confidence scores, current FPS, and current confidence threshold.
- **Interactive Threshold Tuning**: Adjust the confidence threshold on the fly using keyboard keys (`w` / `s`).
- **Single-Key Raw Frame Capture**: Save pristine raw frames (without drawn boxes or overlays) directly into categorized folders (`1`, `2`, `3`) with timestamps.
- **Automatic Per-Frame Logging**: Logs timestamp, person count, max confidence, and current threshold to `log.csv`.
- **Clean Error Handling**: Clear messages if the webcam cannot be opened, with safe camera release on exit.

---

## Project Structure

```text
Frame Filtering/
│
├── live_person.py      # Real-time Person & Hand detector
├── scored_live.py      # Multi-factor frame scoring (Person + Hand Proximity + Motion)
├── requirements.txt    # Project dependencies (ultralytics, opencv-python)
├── README.md           # Instructions and documentation
├── log.csv             # Generated automatically: detection logs
├── scored_log.csv      # Generated automatically: composite frame scores
├── roi_polygon.json    # Saved ROI polygon coordinates
│
└── frames/             # Generated automatically: saved raw frames
    ├── no_person/      # Key [1]: frames without any person
    ├── person_no_dump/ # Key [2]: frames with a person, but not dumping
    └── dumping/        # Key [3]: frames showing waste dumping activity
```

---

## Windows Setup & Installation

Ensure you have **Python 3.8+** installed on Windows.

### 1. Open Terminal / PowerShell
Open **PowerShell** or **Command Prompt** and navigate to this folder:
```powershell
cd "c:\Users\Āḍṁīṇ\Desktop\Solid-Waste\Frame Filtering"
```

### 2. (Recommended) Create and Activate a Virtual Environment
A virtual environment keeps your project dependencies isolated:
```powershell
python -m venv venv
.\venv\Scripts\activate
```
*(You will see `(venv)` in your command prompt once activated).*

### 3. Install Dependencies
Install the required packages using `requirements.txt`:
```powershell
$env:PYTHONUTF8 = '1'
pip install -r requirements.txt
```
> **Tip for Windows**: `$env:PYTHONUTF8 = '1'` ensures Python and pip handle Unicode filenames and paths without encoding errors.
> **Note**: On the first run, `ultralytics` will also automatically download the official `yolov8s.pt` model weights (~22 MB).


---

## Running the Application

Run the script from your terminal:
```powershell
python live_person.py
```

A window titled **"Live Person Detection - Waste System Test"** will open showing your webcam feed.

---

## Keyboard Controls

| Key | Action | Details |
|---|---|---|
| **`q`** | **Quit** | Closes the camera, flushes the log, and exits cleanly. |
| **`w`** | **Increase Threshold** | Raises detection confidence threshold by `+0.05` (max: `0.90`). |
| **`s`** | **Decrease Threshold** | Lowers detection confidence threshold by `-0.05` (min: `0.05`). |
| **`1`** | **Save: No Person** | Saves the current **unmodified raw frame** into `frames/no_person/`. |
| **`2`** | **Save: Person No Dump** | Saves the current **unmodified raw frame** into `frames/person_no_dump/`. |
| **`3`** | **Save: Dumping** | Saves the current **unmodified raw frame** into `frames/dumping/`. |

### Frame Scoring & ROI Controls (`scored_live.py`)

| Key | Action | Details |
|---|---|---|
| **`r`** | **Reset ROI to Default** | Restores the default centered rectangular ROI polygon and displays confirmation. |
| **`c`** | **Clear ROI** | Removes the ROI constraint completely (switches to Full-Frame mode). |
| **`d`** | **Draw Custom ROI** | Toggles interactive drawing mode. Left-click to add points, right-click to confirm (min 3 points). |
| **`Esc`** | **Cancel Draw Mode** | Cancels active ROI drawing without saving. |
| **`Backspace` / `u`** | **Undo Point** | Removes the last clicked vertex while in drawing mode. |

---

## Code Walkthrough: How It Works

The entire tool is contained within [`live_person.py`](file:///c:/Users/%C4%80%E1%B8%8D%E1%B9%81%C4%AB%E1%B9%87/Desktop/Solid-Waste/Frame%20Filtering/live_person.py). Here is a step-by-step breakdown of its components:

### 1. Configuration & Constants
- Defines default values such as `MODEL_WEIGHTS = "yolov8s.pt"` and `PERSON_CLASS_ID = 0` (the COCO dataset index for `person`).
- Sets threshold bounds (`0.05` to `0.90`, default `0.25`) and maps keys `'1'`, `'2'`, `'3'` to their respective subfolder paths.

### 2. Directory Creation & Logging Initialization
- `ensure_directories()`: Uses `os.makedirs(..., exist_ok=True)` to ensure `frames/no_person`, `frames/person_no_dump`, and `frames/dumping` exist.
- `init_logger()`: Opens `log.csv` in append mode. If the file is brand new, it writes the CSV header row (`timestamp`, `number_of_persons`, `max_person_confidence`, `current_threshold`).

### 3. Model Loading
- `model = YOLO("yolov8s.pt")`: Initializes the YOLOv8 small neural network. If the weights are not found locally, Ultralytics downloads them from the official repository automatically.

### 4. Webcam Initialization & Error Handling
- `cap = cv2.VideoCapture(0)`: Connects to your default webcam (index `0`).
- If `not cap.isOpened()`, the script immediately displays helpful troubleshooting advice (camera permissions, device unplugged, or camera in use by Zoom/Teams) and exits safely without crashing.

### 5. The Main Video Loop
For each frame captured from the camera:
1. **FPS Calculation**: Measures elapsed time between frames using `time.time()` and smooths the value using an exponential moving average for steady display.
2. **YOLO Detection**: Calls `model.predict(source=raw_frame, classes=[0], conf=conf_threshold, verbose=False)`. Passing `classes=[0]` instructs the model to ignore cars, animals, and other objects, focusing only on human detections.
3. **Data Logging**: Records the frame's timestamp, total persons detected, maximum confidence score, and the active threshold into `log.csv`, calling `.flush()` so data is written immediately.
4. **Drawing Annotations (HUD)**:
   - Preserves `raw_frame` in its original form.
   - Copies `raw_frame` into `display_frame` where bounding boxes, confidence tags, FPS, and hotkey hints are drawn.
   - Uses `cv2.imshow(...)` to display `display_frame`.
5. **Keyboard Input Handling (`cv2.waitKey(1)`)**:
   - `'w'` and `'s'`: Updates `conf_threshold` between `0.05` and `0.90` and updates future predictions immediately.
   - `'1'`, `'2'`, `'3'`: Uses `cv2.imwrite()` to save `raw_frame` (unmarked) with a timestamped filename (e.g., `frame_20261023_185200_123.jpg`), and flashes an on-screen confirmation banner for 2 seconds.
   - `'q'`: Breaks the loop.

### 6. Resource Cleanup
- Located in a `finally:` block: ensures `cap.release()`, `cv2.destroyAllWindows()`, and `log_handle.close()` are always executed, even if the program is closed with `Ctrl+C`.

---

## Log File Format (`log.csv`)

Each frame appends a row:
```csv
timestamp,number_of_persons,max_person_confidence,current_threshold
2026-10-23 18:52:01.120,1,0.874,0.25
2026-10-23 18:52:01.155,1,0.891,0.25
2026-10-23 18:52:01.190,0,0.000,0.25
```

---

## Troubleshooting

- **Webcam cannot be opened (`[ERROR] Could not open webcam`)**:
  - Open **Windows Settings > Privacy & security > Camera** and make sure **Camera access** and **Let desktop apps access your camera** are toggled **On**.
  - Ensure no other application (like Zoom, Teams, Skype, or a web browser) is using the webcam.
  - If you have an external webcam, try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`.
- **Low FPS**:
  - `yolov8s` runs inference on your CPU if no CUDA-compatible GPU is available. Resolution is set to 640x480 for balanced real-time speed.
