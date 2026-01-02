# Project Setup

## Folder structure

The repo is split into two main roots:

- **`envs/`** – Conda environment definitions (e.g., `datathon2025.yml`).
- **`apps/`** – Frontend, backend, and machine-learning source trees.

## Requirements

- Node 20+, pnpm 10+
- Conda 23+

## Install JavaScript dependencies

```bash
pnpm install
```

## Activate the Conda environment

```bash
conda env create -f envs/datathon2025.yml
conda activate datathon2025
```

## Run project

### Full stack

```bash
pnpm dev
```

### Frontend only

```bash
pnpm fe dev
```

## Outcome

- Web: https://localhost:4000
- Server: https://localhost:5000

## ML pipeline scripts (apps/ml/src)

- [adjust_event_frames.py](apps/ml/src/adjust_event_frames.py): reads GolfDB CSV metadata, queries each video’s FPS, and shifts all `events` by `offset_seconds * fps` to keep phase timestamps aligned after trimming.
- [evaluate.py](apps/ml/src/evaluate.py): normalizes the user skeleton, computes DTW distances against reference templates for each camera view, applies rule-based penalties (chicken wing, head bobbing, soft lead leg), and exports a JSON report with scores plus feedback.
- [extract_skelectons.py](apps/ml/src/extract_skelectons.py): runs YOLOv8 Pose on raw videos, keeps the actor with the highest confidence per frame, and stores full keypoints as `.npy` tensors for downstream tasks.
- [organize_videos.py](apps/ml/src/organize_videos.py): walks the original TDTU folder tree, copies every `.mov` into normalized `id.mp4` files, and emits a CSV metadata file containing `place/band/view` and the original filename.
- [prepare_videos.py](apps/ml/src/prepare_videos.py): resizes every GolfDB/TDTU video to 720p height, applies adaptive CLAHE-based brightness enhancement, and produces clean clips reused by all extraction/evaluation steps (no MediaPipe handoff anymore).
- [rulebased_detector.py](apps/ml/src/rulebased_detector.py): infers Address/Top/Impact/Finish from wrist trajectories inside `.npy` skeletons and can optionally render debug videos with phase markers.
- [score_mapper_trainer.py](apps/ml/src/score_mapper_trainer.py): collects model scores versus ground-truth bands, fits a linear + double-power warp to map scores back to the 0–10 scale, and reports per-band accuracy.

### Suggested execution order

```bash
# 1. Normalize GolfDB + TDTU videos
python apps/ml/src/prepare_videos.py

# 2. Reorganize TDTU assets and emit metadata (update DATA_FOLDER / OUTPUT_* first)
python apps/ml/src/organize_videos.py

# 3. Extract skeletons from the normalized videos
python apps/ml/src/extract_skelectons.py

# 4. Detect Address/Top/Impact/Finish phases (optional debug video)
python apps/ml/src/rulebased_detector.py

# 5. Offset GolfDB CSV event frames (adjust paths beforehand)
python apps/ml/src/adjust_event_frames.py

# 6. Evaluate one user skeleton against references
python apps/ml/src/evaluate.py

# 7. Train the score-to-band mapper
python apps/ml/src/score_mapper_trainer.py
```

> Before running any script, update its hard-coded paths (video folders, skeleton directories, CSV/JSON outputs, YOLO weights, etc.) and make sure the `datathon2025` environment has all dependencies (OpenCV, Ultralytics, NumPy, Pandas, SciPy, ...).

### Related notebooks

- [apps/ml/notebooks/eda_dataset_comparison.ipynb](apps/ml/notebooks/eda_dataset_comparison.ipynb) contains the TDTU vs GolfDB exploratory analysis: camera-view distribution, environment split, skill histogram, and skeleton variance evidence that GolfDB is the authoritative reference set.

## IoT edge node (Device/)

- [Device/src/main.cpp](Device/src/main.cpp#L1-L47) is the ESP32 entrypoint: after receiving a `START` MQTT command it streams IMU data, detects impacts, and publishes the result payload.
- [Device/src/sensors/IMUSensor.*](Device/src/sensors/IMUSensor.cpp#L1-L35) wraps the MPU6050, converts raw readings to acceleration (g) and angular velocity (deg/s), and exposes magnitude helpers.
- [Device/src/detection/ImpactDetector.*](Device/src/detection/ImpactDetector.cpp#L1-L18) compares acceleration/gyro magnitudes against `ACC_THRESHOLD`/`GYRO_THRESHOLD` with a cooldown window to prevent double-triggering.
- [Device/src/communication/MqttClient.*](Device/src/communication/MqttClient.cpp#L1-L69) manages Wi-Fi, MQTT lifecycle, command subscription (`MQTT_TOPIC_CMD`), and publishes JSON results on `MQTT_TOPIC_RESULT` (timestamps, delay, magnitudes).
- Hardware/network constants live in [Device/src/config.h](Device/src/config.h); update SSID, broker, topics, and thresholds before flashing the firmware.

### PlatformIO build & flash quickstart

```bash
cd Device
pio run              # build firmware
pio run -t upload    # flash ESP32 (ESP32 DOIT Devkit V1 per platformio.ini)
pio device monitor   # view serial logs at 115200 bps
```

> Requires PlatformIO CLI (`pip install platformio` or the VS Code extension). Configure `config.h` with real Wi-Fi/MQTT credentials, then reset the board so the node reconnects and waits for commands from the backend/web.
