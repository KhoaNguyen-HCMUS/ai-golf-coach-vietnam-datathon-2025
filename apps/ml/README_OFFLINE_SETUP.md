# Offline Model Setup Guide for ViTPose & YOLO

This guide explains how to set up the ML models for offline execution, as required by the Datathon submission rules.

## Prerequisites

Ensure you are in the project root and have the Conda environment activated:

```bash
conda activate datathon2025
```

## 1. One-Time Setup (Requires Internet)

Before running the main script offline, you must download the models to your local machine. We have a script that handles this.

Run the following command *with internet access*:

```bash
python apps/ml/src/download_models.py
```

This will:
1.  Download the **ViTPose** model (`usyd-community/vitpose-base-simple`) to `apps/models/vitpose-base-simple`.
2.  Copy or download the **YOLO** model (`yolov8n.pt`) to `apps/models/yolov8n.pt`.

## 2. Verification

To ensure everything is set up correctly and works **without internet**, you can run the verification script:

```bash
python apps/ml/src/verify_offline_setup.py
```

If successful, you will see:
> SUCCESS: Offline setup verified.

## 3. Running Skeleton Extraction

Once the models are present in `apps/models`, the main extraction script will automatically use them. It will **NOT** attempt to connect to Hugging Face or download anything.

```bash
python apps/ml/src/extract_skelectons.py
```

## 4. Important for Submission

**CRITICAL**: When packaging your solution for submission, make sure the `apps/models` directory is included in your zip file. The judges will run the code in an offline environment, so these files must be present.

### Directory Structure Reference
```
apps/
  ml/
    src/
      download_models.py
      extract_skelectons.py
      verify_offline_setup.py
    models/                 <-- MUST BE INCLUDED IN SUBMISSION
      vitpose-base-simple/
        config.json
        pytorch_model.bin
        ...
      yolov8n.pt
```
