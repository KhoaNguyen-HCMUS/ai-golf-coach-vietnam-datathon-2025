import sys
from pathlib import Path
import os
import torch

# Add src to path to import extract_skelectons variables
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'apps' / 'ml' / 'src'
sys.path.append(str(SRC))

try:
    # We want to check constants from extract_skelectons
    # But it has code that runs on import if not careful, luckily it's under if __name__ == "__main__"
    # However, global variables are executed.
    import extract_skelectons as es
    
    print(f"Checking configuration in extract_skelectons.py...")
    print(f"VITPOSE_MODEL_ID: {es.VITPOSE_MODEL_ID}")
    print(f"YOLO_DET_MODEL: {es.YOLO_DET_MODEL}")
    
    if not os.path.exists(es.VITPOSE_MODEL_ID):
        print(f"FAIL: ViTPose path does not exist: {es.VITPOSE_MODEL_ID}")
        sys.exit(1)
        
    if not os.path.exists(es.YOLO_DET_MODEL):
        print(f"FAIL: YOLO path does not exist: {es.YOLO_DET_MODEL}")
        sys.exit(1)
        
    print("Files exist. Attempting to load models locally...")
    
    from transformers import AutoImageProcessor, VitPoseForPoseEstimation
    from ultralytics import YOLO
    
    # Load ViTPose
    processor = AutoImageProcessor.from_pretrained(es.VITPOSE_MODEL_ID, local_files_only=True)
    model = VitPoseForPoseEstimation.from_pretrained(es.VITPOSE_MODEL_ID, local_files_only=True)
    print("✓ ViTPose loaded with local_files_only=True")
    
    # Load YOLO
    yolo = YOLO(es.YOLO_DET_MODEL)
    print("✓ YOLO loaded from local file")
    
    print("\nSUCCESS: Offline setup verified.")

except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Verification Failed: {e}")
    sys.exit(1)
