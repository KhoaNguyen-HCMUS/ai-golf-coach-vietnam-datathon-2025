import os
from pathlib import Path
from transformers import AutoImageProcessor, VitPoseForPoseEstimation
from ultralytics import YOLO
import shutil

# Define paths
ROOT = Path(__file__).resolve().parents[2]  # apps/ml
MODELS_DIR = ROOT / 'models'
VITPOSE_DIR = MODELS_DIR / 'vitpose-base-simple'

def download_models():
    print(f"Creating models directory at: {MODELS_DIR}")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Download ViTPose (Hugging Face)
    print("\n--- Downloading ViTPose model ---")
    model_id = "usyd-community/vitpose-base-simple"
    try:
        print(f"Downloading {model_id} to {VITPOSE_DIR}...")
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = VitPoseForPoseEstimation.from_pretrained(model_id)
        
        processor.save_pretrained(VITPOSE_DIR)
        model.save_pretrained(VITPOSE_DIR)
        print("✓ ViTPose model downloaded successfully.")
    except Exception as e:
        print(f"ERROR downloading ViTPose: {e}")

    # 2. Download YOLO (Ultralytics)
    print("\n--- Downloading YOLO model ---")
    yolo_model_name = "yolov8n.pt"
    yolo_dest_path = MODELS_DIR / yolo_model_name
    
    try:
        # Check if we already have it in src or locally
        src_yolo = Path(__file__).parent / yolo_model_name
        if src_yolo.exists():
            print(f"Found {yolo_model_name} in src, copying to models dir...")
            shutil.copy2(src_yolo, yolo_dest_path)
            print(f"✓ Copied to {yolo_dest_path}")
        else:
            print(f"Downloading {yolo_model_name}...")
            # This triggers download if not cached
            model = YOLO(yolo_model_name) 
            # Move/Copy logic might be tricky since YOLO downloads to current dir or global cache.
            # Best way is to load it, then save it? Or just let it download to current dir and move it.
            # simpler: standard usage `YOLO('yolov8n.pt')` downloads to current dir if not present.
            
            # Let's ensure it's in the right place.
            if os.path.exists(yolo_model_name):
                 shutil.move(yolo_model_name, yolo_dest_path)
                 print(f"✓ Moved downloaded file to {yolo_dest_path}")
            # If it didn't strictly download to cwd (maybe in cache), let's raise a warning
            elif not yolo_dest_path.exists():
                 print(f"WARNING: YOLO model initialized but file not found at {yolo_dest_path} or CWD.")
                 print("Please ensure 'yolov8n.pt' is present in the src folder or successfuly downloaded.")

    except Exception as e:
         print(f"ERROR downloading YOLO: {e}")

    print("\nAll downloads finished.")
    print(f"Models are ready in: {MODELS_DIR}")

if __name__ == "__main__":
    download_models()
