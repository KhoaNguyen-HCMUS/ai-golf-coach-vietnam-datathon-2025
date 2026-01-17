"""
Main Prediction Script - Extract Skeletons & Predict
Input: Folder containing video files (.mp4, .mov) OR .npy skeleton files
Output: CSV with columns: file_name, score
"""

import numpy as np
import pandas as pd
import os
import joblib
import json
from glob import glob
from tqdm import tqdm
from extract_features_biomech_augmented import GolfFeatureExtractorBiomech


# ============================================================================
# CONFIGURATION
# ============================================================================

# Skeleton extraction config (only used if extracting from videos)
CONF_THRESHOLD = 0.5
POSE_MODEL_PATH = os.environ.get("YOLO_POSE_MODEL", "yolov8m-pose.pt")

# ============================================================================
# SKELETON EXTRACTION
# ============================================================================

# Config
MAX_WORKERS = 6  # Number of parallel threads


def _import_extraction_deps():
    """Import dependencies for skeleton extraction (lazy loading)."""
    try:
        import cv2
        import torch
        from ultralytics import YOLO
        import threading
        import concurrent.futures
        return cv2, torch, YOLO, threading, concurrent.futures
    except ImportError as e:
        raise ImportError(
            f"Skeleton extraction requires cv2, torch, and ultralytics: {e}\n"
            "Install with: pip install opencv-python torch ultralytics"
        )


def _infer_num_keypoints(model) -> int:
    """Infer number of keypoints from YOLO Pose model."""
    try:
        head = model.model.model[-1]
        kpt_shape = getattr(head, "kpt_shape", None)
        if kpt_shape:
            return int(kpt_shape[0])
    except AttributeError:
        pass
    
    kpt_shape = getattr(model.model, "kpt_shape", None)
    if kpt_shape:
        return int(kpt_shape[0])
    
    return 17  # Default COCO


def extract_skeleton_from_video(video_path, output_path, model, device, lock):
    """
    Worker function for threaded extraction.
    Uses lock for GPU inference safety.
    """
    import cv2
    import numpy as np
    import torch
    
    try:
        # We assume model is already loaded and passed
        # No need to infer num_keypoints every frame, assume known or 17
        # Safe default if accessing model internals is thread-risky
        num_keypoints = 17 
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return f"❌ Error opening {os.path.basename(video_path)}"
        
        video_skeleton_data = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # --- CRITICAL SECTION: GPU INFERENCE ---
            with lock:
                results = model(frame, conf=CONF_THRESHOLD, device=device, verbose=False)
            # ---------------------------------------
            
            # Post-process (CPU - Parallelizable)
            current_frame_landmarks = np.zeros((num_keypoints, 4), dtype=np.float32)
            
            best_keypoints = None
            best_scores = None
            best_conf = -1.0
            
            for result in results:
                kp_struct = getattr(result, "keypoints", None)
                boxes = getattr(result, "boxes", None)
                if kp_struct is None or boxes is None or len(boxes) == 0:
                    continue
                
                # Update num_keypoints if possible from first result result
                # (A bit hacky inside loop but safe enough)
                if kp_struct.xy.shape[1] != num_keypoints:
                    num_keypoints = kp_struct.xy.shape[1]
                    current_frame_landmarks = np.zeros((num_keypoints, 4), dtype=np.float32)

                kp_xy = kp_struct.xy
                kp_conf = getattr(kp_struct, "conf", None)
                
                for idx in range(len(boxes)):
                    conf_tensor = boxes.conf
                    conf_value = float(conf_tensor[idx].item()) if conf_tensor is not None else 0.0
                    
                    if conf_value <= best_conf:
                        continue
                    
                    keypoints_np = kp_xy[idx]
                    if isinstance(keypoints_np, torch.Tensor):
                        keypoints_np = keypoints_np.detach().cpu().numpy()
                    else:
                        keypoints_np = np.asarray(keypoints_np)
                    
                    if kp_conf is not None:
                        kp_scores = kp_conf[idx]
                        if isinstance(kp_scores, torch.Tensor):
                            kp_scores = kp_scores.detach().cpu().numpy()
                        else:
                            kp_scores = np.asarray(kp_scores)
                    else:
                        kp_scores = np.full(num_keypoints, conf_value, dtype=np.float32)
                    
                    best_conf = conf_value
                    best_keypoints = keypoints_np
                    best_scores = kp_scores
            
            if best_keypoints is not None:
                current_frame_landmarks[:, 0:2] = best_keypoints.astype(np.float32)
                current_frame_landmarks[:, 2] = 0.0
                current_frame_landmarks[:, 3] = best_scores.astype(np.float32)
            
            video_skeleton_data.append(current_frame_landmarks)
        
        cap.release()
        
        if len(video_skeleton_data) == 0:
            return f"⚠️ No frames in {os.path.basename(video_path)}"
        
        final_array = np.array(video_skeleton_data, dtype=np.float32)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, final_array)
        
        return None
        
    except Exception as e:
        return f"❌ Error {os.path.basename(video_path)}: {str(e)}"


def extract_skeletons_from_folder(video_folder, skeleton_folder):
    """
    Extract skeletons from all videos in a folder using Threaded Pipeline.
    """
    cv2, torch, YOLO, threading, concurrent_futures = _import_extraction_deps()
    os.makedirs(skeleton_folder, exist_ok=True)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        device = 0
        print(f"\n🔥 GPU Detected: {gpu_name}")
    else:
        device = 'cpu'
        print(f"\n⚠️ GPU NOT detected. Using CPU.")
    
    # Load model once
    print(f"📦 Loading YOLO model on {device}...")
    pose_model = YOLO(POSE_MODEL_PATH)
    
    # Create Lock for GPU
    gpu_lock = threading.Lock()
    
    # Find all video files
    video_files = []
    for ext in ['*.mp4', '*.mov', '*.avi', '*.MP4', '*.MOV']:
        video_files.extend(glob(os.path.join(video_folder, ext)))
    
    # Remove duplicates (Windows glob is case insensitive)
    video_files = sorted(list(set(video_files)))
    
    if len(video_files) == 0:
        print(f"⚠️ No video files found in {video_folder}")
        return []
    
    print(f"\n📹 Extracting skeletons from {len(video_files)} videos...")
    print(f"   Using {MAX_WORKERS} threads (Shared GPU Model)")
    
    skeleton_paths = []
    
    # helper for map
    def process_video_wrapper(args):
        # args = (video_path, output_path)
        return extract_skeleton_from_video(args[0], args[1], pose_model, device, gpu_lock)

    tasks = []
    for video_path in video_files:
        filename = os.path.basename(video_path)
        filename_no_ext = os.path.splitext(filename)[0]
        output_path = os.path.join(skeleton_folder, f"{filename_no_ext}.npy")
        
        if os.path.exists(output_path):
            skeleton_paths.append(output_path)
        else:
            tasks.append((video_path, output_path))
            skeleton_paths.append(output_path)
    
    if not tasks:
        print("✅ All skeletons already extracted.")
        return skeleton_paths

    # Run Threaded extraction
    # We use ThreadPoolExecutor because we share the 'pose_model' object and 'gpu_lock'
    with concurrent_futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(
            executor.map(process_video_wrapper, tasks), 
            total=len(tasks), 
            desc="Extracting"
        ))
    
    for res in results:
        if res:
            print(res)
    
    return skeleton_paths


# ============================================================================
# PREDICTION
# ============================================================================

class GolfSwingPredictor:
    """Predictor using full biomechanical features."""
    
    def __init__(self, model_path, scaler_path, metadata_path):
        """Load model, scaler, and metadata."""
        try:
            self.model = joblib.load(model_path)
        except ValueError as e:
            if "BitGenerator" in str(e):
                import pickle
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            else:
                raise
        
        self.scaler = joblib.load(scaler_path)
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.selected_features = self.metadata['selected_features']
        self.feature_extractor = GolfFeatureExtractorBiomech()
        
        print(f"✓ Model loaded: {len(self.selected_features)} features")
    
    def predict(self, skeleton_path):
        """Predict from skeleton file."""
        features = self.feature_extractor.extract_all_features(skeleton_path)
        features_df = pd.DataFrame([features])
        features_df = features_df[self.selected_features]
        features_scaled = self.scaler.transform(features_df)
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        return prediction, probabilities
    
    def get_band_name(self, band_idx):
        """Convert band index to name."""
        band_names = {
            0: "band 0-2",
            1: "band 2-4",
            2: "band 4-6",
            3: "band 6-8",
            4: "band 8-10"
        }
        return band_names.get(band_idx, "unknown")


def predict_from_skeletons(skeleton_folder, output_csv, model_dir="../models"):
    """
    Predict from skeleton files and save to CSV.
    
    Args:
        skeleton_folder: Folder containing .npy files
        output_csv: Output CSV path
        model_dir: Model directory
    """
    # Find model files
    if not os.path.exists(model_dir):
        print(f"❌ Model directory not found: {model_dir}")
        return None
    
    all_files = os.listdir(model_dir)
    model_files = sorted([f for f in all_files if f.startswith("stage2_model_") and f.endswith(".pkl")])
    
    if not model_files:
        print(f"❌ No models found in {model_dir}")
        return None
    
    latest_model = model_files[-1]
    model_base = latest_model.replace("stage2_model_", "").replace(".pkl", "")
    
    model_path = os.path.join(model_dir, latest_model)
    scaler_path = os.path.join(model_dir, f"stage2_scaler_{model_base}.pkl")
    metadata_path = os.path.join(model_dir, f"stage2_metadata_{model_base}.json")
    
    if not os.path.exists(scaler_path):
        print(f"❌ Scaler not found")
        return None
    
    if not os.path.exists(metadata_path):
        print(f"❌ Metadata not found")
        return None
    
    # Load predictor
    print(f"\n📦 Model: {latest_model}")
    predictor = GolfSwingPredictor(model_path, scaler_path, metadata_path)
    
    # Get skeleton files
    skeleton_files = sorted(glob(os.path.join(skeleton_folder, "*.npy")))
    
    if len(skeleton_files) == 0:
        print(f"❌ No .npy files in {skeleton_folder}")
        return None
    
    print(f"📁 Found {len(skeleton_files)} skeleton files")
    
    # Predict
    results = []
    print("\n🔄 Predicting...")
    
    for skeleton_path in tqdm(skeleton_files, desc="Processing"):
        base_name = os.path.basename(skeleton_path)
        file_name_no_ext = os.path.splitext(base_name)[0]
        video_filename = f"{file_name_no_ext}.mov"
        
        try:
            prediction, probabilities = predictor.predict(skeleton_path)
            band_name = predictor.get_band_name(prediction)
            score = band_name.replace(" ", "_").replace("-", "_")
            
            results.append({
                'file_name': video_filename,
                'score': score
            })
        except Exception as e:
            print(f"\n⚠️ Error: {base_name}: {e}")
            results.append({
                'file_name': video_filename,
                'score': 'unknown'
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    results_df.to_csv(output_csv, index=False)
    
    print("\n" + "="*70)
    print("  ✅ PREDICTION COMPLETE!")
    print("="*70)
    print(f"  Processed: {len(results_df)} files")
    print(f"  Output: {output_csv}")
    print("\n  Preview:")
    print(results_df.head(10).to_string(index=False))
    print("="*70 + "\n")
    
    return results_df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(input_folder, output_csv, model_dir="../models", skeleton_folder=None):
    """
    Main pipeline: Extract skeletons (if needed) → Predict → Save CSV
    
    Args:
        input_folder: Folder containing videos OR skeletons
        output_csv: Output CSV path
        model_dir: Model directory
        skeleton_folder: Temporary folder for skeletons (if extracting from videos)
    """
    print("\n" + "="*70)
    print("  ⛳ GOLF SWING PREDICTION PIPELINE")
    print("="*70)
    
    # Check if input contains videos or skeletons
    video_files = []
    for ext in ['*.mp4', '*.mov', '*.avi', '*.MP4', '*.MOV']:
        video_files.extend(glob(os.path.join(input_folder, ext)))
    
    # Remove duplicates
    video_files = sorted(list(set(video_files)))
    
    skeleton_files = glob(os.path.join(input_folder, "*.npy"))
    
    if len(video_files) > 0:
        # Extract skeletons from videos
        print(f"\n📹 Input: {len(video_files)} video files")
        
        if skeleton_folder is None:
            skeleton_folder = os.path.join(input_folder, "../skeletons")
        
        extract_skeletons_from_folder(input_folder, skeleton_folder)
        
        # Predict from extracted skeletons
        results = predict_from_skeletons(skeleton_folder, output_csv, model_dir)
        
    elif len(skeleton_files) > 0:
        # Directly predict from skeletons
        print(f"\n📁 Input: {len(skeleton_files)} skeleton files")
        results = predict_from_skeletons(input_folder, output_csv, model_dir)
        
    else:
        print(f"\n❌ No video or skeleton files found in {input_folder}")
        return None
    
    return results


if __name__ == "__main__":
    # Default usage: predict from skeleton files
    # results = main(
    #     input_folder="../data/skeletons",
    #     output_csv="../outputs/predictions.csv",
    #     model_dir="../models"
    # )
    
    # Example: Extract from videos and predict
    results = main(
        input_folder="../data/raw",
        output_csv="../outputs/predictions.csv",
        model_dir="../models",
        skeleton_folder="../data/skeletons"
    )
