import numpy as np
import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
try:
    from xgboost import XGBClassifier
except ImportError:
    # Fallback to Random Forest if XGBoost is not installed
    from sklearn.ensemble import RandomForestClassifier
    XGBClassifier = RandomForestClassifier

# --- Config ---
# USER: EDIT THIS PATH TO YOUR NPY FOLDER
SKELETON_FOLDER = "../../../data/TDTU-Golf-Pose-v1/preprocessed_vitpose_npy" 
METADATA_PATH = "../../../data/TDTU-Golf-Pose-v1/video_metadata.csv"
CONFIDENCE_THRESHOLD = 0.3

# --- 1. Load Labels from Metadata ---
def load_labels(metadata_path):
    df = pd.read_csv(metadata_path)
    
    # Create a mapping dictionary: id -> label
    # User confirmed files are named 0.mp4, 1.mp4 ... corresponding to IDs.
    # So we map str(id) -> view.
    
    label_map = {}
    for index, row in df.iterrows():
        # Metadata 'id' column corresponds to the numeric filename
        file_id = str(row['id']).strip()
        label = row['view'].strip().lower() # 'backside' or 'side'
        
        label_map[file_id] = label
        
    return label_map

# --- 2. Feature Extraction from Skeleton ---
def extract_features(npy_path):
    """
    Extract features from a skeleton sequence.
    We take the median skeleton of the sequence to represent the 'average pose'.
    Features:
    - Relative x-distances between keypoints (e.g. shoulders vs hips)
    - Visibility scores
    """
    try:
        data = np.load(npy_path) # Shape (Frames, 17, 4)
        if data.size == 0: return None
        
        # Filter low confidence keypoints
        # Valid data: score > threshold
        # We'll just take the frame with the highest average confidence score as the representative frame,
        # OR take the median of all valid frames.
        
        # Find the frame with the best detection
        scores = data[:, :, 3] # (Frames, 17)
        avg_scores = np.mean(scores, axis=1)
        best_frame_idx = np.argmax(avg_scores)
        best_skeleton = data[best_frame_idx] # (17, 4)
        
        # If the best frame is still garbage, return None
        if np.mean(best_skeleton[:, 3]) < CONFIDENCE_THRESHOLD:
            return None

        # Keypoints: 
        # 0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar, 
        # 5:LShoulder, 6:RShoulder, 7:LElbow, 8:RElbow, 9:LWrist, 10:RWrist,
        # 11:LHip, 12:RHip, 13:LKnee, 14:RKnee, 15:LAnkle, 16:RAnkle
        
        kp = best_skeleton[:, :2] # (17, 2)
        
        # Normalize: Center hip midpoint to (0,0)
        hip_midpoint = (kp[11] + kp[12]) / 2
        kp_norm = kp - hip_midpoint
        
        # Feature Engineering for View Classification:
        # Side view: Shoulders are close in X (stacked). Back view: Shoulders are far apart in X.
        # Side view: Hips are close in X. Back view: Hips are far apart in X.
        
        # Widths
        shoulder_width = np.abs(kp_norm[5, 0] - kp_norm[6, 0])
        hip_width = np.abs(kp_norm[11, 0] - kp_norm[12, 0])
        ear_width = np.abs(kp_norm[3, 0] - kp_norm[4, 0])
        
        # Heights (y-diff)
        # Maybe useful?
        
        # X-positions relative to center
        # For side view, usually one side is clearly in front of the other?
        # But 'backside' vs 'side' relies heavy on width.
        
        features = [
            shoulder_width,
            hip_width,
            ear_width,
            kp_norm[0,0], kp_norm[0,1], # Nose pos
            kp_norm[5,0], kp_norm[6,0], # Shoulder X positions
            kp_norm[11,0], kp_norm[12,0] # Hip X positions
        ]
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error reading {npy_path}: {e}")
        return None

# --- Rule-Based Classifier ---
def predict_view_rule_based(keypoints_norm):
    # KP Indices: 5:LSh, 6:RSh, 11:LHip, 12:RHip
    l_sh = keypoints_norm[5]
    r_sh = keypoints_norm[6]
    l_hip = keypoints_norm[11]
    r_hip = keypoints_norm[12]
    
    # Scale Reference: Torso Height
    shoulder_mid = (l_sh + r_sh) / 2
    hip_mid = (l_hip + r_hip) / 2
    torso_height = np.linalg.norm(shoulder_mid - hip_mid)
    
    if torso_height == 0: return "unknown"
    
    # Widths (X separation)
    shoulder_w = np.abs(l_sh[0] - r_sh[0])
    hip_w = np.abs(l_hip[0] - r_hip[0])
    
    s_ratio = shoulder_w / torso_height
    h_ratio = hip_w / torso_height
    
    avg_ratio = (s_ratio + h_ratio) / 2
    
    # Threshold Tuning:
    # DATA OBSERVATION:
    # Truth: backside -> Ratio ~0.12 - 0.20 (Shoulders stacked, Down-the-Line view)
    # Truth: side     -> Ratio ~0.50 - 0.70 (Shoulders wide, Face-On view)
    
    threshold = 0.40
    
    if avg_ratio < threshold:
        return 'backside'
    else:
        return 'side'

# --- Main Script ---
def main():
    print("Loading labels...")
    label_map = load_labels(METADATA_PATH)
    
    print(f"Scanning first 50 videos (IDs 1-50)...")
    
    y_true = []
    y_pred = []
    
    # Data Collection for Optimization
    data_points = [] # list of (file_id, truth, ratio)
    
    # Scan ALL npy files in folder
    npy_files = glob.glob(os.path.join(SKELETON_FOLDER, "*.npy"))
    print(f"Scanning full dataset: {len(npy_files)} files found.")
    
    target_ids = []
    for f in npy_files:
        filename = os.path.basename(f)
        fid = os.path.splitext(filename)[0]
        # Only process numeric IDs (original structure) or match metadata
        if fid.isdigit():
            target_ids.append(fid)
    
    print(f"Matched {len(target_ids)} IDs with metadata.")
    
    for file_id in target_ids:
        npy_path = os.path.join(SKELETON_FOLDER, f"{file_id}.npy")
        if not os.path.exists(npy_path): continue
        truth = label_map.get(file_id)
        if truth is None: continue
            
        try:
            data = np.load(npy_path)
            if data.shape[0] < 5: continue
            
            # Extract Setup Phase info
            early_frames = data[:15]
            kp_avg = np.mean(early_frames[:, :, :2], axis=0)
            
            # Calculate Ratio
            l_sh = kp_avg[5]; r_sh = kp_avg[6]; l_hip = kp_avg[11]; r_hip = kp_avg[12]
            th = np.linalg.norm((l_sh+r_sh)/2 - (l_hip+r_hip)/2)
            sw = np.abs(l_sh[0] - r_sh[0])
            hw = np.abs(l_hip[0] - r_hip[0])
            
            ratio = ((sw/th) + (hw/th)) / 2 if th > 0 else 0
            
            data_points.append({'id': file_id, 'truth': truth, 'ratio': ratio})
            
        except Exception as e:
            print(f"Error {file_id}: {e}")

    if not data_points:
        print("No valid data points.")
        return

    # optimize Threshold
    best_acc = 0
    best_thresh = 0.0
    
    # Grid search from 0.20 to 0.50
    thresholds = np.linspace(0.20, 0.50, 100)
    
    for t in thresholds:
        correct = 0
        total = len(data_points)
        for dp in data_points:
            # Logic: Ratio < T -> Backside, Ratio >= T -> Side
            pred = 'backside' if dp['ratio'] < t else 'side'
            if pred == dp['truth']:
                correct += 1
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
            
    print(f"\n--- OPTIMIZATION ---")
    print(f"Optimal Threshold found: {best_thresh:.4f}")
    print(f"Max Achievable Accuracy on set: {best_acc*100:.2f}%")
    
    # Final Prediction with Best Threshold
    y_true = []
    y_pred = []
    
    print("\n--- FINAL PREDICTIONS ---")
    print(f"{'ID':<4} | {'Truth':<8} | {'Ratio':<6} | {'Pred':<8} | {'Status'}")
    
    for dp in data_points:
        pred = 'backside' if dp['ratio'] < best_thresh else 'side'
        y_true.append(dp['truth'])
        y_pred.append(pred)
        
        status = "OK" if pred == dp['truth'] else "FAIL"
        if status == "FAIL":
            print(f"{dp['id']:<4} | {dp['truth']:<8} | {dp['ratio']:.2f}   | {pred:<8} | {status}")

    # Evaluation
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=['backside', 'side'])
    
    print("\n--- RESULTS ---")
    print(f"Accuracy: {acc*100:.2f}%")
    print("\nConfusion Matrix [Back, Side]:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['backside', 'side']))

if __name__ == "__main__":
    main()
