"""
Demo: Predict handicap band from a single skeleton file (with automatic view detection)
"""

import numpy as np
import sys
from load_and_predict import GolfSwingPredictor
from view_classifier import GolfViewClassifier
import os
from glob import glob


def predict_single_skeleton(skeleton_path, model_dir="models"):
    """
    Predict handicap band from a single skeleton file with automatic view detection.
    
    Args:
        skeleton_path: Path to .npy skeleton file
        model_dir: Directory containing saved models
    """
    # Find latest model files
    all_files = os.listdir(model_dir)
    model_files = sorted([f for f in all_files if f.startswith("stage2_model_") and f.endswith(".pkl")])
    
    if not model_files:
        print("❌ No saved models found!")
        return
    
    latest_model_file = model_files[-1]
    model_base = latest_model_file.replace("stage2_model_", "").replace(".pkl", "")
    
    scaler_candidates = [f for f in all_files if f.startswith(f"stage2_scaler_{model_base}")]
    metadata_candidates = [f for f in all_files if f.startswith(f"stage2_metadata_{model_base}")]
    
    if not scaler_candidates or not metadata_candidates:
        print("❌ Missing model files!")
        return
    
    model_path = f"{model_dir}/{latest_model_file}"
    scaler_path = f"{model_dir}/{scaler_candidates[0]}"
    metadata_path = f"{model_dir}/{metadata_candidates[0]}"
    
    print("\n" + "="*70)
    print("  🏌️ GOLF SWING ANALYSIS - AUTOMATIC VIEW DETECTION")
    print("="*70)
    
    # Load models
    print("\n📦 Loading models...")
    predictor = GolfSwingPredictor(model_path, scaler_path, metadata_path)
    view_classifier = GolfViewClassifier(model_type='yolov8')  # Using YOLOv8 skeletons
    
    # Load skeleton
    print(f"\n📂 Loading skeleton: {skeleton_path}")
    skeleton_data = np.load(skeleton_path)
    print(f"   Shape: {skeleton_data.shape}")
    
    # Detect view
    print("\n🎯 Detecting view...")
    detected_view, view_info = view_classifier.predict(skeleton_data)
    print(f"   Detected View: {detected_view.upper()}")
    print(f"   Confidence Ratio: {view_info['ratio']:.4f} (threshold: {view_info['threshold']:.4f})")
    print(f"   Shoulder Ratio: {view_info['shoulder_ratio']:.4f}")
    print(f"   Hip Ratio: {view_info['hip_ratio']:.4f}")
    
    # Predict handicap band
    print("\n🔮 Predicting handicap band...")
    prediction, probabilities = predictor.predict_from_skeleton_path(skeleton_path)
    predicted_band_name = predictor.get_band_name(prediction)
    
    # Display results
    print("\n" + "="*70)
    print("  📊 PREDICTION RESULTS")
    print("="*70)
    print(f"\n  View: {detected_view.upper()}")
    print(f"  Predicted Handicap Band: {predicted_band_name.upper()} (Index: {prediction})")
    print(f"\n  Class Probabilities:")
    print(f"  {'Band':<15} | {'Probability':<12} | Bar")
    print(f"  {'-'*50}")
    
    for i, prob in enumerate(probabilities):
        band_name = predictor.get_band_name(i)
        bar = '█' * int(prob * 40)
        marker = ' ← PREDICTED' if i == prediction else ''
        print(f"  {band_name:<15} | {prob*100:>5.1f}% | {bar}{marker}")
    
    print("\n" + "="*70)
    
    return detected_view, prediction, probabilities


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Use skeleton path from command line argument
        skeleton_path = sys.argv[1]
    else:
        # Use first skeleton file in data directory as demo
        skeleton_files = glob("data/TDTU_skeletons_augmented/*.npy")
        if skeleton_files:
            skeleton_path = skeleton_files[0]
            print(f"\nNo skeleton path provided, using demo file: {skeleton_path}")
        else:
            print("\n❌ No skeleton files found!")
            print("\nUsage: python predict_single.py <path_to_skeleton.npy>")
            sys.exit(1)
    
    if not os.path.exists(skeleton_path):
        print(f"\n❌ File not found: {skeleton_path}")
        sys.exit(1)
    
    predict_single_skeleton(skeleton_path)
