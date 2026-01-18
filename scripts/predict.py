"""
Golf Swing Prediction - Single Video Mode
Input: 1 video file
Output: Score + JSON with hybrid insights (overall + phase-based critical issues)

Features:
- 39 biomechanical features extracted from skeleton
- Phase detection (Address, Top, Impact, Finish)
- Overall insights (top 3 strengths/weaknesses)
- Phase-based critical issues (1-2 per phase)
"""

import numpy as np
import os
import joblib
import json
import tempfile
from extract_features_biomech_augmented import GolfFeatureExtractorBiomech

# Config
CONF_THRESHOLD = 0.5
# Default YOLO pose model path - point to models folder
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_POSE_MODEL = os.path.join(_SCRIPT_DIR, "models", "yolov8m-pose.pt")
POSE_MODEL_PATH = os.environ.get("YOLO_POSE_MODEL", _DEFAULT_POSE_MODEL)

# Feature metadata
try:
    from feature_metadata import get_feature_info
except ImportError:
    def get_feature_info(name):
        return {"name": name, "description": "", "unit": "", "category": "Other"}


class GolfSwingPredictor:
    """Predictor with Vietnamese insights."""
    
    def __init__(self, model_path, scaler_path, metadata_path, stats_path=None):
        """Load model, scaler, metadata, and stats."""
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
        
        # Load Stats
        self.stats = None
        if stats_path and os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                self.stats = json.load(f)
        
        self.selected_features = self.metadata['selected_features']
        self.feature_extractor = GolfFeatureExtractorBiomech()
        
        # Extract feature importances
        self.feature_importances = {}
        try:
            if hasattr(self.model, 'named_estimators_') and 'rf' in self.model.named_estimators_:
                rf_imp = self.model.named_estimators_['rf'].feature_importances_
                if len(rf_imp) == len(self.selected_features):
                    self.feature_importances = dict(zip(self.selected_features, rf_imp))
        except Exception:
            pass
    
    def predict(self, skeleton_path):
        """Predict from skeleton file."""
        features = self.feature_extractor.extract_all_features(skeleton_path)
        features_df = pd.DataFrame([features])
        features_df = features_df[self.selected_features]
        features_scaled = self.scaler.transform(features_df)
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        feature_values = features_df.iloc[0].to_dict()
        return prediction, probabilities, feature_values
    
    def get_band_name(self, band_idx):
        """Convert band index to name."""
        band_names = {
            0: "band 0-2", 1: "band 2-4", 2: "band 4-6",
            3: "band 6-8", 4: "band 8-10"
        }
        return band_names.get(band_idx, "unknown")

    def generate_feature_insights(self, feature_values):
        """Generate Vietnamese insights."""
        if not self.stats:
            return {"strengths": [], "weaknesses": []}
        
        deviations = []
        for name, value in feature_values.items():
            if name not in self.stats:
                continue
            
            band_stat = self.stats[name].get("band_4")
            if not band_stat or band_stat.get('count', 0) < 5:
                continue
            
            target_mean = band_stat['mean']
            target_std = band_stat['std']
            importance = self.feature_importances.get(name, 1.0)
            z_score = (value - target_mean) / (target_std + 1e-6)
            weighted_dev = abs(z_score) * importance
            
            deviations.append({
                'name': name, 'value': value, 'target': target_mean,
                'z_score': z_score, 'weighted_dev': weighted_dev,
                'info': get_feature_info(name)
            })
        
        deviations.sort(key=lambda x: x['weighted_dev'])
        top_strengths = deviations[:3]
        top_weaknesses = deviations[-3:]
        top_weaknesses.reverse()
        
        insights = {"strengths": [], "weaknesses": []}
        
        for item in top_strengths:
            info = item['info']
            feature_name = item['name'].replace('_', ' ').title()
            text = f"{feature_name}: Good ({item['value']:.1f} {info['unit']}), close to pro avg ({item['target']:.1f})"
            insights['strengths'].append(text)
        
        for item in top_weaknesses:
            info = item['info']
            feature_name = item['name'].replace('_', ' ').title()
            direction = "below" if item['value'] < item['target'] else "above"
            text = f"{feature_name}: {direction} pro level ({item['value']:.1f} vs {item['target']:.1f} {info['unit']})"
            insights['weaknesses'].append(text)
        
        return insights
    
    def generate_phase_critical_issues(self, feature_values):
        """Generate critical issues organized by swing phase (hybrid approach)."""
        if not self.stats:
            return {}
        
        # Map features to phases
        phase_features = {
            "Setup (Address)": [
                "bio_upper_tilt", "bio_stance_ratio", "stab_overall"
            ],
            "Backswing (Address → Top)": [
                "timing_backswing_frames", "bio_hip_rotation", "bio_shoulder_angle_top",
                "bio_left_arm_angle_top", "bio_right_arm_angle_top", "bio_right_armpit_angle",
                "geo_has_chicken_wing", "stab_head_std"
            ],
            "Downswing (Top → Impact)": [
                "timing_downswing_frames", "kin_peak_velocity", "kin_impact_velocity",
                "bio_weight_shift", "bio_hip_shifted", "bio_left_arm_angle_impact",
                "bio_right_arm_angle_impact", "bio_right_leg_angle_impact",
                "geo_has_soft_lead_leg", "stab_hip_sway"
            ],
            "Follow-Through (Impact → Finish)": [
                "timing_followthrough_frames", "bio_shoulder_hanging_back", 
                "bio_hip_hanging_back", "bio_finish_angle", "bio_shoulder_loc"
            ]
        }
        
        phase_issues = {}
        
        for phase, feature_names in phase_features.items():
            issues = []
            
            for fname in feature_names:
                if fname not in feature_values or fname not in self.stats:
                    continue
                
                value = feature_values[fname]
                band_stat = self.stats[fname].get("band_4")
                if not band_stat or band_stat.get('count', 0) < 5:
                    continue
                
                target_mean = band_stat['mean']
                target_std = band_stat['std']
                importance = self.feature_importances.get(fname, 0.5)
                z_score = (value - target_mean) / (target_std + 1e-6)
                
                # Only flag significant deviations (|z| > 1.5) with reasonable importance
                if abs(z_score) > 1.5 and importance > 0.01:
                    # Determine if this is actually an issue based on feature type
                    is_issue = False
                    higher_is_better = not ('hanging' in fname or 'sway' in fname or 
                                           'error' in fname or 'violation' in fname or
                                           'penalty' in fname or 'has_' in fname)
                    
                    if higher_is_better and z_score < -1.5:
                        is_issue = True
                    elif not higher_is_better and z_score > 1.5:
                        is_issue = True
                    
                    if is_issue:
                        info = get_feature_info(fname)
                        severity_score = abs(z_score) * importance
                        
                        issues.append({
                            'feature': fname.replace('_', ' ').title(),
                            'value': float(value),
                            'pro_avg': float(target_mean),
                            'unit': info['unit'],
                            'severity': severity_score,
                            'description': info.get('description', '')
                        })
            
            # Sort by severity and take top 1-2 issues
            issues.sort(key=lambda x: x['severity'], reverse=True)
            
            if issues:
                phase_issues[phase] = issues[:2]  # Max 2 critical issues per phase
        
        return phase_issues


def extract_skeleton_from_video(video_path, output_path):
    """Extract skeleton from video."""
    import cv2
    import torch
    from ultralytics import YOLO
    
    device = 0 if torch.cuda.is_available() else 'cpu'
    pose_model = YOLO(POSE_MODEL_PATH)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    video_skeleton_data = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = pose_model(frame, conf=CONF_THRESHOLD, device=device, verbose=False)
        current_frame_landmarks = np.zeros((17, 4), dtype=np.float32)
        
        if results and len(results) > 0 and results[0].keypoints is not None:
            kpts = results[0].keypoints
            if hasattr(kpts, 'data') and kpts.data is not None and len(kpts.data) > 0:
                kpt_data = kpts.data[0].cpu().numpy()
                
                # Handle both (17,3) and (17,4) formats
                if kpt_data.shape[1] == 3:
                    # YOLO format: [x, y, conf] - need to add visibility column
                    current_frame_landmarks[:len(kpt_data), :3] = kpt_data
                    current_frame_landmarks[:len(kpt_data), 3] = 1.0  # Set visibility to 1
                else:
                    # Already has 4 columns
                    current_frame_landmarks[:len(kpt_data)] = kpt_data
        
        video_skeleton_data.append(current_frame_landmarks)

    
    cap.release()
    
    if len(video_skeleton_data) == 0:
        raise ValueError("No frames extracted from video")
    
    final_array = np.array(video_skeleton_data, dtype=np.float32)
    np.save(output_path, final_array)
    return output_path


def predict_video(video_path, model_dir=None, output_json=None):
    """
    Predict từ 1 video file.
    
    Args:
        video_path: Đường dẫn video
        model_dir: Thư mục chứa model (None = tự động tìm)
        output_json: Optional - đường dẫn lưu JSON output
    
    Returns:
        dict với keys: score, confidence, json_output
    """
    # Auto-detect model directory
    if model_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Point to models folder inside scripts directory
        if os.path.basename(script_dir) == 'scripts':
            model_dir = os.path.join(script_dir, 'models')
        else:
            # If running from somewhere else, look for scripts/models
            model_dir = os.path.join(script_dir, 'models')
    
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory not found: {model_dir}")

    # Load model
    all_files = os.listdir(model_dir)
    model_files = sorted([f for f in all_files if f.startswith("stage2_model_") and f.endswith(".pkl")])
    
    if not model_files:
        raise ValueError(f"No models found in {model_dir}")
    
    latest_model = model_files[-1]
    model_base = latest_model.replace("stage2_model_", "").replace(".pkl", "")
    
    model_path = os.path.join(model_dir, latest_model)
    scaler_path = os.path.join(model_dir, f"stage2_scaler_{model_base}.pkl")
    metadata_path = os.path.join(model_dir, f"stage2_metadata_{model_base}.json")
    stats_path = os.path.join(model_dir, "feature_statistics.json")
    
    print(f"\nLoading model: {latest_model}")
    predictor = GolfSwingPredictor(model_path, scaler_path, metadata_path, stats_path)
    
    # Extract skeleton (temporary file)
    print("Extracting skeleton...")
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        extract_skeleton_from_video(video_path, tmp_path)
        
        # Predict
        print("Running prediction...")
        pred_idx, probs, feature_values = predictor.predict(tmp_path)
        
        band_name = predictor.get_band_name(pred_idx)
        score = band_name.replace(" ", "_").replace("-", "_")
        
        # Generate insights
        insights = predictor.generate_feature_insights(feature_values)
        
        # Generate phase-based critical issues (hybrid approach)
        phase_critical_issues = predictor.generate_phase_critical_issues(feature_values)
        
        # Build result
        # Add evaluation for each feature
        feature_list = []
        for fname, val in sorted(feature_values.items(), key=lambda x: predictor.feature_importances.get(x[0], 0.0), reverse=True):
            info = get_feature_info(fname)
            
            # Calculate evaluation vs Band 4 (Pro)
            evaluation = "Unknown"
            description = info.get('description', '')
            
            if predictor.stats and fname in predictor.stats:
                band4_stat = predictor.stats[fname].get("band_4")
                if band4_stat and band4_stat.get('count', 0) > 0:
                    target_mean = band4_stat['mean']
                    target_std = band4_stat['std']
                    
                    # Calculate Z-score from pro mean
                    z_score = (val - target_mean) / (target_std + 1e-6)
                    
                    # Determine if higher is better or lower is better based on feature category
                    # For most stability/consistency features: higher is better
                    # For error/fault features (hanging_back, sway): lower is better
                    higher_is_better = not ('hanging' in fname or 'sway' in fname or 'error' in fname)
                    
                    if abs(z_score) < 1.0:
                        evaluation = "Good"
                        description = f"Within pro range ({target_mean:.1f}±{target_std:.1f})"
                    elif abs(z_score) < 2.0:
                        if (higher_is_better and z_score > 0) or (not higher_is_better and z_score < 0):
                            evaluation = "Good"
                            description = f"Better than average pro ({target_mean:.1f})"
                        else:
                            evaluation = "Average"
                            description = f"Slightly below pro level ({target_mean:.1f})"
                    else:
                        if (higher_is_better and z_score < 0) or (not higher_is_better and z_score > 0):
                            evaluation = "Poor"
                            description = f"Significantly below pro level ({target_mean:.1f})"
                        else:
                            evaluation = "Excellent"
                            description = f"Significantly better than pro ({target_mean:.1f})"
            
            feature_list.append({
                "name": fname.replace('_', ' ').title(),
                "key": fname,
                "value": float(val),
                "unit": info['unit'],
                "importance": float(predictor.feature_importances.get(fname, 0.0)),
                "evaluation": evaluation,
                "description": description
            })
        
        result = {
            "score": score,
            "band_index": int(pred_idx),
            "confidence": float(max(probs)),
            "json_output": {
                "file_name": os.path.basename(video_path),
                "prediction": {
                    "score": score,
                    "band_index": int(pred_idx),
                    "confidence": float(max(probs)),
                    "probabilities": {
                        predictor.get_band_name(i).replace(" ", "_"): float(p)
                        for i, p in enumerate(probs)
                    }
                },
                "insights": insights,
                "phase_critical_issues": phase_critical_issues,
                "features": feature_list
            }
        }
        
        # Save JSON
        if output_json:
            os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(result['json_output'], f, indent=2, ensure_ascii=False)
            print(f"JSON saved: {output_json}")
        
        return result
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    import argparse
    import pandas as pd
    
    parser = argparse.ArgumentParser(description="Golf Swing Prediction - Single Video")
    parser.add_argument("video", help="Video file path")
    parser.add_argument("--output", help="Output JSON path (optional)")
    parser.add_argument("--model_dir", default=None, help="Model directory (auto-detect if not specified)")

    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  GOLF SWING PREDICTION")
    print("="*70)
    
    try:
        result = predict_video(args.video, args.model_dir, args.output)
        
        print(f"\n{'='*70}")
        print(f"  Score: {result['score']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print("="*70)
        
        # Print insights
        insights = result['json_output']['insights']
        if insights.get('strengths'):
            print("\nStrengths:")
            for s in insights['strengths']:
                print(f"  + {s}")
        
        if insights.get('weaknesses'):
            print("\nAreas for Improvement:")
            for w in insights['weaknesses']:
                print(f"  - {w}")
        
        # Print phase-based critical issues
        phase_issues = result['json_output'].get('phase_critical_issues', {})
        if phase_issues:
            print("\n" + "="*70)
            print("  CRITICAL ISSUES BY PHASE")
            print("="*70)
            for phase, issues in phase_issues.items():
                print(f"\n📍 {phase}:")
                for issue in issues:
                    print(f"   ⚠️  {issue['feature']}: {issue['value']:.1f}{issue['unit']} (pro avg: {issue['pro_avg']:.1f})")
        
        if not args.output:
            print(f"\n(Use --output to save JSON)")
        
        print()
        
    except Exception as e:
        print(f"\nError: {e}")
        exit(1)
