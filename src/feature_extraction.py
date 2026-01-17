"""
Extended Feature Extractor with Biomechanical Features - AUGMENTED VERSION
Includes 24 base features + 15 biomechanical features = 39 total

This version works with the augmented dataset in data/TDTU_skeletons_augmented
"""

import numpy as np
import pandas as pd
from typing import Dict
import os

from rulebased_detector import GolfNpyDetector
from evaluate import (
    symmetric_preprocess,
    evaluate_geometric_rules, 
    joint_angle
)


class GolfFeatureExtractorBiomech:
    """Extended feature extractor with biomechanical analysis."""
    
    def __init__(self):
        self.detector = GolfNpyDetector()
    
    def extract_all_features(self, source) -> Dict[str, float]:
        """Extract all 39 features (24 base + 15 biomech)."""
        
        if isinstance(source, np.ndarray):
            skeleton = source
        else:
            skeleton = np.load(source)
            
        user = symmetric_preprocess(source, self.detector)
        
        features = {}
        
        # === BASE FEATURES (24 total) ===
        # GROUP 1: Timing (5)
        timing = self._extract_timing_features(user.phases)
        features.update(timing)
        
        # GROUP 2: Geometric (6)
        geo_feats = self._extract_geometric_features(user)
        features.update(geo_feats)
        
        # GROUP 3: Kinematics (8)
        kin_feats = self._extract_kinematics_features(skeleton, user.phases)
        features.update(kin_feats)
        
        # GROUP 4: Stability (5)
        stab_feats = self._extract_stability_features(skeleton, user.phases)
        features.update(stab_feats)
        
        # === NEW: BIOMECHANICAL FEATURES (15 total) ===
        biomech_feats = self._extract_biomechanical_features(skeleton, user.phases, user.layout)
        features.update(biomech_feats)
        
        return features
    
    # ============ BASE FEATURE METHODS (Copy from extract_features_no_dtw.py) ============
    
    def _extract_timing_features(self, phases: Dict[str, int]) -> Dict[str, float]:
        """Timing features."""
        addr = phases["Address"]
        top = phases["Top"]
        imp = phases["Impact"]
        fin = phases["Finish"]
        
        backswing_frames = top - addr
        downswing_frames = imp - top
        followthrough_frames = fin - imp
        total_frames = fin - addr
        tempo_ratio = backswing_frames / max(downswing_frames, 1)
        
        return {
            "timing_backswing_frames": float(backswing_frames),
            "timing_downswing_frames": float(downswing_frames),
            "timing_followthrough_frames": float(followthrough_frames),
            "timing_total_frames": float(total_frames),
            "timing_tempo_ratio": float(tempo_ratio),
        }
    
    def _extract_geometric_features(self, user) -> Dict[str, float]:
        """Geometric rule features."""
        errors, metrics = evaluate_geometric_rules(
            user.normalized, user.phases, user.layout
        )
        
        num_violations = len(errors)
        total_penalty = sum(
            0.8 * {"high": 1.0, "medium": 0.6, "low": 0.3}.get(e["severity"], 0)
            for e in errors
        )
        
        has_chicken_wing = any(e["code"] == "chicken_wing_top" for e in errors)
        has_head_bobbing = any(e["code"] == "head_bobbing_top" for e in errors)
        has_soft_lead_leg = any(e["code"] == "soft_lead_leg" for e in errors)
        
        elbow_angle = next(
            (e["metric"]["angle_deg"] for e in errors if e["code"] == "chicken_wing_top"),
            180.0
        )
        
        return {
            "geo_num_violations": float(num_violations),
            "geo_total_penalty": float(total_penalty),
            "geo_has_chicken_wing": float(int(has_chicken_wing)),
            "geo_has_head_bobbing": float(int(has_head_bobbing)),
            "geo_has_soft_lead_leg": float(int(has_soft_lead_leg)),
            "geo_elbow_angle_top": float(elbow_angle),
        }
    
    def _extract_kinematics_features(
        self, skeleton: np.ndarray, phases: Dict[str, int]
    ) -> Dict[str, float]:
        """Kinematics features."""
        
        num_keypoints = skeleton.shape[1]
        if num_keypoints == 17:
            left_wrist, right_wrist = 9, 10
        elif num_keypoints == 33:
            left_wrist, right_wrist = 15, 16
        else:
            left_wrist, right_wrist = 9, 10
        
        wrist_y = (skeleton[:, left_wrist, 1] + skeleton[:, right_wrist, 1]) / 2
        wrist_x = (skeleton[:, left_wrist, 0] + skeleton[:, right_wrist, 0]) / 2
        
        velocity = np.sqrt(np.diff(wrist_x)**2 + np.diff(wrist_y)**2)
        velocity = np.insert(velocity, 0, 0)
        
        top_idx = phases["Top"]
        imp_idx = phases["Impact"]
        downswing_vel = velocity[top_idx:imp_idx+1]
        
        peak_vel = np.max(downswing_vel) if len(downswing_vel) > 0 else 0
        impact_vel = velocity[imp_idx] if imp_idx < len(velocity) else 0
        
        acceleration = np.diff(velocity)
        acc_variance = np.var(acceleration)
        
        rom_y = np.max(wrist_y) - np.min(wrist_y)
        rom_x = np.max(wrist_x) - np.min(wrist_x)
        
        return {
            "kin_peak_velocity": float(peak_vel),
            "kin_impact_velocity": float(impact_vel),
            "kin_mean_velocity": float(np.mean(velocity)),
            "kin_velocity_std": float(np.std(velocity)),
            "kin_acceleration_variance": float(acc_variance),
            "kin_rom_vertical": float(rom_y),
            "kin_rom_horizontal": float(rom_x),
            "kin_smoothness": float(1.0 / (1.0 + acc_variance)),
        }
    
    def _extract_stability_features(
        self, skeleton: np.ndarray, phases: Dict[str, int]
    ) -> Dict[str, float]:
        """Stability features."""
        
        num_keypoints = skeleton.shape[1]
        if num_keypoints == 17:
            head_idx = 0
            left_hip, right_hip = 11, 12
            left_shoulder, right_shoulder = 5, 6
        elif num_keypoints == 33:
            head_idx = 0
            left_hip, right_hip = 23, 24
            left_shoulder, right_shoulder = 11, 12
        else:
            head_idx = 0
            left_hip, right_hip = 11, 12
            left_shoulder, right_shoulder = 5, 6
        
        head_y = skeleton[:, head_idx, 1]
        head_vertical_range = np.max(head_y) - np.min(head_y)
        head_std = np.std(head_y)
        
        hip_center_x = (skeleton[:, left_hip, 0] + skeleton[:, right_hip, 0]) / 2
        hip_sway = np.max(hip_center_x) - np.min(hip_center_x)
        
        shoulder_center = (skeleton[:, left_shoulder, :2] + skeleton[:, right_shoulder, :2]) / 2
        hip_center = (skeleton[:, left_hip, :2] + skeleton[:, right_hip, :2]) / 2
        spine_angle = np.arctan2(
            shoulder_center[:, 0] - hip_center[:, 0],
            shoulder_center[:, 1] - hip_center[:, 1]
        )
        spine_angle_std = np.std(spine_angle)
        
        return {
            "stab_head_vertical_range": float(head_vertical_range),
            "stab_head_std": float(head_std),
            "stab_hip_sway": float(hip_sway),
            "stab_spine_angle_std": float(spine_angle_std),
            "stab_overall": float(1.0 / (1.0 + head_std + hip_sway + spine_angle_std)),
        }
    
    # ============ NEW: BIOMECHANICAL FEATURES ============
    
    def _extract_biomechanical_features(
        self, skeleton: np.ndarray, phases: Dict[str, int], layout
    ) -> Dict[str, float]:
        """Extract 15 biomechanical features."""
        
        addr_idx = phases["Address"]
        top_idx = phases["Top"]
        imp_idx = phases["Impact"]
        fin_idx = phases["Finish"]
        
        # Get joint indices based on skeleton type
        num_kp = skeleton.shape[1]
        if num_kp == 17:  # COCO
            ls, rs = 5, 6
            le, re = 7, 8
            lw, rw = 9, 10
            lh, rh = 11, 12
            lk, rk = 13, 14
            la, ra = 15, 16
        else:  # MediaPipe 33
            ls, rs = 11, 12
            le, re = 13, 14
            lw, rw = 15, 16
            lh, rh = 23, 24
            lk, rk = 25, 26
            la, ra = 27, 28
        
        features = {}
        
        # 1. SHOULDER-ANGLE (at Top) - angle of shoulder line to horizontal
        shoulder_angle_top = self._calc_shoulder_angle(skeleton[top_idx], ls, rs)
        features["bio_shoulder_angle_top"] = shoulder_angle_top
        
        # 2 & 3. ARM ANGLES (at Top and Impact)
        left_arm_angle_top = joint_angle(
            skeleton[top_idx, ls, :2],
            skeleton[top_idx, le, :2],
            skeleton[top_idx, lw, :2]
        )
        right_arm_angle_top = joint_angle(
            skeleton[top_idx, rs, :2],
            skeleton[top_idx, re, :2],
            skeleton[top_idx, rw, :2]
        )
        left_arm_angle_imp = joint_angle(
            skeleton[imp_idx, ls, :2],
            skeleton[imp_idx, le, :2],
            skeleton[imp_idx, lw, :2]
        )
        right_arm_angle_imp = joint_angle(
            skeleton[imp_idx, rs, :2],
            skeleton[imp_idx, re, :2],
            skeleton[imp_idx, rw, :2]
        )
        
        features["bio_left_arm_angle_top"] = float(left_arm_angle_top or 180.0)
        features["bio_right_arm_angle_top"] = float(right_arm_angle_top or 180.0)
        features["bio_left_arm_angle_impact"] = float(left_arm_angle_imp or 180.0)
        features["bio_right_arm_angle_impact"] = float(right_arm_angle_imp or 180.0)
        
        # 4. RIGHT-LEG-ANGLE (at Impact)
        right_leg_angle = joint_angle(
            skeleton[imp_idx, rh, :2],
            skeleton[imp_idx, rk, :2],
            skeleton[imp_idx, ra, :2]
        )
        features["bio_right_leg_angle_impact"] = float(right_leg_angle or 180.0)
        
        # 5. UPPER-TILT (lower/upper body ratio at Address)
        upper_tilt = self._calc_upper_tilt(skeleton[addr_idx], ls, rs, lh, rh, la, ra)
        features["bio_upper_tilt"] = upper_tilt
        
        # 6. STANCE-RATIO (shoulder width / stride length at Address)
        stance_ratio = self._calc_stance_ratio(skeleton[addr_idx], ls, rs, la, ra)
        features["bio_stance_ratio"] = stance_ratio
        
        # 7. HEAD-LOC (head movement from Address to Impact)
        head_loc = self._calc_head_loc(skeleton, addr_idx, imp_idx, 0)
        features["bio_head_loc"] = head_loc
        
        # 8. SHOULDER-LOC (left shoulder position in stride at Impact)
        shoulder_loc = self._calc_shoulder_loc(skeleton[imp_idx], ls, la, ra)
        features["bio_shoulder_loc"] = shoulder_loc
        
        # 9. HIP-ROTATION (hip rotation from Address to Top)
        hip_rotation = self._calc_hip_rotation(skeleton, addr_idx, top_idx, lh, rh)
        features["bio_hip_rotation"] = hip_rotation
        
        # 10. HIP-SHIFTED (hip center movement from Address to Impact)
        hip_shifted = self._calc_hip_shifted(skeleton, addr_idx, imp_idx, lh, rh)
        features["bio_hip_shifted"] = hip_shifted
        
        # 11. SHOULDER-HANGING-BACK (at Finish)
        shoulder_hanging = self._calc_hanging_back(skeleton[fin_idx], ls, la, ra)
        features["bio_shoulder_hanging_back"] = shoulder_hanging
        
        # 12. HIP-HANGING-BACK (at Finish)
        hip_hanging = self._calc_hanging_back(skeleton[fin_idx], lh, la, ra)
        features["bio_hip_hanging_back"] = hip_hanging
        
        # 13. RIGHT-ARMPIT-ANGLE (at Top)
        # Angle between right elbow, right shoulder, and hip center
        hip_center = (skeleton[top_idx, lh, :2] + skeleton[top_idx, rh, :2]) / 2
        armpit_angle = joint_angle(
            skeleton[top_idx, re, :2],
            skeleton[top_idx, rs, :2],
            hip_center
        )
        features["bio_right_armpit_angle"] = float(armpit_angle or 90.0)
        
        # 14. WEIGHT-SHIFT (at Impact) - angle from left ankle to left hip
        weight_shift = self._calc_weight_shift(skeleton[imp_idx], la, lh)
        features["bio_weight_shift"] = weight_shift
        
        # 15. FINISH-ANGLE (at Finish) - angle from left ankle to right hip
        finish_angle = self._calc_finish_angle(skeleton[fin_idx], la, rh)
        features["bio_finish_angle"] = finish_angle
        
        return features
    
    # Helper methods for biomechanical calculations
    
    def _calc_shoulder_angle(self, frame, ls, rs):
        """Calculate shoulder line angle relative to horizontal."""
        left_shoulder = frame[ls, :2]
        right_shoulder = frame[rs, :2]
        dx = right_shoulder[0] - left_shoulder[0]
        dy = right_shoulder[1] - left_shoulder[1]
        angle = np.degrees(np.arctan2(dy, dx))
        return float(abs(angle))
    
    def _calc_upper_tilt(self, frame, ls, rs, lh, rh, la, ra):
        """Calculate upper/lower body ratio."""
        # Upper body: shoulder to hip distance
        shoulder_center = (frame[ls, :2] + frame[rs, :2]) / 2
        hip_center = (frame[lh, :2] + frame[rh, :2]) / 2
        upper_length = np.linalg.norm(shoulder_center - hip_center)
        
        # Lower body: hip to ankle distance
        ankle_center = (frame[la, :2] + frame[ra, :2]) / 2
        lower_length = np.linalg.norm(hip_center - ankle_center)
        
        if lower_length < 1e-6:
            return 1.0
        return float(upper_length / lower_length)
    
    def _calc_stance_ratio(self, frame, ls, rs, la, ra):
        """Calculate shoulder width to stride length ratio."""
        shoulder_width = np.linalg.norm(frame[ls, :2] - frame[rs, :2])
        stride_length = np.linalg.norm(frame[la, :2] - frame[ra, :2])
        
        if stride_length < 1e-6:
            return 1.0
        return float(shoulder_width / stride_length)
    
    def _calc_head_loc(self, skeleton, addr_idx, imp_idx, head_idx):
        """Calculate head movement ratio."""
        head_addr = skeleton[addr_idx, head_idx, :2]
        head_imp = skeleton[imp_idx, head_idx, :2]
        movement = np.linalg.norm(head_imp - head_addr)
        return float(movement)
    
    def _calc_shoulder_loc(self, frame, ls, la, ra):
        """Calculate left shoulder position within stride."""
        shoulder_pos = frame[ls, 0]
        left_ankle_pos = frame[la, 0]
        right_ankle_pos = frame[ra, 0]
        
        stride_width = abs(right_ankle_pos - left_ankle_pos)
        if stride_width < 1e-6:
            return 0.5
        
        # Normalize to 0-1 range
        relative_pos = (shoulder_pos - left_ankle_pos) / stride_width
        return float(relative_pos)
    
    def _calc_hip_rotation(self, skeleton, addr_idx, top_idx, lh, rh):
        """Calculate hip rotation angle."""
        # Hip line at Address
        hip_addr = skeleton[addr_idx, rh, :2] - skeleton[addr_idx, lh, :2]
        angle_addr = np.degrees(np.arctan2(hip_addr[1], hip_addr[0]))
        
        # Hip line at Top
        hip_top = skeleton[top_idx, rh, :2] - skeleton[top_idx, lh, :2]
        angle_top = np.degrees(np.arctan2(hip_top[1], hip_top[0]))
        
        rotation = abs(angle_top - angle_addr)
        return float(rotation)
    
    def _calc_hip_shifted(self, skeleton, addr_idx, imp_idx, lh, rh):
        """Calculate hip center movement."""
        hip_center_addr = (skeleton[addr_idx, lh, :2] + skeleton[addr_idx, rh, :2]) / 2
        hip_center_imp = (skeleton[imp_idx, lh, :2] + skeleton[imp_idx, rh, :2]) / 2
        shift = np.linalg.norm(hip_center_imp - hip_center_addr)
        return float(shift)
    
    def _calc_hanging_back(self, frame, joint_idx, la, ra):
        """Calculate hanging back ratio (joint to left ankle / stride)."""
        joint_pos = frame[joint_idx, :2]
        left_ankle = frame[la, :2]
        right_ankle = frame[ra, :2]
        
        distance = np.linalg.norm(joint_pos - left_ankle)
        stride = np.linalg.norm(right_ankle - left_ankle)
        
        if stride < 1e-6:
            return 0.0
        return float(distance / stride)
    
    def _calc_weight_shift(self, frame, la, lh):
        """Calculate weight shift angle."""
        ankle = frame[la, :2]
        hip = frame[lh, :2]
        dx = hip[0] - ankle[0]
        dy = hip[1] - ankle[1]
        angle = np.degrees(np.arctan2(dy, dx))
        return float(abs(angle))
    
    def _calc_finish_angle(self, frame, la, rh):
        """Calculate finish angle."""
        ankle = frame[la, :2]
        hip = frame[rh, :2]
        dx = hip[0] - ankle[0]
        dy = hip[1] - ankle[1]
        angle = np.degrees(np.arctan2(dy, dx))
        return float(abs(angle))


def build_feature_dataset_biomech(
    metadata_path: str,
    skeleton_dir: str,
    output_path: str = "outputs/golf_features_biomech_augmented.csv"
) -> pd.DataFrame:
    """Build feature matrix with biomechanical features."""
    
    df = pd.read_csv(metadata_path)
    extractor = GolfFeatureExtractorBiomech()
    
    all_features = []
    labels = []
    valid_ids = []
    
    band_map = {
        "Band 0-2": 0, "Band 1-2": 0,
        "Band 2-4": 1,
        "Band 4-6": 2,
        "Band 6-8": 3,
        "Band 8-10": 4,
    }
    
    print(f"Extracting biomechanical features from {len(df)} samples...")
    print("Features: 24 base + 15 biomech = 39 total")
    
    for idx, row in df.iterrows():
        sample_id = row['id']
        skeleton_path = f"{skeleton_dir}/{sample_id}.npy"
        
        if not os.path.exists(skeleton_path):
            continue
        
        try:
            features = extractor.extract_all_features(skeleton_path)
            all_features.append(features)
            
            band_str = str(row.get("band", "")).strip()
            label = band_map.get(band_str, 2)
            labels.append(label)
            valid_ids.append(sample_id)
            
        except Exception as e:
            print(f"[Error] Sample {sample_id}: {e}")
            continue
    
    if not all_features:
        print("No features extracted!")
        return pd.DataFrame()

    feature_df = pd.DataFrame(all_features)
    feature_df['label'] = labels
    feature_df['sample_id'] = valid_ids
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    feature_df.to_csv(output_path, index=False)
    
    print(f"\nSaved {len(feature_df)} samples")
    print(f"Total features: {len(feature_df.columns)-2}")
    print(f"Output: {output_path}")
    
    return feature_df


if __name__ == "__main__":
    build_feature_dataset_biomech(
        metadata_path="video_metadata.csv",
        skeleton_dir="data/TDTU_skeletons_augmented",
        output_path="outputs/golf_features_biomech_augmented.csv"
    )
