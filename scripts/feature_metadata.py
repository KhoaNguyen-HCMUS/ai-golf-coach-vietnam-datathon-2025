
# English metadata for Golf Swing Features

FEATURE_METADATA = {
    # === BIOMECHANICAL FEATURES (Most Important) ===
    "bio_shoulder_angle_top": {
        "name": "Shoulder Tilt at Top",
        "description": "Shoulder angle relative to horizontal at backswing peak",
        "unit": "degrees",
        "category": "Biomechanics"
    },
    "bio_hip_rotation": {
        "name": "Hip Rotation (Backswing)",
        "description": "Hip rotation angle from setup to top of backswing",
        "unit": "degrees",
        "category": "Biomechanics"
    },
    "bio_shoulder_hanging_back": {
        "name": "Shoulder Hanging Back (Finish)",
        "description": "Ratio of right shoulder lagging behind lead leg at finish",
        "unit": "ratio",
        "category": "Biomechanics"
    },
    "bio_hip_hanging_back": {
        "name": "Hip Hanging Back (Finish)",
        "description": "Ratio of hip lagging behind at finish position",
        "unit": "ratio",
        "category": "Biomechanics"
    },
    "bio_right_armpit_angle": {
        "name": "Right Armpit Angle at Top",
        "description": "Right arm separation from torso (Chicken Wing indicator)",
        "unit": "degrees",
        "category": "Biomechanics"
    },
    "bio_weight_shift": {
        "name": "Weight Shift (Impact)",
        "description": "Angle formed by hip and lead leg, indicating weight transfer",
        "unit": "degrees",
        "category": "Biomechanics"
    },
    "bio_head_loc": {
        "name": "Head Movement",
        "description": "Distance head travels from setup to impact",
        "unit": "normalized",
        "category": "Stability"
    },
    "bio_upper_tilt": {
        "name": "Upper/Lower Body Ratio",
        "description": "Ratio of upper torso to lower torso length at setup",
        "unit": "ratio",
        "category": "Biomechanics"
    },
    "bio_finish_angle": {
        "name": "Finish Angle",
        "description": "Body rotation angle at finish position",
        "unit": "degrees",
        "category": "Biomechanics"
    },
    "bio_right_leg_angle_impact": {
        "name": "Right Leg Angle (Impact)",
        "description": "Right leg bend angle at ball contact",
        "unit": "degrees",
        "category": "Biomechanics"
    },
    "bio_left_arm_angle_top": {
        "name": "Left Arm Angle at Top",
        "description": "Left arm angle at top of backswing",
        "unit": "degrees",
        "category": "Biomechanics"
    },
    "bio_right_arm_angle_top": {
        "name": "Right Arm Angle at Top",
        "description": "Right arm angle at top of backswing",
        "unit": "degrees",
        "category": "Biomechanics"
    },
    "bio_shoulder_loc": {
        "name": "Shoulder Location",
        "description": "Shoulder center position consistency",
        "unit": "",
        "category": "Biomechanics"
    },
    "bio_hip_shifted": {
        "name": "Hip Shift",
        "description": "Lateral hip movement during swing",
        "unit": "normalized",
        "category": "Biomechanics"
    },
    "bio_stance_ratio": {
        "name": "Stance Ratio",
        "description": "Width of stance relative to body height",
        "unit": "ratio",
        "category": "Biomechanics"
    },
    
    # === KINEMATICS (Velocity & ROM) ===
    "kin_peak_velocity": {
        "name": "Peak Hand Velocity",
        "description": "Maximum wrist velocity during downswing",
        "unit": "m/s (norm)",
        "category": "Kinematics"
    },
    "kin_impact_velocity": {
        "name": "Impact Velocity",
        "description": "Wrist velocity at moment of ball contact",
        "unit": "m/s (norm)",
        "category": "Kinematics"
    },
    "kin_rom_vertical": {
        "name": "Vertical Range of Motion",
        "description": "Vertical distance traveled by hands",
        "unit": "normalized",
        "category": "Kinematics"
    },
    "kin_rom_horizontal": {
        "name": "Horizontal Range of Motion",
        "description": "Horizontal distance traveled by hands",
        "unit": "normalized",
        "category": "Kinematics"
    },
    "kin_smoothness": {
        "name": "Movement Smoothness",
        "description": "Acceleration consistency (higher = smoother)",
        "unit": "score",
        "category": "Kinematics"
    },
    "kin_mean_velocity": {
        "name": "Mean Velocity",
        "description": "Average hand velocity throughout swing",
        "unit": "m/s (norm)",
        "category": "Kinematics"
    },
    "kin_velocity_std": {
        "name": "Velocity Variance",
        "description": "Standard deviation of velocity",
        "unit": "std",
        "category": "Kinematics"
    },
    "kin_acceleration_variance": {
        "name": "Acceleration Variance",
        "description": "Variance in acceleration patterns",
        "unit": "variance",
        "category": "Kinematics"
    },

    # === STABILITY ===
    "stab_head_std": {
        "name": "Head Stability",
        "description": "Standard deviation of head position throughout swing",
        "unit": "std",
        "category": "Stability"
    },
    "stab_hip_sway": {
        "name": "Hip Sway",
        "description": "Lateral hip movement amplitude (sway)",
        "unit": "normalized",
        "category": "Stability"
    },
    "stab_spine_angle_std": {
        "name": "Spine Angle Stability",
        "description": "Variability in spine tilt angle during swing",
        "unit": "std",
        "category": "Stability"
    },
    "stab_overall": {
        "name": "Overall Stability Score",
        "description": "Combined stability of head, hip, and spine",
        "unit": "score",
        "category": "Stability"
    },
    "stab_head_vertical_range": {
        "name": "Head Vertical Range",
        "description": "Vertical movement range of head",
        "unit": "normalized",
        "category": "Stability"
    },

    # === TIMING ===
    "timing_tempo_ratio": {
        "name": "Tempo Ratio",
        "description": "Backswing to downswing time ratio (ideal ~3.0)",
        "unit": "ratio",
        "category": "Timing"
    },
    "timing_backswing_frames": {
        "name": "Backswing Duration",
        "description": "Number of frames to complete backswing",
        "unit": "frames",
        "category": "Timing"
    },
    "timing_downswing_frames": {
        "name": "Downswing Duration",
        "description": "Number of frames to complete downswing",
        "unit": "frames",
        "category": "Timing"
    },
    "timing_followthrough_frames": {
        "name": "Follow-through Duration",
        "description": "Number of frames in follow-through phase",
        "unit": "frames",
        "category": "Timing"
    },
    "timing_total_frames": {
        "name": "Total Swing Time",
        "description": "Total frames from setup to finish",
        "unit": "frames",
        "category": "Timing"
    },

    # === GEOMETRIC (Errors & Angles) ===
    "geo_elbow_angle_top": {
        "name": "Right Elbow Angle (Top)",
        "description": "Right elbow bend at top of backswing (<90° is good)",
        "unit": "degrees",
        "category": "Geometric"
    },
    "geo_num_violations": {
        "name": "Technical Violations",
        "description": "Number of basic setup/swing errors detected",
        "unit": "count",
        "category": "Geometric"
    },
    "geo_total_penalty": {
        "name": "Total Penalty Score",
        "description": "Cumulative penalty from technical violations",
        "unit": "score",
        "category": "Geometric"
    },
    "geo_has_chicken_wing": {
        "name": "Chicken Wing",
        "description": "Lead elbow bending/lifting through impact",
        "unit": "binary",
        "category": "Geometric"
    },
    "geo_has_head_bobbing": {
        "name": "Head Bobbing",
        "description": "Excessive head vertical movement",
        "unit": "binary",
        "category": "Geometric"
    },
    "geo_has_soft_lead_leg": {
        "name": "Soft Lead Leg",
        "description": "Lead leg not properly braced at impact",
        "unit": "binary",
        "category": "Geometric"
    }
}

def get_feature_info(feature_name):
    """Get metadata for a feature, handling defaults."""
    return FEATURE_METADATA.get(feature_name, {
        "name": feature_name,
        "description": "Golf swing technical metric",
        "unit": "",
        "category": "Other"
    })
