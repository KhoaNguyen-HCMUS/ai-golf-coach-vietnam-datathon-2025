"""
Helper functions for feature extraction
Simplified version - only essential functions for prediction
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from rulebased_detector import GolfNpyDetector


# ============================================================================
# SKELETON LAYOUT CONFIGURATION
# ============================================================================

PHASE_ORDER = ["Address", "Top", "Impact", "Finish"]
PHASE_TARGET_FRAMES = {"backswing": 40, "downswing": 10, "follow_through": 20}


@dataclass(frozen=True)
class SkeletonLayout:
    """Skeleton joint layout mapping."""
    name: str
    joints: Dict[str, int]

    def idx(self, joint: str) -> int:
        if joint not in self.joints:
            raise ValueError(f"Joint '{joint}' not available for layout {self.name}")
        return self.joints[joint]

    def require(self, *joint_names: str) -> List[int]:
        return [self.idx(name) for name in joint_names]


@dataclass
class PreprocessResult:
    """Result of skeleton preprocessing."""
    path: str
    phases: Dict[str, int]
    normalized: np.ndarray
    segments: Dict[str, np.ndarray]
    layout: SkeletonLayout
    torso_length: float
    num_keypoints: int


LAYOUTS: Dict[int, SkeletonLayout] = {
    17: SkeletonLayout(
        name="coco17",
        joints={
            "head": 0,
            "left_shoulder": 5,
            "right_shoulder": 6,
            "left_elbow": 7,
            "right_elbow": 8,
            "left_wrist": 9,
            "right_wrist": 10,
            "left_hip": 11,
            "right_hip": 12,
            "left_knee": 13,
            "right_knee": 14,
            "left_ankle": 15,
            "right_ankle": 16,
        },
    ),
    33: SkeletonLayout(
        name="mediapipe33",
        joints={
            "head": 0,
            "left_shoulder": 11,
            "right_shoulder": 12,
            "left_elbow": 13,
            "right_elbow": 14,
            "left_wrist": 15,
            "right_wrist": 16,
            "left_hip": 23,
            "right_hip": 24,
            "left_knee": 25,
            "right_knee": 26,
            "left_ankle": 27,
            "right_ankle": 28,
        },
    ),
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_skeleton(source) -> np.ndarray:
    """Load skeleton from file or array."""
    if isinstance(source, np.ndarray):
        return source.astype(np.float32)
    
    path = source
    if not os.path.exists(path):
        raise FileNotFoundError(f"Skeleton file not found: {path}")
    array = np.load(path)
    if array.ndim != 3 or array.shape[2] < 2:
        raise ValueError(f"Invalid skeleton shape {array.shape} for {path}")
    return array.astype(np.float32)


def resolve_layout(num_keypoints: int) -> SkeletonLayout:
    """Get skeleton layout based on number of keypoints."""
    if num_keypoints not in LAYOUTS:
        raise ValueError(f"Unsupported skeleton with {num_keypoints} keypoints")
    return LAYOUTS[num_keypoints]


def enforce_phase_order(phases: Dict[str, int], total_frames: int) -> Dict[str, int]:
    """Ensure phase indices are in correct order."""
    ordered = {}
    last_idx = 0
    for key in PHASE_ORDER:
        idx = int(phases.get(key, last_idx))
        idx = max(last_idx, min(idx, total_frames - 1))
        ordered[key] = idx
        last_idx = idx
    return ordered


def torso_normalize(
    sequence: np.ndarray, address_idx: int, layout: SkeletonLayout
) -> Tuple[np.ndarray, float]:
    """Normalize skeleton by torso length and center on hips."""
    seq = sequence.copy()
    total_frames = len(seq)
    address_idx = int(np.clip(address_idx, 0, total_frames - 1))
    lhip, rhip = layout.require("left_hip", "right_hip")
    lsh, rsh = layout.require("left_shoulder", "right_shoulder")

    hip_addr = (seq[address_idx, lhip, :2] + seq[address_idx, rhip, :2]) / 2.0
    shoulder_addr = (seq[address_idx, lsh, :2] + seq[address_idx, rsh, :2]) / 2.0
    torso_length = float(np.linalg.norm(shoulder_addr - hip_addr))
    if torso_length < 1e-6:
        torso_length = 1.0

    hip_centers = (seq[:, lhip, :2] + seq[:, rhip, :2]) / 2.0
    invalid = np.linalg.norm(hip_centers, axis=1) < 1e-6
    hip_centers[invalid] = hip_addr

    seq[:, :, :2] -= hip_centers[:, None, :]
    seq[:, :, :2] /= torso_length
    return seq.astype(np.float32), torso_length


def extract_phase(sequence: np.ndarray, start: int, end: int) -> np.ndarray:
    """Extract a phase segment from sequence."""
    total = len(sequence)
    start = int(max(0, min(start, total - 1)))
    end = int(max(start + 1, min(end, total - 1)))
    chunk = sequence[start : end + 1]
    if len(chunk) == 1:
        chunk = np.repeat(chunk, 2, axis=0)
    return chunk


def resample_segment(segment: np.ndarray, target_length: int) -> np.ndarray:
    """Resample segment to target length using interpolation."""
    frames = len(segment)
    if frames == target_length:
        return segment.copy()
    flat = segment.reshape(frames, -1)
    x_old = np.linspace(0.0, 1.0, frames)
    x_new = np.linspace(0.0, 1.0, target_length)
    resampled = np.empty((target_length, flat.shape[1]), dtype=np.float32)
    for dim in range(flat.shape[1]):
        resampled[:, dim] = np.interp(x_new, x_old, flat[:, dim])
    return resampled.reshape(target_length, segment.shape[1], segment.shape[2])


def build_phase_segments(
    sequence: np.ndarray, phases: Dict[str, int]
) -> Dict[str, np.ndarray]:
    """Build phase segments from full sequence."""
    addr = phases["Address"]
    top = phases["Top"]
    impact = phases["Impact"]
    finish = phases["Finish"]
    raw_segments = {
        "backswing": extract_phase(sequence, addr, top),
        "downswing": extract_phase(sequence, top, impact),
        "follow_through": extract_phase(sequence, impact, finish),
    }
    return {
        name: resample_segment(chunk, PHASE_TARGET_FRAMES[name])
        for name, chunk in raw_segments.items()
    }


def symmetric_preprocess(source, detector: GolfNpyDetector) -> PreprocessResult:
    """
    Preprocess skeleton: detect phases, normalize, extract segments.
    This is the main preprocessing function used by feature extraction.
    """
    raw = load_skeleton(source)
    layout = resolve_layout(raw.shape[1])
    phases = detector.detect_phases(raw.copy())
    phases = enforce_phase_order(phases, len(raw))
    normalized, torso_length = torso_normalize(raw, phases["Address"], layout)
    segments = build_phase_segments(normalized, phases)
    
    path_str = source if isinstance(source, str) else "in_memory_array"
    
    return PreprocessResult(
        path=path_str,
        phases=phases,
        normalized=normalized,
        segments=segments,
        layout=layout,
        torso_length=torso_length,
        num_keypoints=raw.shape[1],
    )


def joint_angle(pt_a: np.ndarray, pt_b: np.ndarray, pt_c: np.ndarray) -> float:
    """
    Calculate angle at joint B formed by points A-B-C.
    Returns angle in degrees, or None if points are too close.
    """
    vec_ab = pt_a - pt_b
    vec_cb = pt_c - pt_b
    norm_ab = np.linalg.norm(vec_ab)
    norm_cb = np.linalg.norm(vec_cb)
    if norm_ab < 1e-6 or norm_cb < 1e-6:
        return None
    cosang = np.dot(vec_ab, vec_cb) / (norm_ab * norm_cb)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def clamp_idx(idx: int, total: int) -> int:
    """Clamp index to valid range."""
    return int(max(0, min(idx, total - 1)))


def evaluate_geometric_rules(
    normalized: np.ndarray, phases: Dict[str, int], layout: SkeletonLayout
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    """
    Evaluate geometric rules and extract metrics.
    Returns: (errors, metrics)
    - errors: List of rule violations
    - metrics: Dictionary of biomechanical metrics
    """
    errors: List[Dict[str, object]] = []
    total = len(normalized)
    idx_addr = clamp_idx(phases["Address"], total)
    idx_top = clamp_idx(phases["Top"], total)
    idx_imp = clamp_idx(phases["Impact"], total)

    # Rule 1: Chicken Wing at Top (left elbow angle)
    ls, le, lw = layout.require("left_shoulder", "left_elbow", "left_wrist")
    angle_elbow = joint_angle(
        normalized[idx_top, ls, :2],
        normalized[idx_top, le, :2],
        normalized[idx_top, lw, :2],
    )
    if angle_elbow is not None and angle_elbow < 140.0:
        severity = "high" if angle_elbow < 125.0 else "medium"
        errors.append({
            "code": "chicken_wing_top",
            "phase": "Top",
            "severity": severity,
            "metric": {"angle_deg": float(angle_elbow)},
        })

    # Rule 2: Head bobbing at Top
    head_idx = layout.idx("head")
    head_disp = abs(
        float(normalized[idx_top, head_idx, 1] - normalized[idx_addr, head_idx, 1])
    )
    if head_disp > 0.15:
        if head_disp > 0.25:
            severity = "high"
        elif head_disp > 0.18:
            severity = "medium"
        else:
            severity = "low"
        errors.append({
            "code": "head_bobbing_top",
            "phase": "Top",
            "severity": severity,
            "metric": {"displacement": float(head_disp)},
        })

    # Rule 3: Lead leg stability at Impact
    lh, lk, la = layout.require("left_hip", "left_knee", "left_ankle")
    knee_angle = joint_angle(
        normalized[idx_imp, lh, :2],
        normalized[idx_imp, lk, :2],
        normalized[idx_imp, la, :2],
    )
    if knee_angle is not None and knee_angle < 150.0:
        severity = "high" if knee_angle < 130.0 else "medium"
        errors.append({
            "code": "soft_lead_leg",
            "phase": "Impact",
            "severity": severity,
            "metric": {"angle_deg": float(knee_angle)},
        })

    # Additional biomechanical metrics
    metrics = {}
    
    # Swing tempo ratio
    bs_time = max(1, idx_top - idx_addr)
    ds_time = max(1, idx_imp - idx_top)
    metrics["tempo_ratio"] = float(bs_time / ds_time)

    # X-Factor (shoulder-hip separation)
    def get_slope(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if abs(dx) < 1e-6:
            return 90.0
        return np.degrees(np.arctan(dy / dx))

    l_sh, r_sh = layout.require("left_shoulder", "right_shoulder")
    l_hip, r_hip = layout.require("left_hip", "right_hip")
    
    sh_slope = get_slope(normalized[idx_top, l_sh, :2], normalized[idx_top, r_sh, :2])
    hip_slope = get_slope(normalized[idx_top, l_hip, :2], normalized[idx_top, r_hip, :2])
    metrics["x_factor_2d"] = float(abs(sh_slope - hip_slope))

    # Head stability metrics
    head = layout.idx("head")
    metrics["head_sway"] = float(abs(normalized[idx_top, head, 0] - normalized[idx_addr, head, 0]))
    metrics["head_lift"] = float(abs(normalized[idx_imp, head, 1] - normalized[idx_addr, head, 1]))
    metrics["lead_arm_angle_top"] = float(angle_elbow) if angle_elbow is not None else 0.0

    return errors, metrics
