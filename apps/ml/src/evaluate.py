import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pathlib import Path

from rulebased_detector import GolfNpyDetector

# ---------------------------------------------------------------------------
# Configuration (edit these paths directly, no CLI args required)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]  # Repo root
USER_SKELETON_PATH = ROOT / 'data' / 'TDTU_skeletons_npy' / '3.npy'  # Path to user skeleton .npy
OUTPUT_JSON_PATH = ROOT / 'outputs' / '24_eval_layers34.json'
# Reference templates for different camera views (edit paths to match your dataset)
VIEW_REFERENCE_MAP = {
    "front_view": str(ROOT / "data" / "golfdb" / "skeletons_npy" / "8.npy"),
    "back_view": str(ROOT / "data" / "golfdb" / "skeletons_npy" / "2.npy"),
}

PHASE_TARGET_FRAMES = {"backswing": 40, "downswing": 10, "follow_through": 20}
PHASE_WEIGHTS = {"backswing": 0.30, "downswing": 0.50, "follow_through": 0.20}
BODY_WEIGHTS = {"upper": 0.70, "lower": 0.30}
DTW_DECAY = 1.25
PENALTY_FACTOR = 0.8
SEVERITY_WEIGHTS = {"high": 1.0, "medium": 0.6, "low": 0.3}
SEVERITY_LABELS = {"high": "nghiem trong", "medium": "trung binh", "low": "nhe"}

FEEDBACK_TEMPLATES = {
    "chicken_wing_top": "Tai pha Top Swing, tay trai dang co lai ({angle_deg:.0f} do). Day la loi {severity_label}. Hay giu khop khuyu mo rong hon de tao banh quay lon hon.",
    "head_bobbing_top": "Tai pha Top Swing, dau dich chuyen {displacement_pct:.1f}% chieu cao than. Hay giu cam on dinh de tranh mat thang bang.",
    "soft_lead_leg": "Tai pha Impact, chan tru trai mo goc {angle_deg:.0f} do (<150 do). Khoa chan tru de giu luc danh on dinh.",
}

PHASE_ORDER = ["Address", "Top", "Impact", "Finish"]


@dataclass(frozen=True)
class SkeletonLayout:
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


def load_skeleton(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Skeleton file not found: {path}")
    array = np.load(path)
    if array.ndim != 3 or array.shape[2] < 2:
        raise ValueError(f"Invalid skeleton shape {array.shape} for {path}")
    return array.astype(np.float32)


def resolve_layout(num_keypoints: int) -> SkeletonLayout:
    if num_keypoints not in LAYOUTS:
        raise ValueError(f"Unsupported skeleton with {num_keypoints} keypoints")
    return LAYOUTS[num_keypoints]


def enforce_phase_order(phases: Dict[str, int], total_frames: int) -> Dict[str, int]:
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
    total = len(sequence)
    start = int(max(0, min(start, total - 1)))
    end = int(max(start + 1, min(end, total - 1)))
    chunk = sequence[start : end + 1]
    if len(chunk) == 1:
        chunk = np.repeat(chunk, 2, axis=0)
    return chunk


def resample_segment(segment: np.ndarray, target_length: int) -> np.ndarray:
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


def symmetric_preprocess(path: str, detector: GolfNpyDetector) -> PreprocessResult:
    raw = load_skeleton(path)
    layout = resolve_layout(raw.shape[1])
    phases = detector.detect_phases(raw.copy())
    phases = enforce_phase_order(phases, len(raw))
    normalized, torso_length = torso_normalize(raw, phases["Address"], layout)
    segments = build_phase_segments(normalized, phases)
    return PreprocessResult(
        path=path,
        phases=phases,
        normalized=normalized,
        segments=segments,
        layout=layout,
        torso_length=torso_length,
        num_keypoints=raw.shape[1],
    )


def flatten_joints(segment: np.ndarray, joint_indices: List[int]) -> np.ndarray:
    return segment[:, joint_indices, :2].reshape(segment.shape[0], -1)


def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    n, m = len(seq_a), len(seq_b)
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dist = np.linalg.norm(seq_a[i - 1] - seq_b[j - 1])
            cost[i, j] = dist + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n, m] / (n + m))


def compute_quantitative_scores(
    user: PreprocessResult, reference: PreprocessResult
) -> Dict[str, Any]:
    layout = user.layout
    upper_joints = layout.require(
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    )
    lower_joints = layout.require(
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    )

    per_phase = {}
    total_error = 0.0
    for phase_name, weight in PHASE_WEIGHTS.items():
        user_seg = user.segments[phase_name]
        ref_seg = reference.segments[phase_name]
        upper_err = dtw_distance(
            flatten_joints(user_seg, upper_joints),
            flatten_joints(ref_seg, upper_joints),
        )
        lower_err = dtw_distance(
            flatten_joints(user_seg, lower_joints),
            flatten_joints(ref_seg, lower_joints),
        )
        phase_error = (
            BODY_WEIGHTS["upper"] * upper_err + BODY_WEIGHTS["lower"] * lower_err
        )
        total_error += phase_error * weight
        per_phase[phase_name] = {
            "upper_error": float(upper_err),
            "lower_error": float(lower_err),
            "blended_error": float(phase_error),
        }

    base_score = 10.0 * math.exp(-DTW_DECAY * total_error)
    return {
        "base_score": float(base_score),
        "total_error": float(total_error),
        "per_phase": per_phase,
    }


def joint_angle(pt_a: np.ndarray, pt_b: np.ndarray, pt_c: np.ndarray) -> float:
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
    return int(max(0, min(idx, total - 1)))


def evaluate_geometric_rules(
    normalized: np.ndarray, phases: Dict[str, int], layout: SkeletonLayout
) -> List[Dict[str, Any]]:
    errors: List[Dict[str, Any]] = []
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
        errors.append(
            {
                "code": "chicken_wing_top",
                "phase": "Top",
                "severity": severity,
                "metric": {"angle_deg": float(angle_elbow)},
            }
        )

    # Rule 2: Head bobbing at Top (Y displacement vs Address)
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
        errors.append(
            {
                "code": "head_bobbing_top",
                "phase": "Top",
                "severity": severity,
                "metric": {"displacement": float(head_disp)},
            }
        )

    # Rule 3: Lead leg stability at Impact (left knee angle)
    lh, lk, la = layout.require("left_hip", "left_knee", "left_ankle")
    knee_angle = joint_angle(
        normalized[idx_imp, lh, :2],
        normalized[idx_imp, lk, :2],
        normalized[idx_imp, la, :2],
    )
    if knee_angle is not None and knee_angle < 150.0:
        severity = "high" if knee_angle < 130.0 else "medium"
        errors.append(
            {
                "code": "soft_lead_leg",
                "phase": "Impact",
                "severity": severity,
                "metric": {"angle_deg": float(knee_angle)},
            }
        )

    return errors


def apply_penalties(
    base_score: float, errors: List[Dict[str, Any]]
) -> Tuple[float, float]:
    total_penalty = 0.0
    for err in errors:
        sev = err["severity"]
        total_penalty += PENALTY_FACTOR * SEVERITY_WEIGHTS.get(sev, 0.0)
    final_score = max(0.0, min(10.0, base_score - total_penalty))
    return final_score, total_penalty


def generate_feedback(errors: List[Dict[str, Any]]) -> List[str]:
    messages: List[str] = []
    for err in errors:
        template = FEEDBACK_TEMPLATES.get(err["code"])
        severity = err["severity"]
        severity_label = SEVERITY_LABELS.get(severity, severity)
        metric = err.get("metric", {})
        if template:
            message = template.format(
                severity_label=severity_label,
                angle_deg=metric.get("angle_deg", 0.0),
                displacement_pct=metric.get("displacement", 0.0) * 100.0,
            )
        else:
            message = f"{err['code']} ({severity_label})"
        messages.append(message)
    return messages


def save_report(report: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)


def evaluate_view_pipeline(
    view_name: str,
    reference_path: str,
    detector: GolfNpyDetector,
    user: PreprocessResult,
    qualitative_errors: List[Dict[str, Any]],
) -> Dict[str, Any]:
    reference = symmetric_preprocess(reference_path, detector)

    if user.num_keypoints != reference.num_keypoints:
        raise ValueError(
            f"Skeleton mismatch for view '{view_name}': user has {user.num_keypoints} keypoints"
            f" but reference has {reference.num_keypoints}"
        )

    quantitative = compute_quantitative_scores(user, reference)
    final_score, total_penalty = apply_penalties(
        quantitative["base_score"], qualitative_errors
    )

    return {
        "view": view_name,
        "reference": reference.path,
        "reference_phases": reference.phases,
        "quantitative": quantitative,
        "fusion": {
            "final_score": float(final_score),
            "total_penalty": float(total_penalty),
        },
    }


def evaluate_user_skeleton(
    user_path: str,
    detector: Optional[GolfNpyDetector] = None,
) -> Dict[str, Any]:
    local_detector = detector or GolfNpyDetector()
    user = symmetric_preprocess(user_path, local_detector)
    qualitative = evaluate_geometric_rules(user.normalized, user.phases, user.layout)
    feedback = generate_feedback(qualitative)

    view_results: List[Dict[str, Any]] = []
    for view_name, ref_path in VIEW_REFERENCE_MAP.items():
        try:
            result = evaluate_view_pipeline(
                view_name, ref_path, local_detector, user, qualitative
            )
            view_results.append(result)
        except FileNotFoundError as exc:
            print(f"[Skip {view_name}] Missing reference skeleton: {exc}")
        except ValueError as exc:
            print(f"[Skip {view_name}] {exc}")

    if not view_results:
        raise RuntimeError(
            "No valid view evaluations completed. Check reference paths and keypoint layouts."
        )

    best_result = max(view_results, key=lambda item: item["fusion"]["final_score"])

    return {
        "user_path": user.path,
        "user_phases": user.phases,
        "qualitative": qualitative,
        "feedback": feedback,
        "view_results": view_results,
        "best_result": best_result,
    }


def main() -> None:
    evaluation = evaluate_user_skeleton(str(USER_SKELETON_PATH))
    view_results = evaluation["view_results"]
    best_result = evaluation["best_result"]
    qualitative = evaluation["qualitative"]
    feedback = evaluation["feedback"]

    report = {
        "inputs": {"user": evaluation["user_path"]},
        "phases": {"user": evaluation["user_phases"]},
        "qualitative": qualitative,
        "feedback": feedback,
        "views": view_results,
        "best_view": {
            "name": best_result["view"],
            "reference": best_result["reference"],
            "final_score": best_result["fusion"]["final_score"],
            "penalty": best_result["fusion"]["total_penalty"],
        },
    }

    save_report(report, OUTPUT_JSON_PATH)

    print("\n===== Evaluation Layer 3 & 4 =====")
    print(f"User skeleton     : {evaluation['user_path']}")
    print("-- View comparison summary --")
    for res in view_results:
        phases = res.get("quantitative", {}).get("per_phase", {})
        print(
            f"[{res['view']}] ref={res['reference']} | Base={res['quantitative']['base_score']:.2f} | "
            f"Final={res['fusion']['final_score']:.2f}"
        )
        for phase_name, detail in phases.items():
            print(
                f"    {phase_name.title():<15} Upper={detail['upper_error']:.4f} | "
                f"Lower={detail['lower_error']:.4f} | Blend={detail['blended_error']:.4f}"
            )

    print("-- Qualitative rules --")
    if qualitative:
        for err in qualitative:
            print(
                f"  {err['code']} | phase={err['phase']} | sev={err['severity']} | metric={err['metric']}"
            )
    else:
        print("  No qualitative violations")

    print("-- Fusion --")
    print(
        f"Selected view     : {best_result['view']} (Final Score {best_result['fusion']['final_score']:.2f})"
    )
    print(f"Penalty applied   : {best_result['fusion']['total_penalty']:.2f}")
    if feedback:
        print("-- Coach feedback --")
        for msg in feedback:
            print(f"  - {msg}")
    print(f"Report saved to   : {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
