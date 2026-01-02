import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from evaluate_layers34 import evaluate_user_skeleton, GolfNpyDetector

METADATA_PATH = "video_metadata.csv"
SKELETON_DIR = "data/TDTU_skeletons_npy"
WEIGHTS_OUTPUT_PATH = "outputs/score_mapper_weights.json"
EVAL_OUTPUT_PATH = "outputs/score_mapper_eval.json"

BAND_BUCKETS: List[Tuple[float, float, str]] = [
    (1.0, 2.0, "Band 1-2"),
    (2.0, 4.0, "Band 2-4"),
    (4.0, 6.0, "Band 4-6"),
    (6.0, 8.0, "Band 6-8"),
    (8.0, 10.0, "Band 8-10"),
]
BAND_LABELS = [label for _, _, label in BAND_BUCKETS]


def band_to_score(band_label: str) -> float:
    cleaned = band_label.replace("Band", "").strip()
    low_str, high_str = cleaned.split("-")
    low = float(low_str)
    high = float(high_str)
    return (low + high) / 2.0


def score_to_band(score: float) -> str:
    clamped = max(0.0, min(10.0, score))
    for _, high, label in BAND_BUCKETS:
        if clamped <= high:
            return label
    return BAND_BUCKETS[-1][2]


def load_metadata() -> List[Dict[str, str]]:
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")
    with open(METADATA_PATH, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def collect_samples(detector: GolfNpyDetector) -> List[Dict[str, object]]:
    metadata_rows = load_metadata()
    samples: List[Dict[str, object]] = []
    for row in metadata_rows:
        sample_id = int(row["id"])
        skeleton_path = os.path.join(SKELETON_DIR, f"{sample_id}.npy")
        if not os.path.exists(skeleton_path):
            print(f"[Skip id={sample_id}] Missing skeleton file at {skeleton_path}")
            continue
        evaluation = evaluate_user_skeleton(skeleton_path, detector)
        best_view = evaluation["best_result"]
        samples.append(
            {
                "id": sample_id,
                "file_name": row.get("original_name", f"{sample_id}.mp4"),
                "band": row.get("band", "Band 4-6"),
                "target_score": band_to_score(row.get("band", "Band 4-6")),
                "model_score": best_view["fusion"]["final_score"],
                "best_view": best_view["view"],
            }
        )
    if not samples:
        raise RuntimeError("No samples collected for score mapping.")
    return samples


def linear_least_squares(model_scores: np.ndarray, target_scores: np.ndarray) -> Tuple[float, float, np.ndarray]:
    A = np.vstack([model_scores, np.ones(len(model_scores))]).T
    slope, bias = np.linalg.lstsq(A, target_scores, rcond=None)[0]
    linear_pred = slope * model_scores + bias
    return float(slope), float(bias), linear_pred


def double_power_warp(norm_scores: np.ndarray, gamma_low: float, gamma_high: float) -> np.ndarray:
    norm_scores = np.clip(norm_scores, 0.0, 1.0)
    warped = np.empty_like(norm_scores)
    lower_mask = norm_scores < 0.5
    lower_vals = norm_scores[lower_mask]
    upper_vals = norm_scores[~lower_mask]
    if lower_vals.size > 0:
        warped[lower_mask] = 0.5 * np.power(2.0 * lower_vals, gamma_low)
    if upper_vals.size > 0:
        warped[~lower_mask] = 1.0 - 0.5 * np.power(2.0 * (1.0 - upper_vals), gamma_high)
    return np.clip(warped, 0.0, 1.0)


def apply_mapping(raw_scores, params: Dict[str, float]) -> np.ndarray:
    raw_scores = np.asarray(raw_scores, dtype=np.float32)
    norm_linear = np.clip((params["slope"] * raw_scores + params["bias"]) / 10.0, 0.0, 1.0)
    warped = double_power_warp(norm_linear, params["gamma_low"], params["gamma_high"])
    return warped * 10.0


def fit_non_linear_mapping(samples: List[Dict[str, object]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    model_scores = np.array([sample["model_score"] for sample in samples], dtype=np.float32)
    target_scores = np.array([sample["target_score"] for sample in samples], dtype=np.float32)

    slope, bias, linear_pred = linear_least_squares(model_scores, target_scores)
    norm_linear = np.clip(linear_pred / 10.0, 0.0, 1.0)
    norm_target = np.clip(target_scores / 10.0, 0.0, 1.0)

    gamma_candidates = np.linspace(0.2, 0.9, 15)
    best = {"mae": float("inf"), "gamma_low": 0.8, "gamma_high": 0.8}
    for gamma_low in gamma_candidates:
        for gamma_high in gamma_candidates:
            warped = double_power_warp(norm_linear, gamma_low, gamma_high)
            preds = warped * 10.0
            residuals = target_scores - preds
            mae = float(np.mean(np.abs(residuals)))
            if mae < best["mae"]:
                ss_res = float(np.sum(residuals ** 2))
                ss_tot = float(np.sum((target_scores - np.mean(target_scores)) ** 2))
                r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
                best.update({
                    "mae": mae,
                    "r2": r2,
                    "gamma_low": float(gamma_low),
                    "gamma_high": float(gamma_high),
                })

    mapping_params = {
        "slope": slope,
        "bias": bias,
        "gamma_low": best["gamma_low"],
        "gamma_high": best["gamma_high"],
    }
    fit_stats = {"mae": best["mae"], "r2": best["r2"]}
    return mapping_params, fit_stats


def evaluate_classification(
    samples: List[Dict[str, object]],
    mapping_params: Dict[str, float],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    confusion = {actual: {pred: 0 for pred in BAND_LABELS} for actual in BAND_LABELS}
    total = len(samples)
    correct = 0
    per_band_totals = {label: 0 for label in BAND_LABELS}
    per_band_correct = {label: 0 for label in BAND_LABELS}
    annotated: List[Dict[str, object]] = []

    model_scores = np.array([sample["model_score"] for sample in samples], dtype=np.float32)
    scaled_scores = apply_mapping(model_scores, mapping_params)

    abs_errors = []
    for sample, scaled_score in zip(samples, scaled_scores):
        scaled_score = float(scaled_score)
        predicted_band = score_to_band(scaled_score)
        actual_band = sample["band"]
        if actual_band not in BAND_LABELS:
            actual_band = score_to_band(sample["target_score"])
        per_band_totals[actual_band] += 1
        if predicted_band == actual_band:
            correct += 1
            per_band_correct[actual_band] += 1
        confusion[actual_band][predicted_band] += 1
        abs_errors.append(abs(sample["target_score"] - scaled_score))
        annotated.append(
            {
                **sample,
                "scaled_score": scaled_score,
                "predicted_band": predicted_band,
                "is_correct": predicted_band == actual_band,
            }
        )

    accuracy = correct / total if total else 0.0
    per_band_accuracy = {}
    for label in BAND_LABELS:
        total_count = per_band_totals[label]
        per_band_accuracy[label] = per_band_correct[label] / total_count if total_count else None

    metrics = {
        "accuracy": accuracy,
        "mae": float(np.mean(abs_errors)) if abs_errors else 0.0,
        "confusion_matrix": confusion,
        "per_band_accuracy": per_band_accuracy,
        "samples": total,
    }
    return metrics, annotated


def save_json(path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    detector = GolfNpyDetector()
    samples = collect_samples(detector)
    mapping_params, fit_stats = fit_non_linear_mapping(samples)
    metrics, annotated = evaluate_classification(samples, mapping_params)

    weights_payload = {
        **mapping_params,
        "training_samples": len(samples),
        "fit_stats": fit_stats,
    }
    save_json(WEIGHTS_OUTPUT_PATH, weights_payload)

    eval_payload = {
        "stage1": weights_payload,
        "stage2": metrics,
        "samples": annotated,
    }
    save_json(EVAL_OUTPUT_PATH, eval_payload)

    print("\n[Stage 1] Learned non-linear mapping")
    print(
        "  slope={slope:.4f} | bias={bias:.4f} | gamma_low={gamma_low:.2f} | gamma_high={gamma_high:.2f}".format(
            **mapping_params
        )
    )
    print(f"  R2={fit_stats['r2']:.3f} | MAE={fit_stats['mae']:.3f}")
    print("[Stage 2] Classification-style evaluation")
    print(f"  accuracy={metrics['accuracy']*100:.2f}% | MAE={metrics['mae']:.3f}")
    print(f"  weights saved to {WEIGHTS_OUTPUT_PATH}")
    print(f"  eval report saved to {EVAL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
