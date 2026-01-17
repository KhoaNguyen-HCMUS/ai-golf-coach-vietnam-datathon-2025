"""
Data Augmentation - CHỈ augment tập TRAIN để tránh data leakage
Val và Test giữ nguyên không augment
"""

import numpy as np
import os
from typing import List
from scipy.ndimage import gaussian_filter1d
import pandas as pd


def augment_skeleton_heavy(
    skeleton: np.ndarray, 
    num_augments: int = 9,
    seed: int = None
) -> List[np.ndarray]:
    """
    Tạo nhiều biến thể từ 1 skeleton sequence.
    
    Args:
        skeleton: Shape (Frames, Keypoints, 4) - [x, y, z, score]
        num_augments: Số biến thế cần tạo
        seed: Random seed for reproducibility (default: None)
    
    Returns:
        List các augmented skeletons
    """
    # Set seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    results = []
    n_frames = len(skeleton)
    
    # === 1. Time Warping (Co giãn thời gian) ===
    for factor in [0.85, 0.90, 1.10, 1.15]:
        new_n_frames = int(n_frames * factor)
        if new_n_frames < 10:
            continue
        indices = np.linspace(0, n_frames - 1, new_n_frames).astype(int)
        warped = skeleton[indices].copy()
        results.append(warped)
    
    # === 2. Gaussian Noise (Nhiễu nhỏ) ===
    for noise_level in [0.005, 0.01, 0.015]:
        noisy = skeleton.copy()
        noise = np.random.normal(0, noise_level, skeleton[:, :, :2].shape)
        noisy[:, :, :2] += noise
        results.append(noisy)
    
    # === 3. Temporal Jittering ===
    for shift in [-2, 2]:
        shifted = np.roll(skeleton, shift, axis=0)
        results.append(shifted)
    
    # === 4. Spatial Scaling ===
    for scale in [0.92, 1.08]:
        scaled = skeleton.copy()
        scaled[:, :, :2] *= scale
        results.append(scaled)
    
    # === 5. Smoothing ===
    smoothed = skeleton.copy()
    for kp in range(skeleton.shape[1]):
        for coord in range(2):
            smoothed[:, kp, coord] = gaussian_filter1d(
                skeleton[:, kp, coord], sigma=1.0
            )
    results.append(smoothed)

    # === 6. Combined Augmentations (Scale + Noise) ===
    # Thêm các biến thể kết hợp để tăng số lượng sample lên > 20
    for scale in [0.95, 1.05]:
        for noise_level in [0.005, 0.01]:
            combined = skeleton.copy()
            # Apply Scale
            combined[:, :, :2] *= scale
            # Apply Noise
            noise = np.random.normal(0, noise_level, skeleton[:, :, :2].shape)
            combined[:, :, :2] += noise
            results.append(combined)

    # === 7. Combined Augmentations (Time Warp + Scale) ===
    for factor in [0.90, 1.10]:
        new_n_frames = int(n_frames * factor)
        if new_n_frames < 10: continue
        indices = np.linspace(0, n_frames - 1, new_n_frames).astype(int)
        
        for scale in [0.95, 1.05]:
             warped_scaled = skeleton[indices].copy()
             warped_scaled[:, :, :2] *= scale
             results.append(warped_scaled)
    
    return results[:num_augments]


def augment_split(
    split_name: str,
    input_dir: str,
    output_dir: str,
    metadata_path: str,
    augments_per_sample: int = 9,
    augment_enabled: bool = True
):
    """
    Augment một split (train/val/test).
    CHỈ augment train, val và test giữ nguyên.
    
    Args:
        split_name: 'train', 'val', or 'test'
        input_dir: Input directory containing .npy files
        output_dir: Output directory for augmented files
        metadata_path: Path to metadata CSV for this split
        augments_per_sample: Number of augmented versions per sample
        augment_enabled: If False, just copy original files
    """
    
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(metadata_path)
    
    new_rows = []
    aug_id = 10000 + (1000 if split_name == 'train' else 
                      2000 if split_name == 'val' else 3000)
    
    print(f"\n{'='*60}")
    print(f"Processing {split_name.upper()} split...")
    print(f"{'='*60}")
    print(f"Samples: {len(df)}")
    print(f"Augmentation: {'ENABLED' if augment_enabled else 'DISABLED'}")
    
    for idx, row in df.iterrows():
        sample_id = row['id']
        npy_path = os.path.join(input_dir, f"{sample_id}.npy")
        
        if not os.path.exists(npy_path):
            print(f"[Skip] {npy_path} not found")
            continue
        
        skeleton = np.load(npy_path)
        
        # Giữ nguyên original
        new_path = os.path.join(output_dir, f"{sample_id}.npy")
        np.save(new_path, skeleton)
        
        row_dict = row.to_dict()
        if 'is_augmented' not in row_dict:
            row_dict['is_augmented'] = False
        if 'original_id' not in row_dict:
            row_dict['original_id'] = sample_id
        if 'split' not in row_dict:
            row_dict['split'] = split_name
        
        new_rows.append(row_dict)
        
        # Augment CHỈ KHI enabled (only for train)
        if augment_enabled:
            augmented = augment_skeleton_heavy(skeleton, augments_per_sample)
            
            for i, aug_skel in enumerate(augmented):
                aug_id += 1
                aug_path = os.path.join(output_dir, f"{aug_id}.npy")
                np.save(aug_path, aug_skel)
                
                aug_row = row.to_dict()
                aug_row['id'] = aug_id
                aug_row['original_id'] = sample_id
                aug_row['is_augmented'] = True
                aug_row['split'] = split_name
                new_rows.append(aug_row)
    
    # Save metadata
    new_df = pd.DataFrame(new_rows)
    new_metadata_path = os.path.join(output_dir, "metadata.csv")
    new_df.to_csv(new_metadata_path, index=False)
    
    print(f"\nResults:")
    print(f"  Original: {len(df)} samples")
    print(f"  After:    {len(new_df)} samples")
    if augment_enabled:
        print(f"  Augmented: {len(new_df) - len(df)} samples")
        print(f"  Factor: {len(new_df)/len(df):.1f}x")
    else:
        print(f"  (No augmentation - val/test set)")


def augment_all_splits(
    base_dir: str = "data/TDTU_split",
    output_base: str = "data/TDTU_augmented_split",
    augments_per_sample: int = 9
):
    """
    Augment tất cả splits với strategy:
    - Train: AUGMENT (tăng 10x)
    - Val: KHÔNG augment (giữ nguyên)
    - Test: KHÔNG augment (giữ nguyên)
    """
    
    print("="*60)
    print("  DATA AUGMENTATION - TRAIN ONLY")
    print("  (Preventing Data Leakage)")
    print("="*60)
    
    splits = {
        'train': True,   # Augment train
        'val': False,    # NO augment val
        'test': False    # NO augment test
    }
    
    for split_name, should_augment in splits.items():
        input_dir = os.path.join(base_dir, split_name)
        output_dir = os.path.join(output_base, split_name)
        metadata_path = os.path.join(input_dir, "metadata.csv")
        
        if not os.path.exists(metadata_path):
            print(f"\n[Warning] {split_name} metadata not found: {metadata_path}")
            print(f"  Skipping {split_name}...")
            continue
        
        augment_split(
            split_name=split_name,
            input_dir=input_dir,
            output_dir=output_dir,
            metadata_path=metadata_path,
            augments_per_sample=augments_per_sample,
            augment_enabled=should_augment
        )
    
    print(f"\n{'='*60}")
    print("✅ AUGMENTATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Output: {output_base}/")
    print(f"  - train/: Augmented")
    print(f"  - val/:   Original only")
    print(f"  - test/:  Original only")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    augment_all_splits(
        base_dir="data/TDTU_split",
        output_base="data/TDTU_augmented_split",
        augments_per_sample=9
    )
