
import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm
from extract_features_biomech_augmented import GolfFeatureExtractorBiomech

def compute_and_save_stats(
    metadata_path="../data/metadata/video_metadata.csv",
    skeleton_dir="data/TDTU_skeletons_augmented",
    output_path="../models/feature_statistics.json"
):
    """
    Compute mean and std for each feature per band.
    Focus on Band 4 (Best) as reference.
    """
    print(f"Loading metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path)
    
    extractor = GolfFeatureExtractorBiomech()
    
    # Storage structure
    # { feature_name: { 0: {val:[], ...}, 1: {...} } }
    feature_values = {}
    
    # Map bands to indices
    band_map = {
        "band 0-2": 0, "band 1-2": 0,
        "band 2-4": 1,
        "band 4-6": 2,
        "band 6-8": 3,
        "band 8-10": 4,
    }
    
    print("Extracting features and computing stats...")
    valid_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        sample_id = row['id']
        band_str = str(row.get("band", "")).strip().lower()
        band_idx = band_map.get(band_str, -1)
        
        if band_idx == -1:
            continue
            
        skeleton_path = f"{skeleton_dir}/{sample_id}.npy"
        if not os.path.exists(skeleton_path):
            continue
            
        try:
            feats = extractor.extract_all_features(skeleton_path)
            
            for fname, val in feats.items():
                if fname not in feature_values:
                    feature_values[fname] = {b: [] for b in range(5)}
                
                # Check for NaN/Inf
                if np.isfinite(val):
                    feature_values[fname][band_idx].append(float(val))
                    
            valid_count += 1
            
        except Exception:
            continue
            
    print(f"Processed {valid_count} samples.")
    
    # Calculate Mean/Std
    final_stats = {}
    
    for fname, bands_data in feature_values.items():
        final_stats[fname] = {}
        for b_idx, vals in bands_data.items():
            if len(vals) > 0:
                final_stats[fname][f"band_{b_idx}"] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "count": len(vals)
                }
            else:
                final_stats[fname][f"band_{b_idx}"] = {
                    "mean": 0.0, "std": 1.0, "count": 0
                }
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(final_stats, f, indent=2)
        
    print(f"✅ Statistics saved to {output_path}")

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    
    compute_and_save_stats(
        metadata_path="../data/metadata/video_metadata.csv",
        skeleton_dir="../data/skeletons",
        output_path="../models/feature_statistics.json"
    )
