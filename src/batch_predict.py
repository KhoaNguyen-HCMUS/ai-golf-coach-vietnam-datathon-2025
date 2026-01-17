"""
Predict from folder of .npy skeleton files and output CSV
Input: Folder containing .npy skeleton files
Output: CSV with columns: file_name, score
"""

import numpy as np
import pandas as pd
import os
from glob import glob
from tqdm import tqdm
from load_and_predict import GolfSwingPredictor


def predict_folder_to_csv(skeleton_dir, output_csv, model_dir="models"):
    """
    Predict all skeleton files in a folder and save to CSV.
    
    Args:
        skeleton_dir: Directory containing .npy skeleton files
        output_csv: Path to output CSV file
        model_dir: Directory containing saved models
    """
    # Find all model files
    all_files = os.listdir(model_dir)
    model_files = sorted([f for f in all_files if f.startswith("stage2_model_") and f.endswith(".pkl")])
    
    if not model_files:
        print("❌ No saved models found!")
        return
    
    # Get the latest model file
    latest_model_file = model_files[-1]
    print(f"\n📦 Found model: {latest_model_file}")
    
    # Extract base name for finding matching scaler and metadata
    model_base = latest_model_file.replace("stage2_model_", "").replace(".pkl", "")
    
    # Find matching scaler and metadata files
    scaler_candidates = [f for f in all_files if f.startswith(f"stage2_scaler_{model_base}")]
    metadata_candidates = [f for f in all_files if f.startswith(f"stage2_metadata_{model_base}")]
    
    if not scaler_candidates:
        print(f"❌ No matching scaler found for {latest_model_file}")
        return
    if not metadata_candidates:
        print(f"❌ No matching metadata found for {latest_model_file}")
        return
    
    model_path = f"{model_dir}/{latest_model_file}"
    scaler_path = f"{model_dir}/{scaler_candidates[0]}"
    metadata_path = f"{model_dir}/{metadata_candidates[0]}"
    
    print(f"   Scaler: {scaler_candidates[0]}")
    print(f"   Metadata: {metadata_candidates[0]}")
    
    # Load predictor
    print("\n" + "="*70)
    print("  🔮 FOLDER PREDICTION - NPY TO CSV")
    print("="*70)
    
    predictor = GolfSwingPredictor(model_path, scaler_path, metadata_path)
    
    # Get all skeleton files
    skeleton_files = sorted(glob(f"{skeleton_dir}/*.npy"))
    print(f"\n📁 Found {len(skeleton_files)} skeleton files in {skeleton_dir}")
    
    if len(skeleton_files) == 0:
        print("❌ No .npy files found in the specified directory!")
        return
    
    # Prepare results
    results = []
    
    print("\n🔄 Predicting...")
    for skeleton_path in tqdm(skeleton_files, desc="Processing"):
        # Get filename without extension
        base_name = os.path.basename(skeleton_path)
        file_name_no_ext = os.path.splitext(base_name)[0]
        
        # Assume video filename is the same as npy filename but with .mov extension
        video_filename = f"{file_name_no_ext}.mov"
        
        try:
            # Predict
            prediction, probabilities = predictor.predict_from_skeleton_path(skeleton_path)
            
            # Get band name
            band_name = predictor.get_band_name(prediction)
            
            # Format band name to match sample output (e.g., "band_1_2" instead of "band 1-2")
            score = band_name.replace(" ", "_").replace("-", "_")
            
            results.append({
                'file_name': video_filename,
                'score': score
            })
        except Exception as e:
            print(f"\n  ⚠️ Error processing {base_name}: {e}")
            # Add with unknown score if prediction fails
            results.append({
                'file_name': video_filename,
                'score': 'unknown'
            })
            continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    results_df.to_csv(output_csv, index=False)
    
    print("\n" + "="*70)
    print("  ✅ PREDICTION COMPLETE!")
    print("="*70)
    print(f"  Total files processed: {len(results_df)}")
    print(f"  Output saved to: {output_csv}")
    print("\n  Preview of results:")
    print(results_df.head(10).to_string(index=False))
    print("="*70)
    
    return results_df


if __name__ == "__main__":
    # Example usage
    results = predict_folder_to_csv(
        skeleton_dir="../data/skeletons",
        output_csv="../outputs/predictions.csv",
        model_dir="../models"
    )
