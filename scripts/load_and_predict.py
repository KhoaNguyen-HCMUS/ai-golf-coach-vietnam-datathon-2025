"""
Load saved Stage 2 model and make predictions
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
from extract_features_biomech_augmented import GolfFeatureExtractorBiomech

class GolfSwingPredictor:
    def __init__(self, model_path, scaler_path, metadata_path):
        """
        Load the saved model.
        
        Args:
            model_path: Path to saved model (.pkl)
            scaler_path: Path to saved scaler (.pkl)
            metadata_path: Path to metadata (.json)
        """
        # Try to load model with error handling for numpy compatibility
        try:
            self.model = joblib.load(model_path)
        except ValueError as e:
            if "BitGenerator" in str(e):
                # Numpy version mismatch - try alternative loading
                print(f"  ⚠️  Numpy compatibility issue detected, using workaround...")
                import pickle
                with open(model_path, 'rb') as f:
                    # Load with allow_pickle and ignore random state issues
                    self.model = pickle.load(f)
            else:
                raise
        
        self.scaler = joblib.load(scaler_path)
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.selected_features = self.metadata['selected_features']
        self.feature_extractor = GolfFeatureExtractorBiomech()
        
        print("="*70)
        print("  MODEL LOADED SUCCESSFULLY")
        print("="*70)
        print(f"  Stage: {self.metadata['stage']}")
        print(f"  Features: {self.metadata['n_features']}")
        print(f"  Training F1 (Macro): {self.metadata['train_f1_macro']*100:.2f}%")
        print(f"  Training Samples: {self.metadata['n_samples']}")
        print("="*70)
    
    def predict_from_skeleton_path(self, skeleton_path):
        """
        Predict handicap band from skeleton file.
        
        Args:
            skeleton_path: Path to .npy skeleton file
            
        Returns:
            prediction: Predicted band (0-4)
            probabilities: Class probabilities
        """
        # Extract features
        features = self.feature_extractor.extract_all_features(skeleton_path)
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        # Select only the features used in training
        features_df = features_df[self.selected_features]
        
        # Scale
        features_scaled = self.scaler.transform(features_df)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        return prediction, probabilities
    
    def predict_batch(self, skeleton_paths):
        """
        Predict for multiple skeleton files.
        
        Args:
            skeleton_paths: List of paths to .npy skeleton files
            
        Returns:
            predictions: Array of predictions
            probabilities: Array of probabilities
        """
        predictions = []
        probabilities = []
        
        for path in skeleton_paths:
            pred, prob = self.predict_from_skeleton_path(path)
            predictions.append(pred)
            probabilities.append(prob)
        
        return np.array(predictions), np.array(probabilities)
    
    def get_band_name(self, band_idx):
        """Convert band index to name."""
        band_names = {
            0: "band 0-2",
            1: "band 2-4",
            2: "band 4-6",
            3: "band 6-8",
            4: "band 8-10"
        }
        return band_names.get(band_idx, "unknown")


def example_usage():
    """Example of how to use the predictor."""
    
    # Find the latest model files
    model_dir = "models"
    model_files = sorted([f for f in os.listdir(model_dir) if f.startswith("stage2_model_")])
    
    if not model_files:
        print("No saved models found in 'models/' directory")
        return
    
    # Get latest model
    latest_timestamp = model_files[-1].split("_")[2].replace(".pkl", "")
    
    model_path = f"{model_dir}/stage2_model_{latest_timestamp}.pkl"
    scaler_path = f"{model_dir}/stage2_scaler_{latest_timestamp}.pkl"
    metadata_path = f"{model_dir}/stage2_metadata_{latest_timestamp}.json"
    
    # Load predictor
    predictor = GolfSwingPredictor(model_path, scaler_path, metadata_path)
    
    # Example prediction
    skeleton_path = "data/TDTU_skeletons_augmented/1.npy"
    
    if os.path.exists(skeleton_path):
        prediction, probabilities = predictor.predict_from_skeleton_path(skeleton_path)
        
        print("\n" + "="*70)
        print("  PREDICTION RESULT")
        print("="*70)
        print(f"  Skeleton: {skeleton_path}")
        print(f"  Predicted Band: {predictor.get_band_name(prediction)} (index: {prediction})")
        print(f"\n  Class Probabilities:")
        for i, prob in enumerate(probabilities):
            print(f"    {predictor.get_band_name(i):15s}: {prob*100:5.1f}%")
        print("="*70)
    else:
        print(f"\nExample skeleton not found: {skeleton_path}")
        print("Update the path to a real skeleton file to test prediction")


if __name__ == "__main__":
    example_usage()
