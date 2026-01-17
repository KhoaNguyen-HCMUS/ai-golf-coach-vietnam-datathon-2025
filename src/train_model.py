"""
4-Stage Iterative Feature Pruning & Optimization Pipeline
Stage 1: Train with all features → Find top features
Stage 2: Train with only top features → Lighter model
Stage 3: Hyperparameter tuning → Find best params
Stage 4: LOOCV with optimized params → Final accuracy

Automatically finds best features and best hyperparameters, then validates with LOOCV
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from extract_features_biomech_augmented import GolfFeatureExtractorBiomech
from augment_skeletons_v2 import augment_skeleton_heavy
from family_grouping_utils import get_train_ids_excluding_family, print_family_statistics

class IterativeFeaturePruningTrainer:
    def __init__(self, top_n_features=20, n_jobs=-1):
        """
        Two-stage training with feature pruning.
        
        Args:
            top_n_features: Number of top features to keep in stage 2
            n_jobs: Number of parallel jobs (-1 = use all cores, 1 = sequential)
        """
        self.feature_extractor = GolfFeatureExtractorBiomech()
        self.top_n_features = top_n_features
        self.feature_importances = {}
        self.selected_features = None
        self.n_jobs = n_jobs
        
    def create_ensemble(self, params=None):
        """
        Create ensemble model.
        
        Args:
            params: Optional dict of hyperparameters from GridSearch
        """
        # Default params
        rf_n_estimators = 50
        rf_max_depth = 5
        gb_n_estimators = 50
        gb_learning_rate = 0.1
        final_C = 0.3
        
        # Override with optimized params if provided
        if params:
            rf_n_estimators = params.get('rf__n_estimators', rf_n_estimators)
            rf_max_depth = params.get('rf__max_depth', rf_max_depth)
            gb_n_estimators = params.get('gb__n_estimators', gb_n_estimators)
            gb_learning_rate = params.get('gb__learning_rate', gb_learning_rate)
            final_C = params.get('final_estimator__C', final_C)
        
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, 
                                         min_samples_leaf=3, random_state=42,
                                         class_weight='balanced',
                                         ccp_alpha=0.01)),
            ('gb', GradientBoostingClassifier(n_estimators=gb_n_estimators, max_depth=3, 
                                             learning_rate=gb_learning_rate, random_state=42,
                                             subsample=0.8)),
            ('svm', SVC(kernel='rbf', C=0.5, gamma='scale',
                       probability=True, random_state=42,
                       class_weight='balanced')),
            ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance')),
        ]
        
        return StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(
                solver='saga',
                l1_ratio=1.0,  # L1 regularization (equivalent to penalty='l1')
                C=final_C,
                max_iter=2000, 
                random_state=42,
                class_weight='balanced'
            ),
            cv=3,
            stack_method='predict_proba'
        )

    def augment_and_extract_features(self, sample_ids, skeleton_dir, metadata_df, augment=True):
        """Extract features for samples."""
        band_map = {
            "band 0-2": 0, "band 1-2": 0,
            "band 2-4": 1,
            "band 4-6": 2,
            "band 6-8": 3,
            "band 8-10": 4,
        }
        
        all_features = []
        all_labels = []
        
        for sample_id in sample_ids:
            skeleton_path = f"{skeleton_dir}/{sample_id}.npy"
            
            if not os.path.exists(skeleton_path):
                continue
            
            row = metadata_df[metadata_df['id'] == sample_id].iloc[0]
            band_str = str(row.get("band", "")).strip().lower()
            label = band_map.get(band_str, 2)
            
            try:
                features = self.feature_extractor.extract_all_features(skeleton_path)
                all_features.append(features)
                all_labels.append(label)
            except Exception:
                continue
            
            if augment:
                if label == 0:   n_aug = 11
                elif label == 1: n_aug = 14
                elif label == 2: n_aug = 9
                elif label == 3: n_aug = 9
                elif label == 4: n_aug = 14
                else: n_aug = 2
                
                skeleton = np.load(skeleton_path)
                # Use sample_id as seed for reproducible augmentation
                augmented = augment_skeleton_heavy(skeleton, n_aug, seed=sample_id)
                
                for aug_skel in augmented:
                    try:
                        features = self.feature_extractor.extract_all_features(aug_skel)
                        all_features.append(features)
                        all_labels.append(label)
                    except Exception:
                        pass
        
        return all_features, all_labels
    
    def run_loocv_stage(self, tdtu_metadata_path, tdtu_skeleton_dir, stage_name, use_selected_features=False, best_params=None):
        """
        Run one LOOCV stage.
        
        Args:
            best_params: Optional dict of best hyperparameters from GridSearch
        """
        df = pd.read_csv(tdtu_metadata_path)
        
        backside_df = df[df['view'] == 'backside']
        side_df = df[df['view'] == 'side']
        
        backside_original_ids = backside_df[backside_df['id'] <= 50]['id'].tolist()
        side_original_ids = side_df[side_df['id'] <= 50]['id'].tolist()
        
        backside_all_ids = backside_df['id'].tolist()
        side_all_ids = side_df['id'].tolist()
        
        band_map = {
            "band 0-2": 0, "band 1-2": 0,
            "band 2-4": 1,
            "band 4-6": 2,
            "band 6-8": 3,
            "band 8-10": 4,
        }
        
        print("\n" + "="*70)
        print(f"  {stage_name.upper()}")
        if use_selected_features:
            print(f"  Using {len(self.selected_features)} selected features")
            print(f"  Selected: {', '.join(self.selected_features[:5])}...")
        else:
            print(f"  Using all 39 features")
        print("="*70)
        
        all_results = {}
        stage_importances = {}
        
        # Create log directory and files
        log_dir = f"outputs/iterative_pruning_logs"
        os.makedirs(log_dir, exist_ok=True)
        stage_log_name = stage_name.lower().replace(' ', '_').replace(':', '')
        
        for original_test_ids, all_ids, view_name in [
            (backside_original_ids, backside_all_ids, 'Backside'), 
            (side_original_ids, side_all_ids, 'Side')
        ]:
            print(f"\n  Processing {view_name} ({len(original_test_ids)} videos)...")
            
            all_predictions = []
            all_true_labels = []
            all_test_ids = []
            
            # Create log file for this view
            log_path = f"{log_dir}/{stage_log_name}_{view_name.lower()}.csv"
            with open(log_path, 'w') as f:
                f.write("Fold,Test_ID,True_Label,Pred_Label,Correct,Train_F1,Val_F1,Train_Size,Val_Size,Family_Excluded\n")
            
            if view_name not in stage_importances:
                stage_importances[view_name] = {'rf': [], 'gb': []}
            
            # Define function to process one fold
            def process_fold(fold_idx, test_id):
                train_ids = get_train_ids_excluding_family(all_ids, test_id, df)
                test_row = df[df['id'] == test_id].iloc[0]
                true_label = band_map.get(str(test_row.get("band", "")).strip().lower(), 2)
                
                # Count excluded family members
                family_excluded = len(all_ids) - len(train_ids) - 1
                
                X_train, y_train = self.augment_and_extract_features(
                    train_ids, tdtu_skeleton_dir, df, augment=True
                )
                
                if not X_train:
                    return None
                
                X_train_df = pd.DataFrame(X_train)
                
                # Filter to selected features if stage 2/4
                if use_selected_features and self.selected_features:
                    X_train_df = X_train_df[self.selected_features]
                
                # Store feature names for this stage
                feature_names = X_train_df.columns.tolist() if not use_selected_features else None
                
                y_train_arr = np.array(y_train)
                
                try:
                    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                        X_train_df, y_train_arr, test_size=0.2, random_state=42, stratify=y_train_arr
                    )
                except ValueError:
                    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                        X_train_df, y_train_arr, test_size=0.2, random_state=42
                    )
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_split)
                X_val_scaled = scaler.transform(X_val_split)
                
                # Train model with optional best hyperparameters
                model = self.create_ensemble(params=best_params)
                model.fit(X_train_scaled, y_train_split)
                
                # Get importances (only in stage 1)
                rf_importances = None
                gb_importances = None
                if not use_selected_features:
                    rf_model = model.named_estimators_['rf']
                    gb_model = model.named_estimators_['gb']
                    rf_importances = rf_model.feature_importances_
                    gb_importances = gb_model.feature_importances_
                
                # Calculate train and validation F1 scores (macro)
                train_pred = model.predict(X_train_scaled)
                train_f1 = f1_score(y_train_split, train_pred, average='macro', zero_division=0)
                
                val_pred = model.predict(X_val_scaled)
                val_f1 = f1_score(y_val_split, val_pred, average='macro', zero_division=0)
                
                # Test
                X_test, _ = self.augment_and_extract_features(
                    [test_id], tdtu_skeleton_dir, df, augment=False
                )
                
                if not X_test:
                    return None
                
                X_test_df = pd.DataFrame(X_test)
                if use_selected_features and self.selected_features:
                    X_test_df = X_test_df[self.selected_features]
                
                X_test_scaled = scaler.transform(X_test_df)
                pred = model.predict(X_test_scaled)[0]
                
                return {
                    'test_id': test_id,
                    'true_label': true_label,
                    'pred': pred,
                    'train_f1': train_f1,
                    'val_f1': val_f1,
                    'train_size': len(y_train_split),
                    'val_size': len(y_val_split),
                    'family_excluded': family_excluded,
                    'rf_importances': rf_importances,
                    'gb_importances': gb_importances,
                    'feature_names': feature_names
                }
            
            # Process folds in parallel or sequential
            if self.n_jobs == 1:
                # Sequential processing (original behavior)
                results_list = []
                for fold_idx, test_id in enumerate(tqdm(original_test_ids, desc=f"{view_name}")):
                    result = process_fold(fold_idx, test_id)
                    if result:
                        results_list.append(result)
            else:
                # Parallel processing
                print(f"  Using {self.n_jobs if self.n_jobs > 0 else 'all'} CPU cores for parallel training...")
                results_list = Parallel(n_jobs=self.n_jobs)(
                    delayed(process_fold)(fold_idx, test_id) 
                    for fold_idx, test_id in enumerate(tqdm(original_test_ids, desc=f"{view_name}"))
                )
                # Filter out None results
                results_list = [r for r in results_list if r is not None]
            
            # Collect results
            for fold_idx, result in enumerate(results_list):
                all_predictions.append(result['pred'])
                all_test_ids.append(result['test_id'])
                all_true_labels.append(result['true_label'])
                
                # Store feature importances (only stage 1)
                if result['rf_importances'] is not None:
                    if 'feature_names' not in stage_importances[view_name]:
                        stage_importances[view_name]['feature_names'] = result['feature_names']
                    stage_importances[view_name]['rf'].append(result['rf_importances'])
                    stage_importances[view_name]['gb'].append(result['gb_importances'])
                
                # Log this fold
                with open(log_path, 'a') as f:
                    f.write(f"{fold_idx+1},{result['test_id']},{result['true_label']},{result['pred']},{1 if result['pred']==result['true_label'] else 0},"
                           f"{result['train_f1']:.4f},{result['val_f1']:.4f},{result['train_size']},{result['val_size']},{result['family_excluded']}\n")
            
            print(f"  ✓ Logged to: {log_path}")
            results = self.calculate_results(all_predictions, all_true_labels)
            all_results[view_name] = results
        
        return all_results, stage_importances
    
    def calculate_results(self, predictions, true_labels):
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        # Calculate F1 score (macro) as main metric
        f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
        relaxed_correct = sum(abs(t - p) <= 1 for t, p in zip(true_labels, predictions))
        relaxed_acc = relaxed_correct / len(true_labels) if len(true_labels) > 0 else 0
        
        cm = confusion_matrix(true_labels, predictions, labels=range(5))
        
        return {
            'predictions': predictions,
            'true_labels': true_labels,
            'f1_score': f1,
            'relaxed_accuracy': relaxed_acc,
            'confusion_matrix': cm,
        }
    
    def select_top_features(self, stage1_importances):
        """Select top N features based on combined importance."""
        print("\n" + "="*70)
        print("  SELECTING TOP FEATURES")
        print("="*70)
        
        all_feature_scores = {}
        
        for view_name, importances in stage1_importances.items():
            feature_names = importances['feature_names']
            rf_importance = np.mean(importances['rf'], axis=0)
            gb_importance = np.mean(importances['gb'], axis=0)
            combined = (rf_importance + gb_importance) / 2
            
            for fname, score in zip(feature_names, combined):
                if fname not in all_feature_scores:
                    all_feature_scores[fname] = []
                all_feature_scores[fname].append(score)
        
        # Average across views
        avg_scores = {fname: np.mean(scores) for fname, scores in all_feature_scores.items()}
        
        # Sort and select top N
        sorted_features = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        self.selected_features = [f[0] for f in sorted_features[:self.top_n_features]]
        
        print(f"\nSelected top {self.top_n_features} features:")
        for i, (fname, score) in enumerate(sorted_features[:self.top_n_features], 1):
            print(f"  {i:2d}. {fname:35s} (score: {score:.4f})")
        
        print(f"\nDropped {len(sorted_features) - self.top_n_features} features with low importance")
    
    def train_and_save_final_model(self, tdtu_metadata_path, tdtu_skeleton_dir, output_dir="models"):
        """Train final model on all data and save it."""
        print("\n" + "="*70)
        print("  💾 TRAINING & SAVING FINAL MODEL (Stage 2)")
        print("="*70)
        
        df = pd.read_csv(tdtu_metadata_path)
        all_ids = df['id'].tolist()
        
        print("\n  Extracting features from all data...")
        X_all, y_all = self.augment_and_extract_features(
            all_ids, tdtu_skeleton_dir, df, augment=True
        )
        
        if not X_all:
            print("  ❌ No data found!")
            return
        
        X_df = pd.DataFrame(X_all)
        
        # Use selected features from Stage 2
        if self.selected_features:
            X_df = X_df[self.selected_features]
            print(f"  Using {len(self.selected_features)} selected features")
        
        y_arr = np.array(y_all)
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df)
        
        # Train final model with default params (Stage 2)
        print("\n  Training final model...")
        model = self.create_ensemble(params=None)  # Use default params
        model.fit(X_scaled, y_arr)
        
        # Calculate training F1 score
        train_pred = model.predict(X_scaled)
        train_f1 = f1_score(y_arr, train_pred, average='macro', zero_division=0)
        print(f"  Training F1 Score (Macro): {train_f1*100:.2f}%")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = f"{output_dir}/stage2_model_{timestamp}.pkl"
        joblib.dump(model, model_path)
        print(f"\n  ✅ Model saved: {model_path}")
        
        # Save scaler
        scaler_path = f"{output_dir}/stage2_scaler_{timestamp}.pkl"
        joblib.dump(scaler, scaler_path)
        print(f"  ✅ Scaler saved: {scaler_path}")
        
        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "n_features": len(self.selected_features),
            "selected_features": self.selected_features,
            "train_f1_macro": float(train_f1),
            "n_samples": len(y_arr),
            "class_distribution": {int(k): int(v) for k, v in zip(*np.unique(y_arr, return_counts=True))},
            "model_type": "StackingClassifier (RF + GB + SVM + KNN → LogisticRegression)",
            "stage": "Stage 2 - Top Features (Default Hyperparameters)"
        }
        
        metadata_path = f"{output_dir}/stage2_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✅ Metadata saved: {metadata_path}")
        
        print(f"\n  📦 Model Package Summary:")
        print(f"     - Model: {model_path}")
        print(f"     - Scaler: {scaler_path}")
        print(f"     - Metadata: {metadata_path}")
        print(f"     - Features: {len(self.selected_features)} selected")
        print(f"     - Training F1: {train_f1*100:.1f}%")
        
        return model_path, scaler_path, metadata_path

    
    def tune_hyperparameters(self, tdtu_metadata_path, tdtu_skeleton_dir):
        """Stage 3: Find best hyperparameters using GridSearchCV."""
        print("\n" + "🟡"*35)
        print("  STAGE 3: Hyperparameter Tuning")
        print("🟡"*35)
        
        df = pd.read_csv(tdtu_metadata_path)
        
        # Collect ALL training data (not LOOCV, just gather data)
        all_ids = df['id'].tolist()
        
        print("\n  Collecting training data with selected features...")
        X_train, y_train = self.augment_and_extract_features(
            all_ids, tdtu_skeleton_dir, df, augment=True
        )
        
        X_train_df = pd.DataFrame(X_train)
        if self.selected_features:
            X_train_df = X_train_df[self.selected_features]
        
        y_train_arr = np.array(y_train)
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train_df)
        
        # Define hyperparameter grid (REDUCED - only most important params)
        param_grid = {
            # RandomForest - Most important for ensemble
            'rf__n_estimators': [100, 200],        # Number of trees (CRITICAL)
            'rf__max_depth': [5, 7],               # Tree depth (CRITICAL)
            
            # GradientBoosting - Second most important
            'gb__n_estimators': [50, 100],         # Boosting rounds (CRITICAL)
            'gb__learning_rate': [0.05, 0.1],      # Learning rate (CRITICAL)
            
            # Meta-learner (LogisticRegression)
            'final_estimator__C': [0.1, 0.5],      # Regularization (Important)
        }
        
        total_combinations = 2 * 2 * 2 * 2 * 2  # = 32 combinations
        print(f"  Searching {len(param_grid)} critical hyperparameters...")
        print(f"  Total combinations: {total_combinations} (much faster!)")
        
        # Create base ensemble
        base_ensemble = self.create_ensemble()
        
        # GridSearchCV with 3-fold CV
        grid_search = GridSearchCV(
            base_ensemble,
            param_grid,
            cv=3,
            scoring='f1_macro',
            n_jobs=-1,  # Use all CPU cores
            verbose=1
        )
        
        print("\n  Running Grid Search (this may take a while)...")
        grid_search.fit(X_scaled, y_train_arr)
        
        print("\n" + "="*70)
        print("  BEST HYPERPARAMETERS FOUND")
        print("="*70)
        for param, value in grid_search.best_params_.items():
            print(f"  {param:35s}: {value}")
        
        print(f"\n  Best CV F1 Score: {grid_search.best_score_*100:.2f}%")
        
        return grid_search.best_params_, grid_search.best_estimator_
    
    def run_four_stage_training(self, tdtu_metadata_path, tdtu_skeleton_dir):
        """Run full 4-stage pipeline: Feature Selection → Pruning → Hyperparameter Tuning → Final LOOCV."""
        df = pd.read_csv(tdtu_metadata_path)
        print_family_statistics(df)
        
        # STAGE 1: All features
        print("\n" + "🔵"*35)
        print("  STAGE 1: Training with ALL features")
        print("🔵"*35)
        stage1_results, stage1_importances = self.run_loocv_stage(
            tdtu_metadata_path, tdtu_skeleton_dir, 
            "Stage 1: All Features", 
            use_selected_features=False
        )
        
        # Select top features
        self.select_top_features(stage1_importances)
        
        # STAGE 2: Top features only
        print("\n" + "🟢"*35)
        print("  STAGE 2: Training with TOP features")
        print("🟢"*35)
        stage2_results, _ = self.run_loocv_stage(
            tdtu_metadata_path, tdtu_skeleton_dir, 
            "Stage 2: Top Features", 
            use_selected_features=True
        )
        
        # Compare Stage 1 vs Stage 2
        self.print_comparison(stage1_results, stage2_results)
        
        # Save Stage 2 model (before hyperparameter tuning)
        self.train_and_save_final_model(tdtu_metadata_path, tdtu_skeleton_dir)

        
        # STAGE 3: Hyperparameter Tuning
        best_params, best_model = self.tune_hyperparameters(
            tdtu_metadata_path, tdtu_skeleton_dir
        )
        
        # STAGE 4: LOOCV with Best Hyperparameters
        print("\n" + "🟣"*35)
        print("  STAGE 4: LOOCV with OPTIMIZED Hyperparameters")
        print("🟣"*35)
        stage4_results, _ = self.run_loocv_stage(
            tdtu_metadata_path, tdtu_skeleton_dir, 
            "Stage 4: Optimized Model", 
            use_selected_features=True,
            best_params=best_params
        )
        
        # Final Summary
        print("\n" + "="*70)
        print("  🎯 4-STAGE PIPELINE COMPLETE!")
        print("="*70)
        
        # Calculate F1 scores (macro)
        stage2_true = list(stage2_results['Backside']['true_labels']) + list(stage2_results['Side']['true_labels'])
        stage2_pred = list(stage2_results['Backside']['predictions']) + list(stage2_results['Side']['predictions'])
        stage2_f1 = f1_score(stage2_true, stage2_pred, average='macro', zero_division=0)
        
        stage4_true = list(stage4_results['Backside']['true_labels']) + list(stage4_results['Side']['true_labels'])
        stage4_pred = list(stage4_results['Backside']['predictions']) + list(stage4_results['Side']['predictions'])
        stage4_f1 = f1_score(stage4_true, stage4_pred, average='macro', zero_division=0)
        
        print(f"\n  📊 F1 SCORE (MACRO) PROGRESSION:")
        print(f"  {'Stage':<25} | F1 Score")
        print(f"  {'-'*40}")
        print(f"  {'Stage 1 (39 features)':<25} | {f1_score([x for r in stage1_results.values() for x in r['true_labels']], [x for r in stage1_results.values() for x in r['predictions']], average='macro', zero_division=0)*100:>6.1f}%")
        print(f"  {'Stage 2 (22 features)':<25} | {stage2_f1*100:>6.1f}%")
        print(f"  {'Stage 3 (CV only)':<25} | (GridSearchCV)")
        print(f"  {'Stage 4 (22 + optimized)':<25} | {stage4_f1*100:>6.1f}% ⭐")
        
        improvement = (stage4_f1 - stage2_f1) * 100
        if improvement > 0:
            print(f"\n  ✅ Hyperparameter tuning IMPROVED F1 score by {improvement:+.1f}%!")
        elif improvement < 0:
            print(f"\n  ⚠️  Hyperparameter tuning DECREASED F1 score by {improvement:.1f}%")
            print(f"      (Use Stage 2 default params instead)")
        else:
            print(f"\n  ➖ No change in F1 score (both equally good)")
        
        print(f"\n  💡 Final Model Configuration:")
        print(f"     - Features: {self.top_n_features} (reduced from 39)")
        print(f"     - Best Hyperparameters:")
        for param, value in best_params.items():
            print(f"       • {param}: {value}")
        print(f"     - Final LOOCV F1 Score (Macro): {stage4_f1*100:.1f}%")
        
        return stage1_results, stage2_results, stage4_results, best_params
    
    def print_comparison(self, stage1_results, stage2_results):
        """Compare stage 1 vs stage 2 results."""
        print("\n" + "="*70)
        print("  COMPARISON: ALL FEATURES vs TOP FEATURES")
        print("="*70)
        
        # Combined F1 scores
        stage1_preds = []
        stage1_true = []
        stage2_preds = []
        stage2_true = []
        
        for view in ['Backside', 'Side']:
            stage1_preds.extend(stage1_results[view]['predictions'])
            stage1_true.extend(stage1_results[view]['true_labels'])
            stage2_preds.extend(stage2_results[view]['predictions'])
            stage2_true.extend(stage2_results[view]['true_labels'])
        
        stage1_f1 = f1_score(stage1_true, stage1_preds, average='macro', zero_division=0)
        stage2_f1 = f1_score(stage2_true, stage2_preds, average='macro', zero_division=0)
        
        stage1_relaxed = sum(abs(t - p) <= 1 for t, p in zip(stage1_true, stage1_preds)) / len(stage1_true)
        stage2_relaxed = sum(abs(t - p) <= 1 for t, p in zip(stage2_true, stage2_preds)) / len(stage2_true)
        
        print(f"\n{'Metric':<30} | Stage 1 (39)  | Stage 2 ({self.top_n_features})  | Difference")
        print("-" * 70)
        print(f"{'F1 Score (Macro)':<30} | {stage1_f1:>12.1%} | {stage2_f1:>12.1%} | {(stage2_f1-stage1_f1):>+10.1%}")
        print(f"{'±1 Band Accuracy':<30} | {stage1_relaxed:>12.1%} | {stage2_relaxed:>12.1%} | {(stage2_relaxed-stage1_relaxed):>+10.1%}")
        print(f"{'Number of Features':<30} | {39:>12d} | {self.top_n_features:>12d} | {self.top_n_features-39:>10d}")
        print(f"{'Feature Reduction':<30} | {0:>11.1%} | {100*(39-self.top_n_features)/39:>11.1%} |")
        
        if stage2_f1 >= stage1_f1:
            print("\n✅ Stage 2 (pruned) BETTER or EQUAL: Lighter model with same/better F1 score!")
        else:
            print(f"\n⚠️  Stage 2 (pruned) WORSE: Lost {(stage1_f1-stage2_f1)*100:.1f}% F1 score")
            if (stage1_f1 - stage2_f1) < 0.05:
                print("   But difference is small (<5%), acceptable trade-off for lighter model")

def main():
    trainer = IterativeFeaturePruningTrainer(
        top_n_features=20,  # Keep top 20 features
        n_jobs=4  # Use 2-3 parallel jobs (safe + faster than sequential)
    )
    
    trainer.run_four_stage_training(
        tdtu_metadata_path="video_metadata.csv",
        tdtu_skeleton_dir="data/TDTU_skeletons_augmented"
    )

if __name__ == "__main__":
    main()
