# -*- coding: utf-8 -*-
"""
train_3d_cnn.py
Train the attention-enhanced 3D CNN on 28-channel voxel grids.
Features:
- Load separated-channel voxel data (14 adsorbate + 14 solvent features)
- Train the 3D CNN model with its attention mechanisms
- Use GroupKFold to prevent data leakage between adsorbates
- Save and load model checkpoints and cross-validation results
- Support test mode for quick validation
- Command-line interface with argparse
"""

import os
import pandas as pd
import numpy as np
import pickle
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import traceback
import gc
import random  # Add random import

# Import custom functions
from core.path import get_paths
from core.global_vars import ZEOLITE_TYPES, ADSORBATES_BY_ENV
from load_grids_pickle import VoxelGridsLoader

# Import model architecture
from model_3d_cnn import AttentionCNN



class VoxelDataset(Dataset):
    """PyTorch dataset for 28-channel voxel grids."""
    
    def __init__(self, X, y, scaler=None, fit_scaler=False):
        """
        Args:
            X: numpy array of voxel grids with shape (N, 20, 20, 20, 28)
            y: numpy array of labels (N,)
            scaler: StandardScaler for labels
            fit_scaler: whether to fit the scaler
        """
        self.X = torch.FloatTensor(X)
        # Permute from (N, 20, 20, 20, 28) to (N, 28, 20, 20, 20) for PyTorch
        self.X = self.X.permute(0, 4, 1, 2, 3)
        
        if scaler is not None and fit_scaler:
            self.y = torch.FloatTensor(scaler.fit_transform(y.reshape(-1, 1)).flatten())
            self.scaler = scaler
        elif scaler is not None:
            self.y = torch.FloatTensor(scaler.transform(y.reshape(-1, 1)).flatten())
            self.scaler = scaler
        else:
            self.y = torch.FloatTensor(y)
            self.scaler = None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CNN3DTrainer:
    """
    Main class for training 3D CNN models with GroupKFold cross-validation
    """
    
    def __init__(self,
                 zeolite_types,
                 adsorbates_by_env,
                 test_mode=False,
                 retrain=False,
                 box_grids_size=16.0,
                 box_increment=0.8,
                 split_type='random_split',
                 batch_size=32,
                 epochs=200,
                 learning_rate=0.001,
                 verbose=True,
                 auto_run=True,
                 job_id='',
                 random_state=42,
                 output_prefix='cnn'):
        """
        Initialize the 3D CNN trainer
        
        Args:
            zeolite_types: list of zeolite types
            adsorbates_by_env: dict of environments and adsorbates
            test_mode: bool, if True run in test mode (quick training)
            retrain: bool, if True force retrain even if saved models exist
            box_grids_size: float, voxel grid size
            box_increment: float, voxel increment
            split_type: str, cross-validation split strategy ('random_split', 'solvent_split', 'pore_type_split')
            batch_size: int, training batch size
            epochs: int, number of training epochs
            learning_rate: float, learning rate for optimizer
            verbose: bool, print detailed information
            auto_run: bool, if True automatically run training after initialization
            job_id: str, job identifier for multi-job training
            random_state: int, random seed for reproducible splits (default: 42)
            output_prefix: str, prefix for output files (default: 'cnn')
        """
        # Set basic parameters first (before calling any methods that use them)
        self.verbose = verbose
        self.random_state = random_state
        self.output_prefix = output_prefix  # Store output prefix for file naming
        
        # Set random seeds for reproducibility
        self.set_random_seeds()
        
        self.zeolite_types = zeolite_types
        self.adsorbates_by_env = adsorbates_by_env
        self.test_mode = test_mode
        self.retrain = retrain
        self.box_grids_size = box_grids_size
        self.box_increment = box_increment
        self.split_type = split_type
        
        # Set n_folds based on split_type
        if self.split_type == 'random_split':
            self.n_folds = 5
        elif self.split_type == 'solvent_split':
            self.n_folds = 4
        elif self.split_type == 'pore_type_split':
            self.n_folds = 2
        else:
            raise ValueError(f"Unknown split_type: {self.split_type}")
        
        # Training parameters (now configurable)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.job_id = job_id
        
        # Generate job_id if not provided (for multi-job training identification)
        if not self.job_id:
            # Create split_type suffix for better file identification
            split_suffix = ""
            if self.split_type == "random_split":
                split_suffix = "random"
            elif self.split_type == "solvent_split":
                split_suffix = "solvent"
            elif self.split_type == "pore_type_split":
                split_suffix = "pore_type"
            
            # Try to get SLURM job ID first
            slurm_job_id = os.environ.get('SLURM_JOB_ID')
            if slurm_job_id:
                # Use output prefix for consistent naming with files, split type before SLURM job ID
                self.job_id = f"{self.output_prefix}-{split_suffix}-{slurm_job_id}"
            else:
                # Fallback to parameter-based ID for local runs
                self.job_id = f"{self.output_prefix}-lr{learning_rate}-bs{batch_size}-ep{epochs}-{split_type}"

        # Data storage
        self.num_adsorbates = 0
        self.num_points_per_adsorbate = 0
        self.num_total_datapoints = 0
        self.input_data_shape = None
        self.X_data = None
        self.y_data = None
        self.adsorbate_groups = None
        
        # Model storage
        self.model_storage = {}
        self.best_model_idx = None
        
        # Use directly defined path to ensure consistency
        self.model_save_dir = get_paths("output_model_cnn")
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # Initialize the loader for 14 adsorbate and 14 solvent channels.
        self.data_loader = VoxelGridsLoader(
                                            zeolite_types=self.zeolite_types,
                                            adsorbates_by_env=self.adsorbates_by_env,
                                            box_grids_size=self.box_grids_size,
                                            box_increment=self.box_increment,
                                            num_features=28,
                                            verbose=self.verbose
                                            )
        
        if self.verbose:
            print(f"\n--- CNN3DTrainer initialized")
            print(f"    Expected data format: 28 channels (14 adsorbate + 14 solvent features)")
            print(f"    Output directory: {self.model_save_dir}")
            print(f"    Job ID: {self.job_id}")
            print(f"    Device: {self.device}")
            print(f"    Test mode: {self.test_mode}")
            print(f"    Retrain: {self.retrain}")
            print(f"    Model Dir: {self.model_save_dir}")
        
        # Auto-run training if enabled
        if auto_run:
            start_time = time.time()
            self.run_training()
            end_time = time.time()
            
            if self.verbose:
                print(f"\n--- Total training time: {end_time - start_time:.2f} seconds")
                print("=== Training completed successfully! ===")
    
    def set_random_seeds(self):
        """Set random seeds for reproducibility"""
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)
        
        if self.verbose:
            print(f"--- Random seeds set to {self.random_state} for reproducibility")

    def create_model(self, dropout_rate=0.25):
        """Create the 28-channel 3D CNN model."""
        # Use feature names extracted from actual data loading
        # These will be set after load_data() is called
        feature_names = getattr(self, 'feature_names', None)
        
        if feature_names is None:
            # Fallback warning - this should not happen if load_data() was called first
            print("⚠️  Warning: Feature names not available yet. Using None for now.")
            print("    This is normal if called before data loading.")
        elif self.verbose:
            print(f"    Using {len(feature_names)} feature names from loaded data: {feature_names}")
        
        # Construct the model for separated adsorbate and solvent channels.
        model = AttentionCNN(
            in_channels=28, 
            dropout_rate=dropout_rate,
            feature_names=feature_names
        )
        if self.verbose:
            print(f"     Created 3D CNN model (28 channels)")
        
        return model

    def validate_voxel_data(self):
        """Validate the spatial dimensions and 28-channel input layout."""
        if self.X_data is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")
        
        expected_shape = (20, 20, 20, 28)
        actual_shape = self.X_data.shape[1:]
        
        if actual_shape != expected_shape:
            raise ValueError(
                f"Voxel-data shape mismatch: expected {expected_shape}, got {actual_shape}. "
                "Expected 14 adsorbate and 14 solvent channels."
            )
        
        if self.verbose:
            print(f"    Data validation passed: format confirmed (28 channels)")

    def load_data(self):
        """Load and prepare voxel grid data"""
        if self.verbose:
            print("\n=== Loading voxel grid data ===")
        
        # Load all pickle files
        self.loaded_data = self.data_loader.load_all_pickle_files()
        
        if not self.loaded_data:
            raise ValueError("No data loaded. Check your data configuration.")
        
        # Extract feature names from loader (already extracted during loading)
        self.feature_names = self.data_loader.feature_names
        if self.verbose:
            print(f"    Feature names ({len(self.feature_names)}): {self.feature_names}")
        
        # Extract feature channels mapping for molecular interaction analysis
        self.feature_channels = self.data_loader.feature_channels
        if self.verbose and self.feature_channels:
            print(f"    Feature channels mapping available for molecular interaction analysis")
        elif self.verbose:
            print(f"    No feature channels mapping available - using basic feature names")
        
        # Combine all data
        all_grids = []
        all_labels = []
        all_adsorbate_names = []
        all_metadata = []  # Store metadata for each data point
        
        for key, data in self.loaded_data.items():
            grids = data['voxel_grids']                 # length = 240
            labels = data['labels']                     # length = 240
            snapshot_indices = data['snapshot_indices'] # length = 240, Actual snapshot numbers
            voxel_ids = data['voxel_ids']               # length = 240, Rotation indices (1-24)
            adsorbate = data['adsorbate']               # one string
            zeolite = data['zeolite_type']              # one string
            environment = data['environment']           # one string

            # Filter out None labels
            valid_indices = [i for i, label in enumerate(labels) if label is not None]
            
            if valid_indices:
                valid_grids = [grids[i] for i in valid_indices]
                valid_labels = [labels[i] for i in valid_indices]
                
                all_grids.extend(valid_grids)
                all_labels.extend(valid_labels)
                all_adsorbate_names.extend([adsorbate] * len(valid_labels))
                
                # Add metadata for each valid data point using actual snapshot and voxel info
                for i in valid_indices:
                    all_metadata.append({
                        'zeolite': zeolite,
                        'environment': environment,
                        'adsorbate': adsorbate,
                        'snapshot': snapshot_indices[i],  # Use actual snapshot number
                        'voxel_id': voxel_ids[i]          # Use actual voxel rotation ID
                    })
        
        if not all_grids:
            raise ValueError("No valid data found with labels")
        
        # Convert to numpy arrays
        # Shape: (N, 20, 20, 20, 28), with 10 snapshots × 24 rotations per system.
        self.X_data = np.stack(all_grids)
        self.y_data = np.array(all_labels)  # Shape: (N,)
        
        # Create environment-adsorbate combinations for proper grouping
        self.env_adsorbate_groups = np.array([f"{meta['environment']}-{meta['adsorbate']}" for meta in all_metadata])
        self.adsorbate_groups = np.array(all_adsorbate_names)   # Keep for compatibility but use env_adsorbate_groups for CV
        self.metadata = all_metadata  # Store metadata list     # Shape: (N,)
        
        # Calculate statistics using environment-adsorbate combinations
        unique_env_adsorbates = np.unique(self.env_adsorbate_groups)
        self.num_env_adsorbates = len(unique_env_adsorbates)
        self.num_total_datapoints = len(self.X_data)
        self.num_points_per_env_adsorbate = self.num_total_datapoints // self.num_env_adsorbates
        self.input_data_shape = self.X_data.shape[1:]
        
        if self.verbose:
            print(f"\n--- Separated-channel voxel data loaded successfully:")
            print(f"    Number of environment-adsorbate combinations: {self.num_env_adsorbates}")
            print(f"    Total datapoints: {self.num_total_datapoints}")
            print(f"    Points per env-adsorbate: {self.num_points_per_env_adsorbate}")
            print(f"    Input data shape: {self.input_data_shape} (28 channels)")
            print(f"    Label range: [{self.y_data.min():.3f}, {self.y_data.max():.3f}]")
        
        # Validate the spatial dimensions and channel layout before training.
        self.validate_voxel_data()
    
    def get_cv_splits(self):
        """Get cross-validation splits based on split_type"""
        if self.split_type == 'random_split':
            return self._get_random_splits()
        elif self.split_type == 'solvent_split':
            return self._get_solvent_splits()
        elif self.split_type == 'pore_type_split':
            return self._get_pore_type_splits()
        else:
            raise ValueError(f"Unknown split_type: {self.split_type}. "
                           f"Must be one of: 'random_split', 'solvent_split', 'pore_type_split'")
    
    def _get_random_splits(self):
        """Get random splits using GroupKFold (current method)"""
        from sklearn.model_selection import GroupKFold

        unique_groups = np.unique(self.env_adsorbate_groups)
        num_groups = len(unique_groups)

        if num_groups < 5:
            if self.test_mode:
                if self.verbose:
                    print(f"Random Split: test-mode fallback because only {num_groups} group(s) are available")

                num_samples = len(self.X_data)
                test_size = max(1, int(round(num_samples * 0.2)))
                test_idx = np.arange(num_samples - test_size, num_samples)
                train_idx = np.arange(0, num_samples - test_size)
                return [(train_idx, test_idx)]

            raise ValueError(
                f"Random split requires at least 5 groups, but only {num_groups} "
                f"group(s) are available in the loaded dataset."
            )
        
        group_kfold = GroupKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        splits = list(group_kfold.split(self.X_data, self.y_data, self.env_adsorbate_groups))
        
        if self.verbose:
            print(f"Random Split: 5 folds, random_state={self.random_state}")
            
        return splits
    
    def _get_solvent_splits(self):
        """Get solvent-based splits (4 folds)"""
        import numpy as np
        
        # Fixed solvent types - always use these 4 solvents
        solvents = ["methanol_120_water_1080", "methanol_240_water_960", "methanol_600_water_600", "water_pure"]
        
        if self.verbose:
            print(f"Solvent Split: {len(solvents)} folds, solvents: {solvents}")
        
        splits = []
        for test_solvent in solvents:
            # Create masks for test and train
            test_mask = np.array([
                env_ads.rsplit('-', 1)[0].split('-')[0] == test_solvent 
                for env_ads in self.env_adsorbate_groups
            ])
            
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(~test_mask)[0]
            
            splits.append((train_idx, test_idx))
            
            if self.verbose:
                test_env_ads = set(self.env_adsorbate_groups[test_idx])
                train_env_ads = set(self.env_adsorbate_groups[train_idx])
                print(f"  Fold {len(splits)}: test={test_solvent}, "
                      f"train_combinations={len(train_env_ads)}, test_combinations={len(test_env_ads)}")
        
        return splits
    
    def _get_pore_type_splits(self):
        """Get pore-type-based splits (2 folds)"""
        import numpy as np
        
        # Fixed pore types - always use these 2 types
        pore_types = ["hydrophilic", "hydrophobic"]
        
        if self.verbose:
            print(f"Pore Type Split: {len(pore_types)} folds, pore_types: {pore_types}")
        
        splits = []
        for test_pore_type in pore_types:
            # Create masks for test and train
            test_mask = np.array([
                env_ads.rsplit('-', 1)[0].split('-')[-1] == test_pore_type 
                for env_ads in self.env_adsorbate_groups
            ])
            
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(~test_mask)[0]
            
            splits.append((train_idx, test_idx))
            
            if self.verbose:
                test_env_ads = set(self.env_adsorbate_groups[test_idx])
                train_env_ads = set(self.env_adsorbate_groups[train_idx])
                print(f"  Fold {len(splits)}: test={test_pore_type}, "
                      f"train_combinations={len(train_env_ads)}, test_combinations={len(test_env_ads)}")
        
        return splits
    
    
    def get_checkpoint_path(self, fold_idx):
        """Get checkpoint file path for a specific fold with detailed naming"""
        mode_suffix = "-test" if self.test_mode else ""
        
        # Create detailed filename with job_id and configuration parameters
        filename = (
                    f"{self.job_id}"
                    f"-epochs_{self.epochs}"
                    f"-bs_{self.batch_size}"
                    f"-lr_{self.learning_rate}"
                    f"-grid_{self.box_grids_size}_{self.box_increment}"
                    f"-fold_{fold_idx}"
                    f"{mode_suffix}.pth")
        print(f"    Checkpoint filename: {filename}")
        return os.path.join(self.model_save_dir, filename)
    
    def get_results_path(self):
        """Get results file path"""
        mode_suffix = "-test" if self.test_mode else ""
        filename = (
                    f"{self.job_id}"
                    f"-epochs_{self.epochs}"
                    f"-bs_{self.batch_size}"
                    f"-lr_{self.learning_rate}"
                    f"-grid_{self.box_grids_size}_{self.box_increment}"
                    f"{mode_suffix}.pkl")
        return os.path.join(self.model_save_dir, filename)
    
    def save_model(self, model, fold_idx, train_metrics, test_metrics, df_train, df_test, train_idx, test_idx, monitoring_data=None, scaler_info=None):
        """Save model checkpoint and results with monitoring data"""
        checkpoint_path = self.get_checkpoint_path(fold_idx)
        
        # Get environment-adsorbate groups for train and test indices
        train_env_adsorbates = list(set(self.env_adsorbate_groups[train_idx].tolist()))
        test_env_adsorbates = list(set(self.env_adsorbate_groups[test_idx].tolist()))
        
        # Use the scaler information passed from train_single_model
        if scaler_info is None:
            scaler_info = getattr(self, '_current_scaler_info', None)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'fold_idx': fold_idx,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'df_train': df_train.to_dict(),
            'df_test': df_test.to_dict(),
            'monitoring_data': monitoring_data,  # Add monitoring data
            'scaler_info': scaler_info,  # Add scaler information
            'split_info': {  # Add split information
                'train_idx': train_idx.tolist(),
                'test_idx': test_idx.tolist(),
                'train_env_adsorbates': sorted(train_env_adsorbates),
                'test_env_adsorbates': sorted(test_env_adsorbates),
                'random_state': self.random_state
            },
            'model_config': {
                'input_channels': 28,
                'test_mode': self.test_mode,
                'random_state': self.random_state
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        self.model_storage[fold_idx] = {
            'model': model,
            'train_rmse': train_metrics['rmse'],
            'train_r2': train_metrics['r2'],
            'train_mae': train_metrics['mae'],
            'test_rmse': test_metrics['rmse'],
            'test_r2': test_metrics['r2'],
            'test_mae': test_metrics['mae'],
            'df_train': df_train,
            'df_test': df_test,
            'monitoring_data': monitoring_data,  # Add monitoring data
            'scaler_info': scaler_info,  # 🆕 Add scaler information
            'split_info': {  # Add split information to model_storage
                'train_idx': train_idx.tolist(),
                'test_idx': test_idx.tolist(),
                'train_env_adsorbates': sorted(train_env_adsorbates),
                'test_env_adsorbates': sorted(test_env_adsorbates),
                'random_state': self.random_state
            }
        }
        
        if self.verbose:
            print(f"    Model saved: {checkpoint_path}")
            
            # Print detailed fold completion analysis
            self.print_fold_completion_analysis(fold_idx, train_metrics, test_metrics, 
                                               train_env_adsorbates, test_env_adsorbates, 
                                               monitoring_data)
    
    def print_fold_completion_analysis(self, fold_idx, train_metrics, test_metrics, 
                                     train_env_adsorbates, test_env_adsorbates, monitoring_data):
        """Print detailed analysis when a fold is completed"""

        print(f"\n{'='*60}")
        print(f"🎯 FOLD {fold_idx + 1}/{self.n_folds} COMPLETION ANALYSIS")
        print(f"{'='*60}")
        
        # Performance metrics
        generalization_gap = test_metrics['rmse'] - train_metrics['rmse']
        
        print(f"📊 Performance Metrics:")
        print(f"  Train: RMSE={train_metrics['rmse']:.4f}, R²={train_metrics['r2']:.4f}, MAE={train_metrics['mae']:.4f}")
        print(f"  Test:  RMSE={test_metrics['rmse']:.4f}, R²={test_metrics['r2']:.4f}, MAE={test_metrics['mae']:.4f}")
        print(f"  Generalization gap: {generalization_gap:.4f}")
        
        # Data split analysis
        print(f"\n🔄 Data Split Analysis:")
        print(f"  Train env-adsorbates: {len(train_env_adsorbates)}")
        print(f"  Test env-adsorbates:  {len(test_env_adsorbates)}")
        
        # Training monitoring analysis
        if monitoring_data and 'final_analysis' in monitoring_data:
            final_analysis = monitoring_data['final_analysis']
            print(f"\n🔬 Training Dynamics Analysis:")
            print(f"  Epochs completed: {final_analysis.get('total_epochs', 'N/A')}")
            print(f"  Samples per parameter: {final_analysis.get('samples_per_param', 'N/A'):.1f}")
        
        # Loss trend analysis
        if monitoring_data and 'train_losses' in monitoring_data and 'test_losses' in monitoring_data:
            train_losses = monitoring_data['train_losses']
            test_losses = monitoring_data['test_losses']
            
            if len(train_losses) >= 10:
                # Early vs late performance
                early_train = np.mean(train_losses[:5])
                late_train = np.mean(train_losses[-5:])
                early_test = np.mean(test_losses[:5])
                late_test = np.mean(test_losses[-5:])
                
                train_improvement = early_train - late_train
                test_improvement = early_test - late_test
                
                print(f"\n📈 Learning Progress Analysis:")
                print(f"  Train loss improvement: {train_improvement:.4f} ({early_train:.4f} → {late_train:.4f})")
                print(f"  Test loss improvement: {test_improvement:.4f} ({early_test:.4f} → {late_test:.4f})")
                
                if train_improvement < 0.05 and test_improvement < 0.05:
                    print(f"  ⚠️  Learning stagnation - both losses stopped improving")
                elif test_improvement > train_improvement * 0.3:
                    print(f"  ✓ Healthy learning pattern - test follows train")
                
                # Find best epoch
                best_test_epoch = np.argmin(test_losses)
                best_test_loss = test_losses[best_test_epoch]
                final_test_loss = test_losses[-1]
                
                print(f"  Best test loss: {best_test_loss:.4f} at epoch {best_test_epoch + 1}")
                print(f"  Performance degradation: {final_test_loss - best_test_loss:.4f}")
        
        print(f"{'='*60}\n")
    
    
    def load_model(self, fold_idx):
        """Load model checkpoint if exists"""
        checkpoint_path = self.get_checkpoint_path(fold_idx)
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Recreate the 28-channel model before loading its state dictionary.
            model_config = checkpoint.get('model_config', {})
            
            # Use feature names extracted from actual data loading
            feature_names = getattr(self, 'feature_names', None)
            if feature_names is None and self.verbose:
                print("    ⚠️  Warning: Feature names not available, using None for model reconstruction")
            
            model = AttentionCNN(
                in_channels=28,
                dropout_rate=0.10,
                feature_names=feature_names
            )
            if self.verbose:
                print(f"    Loaded 3D CNN model from checkpoint (28 channels)")
                    
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            
            # Load dataframes and split info if they exist in checkpoint
            df_train = None
            df_test = None
            split_info = None
            
            if 'df_train' in checkpoint and 'df_test' in checkpoint:
                df_train = pd.DataFrame.from_dict(checkpoint['df_train'])
                df_test = pd.DataFrame.from_dict(checkpoint['df_test'])
            
            if 'split_info' in checkpoint:
                split_info = checkpoint['split_info']
            
            if self.verbose:
                print(f"    Model loaded")
            
            return model, checkpoint['train_metrics'], checkpoint['test_metrics'], df_train, df_test, split_info
        
        return None, None, None, None, None, None
    
    
    def check_existing_models(self):
        """Check if all fold models exist"""
        all_exist = True
        for fold_idx in range(self.n_folds):
            if not os.path.exists(self.get_checkpoint_path(fold_idx)):
                all_exist = False
                break
        return all_exist
    
    

    def train_single_model(self, X_train, y_train, X_test, y_test, fold_idx, train_idx, test_idx):
        """Train a single model for one cross-validation fold."""
        if self.verbose:
            print(f"\n--- Training Fold {fold_idx + 1}/{self.n_folds} ---")

        scaler = StandardScaler()
        scaler.fit(y_train.reshape(-1, 1))
        self._current_scaler_info = {
            'mean': scaler.mean_[0],
            'std': scaler.scale_[0]
        }

        train_dataset = VoxelDataset(X_train, y_train, scaler=scaler, fit_scaler=True)
        test_dataset = VoxelDataset(X_test, y_test, scaler=scaler, fit_scaler=False)

        if self.verbose:
            print(f"    Dataset Analysis:")
            print(f"      Train set: {len(X_train)} samples")
            print(f"      Test set:  {len(X_test)} samples")
            print(f"      Label distribution (original scale):")
            print(f"        Train: mean={y_train.mean():.4f}, std={y_train.std():.4f}, range=[{y_train.min():.4f}, {y_train.max():.4f}]")
            print(f"        Test:  mean={y_test.mean():.4f}, std={y_test.std():.4f}, range=[{y_test.min():.4f}, {y_test.max():.4f}]")

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        train_eval_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        test_eval_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        if len(X_train) > 1500:
            dropout_rate = 0.15
        elif len(X_train) > 800:
            dropout_rate = 0.12
        else:
            dropout_rate = 0.10

        model = self.create_model(dropout_rate=dropout_rate)
        model.to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        samples_per_param = len(X_train) / trainable_params if trainable_params > 0 else 0

        if self.verbose:
            print(f"    Model Complexity Analysis:")
            print(f"      Total parameters: {total_params:,}")
            print(f"      Trainable parameters: {trainable_params:,}")
            print(f"      Samples per parameter: {samples_per_param:.2f}")

        criterion = nn.MSELoss()
        base_weight_decay = 1e-4
        weight_decay = base_weight_decay

        if self.verbose:
            print(f"    Dropout rate: {dropout_rate:.3f} (minimized for test loss stability)")
            print(f"    Weight decay: {weight_decay:.1e} (reduced for stable training)")

        adjusted_lr = self.learning_rate
        attention_params = []
        other_params = []

        for name, param in model.named_parameters():
            if ('attention' in name.lower() or 'cbam' in name.lower() or 'fc' in name.lower() or
                'pooling_weights' in name.lower()):
                attention_params.append(param)
            else:
                other_params.append(param)

        optimizer = optim.AdamW([
            {'params': other_params, 'weight_decay': weight_decay},
            {'params': attention_params, 'weight_decay': weight_decay * 0.3, 'lr': adjusted_lr * 0.9}
        ], lr=adjusted_lr, eps=1e-8, betas=(0.9, 0.999))

        max_grad_norm = 0.5
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.9,
            patience=25,
            min_lr=1e-9,
            threshold=1e-4,
            cooldown=5
        )

        best_test_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        early_stop_patience = 80
        min_improvement = 5e-5
        if self.test_mode:
            early_stop_patience = 2
        min_epochs = 50

        train_losses = []
        test_losses = []
        learning_rates = []
        if self.verbose:
            print(f"    🎯 OPTIMIZED Training Strategy (for LR=0.0002 slow convergence):")
            print(f"      Learning rate: {self.learning_rate:.6f}")
            print(f"      Weight decay: {weight_decay:.0e}")
            print(f"      Dropout rate: {dropout_rate:.2f}")
            print(f"      Early stop patience: {early_stop_patience} (extended for consistent fold training curves)")
            print(f"      Min improvement: {min_improvement:.0e} (more sensitive for gradual improvements)")
            print(f"      Min epochs: {min_epochs} (ensures consistent training duration across folds)")
            print(f"      Gradient clipping: {max_grad_norm}")

        for epoch in range(self.epochs):
            model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in test_eval_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    test_loss += loss.item()

            test_loss /= len(test_eval_loader)
            test_losses.append(test_loss)

            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            old_lr = current_lr
            scheduler.step(test_loss)
            new_lr = optimizer.param_groups[0]['lr']

            if old_lr != new_lr and self.verbose:
                print(f"  Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")

            improvement_flag = ""
            if test_loss < best_test_loss - min_improvement:
                best_test_loss = test_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                improvement_flag = "✓"
            else:
                patience_counter += 1

            if epoch >= min_epochs and patience_counter >= early_stop_patience:
                if self.verbose:
                    print(f"\n{'─' * 60}")
                    print("✅ EARLY STOPPING")
                    print(f"{'─' * 60}")
                    print(f"  Stopped at epoch {epoch + 1} (no improvement for {early_stop_patience} epochs)")
                    print(f"  Best test loss: {best_test_loss:.4f}")
                    print(f"  Current test loss: {test_loss:.4f}")
                break

            if new_lr < 5e-7:
                if self.verbose:
                    print(f"\n{'─' * 80}")
                    print("✅ TRAINING TERMINATION - LEARNING RATE TOO SMALL")
                    print(f"{'─' * 80}")
                    print(f"  Stopping at epoch {epoch + 1} (learning rate too small: {new_lr:.0e})")
                break

            if self.verbose:
                lr_change = f" → {new_lr:.6f}" if old_lr != new_lr else ""
                epoch_log_line = (
                    f"  Epoch {epoch + 1:3d}/{self.epochs}: "
                    f"Train={train_loss:.4f}, Test={test_loss:.4f} {improvement_flag}, "
                    f"LR={old_lr:.6f}{lr_change}, "
                    f"Patience={patience_counter}/{early_stop_patience}"
                )
                print(f"\n{epoch_log_line}")

                if len(train_losses) >= 5:
                    recent_train_trend = np.mean(train_losses[-5:]) - np.mean(train_losses[-10:-5]) if len(train_losses) >= 10 else 0
                    recent_test_trend = np.mean(test_losses[-5:]) - np.mean(test_losses[-10:-5]) if len(test_losses) >= 10 else 0
                    print("\n    📉 Learning Trends (last 5 epochs):")
                    train_trend_status = "↓ improving" if recent_train_trend < -0.01 else "↑ worsening" if recent_train_trend > 0.01 else "→ stable"
                    test_trend_status = "↓ improving" if recent_test_trend < -0.01 else "↑ worsening" if recent_test_trend > 0.01 else "→ stable"
                    print(f"      Train loss trend: {recent_train_trend:+.4f} ({train_trend_status})")
                    print(f"      Test loss trend:  {recent_test_trend:+.4f} ({test_trend_status})")

                print()

        if self.verbose and epoch == self.epochs - 1:
            print(f"\n{'─' * 80}")
            print("✅ TRAINING COMPLETED - ALL EPOCHS FINISHED")
            print(f"{'─' * 80}")
            print(f"  Completed all {self.epochs} epochs without early stopping")

        if self.verbose:
            print("    🔍 Final Training Analysis:")
            print(f"      Epochs completed: {epoch + 1}/{self.epochs}")
            print("    📈 Loss Convergence Analysis:")
            if len(train_losses) > 10:
                early_train_avg = np.mean(train_losses[:5])
                late_train_avg = np.mean(train_losses[-5:])
                early_test_avg = np.mean(test_losses[:5])
                late_test_avg = np.mean(test_losses[-5:])

                train_improvement = early_train_avg - late_train_avg
                test_improvement = early_test_avg - late_test_avg

                print(f"      Train loss improvement: {train_improvement:.4f} ({early_train_avg:.4f} → {late_train_avg:.4f})")
                print(f"      Test loss improvement: {test_improvement:.4f} ({early_test_avg:.4f} → {late_test_avg:.4f})")

                if train_improvement < 0.1 and test_improvement < 0.1:
                    print("      ⚠️  Learning stagnation: both train and test stopped improving")
                elif test_improvement > train_improvement * 0.5:
                    print("      ✓ Good learning: test improvement follows train improvement")

            print("    💡 Optimization Notes:")
            print(f"      • Current dropout rate: {dropout_rate:.2f}")
            print(f"      • Current weight decay: {weight_decay:.0e}")
            print("      • If performance is unsatisfactory, consider tuning model capacity, epochs, or learning rate")

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            if self.verbose:
                print(f"    ✅ Loaded best model (test loss: {best_test_loss:.4f})")
        else:
            if self.verbose:
                print("    ⚠️ Warning: No improvement found, using final model state")

        model.eval()
        train_predictions = []
        train_targets = []
        test_predictions = []
        test_targets = []

        with torch.no_grad():
            for batch_X, batch_y in train_eval_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = model(batch_X).squeeze()
                train_predictions.extend(outputs.cpu().numpy())
                train_targets.extend(batch_y.cpu().numpy())

            for batch_X, batch_y in test_eval_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = model(batch_X).squeeze()
                test_predictions.extend(outputs.cpu().numpy())
                test_targets.extend(batch_y.cpu().numpy())

        train_predictions = scaler.inverse_transform(np.array(train_predictions).reshape(-1, 1)).flatten()
        train_targets = scaler.inverse_transform(np.array(train_targets).reshape(-1, 1)).flatten()
        test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1)).flatten()
        test_targets = scaler.inverse_transform(np.array(test_targets).reshape(-1, 1)).flatten()

        train_metrics = {
            'rmse': root_mean_squared_error(train_targets, train_predictions),
            'r2': r2_score(train_targets, train_predictions),
            'mae': mean_absolute_error(train_targets, train_predictions)
        }

        test_metrics = {
            'rmse': root_mean_squared_error(test_targets, test_predictions),
            'r2': r2_score(test_targets, test_predictions),
            'mae': mean_absolute_error(test_targets, test_predictions)
        }

        train_metadata = [self.metadata[i] for i in self.current_train_idx]
        test_metadata = [self.metadata[i] for i in self.current_test_idx]

        df_train = pd.DataFrame({
            'zeolite': [meta['zeolite'] for meta in train_metadata],
            'environment': [meta['environment'] for meta in train_metadata],
            'adsorbate': [meta['adsorbate'] for meta in train_metadata],
            'snapshot': [meta['snapshot'] for meta in train_metadata],
            'voxel_id': [meta['voxel_id'] for meta in train_metadata],
            'fold': fold_idx,
            'y_true': train_targets,
            'y_pred': train_predictions
        })

        df_test = pd.DataFrame({
            'zeolite': [meta['zeolite'] for meta in test_metadata],
            'environment': [meta['environment'] for meta in test_metadata],
            'adsorbate': [meta['adsorbate'] for meta in test_metadata],
            'snapshot': [meta['snapshot'] for meta in test_metadata],
            'voxel_id': [meta['voxel_id'] for meta in test_metadata],
            'fold': fold_idx,
            'y_true': test_targets,
            'y_pred': test_predictions
        })

        if self.verbose:
            print(f"    Train RMSE: {train_metrics['rmse']:.4f}, R²: {train_metrics['r2']:.4f}, MAE: {train_metrics['mae']:.4f}")
            print(f"    Test RMSE: {test_metrics['rmse']:.4f}, R²: {test_metrics['r2']:.4f}, MAE: {test_metrics['mae']:.4f}")
            print("    🎯 Prediction Quality Analysis:")
            test_residuals = test_targets - test_predictions
            print(f"      Test residuals: μ={np.mean(test_residuals):.4f}, σ={np.std(test_residuals):.4f}")
            print(f"      Test range: predictions=[{test_predictions.min():.3f}, {test_predictions.max():.3f}], targets=[{test_targets.min():.3f}, {test_targets.max():.3f}]")

            worst_indices = np.argsort(np.abs(test_residuals))[-3:]
            print("      Worst predictions:")
            for i, idx in enumerate(worst_indices):
                error = test_residuals[idx]
                print(f"        {i+1}. True={test_targets[idx]:.4f}, Pred={test_predictions[idx]:.4f}, Error={error:.4f}")

            best_indices = np.argsort(np.abs(test_residuals))[:3]
            print("      Best predictions:")
            for i, idx in enumerate(best_indices):
                error = test_residuals[idx]
                print(f"        {i+1}. True={test_targets[idx]:.4f}, Pred={test_predictions[idx]:.4f}, Error={error:.4f}")

            low_energy_mask = test_targets < np.percentile(test_targets, 33)
            mid_energy_mask = ((test_targets >= np.percentile(test_targets, 33)) &
                               (test_targets <= np.percentile(test_targets, 67)))
            high_energy_mask = test_targets > np.percentile(test_targets, 67)

            if np.sum(low_energy_mask) > 0:
                low_rmse = np.sqrt(np.mean(test_residuals[low_energy_mask] ** 2))
                print(f"      Low energy RMSE: {low_rmse:.4f} (n={np.sum(low_energy_mask)})")
            if np.sum(mid_energy_mask) > 0:
                mid_rmse = np.sqrt(np.mean(test_residuals[mid_energy_mask] ** 2))
                print(f"      Mid energy RMSE: {mid_rmse:.4f} (n={np.sum(mid_energy_mask)})")
            if np.sum(high_energy_mask) > 0:
                high_rmse = np.sqrt(np.mean(test_residuals[high_energy_mask] ** 2))
                print(f"      High energy RMSE: {high_rmse:.4f} (n={np.sum(high_energy_mask)})")

        monitoring_data = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'learning_rates': learning_rates,
            'final_analysis': {
                'samples_per_param': samples_per_param,
                'total_epochs': epoch + 1
            }
        }

        return model, train_metrics, test_metrics, df_train, df_test, monitoring_data, self._current_scaler_info

    def run_training(self):
        """Run the complete training pipeline with GroupKFold CV"""
        if self.verbose:
            print("\n=== Starting 3D CNN Training ===")
        
        # Load data if not already loaded
        if self.X_data is None:
            self.load_data()
        
        # Get splits based on split_type
        splits = self.get_cv_splits()
        # Use the actual number of generated splits, which may differ in test-mode fallback.
        self.n_folds = len(splits)
        
        # Check for existing models
        if not self.retrain and self.check_existing_models():
            if self.verbose:
                print("\n--- Found existing trained models. Loading...")
            self.load_all_models()
            return
        else:
            if self.verbose:
                print("\n--- No existing models found or retraining is enabled. Starting training from scratch.")
        
        if self.verbose:
            print(f"    Training on {self.num_env_adsorbates} environment-adsorbate combinations")
        
        # Cross-validation loop - use predetermined splits
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            
            # Store current indices for metadata access in train_single_model
            self.current_train_idx = train_idx
            self.current_test_idx = test_idx
            
            # Check if model already exists and we're not retraining
            if not self.retrain:
                existing_model, train_metrics, test_metrics, df_train, df_test, split_info = self.load_model(fold_idx)
                if existing_model is not None:
                    if self.verbose:
                        print(f"Fold {fold_idx + 1}: Loaded existing model")
                        if split_info:
                            print(f"    Split info: train_env_adsorbates={len(split_info['train_env_adsorbates'])}, test_env_adsorbates={len(split_info['test_env_adsorbates'])}")
                    
                    # Store the loaded model data
                    if df_train is not None and df_test is not None:
                        self.model_storage[fold_idx] = {
                            'model': existing_model,
                            'train_rmse': train_metrics['rmse'],
                            'train_r2': train_metrics['r2'],
                            'train_mae': train_metrics['mae'],
                            'test_rmse': test_metrics['rmse'],
                            'test_r2': test_metrics['r2'],
                            'test_mae': test_metrics['mae'],
                            'df_train': df_train,
                            'df_test': df_test,
                            'split_info': split_info  # Add split_info
                        }
                    continue
            
            # Split data
            X_train, X_test = self.X_data[train_idx], self.X_data[test_idx]
            y_train, y_test = self.y_data[train_idx], self.y_data[test_idx]
            
            # Get environment-adsorbate information for this fold
            train_env_adsorbates = set(self.env_adsorbate_groups[train_idx].tolist())
            test_env_adsorbates = set(self.env_adsorbate_groups[test_idx].tolist())
            
            if self.verbose:
                print(f"\n--- Fold {fold_idx + 1}:")
                print(f"    Train env-adsorbates ({len(train_env_adsorbates)}): {sorted(train_env_adsorbates)}")
                print(f"    Test env-adsorbates ({len(test_env_adsorbates)}): {sorted(test_env_adsorbates)}")
                print(f"    Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
            # Train model
            model, train_metrics, test_metrics, df_train, df_test, monitoring_data, scaler_info = self.train_single_model(
                X_train, y_train, X_test, y_test, fold_idx, train_idx, test_idx
            )
            
            # Save model with split information
            self.save_model(model, fold_idx, train_metrics, test_metrics, df_train, df_test, train_idx, test_idx, monitoring_data, scaler_info)

            # Explicitly free memory
            del X_train, X_test, y_train, y_test, model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Save overall results
        self.save_overall_results()
        
        if self.verbose:
            print("\n=== Training completed ===")
            self.print_summary()
    
    
    def load_all_models(self):
        """Load all existing models"""
        for fold_idx in range(self.n_folds):
            model, train_metrics, test_metrics, df_train, df_test, split_info = self.load_model(fold_idx)
            if model is not None:
                # Use loaded dataframes if available, otherwise create placeholders
                if df_train is None or df_test is None:
                    if self.verbose:
                        print(f"    Warning: No prediction data found for fold {fold_idx}, creating placeholders")
                    df_train = pd.DataFrame({'fold': [fold_idx]})
                    df_test = pd.DataFrame({'fold': [fold_idx]})
                
                self.model_storage[fold_idx] = {
                    'model': model,
                    'train_rmse': train_metrics['rmse'],
                    'train_r2': train_metrics['r2'],
                    'train_mae': train_metrics['mae'],
                    'test_rmse': test_metrics['rmse'],
                    'test_r2': test_metrics['r2'],
                    'test_mae': test_metrics['mae'],
                    'df_train': df_train,
                    'df_test': df_test,
                    'split_info': split_info  # Add split_info
                }
        
        # Save pkl file after loading all models
        if self.model_storage:
            self.save_overall_results()
            if self.verbose:
                print(f"--- Generated model_storage for {len(self.model_storage)} folds and saved pkl file")

    def save_overall_results(self):
        """Save overall results summary"""
        results_path = self.get_results_path()
        
        overall_results = {
            'model_storage': self.model_storage,  # Each fold contains its own scaler_info
            'data_info': {
                'num_env_adsorbates': self.num_env_adsorbates,
                'num_total_datapoints': self.num_total_datapoints,
                'input_data_shape': self.input_data_shape
            },
            'training_config': {
                'test_mode': self.test_mode,
                'split_type': self.split_type,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'random_state': self.random_state,  # Add random_state to config
                'job_id': self.job_id  # Add job_id to config
            }
        }
        
        with open(results_path, 'wb') as f:
            pickle.dump(overall_results, f)
        
        if self.verbose:
            print(f"\n--- Overall results saved: {results_path}")
    
    def print_summary(self):
        """Print comprehensive training summary with detailed analysis"""
        if not self.model_storage:
            print("No models trained yet")
            return
        
        print("\n" + "="*80)
        print("=== COMPREHENSIVE TRAINING SUMMARY ===")
        print("="*80)
        
        # Calculate average metrics
        train_rmses = [self.model_storage[i]['train_rmse'] for i in range(self.n_folds)]
        train_r2s = [self.model_storage[i]['train_r2'] for i in range(self.n_folds)]
        train_maes = [self.model_storage[i]['train_mae'] for i in range(self.n_folds)]
        
        test_rmses = [self.model_storage[i]['test_rmse'] for i in range(self.n_folds)]
        test_r2s = [self.model_storage[i]['test_r2'] for i in range(self.n_folds)]
        test_maes = [self.model_storage[i]['test_mae'] for i in range(self.n_folds)]
        
        print("📊 Per-fold Performance Analysis:")
        print("-" * 80)
        for i in range(self.n_folds):
            print(f"  Fold {i+1}: Train RMSE={train_rmses[i]:.4f}, R²={train_r2s[i]:.4f} | "
                  f"Test RMSE={test_rmses[i]:.4f}, R²={test_r2s[i]:.4f}")
            
            # Add training details for each fold if available
            if 'monitoring_data' in self.model_storage[i] and self.model_storage[i]['monitoring_data']:
                monitor_data = self.model_storage[i]['monitoring_data']
                if 'final_analysis' in monitor_data:
                    final_analysis = monitor_data['final_analysis']
                    epochs_completed = final_analysis.get('total_epochs', 'N/A')
                    samples_per_param = final_analysis.get('samples_per_param', 0)
                    print(f"    Training details: {epochs_completed} epochs, "
                          f"samples/param={samples_per_param:.1f}")
        
        print(f"\n📈 Cross-Validation Performance Summary:")
        print("-" * 50)
        print(f"  Train: RMSE={np.mean(train_rmses):.4f}±{np.std(train_rmses):.4f}, "
              f"R²={np.mean(train_r2s):.4f}±{np.std(train_r2s):.4f}, "
              f"MAE={np.mean(train_maes):.4f}±{np.std(train_maes):.4f}")
        print(f"  Test:  RMSE={np.mean(test_rmses):.4f}±{np.std(test_rmses):.4f}, "
              f"R²={np.mean(test_r2s):.4f}±{np.std(test_r2s):.4f}, "
              f"MAE={np.mean(test_maes):.4f}±{np.std(test_maes):.4f}")
        
        # Performance consistency analysis
        print(f"\n🎯 Model Consistency Analysis:")
        print("-" * 40)
        test_rmse_cv = np.std(test_rmses) / np.mean(test_rmses) if np.mean(test_rmses) > 0 else 0
        test_r2_cv = np.std(test_r2s) / np.mean(test_r2s) if np.mean(test_r2s) > 0 else 0
        
        print(f"  Test RMSE consistency (CV): {test_rmse_cv:.3f} {'✓ Consistent' if test_rmse_cv < 0.1 else '⚠️ Variable' if test_rmse_cv < 0.2 else '❌ Highly Variable'}")
        print(f"  Test R² consistency (CV): {test_r2_cv:.3f} {'✓ Consistent' if test_r2_cv < 0.1 else '⚠️ Variable' if test_r2_cv < 0.2 else '❌ Highly Variable'}")
        
        # Find best and worst models
        self.best_model_idx = np.argmin(test_rmses)
        worst_model_idx = np.argmax(test_rmses)
        
        print(f"\n🏆 Best vs Worst Model Comparison:")
        print("-" * 45)
        print(f"  Best model (Fold {self.best_model_idx + 1}):")
        print(f"    Test RMSE: {test_rmses[self.best_model_idx]:.4f}, R²: {test_r2s[self.best_model_idx]:.4f}")
        print(f"  Worst model (Fold {worst_model_idx + 1}):")
        print(f"    Test RMSE: {test_rmses[worst_model_idx]:.4f}, R²: {test_r2s[worst_model_idx]:.4f}")
        print(f"  Performance gap: RMSE {test_rmses[worst_model_idx] - test_rmses[self.best_model_idx]:.4f}, "
              f"R² {test_r2s[self.best_model_idx] - test_r2s[worst_model_idx]:.4f}")
        
        # Overall model recommendations
        print(f"\n💡 Overall Model Optimization Recommendations:")
        print("-" * 55)
        print(f"  • Review RMSE/R² across folds and tune capacity, regularization, or epochs if needed")
        
        if test_rmse_cv > 0.15:
            print(f"  ⚠️  High performance variability across folds:")
            print(f"     • Consider stratified sampling")
            print(f"     • Check for data leakage")
            print(f"     • Increase training stability (lower learning rate)")
        
        # Training patterns analysis across folds
        print(f"\n📚 Cross-Fold Training Patterns:")
        print("-" * 45)
        
        # Analyze convergence patterns if available
        early_stopped_folds = 0
        avg_epochs_completed = []
        
        for i in range(self.n_folds):
            if 'monitoring_data' in self.model_storage[i] and self.model_storage[i]['monitoring_data']:
                monitor_data = self.model_storage[i]['monitoring_data']
                if 'final_analysis' in monitor_data:
                    final_analysis = monitor_data['final_analysis']
                    epochs = final_analysis.get('total_epochs', self.epochs)
                    avg_epochs_completed.append(epochs)
                    if epochs < self.epochs:
                        early_stopped_folds += 1
        
        if avg_epochs_completed:
            print(f"  Average epochs completed: {np.mean(avg_epochs_completed):.1f} / {self.epochs}")
            print(f"  Early stopping rate: {early_stopped_folds}/{self.n_folds} folds ({early_stopped_folds/self.n_folds*100:.1f}%)")
            
            if early_stopped_folds > self.n_folds * 0.8:
                print(f"  ⚠️  Most folds stopped early - consider more patience or different stopping criteria")
            elif early_stopped_folds == 0:
                print(f"  ⚠️  No early stopping occurred - might need more epochs or different criteria")
            else:
                print(f"  ✓ Reasonable early stopping pattern")
        
        avg_test_rmse = np.mean(test_rmses)
        print(f"\n📋 Performance Context:")
        print(f"  Current average test RMSE: {avg_test_rmse:.4f} eV")
        print("="*80)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train 3D CNN models with GroupKFold cross-validation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode settings
    parser.add_argument('--test-mode', action='store_true', default=False,
                       help='Run in test mode (quick training with limited dataset)')
    parser.add_argument('--retrain', action='store_true', default=False,
                       help='Force retrain even if saved models exist')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print detailed information during training')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0002,
                       help='Learning rate for optimizer')
    parser.add_argument('--split-type', type=str, default='random_split', 
                       choices=['random_split', 'solvent_split', 'pore_type_split'],
                       help='Cross-validation split strategy: random_split (5-fold), solvent_split (4-fold), pore_type_split (2-fold)')
    
    # Add parameter combination identifier for multi-job training
    parser.add_argument('--job-id', type=str, default='',
                       help='Job identifier for multi-job training (auto-generated if not provided)')
    
    # Data parameters
    parser.add_argument('--box-grids-size', type=float, default=16.0,
                       help='Voxel grid size')
    parser.add_argument('--box-increment', type=float, default=0.8,
                       help='Voxel increment')
    
    # Dataset selection
    parser.add_argument('--zeolite-types', nargs='+', default=None,
                       help='List of zeolite types to use (default: all from ZEOLITE_TYPES)')
    
    # Add random state argument
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducible cross-validation splits')
    
    # Output file prefix argument - hardcoded for train_3d_cnn.py
    parser.add_argument('--output-prefix', type=str, default='model',
                       help='Prefix for output files (e.g., model checkpoints, CSV files). Defaults to "model_2_1"')
    
    return parser.parse_args()


def get_dataset_config(args):
    """Get dataset configuration based on arguments"""
    if args.zeolite_types:
        zeolite_types = args.zeolite_types
    elif args.test_mode:
        zeolite_types = ['FAU']
    else:
        zeolite_types = ZEOLITE_TYPES
    
    if args.test_mode:
        adsorbates_by_env = {
            'methanol_240_water_960-hydrophilic': [
                '02_01_02_propanol'
            ]
        }
    else:
        adsorbates_by_env = ADSORBATES_BY_ENV
    
    return zeolite_types, adsorbates_by_env


def get_training_config(args):
    """Get training configuration based on arguments and mode"""
    if args.test_mode:
        # Override certain parameters for test mode
        config = {
            'epochs': 1,  # Fixed for test mode
            'batch_size': 8,  # Fixed for test mode
            'learning_rate': args.learning_rate,
            'split_type': args.split_type,
        }
    else:
        config = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'split_type': args.split_type,
        }
    
    return config

if __name__ == "__main__":
    try:
        print("\n=== 3D CNN Training Script ===")
        
        # Parse command line arguments
        args = parse_arguments()
        
        # Print configuration
        if args.verbose:
            print(f"--- Configuration:")
            print(f"    Job ID: {args.job_id if args.job_id else 'auto-generated'}")
            print(f"    Random state: {args.random_state}")
            print(f"    Test mode: {args.test_mode}")
            print(f"    Retrain: {args.retrain}")
            print(f"    Epochs: {1 if args.test_mode else args.epochs}")
            print(f"    Batch size: {8 if args.test_mode else args.batch_size}")
            print(f"    Learning rate: {args.learning_rate}")
            print(f"    Split type: {args.split_type}")
            print(f"    Box grid size: {args.box_grids_size}")
            print(f"    Box increment: {args.box_increment}")
            print(f"    Verbose: {args.verbose}")

        # Get dataset and training configuration
        zeolite_types, adsorbates_by_env = get_dataset_config(args)
        training_config = get_training_config(args)
    
        # Create and run trainer
        trainer = CNN3DTrainer(
            zeolite_types=zeolite_types,
            adsorbates_by_env=adsorbates_by_env,
            test_mode=args.test_mode,
            retrain=args.retrain,
            box_grids_size=args.box_grids_size,
            box_increment=args.box_increment,
            split_type=training_config['split_type'],
            batch_size=training_config['batch_size'],
            epochs=training_config['epochs'],
            learning_rate=training_config['learning_rate'],
            verbose=args.verbose,
            auto_run=True,
            job_id=args.job_id,
            random_state=args.random_state,
            output_prefix=args.output_prefix
        )
        
        #########################################
        ###### FOR DEBUGGING PURPOSES ONLY ######
        loaded_data = trainer.loaded_data
        model_storage = trainer.model_storage
        metadata = trainer.metadata
        ###### FOR DEBUGGING PURPOSES ONLY ######
        #########################################
        
        if args.verbose:
            print(f"\n=== Script completed successfully ===")
    
    except Exception as e:
        print("\n--- AN ERROR OCCURRED ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        print("\n--- Traceback ---")
        traceback.print_exc()
        sys.exit(1)
