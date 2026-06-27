# -*- coding: utf-8 -*-
"""
train_3d_cnn_2_8.py
Training script specifically for Type_2 voxel format (28 channels).
Features:
- Load Type_2 voxel grid data (28 channels: 14 adsorbate + 14 solvent features)
- Train Type_2 3D CNN model with enhanced attention mechanism
- Use GroupKFold to prevent data leakage between adsorbates
- Save/load model checkpoints with Type_2 format support
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
from model_3d_cnn import AttentionCNN_2_8



class VoxelDataset(Dataset):
    """Custom PyTorch Dataset for Type_2 voxel grids (28 channels)"""
    
    def __init__(self, X, y, scaler=None, fit_scaler=False):
        """
        Args:
            X: numpy array of voxel grids (N, 20, 20, 20, 28)  # Type_2 format: 28 channels
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
        
        # Initialize data loader with Type_2 format (28 channels)
        self.data_loader = VoxelGridsLoader(
                                            zeolite_types=self.zeolite_types,
                                            adsorbates_by_env=self.adsorbates_by_env,
                                            box_grids_size=self.box_grids_size,
                                            box_increment=self.box_increment,
                                            num_features=28,  # Type_2 format: 28 channels
                                            verbose=self.verbose
                                            )
        
        if self.verbose:
            print(f"\n--- Type_2 CNN3DTrainer initialized")
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
        """Create Type_2 CNN model"""
        # Use feature names extracted from actual data loading
        # These will be set after load_data() is called
        feature_names = getattr(self, 'feature_names', None)
        
        if feature_names is None:
            # Fallback warning - this should not happen if load_data() was called first
            print("⚠️  Warning: Feature names not available yet. Using None for now.")
            print("    This is normal if called before data loading.")
        elif self.verbose:
            print(f"    Using {len(feature_names)} feature names from loaded data: {feature_names}")
        
        # Type_2 CNN model (28 channels)
        model = AttentionCNN_2_8(
            in_channels=28, 
            dropout_rate=dropout_rate,
            feature_names=feature_names
        )
        if self.verbose:
            print(f"     Created Type_2 CNN model (28 channels)")
        
        return model

    def validate_type2_data(self):
        """Validate that loaded data is in Type_2 format (28 channels)"""
        if self.X_data is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")
        
        expected_shape = (20, 20, 20, 28)  # Type_2 format
        actual_shape = self.X_data.shape[1:]
        
        if actual_shape != expected_shape:
            raise ValueError(
                f"Data format mismatch! Expected Type_2 format with shape {expected_shape}, "
                f"but got shape {actual_shape}. Please ensure you're using Type_2 voxel data with 28 channels."
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
        self.X_data = np.stack(all_grids)   # Shape: (N, 20, 20, 20, 28)  N = (Number of adsorbates) * 10 * 24 for Type_2 format
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
            print(f"\n--- Type_2 format data loaded successfully:")
            print(f"    Number of environment-adsorbate combinations: {self.num_env_adsorbates}")
            print(f"    Total datapoints: {self.num_total_datapoints}")
            print(f"    Points per env-adsorbate: {self.num_points_per_env_adsorbate}")
            print(f"    Input data shape: {self.input_data_shape} (Type_2: 28 channels)")
            print(f"    Label range: [{self.y_data.min():.3f}, {self.y_data.max():.3f}]")
        
        # Validate that this is Type_2 format data
        self.validate_type2_data()
    
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
                'input_channels': 28,  # Type_2 format: 28 channels
                'test_mode': self.test_mode,
                'random_state': self.random_state,
                'dropout_rate': 0.25
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
        
        # First, print complete epoch history for easy copying
        print(f"\n{'='*80}")
        print(f"📋 FOLD {fold_idx + 1}/{self.n_folds} EPOCH HISTORY")
        print(f"{'='*80}")
        
        if monitoring_data and 'epoch_history' in monitoring_data:
            epoch_history = monitoring_data['epoch_history']
            if epoch_history:
                print("# Epoch-by-epoch training log:")
                for epoch_info in epoch_history:
                    print(epoch_info['log_line'])
                print(f"# Training completed after {len(epoch_history)} epochs")
            else:
                print("# No epoch history available")
        else:
            print("# No epoch history available in monitoring data")
        
        print(f"{'='*80}")
        
        print(f"\n{'='*60}")
        print(f"🎯 FOLD {fold_idx + 1}/{self.n_folds} COMPLETION ANALYSIS")
        print(f"{'='*60}")
        
        # Performance metrics
        overfitting_ratio = (test_metrics['rmse']**2) / (train_metrics['rmse']**2) if train_metrics['rmse'] > 0 else float('inf')
        generalization_gap = test_metrics['rmse'] - train_metrics['rmse']
        
        print(f"📊 Performance Metrics:")
        print(f"  Train: RMSE={train_metrics['rmse']:.4f}, R²={train_metrics['r2']:.4f}, MAE={train_metrics['mae']:.4f}")
        print(f"  Test:  RMSE={test_metrics['rmse']:.4f}, R²={test_metrics['r2']:.4f}, MAE={test_metrics['mae']:.4f}")
        print(f"  Overfitting ratio: {overfitting_ratio:.2f}")
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
            print(f"  Final overfitting ratio: {final_analysis.get('final_overfitting_ratio', 'N/A'):.2f}")
            print(f"  Max overfitting ratio: {final_analysis.get('max_overfitting_ratio', 'N/A'):.2f}")
            print(f"  Average gradient norm: {final_analysis.get('avg_gradient_norm', 'N/A'):.4f}")
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
                
                if train_improvement > 0.5 and test_improvement < 0.1:
                    print(f"  ⚠️  Strong overfitting pattern detected during training")
                elif train_improvement < 0.05 and test_improvement < 0.05:
                    print(f"  ⚠️  Learning stagnation - both losses stopped improving")
                elif test_improvement > train_improvement * 0.3:
                    print(f"  ✓ Healthy learning pattern - test follows train")
                
                # Find best epoch
                best_test_epoch = np.argmin(test_losses)
                best_test_loss = test_losses[best_test_epoch]
                final_test_loss = test_losses[-1]
                
                print(f"  Best test loss: {best_test_loss:.4f} at epoch {best_test_epoch + 1}")
                print(f"  Performance degradation: {final_test_loss - best_test_loss:.4f}")
        
        # Feature importance summary (if available in monitoring data)
        if monitoring_data and 'feature_importance' in monitoring_data:
            feat_importance = monitoring_data['feature_importance']
            print(f"\n🌟 Key Features (Top 3):")
            # Use dynamically loaded feature names
            feature_names = getattr(self, 'feature_names', None)
            top_indices = np.argsort(feat_importance)[-3:][::-1]
            for i, idx in enumerate(top_indices):
                feat_name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
                print(f"  {i+1}. {feat_name}: {feat_importance[idx]:.4f}")
        
        print(f"{'='*60}\n")
    
    
    def load_model(self, fold_idx):
        """Load model checkpoint if exists"""
        checkpoint_path = self.get_checkpoint_path(fold_idx)
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Create Type_2 model 
            model_config = checkpoint.get('model_config', {})
            
            # Use feature names extracted from actual data loading
            feature_names = getattr(self, 'feature_names', None)
            if feature_names is None and self.verbose:
                print("    ⚠️  Warning: Feature names not available, using None for model reconstruction")
            
            model = AttentionCNN_2_8(
                in_channels=28,
                dropout_rate=model_config.get('dropout_rate', 0.25),
                feature_names=feature_names
            )
            if self.verbose:
                print(f"    � Loaded Type_2 CNN model from checkpoint (28 channels)")
                    
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
        """Train a single model for one fold with improved training strategies and detailed monitoring"""
        if self.verbose:
            print(f"\n--- Training Fold {fold_idx + 1}/{self.n_folds} ---")
        
        # Create label scaler
        scaler = StandardScaler()
        
        # Store scaler info for saving in checkpoint
        scaler.fit(y_train.reshape(-1, 1))
        self._current_scaler_info = {
            'mean': scaler.mean_[0],
            'std': scaler.scale_[0]
        }
        
        # Create datasets
        train_dataset = VoxelDataset(X_train, y_train, scaler=scaler, fit_scaler=True)
        test_dataset = VoxelDataset(X_test, y_test, scaler=scaler, fit_scaler=False)
        
        # Data distribution analysis
        if self.verbose:
            print(f"    Dataset Analysis:")
            print(f"      Train set: {len(X_train)} samples")
            print(f"      Test set:  {len(X_test)} samples")
            print(f"      Label distribution (original scale):")
            print(f"        Train: mean={y_train.mean():.4f}, std={y_train.std():.4f}, range=[{y_train.min():.4f}, {y_train.max():.4f}]")
            print(f"        Test:  mean={y_test.mean():.4f}, std={y_test.std():.4f}, range=[{y_test.min():.4f}, {y_test.max():.4f}]")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Create separate evaluation loaders WITHOUT shuffling
        train_eval_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        test_eval_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Initialize model with STABILITY-FOCUSED dropout to reduce test loss volatility
        # Lower dropout significantly to reduce randomness-induced fluctuations
        if len(X_train) > 1500:
            dropout_rate = 0.15   # Significantly reduced dropout for stability
        elif len(X_train) > 800:
            dropout_rate = 0.12   # Lower dropout to reduce volatility
        else:
            dropout_rate = 0.10   # Minimal dropout for maximum stability
        
        model = self.create_model(dropout_rate=dropout_rate)
        model.to(self.device)
        
        # Model complexity analysis
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        samples_per_param = len(X_train) / trainable_params if trainable_params > 0 else 0
        
        if self.verbose:
            print(f"    Model Complexity Analysis:")
            print(f"      Total parameters: {total_params:,}")
            print(f"      Trainable parameters: {trainable_params:,}")
            print(f"      Samples per parameter: {samples_per_param:.2f}")
            
            # Add detailed model analysis
            print(f"    📊 Detailed Model Architecture Analysis:")
            model.print_model_analysis()
        
        # Loss and optimizer - STABILITY-FOCUSED to reduce test loss volatility
        criterion = nn.MSELoss()
        
        # 🎯 STABILITY regularization - reduced weight decay to minimize training instability
        base_weight_decay = 1e-4  # Reduced from 3e-4 for more stable training
        weight_decay = base_weight_decay
        
        if self.verbose:
            print(f"    Dropout rate: {dropout_rate:.3f} (minimized for test loss stability)")
            print(f"    Weight decay: {weight_decay:.1e} (reduced for stable training)")
        
        # 🎯 DIRECT learning rate - use input LR directly for lr=0.0002 testing
        adjusted_lr = self.learning_rate  # No adjustment, use input LR directly
        
        # Use AdamW with differentiated learning for attention vs conv layers
        attention_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            # Include pooling_weights as attention parameters since they control attention-related pooling
            if ('attention' in name.lower() or 'cbam' in name.lower() or 'fc' in name.lower() or 
                'pooling_weights' in name.lower()):
                attention_params.append(param)  # Include attention, FC layers, and pooling weights
            else:
                other_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': other_params, 'weight_decay': weight_decay},
            {'params': attention_params, 'weight_decay': weight_decay * 0.3, 'lr': adjusted_lr * 0.9}  # Stable attention LR
        ], lr=adjusted_lr, eps=1e-8, betas=(0.9, 0.999))  # More stable betas for reduced volatility
        
        # 🔥 CRITICAL: Stronger gradient clipping to prevent sudden large updates that cause volatility
        max_grad_norm = 0.5  # Reduced from 1.0 for better stability
        
        # 🎯 LR scheduler optimized for stable training - very conservative
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.9,       # Very conservative reduction for stability
            patience=25,      # Much longer patience to avoid early reduction
            min_lr=1e-9,      # Lower minimum for extended fine-tuning
            threshold=1e-4,   # Less sensitive threshold to avoid noise
            cooldown=5        # Longer cooldown for stability
        )
        
        # 🛑 SIMPLIFIED EARLY STOPPING - remove complexity and inconsistency
        best_test_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        # � Simple tracking for monitoring (no complex stability scoring)
        recent_test_losses = []  # Track recent test losses for display only
        
        # 🎯 OPTIMIZED Early stopping parameters for consistent fold training duration
        early_stop_patience = 80  # Much longer patience for more consistent fold training curves
        min_improvement = 5e-5    # More sensitive to tiny gradual improvements
        
        # For test mode, allow minimal patience
        if self.test_mode:
            early_stop_patience = 2
        
        # Extended minimum epochs for more consistent training curves across folds
        min_epochs = 50  # Ensure all folds train for at least 50 epochs for curve consistency
        
        train_losses = []
        test_losses = []
        learning_rates = []
        
        # Add detailed monitoring lists
        overfitting_ratios = []
        gradient_norms = []
        weight_norms_history = []
        activation_stats_history = []
        
        # Add epoch history for fold summary printing
        epoch_history = []
        
        if self.verbose:
            print(f"    🎯 OPTIMIZED Training Strategy (for LR=0.0002 slow convergence):")
            print(f"      Learning rate: {self.learning_rate:.6f}")
            print(f"      Weight decay: {weight_decay:.0e}")
            print(f"      Dropout rate: {dropout_rate:.2f}")
            print(f"      Early stop patience: {early_stop_patience} (extended for consistent fold training curves)")
            print(f"      Min improvement: {min_improvement:.0e} (more sensitive for gradual improvements)")
            print(f"      Min epochs: {min_epochs} (ensures consistent training duration across folds)")
            print(f"      Gradient clipping: {max_grad_norm}")
            
            # Initial model analysis using real training data
            print(f"    🔬 Initial Model Analysis:")
            # Use a small batch of real training data for analysis
            sample_batch_size = min(4, len(train_dataset))
            sample_loader = DataLoader(train_dataset, batch_size=sample_batch_size, shuffle=False)
            sample_batch_X, sample_batch_y = next(iter(sample_loader))
            sample_batch_X = sample_batch_X.to(self.device)
            sample_batch_y = sample_batch_y.to(self.device)
            
            model.eval()
            with torch.no_grad():
                # Test forward pass with real data
                sample_output = model(sample_batch_X, feature_channels=self.feature_channels)
                print(f"      Model forward pass successful: input {sample_batch_X.shape} → output {sample_output.shape}")
                print(f"      Sample input range: [{sample_batch_X.min():.3f}, {sample_batch_X.max():.3f}]")
                print(f"      Sample output range: [{sample_output.min():.3f}, {sample_output.max():.3f}]")
                print(f"      Sample target range: [{sample_batch_y.min():.3f}, {sample_batch_y.max():.3f}]")
                
                # Initial prediction quality check
                sample_predictions = sample_output.squeeze().cpu().numpy()
                sample_targets = sample_batch_y.cpu().numpy()
                initial_mse = np.mean((sample_predictions - sample_targets) ** 2)
                print(f"      Initial sample MSE (normalized): {initial_mse:.4f}")
            
            # Layer-wise parameter count
            print(f"    📋 Layer-wise Parameter Count:")
            total_conv_params = 0
            total_fc_params = 0
            for name, param in model.named_parameters():
                if 'conv' in name.lower() or 'bn' in name.lower():
                    total_conv_params += param.numel()
                elif 'regressor' in name.lower() or 'fc' in name.lower():
                    total_fc_params += param.numel()
            
            print(f"      Convolutional layers: {total_conv_params:,} parameters ({total_conv_params/total_params*100:.1f}%)")
            print(f"      Regressor layers: {total_fc_params:,} parameters ({total_fc_params/total_params*100:.1f}%)")
            
            model.train()  # Switch back to training mode
        
        # Training loop - STABILITY-FOCUSED (no warmup for lr=0.0002 to avoid complexity)
        
        for epoch in range(self.epochs):
            # Training phase with enhanced monitoring
            model.train()
            train_loss = 0.0
            epoch_gradient_norms = []
            
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Unified monitoring strategy - EVERY EPOCH SAME INFORMATION
                monitor_activations = (batch_idx == 0)  # Monitor activations on first batch of each epoch
                
                # Unified monitoring - SAME INFORMATION EVERY EPOCH
                if monitor_activations:
                    # Get comprehensive monitoring data every epoch
                    outputs, activation_stats = model(batch_X, monitor_activations=True, feature_channels=self.feature_channels)
                    
                    # Store monitoring data
                    activation_stats_history.append({
                        'epoch': epoch,
                        'activation_data': activation_stats,
                        'comprehensive_data': activation_stats  # Make sure data is accessible
                    })
                else:
                    # Normal forward pass without monitoring overhead
                    outputs = model(batch_X, feature_channels=self.feature_channels)
                
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                
                # Monitor gradients before clipping
                grad_info = model.get_gradient_norms()
                epoch_gradient_norms.append(grad_info['total_norm'])
                
                # Enhanced gradient clipping based on log analysis (large gradients observed)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Store gradient and weight norms
            avg_grad_norm = np.mean(epoch_gradient_norms) if epoch_gradient_norms else 0
            gradient_norms.append(avg_grad_norm)
            weight_norms = model.get_layer_weight_norms()
            weight_norms_history.append({'epoch': epoch, 'norms': weight_norms})
            
            # Validation phase
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in test_eval_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X, feature_channels=self.feature_channels).squeeze()
                    loss = criterion(outputs, batch_y)
                    test_loss += loss.item()
            
            test_loss /= len(test_eval_loader)
            test_losses.append(test_loss)
            
            # Calculate overfitting metrics
            overfitting_ratio = test_loss / train_loss if train_loss > 0 else float('inf')
            overfitting_ratios.append(overfitting_ratio)
            
            # Store current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            
            # Learning rate scheduling - be more conservative
            old_lr = current_lr
            scheduler.step(test_loss)
            new_lr = optimizer.param_groups[0]['lr']
            
            # Manual logging of learning rate changes since verbose is deprecated
            if old_lr != new_lr and self.verbose:
                print(f"  Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
            
            # 🎯 SIMPLIFIED Early Stopping Logic (remove complex stability scoring)
            
            # Add current test loss to recent losses for monitoring display only
            recent_test_losses.append(test_loss)
            if len(recent_test_losses) > 5:  # Keep last 5 for simple monitoring
                recent_test_losses.pop(0)
            
            # 🔥 CLEAN & SIMPLE: Best model tracking
            improvement_flag = ""
            if test_loss < best_test_loss - min_improvement:  # Simple improvement check
                best_test_loss = test_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                improvement_flag = "✓"
            else:
                patience_counter += 1
                improvement_flag = ""
            
            # 🎯 SIMPLIFIED Early Stopping - single, clear condition
            if epoch >= min_epochs and patience_counter >= early_stop_patience:
                if self.verbose:
                    print(f"\n{'─'*60}")
                    print(f"✅ EARLY STOPPING")
                    print(f"{'─'*60}")
                    print(f"  Stopped at epoch {epoch + 1} (no improvement for {early_stop_patience} epochs)")
                    print(f"  Best test loss: {best_test_loss:.4f}")
                    print(f"  Current test loss: {test_loss:.4f}")
                break
            
            # Overfitting analysis for monitoring
            overfitting_status = ""
            if overfitting_ratio > 2.0:
                overfitting_status = " ⚠️ OVERFIT"
            elif overfitting_ratio > 1.5:
                overfitting_status = " ⚡MILD_OVERFIT"
            elif overfitting_ratio < 1.1:
                overfitting_status = " ✓GOOD_FIT"
            
            # Also stop if learning rate becomes too small (but less aggressive)
            if new_lr < 5e-7:
                if self.verbose:
                    print(f"\n{'─'*80}")
                    print(f"✅ TRAINING TERMINATION - LEARNING RATE TOO SMALL")
                    print(f"{'─'*80}")
                    print(f"  Stopping at epoch {epoch + 1} (learning rate too small: {new_lr:.0e})")
                break
            
            # Unified Enhanced logging with STABILITY information - EVERY EPOCH
            if self.verbose:
                lr_change = f" → {new_lr:.6f}" if old_lr != new_lr else ""
                grad_info = f", Grad={avg_grad_norm:.3f}" if avg_grad_norm > 0 else ""
                
                # Simplified epoch logging (remove stability metrics)
                epoch_log_line = (f"  Epoch {epoch + 1:3d}/{self.epochs}: "
                                f"Train={train_loss:.4f}, Test={test_loss:.4f} {improvement_flag}, "
                                f"Ratio={overfitting_ratio:.2f}{overfitting_status}, "
                                f"LR={old_lr:.6f}{lr_change}{grad_info}, "
                                f"Patience={patience_counter}/{early_stop_patience}")
                
                print(f"\n{epoch_log_line}")
                
                # Store epoch information for fold summary
                epoch_info = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'overfitting_ratio': overfitting_ratio,
                    'overfitting_status': overfitting_status.strip(),
                    'lr': old_lr,
                    'lr_new': new_lr if old_lr != new_lr else None,
                    'grad_norm': avg_grad_norm,
                    'patience': patience_counter,
                    'improvement_flag': improvement_flag,
                    'log_line': epoch_log_line
                }
                epoch_history.append(epoch_info)
                
                # 🎯 UNIFIED MONITORING CALL - All analysis in one place!
                current_monitoring_data = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'overfitting_ratio': overfitting_ratio,
                    'gradient_norms': epoch_gradient_norms,
                    'avg_grad_norm': avg_grad_norm,
                    'current_weight_norms': model.get_layer_weight_norms(),
                    'train_losses': train_losses,
                    'test_losses': test_losses
                }
                
                # Add activation stats if available
                if activation_stats_history and len(activation_stats_history) > 0:
                    latest_entry = activation_stats_history[-1]
                    # Get both comprehensive_data and activation_data
                    latest_stats = latest_entry.get('comprehensive_data') or latest_entry.get('activation_data') or latest_entry.get('basic_data')
                    if latest_stats:
                        # Merge activation stats into monitoring data
                        current_monitoring_data.update(latest_stats)
                
                
                # 📊 UNIFIED monitoring function call - replaces all scattered monitoring
                # Only call once per epoch with complete data
                if self.verbose:
                    self._print_monitoring_summary(current_monitoring_data, epoch, 0, model)
                
                # Learning dynamics
                if len(train_losses) >= 5:
                    recent_train_trend = np.mean(train_losses[-5:]) - np.mean(train_losses[-10:-5]) if len(train_losses) >= 10 else 0
                    recent_test_trend = np.mean(test_losses[-5:]) - np.mean(test_losses[-10:-5]) if len(test_losses) >= 10 else 0
                    print(f"\n    📉 Learning Trends (last 5 epochs):")
                    train_trend_status = "↓ improving" if recent_train_trend < -0.01 else "↑ worsening" if recent_train_trend > 0.01 else "→ stable"
                    test_trend_status = "↓ improving" if recent_test_trend < -0.01 else "↑ worsening" if recent_test_trend > 0.01 else "→ stable"
                    print(f"      Train loss trend: {recent_train_trend:+.4f} ({train_trend_status})")
                    print(f"      Test loss trend:  {recent_test_trend:+.4f} ({test_trend_status})")
                
                print()  # Add spacing for readability
        
        # Check if training completed all epochs (no early stopping)
        if self.verbose and epoch == self.epochs - 1:
            print(f"\n{'─'*80}")
            print(f"✅ TRAINING COMPLETED - ALL EPOCHS FINISHED")
            print(f"{'─'*80}")
            print(f"  Completed all {self.epochs} epochs without early stopping")
        
        # Training completion analysis
        final_overfitting_ratio = overfitting_ratios[-1] if overfitting_ratios else 0
        max_overfitting_ratio = max(overfitting_ratios) if overfitting_ratios else 0
        avg_gradient_norm = np.mean(gradient_norms) if gradient_norms else 0
        
        if self.verbose:
            print(f"    🔍 Final Training Analysis:")
            print(f"      Final overfitting ratio: {final_overfitting_ratio:.2f}")
            print(f"      Max overfitting ratio: {max_overfitting_ratio:.2f}")
            print(f"      Average gradient norm: {avg_gradient_norm:.4f}")
            print(f"      Epochs completed: {epoch + 1}/{self.epochs}")
            
            # Final attention analysis
            final_attention_summary = model.get_attention_summary()
            if final_attention_summary != "No attention data available":
                print(f"    🔍 Final Attention Analysis:")
                for layer_name, attention_info in final_attention_summary.items():
                    print(f"      {layer_name}:")
                    if isinstance(attention_info, dict):
                        # 显示注意力权重统计信息
                        if 'mean' in attention_info:
                            print(f"        📊 Attention weights: μ={attention_info['mean']:.4f}, σ={attention_info['std']:.4f}")
                            print(f"           Range: [{attention_info['min']:.4f}, {attention_info['max']:.4f}]")
                        if 'channels' in attention_info:
                            print(f"           Channels: {attention_info['channels']}")
                        if 'group_interaction_enabled' in attention_info:
                            print(f"           Group interaction: {'enabled' if attention_info['group_interaction_enabled'] else 'disabled'}")
                            if attention_info.get('group_split'):
                                print(f"           Group split: {attention_info['group_split']}")
                    else:
                        # 备用显示格式
                        print(f"        📊 Data: {attention_info}")
            else:
                print(f"    🔍 Final Attention Analysis: No attention data available")
            
            # Loss convergence analysis
            print(f"    📈 Loss Convergence Analysis:")
            if len(train_losses) > 10:
                early_train_avg = np.mean(train_losses[:5])
                late_train_avg = np.mean(train_losses[-5:])
                early_test_avg = np.mean(test_losses[:5])
                late_test_avg = np.mean(test_losses[-5:])
                
                train_improvement = early_train_avg - late_train_avg
                test_improvement = early_test_avg - late_test_avg
                
                print(f"      Train loss improvement: {train_improvement:.4f} ({early_train_avg:.4f} → {late_train_avg:.4f})")
                print(f"      Test loss improvement: {test_improvement:.4f} ({early_test_avg:.4f} → {late_test_avg:.4f})")
                
                if train_improvement > 0.5 and test_improvement < 0.1:
                    print(f"      ⚠️  Strong overfitting detected: train improved significantly but test didn't")
                elif train_improvement < 0.1 and test_improvement < 0.1:
                    print(f"      ⚠️  Learning stagnation: both train and test stopped improving")
                elif test_improvement > train_improvement * 0.5:
                    print(f"      ✓ Good learning: test improvement follows train improvement")
            
            # Weight analysis with improved categorization for dual-branch architecture
            final_weight_norms = model.get_layer_weight_norms()
            print(f"    ⚖️  Final Weight Analysis:")
            
            # Categorize weights by new architecture components
            dual_branch_weights = [norm for name, norm in final_weight_norms.items() if any(keyword in name for keyword in ['adsorbate_branch', 'solvent_branch', 'interaction'])]
            cnn_weights = [norm for name, norm in final_weight_norms.items() if 'cnn_backbone' in name or ('layer' in name and not any(keyword in name for keyword in ['adsorbate', 'solvent', 'interaction']))]
            regressor_weights = [norm for name, norm in final_weight_norms.items() if 'regressor' in name]
            
            if dual_branch_weights:
                print(f"      Dual-branch weights: μ={np.mean(dual_branch_weights):.3f}, σ={np.std(dual_branch_weights):.3f}, range=[{min(dual_branch_weights):.3f}, {max(dual_branch_weights):.3f}]")
            if cnn_weights:
                print(f"      CNN backbone weights: μ={np.mean(cnn_weights):.3f}, σ={np.std(cnn_weights):.3f}, range=[{min(cnn_weights):.3f}, {max(cnn_weights):.3f}]")
            if regressor_weights:
                print(f"      Regressor weights: μ={np.mean(regressor_weights):.3f}, σ={np.std(regressor_weights):.3f}, range=[{min(regressor_weights):.3f}, {max(regressor_weights):.3f}]")
            
            # Recommendations based on analysis
            print(f"    💡 Optimization Recommendations:")
            if final_overfitting_ratio > 2.0:
                print(f"      🔧 Overfitting mitigation:")
                print(f"        • Increase dropout rate (current: {dropout_rate:.2f} → {min(0.5, dropout_rate + 0.1):.2f})")
                print(f"        • Increase weight decay (current: {weight_decay:.0e} → {weight_decay * 10:.0e})")
                print(f"        • Consider reducing model complexity")
                print(f"        • Add more data augmentation")
            elif final_overfitting_ratio < 1.2:
                print(f"      🚀 Underfitting mitigation:")
                print(f"        • Reduce dropout rate (current: {dropout_rate:.2f} → {max(0.1, dropout_rate - 0.1):.2f})")
                print(f"        • Increase model capacity (add more channels/layers)")
                print(f"        • Train for more epochs")
                print(f"        • Increase learning rate")
            else:
                print(f"      ✓ Good balance between overfitting and underfitting")
            
            if avg_gradient_norm < 0.01:
                print(f"      🔄 Gradient flow issues:")
                print(f"        • Consider increasing learning rate")
                print(f"        • Check for vanishing gradients")
                print(f"        • Consider residual connections or gradient clipping adjustments")
            elif avg_gradient_norm > 1.0:
                print(f"      ⚡ Large gradient issues:")
                print(f"        • Reduce learning rate") 
                print(f"        • Increase gradient clipping (current max_norm=0.5)")
                print(f"        • Consider batch normalization tuning")
        
        # Use model's built-in comprehensive training insights
        if hasattr(model, 'print_training_insights'):
            # Create comprehensive monitoring data
            final_monitoring_data = {
                'feature_analysis': {
                    'learned_importance': model.get_learned_feature_importance() if hasattr(model, 'get_learned_feature_importance') else {}
                },
                'training_metrics': {
                    'final_overfitting_ratio': final_overfitting_ratio,
                    'max_overfitting_ratio': max_overfitting_ratio,
                    'avg_gradient_norm': avg_gradient_norm,
                    'total_epochs': epoch + 1
                }
            }
            # Add final activation stats if available
            if activation_stats_history:
                latest_entry = activation_stats_history[-1]
                latest_stats = latest_entry.get('comprehensive_data') or latest_entry.get('basic_data')
                if latest_stats:
                    final_monitoring_data.update(latest_stats)
            
            # Use compatibility check for different model interfaces
            try:
                # Call with monitoring data - model method now supports it
                model.print_training_insights(monitoring_data=final_monitoring_data, epoch=(epoch + 1))
                    
            except Exception as e:
                print(f"    ⚠️ Training insights generation failed: {e}")
        
        # Load best model state - SIMPLIFIED approach
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            if self.verbose:
                print(f"    ✅ Loaded best model (test loss: {best_test_loss:.4f})")
        else:
            if self.verbose:
                print(f"    ⚠️ Warning: No improvement found, using final model state")
        
        # Final evaluation using UNSHUFFLED loaders to maintain order
        model.eval()
        train_predictions = []
        train_targets = []
        test_predictions = []
        test_targets = []
        
        with torch.no_grad():
            # Train set evaluation with UNSHUFFLED data
            for batch_X, batch_y in train_eval_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = model(batch_X, feature_channels=self.feature_channels).squeeze()
                train_predictions.extend(outputs.cpu().numpy())
                train_targets.extend(batch_y.cpu().numpy())
            
            # Test set evaluation with UNSHUFFLED data
            for batch_X, batch_y in test_eval_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = model(batch_X, feature_channels=self.feature_channels).squeeze()
                test_predictions.extend(outputs.cpu().numpy())
                test_targets.extend(batch_y.cpu().numpy())
        
        # Convert back to original scale
        train_predictions = scaler.inverse_transform(np.array(train_predictions).reshape(-1, 1)).flatten()
        train_targets = scaler.inverse_transform(np.array(train_targets).reshape(-1, 1)).flatten()
        test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1)).flatten()
        test_targets = scaler.inverse_transform(np.array(test_targets).reshape(-1, 1)).flatten()
        
        # Calculate metrics
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
        
        # Get metadata for train and test indices (passed from run_training)
        train_metadata = [self.metadata[i] for i in self.current_train_idx]
        test_metadata = [self.metadata[i] for i in self.current_test_idx]
        
        # Create result dataframes with metadata - now the order matches because we used unshuffled evaluation loaders
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
            
            # Prediction quality analysis
            print(f"    🎯 Prediction Quality Analysis:")
            
            # Residual analysis for test set
            test_residuals = test_targets - test_predictions
            print(f"      Test residuals: μ={np.mean(test_residuals):.4f}, σ={np.std(test_residuals):.4f}")
            print(f"      Test range: predictions=[{test_predictions.min():.3f}, {test_predictions.max():.3f}], targets=[{test_targets.min():.3f}, {test_targets.max():.3f}]")
            
            # Find worst predictions
            worst_indices = np.argsort(np.abs(test_residuals))[-3:]  # Top 3 worst predictions
            print(f"      Worst predictions:")
            for i, idx in enumerate(worst_indices):
                error = test_residuals[idx]
                print(f"        {i+1}. True={test_targets[idx]:.4f}, Pred={test_predictions[idx]:.4f}, Error={error:.4f}")
            
            # Best predictions
            best_indices = np.argsort(np.abs(test_residuals))[:3]  # Top 3 best predictions
            print(f"      Best predictions:")
            for i, idx in enumerate(best_indices):
                error = test_residuals[idx]
                print(f"        {i+1}. True={test_targets[idx]:.4f}, Pred={test_predictions[idx]:.4f}, Error={error:.4f}")
            
            # Prediction bias analysis
            low_energy_mask = test_targets < np.percentile(test_targets, 33)
            mid_energy_mask = (test_targets >= np.percentile(test_targets, 33)) & (test_targets <= np.percentile(test_targets, 67))
            high_energy_mask = test_targets > np.percentile(test_targets, 67)
            
            if np.sum(low_energy_mask) > 0:
                low_rmse = np.sqrt(np.mean(test_residuals[low_energy_mask]**2))
                print(f"      Low energy RMSE: {low_rmse:.4f} (n={np.sum(low_energy_mask)})")
            if np.sum(mid_energy_mask) > 0:
                mid_rmse = np.sqrt(np.mean(test_residuals[mid_energy_mask]**2))
                print(f"      Mid energy RMSE: {mid_rmse:.4f} (n={np.sum(mid_energy_mask)})")
            if np.sum(high_energy_mask) > 0:
                high_rmse = np.sqrt(np.mean(test_residuals[high_energy_mask]**2))
                print(f"      High energy RMSE: {high_rmse:.4f} (n={np.sum(high_energy_mask)})")
        
        # 🔬 ENHANCED ARCHITECTURE ANALYSIS
        if self.verbose and hasattr(model, 'analyze_layer_importance_scores'):
            print(f"\n🔬 ARCHITECTURE PERFORMANCE ANALYSIS:")
            print("=" * 60)
            
            try:
                # Analyze layer importance with a sample batch
                sample_batch = X_test[:min(4, len(X_test))]  # Use small batch for analysis
                sample_tensor = torch.FloatTensor(sample_batch).permute(0, 4, 1, 2, 3).to(self.device)
                
                try:
                    layer_importance = model.analyze_layer_importance_scores(sample_tensor)
                    if layer_importance:
                        print(f"\n📊 Layer Importance Ranking:")
                        sorted_layers = sorted(layer_importance.items(), 
                                             key=lambda x: x[1]['gradient_importance'], reverse=True)
                        for layer_name, metrics in sorted_layers:
                            print(f"    {layer_name}: Importance={metrics['gradient_importance']:.6f}, "
                                  f"Efficiency={metrics['parameter_efficiency']:.6f}")
                    else:
                        print("    ⚠️ Could not analyze layer importance - no data returned")
                except Exception as e:
                    print(f"    ⚠️ Layer importance analysis failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Regressor bottleneck analysis - need to get proper inputs
                if hasattr(model, 'analyze_regressor_bottlenecks'):
                    
                    model.eval()
                    with torch.no_grad():
                        try:
                            # Use model's own forward pass to avoid parameter mismatch
                            full_output = model(sample_tensor)
                            
                            # Get the regressor input by manual forward pass
                            sample_input = sample_tensor
                            
                            # Process through model components step by step
                            adsorbate_features = sample_input[:, :14, :, :, :]
                            solvent_features = sample_input[:, 14:, :, :, :]
                            
                            # Let the model process this properly
                            adsorbate_out = model.adsorbate_processor(adsorbate_features)
                            solvent_out = model.solvent_processor(solvent_features)
                            
                            # Combine features and process through attention
                            combined_features = torch.cat([adsorbate_out, solvent_out], dim=1)
                            attended_features = model.interaction_attention(combined_features)
                            
                            # Process through interaction conv (important for dimension reduction)
                            interaction_features = model.interaction_conv(attended_features)
                            
                            # Process through CNN backbone
                            cnn_output = interaction_features
                            for layer_name in ['layer1', 'layer2', 'layer3']:
                                if hasattr(model, layer_name):
                                    layer = getattr(model, layer_name)
                                    cnn_output = layer(cnn_output)
                            
                            # Final pooling and flattening
                            pooled_output = model.adaptive_pool(cnn_output)
                            x_before_regressor = pooled_output.view(pooled_output.size(0), -1)
                            predictions = model.regressor(x_before_regressor)
                            
                            # Run regressor analysis
                            regressor_analysis = model.analyze_regressor_bottlenecks(x_before_regressor, predictions)
                            
                            if regressor_analysis:
                                print(f"\n🎯 Regressor Analysis:")
                                if 'regressor_input_analysis' in regressor_analysis:
                                    input_analysis = regressor_analysis['regressor_input_analysis']
                                    print(f"    Feature Utilization: {input_analysis['feature_utilization_ratio']:.3f}")
                                    print(f"    Unused Features: {input_analysis['unused_feature_percentage']:.1f}%")
                                    print(f"    Effective Features: {input_analysis['effective_features']}/{input_analysis['total_features']}")
                                
                                if 'regressor_capacity_analysis' in regressor_analysis:
                                    capacity_analysis = regressor_analysis['regressor_capacity_analysis']
                                    print(f"    Capacity Status: {capacity_analysis['capacity_utilization']}")
                                    print(f"    Samples per Parameter: {capacity_analysis['samples_per_parameter']:.2f}")
                                
                                if 'recommendations' in regressor_analysis:
                                    print(f"    Recommendations:")
                                    for rec in regressor_analysis['recommendations']:
                                        print(f"      {rec}")
                            else:
                                print("    ⚠️ Could not analyze regressor bottlenecks - no data returned")
                                
                        except Exception as e:
                            print(f"    ⚠️ Regressor bottleneck analysis failed: {e}")
                            import traceback
                            traceback.print_exc()
                
                # Architecture efficiency comparison
                if hasattr(model, 'compare_architecture_efficiency'):
                    efficiency_comparison = model.compare_architecture_efficiency()
                    if efficiency_comparison:
                        print(f"\n⚡ Architecture Efficiency:")
                        if 'current_efficiency' in efficiency_comparison:
                            current_eff = efficiency_comparison['current_efficiency']
                            if 'parameter_efficiency_score' in current_eff:
                                print(f"    Parameter Efficiency Score: {current_eff['parameter_efficiency_score']:.4f}")
                            if 'cnn_regressor_balance' in current_eff:
                                print(f"    CNN/Regressor Balance: {current_eff['cnn_regressor_balance']}")
                
                # Architecture optimization recommendations
                if hasattr(model, 'get_architecture_optimization_recommendations'):
                    current_performance = {
                        'test_rmse': test_metrics['rmse'],
                        'train_rmse': train_metrics['rmse'],
                        'overfitting_ratio': (test_metrics['rmse'] / train_metrics['rmse']) ** 2 if train_metrics['rmse'] > 0 else 1.0
                    }
                    
                    recommendations = model.get_architecture_optimization_recommendations(current_performance)
                    
                    if recommendations.get('immediate_actions'):
                        print(f"\n⚡ Immediate Actions:")
                        for action in recommendations['immediate_actions']:
                            print(f"    {action}")
                    
                    if recommendations.get('experimental_changes'):
                        print(f"\n🧪 Experimental Changes:")
                        for change in recommendations['experimental_changes']:
                            print(f"    {change}")
                
                # Record this architecture's performance
                if hasattr(model, 'record_architecture_performance'):
                    performance_record = model.record_architecture_performance(
                        train_metrics, test_metrics, final_monitoring_data
                    )
                    print(f"\n📝 Architecture Performance Recorded:")
                    print(f"    Total Parameters: {performance_record['architecture']['total_params']:,}")
                    print(f"    CNN/Regressor Ratio: {performance_record['architecture']['cnn_regressor_ratio']:.2f}")
                
                print("=" * 60)
                    
            except Exception as e:
                print(f"  ⚠️ Architecture analysis failed: {e}")
                import traceback
                traceback.print_exc()
        
        
        # Store monitoring data in model storage
        monitoring_data = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'learning_rates': learning_rates,
            'overfitting_ratios': overfitting_ratios,
            'gradient_norms': gradient_norms,
            'weight_norms_history': weight_norms_history,
            'activation_stats_history': activation_stats_history,
            'epoch_history': epoch_history,  # Add epoch history for summary printing
            'final_analysis': {
                'final_overfitting_ratio': final_overfitting_ratio,
                'max_overfitting_ratio': max_overfitting_ratio,
                'avg_gradient_norm': avg_gradient_norm,
                'samples_per_param': samples_per_param,
                'total_epochs': epoch + 1
            }
        }
        
        # Add CBAM group interaction monitoring data
        try:
            # Collect CBAM group interaction statistics from all CBAM layers
            cbam_group_summary = {}
            cbam_layer_count = 0
            total_ads_attention = 0
            total_solv_attention = 0
            total_interaction_strength = 0
            
            for name, module in model.named_modules():
                if hasattr(module, 'channel_attention') and hasattr(module.channel_attention, 'group_interaction_stats'):
                    stats = module.channel_attention.group_interaction_stats
                    if stats and any(v != 0.0 for v in stats.values()):
                        cbam_layer_count += 1
                        total_ads_attention += stats.get('ads_attention_mean', 0)
                        total_solv_attention += stats.get('solv_attention_mean', 0)
                        total_interaction_strength += stats.get('interaction_strength_value', 0)
            
            if cbam_layer_count > 0:
                cbam_group_summary = {
                    'cbam_layers_with_interaction': cbam_layer_count,
                    'average_ads_attention': total_ads_attention / cbam_layer_count,
                    'average_solv_attention': total_solv_attention / cbam_layer_count,
                    'average_interaction_strength': total_interaction_strength / cbam_layer_count,
                    'solvent_emphasis_ratio': (total_solv_attention / cbam_layer_count) / (total_ads_attention / cbam_layer_count) if total_ads_attention > 0 else 0
                }
                monitoring_data['cbam_group_interaction_summary'] = cbam_group_summary
                
                if self.verbose:
                    print(f"    🧬 CBAM Group Interaction Summary:")
                    print(f"      • Active CBAM layers: {cbam_group_summary['cbam_layers_with_interaction']}")
                    print(f"      • Avg adsorbate attention: {cbam_group_summary['average_ads_attention']:.4f}")
                    print(f"      • Avg solvent attention: {cbam_group_summary['average_solv_attention']:.4f}")
                    print(f"      • Avg interaction strength: {cbam_group_summary['average_interaction_strength']:.4f}")
                    print(f"      • Solvent emphasis ratio: {cbam_group_summary['solvent_emphasis_ratio']:.3f}")
            
        except Exception as e:
            if self.verbose:
                print(f"    ⚠️  Warning: Could not collect CBAM group interaction data: {e}")
        
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
        # Note: self.n_folds already set in __init__ based on split_type
        
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
    
    def _format_bn_layer_name(self, layer_name):
        """Format BatchNorm layer name for display"""
        if '.' in layer_name:
            parts = layer_name.split('.')
            if 'adsorbate_processor' in layer_name:
                return f"ads_branch.{parts[-1]}"
            elif 'solvent_processor' in layer_name:
                return f"solv_branch.{parts[-1]}"
            elif 'layer' in layer_name and 'cnn' not in layer_name:
                return f"cnn_{parts[-2]}.{parts[-1]}"
            elif 'interaction' in layer_name:
                return f"interact.{parts[-1]}"
            else:
                return f"{parts[-2]}.{parts[-1]}" if len(parts) >= 2 else parts[-1]
        else:
            return layer_name
    
    def _print_monitoring_summary(self, monitoring_data, epoch, batch_idx, model=None):
        """🎯 UNIFIED MONITORING - All analysis in one place, every epoch!"""
        if not monitoring_data or not isinstance(monitoring_data, dict):
            return
        
        print(f"\n    🔬 [Epoch {epoch+1}] UNIFIED Comprehensive Analysis:")
        
        # ========== 1. WEIGHT NORMS ANALYSIS ==========
        print(f"\n    🔧 Weight Norms:")
        if 'current_weight_norms' in monitoring_data:
            current_weight_norms = monitoring_data['current_weight_norms']
            
            # Group weights by component for better organization
            component_groups = {
                'Dual Branch': [],
                'CNN Backbone': [],
                'Regressor': [],
                'Other': []
            }
            
            # Categorize weights by component
            for layer_name, norm in current_weight_norms.items():
                if any(keyword in layer_name for keyword in ['adsorbate_branch', 'solvent_branch', 'interaction']):
                    component_groups['Dual Branch'].append((layer_name, norm))
                elif any(keyword in layer_name for keyword in ['cnn_backbone', 'layer']):
                    component_groups['CNN Backbone'].append((layer_name, norm))
                elif 'regressor' in layer_name:
                    component_groups['Regressor'].append((layer_name, norm))
                else:
                    component_groups['Other'].append((layer_name, norm))
            
            # Display organized by component (show max 8 most important)
            total_shown = 0
            for group_name, group_weights in component_groups.items():
                if group_weights and total_shown < 8:
                    for layer_name, norm in group_weights[:min(3, 8-total_shown)]:  # Max 3 per group
                        print(f"      {layer_name:<25}: {norm:.4f}")
                        total_shown += 1
                        if total_shown >= 8:
                            break
        else:
            print(f"      No weight norm data available")
        
        # ========== 2. GRADIENT ANALYSIS ==========
        print(f"\n    📈 Gradient Analysis:")
        if 'gradient_norms' in monitoring_data and monitoring_data['gradient_norms']:
            gradient_norms = monitoring_data['gradient_norms']
            avg_grad_norm = monitoring_data.get('avg_grad_norm', 0)
            print(f"      Average gradient norm: {avg_grad_norm:.4f}")
            
            # gradient_norms is a dict, so get values for min/max
            if isinstance(gradient_norms, dict):
                grad_values = list(gradient_norms.values())
                print(f"      Max gradient norm: {max(grad_values):.4f}")
                print(f"      Min gradient norm: {min(grad_values):.4f}")
                print(f"      Number of parameters with gradients: {len(grad_values)}")
            else:
                print(f"      Max gradient norm: {max(gradient_norms):.4f}")
                print(f"      Min gradient norm: {min(gradient_norms):.4f}")
                
            if avg_grad_norm < 0.001:
                print(f"      ⚠️  Very small gradients - possible vanishing gradient problem")
            elif avg_grad_norm > 2.0:
                print(f"      ⚠️  Large gradients - consider stronger gradient clipping")
        else:
            print(f"      No gradient data available")
        
        # ========== 3. ACTIVATION ANALYSIS ==========
        print(f"\n    🎯 Activation Analysis:")
        activation_analysis_printed = False
        
        # Dynamic activation keys detection based on actual model architecture
        # This automatically adapts to model changes (layer3 removal/addition)
        def get_dynamic_activation_keys(model):
            """Dynamically detect available layers in the model"""
            base_keys = ['adsorbate_branch', 'solvent_branch', 'combined_features', 
                        'attended_features', 'interaction_features']
            
            # Dynamically check for CNN layers
            cnn_layers = []
            for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
                if hasattr(model, layer_name):
                    cnn_layers.append(layer_name)
            
            # Add remaining components
            end_keys = ['adaptive_pool', 'output']
            
            return base_keys + cnn_layers + end_keys
        
        activation_keys = get_dynamic_activation_keys(model)
        
        # Look for activation data in the correct format
        for key in activation_keys:
            # Check for direct activation stats (new format)
            if f'{key}_mean' in monitoring_data:
                mean_val = monitoring_data.get(f'{key}_mean', 0)
                std_val = monitoring_data.get(f'{key}_std', 0)
                zeros_pct = monitoring_data.get(f'{key}_zeros_pct', 0)
                
                print(f"      {key:<18}: μ={mean_val:.3f}, σ={std_val:.3f}, zeros={zeros_pct:.1f}%")
                activation_analysis_printed = True
                
                # Show abs statistics for important layers
                if f'{key}_abs_mean' in monitoring_data and key in ['adsorbate_branch', 'solvent_branch', 'output']:
                    abs_mean = monitoring_data.get(f'{key}_abs_mean', 0)
                    abs_std = monitoring_data.get(f'{key}_abs_std', 0)
                    print(f"      {key}_abs     : μ={abs_mean:.3f}, σ={abs_std:.3f}, zeros=0.0%")
        
        # Show any additional activation stats not in the expected list
        additional_keys = []
        for key in monitoring_data.keys():
            if key.endswith('_mean') and not any(expected in key for expected in activation_keys):
                base_key = key.replace('_mean', '')
                if base_key not in additional_keys:
                    additional_keys.append(base_key)
        
        for key in additional_keys:
            if f'{key}_mean' in monitoring_data:
                mean_val = monitoring_data.get(f'{key}_mean', 0)
                std_val = monitoring_data.get(f'{key}_std', 0)
                zeros_pct = monitoring_data.get(f'{key}_zeros_pct', 0)
                print(f"      {key:<18}: μ={mean_val:.3f}, σ={std_val:.3f}, zeros={zeros_pct:.1f}%")
                activation_analysis_printed = True
        
        if not activation_analysis_printed:
            print(f"      No activation analysis data available")
        
        # ========== 4. FEATURE ANALYSIS ==========
        print(f"\n    🧪 Feature Analysis:")
        if model is not None:
            try:
                # 🎯 NEW: Dual-branch specific feature analysis
                if hasattr(model, 'adsorbate_processor') and hasattr(model, 'solvent_processor'):
                    print(f"      🔀 Dual-Branch Feature Analysis:")
                    
                    # Analyze processor channel importance via activation magnitudes
                    if hasattr(model, 'branch_statistics') and model.branch_statistics:
                        stats = model.branch_statistics
                        ads_mag = stats.get('adsorbate_magnitude', 0)
                        sol_mag = stats.get('solvent_magnitude', 0)
                        total_mag = ads_mag + sol_mag
                        
                        if total_mag > 0:
                            ads_importance = ads_mag / total_mag
                            sol_importance = sol_mag / total_mag
                            
                            print(f"        Adsorbate branch importance: {ads_importance:.3f} ({ads_importance*100:.1f}%)")
                            print(f"        Solvent branch importance: {sol_importance:.3f} ({sol_importance*100:.1f}%)")
                            
                            # 🔥 NEW: Enhanced adsorbate importance monitoring with optimization tracking
                            ads_pct = ads_importance * 100
                            if ads_pct < 20.0:
                                print(f"        └─ ⚠️ LOW ADSORBATE IMPORTANCE: {ads_pct:.1f}% < 20% target")
                                print(f"        └─ 💡 Optimization Status: Enhanced processing should improve this")
                            elif ads_pct >= 25.0 and ads_pct <= 35.0:
                                print(f"        └─ ✅ TARGET ACHIEVED: {ads_pct:.1f}% in optimal range (25-35%)")
                                print(f"        └─ 🎯 Optimization Status: Enhanced adsorbate processing working!")
                            elif ads_pct >= 20.0 and ads_pct < 25.0:
                                print(f"        └─ 🔄 IMPROVING: {ads_pct:.1f}% approaching 25-35% target")
                            else:
                                print(f"        └─ ⚠️ TOO HIGH: {ads_pct:.1f}% > 35% may indicate overfitting")
                            
                            # Feature group analysis
                            print(f"        Feature group analysis:")
                            print(f"          Channels 0-13 (adsorbate): {ads_importance:.3f} relative importance")
                            print(f"          Channels 14-27 (solvent): {sol_importance:.3f} relative importance")
                            
                            # Determine dominant feature type
                            if sol_importance > ads_importance * 1.5:
                                print(f"        🧪 Solvent-dominated feature learning (expected for interaction energy)")
                            elif ads_importance > sol_importance * 1.5:
                                print(f"        ⚠️ Adsorbate-dominated feature learning (unexpected)")
                            else:
                                print(f"        ⚖️ Balanced feature learning")
                        else:
                            print(f"        No activation data available for feature analysis")
                    else:
                        print(f"        Branch statistics not available")
                    
                    # CBAM attention-based feature importance
                    cbam_found = False
                    for name, module in model.named_modules():
                        if hasattr(module, 'channel_attention') and hasattr(module.channel_attention, 'group_interaction_stats'):
                            stats = module.channel_attention.group_interaction_stats
                            if stats and any(v != 0.0 for v in stats.values()):
                                if not cbam_found:
                                    print(f"        📊 CBAM Attention-based Feature Importance:")
                                    cbam_found = True
                                
                                layer_display = name.split('.')[-2] if '.' in name else name
                                ads_attn = stats.get('ads_attention_mean', 0)
                                solv_attn = stats.get('solv_attention_mean', 0)
                                
                                if ads_attn + solv_attn > 0:
                                    ads_ratio = ads_attn / (ads_attn + solv_attn)
                                    solv_ratio = solv_attn / (ads_attn + solv_attn)
                                    print(f"          {layer_display}: ads={ads_ratio:.3f}, solv={solv_ratio:.3f}")
                    
                    if not cbam_found:
                        print(f"        No CBAM attention data available")
                
                # Alternative: learned feature importance method
                if hasattr(model, 'get_learned_feature_importance'):
                    print(f"      🧠 Learned Feature Importance Analysis:")
                    try:
                        importance = model.get_learned_feature_importance()
                        if importance:
                            print(f"        Available features: {len(importance)}")
                            
                            # Group by adsorbate vs solvent features
                            ads_features = {k: v for k, v in importance.items() if 'adsorbate' in k.lower()}
                            solv_features = {k: v for k, v in importance.items() if 'solvent' in k.lower()}
                            
                            if ads_features and solv_features:
                                # Calculate group importance
                                ads_avg_weight = np.mean([v['normalized_weight'] for v in ads_features.values()])
                                solv_avg_weight = np.mean([v['normalized_weight'] for v in solv_features.values()])
                                
                                print(f"        Adsorbate features avg importance: {ads_avg_weight:.4f}")
                                print(f"        Solvent features avg importance: {solv_avg_weight:.4f}")
                                
                                # Top features overall
                                top_3 = sorted(importance.items(), key=lambda x: x[1].get('normalized_weight', 0), reverse=True)[:3]
                                print(f"        Top 3 features:")
                                for i, (name, data) in enumerate(top_3, 1):
                                    print(f"          {i}. {name}: {data.get('normalized_weight', 0):.4f}")
                            else:
                                # Fallback to top features
                                top_3 = sorted(importance.items(), key=lambda x: x[1].get('normalized_weight', 0), reverse=True)[:3]
                                print(f"        Top 3 features:")
                                for i, (name, data) in enumerate(top_3, 1):
                                    print(f"          {i}. {name}: {data.get('normalized_weight', 0):.4f}")
                        else:
                            print(f"        No learned feature importance data")
                    except Exception as imp_e:
                        print(f"        Feature importance error: {str(imp_e)[:40]}...")
                
                else:
                    print(f"      ❌ No feature analysis methods available")
                    print(f"        Model type: {type(model).__name__}")
                    print(f"        Available attributes: {[attr for attr in dir(model) if not attr.startswith('_')][:5]}...")
                    
            except Exception as e:
                print(f"      Feature analysis error: {str(e)[:50]}...")
        else:
            print(f"      No model available for feature analysis")
        
        # ========== 🎯 5. DUAL-BRANCH ANALYSIS (NEW) ==========
        print(f"\n    🏗️ Dual-Branch Architecture Analysis:")
        if model is not None and hasattr(model, 'get_branch_analysis'):
            try:
                branch_analysis = model.get_branch_analysis()
                if branch_analysis and 'current_statistics' in branch_analysis:
                    stats = branch_analysis['current_statistics']
                    
                    # Branch magnitudes comparison
                    ads_mag = stats.get('adsorbate_magnitude', 0)
                    sol_mag = stats.get('solvent_magnitude', 0)
                    mag_ratio = stats.get('magnitude_ratio', 0)
                    
                    print(f"      Adsorbate magnitude: {ads_mag:.4f}")
                    print(f"      Solvent magnitude: {sol_mag:.4f}")
                    print(f"      Magnitude ratio (ads/sol): {mag_ratio:.3f}")
                    
                    # Sparsity analysis
                    ads_sparsity = stats.get('adsorbate_sparsity', 0)
                    sol_sparsity = stats.get('solvent_sparsity', 0)
                    sparsity_ratio = stats.get('sparsity_ratio', 0)
                    
                    print(f"      Adsorbate sparsity: {ads_sparsity:.3f}")
                    print(f"      Solvent sparsity: {sol_sparsity:.3f}")
                    print(f"      Sparsity ratio: {sparsity_ratio:.2f}")
                    
                    # Architecture insights
                    if 'architecture_insights' in branch_analysis:
                        insights = branch_analysis['architecture_insights']
                        param_ratio = insights.get('parameter_ratio', 0)
                        print(f"      Parameter ratio (sol/ads): {param_ratio:.2f}")
                        
                        if 'architecture_design' in insights:
                            design = insights['architecture_design']
                            print(f"      Design emphasis: {design.get('natural_emphasis', 'unknown')} ({design.get('design_description', 'unknown')})")
                    
                    # Balance assessment
                    balance = stats.get('branch_balance', 'unknown')
                    if balance == 'solvent_dominant':
                        print(f"      ✓ Branch balance: {balance} (as expected)")
                    elif balance == 'adsorbate_dominant':
                        print(f"      ⚠️ Branch balance: {balance} (unexpected)")
                    else:
                        print(f"      ⚖️ Branch balance: {balance}")
                    
                    # Recommendations
                    if 'recommendations' in branch_analysis:
                        recommendations = branch_analysis['recommendations']
                        if recommendations and len(recommendations) > 1:  # More than just "well-balanced"
                            print(f"      💡 Recommendations:")
                            for rec in recommendations[:2]:  # Show top 2
                                print(f"        • {rec}")
                else:
                    print(f"      No branch analysis data available")
            except Exception as e:
                print(f"      Branch analysis error: {str(e)[:50]}...")
        else:
            print(f"      Model does not support branch analysis")
        
        # ========== 🎯 6. CBAM GROUP INTERACTION ANALYSIS (NEW) ==========
        print(f"\n    🧬 CBAM Group Interaction Analysis:")
        if model is not None:
            try:
                # Check all CBAM layers for group interaction statistics
                cbam_group_stats = []
                
                # Look for CBAM modules in the model
                for name, module in model.named_modules():
                    if hasattr(module, 'channel_attention') and hasattr(module.channel_attention, 'group_interaction_stats'):
                        stats = module.channel_attention.group_interaction_stats
                        if stats and any(v != 0.0 for v in stats.values()):
                            cbam_group_stats.append({
                                'layer': name,
                                'stats': stats
                            })
                
                if cbam_group_stats:
                    # Show statistics from the most important CBAM layers
                    for layer_data in cbam_group_stats:
                        layer_name = layer_data['layer']
                        stats = layer_data['stats']
                        
                        # Simplify layer name for display
                        layer_display = layer_name.split('.')[-2] if '.' in layer_name else layer_name
                        
                        print(f"      {layer_display}:")
                        print(f"        ads_attention_mean: {stats.get('ads_attention_mean', 0):.4f}")
                        print(f"        solv_attention_mean: {stats.get('solv_attention_mean', 0):.4f}")
                        print(f"        interaction_strength: {stats.get('interaction_strength_value', 0):.4f}")
                        
                        # Assess group interaction effectiveness
                        ads_mean = stats.get('ads_attention_mean', 0)
                        solv_mean = stats.get('solv_attention_mean', 0)
                        interaction_strength = stats.get('interaction_strength_value', 0)
                        
                        if solv_mean > ads_mean and interaction_strength > 0.3:
                            print(f"        ✓ Effective group interaction: solvent emphasized with {interaction_strength:.3f} strength")
                        elif interaction_strength < 0.1:
                            print(f"        ⚠️ Weak group interaction: strength {interaction_strength:.3f} too low")
                        else:
                            print(f"        📊 Active group interaction: strength {interaction_strength:.3f}")
                        
                        # Cross-group correlation analysis
                        correlation = stats.get('cross_group_correlation', 0)
                        if abs(correlation) > 0.3:
                            print(f"        💫 Cross-group correlation: {correlation:.3f} (strong coupling)")
                        elif abs(correlation) > 0.1:
                            print(f"        🔗 Cross-group correlation: {correlation:.3f} (moderate coupling)")
                        else:
                            print(f"        ⚪ Cross-group correlation: {correlation:.3f} (weak coupling)")
                else:
                    # Check if group interaction is enabled but not yet active
                    has_group_cbam = False
                    for name, module in model.named_modules():
                        if hasattr(module, 'channel_attention') and hasattr(module.channel_attention, 'enable_group_interaction'):
                            has_group_cbam = True
                            if module.channel_attention.enable_group_interaction:
                                print(f"      Group interaction enabled but no stats available yet")
                                break
                    
                    if not has_group_cbam:
                        print(f"      No CBAM group interaction modules found")
                    else:
                        print(f"      CBAM group interaction modules detected but not active")
                        
            except Exception as e:
                print(f"      CBAM group interaction analysis error: {str(e)[:50]}...")
        else:
            print(f"      Model not available for CBAM group interaction analysis")
        
        # ========== 7. LAYER CONTRIBUTIONS & ARCHITECTURE ANALYSIS ==========
        print(f"\n    🏗️ Layer Contributions & Architecture Analysis:")
        
        # Dynamic layer detection for statistics (adapts to architecture changes)
        def get_dynamic_layer_patterns():
            """Get dynamic layer patterns based on model architecture"""
            base_layers = ['initial_conv', 'adsorbate_processor', 'solvent_processor', 'interaction_attention']
            
            # Dynamically detect CNN layers
            cnn_layers = []
            for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
                if hasattr(model, layer_name):
                    cnn_layers.append(layer_name)
            
            return base_layers + cnn_layers
        
        dynamic_layer_patterns = get_dynamic_layer_patterns()
        
        # Traditional layer statistics from monitoring data (dynamically adapted)
        layer_stats = {}
        for key, value in monitoring_data.items():
            if any(layer in key for layer in dynamic_layer_patterns):
                if isinstance(value, (int, float)):
                    layer_name = key.split('_')[0] if '_' in key else key
                    if layer_name not in layer_stats:
                        layer_stats[layer_name] = []
                    layer_stats[layer_name].append(abs(value))
        
        for layer, values in layer_stats.items():
            if values:
                avg_magnitude = np.mean(values)
                print(f"      {layer}: avg_magnitude={avg_magnitude:.3f}")
        
        # Enhanced architecture analysis for dual-branch model
        if model is not None:
            try:
                # Check for dual-branch processors
                if hasattr(model, 'adsorbate_processor') and hasattr(model, 'solvent_processor'):
                    print(f"      🔀 Dual-Branch Processors detected:")
                    
                    # Get processor statistics if available
                    if hasattr(model, 'get_processor_statistics'):
                        processor_stats = model.get_processor_statistics()
                        if processor_stats:
                            # Adsorbate processor stats
                            if 'adsorbate' in processor_stats:
                                ads_stats = processor_stats['adsorbate']
                                print(f"        🔗 Adsorbate processor:")
                                print(f"          Params: {ads_stats.get('total_params', 'N/A'):,}, Efficiency: {ads_stats.get('param_efficiency', 0):.3f}")
                                print(f"          Avg weight norm: {ads_stats.get('avg_weight_norm', 0):.3f}, Architecture: {ads_stats.get('architecture', 'N/A')}")
                            
                            # Solvent processor stats
                            if 'solvent' in processor_stats:
                                sol_stats = processor_stats['solvent']
                                print(f"        💧 Solvent processor:")
                                print(f"          Params: {sol_stats.get('total_params', 'N/A'):,}, Efficiency: {sol_stats.get('param_efficiency', 0):.3f}")
                                print(f"          Avg weight norm: {sol_stats.get('avg_weight_norm', 0):.3f}, Architecture: {sol_stats.get('architecture', 'N/A')}")
                            
                            # Branch balance analysis
                            if 'branch_balance' in processor_stats:
                                balance = processor_stats['branch_balance']
                                print(f"        ⚖️ Branch Balance:")
                                print(f"          Adsorbate share: {balance.get('adsorbate_share', 0):.1%}, Solvent share: {balance.get('solvent_share', 0):.1%}")
                                print(f"          Param ratio (sol/ads): {balance.get('param_ratio', 0):.2f}, Design: {balance.get('design_philosophy', 'N/A')}")
                                
                                # Architecture optimization insights
                                param_ratio = balance.get('param_ratio', 0)
                                if param_ratio > 15:
                                    print(f"        💡 Very high solvent emphasis - good for interaction-heavy systems")
                                elif param_ratio > 8:
                                    print(f"        💡 Strong solvent emphasis - suitable for most molecular systems")
                                elif param_ratio > 3:
                                    print(f"        💡 Moderate solvent emphasis - balanced approach")
                                else:
                                    print(f"        💡 Balanced or adsorbate-heavy - may need adjustment for interactions")
                        else:
                            print(f"        Processors active but statistics not available")
                    else:
                        print(f"        Processors active but statistics not available")
                
                # Enhanced fusion layer analysis
                if hasattr(model, 'interaction_conv'):
                    print(f"      🔗 Interaction Fusion Layer:")
                    try:
                        # Get interaction layer statistics from processor_stats
                        if hasattr(model, 'get_processor_statistics'):
                            processor_stats = model.get_processor_statistics()
                            if processor_stats and 'interaction' in processor_stats:
                                int_stats = processor_stats['interaction']
                                print(f"        Layer type: {int_stats.get('architecture', 'feature_mixing')}")
                                print(f"        Input channels: {int_stats.get('input_channels', 72)} → Output channels: {int_stats.get('output_channels', 32)}")
                                print(f"        Parameters: {int_stats.get('total_params', 'N/A'):,}, Efficiency: {int_stats.get('param_efficiency', 0):.3f}")
                                print(f"        Avg weight norm: {int_stats.get('avg_weight_norm', 0):.3f}")
                                
                                # Channel reduction analysis
                                if int_stats.get('input_channels', 0) > 0 and int_stats.get('output_channels', 0) > 0:
                                    reduction_ratio = int_stats['output_channels'] / int_stats['input_channels']
                                    print(f"        Channel reduction ratio: {reduction_ratio:.3f}")
                                    if reduction_ratio < 0.5:
                                        print(f"        💡 Aggressive dimensionality reduction - efficient but may lose info")
                                    elif reduction_ratio > 0.8:
                                        print(f"        💡 Conservative feature preservation - safe but potentially redundant")
                                    else:
                                        print(f"        💡 Balanced reduction - good compromise")
                                
                                # Weight norm analysis for fusion quality
                                avg_weight_norm = int_stats.get('avg_weight_norm', 0)
                                if avg_weight_norm > 2.0:
                                    print(f"        ⚠️ High weight norms - may indicate overfitting or instability")
                                elif avg_weight_norm < 0.1:
                                    print(f"        ⚠️ Very low weight norms - may indicate undertraining or dying neurons")
                                else:
                                    print(f"        ✅ Healthy weight norms - good training progress")
                            else:
                                # Fallback to basic weight analysis
                                fusion_layer = model.interaction_conv
                                if hasattr(fusion_layer, '__iter__'):  # Sequential
                                    total_params = sum(p.numel() for p in fusion_layer.parameters())
                                    print(f"        Sequential fusion layer with {total_params:,} parameters")
                                    # Get weight norm from first conv layer
                                    for layer in fusion_layer:
                                        if hasattr(layer, 'weight') and layer.weight is not None:
                                            weight_norm = torch.norm(layer.weight).item()
                                            print(f"        First conv weight norm: {weight_norm:.3f}")
                                            break
                                elif hasattr(fusion_layer, 'weight'):
                                    weight_norm = torch.norm(fusion_layer.weight).item()
                                    print(f"        Fusion weight norm: {weight_norm:.3f}")
                        else:
                            print(f"        Basic fusion layer detected but detailed stats unavailable")
                    except Exception as e:
                        print(f"        Fusion layer analysis error: {str(e)[:50]}...")
                elif hasattr(model, 'fusion_layer'):
                    print(f"      🔗 Alternative Fusion Layer detected:")
                    fusion_layer = model.fusion_layer
                    if hasattr(fusion_layer, 'weight'):
                        weight_norm = torch.norm(fusion_layer.weight).item()
                        print(f"        Fusion weight norm: {weight_norm:.3f}")
                else:
                    print(f"      ❌ No fusion layer detected in current architecture")
            except Exception as e:
                print(f"      Architecture analysis error: {str(e)[:50]}...")
        
        if not layer_stats:
            print(f"      No layer contribution data available")
        
        
        # ========== 8. ATTENTION MECHANISM ANALYSIS ==========
        print(f"\n    🧠 Attention Mechanism Analysis:")
        attention_analysis_printed = False
        
        # - General Attention Summary:
        if model is not None and hasattr(model, 'get_attention_summary'):
            try:
                attention_summary = model.get_attention_summary()
                if attention_summary and attention_summary != "No attention data available":
                    if isinstance(attention_summary, dict):
                        print(f"      - General Attention Summary:")
                        for layer_name, layer_data in attention_summary.items():
                            if isinstance(layer_data, dict) and layer_data:
                                print(f"        {layer_name}:")
                                # Handle all possible attention fields
                                for field_name, field_value in layer_data.items():
                                    print(f"          {field_name}: {field_value}")
                        attention_analysis_printed = True
                    else:
                        # Handle string response
                        print(f"      - General Attention Summary: {attention_summary}")
                        attention_analysis_printed = True
            except Exception as e:
                print(f"      General attention analysis error: {e}")

        
        attention_analysis_printed = True
        
        if not attention_analysis_printed:
            attention_count = sum(1 for key in monitoring_data.keys() if 'attention' in key.lower())
            if attention_count > 0:
                print(f"      Attention layers monitored: {attention_count}")
            else:
                print(f"      No attention mechanism data available")
        
        # ========== 9. ENHANCED MONITORING (BatchNorm, Feature Utilization) ==========
        print(f"\n    🔬 Enhanced Monitoring Analysis:")
        if model is not None:
            try:
                # BatchNorm statistics
                if hasattr(model, 'get_batchnorm_statistics'):
                    bn_stats = model.get_batchnorm_statistics()
                    if bn_stats:
                        total_bn_layers = len(bn_stats)
                        
                        # Analyze all layers for health issues
                        healthy_layers = []
                        problematic_layers = []
                        
                        for layer, stats in bn_stats.items():
                            var_mean = stats['running_var_mean']
                            weight_norm = stats['weight_norm']
                            
                            issues = []
                            if var_mean < 0.5:
                                issues.append("low_variance")
                            elif var_mean > 2.0:
                                issues.append("high_variance")
                            if weight_norm > 10.0:
                                issues.append("large_weights")
                            if weight_norm < 0.1:
                                issues.append("small_weights")
                            
                            if issues:
                                problematic_layers.append((layer, stats, issues))
                            else:
                                healthy_layers.append((layer, stats))
                        
                        print(f"      🏥 BatchNorm Health ({total_bn_layers} layers total):")
                        print(f"        ✅ Healthy layers: {len(healthy_layers)}")
                        print(f"        ⚠️ Layers needing attention: {len(problematic_layers)}")
                        
                        # Always show first 4 representative layers
                        print(f"\n        📊 Representative Layers (first 4):")
                        for layer, stats in list(bn_stats.items())[:4]:
                            layer_display = self._format_bn_layer_name(layer)
                            var_mean = stats['running_var_mean']
                            weight_norm = stats['weight_norm']
                            
                            # Health status
                            if 0.7 <= var_mean <= 1.5 and 1.0 <= weight_norm <= 8.0:
                                status = "✅"
                            elif 0.5 <= var_mean <= 2.0 and 0.5 <= weight_norm <= 10.0:
                                status = "🟡"
                            else:
                                status = "❌"
                            
                            print(f"          {status} {layer_display}: var={var_mean:.3f}, weight={weight_norm:.3f}")
                        
                        # Show ALL problematic layers for debugging
                        if problematic_layers:
                            print(f"\n        ⚠️ Problematic Layers (need attention):")
                            for layer, stats, issues in problematic_layers[:8]:  # Show up to 8 problematic
                                layer_display = self._format_bn_layer_name(layer)
                                var_mean = stats['running_var_mean']
                                weight_norm = stats['weight_norm']
                                issue_str = ", ".join(issues)
                                print(f"          ❌ {layer_display}: var={var_mean:.3f}, weight={weight_norm:.3f} ({issue_str})")
                            
                            if len(problematic_layers) > 8:
                                print(f"          ... and {len(problematic_layers) - 8} more problematic layers")
                        
                        # Layer distribution summary
                        if total_bn_layers > 4:
                            print(f"\n        📋 Layer Distribution:")
                            layer_types = {}
                            for layer_name in bn_stats.keys():
                                if 'adsorbate' in layer_name:
                                    layer_types['ads_branch'] = layer_types.get('ads_branch', 0) + 1
                                elif 'solvent' in layer_name:
                                    layer_types['solv_branch'] = layer_types.get('solv_branch', 0) + 1
                                elif 'layer' in layer_name and 'cnn' not in layer_name:
                                    layer_types['cnn_backbone'] = layer_types.get('cnn_backbone', 0) + 1
                                elif 'interaction' in layer_name:
                                    layer_types['interaction'] = layer_types.get('interaction', 0) + 1
                                else:
                                    layer_types['other'] = layer_types.get('other', 0) + 1
                            
                            for layer_type, count in layer_types.items():
                                print(f"          {layer_type}: {count} layers")
                
                # Attention evolution (if epoch > 0)
                if epoch > 0 and hasattr(model, 'analyze_attention_evolution'):
                    attn_evolution = model.analyze_attention_evolution()
                    if attn_evolution:
                        print(f"\n      🎯 Attention Evolution:")
                        for layer_name, layer_evolution in attn_evolution.items():
                            if 'channel_std_change' in layer_evolution:
                                print(f"        {layer_name}: ch_std={layer_evolution['channel_std_change']:.4f}")
                            if 'spatial_std_change' in layer_evolution:
                                print(f"        {layer_name}: sp_std={layer_evolution['spatial_std_change']:.4f}")
                    else:
                        print(f"\n      🎯 Attention Evolution: No data available yet")
                
                # Learned Feature Importance
                if hasattr(model, 'get_learned_feature_importance'):
                    learned_importance = model.get_learned_feature_importance()
                    if learned_importance:
                        print(f"      🧠 Learned Feature Importance (Top 5):")
                        sorted_features = sorted(learned_importance.items(), 
                                               key=lambda x: x[1]['normalized_weight'], reverse=True)[:5]
                        for i, (name, data) in enumerate(sorted_features, 1):
                            print(f"        {i}. {name:<12}: {data['normalized_weight']:.4f} (rank {data['rank']})")
                
            except Exception as e:
                print(f"      Enhanced analysis error: {str(e)[:50]}...")
        
        # ========== 10. TRAINING INSIGHTS SUMMARY ==========
        print(f"\n    💡 Training Insights Summary:")
        # Calculate insights from monitoring data
        if 'overfitting_ratio' in monitoring_data:
            ratio = monitoring_data['overfitting_ratio']
            if ratio > 2.0:
                print(f"      🔴 Overfitting: {ratio:.2f} - Model struggling to generalize")
            elif ratio < 1.2:
                print(f"      🟢 Good fit: {ratio:.2f} - Healthy train/test balance")
            else:
                print(f"      🟡 Moderate fit: {ratio:.2f} - Acceptable generalization")
        
        if 'avg_grad_norm' in monitoring_data:
            grad_norm = monitoring_data['avg_grad_norm']
            if grad_norm < 0.01:
                print(f"      📉 Small gradients: {grad_norm:.4f} - May need LR increase")
            elif grad_norm > 1.0:
                print(f"      📈 Large gradients: {grad_norm:.4f} - May need LR reduction")
            else:
                print(f"      ✅ Normal gradients: {grad_norm:.4f} - Good learning dynamics")
        
        # ========== 11. SIMPLIFIED HEALTH ASSESSMENT ==========
        if model is not None:
            print(f"\n    🏥 TRAINING HEALTH ASSESSMENT:")
            try:
                # Get training health summary
                health_summary = model.get_training_health_summary()
                status_emoji = {'healthy': '🟢', 'warning': '🟡', 'critical': '🔴'}.get(health_summary['overall_status'], '❓')
                print(f"      {status_emoji} Overall Status: {health_summary['overall_status'].upper()}")
                print(f"      📊 Confidence Score: {health_summary['confidence_score']:.3f}")
                print(f"      🧬 Molecular Interaction: {health_summary['molecular_interaction_health']}")
                
                if health_summary['primary_issues']:
                    print(f"      ⚠️ Key Issues: {', '.join(health_summary['primary_issues'])}")
                
                if health_summary['priority_actions']:
                    print(f"      🎯 Priority Actions:")
                    for action in health_summary['priority_actions']:
                        print(f"        • {action}")
                
                # Get molecular interaction insights
                molecular_insights = model.get_molecular_interaction_insights()
                if molecular_insights:
                    print(f"      � Molecular Learning: {molecular_insights.get('interaction_learning', 'unknown')}")
                    if 'avg_interaction_strength' in molecular_insights:
                        strength = molecular_insights['avg_interaction_strength']
                        print(f"      ⚡ Interaction Strength: {strength:.3f}")
                    
                    if 'adsorbate_contribution' in molecular_insights:
                        ads_contrib = molecular_insights['adsorbate_contribution']
                        print(f"      🧪 Adsorbate:Solvent = {ads_contrib:.1%}:{1-ads_contrib:.1%}")
                
                # Get learning efficiency
                current_loss = monitoring_data.get('test_loss', monitoring_data.get('train_loss'))
                previous_loss = getattr(self, '_previous_loss', None) if hasattr(self, '_previous_loss') else None
                grad_norm = monitoring_data.get('avg_grad_norm')
                
                efficiency_stats = model.get_learning_efficiency_stats(current_loss, previous_loss, grad_norm)
                if efficiency_stats:
                    if 'convergence_speed' in efficiency_stats:
                        speed_emoji = {'fast': '🚀', 'moderate': '🚶', 'slow': '🐌', 'stagnant': '🛑'}.get(efficiency_stats['convergence_speed'], '❓')
                        print(f"      {speed_emoji} Learning Speed: {efficiency_stats['convergence_speed']}")
                    
                    if 'gradient_status' in efficiency_stats:
                        grad_emoji = {'too_high': '📈', 'normal': '✅', 'weak': '📉'}.get(efficiency_stats['gradient_status'], '❓')
                        print(f"      {grad_emoji} Gradient Status: {efficiency_stats['gradient_status']}")
                
                # Store current loss for next iteration
                if current_loss is not None:
                    self._previous_loss = current_loss
                    
            except Exception as e:
                print(f"      ❌ Health assessment failed: {e}")
        
        print(f"    {'='*50}")  # Separator for readability
    
    def _print_formatted_attention_analysis(self, attention_dict, indent="        ", brief=True):
        """
        Format and print the Attention Analysis dictionary in a clean, readable way
        
        Args:
            attention_dict: The attention analysis dictionary
            indent: Indentation string for output formatting
            brief: If True, show brief summary; if False, show full details
        """
        if not isinstance(attention_dict, dict):
            print(f"{indent}📊 Attention Analysis: (invalid data format)")
            return
        
        if brief:
            # Brief version for training logs
            print(f"{indent}📊 Attention Summary:")
            
            # Overall Health Status (brief)
            if 'overall_health' in attention_dict:
                health = attention_dict['overall_health']
                status = health.get('status', 'unknown')
                status_icon = "🟢" if status == 'healthy' else "🟡" if status == 'needs_optimization' else "🔴"
                print(f"{indent}  {status_icon} Status: {status.upper()}")
                
                # Key metrics summary
                if 'metrics' in health:
                    metrics = health['metrics']
                    head_div = metrics.get('head_diversity', 0)
                    sat_level = metrics.get('saturation_level', 0)
                    entropy = metrics.get('attention_entropy', 0)
                    print(f"{indent}  📊 Metrics: diversity={head_div:.3f} | saturation={sat_level:.1%} | entropy={entropy:.1f}")
            
            # Layer count (brief)
            if 'attention_layers' in attention_dict:
                layers = attention_dict['attention_layers']
                layer_count = len(layers)
                healthy_count = sum(1 for l in layers.values() if l.get('attention_health') == 'healthy')
                print(f"{indent}  🔍 Layers: {healthy_count}/{layer_count} healthy")
            
            # Pooling strategy (brief)
            if 'pooling_analysis' in attention_dict:
                pooling = attention_dict['pooling_analysis']
                strategy = pooling.get('learned_strategy', 'unknown')
                print(f"{indent}  🎯 Pooling: {strategy}")
            
        else:
            # Full detailed version
            print(f"{indent}📊 Attention Analysis Summary:")
            
            # Overall Health Status
            if 'overall_health' in attention_dict:
                health = attention_dict['overall_health']
                status = health.get('status', 'unknown')
                issues = health.get('issues', [])
                recommendations = health.get('recommendations', [])
                
                status_icon = "🟢" if status == 'healthy' else "🟡" if status == 'needs_optimization' else "🔴"
                print(f"{indent}  {status_icon} Overall Status: {status.upper()}")
                
                if issues:
                    print(f"{indent}    Issues: {', '.join(issues)}")
                if recommendations:
                    for i, rec in enumerate(recommendations[:2], 1):  # Show top 2 recommendations
                        print(f"{indent}    {i}. {rec}")
                
                # Overall metrics summary
                if 'metrics' in health:
                    metrics = health['metrics']
                    head_div = metrics.get('head_diversity', 0)
                    sat_level = metrics.get('saturation_level', 0)
                    entropy = metrics.get('attention_entropy', 0)
                    print(f"{indent}    Global: diversity={head_div:.3f}, saturation={sat_level:.1%}, entropy={entropy:.1f}")
            
            # Layer-by-layer Analysis (detailed)
            if 'attention_layers' in attention_dict:
                layers = attention_dict['attention_layers']
                print(f"{indent}  🔍 Layers: {len(layers)} attention layers")
                
                for layer_name, layer_data in layers.items():
                    if isinstance(layer_data, dict):
                        head_div = layer_data.get('head_diversity', 0)
                        entropy = layer_data.get('attention_entropy', 0)
                        health = layer_data.get('attention_health', 'unknown')
                        health_icon = "✅" if health == 'healthy' else "⚠️"
                        
                        print(f"{indent}    {health_icon} {layer_name}: diversity={head_div:.3f}, entropy={entropy:.1f}")
            
            # Pooling Strategy (detailed)
            if 'pooling_analysis' in attention_dict:
                pooling = attention_dict['pooling_analysis']
                strategy = pooling.get('learned_strategy', 'unknown')
                print(f"{indent}  🎯 Pooling: {strategy} strategy")
            
        print(f"{indent}  └─ Brief summary (use detailed logs for full analysis)")
    
    
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
        
        # Calculate overfitting metrics for all folds
        overfitting_ratios = []
        for i in range(self.n_folds):
            if 'monitoring_data' in self.model_storage[i] and self.model_storage[i]['monitoring_data']:
                monitor_data = self.model_storage[i]['monitoring_data']
                if 'final_analysis' in monitor_data:
                    overfitting_ratios.append(monitor_data['final_analysis']['final_overfitting_ratio'])
                elif 'overfitting_ratios' in monitor_data and monitor_data['overfitting_ratios']:
                    overfitting_ratios.append(monitor_data['overfitting_ratios'][-1])
                else:
                    overfitting_ratios.append(test_rmses[i]**2 / train_rmses[i]**2)  # Approximate ratio
            else:
                overfitting_ratios.append(test_rmses[i]**2 / train_rmses[i]**2)  # Approximate ratio
        
        print("📊 Per-fold Performance Analysis:")
        print("-" * 80)
        for i in range(self.n_folds):
            overfitting_status = ""
            if overfitting_ratios[i] > 2.0:
                overfitting_status = " ⚠️ OVERFIT"
            elif overfitting_ratios[i] > 1.5:
                overfitting_status = " ⚡MILD_OVERFIT"
            elif overfitting_ratios[i] < 1.1:
                overfitting_status = " ✓GOOD_FIT"
            
            print(f"  Fold {i+1}: Train RMSE={train_rmses[i]:.4f}, R²={train_r2s[i]:.4f} | "
                  f"Test RMSE={test_rmses[i]:.4f}, R²={test_r2s[i]:.4f} | "
                  f"Ratio={overfitting_ratios[i]:.2f}{overfitting_status}")
            
            # Add training details for each fold if available
            if 'monitoring_data' in self.model_storage[i] and self.model_storage[i]['monitoring_data']:
                monitor_data = self.model_storage[i]['monitoring_data']
                if 'final_analysis' in monitor_data:
                    final_analysis = monitor_data['final_analysis']
                    epochs_completed = final_analysis.get('total_epochs', 'N/A')
                    avg_grad_norm = final_analysis.get('avg_gradient_norm', 0)
                    samples_per_param = final_analysis.get('samples_per_param', 0)
                    print(f"    Training details: {epochs_completed} epochs, "
                          f"grad_norm={avg_grad_norm:.4f}, "
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
        overfitting_cv = np.std(overfitting_ratios) / np.mean(overfitting_ratios) if np.mean(overfitting_ratios) > 0 else 0
        
        print(f"  Test RMSE consistency (CV): {test_rmse_cv:.3f} {'✓ Consistent' if test_rmse_cv < 0.1 else '⚠️ Variable' if test_rmse_cv < 0.2 else '❌ Highly Variable'}")
        print(f"  Test R² consistency (CV): {test_r2_cv:.3f} {'✓ Consistent' if test_r2_cv < 0.1 else '⚠️ Variable' if test_r2_cv < 0.2 else '❌ Highly Variable'}")
        print(f"  Overfitting consistency (CV): {overfitting_cv:.3f} {'✓ Consistent' if overfitting_cv < 0.2 else '⚠️ Variable' if overfitting_cv < 0.4 else '❌ Highly Variable'}")
        print(f"  Average overfitting ratio: {np.mean(overfitting_ratios):.2f} ± {np.std(overfitting_ratios):.2f}")
        
        # Find best and worst models
        self.best_model_idx = np.argmin(test_rmses)
        worst_model_idx = np.argmax(test_rmses)
        
        print(f"\n🏆 Best vs Worst Model Comparison:")
        print("-" * 45)
        print(f"  Best model (Fold {self.best_model_idx + 1}):")
        print(f"    Test RMSE: {test_rmses[self.best_model_idx]:.4f}, R²: {test_r2s[self.best_model_idx]:.4f}, "
              f"Overfitting: {overfitting_ratios[self.best_model_idx]:.2f}")
        print(f"  Worst model (Fold {worst_model_idx + 1}):")
        print(f"    Test RMSE: {test_rmses[worst_model_idx]:.4f}, R²: {test_r2s[worst_model_idx]:.4f}, "
              f"Overfitting: {overfitting_ratios[worst_model_idx]:.2f}")
        print(f"  Performance gap: RMSE {test_rmses[worst_model_idx] - test_rmses[self.best_model_idx]:.4f}, "
              f"R² {test_r2s[self.best_model_idx] - test_r2s[worst_model_idx]:.4f}")
        
        # Overall model recommendations
        print(f"\n💡 Overall Model Optimization Recommendations:")
        print("-" * 55)
        avg_overfitting = np.mean(overfitting_ratios)
        if avg_overfitting > 2.5:
            print(f"  🔧 Strong overfitting detected (avg ratio: {avg_overfitting:.2f}):")
            print(f"     • Increase regularization (dropout, weight decay)")
            print(f"     • Reduce model complexity")
            print(f"     • Collect more training data")
            print(f"     • Improve data augmentation")
        elif avg_overfitting < 1.3:
            print(f"  🚀 Potential underfitting (avg ratio: {avg_overfitting:.2f}):")
            print(f"     • Increase model capacity")
            print(f"     • Reduce regularization")
            print(f"     • Train for more epochs")
            print(f"     • Increase learning rate")
        else:
            print(f"  ✓ Good bias-variance tradeoff (avg ratio: {avg_overfitting:.2f})")
        
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
        
        # Attention mechanism effectiveness summary
        print(f"\n🔍 Attention Mechanism Summary:")
        print("-" * 40)
        
        attention_effectiveness_scores = []
        for i in range(self.n_folds):
            # Use performance as a proxy for attention effectiveness
            # Better performance might indicate more effective attention
            fold_score = test_r2s[i] * (1 / max(overfitting_ratios[i], 0.1))  # Weighted by overfitting control
            attention_effectiveness_scores.append(fold_score)
        
        avg_attention_score = np.mean(attention_effectiveness_scores)
        print(f"  Average attention effectiveness score: {avg_attention_score:.3f}")
        
        if avg_attention_score > 0.8:
            print(f"  🎉 Attention mechanism appears highly effective!")
            print(f"     • Good balance of performance and overfitting control")
        elif avg_attention_score > 0.5:
            print(f"  ✓ Attention mechanism working reasonably well")
        else:
            print(f"  ⚠️  Attention mechanism may need optimization:")
            print(f"     • Consider different attention architectures")
            print(f"     • Adjust attention parameters")
            print(f"     • Validate attention weight patterns")

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
    parser.add_argument('--learning-rate', type=float, default=0.001,
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
    
    # Output file prefix argument - hardcoded for train_3d_cnn_2_8.py
    parser.add_argument('--output-prefix', type=str, default='model_2_8',
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
            'water_pure-hydrophilic': [
                '01_methanol',
                '02_propanol',
                '03_01_1_3_propanediol',
                '04_04_glycerol',
                '05_3c_aldehyde',
            ],
            'methanol_240_water_960-hydrophilic': [
                '01_methanol'  # Test mixed solvent system
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


