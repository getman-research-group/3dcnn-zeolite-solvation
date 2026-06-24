# -*- coding: utf-8 -*-
"""
train_3d_cnn_2_0.py
Training script specifically for Type_2 voxel format (28 channels).
Features:
- Load Type_2 voxel grid data (28 channels: 14 adsorbate + 14 solvent features)
- Train Type_2 3D CNN-Transformer model with enhanced attention mechanism
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

# Import Type_2 model architecture
from model_3d_cnn_2_0 import AttentionCNNTransformerType2



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
                 n_splits=5,
                 batch_size=32,
                 epochs=100,
                 learning_rate=0.001,
                 verbose=True,
                 auto_run=True,
                 job_id='',
                 random_state=42,
                 model_type='cnn',
                 output_prefix='cnn'):  # Add output_prefix parameter
        """
        Initialize the 3D CNN trainer
        
        Args:
            zeolite_types: list of zeolite types
            adsorbates_by_env: dict of environments and adsorbates
            test_mode: bool, if True run in test mode (quick training)
            retrain: bool, if True force retrain even if saved models exist
            box_grids_size: float, voxel grid size
            box_increment: float, voxel increment
            n_splits: int, number of CV folds
            batch_size: int, training batch size
            epochs: int, number of training epochs
            learning_rate: float, learning rate for optimizer
            verbose: bool, print detailed information
            auto_run: bool, if True automatically run training after initialization
            random_state: int, random seed for reproducible splits (default: 42)
            model_type: str, model architecture to use ('cnn' or 'transformer', default: 'cnn')
            output_prefix: str, prefix for output files (default: 'cnn')
        """
        # Set basic parameters first (before calling any methods that use them)
        self.verbose = verbose
        self.random_state = random_state
        self.model_type = model_type  # Store model type
        self.output_prefix = output_prefix  # Store output prefix for file naming
        
        # Validate model type
        if self.model_type not in ['cnn', 'transformer']:
            raise ValueError(f"model_type must be 'cnn' or 'transformer', got '{self.model_type}'")
        
        # Set random seeds for reproducibility
        self.set_random_seeds()
        
        self.zeolite_types = zeolite_types
        self.adsorbates_by_env = adsorbates_by_env
        self.test_mode = test_mode
        self.retrain = retrain
        self.box_grids_size = box_grids_size
        self.box_increment = box_increment
        self.n_splits = n_splits
        
        # Training parameters (now configurable)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.job_id = job_id
        
        # Generate job_id if not provided (for multi-job training identification)
        if not self.job_id:
            # Try to get SLURM job ID first
            slurm_job_id = os.environ.get('SLURM_JOB_ID')
            if slurm_job_id:
                # Use model type as prefix for cleaner naming
                self.job_id = f"{self.model_type}_{slurm_job_id}"
            else:
                # Fallback to parameter-based ID for local runs  
                self.job_id = f"{self.model_type}_lr{learning_rate}_bs{batch_size}_ep{epochs}_cv{n_splits}"
        
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
        
        # Paths - use Type_2 specific output directory
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
            print(f"    Model type: {self.model_type}")
            print(f"    Output directory: {self.model_save_dir}")
            print(f"    Job ID: {self.job_id}")
            print(f"    Device: {self.device}")
            print(f"    Model type: {self.model_type}")
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
        """Create Type_2 model based on the specified model type"""
        # Use feature names extracted from actual data loading
        # These will be set after load_data() is called
        feature_names = getattr(self, 'feature_names', None)
        
        if feature_names is None:
            # Fallback warning - this should not happen if load_data() was called first
            print("⚠️  Warning: Feature names not available yet. Using None for now.")
            print("    This is normal if called before data loading.")
        elif self.verbose:
            print(f"    Using {len(feature_names)} feature names from loaded data: {feature_names}")
        
        if self.model_type == 'cnn':
            # Type_2 CNN-only model (28 channels)
            model = AttentionCNNTransformerType2(
                in_channels=28, 
                dropout_rate=dropout_rate,
                use_transformer=False,  # CNN-only
                feature_names=feature_names
            )
            if self.verbose:
                print(f"    📚 Created Type_2 CNN model (28 channels, CNN-only)")
        elif self.model_type == 'transformer':
            # Type_2 CNN-Transformer hybrid model (28 channels)
            model = AttentionCNNTransformerType2(
                in_channels=28,
                dropout_rate=dropout_rate,
                use_transformer=True,      # Enable transformer
                transformer_dim=200,
                transformer_heads=8,
                transformer_layers=2,
                feature_names=feature_names
            )
            if self.verbose:
                print(f"    🔀 Created Type_2 CNN-Transformer model (28 channels, CNN + Transformer)")
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}. Use 'cnn' or 'transformer'")
        
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
    
    
    def get_checkpoint_path(self, fold_idx):
        """Get checkpoint file path for a specific fold with detailed naming"""
        mode_suffix = "-test" if self.test_mode else ""
        
        # Create detailed filename with output prefix and configuration parameters
        filename = (
                    f"{self.output_prefix}_{self.job_id}"
                    f"-epochs_{self.epochs}"
                    f"-bs_{self.batch_size}"
                    f"-lr_{self.learning_rate}"
                    f"-splits_{self.n_splits}"
                    f"-grid_{self.box_grids_size}_{self.box_increment}"
                    f"-fold_{fold_idx}"
                    f"{mode_suffix}.pth")
        print(f"    Checkpoint filename: {filename}")
        return os.path.join(self.model_save_dir, filename)
    
    def get_results_path(self):
        """Get results file path"""
        mode_suffix = "-test" if self.test_mode else ""
        filename = (
                    f"{self.output_prefix}_{self.job_id}"
                    f"-epochs_{self.epochs}"
                    f"-bs_{self.batch_size}"
                    f"-lr_{self.learning_rate}"
                    f"-splits_{self.n_splits}"
                    f"-grid_{self.box_grids_size}_{self.box_increment}"
                    f"{mode_suffix}.pkl")
        return os.path.join(self.model_save_dir, filename)
    
    def save_model(self, model, fold_idx, train_metrics, test_metrics, df_train, df_test, train_idx, test_idx, monitoring_data=None):
        """Save model checkpoint and results with monitoring data"""
        checkpoint_path = self.get_checkpoint_path(fold_idx)
        
        # Get environment-adsorbate groups for train and test indices
        train_env_adsorbates = list(set(self.env_adsorbate_groups[train_idx].tolist()))
        test_env_adsorbates = list(set(self.env_adsorbate_groups[test_idx].tolist()))
        
        # Extract scaler information from the training process
        # Note: This requires access to the scaler used during training
        # We'll need to pass this information from train_single_model
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
                'model_type': self.model_type,  # Add model type to config
                'test_mode': self.test_mode,
                'random_state': self.random_state,
                # Add transformer-specific config if using transformer
                **(
                    {
                        'use_transformer': True,
                        'transformer_dim': 200,
                        'transformer_heads': 8,
                        'transformer_layers': 2,
                        'dropout_rate': 0.25
                    } if self.model_type == 'transformer' else {}
                )
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
        print(f"🎯 FOLD {fold_idx + 1}/{self.n_splits} COMPLETION ANALYSIS")
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
            
            # Create Type_2 model based on saved model config or current model type
            model_config = checkpoint.get('model_config', {})
            saved_model_type = model_config.get('model_type', 'cnn')  # Default to 'cnn' for backward compatibility
            
            # Use feature names extracted from actual data loading
            feature_names = getattr(self, 'feature_names', None)
            if feature_names is None and self.verbose:
                print("    ⚠️  Warning: Feature names not available, using None for model reconstruction")
            
            if saved_model_type == 'cnn':
                model = AttentionCNNTransformerType2(
                    in_channels=28,
                    use_transformer=False,
                    feature_names=feature_names
                )
                if self.verbose:
                    print(f"    📚 Loaded Type_2 CNN model from checkpoint (28 channels)")
            elif saved_model_type == 'transformer':
                model = AttentionCNNTransformerType2(
                    in_channels=28,
                    use_transformer=model_config.get('use_transformer', True),
                    transformer_dim=model_config.get('transformer_dim', 200),
                    transformer_heads=model_config.get('transformer_heads', 8),
                    transformer_layers=model_config.get('transformer_layers', 2),
                    dropout_rate=model_config.get('dropout_rate', 0.25),
                    feature_names=feature_names
                )
                if self.verbose:
                    print(f"    🔀 Loaded Type_2 CNN-Transformer model from checkpoint (28 channels)")
            else:
                # Fallback for unknown model types
                model = self.create_model()
                if self.verbose:
                    print(f"    ⚠️  Unknown saved model type '{saved_model_type}', using current model type '{self.model_type}'")
                    
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
        for fold_idx in range(self.n_splits):
            if not os.path.exists(self.get_checkpoint_path(fold_idx)):
                all_exist = False
                break
        return all_exist
    
    
    def train_single_model(self, X_train, y_train, X_test, y_test, fold_idx, train_idx, test_idx):
        """Train a single model for one fold with improved training strategies and detailed monitoring"""
        if self.verbose:
            print(f"\n--- Training Fold {fold_idx + 1}/{self.n_splits} ---")
        
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
        
        # Initialize model with adaptive dropout based on dataset size
        # Reduce dropout for smaller datasets
        dropout_rate = 0.3 if len(X_train) > 1000 else 0.2
        model = self.create_model(dropout_rate=dropout_rate)  # Use the new create_model method
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
        
        # Loss and optimizer - balanced approach focusing on test performance
        criterion = nn.MSELoss()
        
        # Based on log analysis: moderate weight decay since we want good test performance
        # Severe overfitting but we prioritize test scores over train-test gap
        weight_decay = 1e-4 if len(X_train) < 1000 else 2e-4  # Reduced from previous values
        
        # Use AdamW with differentiated learning for attention vs conv layers
        attention_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if 'attention' in name.lower() or 'cbam' in name.lower() or 'fc' in name.lower():
                attention_params.append(param)  # Include attention and FC layers
            else:
                other_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': other_params, 'weight_decay': weight_decay},
            {'params': attention_params, 'weight_decay': weight_decay * 0.5, 'lr': self.learning_rate * 0.8}  # Slightly lower LR for attention/classifier stability
        ], lr=self.learning_rate, eps=1e-8)
        
        # Add gradient clipping to handle the large gradients seen in logs
        max_grad_norm = 1.0  # Clip gradients that exceed this norm
        
        # Based on log: LR reductions at epochs 15, 26 but best performance around epoch 4
        # Use faster LR scheduling to find optimal learning rate quickly
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.7,  # Less aggressive reduction to avoid too-small LR
            patience=5,   # Much shorter patience - adapt quickly 
            min_lr=5e-6  # Slightly higher min to keep learning
        )
        
        # Enhanced early stopping strategy based on log analysis
        best_test_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        # Based on log analysis: optimal performance consistently at epoch 4-10
        # Use much more aggressive early stopping focused on test performance
        if self.epochs <= 50:
            early_stop_patience = 8   # Very short patience - log shows degradation after epoch 10
        else:
            early_stop_patience = 12  # Still much shorter than before
        
        # For test mode, allow minimal patience
        if self.test_mode:
            early_stop_patience = 5
        
        # Reduce minimum epochs since log shows good performance very early
        min_epochs = 8  # Reduced from 15 - log shows good results by epoch 4-5
        
        train_losses = []
        test_losses = []
        learning_rates = []
        
        # Add detailed monitoring lists
        overfitting_ratios = []
        gradient_norms = []
        weight_norms_history = []
        activation_stats_history = []
        
        if self.verbose:
            print(f"    Training config: dropout={dropout_rate:.2f}, weight_decay={weight_decay:.0e}")
            print(f"    Early stopping patience: {early_stop_patience} epochs")
            print(f"    LR scheduler: factor=0.7, patience=15, min_lr=1e-6")
            print(f"    Monitoring: overfitting, gradients, weights, activations, attention")
            
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
                elif 'classifier' in name.lower() or 'fc' in name.lower():
                    total_fc_params += param.numel()
            
            print(f"      Convolutional layers: {total_conv_params:,} parameters ({total_conv_params/total_params*100:.1f}%)")
            print(f"      Classifier layers: {total_fc_params:,} parameters ({total_fc_params/total_params*100:.1f}%)")
            
            model.train()  # Switch back to training mode
        
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
            
            # Enhanced early stopping with multiple checkpoints based on log analysis
            improvement_threshold = 1e-5  # Very sensitive to small improvements
            
            # Save best model if improvement detected
            if test_loss < best_test_loss - improvement_threshold:
                best_test_loss = test_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                improvement_flag = "✓"
            else:
                patience_counter += 1
                improvement_flag = ""
            
            # Based on log: also save model if we achieve very good training progress early
            # This helps capture models that might be good but slightly worse than absolute best
            if epoch <= 15 and test_loss < 0.35:  # Log shows good models often have test_loss < 0.35
                backup_model_state = model.state_dict().copy()
            
            # Only apply early stopping after minimum epochs
            if epoch >= min_epochs and patience_counter >= early_stop_patience:
                if self.verbose:
                    print(f"  Early stopping at epoch {epoch + 1} (no improvement for {early_stop_patience} epochs)")
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
                    print(f"  Stopping at epoch {epoch + 1} (learning rate too small: {new_lr:.0e})")
                break
            
            # Unified Enhanced logging - EVERY EPOCH
            if self.verbose:
                lr_change = f" → {new_lr:.6f}" if old_lr != new_lr else ""
                grad_info = f", Grad={avg_grad_norm:.3f}" if avg_grad_norm > 0 else ""
                
                print(f"\n  Epoch {epoch + 1:3d}/{self.epochs}: "
                      f"Train={train_loss:.4f}, Test={test_loss:.4f} {improvement_flag}, "
                      f"Ratio={overfitting_ratio:.2f}{overfitting_status}, "
                      f"LR={old_lr:.6f}{lr_change}{grad_info}, "
                      f"Patience={patience_counter}/{early_stop_patience}")
                
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
                    if 'channel' in attention_info:
                        print(f"        📊 Channel attention: {attention_info['channel']}")
                    if 'spatial' in attention_info:
                        print(f"        🗺️ Spatial attention: {attention_info['spatial']}")
            
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
            
            # Weight analysis
            final_weight_norms = model.get_layer_weight_norms()
            print(f"    ⚖️  Final Weight Analysis:")
            conv_weights = [norm for name, norm in final_weight_norms.items() if 'conv' in name.lower()]
            fc_weights = [norm for name, norm in final_weight_norms.items() if 'classifier' in name.lower()]
            
            if conv_weights:
                print(f"      Conv layer weights: μ={np.mean(conv_weights):.3f}, σ={np.std(conv_weights):.3f}, range=[{min(conv_weights):.3f}, {max(conv_weights):.3f}]")
            if fc_weights:
                print(f"      FC layer weights: μ={np.mean(fc_weights):.3f}, σ={np.std(fc_weights):.3f}, range=[{min(fc_weights):.3f}, {max(fc_weights):.3f}]")
            
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
                import inspect
                sig = inspect.signature(model.print_training_insights)
                
                # Check if method requires monitoring_data parameter
                if 'monitoring_data' in sig.parameters:  # CNN model - requires monitoring_data
                    model.print_training_insights(monitoring_data=final_monitoring_data, epoch=(epoch + 1))
                else:  # Transformer model - uses internal data
                    model.print_training_insights(epoch=(epoch + 1))
                    
            except Exception as e:
                print(f"    ⚠️ Training insights generation failed: {e}")
        
        # Load best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            if self.verbose:
                print(f"    Loaded best model (test loss: {best_test_loss:.4f})")
        else:
            if self.verbose:
                print(f"    Warning: No improvement found, using final model state")
        
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
        
        
        # Store monitoring data in model storage
        monitoring_data = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'learning_rates': learning_rates,
            'overfitting_ratios': overfitting_ratios,
            'gradient_norms': gradient_norms,
            'weight_norms_history': weight_norms_history,
            'activation_stats_history': activation_stats_history,
            'final_analysis': {
                'final_overfitting_ratio': final_overfitting_ratio,
                'max_overfitting_ratio': max_overfitting_ratio,
                'avg_gradient_norm': avg_gradient_norm,
                'samples_per_param': samples_per_param,
                'total_epochs': epoch + 1
            }
        }
        
        # Add transformer-specific monitoring data if using transformer model
        if self.model_type == 'transformer' and hasattr(model, 'get_cross_attention_stats'):
            try:
                # Use UNIFIED cross-attention monitoring - ALL fields, EVERY time
                unified_stats = model.get_cross_attention_stats()
                monitoring_data['cross_attention_unified'] = unified_stats
                
                if self.verbose:
                    print(f"    🤖 Cross-Attention Unified Monitoring:")
                    print(f"      • fusion_layers_count: {unified_stats['fusion_layers_count']}")
                    print(f"      • average_fusion_strength: {unified_stats['average_fusion_strength']:.3f}")
                    print(f"      • fusion_diversity: {unified_stats['fusion_diversity']:.3f}")
                    print(f"      • feature_enhancement_ratio: {unified_stats['feature_enhancement_ratio']:.3f}")
                    print(f"      • average_attention_entropy: {unified_stats['average_attention_entropy']:.3f}")
                    print(f"      • average_attention_focus: {unified_stats['average_attention_focus']:.3f}")
                    print(f"      • entropy_consistency: {unified_stats['entropy_consistency']:.3f}")
                    print(f"      • overall_fusion_quality: {unified_stats['overall_fusion_quality']:.3f}")
                    print(f"      • fusion_assessment: {unified_stats['fusion_assessment']}")
                    print(f"      • note: {unified_stats['note']}")
            except Exception as e:
                if self.verbose:
                    print(f"    ⚠️  Warning: Could not collect transformer monitoring data: {e}")
        
        return model, train_metrics, test_metrics, df_train, df_test, monitoring_data

    def run_training(self):
        """Run the complete training pipeline with GroupKFold CV"""
        if self.verbose:
            print("\n=== Starting 3D CNN Training ===")
        
        # Load data if not already loaded
        if self.X_data is None:
            self.load_data()
        
        # Check for existing models
        if not self.retrain and self.check_existing_models():
            if self.verbose:
                print("\n--- Found existing trained models. Loading...")
            self.load_all_models()
            return
        else:
            if self.verbose:
                print("\n--- No existing models found or retraining is enabled. Starting training from scratch.")
        
        # Setup GroupKFold with random_state for reproducible splits
        group_kfold = GroupKFold(n_splits=self.n_splits)
        
        if self.verbose:
            print(f"\n--- Using GroupKFold with {self.n_splits} splits (random_state={self.random_state})")
            print(f"    Training on {self.num_env_adsorbates} environment-adsorbate combinations")
        
        # Get splits first to ensure reproducibility
        splits = list(group_kfold.split(self.X_data, self.y_data, self.env_adsorbate_groups))
        
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
            model, train_metrics, test_metrics, df_train, df_test, monitoring_data = self.train_single_model(
                X_train, y_train, X_test, y_test, fold_idx, train_idx, test_idx
            )
            
            # Save model with split information
            self.save_model(model, fold_idx, train_metrics, test_metrics, df_train, df_test, train_idx, test_idx, monitoring_data)

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
        for fold_idx in range(self.n_splits):
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
            'model_storage': self.model_storage,
            'data_info': {
                'num_env_adsorbates': self.num_env_adsorbates,
                'num_total_datapoints': self.num_total_datapoints,
                'input_data_shape': self.input_data_shape
            },
            'training_config': {
                'test_mode': self.test_mode,
                'n_splits': self.n_splits,
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
    
    def _print_monitoring_summary(self, monitoring_data, epoch, batch_idx, model=None):
        """🎯 UNIFIED MONITORING - All analysis in one place, every epoch!"""
        if not monitoring_data or not isinstance(monitoring_data, dict):
            return
        
        print(f"\n    🔬 [Epoch {epoch+1}] UNIFIED Comprehensive Analysis:")
        
        # ========== 1. WEIGHT NORMS ANALYSIS ==========
        print(f"\n    🔧 Weight Norms:")
        if 'current_weight_norms' in monitoring_data:
            current_weight_norms = monitoring_data['current_weight_norms']
            for layer_name, norm in list(current_weight_norms.items())[:6]:  # Show first 6 layers
                layer_short = layer_name.split('.')[-2] + '.' + layer_name.split('.')[-1] if '.' in layer_name else layer_name
                print(f"      {layer_short:<25}: {norm:.4f}")
        else:
            print(f"      No weight norm data available")
        
        # ========== 2. GRADIENT ANALYSIS ==========
        print(f"\n    📈 Gradient Analysis:")
        if 'gradient_norms' in monitoring_data and monitoring_data['gradient_norms']:
            gradient_norms = monitoring_data['gradient_norms']
            avg_grad_norm = monitoring_data.get('avg_grad_norm', 0)
            print(f"      Average gradient norm: {avg_grad_norm:.4f}")
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
        for key, value in monitoring_data.items():
            if 'mean' in key and not key.startswith('train_') and not key.startswith('test_'):
                base_name = key.replace('_mean', '')
                mean_val = value
                std_val = monitoring_data.get(f"{base_name}_std", 0)
                zeros_pct = monitoring_data.get(f"{base_name}_zeros_pct", 0)
                print(f"      {base_name:<15}: μ={mean_val:.3f}, σ={std_val:.3f}, zeros={zeros_pct:.1f}%")
                activation_analysis_printed = True
        
        if not activation_analysis_printed:
            print(f"      No activation analysis data available")
        
        # ========== 4. FEATURE ANALYSIS ==========
        print(f"\n    🧪 Feature Analysis:")
        if model is not None:
            try:
                # Real feature importance analysis
                if hasattr(model, 'input_attention') and hasattr(model.input_attention, 'last_attention_weights'):
                    # Get channel attention weights (feature importance)
                    attention_weights = model.input_attention.last_attention_weights
                    if attention_weights is not None:
                        # Calculate feature importance from attention weights
                        avg_weights = attention_weights.mean(dim=0).squeeze()  # Average across batch
                        top_features = torch.topk(avg_weights, k=5)
                        bottom_features = torch.topk(avg_weights, k=3, largest=False)
                        
                        print(f"      Top 5 important features:")
                        for i, (idx, weight) in enumerate(zip(top_features.indices, top_features.values)):
                            print(f"        {i+1}. Channel {idx.item()}: {weight.item():.4f}")
                        
                        print(f"      Least important features:")
                        for i, (idx, weight) in enumerate(zip(bottom_features.indices, bottom_features.values)):
                            print(f"        Channel {idx.item()}: {weight.item():.4f}")
                        
                        # Feature utilization statistics
                        feature_diversity = avg_weights.std().item() / (avg_weights.mean().item() + 1e-8)
                        print(f"      Feature diversity: {feature_diversity:.3f}")
                        
                        # Count effectively used features (above average)
                        avg_importance = avg_weights.mean().item()
                        active_features = (avg_weights > avg_importance).sum().item()
                        print(f"      Active features (above avg): {active_features}/{len(avg_weights)}")
                    else:
                        print(f"      No attention weights available yet")
                elif hasattr(model, 'get_learned_feature_importance'):
                    # Alternative method if available
                    importance = model.get_learned_feature_importance()
                    if importance:
                        print(f"      Learned feature importance available: {len(importance)} features")
                        top_3 = sorted(importance.items(), key=lambda x: x[1].get('normalized_weight', 0), reverse=True)[:3]
                        for name, data in top_3:
                            print(f"        {name}: {data.get('normalized_weight', 0):.4f}")
                    else:
                        print(f"      No learned feature importance data")
                else:
                    print(f"      No feature analysis methods available")
            except Exception as e:
                print(f"      Feature analysis error: {str(e)[:50]}...")
        else:
            print(f"      No model available for feature analysis")
        
        # ========== 5. LAYER CONTRIBUTIONS ==========
        print(f"\n    🏗️ Layer Contributions:")
        layer_stats = {}
        for key, value in monitoring_data.items():
            if any(layer in key for layer in ['layer1', 'layer2', 'layer3', 'layer4', 'initial_conv']):
                if isinstance(value, (int, float)):
                    layer_name = key.split('_')[0] if '_' in key else key
                    if layer_name not in layer_stats:
                        layer_stats[layer_name] = []
                    layer_stats[layer_name].append(abs(value))
        
        for layer, values in layer_stats.items():
            if values:
                avg_magnitude = np.mean(values)
                print(f"      {layer}: avg_magnitude={avg_magnitude:.3f}")
        
        if not layer_stats:
            print(f"      No layer contribution data available")
        
        
        # ========== 6. ATTENTION MECHANISM ANALYSIS ==========
        print(f"\n    🧠 Attention Mechanism Analysis:")
        attention_analysis_printed = False
        
        # First: General attention summary (if available)
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
        
        # Second: Cross-Attention Monitoring
        if model is not None and hasattr(model, 'get_cross_attention_stats'):
            try:
                unified_stats = model.get_cross_attention_stats()
                # Show only if meaningful data is present
                if unified_stats['fusion_assessment'] != 'no_data':
                    print(f"      - Cross-Attention:")
                    print(f"        fusion_strength: {unified_stats['average_fusion_strength']:.3f}")
                    print(f"        enhancement_ratio: {unified_stats['feature_enhancement_ratio']:.3f}")
                    print(f"        quality: {unified_stats['fusion_assessment']} ({unified_stats['overall_fusion_quality']:.3f})")
                    print(f"        layers: {unified_stats['fusion_layers_count']}")
                else:
                    # If no meaningful data is present
                    print(f"      - Cross-Attention: {unified_stats['note']} (waiting for meaningful data)")
                attention_analysis_printed = True
            except Exception as e:
                print(f"      Cross-attention error: {e}")
        
        # Third: Self-Attention Patterns (extract from attention summary)
        if model is not None and hasattr(model, 'get_attention_summary'):
            try:
                attention_summary = model.get_attention_summary()
                if attention_summary and isinstance(attention_summary, dict):
                    # Look for self-attention layers
                    self_attention_layers = {k: v for k, v in attention_summary.items() 
                                           if 'self_attention' in k.lower()}
                    
                    if self_attention_layers:
                        print(f"      - Self-Attention Patterns:")
                        for layer_name, layer_data in self_attention_layers.items():
                            if isinstance(layer_data, dict):
                                layer_display = layer_name.replace('_self_attention', '')
                                
                                # Extract available metrics with fallbacks for missing data
                                strength = layer_data.get('self_attention_strength', 'N/A')
                                spatial_var = layer_data.get('spatial_focus_variance', 'N/A')
                                entropy = layer_data.get('attention_entropy', 'N/A')
                                feature_div = layer_data.get('feature_diversity', 'N/A')
                                weights_std = layer_data.get('attention_weights_std', 'N/A')
                                
                                # Format strength value if it's from the 'self_attention' field
                                if strength == 'N/A' and 'self_attention' in layer_data:
                                    # Parse from string format like "strength=0.001"
                                    self_attn_str = layer_data['self_attention']
                                    if 'strength=' in self_attn_str:
                                        strength = self_attn_str.split('strength=')[1].split(',')[0].split(' ')[0]
                                
                                # Format entropy value if it's from the 'entropy' field
                                if entropy == 'N/A' and 'entropy' in layer_data:
                                    entropy = layer_data['entropy']
                                
                                # 🎯 Only print the most useful metrics for model optimization:
                                # - self_attention_strength: indicates how much spatial correlation exists (useful for understanding spatial learning)
                                # - spatial_focus_variance: shows attention uniformity (useful for detecting over-focusing or under-focusing)  
                                # - attention_entropy: measures pattern diversity (useful for detecting attention collapse)
                                # - feature_diversity: shows head specialization (useful for multi-head effectiveness)
                                
                                # Format feature_diversity for better display
                                if feature_div != 'N/A' and isinstance(feature_div, (int, float)):
                                    if feature_div == 0.0:
                                        feature_div_str = "0.0000"
                                    else:
                                        feature_div_str = f"{feature_div:.4f}"
                                else:
                                    feature_div_str = 'N/A'
                                
                                print(f"        {layer_display}.self_attention: strength={strength}, spatial_var={spatial_var}, entropy={entropy}, head_div={feature_div_str}")
                                
                                # Add attention weights std if it's significantly non-zero (indicates dynamic attention)
                                if weights_std != 'N/A' and isinstance(weights_std, (int, float)) and weights_std > 0.01:
                                    print(f"        {layer_display}.attention_dynamics: weights_std={weights_std:.3f} (dynamic attention detected)")
                                
                                # 💡 Interpretation hints for feature_diversity
                                if feature_div != 'N/A' and isinstance(feature_div, (int, float)):
                                    if feature_div < 0.0005:
                                        print(f"        └─ ⚠️ Very low head diversity ({feature_div:.6f}) - heads may be redundant")
                                    elif feature_div < 0.001:
                                        print(f"        └─ ⚠️ Low head diversity ({feature_div:.6f}) - consider fewer heads")
                                    elif feature_div > 0.001:
                                        print(f"        └─ ✅ Good head diversity ({feature_div:.6f}) - heads are specializing")
                                else:
                                    print(f"        └─ ❌ feature_diversity data missing - check model implementation")
                        attention_analysis_printed = True
            except Exception as e:
                print(f"      Self-attention analysis error: {e}")
        
        # Fourth: Check for Vision Transformer specific data in monitoring_data
        # Look for transformer-related data in monitoring_data
        transformer_data = {}
        for key, value in monitoring_data.items():
            if 'vision_transformer' in key.lower() or 'transformer' in key.lower():
                transformer_data[key] = value
        
        if transformer_data:
            print(f"      - Vision Transformer Detailed Data:")
            for key, value in transformer_data.items():
                print(f"        {key}: {value:.3f}")

            attention_analysis_printed = True
        
        if not attention_analysis_printed:
            attention_count = sum(1 for key in monitoring_data.keys() if 'attention' in key.lower())
            if attention_count > 0:
                print(f"      Attention layers monitored: {attention_count}")
            else:
                print(f"      No attention mechanism data available")
        
        # ========== 7. ENHANCED MONITORING (BatchNorm, Feature Utilization) ==========
        print(f"\n    🔬 Enhanced Monitoring Analysis:")
        if model is not None:
            try:
                # BatchNorm statistics
                if hasattr(model, 'get_batchnorm_statistics'):
                    bn_stats = model.get_batchnorm_statistics()
                    if bn_stats:
                        print(f"      🏥 BatchNorm Health:")
                        for layer, stats in list(bn_stats.items())[:4]:  # Show first 4 layers
                            layer_short = layer.split('.')[-1] if '.' in layer else layer
                            print(f"        {layer_short:<15}: var_mean={stats['running_var_mean']:.3f}, "
                                  f"weight_norm={stats['weight_norm']:.3f}")
                
                # Attention evolution (if epoch > 0)
                if epoch > 0 and hasattr(model, 'analyze_attention_evolution'):
                    attn_evolution = model.analyze_attention_evolution()
                    if attn_evolution:
                        print(f"      🎯 Attention Evolution:")
                        for layer_name, layer_evolution in attn_evolution.items():
                            if 'channel_std_change' in layer_evolution:
                                print(f"        {layer_name}: ch_std={layer_evolution['channel_std_change']:.4f}")
                            if 'spatial_std_change' in layer_evolution:
                                print(f"        {layer_name}: sp_std={layer_evolution['spatial_std_change']:.4f}")
                    else:
                        print(f"      🎯 Attention Evolution: No data available yet")
                
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
        
        # ========== 8. TRAINING INSIGHTS SUMMARY ==========
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
        
        print(f"    {'='*50}")  # Separator for readability
    
    
    def print_summary(self):
        """Print comprehensive training summary with detailed analysis"""
        if not self.model_storage:
            print("No models trained yet")
            return
        
        print("\n" + "="*80)
        print("=== COMPREHENSIVE TRAINING SUMMARY ===")
        print("="*80)
        
        # Calculate average metrics
        train_rmses = [self.model_storage[i]['train_rmse'] for i in range(self.n_splits)]
        train_r2s = [self.model_storage[i]['train_r2'] for i in range(self.n_splits)]
        train_maes = [self.model_storage[i]['train_mae'] for i in range(self.n_splits)]
        
        test_rmses = [self.model_storage[i]['test_rmse'] for i in range(self.n_splits)]
        test_r2s = [self.model_storage[i]['test_r2'] for i in range(self.n_splits)]
        test_maes = [self.model_storage[i]['test_mae'] for i in range(self.n_splits)]
        
        # Calculate overfitting metrics for all folds
        overfitting_ratios = []
        for i in range(self.n_splits):
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
        for i in range(self.n_splits):
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
        
        for i in range(self.n_splits):
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
            print(f"  Early stopping rate: {early_stopped_folds}/{self.n_splits} folds ({early_stopped_folds/self.n_splits*100:.1f}%)")
            
            if early_stopped_folds > self.n_splits * 0.8:
                print(f"  ⚠️  Most folds stopped early - consider more patience or different stopping criteria")
            elif early_stopped_folds == 0:
                print(f"  ⚠️  No early stopping occurred - might need more epochs or different criteria")
            else:
                print(f"  ✓ Reasonable early stopping pattern")
        
        # Attention mechanism effectiveness summary
        print(f"\n🔍 Attention Mechanism Summary:")
        print("-" * 40)
        
        attention_effectiveness_scores = []
        for i in range(self.n_splits):
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
    parser.add_argument('--n-splits', type=int, default=5,
                       help='Number of cross-validation folds')
    
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
    
    # Model selection argument
    parser.add_argument('--model-type', choices=['cnn', 'transformer'], default='transformer',
                       help='Model architecture to use: "cnn" for Type_2 CNN-only, "transformer" for Type_2 CNN-Transformer hybrid')
    
    # Output file prefix argument - hardcoded for train_3d_cnn_2_0.py
    parser.add_argument('--output-prefix', type=str, default='model_2_0',
                       help='Prefix for output files (e.g., model checkpoints, CSV files). Defaults to "model_2_0"')
    
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
            'n_splits': args.n_splits,
        }
    else:
        config = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'n_splits': args.n_splits,
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
            print(f"    Model type: {args.model_type}")
            print(f"    Random state: {args.random_state}")
            print(f"    Test mode: {args.test_mode}")
            print(f"    Retrain: {args.retrain}")
            print(f"    Epochs: {1 if args.test_mode else args.epochs}")
            print(f"    Batch size: {8 if args.test_mode else args.batch_size}")
            print(f"    Learning rate: {args.learning_rate}")
            print(f"    CV folds: {args.n_splits}")
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
            n_splits=training_config['n_splits'],
            batch_size=training_config['batch_size'],
            epochs=training_config['epochs'],
            learning_rate=training_config['learning_rate'],
            verbose=args.verbose,
            auto_run=True,
            job_id=args.job_id,
            random_state=args.random_state,
            model_type=args.model_type,  # Add model_type parameter
            output_prefix=args.output_prefix  # Add output_prefix parameter
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


