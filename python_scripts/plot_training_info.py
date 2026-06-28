# -*- coding: utf-8 -*-
"""
plot_training_enhanced.py
Enhanced training visualization script for 3D CNN models
Plots training information like loss curves, overfitting analysis, etc.
"""

import os
import sys
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
import io
warnings.filterwarnings('ignore')

# Add the parent directory to the path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from core.path import get_paths

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 100

class EnhancedTrainingPlotter:
    """Enhanced training visualization class for 3D CNN models"""
    
    def __init__(self,
                 model_file=None,
                 output_dir=None,
                 verbose=True,
                 show_plot=False,
                 save_plot=True,
                 font_size=18):
        """
        Initialize training plotter
        
        Args:
            model_file: Path to model file to analyze (optional)
            output_dir: Output directory for plots (optional)
            verbose: Whether to print detailed information
            show_plot: Whether to display plots interactively (default: False)
            save_plot: Whether to save plots to files (default: True)
            font_size: Font size for all text in plots (default: 12)
        """
        self.verbose = verbose
        self.show_plot = show_plot
        self.save_plot = save_plot
        self.font_size = font_size
        self.model_dir = get_paths("output_model_cnn")
        
        # Set output directory to cnn_training_results folder
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.join(get_paths("output_figure_path"), "cnn_training_results")
        
        self.model_file = model_file
        self.model_name = os.path.basename(model_file) if model_file else None
        
        # Initialize data storage
        self.training_data = {'folds': {}}
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Data storage - simplified initialization
        self.training_data = {'folds': {}}
        
        if self.verbose:
            print(f"=== Enhanced Training Plotter Initialized ===")
            print(f"Model directory: {self.model_dir}")
            print(f"Output directory: {self.output_dir}")
        
        # Auto-load model data if model_file is provided - using simplified loading function
        if self.model_file:
            if self.verbose:
                print(f"\n📂 Auto-loading model data...")
            if self.load_model_data(self.model_file):
                if self.verbose:
                    print(f"✅ Successfully loaded model: {self.model_name}")
                    print(f"📊 Available folds: {len(self.training_data['folds'])}")
            else:
                if self.verbose:
                    print("❌ Failed to auto-load model data.")
    
    def load_model_data(self, model_file):
        """Simplified model data loading function - only reads pickle file information"""
        try:
            # Determine file path
            if os.path.isabs(model_file):
                file_path = model_file
            else:
                possible_paths = [
                    os.path.join(self.model_dir, model_file),
                    model_file,
                    os.path.join(os.getcwd(), model_file)
                ]
                
                file_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        file_path = path
                        break
                
                if not file_path:
                    print(f"❌ Model file not found: {model_file}")
                    return False
            
            if self.verbose:
                print(f"📂 Loading model file: {file_path}")
            
            # Robust pickle loading - handling complex CUDA issues
            data = None
            
            # Method 1: Direct CPU mapping
            try:
                data = torch.load(file_path, map_location='cpu', weights_only=False)
            except RuntimeError as e:
                if "CUDA" in str(e):
                    if self.verbose:
                        print("    Method 1 failed, using custom unpickler...")
                    
                    # Method 2: Custom unpickler to handle deep CUDA issues
                    import pickle
                    import io
                    
                    class CPUUnpickler(pickle.Unpickler):
                        def find_class(self, module, name):
                            if module == 'torch.storage' and name == '_load_from_bytes':
                                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                            return super().find_class(module, name)
                    
                    try:
                        with open(file_path, 'rb') as f:
                            data = CPUUnpickler(f).load()
                    except Exception as e2:
                        if self.verbose:
                            print(f"    Custom unpickler also failed: {str(e2)[:100]}")
                        raise RuntimeError(f"Cannot load file {file_path}: all methods failed")
                else:
                    raise
            
            if data is None:
                raise RuntimeError("All loading methods failed")
            
            # Extract data directly from model_storage
            if 'model_storage' in data:
                self.training_data = {
                    'folds': {},
                    'summary': data.get('training_config', {}),
                    'data_info': data.get('data_info', {})
                }
                
                # Extract data for each fold
                for fold_idx, fold_data in data['model_storage'].items():
                    fold_training_data = {
                        'train_rmse': fold_data.get('train_rmse'),
                        'test_rmse': fold_data.get('test_rmse'),
                        'train_r2': fold_data.get('train_r2'),
                        'test_r2': fold_data.get('test_r2'),
                        'train_mae': fold_data.get('train_mae', 0),
                        'test_mae': fold_data.get('test_mae', 0),
                    }
                    
                    # Extract training curve data (if available)
                    if 'monitoring_data' in fold_data and fold_data['monitoring_data']:
                        monitoring = fold_data['monitoring_data']
                        fold_training_data.update({
                            'train_losses': monitoring.get('train_losses', []),
                            'test_losses': monitoring.get('test_losses', []),
                            'learning_rates': monitoring.get('learning_rates', []),
                            'overfitting_ratios': monitoring.get('overfitting_ratios', []),
                            'gradient_norms': monitoring.get('gradient_norms', [])
                        })
                    else:
                        fold_training_data.update({
                            'train_losses': [],
                            'test_losses': [],
                            'learning_rates': [],
                            'overfitting_ratios': [],
                            'gradient_norms': []
                        })
                    
                    # 🆕 Extract scaler_info from fold_data
                    if 'scaler_info' in fold_data and fold_data['scaler_info']:
                        fold_training_data['scaler_info'] = fold_data['scaler_info']
                    
                    self.training_data['folds'][fold_idx] = fold_training_data
            else:
                self.training_data = data
            
            # 🆕 Check fold-specific scaler information
            fold_scalers_available = any(
                'scaler_info' in self.training_data['folds'][fold_idx] 
                and self.training_data['folds'][fold_idx]['scaler_info'] is not None
                for fold_idx in self.training_data['folds']
            )
            
            if not fold_scalers_available:
                if self.verbose:
                    print("🔍 Fold-specific scaler info not found, trying to extract from .pth files...")
                self._try_extract_scaler_from_pth_files(file_path)
            
            self.model_name = os.path.basename(file_path)
            
            if self.verbose:
                print(f"✅ Successfully loaded model data")
                print(f"- Folds: {len(self.training_data.get('folds', {}))}")
                
                # Display fold-specific scaler information
                fold_scalers_count = sum(
                    1 for fold_idx in self.training_data.get('folds', {})
                    if 'scaler_info' in self.training_data['folds'][fold_idx] 
                    and self.training_data['folds'][fold_idx]['scaler_info'] is not None
                )
                total_folds = len(self.training_data.get('folds', {}))
                
                if fold_scalers_count > 0:
                    print(f"📊 Fold-specific scalers available: {fold_scalers_count}/{total_folds} folds")
                    if fold_scalers_count == total_folds:
                        print("✅ All folds have independent scaler information")
                    else:
                        print(f"⚠️  Missing scaler info for {total_folds - fold_scalers_count} folds")
                else:
                    print("❌ No fold-specific scaler information - will use standardized scale")
                
                if self.training_data.get('folds'):
                    first_fold = list(self.training_data['folds'].values())[0]
                    has_metrics = any(key in first_fold for key in ['train_rmse', 'test_rmse'])
                    has_curves = len(first_fold.get('train_losses', [])) > 0
                    print(f"- Performance metrics: {'✅' if has_metrics else '❌'}")
                    print(f"- Training curves: {'✅' if has_curves else '❌'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model file: {str(e)}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False
    
    def _try_extract_scaler_from_pth_files(self, pkl_file_path):
        """Try to extract scaler information from corresponding .pth files"""
        try:
            # Get the base name without extension
            base_name = os.path.splitext(pkl_file_path)[0]
            model_dir = os.path.dirname(pkl_file_path)
            
            # Look for .pth files with the same base name
            pth_files = []
            for file in os.listdir(model_dir):
                if file.startswith(os.path.basename(base_name)) and file.endswith('.pth'):
                    pth_files.append(os.path.join(model_dir, file))
            
            if not pth_files:
                if self.verbose:
                    print("⚠️  No corresponding .pth files found for scaler extraction")
                return
            
            # Try to extract scaler info from the first .pth file
            for pth_file in pth_files[:1]:  # Just try the first one
                try:
                    if self.verbose:
                        print(f"🔍 Trying to extract scaler info from: {os.path.basename(pth_file)}")
                    
                    checkpoint = torch.load(pth_file, map_location='cpu', weights_only=False)
                    
                    if isinstance(checkpoint, dict) and 'scaler_info' in checkpoint:
                        scaler_info = checkpoint['scaler_info']
                        if scaler_info and 'mean' in scaler_info and 'std' in scaler_info:
                            self.training_data['scaler'] = {
                                'mean': scaler_info['mean'],
                                'std': scaler_info['std']
                            }
                            if self.verbose:
                                print(f"✅ Successfully extracted scaler info: mean={scaler_info['mean']:.6f}, std={scaler_info['std']:.6f}")
                            return
                    
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Failed to load {os.path.basename(pth_file)}: {str(e)}")
                    continue
            
            if self.verbose:
                print("❌ Could not extract scaler information from .pth files")
                
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Error during scaler extraction: {str(e)}")
    
    def plot_loss_curves(self):
        """Plot training and validation loss curves for all folds"""
        if not self.training_data['folds']:
            print("❌ No training data available for plotting loss curves")
            return
        
        # Check if we have loss curves
        folds_with_curves = []
        for fold_idx, fold_data in self.training_data['folds'].items():
            if 'train_losses' in fold_data and len(fold_data.get('train_losses', [])) > 0:
                folds_with_curves.append(fold_idx)
        
        if not folds_with_curves:
            print("❌ No loss curve data available")
            return
        
        # Create subplots
        n_folds = len(folds_with_curves)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training and Validation Loss Curves - All Folds', fontsize=self.font_size, fontweight='bold')
        
        # Plot individual fold curves
        for i, fold_idx in enumerate(folds_with_curves[:5]):  # Max 5 folds
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            fold_data = self.training_data['folds'][fold_idx]
            train_losses = fold_data['train_losses']
            test_losses = fold_data['test_losses']
            
            epochs = range(1, len(train_losses) + 1)
            
            ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.8)
            ax.plot(epochs, test_losses, 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
            
            # Add best epoch marker
            best_epoch = np.argmin(test_losses)
            ax.plot(best_epoch + 1, test_losses[best_epoch], 'ro', markersize=8, 
                   label=f'Best (Epoch {best_epoch + 1})')
            
            ax.set_title(f'Fold {fold_idx + 1}', fontsize=self.font_size, fontweight='bold')
            ax.set_xlabel('Number of epochs', fontsize=self.font_size)
            ax.set_ylabel('Loss', fontsize=self.font_size)
            ax.legend(fontsize=self.font_size)
            ax.grid(True, alpha=0.3)
            
            # Add final metrics as text
            final_train_rmse = fold_data.get('train_rmse', 'N/A')
            final_test_rmse = fold_data.get('test_rmse', 'N/A')
            if final_train_rmse != 'N/A' and final_test_rmse != 'N/A':
                ax.text(0.02, 0.98, f'Train RMSE: {final_train_rmse:.4f}\nTest RMSE: {final_test_rmse:.4f}', 
                       transform=ax.transAxes, verticalalignment='top', fontsize=self.font_size,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot average curve in the last subplot
        if n_folds > 1:
            ax = axes[1, 2]
            
            # Calculate average curves
            max_epochs = max(len(self.training_data['folds'][fold]['train_losses']) 
                           for fold in folds_with_curves)
            
            avg_train_losses = []
            avg_test_losses = []
            std_train_losses = []
            std_test_losses = []
            
            for epoch in range(max_epochs):
                train_values = []
                test_values = []
                
                for fold_idx in folds_with_curves:
                    fold_data = self.training_data['folds'][fold_idx]
                    if epoch < len(fold_data['train_losses']):
                        train_values.append(fold_data['train_losses'][epoch])
                        test_values.append(fold_data['test_losses'][epoch])
                
                if train_values:
                    avg_train_losses.append(np.mean(train_values))
                    avg_test_losses.append(np.mean(test_values))
                    std_train_losses.append(np.std(train_values))
                    std_test_losses.append(np.std(test_values))
            
            epochs = range(1, len(avg_train_losses) + 1)
            
            # Plot average with error bands
            ax.plot(epochs, avg_train_losses, 'b-', label='Avg Train Loss', linewidth=3)
            ax.fill_between(epochs, 
                          np.array(avg_train_losses) - np.array(std_train_losses),
                          np.array(avg_train_losses) + np.array(std_train_losses),
                          alpha=0.2, color='blue')
            
            ax.plot(epochs, avg_test_losses, 'r-', label='Avg Validation Loss', linewidth=3)
            ax.fill_between(epochs, 
                          np.array(avg_test_losses) - np.array(std_test_losses),
                          np.array(avg_test_losses) + np.array(std_test_losses),
                          alpha=0.2, color='red')
            
            ax.set_title('Average Across All Folds', fontsize=self.font_size, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=self.font_size)
            ax.set_ylabel('Loss', fontsize=self.font_size)
            ax.legend(fontsize=self.font_size)
            ax.grid(True, alpha=0.3)
        
        # Remove unused subplot if necessary
        if n_folds < 5:
            for i in range(n_folds, 5):
                row = i // 3
                col = i % 3
                if row < 2 and col < 3:
                    axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # Save the plot
        if self.save_plot:
            save_path = os.path.join(self.output_dir, 'loss_curves_all_folds.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"✅ Loss curves plot saved to: {save_path}")
        
        if self.show_plot:
            plt.show()
        else:
            # Keep figure open for Spyder plots panel when show_plot=False
            pass
    
    def plot_performance_metrics(self):
        """Plot performance metrics comparison across folds"""
        if not self.training_data['folds']:
            print("❌ No performance metrics data available")
            return
        
        # Extract metrics
        metrics_data = {
            'fold': [],
            'train_rmse': [],
            'test_rmse': [],
            'train_r2': [],
            'test_r2': [],
            'train_mae': [],
            'test_mae': []
        }
        
        for fold_idx, fold_data in self.training_data['folds'].items():
            if all(key in fold_data for key in ['train_rmse', 'test_rmse', 'train_r2', 'test_r2']):
                metrics_data['fold'].append(f'Fold {fold_idx + 1}')
                metrics_data['train_rmse'].append(fold_data['train_rmse'])
                metrics_data['test_rmse'].append(fold_data['test_rmse'])
                metrics_data['train_r2'].append(fold_data['train_r2'])
                metrics_data['test_r2'].append(fold_data['test_r2'])
                metrics_data['train_mae'].append(fold_data.get('train_mae', 0))
                metrics_data['test_mae'].append(fold_data.get('test_mae', 0))
        
        if not metrics_data['fold']:
            print("❌ No complete performance metrics data available")
            return
        
        df = pd.DataFrame(metrics_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Metrics Across Folds', fontsize=self.font_size, fontweight='bold')
        
        # RMSE comparison
        ax1 = axes[0, 0]
        x = np.arange(len(df))
        width = 0.35
        
        ax1.bar(x - width/2, df['train_rmse'], width, label='Train RMSE', alpha=0.8, color='skyblue')
        ax1.bar(x + width/2, df['test_rmse'], width, label='Test RMSE', alpha=0.8, color='lightcoral')
        
        ax1.set_title('RMSE Comparison', fontsize=self.font_size, fontweight='bold')
        ax1.set_xlabel('Fold', fontsize=self.font_size)
        ax1.set_ylabel('RMSE', fontsize=self.font_size)
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['fold'], fontsize=self.font_size)
        ax1.legend(fontsize=self.font_size)
        ax1.grid(True, alpha=0.3)
        
        # Add average lines
        ax1.axhline(y=df['train_rmse'].mean(), color='blue', linestyle='--', alpha=0.7, label=f'Avg Train: {df["train_rmse"].mean():.4f}')
        ax1.axhline(y=df['test_rmse'].mean(), color='red', linestyle='--', alpha=0.7, label=f'Avg Test: {df["test_rmse"].mean():.4f}')
        
        # R² comparison
        ax2 = axes[0, 1]
        ax2.bar(x - width/2, df['train_r2'], width, label='Train R²', alpha=0.8, color='lightgreen')
        ax2.bar(x + width/2, df['test_r2'], width, label='Test R²', alpha=0.8, color='orange')
        
        ax2.set_title('R² Score Comparison', fontsize=self.font_size, fontweight='bold')
        ax2.set_xlabel('Fold', fontsize=self.font_size)
        ax2.set_ylabel('R² Score', fontsize=self.font_size)
        ax2.set_xticks(x)
        ax2.set_xticklabels(df['fold'], fontsize=self.font_size)
        ax2.legend(fontsize=self.font_size)
        ax2.grid(True, alpha=0.3)
        
        # Add average lines
        ax2.axhline(y=df['train_r2'].mean(), color='green', linestyle='--', alpha=0.7)
        ax2.axhline(y=df['test_r2'].mean(), color='darkorange', linestyle='--', alpha=0.7)
        
        # Overfitting analysis
        ax3 = axes[1, 0]
        overfitting_ratios = [test_rmse**2 / train_rmse**2 for train_rmse, test_rmse in zip(df['train_rmse'], df['test_rmse'])]
        colors = ['red' if ratio > 2.0 else 'orange' if ratio > 1.5 else 'green' for ratio in overfitting_ratios]
        
        bars = ax3.bar(df['fold'], overfitting_ratios, color=colors, alpha=0.7)
        ax3.set_title('Overfitting Analysis (Test²/Train² RMSE Ratio)', fontsize=self.font_size, fontweight='bold')
        ax3.set_xlabel('Fold', fontsize=self.font_size)
        ax3.set_ylabel('Overfitting Ratio', fontsize=self.font_size)
        ax3.axhline(y=1.5, color='orange', linestyle='--', alpha=0.7, label='Mild Overfitting (1.5)')
        ax3.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Strong Overfitting (2.0)')
        ax3.legend(fontsize=self.font_size)
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.get_xticklabels(), rotation=45, fontsize=self.font_size)
        
        # Add ratio values on bars
        for bar, ratio in zip(bars, overfitting_ratios):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{ratio:.2f}', ha='center', va='bottom', fontsize=self.font_size)
        
        # Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate summary statistics
        summary_text = f"""Summary Statistics:

Train RMSE: {df['train_rmse'].mean():.4f} ± {df['train_rmse'].std():.4f}
Test RMSE:  {df['test_rmse'].mean():.4f} ± {df['test_rmse'].std():.4f}

Train R²:   {df['train_r2'].mean():.4f} ± {df['train_r2'].std():.4f}
Test R²:    {df['test_r2'].mean():.4f} ± {df['test_r2'].std():.4f}

Best Fold:  {df.loc[df['test_rmse'].idxmin(), 'fold']} (Test RMSE: {df['test_rmse'].min():.4f})
Worst Fold: {df.loc[df['test_rmse'].idxmax(), 'fold']} (Test RMSE: {df['test_rmse'].max():.4f})

Avg Overfitting Ratio: {np.mean(overfitting_ratios):.2f}
Performance CV: {df['test_rmse'].std() / df['test_rmse'].mean() * 100:.1f}%
"""
        
        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=self.font_size,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        if self.save_plot:
            save_path = os.path.join(self.output_dir, 'performance_metrics_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"✅ Performance metrics comparison plot saved to: {save_path}")
        
        if self.show_plot:
            plt.show()
        else:
            # Keep figure open for Spyder plots panel when show_plot=False
            pass
    
    def plot_training_dynamics(self):
        """Plot training dynamics (overfitting, learning rates, gradients)"""
        if not self.training_data['folds']:
            print("❌ No training dynamics data available")
            return
        
        # Check for available dynamics data
        folds_with_dynamics = []
        for fold_idx, fold_data in self.training_data['folds'].items():
            if ('overfitting_ratios' in fold_data and len(fold_data.get('overfitting_ratios', [])) > 0) or \
               ('learning_rates' in fold_data and len(fold_data.get('learning_rates', [])) > 0):
                folds_with_dynamics.append(fold_idx)
        
        if not folds_with_dynamics:
            print("❌ No available training dynamics data")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Training Dynamics Analysis', fontsize=self.font_size, fontweight='bold')
        
        # Plot 1: Overfitting ratios over epochs
        ax1 = axes[0, 0]
        for fold_idx in folds_with_dynamics[:5]:  # Show up to 5 folds for complete analysis
            fold_data = self.training_data['folds'][fold_idx]
            if 'overfitting_ratios' in fold_data and len(fold_data['overfitting_ratios']) > 0:
                epochs = range(1, len(fold_data['overfitting_ratios']) + 1)
                ax1.plot(epochs, fold_data['overfitting_ratios'], 
                        label=f'Fold {fold_idx + 1}', linewidth=2, alpha=0.8)
        
        ax1.axhline(y=1.5, color='orange', linestyle='--', alpha=0.7, label='Mild Overfitting')
        ax1.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Strong Overfitting')
        ax1.set_title('Overfitting Ratio Over Training', fontsize=self.font_size, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=self.font_size)
        ax1.set_ylabel('Test/Train Loss Ratio', fontsize=self.font_size)
        ax1.legend(fontsize=self.font_size)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Learning rates over epochs
        ax2 = axes[0, 1]
        for fold_idx in folds_with_dynamics[:5]:  # Show up to 5 folds for complete analysis
            fold_data = self.training_data['folds'][fold_idx]
            if 'learning_rates' in fold_data and len(fold_data['learning_rates']) > 0:
                epochs = range(1, len(fold_data['learning_rates']) + 1)
                ax2.semilogy(epochs, fold_data['learning_rates'], 
                           label=f'Fold {fold_idx + 1}', linewidth=2, alpha=0.8)
        
        ax2.set_title('Learning Rate Schedule', fontsize=self.font_size, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=self.font_size)
        ax2.set_ylabel('Learning Rate (log scale)', fontsize=self.font_size)
        ax2.legend(fontsize=self.font_size)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Gradient norms over epochs
        ax3 = axes[1, 0]
        for fold_idx in folds_with_dynamics[:5]:  # Show up to 5 folds for complete analysis
            fold_data = self.training_data['folds'][fold_idx]
            if 'gradient_norms' in fold_data and len(fold_data['gradient_norms']) > 0:
                epochs = range(1, len(fold_data['gradient_norms']) + 1)
                ax3.plot(epochs, fold_data['gradient_norms'], 
                        label=f'Fold {fold_idx + 1}', linewidth=2, alpha=0.8)
        
        ax3.set_title('Gradient Norms During Training', fontsize=self.font_size, fontweight='bold')
        ax3.set_xlabel('Epoch', fontsize=self.font_size)
        ax3.set_ylabel('Average Gradient Norm', fontsize=self.font_size)
        ax3.legend(fontsize=self.font_size)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Learning curves comparison (loss improvement rate)
        ax4 = axes[1, 1]
        for fold_idx in folds_with_dynamics[:5]:  # Show up to 5 folds for complete analysis
            fold_data = self.training_data['folds'][fold_idx]
            if 'test_losses' in fold_data and len(fold_data['test_losses']) > 5:
                test_losses = np.array(fold_data['test_losses'])
                # Calculate smoothed improvement rate
                window_size = 5
                improvement_rate = []
                for i in range(window_size, len(test_losses)):
                    recent_avg = np.mean(test_losses[i-window_size:i])
                    previous_avg = np.mean(test_losses[i-window_size*2:i-window_size]) if i >= window_size*2 else recent_avg
                    improvement = (previous_avg - recent_avg) / previous_avg if previous_avg > 0 else 0
                    improvement_rate.append(improvement)
                
                if improvement_rate:
                    epochs = range(window_size + 1, len(test_losses) + 1)
                    ax4.plot(epochs, improvement_rate, 
                            label=f'Fold {fold_idx + 1}', linewidth=2, alpha=0.8)
        
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.set_title('Loss Improvement Rate', fontsize=self.font_size, fontweight='bold')
        ax4.set_xlabel('Epoch', fontsize=self.font_size)
        ax4.set_ylabel('Relative Improvement Rate', fontsize=self.font_size)
        ax4.legend(fontsize=self.font_size)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        if self.save_plot:
            save_path = os.path.join(self.output_dir, 'training_dynamics.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"✅ Training dynamics analysis plot saved to: {save_path}")
        
        if self.show_plot:
            plt.show()
        else:
            # Keep figure open for Spyder plots panel when show_plot=False
            pass
    
    
    def plot_early_stopping_analysis(self):
        """Early stopping effectiveness analysis"""
        convergence_data = []
        
        for fold_idx, fold_data in self.training_data['folds'].items():
            if 'test_losses' in fold_data and len(fold_data['test_losses']) > 0:
                test_losses = fold_data['test_losses']
                best_epoch = np.argmin(test_losses)
                total_epochs = len(test_losses)
                
                convergence_data.append({
                    'fold': fold_idx + 1,
                    'best_epoch': best_epoch + 1,
                    'total_epochs': total_epochs,
                    'best_loss': test_losses[best_epoch],
                    'final_loss': test_losses[-1],
                    'early_stopped': total_epochs < 100  # Assuming 100 is max epochs
                })
        
        if not convergence_data:
            print("❌ No convergence data available")
            return
        
        df = pd.DataFrame(convergence_data)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Early Stopping and Convergence Analysis', fontsize=self.font_size, fontweight='bold')
        
        # Plot 1: Best epoch distribution
        ax1 = axes[0]
        ax1.bar(df['fold'], df['best_epoch'], alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Fold', fontsize=self.font_size, fontweight='bold')
        ax1.set_ylabel('Best Epoch', fontsize=self.font_size, fontweight='bold')
        ax1.set_title('Optimal Stopping Point per Fold', fontsize=self.font_size, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add average line
        avg_best = df['best_epoch'].mean()
        ax1.axhline(y=avg_best, color='red', linestyle='--', linewidth=2, 
                   label=f'Average: {avg_best:.1f}')
        ax1.legend()
        
        # Plot 2: Performance degradation after best epoch
        ax2 = axes[1]
        degradation = ((df['final_loss'] - df['best_loss']) / df['best_loss'] * 100)
        colors = ['red' if d > 5 else 'orange' if d > 1 else 'green' for d in degradation]
        
        bars = ax2.bar(df['fold'], degradation, alpha=0.7, color=colors, edgecolor='black')
        ax2.set_xlabel('Fold', fontsize=self.font_size, fontweight='bold')
        ax2.set_ylabel('Performance Degradation (%)', fontsize=self.font_size, fontweight='bold')
        ax2.set_title('Overfitting After Best Epoch', fontsize=self.font_size, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Add value labels on bars
        for bar, val in zip(bars, degradation):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{val:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if self.save_plot:
            save_path = os.path.join(self.output_dir, 'early_stopping_analysis.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            if self.verbose:
                print(f"✅ Early stopping analysis saved to: {save_path}")
        
        if self.show_plot:
            plt.show()
        else:
            # Keep figure open for Spyder plots panel when show_plot=False
            pass
    
    def plot_cv_stability(self):
        """Cross-validation stability and variance analysis"""
        # Collect performance metrics
        performance_data = []
        for fold_idx, fold_data in self.training_data['folds'].items():
            if all(key in fold_data for key in ['test_rmse', 'test_r2']):
                performance_data.append({
                    'fold': fold_idx + 1,
                    'rmse': fold_data['test_rmse'],
                    'r2': fold_data['test_r2']
                })
        
        if len(performance_data) < 3:
            print("❌ Need at least 3 folds of data for stability analysis")
            return
        
        df = pd.DataFrame(performance_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Cross-Validation Stability Analysis', fontsize=self.font_size, fontweight='bold')
        
        # Plot 1: Performance variance
        ax1 = axes[0, 0]
        ax1.plot(df['fold'], df['rmse'], 'bo-', linewidth=2, markersize=8, label='RMSE')
        ax1.fill_between(df['fold'], df['rmse'], alpha=0.3, color='blue')
        
        # Add mean and std bands
        mean_rmse = df['rmse'].mean()
        std_rmse = df['rmse'].std()
        ax1.axhline(y=mean_rmse, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rmse:.4f}')
        ax1.axhspan(mean_rmse - std_rmse, mean_rmse + std_rmse, alpha=0.2, color='red', label=f'±1σ: {std_rmse:.4f}')
        
        ax1.set_xlabel('Fold', fontsize=self.font_size, fontweight='bold')
        ax1.set_ylabel('Test RMSE (eV)', fontsize=self.font_size, fontweight='bold')
        ax1.set_title('RMSE Variance Across Folds', fontsize=self.font_size, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: R² variance
        ax2 = axes[0, 1]
        ax2.plot(df['fold'], df['r2'], 'go-', linewidth=2, markersize=8, label='R²')
        ax2.fill_between(df['fold'], df['r2'], alpha=0.3, color='green')
        
        mean_r2 = df['r2'].mean()
        std_r2 = df['r2'].std()
        ax2.axhline(y=mean_r2, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_r2:.4f}')
        ax2.axhspan(mean_r2 - std_r2, mean_r2 + std_r2, alpha=0.2, color='red', label=f'±1σ: {std_r2:.4f}')
        
        ax2.set_xlabel('Fold', fontsize=self.font_size, fontweight='bold')
        ax2.set_ylabel('Test R²', fontsize=self.font_size, fontweight='bold')
        ax2.set_title('R² Variance Across Folds', fontsize=self.font_size, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Plot 3: Stability metrics
        ax3 = axes[1, 0]
        cv_rmse = std_rmse / mean_rmse * 100  # Coefficient of variation
        cv_r2 = std_r2 / mean_r2 * 100 if mean_r2 > 0 else 0
        
        metrics = ['RMSE CV (%)', 'R² CV (%)']
        values = [cv_rmse, cv_r2]
        colors = ['red' if v > 10 else 'orange' if v > 5 else 'green' for v in values]
        
        bars = ax3.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Coefficient of Variation (%)', fontsize=self.font_size, fontweight='bold')
        ax3.set_title('Performance Stability', fontsize=self.font_size, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add threshold lines
        ax3.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Good (5%)')
        ax3.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Poor (10%)')
        ax3.legend()
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Statistical summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate additional statistics
        range_rmse = df['rmse'].max() - df['rmse'].min()
        range_r2 = df['r2'].max() - df['r2'].min()
        
        summary_text = f"""Statistical Summary:

RMSE Statistics:
• Mean: {mean_rmse:.4f} ± {std_rmse:.4f} eV
• Range: {range_rmse:.4f} eV
• CV: {cv_rmse:.1f}%
• Best Fold: {df.loc[df['rmse'].idxmin(), 'fold']} ({df['rmse'].min():.4f})
• Worst Fold: {df.loc[df['rmse'].idxmax(), 'fold']} ({df['rmse'].max():.4f})

R² Statistics:
• Mean: {mean_r2:.4f} ± {std_r2:.4f}
• Range: {range_r2:.4f}
• CV: {cv_r2:.1f}%
• Best Fold: {df.loc[df['r2'].idxmax(), 'fold']} ({df['r2'].max():.4f})
• Worst Fold: {df.loc[df['r2'].idxmin(), 'fold']} ({df['r2'].min():.4f})

Model Stability: {'Excellent' if cv_rmse < 5 else 'Good' if cv_rmse < 10 else 'Poor'}
"""
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=self.font_size,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if self.save_plot:
            save_path = os.path.join(self.output_dir, 'cv_stability_analysis.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            if self.verbose:
                print(f"✅ CV stability analysis saved to: {save_path}")
        
        if self.show_plot:
            plt.show()
        else:
            # Keep figure open for Spyder plots panel when show_plot=False
            pass
    
    def plot_training_info_publish(self,
                                   x_pos=-0.16,
                                   y_pos=1.08,
                                   loss_y_max=None,
                                   loss_y_min=None,
                                   train_loss_start_epoch=1,
                                   val_loss_start_epoch=1,
                                   gradient_y=None,
                                   gradient_start_epoch=1,
                                   show_best_epoch=False,
                                   convert_to_original_scale=True,
                                   plot_raw_training_loss=False,
                                   smooth_val_loss=False,
                                   smooth_window=5,
                                   ):
        """
        Create publication-ready training information plots for scientific journals
        
        Two subplots:
        - Left: Average training loss curves across all folds with confidence intervals
        - Right: Gradient norms during training across all folds
        
        Formatted according to scientific journal standards.
        Uses fold-specific scaler information to convert standardized loss values 
        back to original eV scale when available.
        
        Args:
            x_pos (float): x position for subplot labels (a) and (b)
            y_pos (float): y position for subplot labels (a) and (b)
            loss_y_max (float): Maximum y-axis value for loss plot (left subplot)
            loss_y_min (float): Minimum y-axis value for loss plot (left subplot)
            train_loss_start_epoch (int): Starting epoch for training loss curve (default: 1)
            val_loss_start_epoch (int): Starting epoch for validation loss curve (default: 1)
            gradient_y (float): Maximum y-axis value for gradient norm plot (right subplot)
            gradient_start_epoch (int): Starting epoch for gradient plot (default: 1)
            convert_to_original_scale (bool): Whether to convert loss values to original eV scale 
                                            using fold-specific scaler info (default: True)
            plot_raw_training_loss (bool): If True, plot raw MSE training loss (standardized scale);
                                         If False, plot RMSE in original eV scale (default: False)
            smooth_val_loss (bool): If True, apply smoothing to validation loss curve to reduce fluctuations (default: False)
            smooth_window (int): Window size for smoothing (default: 5)
        """
        if not self.training_data['folds']:
            print("❌ No training data available for publication plots")
            return
        
        # Check for available training data
        folds_with_curves = []
        folds_with_gradients = []
        
        for fold_idx, fold_data in self.training_data['folds'].items():
            if 'train_losses' in fold_data and len(fold_data.get('train_losses', [])) > 0:
                folds_with_curves.append(fold_idx)
            if 'gradient_norms' in fold_data and len(fold_data.get('gradient_norms', [])) > 0:
                folds_with_gradients.append(fold_idx)
        
        # Check if fold-specific scaler information is available
        fold_scalers = {}
        scalers_found = 0
        
        if convert_to_original_scale:
            for fold_idx, fold_data in self.training_data['folds'].items():
                if 'scaler_info' in fold_data and fold_data['scaler_info']:
                    scaler_info = fold_data['scaler_info']
                    if 'mean' in scaler_info and 'std' in scaler_info:
                        fold_scalers[fold_idx] = scaler_info
                        scalers_found += 1
            
            if scalers_found == 0:
                print("❌ No fold-specific scaler information found. Cannot convert to original scale.")
                convert_to_original_scale = False
            else:
                print(f"📊 Using fold-specific scaler information ({scalers_found} folds)")
        
        if not folds_with_curves and not folds_with_gradients:
            print("❌ No training curves or gradient data available for publication plots")
            return
        
        # Set publication-ready style
        plt.style.use('default')  # Clean default style
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Publication color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Left subplot: Average training loss with confidence intervals
        ax1 = axes[0]
        
        if folds_with_curves:
            # Calculate average training and validation loss curves
            # For each epoch, convert each fold's loss to eV scale first, then average
            max_epochs = max(len(self.training_data['folds'][fold]['train_losses']) 
                           for fold in folds_with_curves)
            
            avg_train_losses = []
            avg_val_losses = []
            std_train_losses = []
            std_val_losses = []
            
            # Loss conversion function with raw training loss option
            def convert_fold_loss_to_display(mse_loss, fold_idx):
                """Convert MSE loss for display based on plotting preferences"""
                if plot_raw_training_loss:
                    # Return raw MSE training loss (standardized scale)
                    return mse_loss
                else:
                    # Convert MSE to RMSE in original eV scale (default behavior)
                    rmse_standardized = np.sqrt(mse_loss)
                    if convert_to_original_scale and fold_idx in fold_scalers:
                        return rmse_standardized * fold_scalers[fold_idx]['std']
                    return rmse_standardized
            
            # Smoothing function for validation loss
            def smooth_curve(data, window_size):
                """Apply moving average smoothing to reduce fluctuations"""
                if len(data) < window_size:
                    return data
                smoothed = []
                for i in range(len(data)):
                    start_idx = max(0, i - window_size // 2)
                    end_idx = min(len(data), i + window_size // 2 + 1)
                    smoothed.append(np.mean(data[start_idx:end_idx]))
                return smoothed
            
            for epoch in range(max_epochs):
                train_values_eV = []
                val_values_eV = []
                
                for fold_idx in folds_with_curves:
                    fold_data = self.training_data['folds'][fold_idx]
                    
                    # Convert train loss for this fold
                    if epoch < len(fold_data['train_losses']):
                        raw_loss = fold_data['train_losses'][epoch]
                        converted_loss = convert_fold_loss_to_display(raw_loss, fold_idx)
                        train_values_eV.append(converted_loss)
                    
                    # Convert validation loss for this fold  
                    if epoch < len(fold_data.get('test_losses', [])):
                        raw_loss = fold_data['test_losses'][epoch]
                        converted_loss = convert_fold_loss_to_display(raw_loss, fold_idx)
                        val_values_eV.append(converted_loss)
                
                # Now calculate statistics on the converted values
                if train_values_eV:
                    avg_train_losses.append(np.mean(train_values_eV))
                    std_train_losses.append(np.std(train_values_eV))
                if val_values_eV:
                    avg_val_losses.append(np.mean(val_values_eV))
                    std_val_losses.append(np.std(val_values_eV))
            
            epochs = np.arange(1, len(avg_train_losses) + 1)
            
            # Print conversion status
            if plot_raw_training_loss:
                print(f"📊 Displaying raw MSE training loss (standardized scale)")
            elif convert_to_original_scale:
                print(f"🔄 Loss values converted to original eV scale (RMSE)")
            else:
                print(f"📊 Displaying RMSE loss (standardized scale)")
            
            # Apply separate filtering for train and validation losses
            # Prepare training loss data
            train_epochs = epochs.copy()
            train_avg_losses = avg_train_losses.copy()
            train_std_losses = std_train_losses.copy()
            
            if train_loss_start_epoch > 1:
                train_start_idx = train_loss_start_epoch - 1
                if train_start_idx < len(train_avg_losses):
                    train_epochs = epochs[train_start_idx:]
                    train_avg_losses = avg_train_losses[train_start_idx:]
                    train_std_losses = std_train_losses[train_start_idx:]
            
            # Prepare validation loss data
            val_epochs = epochs.copy()
            val_avg_losses = avg_val_losses.copy() if avg_val_losses else []
            val_std_losses = std_val_losses.copy() if std_val_losses else []
            
            # Apply smoothing to validation loss if requested
            if smooth_val_loss and val_avg_losses:
                val_avg_losses = smooth_curve(val_avg_losses, smooth_window)
                # Also smooth the std for consistency, but with smaller window
                val_std_losses = smooth_curve(val_std_losses, max(3, smooth_window // 2))
                if convert_to_original_scale:
                    print(f"🎯 Applied smoothing to validation loss (window={smooth_window})")
            
            if val_loss_start_epoch > 1 and val_avg_losses:
                val_start_idx = val_loss_start_epoch - 1
                if val_start_idx < len(val_avg_losses):
                    val_epochs = epochs[val_start_idx:]
                    val_avg_losses = val_avg_losses[val_start_idx:]
                    val_std_losses = val_std_losses[val_start_idx:]
            
            # Plot training loss with confidence interval (using filtered data)
            ax1.plot(train_epochs, train_avg_losses, color=colors[0], linewidth=2.5, 
                    label='Training Loss', alpha=0.9)
            ax1.fill_between(train_epochs, 
                           np.array(train_avg_losses) - np.array(train_std_losses),
                           np.array(train_avg_losses) + np.array(train_std_losses),
                           alpha=0.25, color=colors[0])
            
            # Plot validation loss if available (using separately filtered data)
            if val_avg_losses:
                ax1.plot(val_epochs, val_avg_losses, color=colors[1], linewidth=2.5, 
                        label='Validation Loss', alpha=0.9)
                ax1.fill_between(val_epochs, 
                               np.array(val_avg_losses) - np.array(val_std_losses),
                               np.array(val_avg_losses) + np.array(val_std_losses),
                               alpha=0.25, color=colors[1])
            
            # Find and mark best epoch (using filtered validation data)
            if show_best_epoch and val_avg_losses:
                best_idx = np.argmin(val_avg_losses)
                best_epoch = val_epochs[best_idx]  # Use filtered validation epochs
                best_loss = val_avg_losses[best_idx]
                ax1.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
                ax1.plot(best_epoch, best_loss, 'ro', markersize=8, alpha=0.8)
                ax1.text(best_epoch - (max(val_epochs) - min(val_epochs)) * 0.05,
                         best_loss + (max(val_avg_losses) - min(val_avg_losses)) * 0.2, 
                         f'Best\n(Epoch {best_epoch})', 
                         fontsize=self.font_size, ha='right', va='center',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax1.set_xlabel('Number of epochs', fontsize=self.font_size)
        
        # Set Y-axis label based on loss type
        if plot_raw_training_loss:
            ax1.set_ylabel('MSE Loss (standardized)', fontsize=self.font_size)
        elif convert_to_original_scale:
            ax1.set_ylabel('RMSE (eV)', fontsize=self.font_size)
        else:
            ax1.set_ylabel('RMSE (Standardized)', fontsize=self.font_size)
        # Add (a) label in the top-left corner
        ax1.text(x_pos, y_pos, '(a)', transform=ax1.transAxes, fontsize=self.font_size, 
                fontweight='bold', va='top', ha='left')
        ax1.legend(loc='upper right', fontsize=self.font_size, frameon=True, fancybox=True, shadow=True)
        # Remove grid for pure white background
        # ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax1.tick_params(labelsize=self.font_size)
        
        # Set y-axis limits for loss plot (using filtered data)
        auto_y_min = min(min(train_avg_losses) if train_avg_losses else [0], 
                        min(val_avg_losses) if val_avg_losses else [0]) * 0.9
        
        # Use provided limits or auto-calculated ones
        y_min_final = loss_y_min if loss_y_min is not None else max(0, auto_y_min)
        
        if loss_y_max is not None:
            ax1.set_ylim(bottom=y_min_final, top=loss_y_max)
        else:
            ax1.set_ylim(bottom=y_min_final)
        
        # Right subplot: Gradient norms during training
        ax2 = axes[1]
        
        if folds_with_gradients:
            # Calculate average gradient norm across all folds
            max_grad_epochs = max(len(self.training_data['folds'][fold]['gradient_norms']) 
                                for fold in folds_with_gradients)
            
            avg_grad_norms = []
            std_grad_norms = []
            
            for epoch in range(max_grad_epochs):
                grad_values = []
                for fold_idx in folds_with_gradients:
                    fold_data = self.training_data['folds'][fold_idx]
                    if epoch < len(fold_data['gradient_norms']):
                        grad_values.append(fold_data['gradient_norms'][epoch])
                
                if grad_values:
                    avg_grad_norms.append(np.mean(grad_values))
                    std_grad_norms.append(np.std(grad_values))
            
            if avg_grad_norms:
                grad_epochs = np.arange(1, len(avg_grad_norms) + 1)
                
                # Apply gradient_start_epoch filter to average data
                if gradient_start_epoch > 1:
                    start_idx = gradient_start_epoch - 1
                    if start_idx < len(avg_grad_norms):
                        grad_epochs = grad_epochs[start_idx:]
                        avg_grad_norms = avg_grad_norms[start_idx:]
                        std_grad_norms = std_grad_norms[start_idx:]
                
                # Plot average gradient norm with confidence interval
                ax2.plot(grad_epochs, avg_grad_norms, color=colors[2], linewidth=3, 
                        label='Average Gradient Norm', alpha=0.9)
                
                # Add shaded region for variability across folds
                ax2.fill_between(grad_epochs,
                               np.array(avg_grad_norms) - np.array(std_grad_norms),
                               np.array(avg_grad_norms) + np.array(std_grad_norms),
                               alpha=0.25, color=colors[2], label='±1σ across folds')
                
                # Remove horizontal line at mean gradient norm
                # if avg_grad_norms:  # Check if we still have data after filtering
                #     mean_grad = np.mean(avg_grad_norms)
                #     ax2.axhline(y=mean_grad, color='gray', linestyle=':', alpha=0.7, linewidth=2)
        
        ax2.set_xlabel('Number of epochs', fontsize=self.font_size)
        ax2.set_ylabel('Gradient Norm', fontsize=self.font_size)
        # Add (b) label in the top-left corner
        ax2.text(x_pos, y_pos, '(b)', transform=ax2.transAxes, fontsize=self.font_size, 
                fontweight='bold', va='top', ha='left')
        ax2.legend(loc='best', fontsize=self.font_size, frameon=True, fancybox=True, shadow=True)
        # Remove grid for pure white background
        # ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax2.tick_params(labelsize=self.font_size)
        
        # Set y-axis limits and scale for gradient plot
        if folds_with_gradients:
            all_grads = []
            for fold_idx in folds_with_gradients:
                grad_data = self.training_data['folds'][fold_idx]['gradient_norms']
                # Apply gradient_start_epoch filter to all_grads for scale determination
                if gradient_start_epoch > 1:
                    start_idx = gradient_start_epoch - 1
                    if start_idx < len(grad_data):
                        grad_data = grad_data[start_idx:]
                all_grads.extend(grad_data)
            
            # Set y-axis maximum if specified
            if gradient_y is not None:
                ax2.set_ylim(top=gradient_y)
            
            # Set logarithmic scale if gradient norms span large range (using filtered data)
            if all_grads and max(all_grads) / min(all_grads) > 100:  # Large range
                ax2.set_yscale('log')
                ax2.set_ylabel('Gradient Norm (log scale)', fontsize=self.font_size, fontweight='bold')
        
        # Overall figure formatting for publication
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3)  # Add space between subplots
        
        # Save the plot
        if self.save_plot:
            parts = self.model_name.split('-')
            key_info = '-'.join(parts[:3])
            filename =  f"training_info-f{key_info}.png"
            save_path = os.path.join(self.output_dir, filename)
            plt.savefig(save_path, dpi=1000, bbox_inches='tight', facecolor='white')
            
            if self.verbose:
                print(f"✅ Publication-ready training info plots saved to:")
                print(f"   PNG: {save_path}")
        
        if self.show_plot:
            plt.show()
        else:
            # Keep figure open for Spyder plots panel when show_plot=False
            pass
    
    
    def analyze_and_plot_all(self, pattern=None):
        """Complete analysis workflow with all common scientific plots"""
        print("\n" + "="*60)
        print("🔬 Complete Research-Level Training Analysis")
        print("="*60)
        
        # Use specified pattern or default
        if pattern is None:
            pattern = "cnn_2016514-epochs_100-bs_16-lr_0.0005-splits_5-grid_20.0_1.0"
        
        # Load and validate data
        if not self.load_data(pattern):
            return
        
        print(f"\n📊 Data Overview:")
        print(f"- Model file: {self.model_name}")
        print(f"- Cross-validation folds: {len(self.training_data['folds'])}")
        print(f"- Output directory: {self.output_dir}")
        
        # Check what data is available
        available_data = self._check_available_data()
        
        print(f"\n📈 Available data types:")
        for data_type, available in available_data.items():
            status = "✅" if available else "❌"
            print(f"{status} {data_type}")
        
        # Generate all available plots
        print(f"\n🎨 Generating scientific paper plots...")
        
        try:
            # 1. Basic performance metrics (always try)
            print("\n1️⃣ Generating performance metrics plot...")
            self.plot_performance_metrics()
            
            # 2. Training curves (if available)
            if available_data['Training Curves']:
                print("\n2️⃣ Generating training curves plot...")
                self.plot_loss_curves()
                
                print("\n3️⃣ Generating publication-ready plots...")
                self.plot_publication_ready_figures()
            else:
                print("\n⚠️ Training curve data not available - need to retrain model to get loss curves")
            
            # 3. Training dynamics (if available)
            if available_data['Training Dynamics']:
                print("\n4️⃣ Generating training dynamics analysis...")
                self.plot_training_dynamics()
            else:
                print("\n⚠️ Training dynamics data not available")
            
            print(f"\n✅ All available plots generated and saved to: {self.output_dir}")
            
        except Exception as e:
            print(f"\n❌ Error generating plots: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _check_available_data(self):
        """Check what types of data are available"""
        available = {
            'Performance Metrics': False,
            'Training Curves': False,
            'Training Dynamics': False,
            'Learning Rate Schedule': False,
            'Gradient Analysis': False
        }
        
        if not self.training_data['folds']:
            return available
        
        # Check basic performance metrics
        first_fold = list(self.training_data['folds'].values())[0]
        if any(key in first_fold for key in ['test_rmse', 'test_r2', 'train_rmse', 'train_r2']):
            available['Performance Metrics'] = True
        
        # Check training curves
        if any('train_losses' in fold_data and len(fold_data.get('train_losses', [])) > 0 
               for fold_data in self.training_data['folds'].values()):
            available['Training Curves'] = True
        
        # Check training dynamics
        if any(('overfitting_ratios' in fold_data and len(fold_data.get('overfitting_ratios', [])) > 0) or
               ('gradient_norms' in fold_data and len(fold_data.get('gradient_norms', [])) > 0)
               for fold_data in self.training_data['folds'].values()):
            available['Training Dynamics'] = True
        
        # Check learning rate data
        if any('learning_rates' in fold_data and len(fold_data.get('learning_rates', [])) > 0
               for fold_data in self.training_data['folds'].values()):
            available['Learning Rate Schedule'] = True
        
        # Check gradient data
        if any('gradient_norms' in fold_data and len(fold_data.get('gradient_norms', [])) > 0
               for fold_data in self.training_data['folds'].values()):
            available['Gradient Analysis'] = True
        
        return available
    
    def analyze_training_and_suggest_hyperparameters(self):
        """
        Comprehensive training analysis and hyperparameter optimization suggestions
        
        Analyzes:
        - Training/validation loss curves
        - Learning rate schedules
        - Early stopping patterns
        - Overfitting behavior
        - Convergence patterns
        
        Returns optimized suggestions for:
        - Learning rate schedule
        - Optimal epoch count
        - Early stopping strategy
        """
        if not self.training_data['folds']:
            print("❌ No training data available for analysis")
            return None
            
        print("\n" + "="*80)
        print("🔬 COMPREHENSIVE TRAINING ANALYSIS & HYPERPARAMETER OPTIMIZATION")
        print("="*80)
        
        analysis_results = {
            'convergence_analysis': {},
            'overfitting_analysis': {},
            'learning_rate_analysis': {},
            'early_stopping_analysis': {},
            'recommendations': {}
        }
        
        # 1. Convergence Analysis
        print("\n📈 1. CONVERGENCE ANALYSIS")
        print("-" * 40)
        
        convergence_info = self._analyze_convergence_patterns()
        analysis_results['convergence_analysis'] = convergence_info
        
        # 2. Overfitting Analysis  
        print("\n⚠️  2. OVERFITTING ANALYSIS")
        print("-" * 40)
        
        overfitting_info = self._analyze_overfitting_patterns()
        analysis_results['overfitting_analysis'] = overfitting_info
        
        # 3. Learning Rate Analysis
        print("\n📊 3. LEARNING RATE ANALYSIS") 
        print("-" * 40)
        
        lr_info = self._analyze_learning_rate_effectiveness()
        analysis_results['learning_rate_analysis'] = lr_info
        
        # 4. Early Stopping Analysis
        print("\n🛑 4. EARLY STOPPING ANALYSIS")
        print("-" * 40)
        
        early_stop_info = self._analyze_early_stopping_effectiveness()
        analysis_results['early_stopping_analysis'] = early_stop_info
        
        # 5. Generate Comprehensive Recommendations
        print("\n💡 5. OPTIMIZATION RECOMMENDATIONS")
        print("-" * 40)
        
        recommendations = self._generate_hyperparameter_recommendations(
            convergence_info, overfitting_info, lr_info, early_stop_info
        )
        analysis_results['recommendations'] = recommendations
        
        # 6. Print Summary
        self._print_optimization_summary(recommendations)
        
        return analysis_results
    
    def _analyze_convergence_patterns(self):
        """Analyze convergence patterns across folds"""
        convergence_info = {
            'avg_epochs_to_best': 0,
            'convergence_stability': 0,
            'plateau_detection': [],
            'convergence_speed': 'unknown'
        }
        
        epochs_to_best = []
        plateau_lengths = []
        loss_improvements = []
        
        for fold_idx, fold_data in self.training_data['folds'].items():
            if 'test_losses' in fold_data and len(fold_data['test_losses']) > 5:
                test_losses = np.array(fold_data['test_losses'])
                
                # Find best epoch
                best_epoch = np.argmin(test_losses)
                epochs_to_best.append(best_epoch)
                
                # Analyze convergence speed (loss improvement rate)
                if len(test_losses) > 10:
                    early_loss = np.mean(test_losses[:5])
                    mid_loss = np.mean(test_losses[best_epoch-5:best_epoch+5]) if best_epoch > 5 else test_losses[best_epoch]
                    improvement_rate = (early_loss - mid_loss) / early_loss
                    loss_improvements.append(improvement_rate)
                
                # Detect plateaus (consecutive epochs with minimal improvement)
                plateau_length = 0
                for i in range(1, len(test_losses)):
                    if abs(test_losses[i] - test_losses[i-1]) / test_losses[i-1] < 0.001:
                        plateau_length += 1
                    else:
                        if plateau_length > 5:
                            plateau_lengths.append(plateau_length)
                        plateau_length = 0
        
        if epochs_to_best:
            convergence_info['avg_epochs_to_best'] = np.mean(epochs_to_best)
            convergence_info['convergence_stability'] = np.std(epochs_to_best)
            
            # Determine convergence speed
            if loss_improvements:
                avg_improvement = np.mean(loss_improvements)
                if avg_improvement > 0.3:
                    convergence_info['convergence_speed'] = 'fast'
                elif avg_improvement > 0.1:
                    convergence_info['convergence_speed'] = 'moderate'
                else:
                    convergence_info['convergence_speed'] = 'slow'
        
        convergence_info['plateau_detection'] = plateau_lengths
        
        print(f"✅ Average epochs to best performance: {convergence_info['avg_epochs_to_best']:.1f}")
        print(f"✅ Convergence stability (std): {convergence_info['convergence_stability']:.1f}")
        print(f"✅ Convergence speed: {convergence_info['convergence_speed']}")
        if plateau_lengths:
            print(f"⚠️  Detected plateaus of length: {plateau_lengths}")
        
        return convergence_info
    
    
    def _analyze_overfitting_patterns(self):
        """Analyze overfitting patterns and severity"""
        overfitting_info = {
            'overfitting_severity': 'none',
            'overfitting_onset': 0,
            'train_val_gap': 0,
            'overfitting_stability': 0
        }
        
        train_val_gaps = []
        overfitting_onsets = []
        final_gaps = []
        
        for fold_idx, fold_data in self.training_data['folds'].items():
            if 'train_losses' in fold_data and 'test_losses' in fold_data:
                train_losses = np.array(fold_data['train_losses'])
                test_losses = np.array(fold_data['test_losses'])
                
                if len(train_losses) == len(test_losses) and len(train_losses) > 10:
                    # Calculate train-validation gap over time
                    gaps = test_losses - train_losses
                    train_val_gaps.extend(gaps)
                    final_gaps.append(gaps[-1])
                    
                    # Find overfitting onset (when gap starts increasing consistently)
                    gap_increases = 0
                    onset_epoch = len(gaps)
                    for i in range(5, len(gaps)):
                        if gaps[i] > gaps[i-1]:
                            gap_increases += 1
                        else:
                            gap_increases = 0
                        
                        if gap_increases >= 3 and onset_epoch == len(gaps):  # 3 consecutive increases
                            onset_epoch = i - 2
                            break
                    
                    overfitting_onsets.append(onset_epoch)
        
        if train_val_gaps:
            avg_gap = np.mean(final_gaps)
            overfitting_info['train_val_gap'] = avg_gap
            overfitting_info['overfitting_stability'] = np.std(final_gaps)
            
            # Determine overfitting severity
            if avg_gap > 0.5:
                overfitting_info['overfitting_severity'] = 'severe'
            elif avg_gap > 0.2:
                overfitting_info['overfitting_severity'] = 'moderate'
            elif avg_gap > 0.05:
                overfitting_info['overfitting_severity'] = 'mild'
            else:
                overfitting_info['overfitting_severity'] = 'none'
        
        if overfitting_onsets:
            overfitting_info['overfitting_onset'] = np.mean(overfitting_onsets)
        
        print(f"📊 Overfitting severity: {overfitting_info['overfitting_severity']}")
        print(f"📊 Average train-val gap: {overfitting_info['train_val_gap']:.4f}")
        print(f"📊 Overfitting onset (avg epoch): {overfitting_info['overfitting_onset']:.1f}")
        
        return overfitting_info
    
    def _analyze_learning_rate_effectiveness(self):
        """Analyze learning rate schedule effectiveness"""
        lr_info = {
            'lr_schedule_type': 'unknown',
            'lr_reduction_points': [],
            'lr_effectiveness': 'unknown',
            'optimal_initial_lr': None
        }
        
        # Analyze learning rate patterns
        for fold_idx, fold_data in self.training_data['folds'].items():
            if 'learning_rates' in fold_data and len(fold_data['learning_rates']) > 1:
                lr_schedule = np.array(fold_data['learning_rates'])
                
                # Detect schedule type
                if len(set(lr_schedule)) == 1:
                    lr_info['lr_schedule_type'] = 'constant'
                elif np.all(lr_schedule[1:] <= lr_schedule[:-1]):
                    lr_info['lr_schedule_type'] = 'decreasing'
                else:
                    lr_info['lr_schedule_type'] = 'variable'
                
                # Find LR reduction points
                reductions = []
                for i in range(1, len(lr_schedule)):
                    if lr_schedule[i] < lr_schedule[i-1] * 0.9:  # 10% reduction threshold
                        reductions.append(i)
                lr_info['lr_reduction_points'] = reductions
                
                # Estimate effectiveness based on loss improvement after LR reductions
                if 'test_losses' in fold_data and len(fold_data['test_losses']) == len(lr_schedule):
                    test_losses = np.array(fold_data['test_losses'])
                    effectiveness_scores = []
                    
                    for reduction_point in reductions:
                        if reduction_point < len(test_losses) - 5:
                            before_loss = np.mean(test_losses[max(0, reduction_point-3):reduction_point])
                            after_loss = np.mean(test_losses[reduction_point:reduction_point+3])
                            if before_loss > 0:
                                improvement = (before_loss - after_loss) / before_loss
                                effectiveness_scores.append(improvement)
                    
                    if effectiveness_scores:
                        avg_effectiveness = np.mean(effectiveness_scores)
                        if avg_effectiveness > 0.05:
                            lr_info['lr_effectiveness'] = 'high'
                        elif avg_effectiveness > 0.01:
                            lr_info['lr_effectiveness'] = 'moderate'
                        else:
                            lr_info['lr_effectiveness'] = 'low'
                
                break  # Use first fold with LR data
        
        print(f"📈 Learning rate schedule: {lr_info['lr_schedule_type']}")
        print(f"📈 LR reduction effectiveness: {lr_info['lr_effectiveness']}")
        if lr_info['lr_reduction_points']:
            print(f"📈 LR reductions at epochs: {lr_info['lr_reduction_points']}")
        
        return lr_info
    
    def _analyze_early_stopping_effectiveness(self):
        """Analyze early stopping effectiveness"""
        early_stop_info = {
            'early_stop_used': False,
            'optimal_stopping_epoch': 0,
            'patience_needed': 0,
            'performance_after_best': 0
        }
        
        actual_epochs = []
        best_epochs = []
        performance_degradations = []
        
        for fold_idx, fold_data in self.training_data['folds'].items():
            if 'test_losses' in fold_data and len(fold_data['test_losses']) > 0:
                test_losses = np.array(fold_data['test_losses'])
                actual_epochs.append(len(test_losses))
                
                best_epoch = np.argmin(test_losses)
                best_epochs.append(best_epoch)
                
                # Calculate performance degradation after best epoch
                if best_epoch < len(test_losses) - 1:
                    best_loss = test_losses[best_epoch]
                    final_loss = test_losses[-1]
                    degradation = (final_loss - best_loss) / best_loss
                    performance_degradations.append(degradation)
        
        if best_epochs and actual_epochs:
            avg_best_epoch = np.mean(best_epochs)
            avg_actual_epochs = np.mean(actual_epochs)
            
            early_stop_info['early_stop_used'] = avg_actual_epochs < 90  # Assuming max epochs was ~100
            early_stop_info['optimal_stopping_epoch'] = avg_best_epoch
            early_stop_info['patience_needed'] = max(5, int(avg_best_epoch * 0.2))  # 20% of convergence time
            
            if performance_degradations:
                early_stop_info['performance_after_best'] = np.mean(performance_degradations)
        
        print(f"🛑 Early stopping used: {early_stop_info['early_stop_used']}")
        print(f"🛑 Optimal stopping epoch: {early_stop_info['optimal_stopping_epoch']:.1f}")
        print(f"🛑 Recommended patience: {early_stop_info['patience_needed']}")
        print(f"🛑 Performance degradation after best: {early_stop_info['performance_after_best']:.3f}")
        
        return early_stop_info
    
    def _generate_hyperparameter_recommendations(self, convergence_info, overfitting_info, lr_info, early_stop_info):
        """Generate comprehensive hyperparameter recommendations"""
        recommendations = {
            'learning_rate': {
                'initial_lr': 0.001,
                'schedule': 'reduce_on_plateau',
                'schedule_params': {}
            },
            'epochs': {
                'max_epochs': 100,
                'expected_convergence': 50
            },
            'early_stopping': {
                'use_early_stopping': True,
                'patience': 10,
                'min_delta': 1e-4
            },
            'regularization': {
                'recommendations': []
            }
        }
        
        # Learning Rate Recommendations
        if convergence_info['convergence_speed'] == 'slow':
            recommendations['learning_rate']['initial_lr'] = 0.005  # Higher initial LR
            print("💡 Slow convergence detected → Recommend higher initial learning rate (0.005)")
        elif convergence_info['convergence_speed'] == 'fast':
            recommendations['learning_rate']['initial_lr'] = 0.0005  # Lower initial LR
            print("💡 Fast convergence detected → Recommend lower initial learning rate (0.0005)")
        else:
            recommendations['learning_rate']['initial_lr'] = 0.001
            print("💡 Moderate convergence → Keep current learning rate (0.001)")
        
        # Learning Rate Schedule
        if lr_info['lr_effectiveness'] == 'high':
            recommendations['learning_rate']['schedule'] = 'step_decay'
            recommendations['learning_rate']['schedule_params'] = {
                'step_size': max(20, int(convergence_info['avg_epochs_to_best'] * 0.6)),
                'gamma': 0.5
            }
            print(f"💡 Effective LR reductions → Recommend step decay every {recommendations['learning_rate']['schedule_params']['step_size']} epochs")
        elif lr_info['lr_effectiveness'] == 'moderate':
            recommendations['learning_rate']['schedule'] = 'reduce_on_plateau'
            recommendations['learning_rate']['schedule_params'] = {
                'patience': max(10, int(convergence_info['avg_epochs_to_best'] * 0.3)),
                'factor': 0.7
            }
            print(f"💡 Moderate LR effectiveness → Recommend ReduceLROnPlateau with patience={recommendations['learning_rate']['schedule_params']['patience']}")
        else:
            recommendations['learning_rate']['schedule'] = 'cosine_annealing'
            print("💡 Low LR effectiveness → Recommend cosine annealing schedule")
        
        # Epoch Recommendations
        optimal_epochs = int(convergence_info['avg_epochs_to_best'] * 1.5)  # 50% buffer after convergence
        if overfitting_info['overfitting_severity'] in ['moderate', 'severe']:
            optimal_epochs = min(optimal_epochs, int(overfitting_info['overfitting_onset'] * 1.2))
        
        recommendations['epochs']['max_epochs'] = max(50, min(150, optimal_epochs))
        recommendations['epochs']['expected_convergence'] = int(convergence_info['avg_epochs_to_best'])
        
        print(f"💡 Recommend max epochs: {recommendations['epochs']['max_epochs']}")
        print(f"💡 Expected convergence: {recommendations['epochs']['expected_convergence']}")
        
        # Early Stopping Recommendations
        if overfitting_info['overfitting_severity'] != 'none':
            recommendations['early_stopping']['use_early_stopping'] = True
            recommendations['early_stopping']['patience'] = max(5, int(convergence_info['avg_epochs_to_best'] * 0.2))
            recommendations['early_stopping']['min_delta'] = 1e-4
            print(f"💡 Overfitting detected → Enable early stopping with patience={recommendations['early_stopping']['patience']}")
        else:
            recommendations['early_stopping']['patience'] = max(15, int(convergence_info['avg_epochs_to_best'] * 0.3))
            print(f"💡 No overfitting → Use longer patience={recommendations['early_stopping']['patience']}")
        
        # Regularization Recommendations
        if overfitting_info['overfitting_severity'] == 'severe':
            recommendations['regularization']['recommendations'].extend([
                'Increase dropout rate to 0.3-0.5',
                'Add L2 regularization (weight_decay=1e-4)',
                'Consider reducing model complexity'
            ])
        elif overfitting_info['overfitting_severity'] == 'moderate':
            recommendations['regularization']['recommendations'].extend([
                'Add dropout (0.2-0.3)',
                'Consider L2 regularization (weight_decay=5e-5)'
            ])
        
        return recommendations
    
    def _print_optimization_summary(self, recommendations):
        """Print a comprehensive optimization summary"""
        print("\n" + "="*80)
        print("🎯 FINAL OPTIMIZATION RECOMMENDATIONS")
        print("="*80)
        
        print("\n🔧 HYPERPARAMETER SETTINGS:")
        print(f"├─ Initial Learning Rate: {recommendations['learning_rate']['initial_lr']}")
        print(f"├─ LR Schedule: {recommendations['learning_rate']['schedule']}")
        if recommendations['learning_rate']['schedule_params']:
            for key, value in recommendations['learning_rate']['schedule_params'].items():
                print(f"│  └─ {key}: {value}")
        
        print(f"├─ Max Epochs: {recommendations['epochs']['max_epochs']}")
        print(f"├─ Expected Convergence: ~{recommendations['epochs']['expected_convergence']} epochs")
        
        print(f"├─ Early Stopping: {'Enabled' if recommendations['early_stopping']['use_early_stopping'] else 'Disabled'}")
        print(f"│  ├─ Patience: {recommendations['early_stopping']['patience']}")
        print(f"│  └─ Min Delta: {recommendations['early_stopping']['min_delta']}")
        
        if recommendations['regularization']['recommendations']:
            print("└─ Regularization:")
            for rec in recommendations['regularization']['recommendations']:
                print(f"   └─ {rec}")
        
        print("\n📋 PYTORCH IMPLEMENTATION EXAMPLE:")
        print("-" * 50)
        print(f"""
# Optimizer setup
optimizer = torch.optim.Adam(model.parameters(), 
                           lr={recommendations['learning_rate']['initial_lr']})

# Learning rate scheduler
""", end="")
        
        if recommendations['learning_rate']['schedule'] == 'step_decay':
            params = recommendations['learning_rate']['schedule_params']
            print(f"""scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size={params.get('step_size', 30)}, gamma={params.get('gamma', 0.5)}
)""")
        elif recommendations['learning_rate']['schedule'] == 'reduce_on_plateau':
            params = recommendations['learning_rate']['schedule_params']
            print(f"""scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience={params.get('patience', 10)}, factor={params.get('factor', 0.7)}
)""")
        elif recommendations['learning_rate']['schedule'] == 'cosine_annealing':
            print(f"""scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max={recommendations['epochs']['max_epochs']}
)""")
        
        print(f"""
# Training loop settings
max_epochs = {recommendations['epochs']['max_epochs']}
early_stopping_patience = {recommendations['early_stopping']['patience']}
min_delta = {recommendations['early_stopping']['min_delta']}
""")
        
        print("\n🎯 EXPECTED IMPROVEMENTS:")
        print("├─ Better convergence stability")
        print("├─ Reduced overfitting risk") 
        print("├─ More efficient training time")
        print("└─ Improved generalization performance")
        
        print("\n" + "="*80)


# Example usage and additional utility functions
def plot_model_comparison(model_files, output_dir=None, font_size=18):
    """Compare multiple models side by side"""
    if not isinstance(model_files, list):
        model_files = [model_files]
    
    # Set default output directory to cnn_training_results folder
    if output_dir is None:
        output_dir = os.path.join(get_paths("output_figure_path"), "cnn_training_results")
    
    print(f"\n🔬 Comparing {len(model_files)} model performance")
    
    comparison_data = []
    for model_file in model_files:
        plotter = EnhancedTrainingPlotter(model_file, output_dir=output_dir, verbose=False, save_plot=False)
        if plotter.load_data():
            # Extract summary statistics
            rmse_values = []
            r2_values = []
            
            for fold_data in plotter.training_data['folds'].values():
                if 'test_rmse' in fold_data:
                    rmse_values.append(fold_data['test_rmse'])
                if 'test_r2' in fold_data:
                    r2_values.append(fold_data['test_r2'])
            
            if rmse_values and r2_values:
                comparison_data.append({
                    'model': os.path.basename(model_file).replace('.pkl', '').replace('.pth', ''),
                    'mean_rmse': np.mean(rmse_values),
                    'std_rmse': np.std(rmse_values),
                    'mean_r2': np.mean(r2_values),
                    'std_r2': np.std(r2_values),
                    'n_folds': len(rmse_values)
                })
    
    if not comparison_data:
        print("❌ No comparable model data found")
        return
    
    # Create comparison plot
    df = pd.DataFrame(comparison_data)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Model Performance Comparison', fontsize=font_size, fontweight='bold')
    
    # RMSE comparison
    ax1 = axes[0]
    ax1.errorbar(range(len(df)), df['mean_rmse'], yerr=df['std_rmse'], 
                fmt='o-', linewidth=2, markersize=8, capsize=5, capthick=2)
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df['model'], rotation=45, ha='right', fontsize=font_size)
    ax1.set_ylabel('Test RMSE (eV)', fontsize=font_size, fontweight='bold')
    ax1.set_title('Root Mean Square Error Comparison', fontsize=font_size, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # R² comparison
    ax2 = axes[1]
    ax2.errorbar(range(len(df)), df['mean_r2'], yerr=df['std_r2'], 
                fmt='o-', linewidth=2, markersize=8, capsize=5, capthick=2, color='green')
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(df['model'], rotation=45, ha='right', fontsize=font_size)
    ax2.set_ylabel('Test R² Score', fontsize=font_size, fontweight='bold')
    ax2.set_title('R² Score Comparison', fontsize=font_size, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save comparison plot
    save_path = os.path.join(output_dir, 'model_comparison.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()  # Don't show by default for comparison function
    
    print(f"✅ Model comparison plot saved to: {save_path}")
    
    # Print comparison table
    print(f"\n📊 Model Performance Comparison Table:")
    print(df.to_string(index=False, float_format='%.4f'))




    
if __name__ == "__main__":
    
    # Test with the specified model file
    # model_filename = "model-random-2546193-epochs_200-bs_32-lr_0.0001-grid_16.0_0.8.pkl"
    # model_filename = "model-random-2546194-epochs_200-bs_32-lr_0.0001-grid_16.0_0.8.pkl"
    # model_filename = "model-random-2546195-epochs_200-bs_32-lr_0.0001-grid_16.0_0.8.pkl"
    # model_filename = "model-random-2546197-epochs_200-bs_32-lr_0.0001-grid_16.0_0.8.pkl"
    # model_filename = "model-random-2546199-epochs_200-bs_32-lr_0.0001-grid_16.0_0.8.pkl"
    # model_filename = "model-random-2546201-epochs_200-bs_32-lr_0.0001-grid_16.0_0.8.pkl"
    # model_filename = "model-random-2546213-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Good
    # model_filename = "model-random-2546214-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl"
    # model_filename = "model-random-2546215-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl"
    # model_filename = "model-random-2546216-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl"
    # model_filename = "model-random-2546217-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Good
    # model_filename = "model-random-2546218-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl"
    # model_filename = "model-random-2546220-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl"
    # model_filename = "model-random-2546223-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl"
    # model_filename = "model-random-2546226-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl"
    # model_filename = "model-random-2546227-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Very good  # 0.1166
    # model_filename = "model-random-2546228-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl"
    # model_filename = "model-random-2546229-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl"
    # model_filename = "model-random-2546238-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Very good # 0.1140
    # model_filename = "model-random-2546239-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl"
    model_filename = "model-random-2546240-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Very good # 0.1150
    # model_filename = "model-random-2546241-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Very good # 0.1181
    # model_filename = "model-random-2546243-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl"
    # model_filename = "model-random-2546244-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl"
    # model_filename = "model-random-2546246-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl"
    # model_filename = "model-random-2546247-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl"

    
    # Create plotter instance with specific model file (auto-loads data)
    plotter = EnhancedTrainingPlotter(model_file=model_filename,
                                      verbose=True,
                                      font_size=24,
                                      show_plot=False,
                                      save_plot=True)
    
    # # Performance metrics comparison
    # print("\n1️⃣ Generating performance metrics plot...")
    # plotter.plot_performance_metrics()
    
    # # Training loss curves
    # print("\n2️⃣ Generating training loss curves...")
    # plotter.plot_loss_curves()
    
    # # Training dynamics analysis
    # print("\n3️⃣ Generating training dynamics analysis...")
    # plotter.plot_training_dynamics()
    
    # # Early stopping analysis
    # print("\n6️⃣ Generating early stopping analysis...")
    # plotter.plot_early_stopping_analysis()
    
    # # Cross-validation stability analysis
    # print("\n7️⃣ Generating cross-validation stability analysis...")
    # plotter.plot_cv_stability()
    
    # print("\n8️⃣ Comprehensive training analysis and hyperparameter suggestions...")
    # plotter.analyze_training_and_suggest_hyperparameters()
    
    # Generate publication-ready plots
    print("\n9️⃣ Generating publication-ready training info plots...")
    plotter.plot_training_info_publish(loss_y_max=1.5,
                                       loss_y_min=-0.05,
                                       train_loss_start_epoch=0,
                                       val_loss_start_epoch=8,
                                       gradient_y=3,
                                       gradient_start_epoch=2,
                                       show_best_epoch=False,
                                       plot_raw_training_loss=True,
                                       convert_to_original_scale=False,
                                       smooth_val_loss=True,
                                       smooth_window=5,
                                       )