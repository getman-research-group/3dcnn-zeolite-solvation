# -*- coding: utf-8 -*-
"""
extract_3d_cnn.py
The purpose of this script is to extract and visualize results from trained 3D CNN models.
Features:
- Load and parse pkl files containing training results
- Extract metrics from all cross-validation folds
- Generate comprehensive visualization plots
- Compare different model configurations
- Export results to various formats
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Import custom functions
from core.path import get_paths


class CNN3DResultsExtractor:
    """
    A comprehensive class for extracting and visualizing 3D CNN training results
    """
    
    def __init__(self, pkl_name: str = None, verbose: bool = True, 
                 export_csv: bool = False,
                 font_size: int = 24,
                 blue_color: str = '#488cca',
                 red_color: str = '#ff0000',
                 green_color: str = '#009E73',
                 alpha: float = 0.6,
                 alpha_train: float = 0.6,
                 point_size: int = 20,
                 point_size_deltaEint: int = 20):
        """
        Initialize the results extractor
        
        Args:
            pkl_name: Name of the pkl file to load (e.g., 'epochs_1-bs_8-lr_0.001-splits_5-grid_20.0_1.0-test.pkl')
            verbose: Whether to print detailed information
            export_csv: Whether to export results to CSV
            font_size: Font size for all plot elements (default: 24)
            blue_color: Color for training data points (default: '#488cca')
            red_color: Color for test data points (default: '#ff0000')
            green_color: Color for MD data points (default: '#009E73')
            alpha: Transparency level for test and MD data points (default: 0.6)
            alpha_train: Transparency level for training data points (default: 0.6)
            point_size: Size of scatter plot points for deltaEsol plots (default: 20)
            point_size_deltaEint: Size of scatter plot points for deltaEint plots (default: 20)
        """
        self.verbose = verbose
        self.model_dir = get_paths("output_model_cnn")
        self.pkl_name = pkl_name
        self.export_csv = export_csv
        
        # Plotting parameters
        self.font_size = font_size
        self.blue_color = blue_color
        self.red_color = red_color
        self.green_color = green_color
        self.alpha = alpha
        self.alpha_train = alpha_train
        self.point_size = point_size
        self.point_size_deltaEint = point_size_deltaEint
        
        # Data storage
        self.results_data = None
        self.model_storage = None
        self.data_info = None
        self.training_config = None
        self.summary_stats = None
        self.fold_results = None
        self.md_data = None  # Store MD intE data for comparison
        
        # Extract split type from pkl name for file naming
        self.split_type = self._extract_split_type(pkl_name) if pkl_name else 'unknown'
        
        # Save path for plots
        self.output_figure_path = get_paths("output_figure_path")
        
        # Plotting configuration
        plt.style.use('default')
        sns.set_palette("husl")
        
        if self.verbose:
            print(f"\n=== CNN3DResultsExtractor initialized ===")
            print(f"    Model directory: {self.model_dir}")
            if pkl_name:
                print(f"    Target pkl file: {pkl_name}")
        
        # Auto-load if pkl_name is provided
        if pkl_name:
            success = self.load_results(pkl_name)
            
            # Check if results were loaded successfully and execute based on settings
            if not success or self.results_data is None:
                print("\nAvailable pkl files:")
                self.list_available_pkl_files()
                print("\nPlease check the pkl_name and try again.")
            else:
                # Only show basic summary, don't plot automatically
                self.print_summary()
                
                # Export to CSV if requested
                if self.export_csv:
                    self.export_results_to_csv()
    
    
    def list_available_pkl_files(self) -> List[str]:
        """List all available pkl files in the model directory"""
        pkl_files = []
        if os.path.exists(self.model_dir):
            for file in os.listdir(self.model_dir):
                if file.endswith('.pkl'):
                    pkl_files.append(file)
        
        if self.verbose:
            print(f"\n--- Available pkl files ({len(pkl_files)}):")
            for i, file in enumerate(pkl_files, 1):
                print(f"    {i}. {file}")
        
        return sorted(pkl_files)
    
    
    def _extract_split_type(self, pkl_name):
        """Extract split type from pkl filename"""
        if not pkl_name:
            return 'random'
        
        # Expected format: "model-{split_type}-{job_id}-epochs_100-bs_32-lr_0.0002-grid_16.0_0.8.pkl"
        # Or old format: "model-{job_id}-epochs_100-bs_32-lr_0.0002-grid_16.0_0.8.pkl"
        pkl_parts = pkl_name.split('-')
        if len(pkl_parts) >= 2:
            potential_split_type = pkl_parts[1]
            if potential_split_type in ['random', 'solvent', 'pore_type']:
                return potential_split_type
        
        # If not explicitly specified, assume it's random split (default/old format)
        return 'random'
    
    
    def load_results(self, pkl_name: str) -> bool:
        """
        Load results from a pkl file
        
        Args:
            pkl_name: Name of the pkl file to load
            
        Returns:
            bool: True if successful, False otherwise
        """
        pkl_path = os.path.join(self.model_dir, pkl_name)
        
        if not os.path.exists(pkl_path):
            print(f"Error: File not found: {pkl_path}")
            return False
        
        try:
            with open(pkl_path, 'rb') as f:
                # Load with CPU mapping to handle GPU-trained models on CPU machines
                import torch
                if torch.cuda.is_available():
                    self.results_data = pickle.load(f)
                else:
                    # For CPU-only machines, we need to handle torch objects specially
                    original_load = torch.load
                    torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, map_location='cpu')
                    try:
                        self.results_data = pickle.load(f)
                    finally:
                        torch.load = original_load
            
            # Extract components
            self.model_storage = self.results_data.get('model_storage', {})
            self.data_info = self.results_data.get('data_info', {})
            self.training_config = self.results_data.get('training_config', {})
            self.pkl_name = pkl_name
            
            # Update split type in case it wasn't set during init
            self.split_type = self._extract_split_type(pkl_name)
            
            # Process results
            self.process_results()
            
            if self.verbose:
                print(f"\n--- Results loaded successfully from: {pkl_name}")
                print(f"    Number of folds: {len(self.model_storage)}")
                print(f"    Data points: {self.data_info.get('num_total_datapoints', 'N/A')}")
                print(f"    Adsorbates: {self.data_info.get('num_adsorbates', 'N/A')}")
            
            return True
            
        except Exception as e:
            print(f"Error loading pkl file: {e}")
            return False
    
    
    def load_md_data(self):
        """
        Load MD intE data from CSV files for comparison in snapshot-averaged plots
        
        Returns:
            pandas.DataFrame: Combined MD data with intE_MD values
        """
        database_path = get_paths("database_path")
        
        # Define the environments that might have data (similar to load_grids_pickle.py)
        # We'll try to load all available CSV files in the database
        csv_files = []
        if os.path.exists(database_path):
            for file in os.listdir(database_path):
                if file.endswith('.csv') and not file.startswith('.'):
                    csv_files.append(file)
        
        if not csv_files:
            if self.verbose:
                print("Warning: No CSV files found in database directory for MD data")
            return None
        
        combined_md_data = []
        
        for csv_filename in csv_files:
            csv_path = os.path.join(database_path, csv_filename)
            try:
                df = pd.read_csv(csv_path)
                
                # Check if intE_MD column exists
                if 'intE_MD' not in df.columns:
                    continue
                
                # For each adsorbate, we only need one row since intE_MD is the same for all snapshots
                # Take the first snapshot of each adsorbate to get the intE_MD value
                md_subset = df.groupby(['zeolite', 'environment', 'adsorbate']).first().reset_index()
                md_subset = md_subset[['zeolite', 'environment', 'adsorbate', 'intE_MD']]
                
                combined_md_data.append(md_subset)
                    
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Error loading {csv_filename}: {e}")
                continue
        
        if not combined_md_data:
            if self.verbose:
                print("Warning: No valid MD data found in CSV files")
            return None
        
        # Combine all MD data
        self.md_data = pd.concat(combined_md_data, ignore_index=True)
        
        if self.verbose:
            print(f"    Loaded MD data: {len(self.md_data)} adsorbate combinations")
        
        return self.md_data
    
    
    def process_results(self):
        """Process and organize the loaded results, including voxel and snapshot averaging"""
        if not self.model_storage:
            return
        
        # Create fold results dataframe
        fold_data = []
        for fold_idx, fold_results in self.model_storage.items():
            fold_data.append({
                'fold': fold_idx + 1,  # 1-indexed for display
                'train_rmse': fold_results.get('train_rmse', np.nan),
                'train_r2': fold_results.get('train_r2', np.nan),
                'train_mae': fold_results.get('train_mae', np.nan),
                'test_rmse': fold_results.get('test_rmse', np.nan),
                'test_r2': fold_results.get('test_r2', np.nan),
                'test_mae': fold_results.get('test_mae', np.nan)
            })
        
        self.fold_results = pd.DataFrame(fold_data)
        
        # Add voxel and snapshot averaging to all dataframes
        self.add_averaged_predictions()
        
        # Load MD data for comparison
        self.load_md_data()
        
        # Calculate summary statistics (including averaged metrics)
        self.calculate_summary_stats()
    
    
    def add_averaged_predictions(self):
        """Add voxel-averaged and snapshot-averaged prediction columns to all fold dataframes"""
        if not self.model_storage:
            return
        
        if self.verbose:
            print(f"\n--- Adding averaged prediction columns to dataframes...")
        
        # Combine all fold results for global averaging calculations
        all_train_data = []
        all_test_data = []
        
        for fold_idx, fold_results in self.model_storage.items():
            if 'df_train' in fold_results and 'df_test' in fold_results:
                df_train = fold_results['df_train']
                df_test = fold_results['df_test']
                
                # Check if required columns exist
                required_cols = ['zeolite', 'environment', 'adsorbate', 'snapshot', 'y_true', 'y_pred']
                if all(col in df_train.columns for col in required_cols):
                    all_train_data.append(df_train)
                    all_test_data.append(df_test)
        
        if not all_train_data or not all_test_data:
            if self.verbose:
                print("    Warning: Required columns not found, skipping averaged predictions")
            return
        
        # Combine all folds
        combined_train = pd.concat(all_train_data, ignore_index=True)
        combined_test = pd.concat(all_test_data, ignore_index=True)
        
        # Calculate voxel averages (24 voxel grids per snapshot)
        train_voxel_avg = combined_train.groupby(['zeolite', 'environment', 'adsorbate', 'snapshot', 'fold']).agg({
            'y_true': 'mean',  # Should be the same for all 24 voxels
            'y_pred': 'mean'   # Average the 24 voxel predictions
        }).reset_index()
        
        test_voxel_avg = combined_test.groupby(['zeolite', 'environment', 'adsorbate', 'snapshot', 'fold']).agg({
            'y_true': 'mean',
            'y_pred': 'mean'
        }).reset_index()
        
        # Calculate snapshot averages (10 snapshots per env-adsorbate combination)
        train_snapshot_avg = train_voxel_avg.groupby(['zeolite', 'environment', 'adsorbate', 'fold']).agg({
            'y_true': 'mean',  # Average across 10 snapshots
            'y_pred': 'mean'   # Average across 10 snapshots
        }).reset_index()
        
        test_snapshot_avg = test_voxel_avg.groupby(['zeolite', 'environment', 'adsorbate', 'fold']).agg({
            'y_true': 'mean',
            'y_pred': 'mean'
        }).reset_index()
        
        if self.verbose:
            print(f"    Voxel averaging results:")
            print(f"      Train: {len(combined_train)} → {len(train_voxel_avg)} points")
            print(f"      Test:  {len(combined_test)} → {len(test_voxel_avg)} points")
            print(f"    Snapshot averaging results:")
            print(f"      Train: {len(train_voxel_avg)} → {len(train_snapshot_avg)} points")
            print(f"      Test:  {len(test_voxel_avg)} → {len(test_snapshot_avg)} points")
        
        # Add averaged columns to each fold's dataframes
        for fold_idx, fold_results in self.model_storage.items():
            if 'df_train' in fold_results and 'df_test' in fold_results:
                df_train = fold_results['df_train'].copy()
                df_test = fold_results['df_test'].copy()
                
                # Get fold-specific averaged data
                train_voxel_fold = train_voxel_avg[train_voxel_avg['fold'] == fold_idx]
                test_voxel_fold = test_voxel_avg[test_voxel_avg['fold'] == fold_idx]
                train_snapshot_fold = train_snapshot_avg[train_snapshot_avg['fold'] == fold_idx]
                test_snapshot_fold = test_snapshot_avg[test_snapshot_avg['fold'] == fold_idx]
                
                # Merge voxel averages
                df_train = df_train.merge(
                    train_voxel_fold[['zeolite', 'environment', 'adsorbate', 'snapshot', 'y_true', 'y_pred']].rename(
                        columns={'y_true': 'y_true_voxel_avg', 'y_pred': 'y_pred_voxel_avg'}
                    ),
                    on=['zeolite', 'environment', 'adsorbate', 'snapshot'],
                    how='left'
                )
                
                df_test = df_test.merge(
                    test_voxel_fold[['zeolite', 'environment', 'adsorbate', 'snapshot', 'y_true', 'y_pred']].rename(
                        columns={'y_true': 'y_true_voxel_avg', 'y_pred': 'y_pred_voxel_avg'}
                    ),
                    on=['zeolite', 'environment', 'adsorbate', 'snapshot'],
                    how='left'
                )
                
                # Merge snapshot averages
                df_train = df_train.merge(
                    train_snapshot_fold[['zeolite', 'environment', 'adsorbate', 'y_true', 'y_pred']].rename(
                        columns={'y_true': 'y_true_snapshot_avg', 'y_pred': 'y_pred_snapshot_avg'}
                    ),
                    on=['zeolite', 'environment', 'adsorbate'],
                    how='left'
                )
                
                df_test = df_test.merge(
                    test_snapshot_fold[['zeolite', 'environment', 'adsorbate', 'y_true', 'y_pred']].rename(
                        columns={'y_true': 'y_true_snapshot_avg', 'y_pred': 'y_pred_snapshot_avg'}
                    ),
                    on=['zeolite', 'environment', 'adsorbate'],
                    how='left'
                )
                
                # Update model_storage
                self.model_storage[fold_idx]['df_train'] = df_train
                self.model_storage[fold_idx]['df_test'] = df_test
        
        if self.verbose:
            print(f"    Added columns: y_true_voxel_avg, y_pred_voxel_avg, y_true_snapshot_avg, y_pred_snapshot_avg")

    
    def calculate_summary_stats(self):
        """Calculate summary statistics across all folds for raw, voxel-averaged, and snapshot-averaged data"""
        if self.fold_results is None or self.fold_results.empty:
            return
        
        # Calculate raw metrics (existing code)
        raw_metrics = ['train_rmse', 'train_r2', 'train_mae', 'test_rmse', 'test_r2', 'test_mae']
        
        self.summary_stats = {}
        for metric in raw_metrics:
            values = self.fold_results[metric].dropna()
            if len(values) > 0:
                self.summary_stats[metric] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'median': values.median()
                }
        
        # Calculate voxel-averaged and snapshot-averaged metrics
        self.voxel_avg_stats = self.calculate_averaged_metrics('voxel')
        self.snapshot_avg_stats = self.calculate_averaged_metrics('snapshot')
        
        if self.verbose and (self.voxel_avg_stats or self.snapshot_avg_stats):
            print(f"    Calculated averaged metrics:")
            print(f"      Voxel-averaged metrics: {list(self.voxel_avg_stats.keys())}")
            print(f"      Snapshot-averaged metrics: {list(self.snapshot_avg_stats.keys())}")

    
    def calculate_averaged_metrics(self, avg_type: str) -> Dict:
        """
        Calculate performance metrics for averaged predictions
        
        Args:
            avg_type: 'voxel' or 'snapshot' averaging type
            
        Returns:
            dict: Calculated metrics with mean and std across folds
        """
        if avg_type not in ['voxel', 'snapshot']:
            raise ValueError("avg_type must be 'voxel' or 'snapshot'")
        
        fold_metrics = {'rmse': {'train': [], 'test': []}, 
                       'mae': {'train': [], 'test': []}, 
                       'r2': {'train': [], 'test': []}}
        
        for fold_idx, fold_results in self.model_storage.items():
            if 'df_train' in fold_results and 'df_test' in fold_results:
                df_train = fold_results['df_train']
                df_test = fold_results['df_test']
                
                # Determine which columns to use
                if avg_type == 'voxel':
                    true_col = 'y_true_voxel_avg'
                    pred_col = 'y_pred_voxel_avg'
                else:  # snapshot
                    true_col = 'y_true_snapshot_avg'
                    pred_col = 'y_pred_snapshot_avg'
                
                # Check if the required columns exist
                if true_col in df_train.columns and pred_col in df_train.columns:
                    # Calculate metrics for train set
                    train_true = df_train[true_col].dropna()
                    train_pred = df_train[pred_col].dropna()
                    
                    if len(train_true) > 0:
                        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                        
                        train_rmse = np.sqrt(mean_squared_error(train_true, train_pred))
                        train_mae = mean_absolute_error(train_true, train_pred)
                        train_r2 = r2_score(train_true, train_pred)
                        
                        fold_metrics['rmse']['train'].append(train_rmse)
                        fold_metrics['mae']['train'].append(train_mae)
                        fold_metrics['r2']['train'].append(train_r2)
                
                if true_col in df_test.columns and pred_col in df_test.columns:
                    # Calculate metrics for test set
                    test_true = df_test[true_col].dropna()
                    test_pred = df_test[pred_col].dropna()
                    
                    if len(test_true) > 0:
                        test_rmse = np.sqrt(mean_squared_error(test_true, test_pred))
                        test_mae = mean_absolute_error(test_true, test_pred)
                        test_r2 = r2_score(test_true, test_pred)
                        
                        fold_metrics['rmse']['test'].append(test_rmse)
                        fold_metrics['mae']['test'].append(test_mae)
                        fold_metrics['r2']['test'].append(test_r2)
        
        # Calculate summary statistics
        summary_metrics = {}
        for metric in ['rmse', 'mae', 'r2']:
            if fold_metrics[metric]['train'] and fold_metrics[metric]['test']:
                train_values = np.array(fold_metrics[metric]['train'])
                test_values = np.array(fold_metrics[metric]['test'])
                
                summary_metrics[metric] = {
                    'train_mean': train_values.mean(),
                    'train_std': train_values.std(),
                    'test_mean': test_values.mean(),
                    'test_std': test_values.std()
                }
        
        return summary_metrics
    
    
    def print_summary(self):
        """Print comprehensive summary of results"""
        if not self.results_data:
            print("No results loaded. Please load a pkl file first.")
            return
        
        # Split strategy information
        print(f"\n--- Split Strategy Information ---")
        print(f"    Split type: {self.split_type}")
        if self.model_storage:
            print(f"    Number of folds: {len(self.model_storage)}")
        
        # Training configuration
        print(f"\n--- Training Configuration ---")
        for key, value in self.training_config.items():
            print(f"    {key}: {value}")
        
        # Data information
        print(f"\n--- Data Information ---")
        for key, value in self.data_info.items():
            print(f"    {key}: {value}")
        
        # Performance summary
        if self.summary_stats:
            print(f"\n--- Performance Summary ---")
            print(f"    {'Metric':<12} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'Median':<8}")
            print("    " + "-" * 60)
            
            for metric, stats in self.summary_stats.items():
                print(f"    {metric:<12} {stats['mean']:<8.3f} {stats['std']:<8.3f} "
                  f"{stats['min']:<8.3f} {stats['max']:<8.3f} {stats['median']:<8.3f}")
        
        # Per-fold results
        if self.fold_results is not None:
            print(f"\n--- Per-Fold Results ---")
            fold_results_str = self.fold_results.round(3).to_string(index=False)
            # Add 4 spaces at the beginning of each line
            indented_results = '\n'.join('   ' + line for line in fold_results_str.split('\n'))
            print(indented_results)
    
    
    def export_results_to_csv(self, output_path: str = None) -> str:
        """
        Export results to CSV file
        
        Args:
            output_path: Path for the output CSV file
            
        Returns:
            str: Path of the saved CSV file
        """
        if self.fold_results is None or self.fold_results.empty:
            print("No results to export")
            return None
        
        if output_path is None:
            base_name = self.pkl_name.replace('.pkl', '_results.csv') if self.pkl_name else 'cnn_results.csv'
            output_path = os.path.join(self.model_dir, base_name)
        
        # Add summary statistics as additional rows
        export_df = self.fold_results.copy()
        
        # Add summary rows
        if self.summary_stats:
            summary_rows = []
            for stat_type in ['mean', 'std', 'min', 'max', 'median']:
                row = {'fold': f'{stat_type.upper()}'}
                for metric in ['train_rmse', 'train_r2', 'train_mae', 'test_rmse', 'test_r2', 'test_mae']:
                    if metric in self.summary_stats:
                        row[metric] = self.summary_stats[metric][stat_type]
                    else:
                        row[metric] = np.nan
                summary_rows.append(row)
            
            summary_df = pd.DataFrame(summary_rows)
            export_df = pd.concat([export_df, summary_df], ignore_index=True)
        
        export_df.to_csv(output_path, index=False)
        
        if self.verbose:
            print(f"Results exported to: {output_path}")
        
        return output_path
    
        
    def plot_parity_plots_using_test_augment(self,
                                            figsize: Tuple[int, int] = (24, 16),
                                            show_plot: bool = False,
                                            save_plot: bool = False):
        """
        Professional-grade parity plots comparing WITH and WITHOUT test time augmentation
        Enhanced comparison showing the impact of TTA on prediction performance
        Creates 5 subplots: 3 with TTA (top row) + 2 without TTA (bottom row) + comparison table
        
        Args:
            figsize: Figure size for the overall plot (default: (24, 16) for 2x3 layout)
            show_plot: Whether to show the plot (default: False)
            save_plot: Whether to save the plot (default: False)
        """
        if not self.model_storage:
            print("No results to plot")
            return

        # Professional color scheme (colorblind-friendly)
        colors = {
            'train': self.blue_color,    # Use class attribute
            'test': self.red_color,      # Use class attribute
            'md': self.green_color,      # Use class attribute
            'unity': '#2c2c2c',          # Dark gray for unity line
            'stats_bg': '#f8f9fa'        # Light background for stats box
        }
        
        # Keep original matplotlib settings (minimal changes)
        original_rcParams = plt.rcParams.copy()  # Save original settings

        # subplot number positions
        num_pos_x = -0.245
        num_pos_y = 1.065
        
        # Create figure with 2x3 subplots
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        if self.verbose:
            print(f"\n--- TTA vs No-TTA Comparison Parity Plot:")

        # Plot configurations for 5 subplots: 3 with TTA (top row) + 2 without TTA (bottom row)
        plot_configs = [
            # TOP ROW - WITH TTA
            {
                'row': 0, 'col': 0,
                'data_type': 'raw_with_tta',
                'label': '(a)',
                'title': 'With TTA: All Voxel Grids',
                'alpha_train': self.alpha_train,
                'alpha_test': self.alpha,
                'marker_size': self.point_size_deltaEint,
                'xlabel': r'DFT calculated $\Delta E^{\mathit{DFT}}_{\mathit{int}}$ (eV)',
                'ylabel': r'ML predicted $\Delta E^{\mathit{ML}}_{\mathit{int}}$ (eV)',
                'show_error_bars': True  # NEW: Enable error bars for per-snapshot SD
            },
            {
                'row': 0, 'col': 1,
                'data_type': 'voxel_avg_with_tta',
                'label': '(b)',
                'title': 'With TTA: Voxel Averaged',
                'alpha_train': self.alpha_train,
                'alpha_test': self.alpha,
                'marker_size': self.point_size_deltaEint,
                'xlabel': r'DFT calculated $\Delta E^{\mathit{DFT}}_{\mathit{int}}$ (eV)',
                'ylabel': r'ML predicted $\Delta E^{\mathit{ML}}_{\mathit{int}}$ (eV)'
            },
            {
                'row': 0, 'col': 2,
                'data_type': 'snapshot_avg_with_tta',
                'label': '(c)',
                'title': 'With TTA: Snapshot Averaged',
                'alpha_train': self.alpha_train,
                'alpha_test': self.alpha,
                'marker_size': self.point_size,
                'xlabel': r'DFT calculated $\Delta E^{\mathit{DFT}}_{\mathit{sol}}$ (eV)',
                'ylabel': r'ML predicted $\Delta E^{\mathit{ML}}_{\mathit{sol}}$ (eV)'
            },
            # BOTTOM ROW - WITHOUT TTA
            {
                'row': 1, 'col': 0,
                'data_type': 'raw_no_tta',
                'label': '(d)',
                'title': 'No TTA: Original Voxels Only',
                'alpha_train': self.alpha_train,
                'alpha_test': self.alpha,
                'marker_size': self.point_size_deltaEint,
                'xlabel': r'DFT calculated $\Delta E^{\mathit{DFT}}_{\mathit{int}}$ (eV)',
                'ylabel': r'ML predicted $\Delta E^{\mathit{ML}}_{\mathit{int}}$ (eV)'
            },
            {
                'row': 1, 'col': 1,
                'data_type': 'snapshot_avg_no_tta',
                'label': '(e)',
                'title': 'No TTA: Snapshot Averaged',
                'alpha_train': self.alpha_train,
                'alpha_test': self.alpha,
                'marker_size': self.point_size,
                'xlabel': r'DFT calculated $\Delta E^{\mathit{DFT}}_{\mathit{sol}}$ (eV)',
                'ylabel': r'ML predicted $\Delta E^{\mathit{ML}}_{\mathit{sol}}$ (eV)'
            }
        ]
        
        # Store metrics for comparison table
        comparison_metrics = {}
        
        for config in plot_configs:
            ax = axes[config['row'], config['col']]
            
            # Collect data based on type
            all_train_true, all_train_pred = [], []
            all_test_true, all_test_pred = [], []
            test_unique_for_md = None  # Store test_unique for MD data matching
            
            # NEW: Store per-snapshot data for SD calculation (only for raw_with_tta)
            test_snapshot_groups = []  # List of DataFrames, one per snapshot
            train_snapshot_groups = []
            
            for fold_idx, fold_results in self.model_storage.items():
                if 'df_train' in fold_results and 'df_test' in fold_results:
                    df_train = fold_results['df_train']
                    df_test = fold_results['df_test']
                    
                    if config['data_type'] == 'raw_with_tta':
                        if 'y_true' in df_train.columns and 'y_pred' in df_train.columns:
                            all_train_true.extend(df_train['y_true'].tolist())
                            all_train_pred.extend(df_train['y_pred'].tolist())
                            all_test_true.extend(df_test['y_true'].tolist())
                            all_test_pred.extend(df_test['y_pred'].tolist())
                            
                            # NEW: Group by snapshot for SD calculation
                            required_cols = ['zeolite', 'environment', 'adsorbate', 'snapshot']
                            if all(col in df_test.columns for col in required_cols):
                                test_snapshot_groups.append(df_test)
                            if all(col in df_train.columns for col in required_cols):
                                train_snapshot_groups.append(df_train)
                            
                    elif config['data_type'] == 'voxel_avg_with_tta':
                        if 'y_true_voxel_avg' in df_train.columns:
                            train_unique = df_train[['zeolite', 'environment', 'adsorbate', 'snapshot', 
                                                   'y_true_voxel_avg', 'y_pred_voxel_avg']].drop_duplicates()
                            test_unique = df_test[['zeolite', 'environment', 'adsorbate', 'snapshot', 
                                                 'y_true_voxel_avg', 'y_pred_voxel_avg']].drop_duplicates()
                            
                            all_train_true.extend(train_unique['y_true_voxel_avg'].tolist())
                            all_train_pred.extend(train_unique['y_pred_voxel_avg'].tolist())
                            all_test_true.extend(test_unique['y_true_voxel_avg'].tolist())
                            all_test_pred.extend(test_unique['y_pred_voxel_avg'].tolist())
                            
                    elif config['data_type'] == 'snapshot_avg_with_tta':
                        if 'y_true_snapshot_avg' in df_train.columns:
                            train_unique = df_train[['zeolite', 'environment', 'adsorbate', 
                                                   'y_true_snapshot_avg', 'y_pred_snapshot_avg']].drop_duplicates()
                            test_unique = df_test[['zeolite', 'environment', 'adsorbate', 
                                                 'y_true_snapshot_avg', 'y_pred_snapshot_avg']].drop_duplicates()
                            
                            # Store test_unique for MD data matching
                            if test_unique_for_md is None:
                                test_unique_for_md = test_unique
                            else:
                                test_unique_for_md = pd.concat([test_unique_for_md, test_unique], ignore_index=True)
                            
                            all_train_true.extend(train_unique['y_true_snapshot_avg'].tolist())
                            all_train_pred.extend(train_unique['y_pred_snapshot_avg'].tolist())
                            all_test_true.extend(test_unique['y_true_snapshot_avg'].tolist())
                            all_test_pred.extend(test_unique['y_pred_snapshot_avg'].tolist())
                            
                    elif config['data_type'] == 'raw_no_tta':
                        # Only original voxels (every 24th or first of each group)
                        if 'y_true' in df_train.columns and 'y_pred' in df_train.columns:
                            if 'is_original_voxel' in df_train.columns:
                                train_original = df_train[df_train['is_original_voxel'] == True]
                                test_original = df_test[df_test['is_original_voxel'] == True]
                            else:
                                # Group by snapshot and take the first sample (assuming it's the original)
                                required_cols = ['zeolite', 'environment', 'adsorbate', 'snapshot']
                                if all(col in df_train.columns for col in required_cols):
                                    train_original = df_train.groupby(required_cols).first().reset_index()
                                    test_original = df_test.groupby(required_cols).first().reset_index()
                                else:
                                    # Fallback - assume every 24th row is original
                                    train_original = df_train.iloc[::24].copy()
                                    test_original = df_test.iloc[::24].copy()
                            
                            if len(train_original) > 0 and len(test_original) > 0:
                                all_train_true.extend(train_original['y_true'].tolist())
                                all_train_pred.extend(train_original['y_pred'].tolist())
                                all_test_true.extend(test_original['y_true'].tolist())
                                all_test_pred.extend(test_original['y_pred'].tolist())
                                
                    elif config['data_type'] == 'snapshot_avg_no_tta':
                        # Snapshot averages from original voxels only
                        if 'y_true' in df_train.columns and 'y_pred' in df_train.columns:
                            if 'is_original_voxel' in df_train.columns:
                                train_original = df_train[df_train['is_original_voxel'] == True]
                                test_original = df_test[df_test['is_original_voxel'] == True]
                            else:
                                required_cols = ['zeolite', 'environment', 'adsorbate', 'snapshot']
                                if all(col in df_train.columns for col in required_cols):
                                    train_original = df_train.groupby(required_cols).first().reset_index()
                                    test_original = df_test.groupby(required_cols).first().reset_index()
                                else:
                                    train_original = df_train.iloc[::24].copy()
                                    test_original = df_test.iloc[::24].copy()
                            
                            if len(train_original) > 0 and len(test_original) > 0:
                                # Calculate snapshot averages (average over 10 snapshots per env-adsorbate)
                                required_group_cols = ['zeolite', 'environment', 'adsorbate']
                                if all(col in train_original.columns for col in required_group_cols):
                                    train_snapshot_avg = train_original.groupby(required_group_cols).agg({
                                        'y_true': 'mean', 'y_pred': 'mean'
                                    }).reset_index()
                                    
                                    test_snapshot_avg = test_original.groupby(required_group_cols).agg({
                                        'y_true': 'mean', 'y_pred': 'mean'
                                    }).reset_index()
                                    
                                    all_train_true.extend(train_snapshot_avg['y_true'].tolist())
                                    all_train_pred.extend(train_snapshot_avg['y_pred'].tolist())
                                    all_test_true.extend(test_snapshot_avg['y_true'].tolist())
                                    all_test_pred.extend(test_snapshot_avg['y_pred'].tolist())
            
            if all_train_true and all_test_true:
                # Convert to numpy arrays for easier manipulation
                train_true = np.array(all_train_true)
                train_pred = np.array(all_train_pred)
                test_true = np.array(all_test_true)
                test_pred = np.array(all_test_pred)
                
                # Calculate comprehensive statistics
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                # Use pre-calculated stats for consistent fold-wise averaging
                if config['data_type'] in ['raw_with_tta', 'raw_no_tta'] and self.summary_stats:
                    # For raw data (deltaEint), use summary stats
                    train_mae = self.summary_stats.get('train_mae', {}).get('mean', mean_absolute_error(train_true, train_pred))
                    test_mae = self.summary_stats.get('test_mae', {}).get('mean', mean_absolute_error(test_true, test_pred))
                elif config['data_type'] in ['snapshot_avg_with_tta', 'snapshot_avg_no_tta'] and hasattr(self, 'snapshot_avg_stats') and self.snapshot_avg_stats:
                    # For snapshot-averaged data (deltaEsol), use snapshot-averaged stats
                    train_mae = self.snapshot_avg_stats.get('mae', {}).get('train_mean', mean_absolute_error(train_true, train_pred))
                    test_mae = self.snapshot_avg_stats.get('mae', {}).get('test_mean', mean_absolute_error(test_true, test_pred))
                elif config['data_type'] in ['voxel_avg_with_tta'] and hasattr(self, 'voxel_avg_stats') and self.voxel_avg_stats:
                    # For voxel-averaged data, use voxel-averaged stats
                    train_mae = self.voxel_avg_stats.get('mae', {}).get('train_mean', mean_absolute_error(train_true, train_pred))
                    test_mae = self.voxel_avg_stats.get('mae', {}).get('test_mean', mean_absolute_error(test_true, test_pred))
                else:
                    train_mae = mean_absolute_error(train_true, train_pred)
                    test_mae = mean_absolute_error(test_true, test_pred)
                
                train_rmse = np.sqrt(mean_squared_error(train_true, train_pred))
                train_r2 = r2_score(train_true, train_pred)
                train_mape = np.mean(np.abs((train_true - train_pred) / np.abs(train_true))) * 100
                
                test_rmse = np.sqrt(mean_squared_error(test_true, test_pred))
                test_r2 = r2_score(test_true, test_pred)
                
                # Store metrics for comparison table
                comparison_metrics[config['data_type']] = {
                    'rmse': test_rmse,
                    'mae': test_mae,
                    'r2': test_r2,
                    'n_points': len(test_true)
                }
                
                # NEW: Calculate per-snapshot SD for error bars (only for raw_with_tta)
                test_snapshot_sd_data = None
                if config.get('show_error_bars', False) and test_snapshot_groups:
                    # Combine all test data from all folds
                    test_combined = pd.concat(test_snapshot_groups, ignore_index=True)
                    
                    # Group by snapshot identifier
                    required_cols = ['zeolite', 'environment', 'adsorbate', 'snapshot']
                    if all(col in test_combined.columns for col in required_cols):
                        # Calculate mean and SD for each snapshot (24 TTA predictions per snapshot)
                        snapshot_stats = test_combined.groupby(required_cols).agg({
                            'y_true': 'first',  # All 24 have same DFT value
                            'y_pred': ['mean', 'std', 'count']
                        }).reset_index()
                        
                        # Flatten column names
                        snapshot_stats.columns = ['zeolite', 'environment', 'adsorbate', 'snapshot', 
                                                 'y_true', 'y_pred_mean', 'y_pred_sd', 'n_predictions']
                        
                        # Store for plotting
                        test_snapshot_sd_data = snapshot_stats
                        
                        # Print detailed TTA robustness statistics
                        if self.verbose and config['label'] == '(a)':
                            sd_values = snapshot_stats['y_pred_sd'].values
                            n_predictions = snapshot_stats['n_predictions'].values
                            
                            print(f"\n--- TTA Rotational Robustness Analysis (Per-Snapshot SD) ---")
                            print(f"    Data Structure Verification:")
                            print(f"      Total data points in test_combined: {len(test_combined)}")
                            print(f"      Number of folds: {len(test_snapshot_groups)}")
                            print(f"      Unique snapshots: {len(snapshot_stats)}")
                            print(f"      Expected snapshots (80 adsorbates × 10 snapshots): 800")
                            print(f"      Predictions per snapshot - Min: {n_predictions.min()}, Max: {n_predictions.max()}, Mean: {n_predictions.mean():.1f}")
                            print(f"      Expected predictions per snapshot: 24")
                            
                            # Check for anomalies
                            if len(snapshot_stats) != 800:
                                print(f"      ⚠️  WARNING: Expected 800 snapshots but found {len(snapshot_stats)}")
                            if n_predictions.min() != 24 or n_predictions.max() != 24:
                                print(f"      ⚠️  WARNING: Not all snapshots have exactly 24 predictions!")
                                print(f"         Snapshots with != 24 predictions: {np.sum(n_predictions != 24)}")
                            
                            print(f"\n    Per-snapshot SD statistics:")
                            print(f"      Mean SD:    {np.mean(sd_values):.4f} eV")
                            print(f"      Median SD:  {np.median(sd_values):.4f} eV")
                            print(f"      Min SD:     {np.min(sd_values):.4f} eV")
                            print(f"      Max SD:     {np.max(sd_values):.4f} eV")
                            print(f"      Std of SD:  {np.std(sd_values):.4f} eV")
                            
                            # Calculate range statistics
                            snapshot_ranges = test_combined.groupby(required_cols)['y_pred'].agg(
                                lambda x: x.max() - x.min()
                            ).values
                            print(f"    Per-snapshot Range (Max-Min) statistics:")
                            print(f"      Mean Range: {np.mean(snapshot_ranges):.4f} eV")
                            print(f"      Max Range:  {np.max(snapshot_ranges):.4f} eV")
                            
                            # Correlation between SD and prediction error
                            snapshot_stats['abs_error'] = np.abs(snapshot_stats['y_true'] - snapshot_stats['y_pred_mean'])
                            correlation = np.corrcoef(snapshot_stats['y_pred_sd'], snapshot_stats['abs_error'])[0, 1]
                            print(f"    Correlation (SD vs Absolute Error): {correlation:.3f}")
                            print(f"    --> {'Low' if abs(correlation) < 0.3 else 'Moderate' if abs(correlation) < 0.6 else 'High'} correlation suggests SD {'may not be' if abs(correlation) < 0.3 else 'could be'} a useful uncertainty indicator")
                
                # Set up the plot area with clean white background
                ax.set_facecolor('white')
                ax.set_axisbelow(True)
                
                # Plot data points with enhanced styling
                train_scatter = ax.scatter(train_true, train_pred,
                                         alpha=config['alpha_train'], 
                                         s=config['marker_size'],
                                         c=colors['train'], 
                                         marker='^',
                                         edgecolors='white', 
                                         linewidths=0.3,
                                         label='Train')
                
                # NEW: Plot with error bars if SD data available
                if test_snapshot_sd_data is not None:
                    # First plot the scatter points (same as before)
                    test_scatter = ax.scatter(test_snapshot_sd_data['y_true'], 
                                            test_snapshot_sd_data['y_pred_mean'],
                                            alpha=config['alpha_test'], 
                                            s=config['marker_size'],
                                            c=colors['test'], 
                                            marker='o',
                                            edgecolors='white', 
                                            linewidths=0.3,
                                            label='Test',
                                            zorder=3)
                    
                    # Then add error bars on top (no markers, just bars)
                    ax.errorbar(test_snapshot_sd_data['y_true'], 
                               test_snapshot_sd_data['y_pred_mean'],
                               yerr=test_snapshot_sd_data['y_pred_sd'],
                               fmt='none',  # No markers, only error bars
                               ecolor=colors['test'],
                               elinewidth=2,    # Error bar line width
                               capsize=4,       # Error bar cap size
                               capthick=2,      # Error bar cap thickness
                               alpha=config['alpha_test'] * 0.8,  # Slightly more transparent for bars
                               zorder=2)
                else:
                    # Plot without error bars
                    test_scatter = ax.scatter(test_true, test_pred,
                                            alpha=config['alpha_test'], 
                                            s=config['marker_size'],
                                            c=colors['test'], 
                                            marker='o',
                                            edgecolors='white', 
                                            linewidths=0.3,
                                            label='Test')
                
                # Initialize all_values for plot limits
                all_values = [train_true, train_pred, test_true, test_pred]
                
                # Note: MD data comparison removed for TTA comparison focus
                
                # Calculate plot limits with padding
                all_values = np.concatenate(all_values)
                data_min, data_max = np.min(all_values), np.max(all_values)
                padding = (data_max - data_min) * 0.05
                plot_min = data_min - padding
                plot_max = data_max + padding
                
                # Perfect prediction line (unity line) with enhanced styling
                unity_line = ax.plot([plot_min, plot_max], [plot_min, plot_max], 
                                   color=colors['unity'], 
                                   linewidth=2, 
                                   linestyle='--', 
                                   alpha=0.8,
                                   zorder=5)
                
                # Add confidence bands for the test data (professional regression-based)
                if len(test_true) > 3:  # Need sufficient data points
                    from scipy import stats
                    
                    # Linear regression for test data
                    slope, intercept, r_value, p_value, std_err = stats.linregress(test_true, test_pred)
                    
                    # Calculate plot limits for confidence bands
                    data_min, data_max = np.min(test_true), np.max(test_true)
                    x_smooth = np.linspace(data_min, data_max, 100)
                    
                    # Calculate residuals and prediction intervals
                    test_predicted = slope * test_true + intercept
                    residuals = test_pred - test_predicted
                    mse = np.sum(residuals**2) / max(len(test_true) - 2, 1)
                    
                    # 95% confidence interval (like R's geom_smooth)
                    t_val = stats.t.ppf(0.975, max(len(test_true) - 2, 1))
                    confidence_std = np.sqrt(mse * (1/len(test_true) + 
                                                   (x_smooth - np.mean(test_true))**2 / 
                                                   np.sum((test_true - np.mean(test_true))**2)))
                    
                    # Calculate confidence bands (narrower than prediction bands)
                    y_pred = slope * x_smooth + intercept
                    upper_band = y_pred + t_val * confidence_std
                    lower_band = y_pred - t_val * confidence_std
                    
                    # Fill confidence region for test data
                    ax.fill_between(x_smooth, lower_band, upper_band, 
                                  alpha=0.15, color=colors['test'], 
                                  label='95% Confidence Band')
                
                # Professional axis formatting
                ax.set_xlim(plot_min, plot_max)
                ax.set_ylim(plot_min, plot_max)
                ax.set_aspect('equal', adjustable='box')
                
                # Enhanced labels with proper LaTeX formatting
                ax.set_xlabel(config['xlabel'], 
                             fontsize=self.font_size, fontweight='normal')
                ax.set_ylabel(config['ylabel'], 
                             fontsize=self.font_size, fontweight='normal')
                
                # Create simplified legend - just show test metrics
                legend_lines = [
                    f"Test MAE: {test_mae:.3f} eV",
                    f"Test R²: {test_r2:.3f}"
                ]
                
                # Create a simple text box for legend
                legend_text = '\n'.join(legend_lines)
                ax.text(0.07, 0.93, legend_text,
                       transform=ax.transAxes,
                       fontsize=self.font_size,  # Slightly smaller for space
                       verticalalignment='top',
                       horizontalalignment='left',
                       bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='white',
                                edgecolor='black',
                                alpha=0.9,
                                linewidth=0.8))
                
                # Add subplot title
                ax.set_title(config['title'], fontsize=self.font_size, fontweight='bold', pad=10)
                
                # Professional subplot label
                ax.text(num_pos_x, num_pos_y, config['label'], 
                       transform=ax.transAxes,
                       fontsize=self.font_size, 
                       fontweight='bold',
                       verticalalignment='top', 
                       horizontalalignment='left')
                
                # Enhanced tick formatting
                ax.tick_params(axis='both', which='major', labelsize=self.font_size)
                
                if self.verbose:
                    print(f"    {config['label']} {config['data_type']}: Train={len(train_true)}, Test={len(test_true)}")
        
        # Use the empty subplot (1,2) for comparison bar plot
        ax_comparison = axes[1, 2]
        
        # Create comparison bar plot
        if 'snapshot_avg_with_tta' in comparison_metrics and 'snapshot_avg_no_tta' in comparison_metrics:
            tta_metrics = comparison_metrics['snapshot_avg_with_tta']
            no_tta_metrics = comparison_metrics['snapshot_avg_no_tta']
            
            # Data for bar plot (removed R²)
            methods = ['With TTA', 'No TTA']
            rmse_values = [tta_metrics['rmse'], no_tta_metrics['rmse']]
            mae_values = [tta_metrics['mae'], no_tta_metrics['mae']]
            
            # Set up bar positions
            x = np.arange(len(methods))
            width = 0.35  # Wider bars since we only have 2 metrics
            
            # Create bars for each metric with new colors (avoid train/test confusion)
            bars1 = ax_comparison.bar(x - width/2, rmse_values, width, 
                                    label='RMSE (eV)', color='#2ca02c', alpha=0.8)  # green
            bars2 = ax_comparison.bar(x + width/2, mae_values, width, 
                                    label='MAE (eV)', color='#ff7f0e', alpha=0.8)  # orange

            # Add value labels on bars
            def add_value_labels(bars, values):
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax_comparison.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                                     f'{value:.3f}', ha='center', va='bottom', 
                                     fontsize=self.font_size, fontweight='bold')
            
            add_value_labels(bars1, rmse_values)
            add_value_labels(bars2, mae_values)
            
            # Add subplot label (f)
            ax_comparison.text(num_pos_x, num_pos_y, '(f)', 
                             transform=ax_comparison.transAxes,
                             fontsize=self.font_size, 
                             fontweight='bold',
                             verticalalignment='top', 
                             horizontalalignment='left')
            
            # Customize the plot
            ax_comparison.set_aspect('auto')
            ax_comparison.set_xlabel('Method', fontsize=self.font_size)
            ax_comparison.set_ylabel('Test Error (eV)', fontsize=self.font_size)
            ax_comparison.set_title(r"Performance on $\mathbf{\Delta E_{\mathbf{sol}}}$", fontsize=self.font_size, fontweight='bold')
            ax_comparison.set_xticks(x)
            ax_comparison.set_xticklabels(methods, fontsize=self.font_size)
            ax_comparison.tick_params(axis='y', labelsize=self.font_size)
            ax_comparison.legend(fontsize=self.font_size, loc='upper right')
            ax_comparison.grid(False)
            ax_comparison.set_ylim(0, max(max(rmse_values), max(mae_values)) * 1.55)
            
        else:
            # If no data available, show message
            ax_comparison.text(0.5, 0.5, 'No comparison data\navailable', 
                             transform=ax_comparison.transAxes,
                             fontsize=self.font_size, ha='center', va='center',
                             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
            ax_comparison.set_title('TTA Performance\nComparison', fontsize=self.font_size, fontweight='bold')
        
        plt.tight_layout()
        
        # Manually adjust the height of subplot f (bar plot) after tight_layout
        pos = axes[1, 2].get_position()
        new_height = pos.height * 0.94  # Reduce height to 92% of original
        axes[1, 2].set_position([pos.x0, pos.y0 + (pos.height - new_height)/2, pos.width, new_height])
        
        # Save with high quality settings
        if save_plot:
            parts = self.pkl_name.split('-')
            job_code = '-'.join(parts[:3])
            file_name = f"{job_code}-tta_comparison_analysis.png"
            save_path = os.path.join(self.output_figure_path, 'cnn_results', file_name)
            plt.savefig(save_path, dpi=500, bbox_inches='tight')
            
            if self.verbose:
                print(f"TTA comparison analysis saved to: {save_path}")
                
                # Print detailed comparison
                print(f"\n--- TTA vs No-TTA Performance Comparison (ΔEsol) ---")
                if 'snapshot_avg_with_tta' in comparison_metrics and 'snapshot_avg_no_tta' in comparison_metrics:
                    tta_metrics = comparison_metrics['snapshot_avg_with_tta']
                    no_tta_metrics = comparison_metrics['snapshot_avg_no_tta']
                    
                    print(f"    With TTA    - RMSE: {tta_metrics['rmse']:.3f} eV, MAE: {tta_metrics['mae']:.3f} eV")
                    print(f"    Without TTA - RMSE: {no_tta_metrics['rmse']:.3f} eV, MAE: {no_tta_metrics['mae']:.3f} eV")
                    
                    rmse_improvement = ((no_tta_metrics['rmse'] - tta_metrics['rmse']) / no_tta_metrics['rmse']) * 100
                    mae_improvement = ((no_tta_metrics['mae'] - tta_metrics['mae']) / no_tta_metrics['mae']) * 100
                    
                    print(f"    TTA Effect  - RMSE: {rmse_improvement:+.2f}%, MAE: {mae_improvement:+.2f}%")
        
        # Reset matplotlib parameters to avoid affecting other plots
        plt.rcParams.update(original_rcParams)

        if show_plot:
            plt.show()

    
    def plot_parity_plot_no_test_augment_with_Train_2_subfigures(self,
                                                                 figsize: Tuple[int, int] = (16, 8),
                                                                 show_plot: bool = False,
                                                                 save_plot: bool = False):
        """
        Professional-grade parity plots without test time augmentation - using only original voxels
        Enhanced with statistical information, improved aesthetics, and journal-quality formatting
        Creates two subplots (a), (b) showing original voxel predictions and their averages
        
        Args:
            figsize: Figure size for the overall plot
            show_plot: Whether to show the plot (default: False)
        """
        if not self.model_storage:
            print("No results to plot")
            return

        # Professional color scheme (colorblind-friendly)
        colors = {
            'train': self.blue_color,    # Use class attribute
            'test': self.red_color,      # Use class attribute
            'unity': '#2c2c2c',          # Dark gray for unity line
            'stats_bg': '#f8f9fa'        # Light background for stats box
        }
        
        # Keep original matplotlib settings (minimal changes)
        original_rcParams = plt.rcParams.copy()  # Save original settings

        # subplot number positions
        num_pos_x = -0.235
        num_pos_y = 1.05
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        if self.verbose:
            print(f"\n--- Professional No-TTA Parity Plot Data Points:")

        # Plot configurations for each subplot - different point sizes for deltaEint vs deltaEsol
        plot_configs = [
            {
                'data_type': 'original_voxel',
                'label': '(a)',
                'alpha_train': self.alpha_train,         # Use class attribute directly
                'alpha_test': self.alpha,                # Unified: test uses base alpha
                'marker_size': self.point_size_deltaEint, # Use deltaEint point size (more data points)
                'xlabel': r'DFT calculated $\Delta E^{\mathit{DFT}}_{\mathit{int}}$ (eV)',
                'ylabel': r'ML predicted $\Delta E^{\mathit{ML}}_{\mathit{int}}$ (eV)'
            },
            {
                'data_type': 'original_snapshot_avg',
                'label': '(b)',
                'alpha_train': self.alpha_train,     # Use class attribute directly
                'alpha_test': self.alpha,            # Unified: test uses base alpha
                'marker_size': self.point_size,      # Use regular point size for deltaEsol (fewer data points)
                'xlabel': r'DFT calculated $\Delta E^{\mathit{DFT}}_{\mathit{sol}}$ (eV)',
                'ylabel': r'ML predicted $\Delta E^{\mathit{ML}}_{\mathit{sol}}$ (eV)'
            }
        ]
        
        for idx, config in enumerate(plot_configs):
            ax = axes[idx]
            
            # Collect data based on type
            all_train_true, all_train_pred = [], []
            all_test_true, all_test_pred = [], []
            
            for fold_idx, fold_results in self.model_storage.items():
                if 'df_train' in fold_results and 'df_test' in fold_results:
                    df_train = fold_results['df_train']
                    df_test = fold_results['df_test']
                    
                    if config['data_type'] == 'original_voxel':
                        if 'y_true' in df_train.columns and 'y_pred' in df_train.columns:
                            # Method 1: Try to identify original voxels by checking for identifier column
                            if 'is_original_voxel' in df_train.columns:
                                train_original = df_train[df_train['is_original_voxel'] == True]
                                test_original = df_test[df_test['is_original_voxel'] == True]
                            else:
                                # Method 2: Assume every 24th sample starting from 0 is original
                                # Group by snapshot and take the first sample of each group
                                required_cols = ['zeolite', 'environment', 'adsorbate', 'snapshot']
                                if all(col in df_train.columns for col in required_cols):
                                    # Group by snapshot and take first entry (assuming it's the original)
                                    train_grouped = df_train.groupby(required_cols)
                                    test_grouped = df_test.groupby(required_cols)
                                    
                                    train_original = train_grouped.first().reset_index()
                                    test_original = test_grouped.first().reset_index()
                                else:
                                    # Method 3: Fallback - assume every 24th row is original
                                    train_original = df_train.iloc[::24].copy()
                                    test_original = df_test.iloc[::24].copy()
                            
                            if len(train_original) > 0 and len(test_original) > 0:
                                all_train_true.extend(train_original['y_true'].tolist())
                                all_train_pred.extend(train_original['y_pred'].tolist())
                                all_test_true.extend(test_original['y_true'].tolist())
                                all_test_pred.extend(test_original['y_pred'].tolist())
                                
                    elif config['data_type'] == 'original_snapshot_avg':
                        if 'y_true' in df_train.columns and 'y_pred' in df_train.columns:
                            # Get original voxel data (same logic as above)
                            if 'is_original_voxel' in df_train.columns:
                                train_original = df_train[df_train['is_original_voxel'] == True]
                                test_original = df_test[df_test['is_original_voxel'] == True]
                            else:
                                required_cols = ['zeolite', 'environment', 'adsorbate', 'snapshot']
                                if all(col in df_train.columns for col in required_cols):
                                    train_grouped = df_train.groupby(required_cols)
                                    test_grouped = df_test.groupby(required_cols)
                                    train_original = train_grouped.first().reset_index()
                                    test_original = test_grouped.first().reset_index()
                                else:
                                    train_original = df_train.iloc[::24].copy()
                                    test_original = df_test.iloc[::24].copy()
                            
                            if len(train_original) > 0 and len(test_original) > 0:
                                # Calculate snapshot averages (average over 10 snapshots per env-adsorbate)
                                required_group_cols = ['zeolite', 'environment', 'adsorbate']
                                if all(col in train_original.columns for col in required_group_cols):
                                    train_snapshot_avg = train_original.groupby(required_group_cols).agg({
                                        'y_true': 'mean',
                                        'y_pred': 'mean'
                                    }).reset_index()
                                    
                                    test_snapshot_avg = test_original.groupby(required_group_cols).agg({
                                        'y_true': 'mean',
                                        'y_pred': 'mean'
                                    }).reset_index()
                                    
                                    all_train_true.extend(train_snapshot_avg['y_true'].tolist())
                                    all_train_pred.extend(train_snapshot_avg['y_pred'].tolist())
                                    all_test_true.extend(test_snapshot_avg['y_true'].tolist())
                                    all_test_pred.extend(test_snapshot_avg['y_pred'].tolist())
            
            if all_train_true and all_test_true:
                # Convert to numpy arrays for easier manipulation
                train_true = np.array(all_train_true)
                train_pred = np.array(all_train_pred)
                test_true = np.array(all_test_true)
                test_pred = np.array(all_test_pred)
                
                # Calculate comprehensive statistics
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                # Use pre-calculated stats for consistent fold-wise averaging
                if config['data_type'] == 'original_voxel' and self.summary_stats:
                    # For deltaEint data, use raw summary stats
                    train_mae = self.summary_stats.get('train_mae', {}).get('mean', mean_absolute_error(train_true, train_pred))
                    test_mae = self.summary_stats.get('test_mae', {}).get('mean', mean_absolute_error(test_true, test_pred))
                elif config['data_type'] == 'original_snapshot_avg' and hasattr(self, 'snapshot_avg_stats') and self.snapshot_avg_stats:
                    # For deltaEsol data, use snapshot-averaged stats
                    train_mae = self.snapshot_avg_stats.get('mae', {}).get('train_mean', mean_absolute_error(train_true, train_pred))
                    test_mae = self.snapshot_avg_stats.get('mae', {}).get('test_mean', mean_absolute_error(test_true, test_pred))
                else:
                    train_mae = mean_absolute_error(train_true, train_pred)
                    test_mae = mean_absolute_error(test_true, test_pred)
                
                train_rmse = np.sqrt(mean_squared_error(train_true, train_pred))
                train_r2 = r2_score(train_true, train_pred)
                
                test_rmse = np.sqrt(mean_squared_error(test_true, test_pred))
                test_r2 = r2_score(test_true, test_pred)
                
                # Set up the plot area with clean white background
                ax.set_facecolor('white')
                ax.set_axisbelow(True)
                
                # Plot data points with enhanced styling
                train_scatter = ax.scatter(train_true, train_pred,
                                         alpha=config['alpha_train'], 
                                         s=config['marker_size'],
                                         c=colors['train'], 
                                         marker='^',
                                         edgecolors='white', 
                                         linewidths=0.3)
                
                test_scatter = ax.scatter(test_true, test_pred,
                                        alpha=config['alpha_test'], 
                                        s=config['marker_size'],
                                        c=colors['test'], 
                                        marker='o',
                                        edgecolors='white', 
                                        linewidths=0.3)
                
                # Calculate plot limits with padding
                all_values = np.concatenate([train_true, train_pred, test_true, test_pred])
                data_min, data_max = np.min(all_values), np.max(all_values)
                padding = (data_max - data_min) * 0.05
                plot_min = data_min - padding
                plot_max = data_max + padding
                
                # Perfect prediction line (unity line) with enhanced styling
                unity_line = ax.plot([plot_min, plot_max], [plot_min, plot_max], 
                                   color=colors['unity'], 
                                   linewidth=2, 
                                   linestyle='--', 
                                   alpha=0.8,
                                   zorder=5)
                
                # Add regression confidence bands for the test data
                if len(test_true) > 50:  # Only for sufficient data points  
                    from scipy import stats
                    
                    # Perform linear regression on test data
                    slope, intercept, r_value, p_value, std_err = stats.linregress(test_true, test_pred)
                    
                    # Calculate regression line
                    x_reg = np.linspace(plot_min, plot_max, 100)
                    y_reg = slope * x_reg + intercept
                    
                    # Calculate confidence interval for regression line
                    n = len(test_true)
                    x_mean = np.mean(test_true)
                    sum_x_sq = np.sum((test_true - x_mean)**2)
                    
                    # Standard error of regression
                    residuals = test_pred - (slope * test_true + intercept)
                    mse = np.sum(residuals**2) / (n - 2) if n > 2 else np.var(residuals)
                    
                    # Calculate confidence bands (95% confidence)
                    t_val = stats.t.ppf(0.975, max(n-2, 1))  # Protect against n<2
                    
                    # Standard error at each point
                    se_reg = np.sqrt(mse * (1/n + (x_reg - x_mean)**2 / max(sum_x_sq, 1e-8)))
                    
                    # Confidence bands
                    upper_band = y_reg + t_val * se_reg
                    lower_band = y_reg - t_val * se_reg
                    
                    # Fill confidence region (much lighter than your green example)
                    ax.fill_between(x_reg, lower_band, upper_band, 
                                  alpha=0.2, color=colors['test'], 
                                  label='95% Confidence Band')
                
                # Professional axis formatting
                ax.set_xlim(plot_min, plot_max)
                ax.set_ylim(plot_min, plot_max)
                ax.set_aspect('equal', adjustable='box')
                
                # Enhanced labels with proper LaTeX formatting
                ax.set_xlabel(config['xlabel'], 
                             fontsize=self.font_size, fontweight='normal')
                ax.set_ylabel(config['ylabel'], 
                             fontsize=self.font_size, fontweight='normal')
                
                # Simple legend with only test metrics
                legend_lines = [
                    f"Test MAE: {test_mae:.3f} eV",
                    f"Test R²: {test_r2:.3f}"
                ]
                
                # Create a simple text box for legend (adjusted position)
                legend_text = '\n'.join(legend_lines)
                ax.text(0.07, 0.93, legend_text,
                       transform=ax.transAxes,
                       fontsize=self.font_size,
                       verticalalignment='top',
                       horizontalalignment='left',
                       bbox=dict(boxstyle='round,pad=0.5',
                                facecolor='white',
                                edgecolor='black',
                                alpha=0.9,
                                linewidth=1.0))
                
                # Professional subplot label
                ax.text(num_pos_x, num_pos_y, config['label'], 
                       transform=ax.transAxes,
                       fontsize=self.font_size, 
                       fontweight='bold',
                       verticalalignment='top', 
                       horizontalalignment='left')
                
                # Enhanced tick formatting
                ax.tick_params(axis='both', which='major', labelsize=self.font_size)
                
                if self.verbose:
                    print(f"    {config['label']} {config['data_type']}: Train={len(train_true)}, Test={len(test_true)}")
        
        plt.tight_layout()
        
        # Save the figure with job code prefix (matching TTA version style)
        if save_plot:
            # Extract job code from pkl_name (same logic as other functions)
            job_code = self.pkl_name.split('-')[0] if '-' in self.pkl_name else self.pkl_name.replace('.pkl', '')
            save_filename = f"{job_code}-{self.split_type}_split-parity_plot_no_test_augment.png"
            full_save_path = os.path.join(self.output_figure_path, 'cnn_results', save_filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
            
            plt.savefig(full_save_path, dpi=1000, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            
            if self.verbose:
                print(f"    Professional No-TTA parity plot saved to: {full_save_path}")
        
        if show_plot:
            plt.show()
        
        # Restore original matplotlib settings
        plt.rcParams.update(original_rcParams)


    def plot_parity_plot_no_test_augment_with_Train_with_MD_3_subfigures(self,
                                                                         figsize: Tuple[int, int] = (24, 8),
                                                                         linear_fit: bool = False,
                                                                         confidence_band: bool = False,
                                                                         shade: str = 'prediction_interval',
                                                                         ml_slope_x: float = 0.1,
                                                                         ml_slope_y: float = 0.02,
                                                                         md_slope_x: float = 0.015,
                                                                         md_slope_y: float = 0.5,
                                                                         show_plot: bool = False,
                                                                         save_plot: bool = False):
        """
        Professional-grade parity plots without test time augmentation - using only original voxels
        Enhanced version with MD data comparison in third subplot
        Creates three subplots (a), (b), (c) where first two match no_test_augment and third shows ML vs MD
        
        Args:
            figsize: Figure size for the overall plot
            linear_fit: Whether to add linear regression lines with slope values (default: False)
            confidence_band: Whether to add shaded uncertainty bands (default: False)
            shade: Type of uncertainty band to display (default: 'prediction_interval')
                - 'confidence_interval': 95% confidence band for regression line (narrower, shows parameter uncertainty)
                - 'prediction_interval': 95% prediction interval for individual points (wider, shows prediction uncertainty)
            ml_slope_x: X offset for ML slope text position (default: 0.1)
            ml_slope_y: Y offset for ML slope text position (default: 0.02)
            md_slope_x: X offset for MD slope text position (default: 0.015)
            md_slope_y: Y offset for MD slope text position (default: 0.5)
            show_plot: Whether to show the plot (default: False)
            save_plot: Whether to save the plot (default: False)
        """
        # Validate shade parameter
        valid_shade_types = ['confidence_interval', 'prediction_interval']
        if shade not in valid_shade_types:
            raise ValueError(f"shade must be one of {valid_shade_types}, got '{shade}'")
        
        if not self.model_storage:
            print("No results to plot")
            return

        # Professional color scheme (colorblind-friendly)
        colors = {
            'train': self.blue_color,    # Use class attribute
            'test': self.red_color,      # Use class attribute
            'md': self.green_color,      # Use class attribute
            'unity': '#2c2c2c',          # Dark gray for unity line
            'stats_bg': '#f8f9fa'        # Light background for stats box
        }
        
        # Keep original matplotlib settings (minimal changes)
        original_rcParams = plt.rcParams.copy()  # Save original settings

        # subplot number positions
        num_pos_x = -0.235
        num_pos_y = 1.05
        
        # Create figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        if self.verbose:
            print(f"\n--- Professional No-TTA with MD Parity Plot Data Points:")

        # Plot configurations for each subplot - different point sizes for deltaEint vs deltaEsol
        plot_configs = [
            {
                'data_type': 'original_voxel',
                'label': '(a)',
                'alpha_train': self.alpha_train,         # Use class attribute directly
                'alpha_test': self.alpha,                # Unified: test uses base alpha
                'marker_size': self.point_size_deltaEint, # Use deltaEint point size (more data points)
                'xlabel': r'DFT calculated $\Delta E^{\mathit{DFT}}_{\mathit{int}}$ (eV)',
                'ylabel': r'ML predicted $\Delta E^{\mathit{ML}}_{\mathit{int}}$ (eV)'
            },
            {
                'data_type': 'original_snapshot_avg',
                'label': '(b)',
                'alpha_train': self.alpha_train,     # Use class attribute directly
                'alpha_test': self.alpha,            # Unified: test uses base alpha
                'marker_size': self.point_size,      # Use regular point size for deltaEsol (fewer data points)
                'xlabel': r'DFT calculated $\Delta E^{\mathit{DFT}}_{\mathit{sol}}$ (eV)',
                'ylabel': r'ML predicted $\Delta E^{\mathit{ML}}_{\mathit{sol}}$ (eV)'
            },
            {
                'data_type': 'ml_vs_md',
                'label': '(c)',
                'alpha_train': self.alpha_train,     # Use class attribute directly (not used in this plot)
                'alpha_test': self.alpha,            # Unified: ML test uses base alpha
                'alpha_md': self.alpha,              # Unified: MD uses base alpha (same as test)
                'marker_size': self.point_size,      # Use regular point size for ML vs MD comparison
                'xlabel': r'DFT calculated $\Delta E^{\mathit{DFT}}_{\mathit{sol}}$ (eV)',
                'ylabel': r'ML/MD predicted $\Delta E_{\mathit{sol}}$ (eV)'
            }
        ]
        
        # Collect snapshot averaged data for MD comparison
        snapshot_avg_data = None
        
        for idx, config in enumerate(plot_configs):
            ax = axes[idx]
            
            # Collect data based on type
            all_train_true, all_train_pred = [], []
            all_test_true, all_test_pred = [], []
            
            for fold_idx, fold_results in self.model_storage.items():
                if 'df_train' in fold_results and 'df_test' in fold_results:
                    df_train = fold_results['df_train']
                    df_test = fold_results['df_test']
                    
                    if config['data_type'] == 'original_voxel':
                        if 'y_true' in df_train.columns and 'y_pred' in df_train.columns:
                            # Use original voxel data (first voxel of each snapshot, assuming it's original)
                            # This is a simplified approach - in practice you might need voxel identifiers
                            train_sample = df_train.groupby(['zeolite', 'environment', 'adsorbate', 'snapshot']).first().reset_index()
                            test_sample = df_test.groupby(['zeolite', 'environment', 'adsorbate', 'snapshot']).first().reset_index()
                            
                            all_train_true.extend(train_sample['y_true'].tolist())
                            all_train_pred.extend(train_sample['y_pred'].tolist())
                            all_test_true.extend(test_sample['y_true'].tolist())
                            all_test_pred.extend(test_sample['y_pred'].tolist())
                            
                    elif config['data_type'] == 'original_snapshot_avg':
                        if 'y_true' in df_train.columns and 'y_pred' in df_train.columns:
                            # Average across snapshots for each environment-adsorbate combination
                            train_avg = df_train.groupby(['zeolite', 'environment', 'adsorbate', 'snapshot']).first().reset_index()
                            train_avg = train_avg.groupby(['zeolite', 'environment', 'adsorbate']).agg({
                                'y_true': 'mean',
                                'y_pred': 'mean'
                            }).reset_index()
                            
                            test_avg = df_test.groupby(['zeolite', 'environment', 'adsorbate', 'snapshot']).first().reset_index()
                            test_avg = test_avg.groupby(['zeolite', 'environment', 'adsorbate']).agg({
                                'y_true': 'mean',
                                'y_pred': 'mean'
                            }).reset_index()
                            
                            # Store test data for MD comparison
                            if snapshot_avg_data is None:
                                snapshot_avg_data = test_avg.copy()
                            else:
                                snapshot_avg_data = pd.concat([snapshot_avg_data, test_avg], ignore_index=True)
                            
                            all_train_true.extend(train_avg['y_true'].tolist())
                            all_train_pred.extend(train_avg['y_pred'].tolist())
                            all_test_true.extend(test_avg['y_true'].tolist())
                            all_test_pred.extend(test_avg['y_pred'].tolist())
                            
                    elif config['data_type'] == 'ml_vs_md':
                        # This will be handled separately after the loop
                        continue
            
            # Handle ML vs MD comparison for third subplot
            if config['data_type'] == 'ml_vs_md' and self.md_data is not None and snapshot_avg_data is not None:
                # Remove duplicates and prepare for MD matching
                test_unique_clean = snapshot_avg_data.drop_duplicates()
                
                # Create comparison data for matching
                test_combinations = test_unique_clean[['zeolite', 'environment', 'adsorbate', 'y_true', 'y_pred']].copy()
                test_combinations.rename(columns={'y_true': 'dft_true', 'y_pred': 'ml_pred'}, inplace=True)
                
                # Merge with MD data
                md_matched = test_combinations.merge(
                    self.md_data, 
                    on=['zeolite', 'environment', 'adsorbate'], 
                    how='inner'
                )
                
                if len(md_matched) > 0:
                    # Plot ML Test vs DFT
                    ml_dft_true = md_matched['dft_true'].values
                    ml_dft_pred = md_matched['ml_pred'].values
                    
                    # Plot MD vs DFT  
                    md_dft_true = md_matched['dft_true'].values
                    md_dft_pred = md_matched['intE_MD'].values
                    
                    all_values = [ml_dft_true, ml_dft_pred, md_dft_true, md_dft_pred]
                    
                    if self.verbose:
                        print(f"    {config['label']} ML vs MD comparison: {len(md_matched)} data points")
                else:
                    if self.verbose:
                        print(f"    {config['label']} Warning: No MD data matches found")
                    continue
            
            # Plot regular data (first two subplots) or ML vs MD (third subplot)
            if config['data_type'] != 'ml_vs_md':
                if all_train_true and all_test_true:
                    # Convert to numpy arrays for easier manipulation
                    train_true = np.array(all_train_true)
                    train_pred = np.array(all_train_pred)
                    test_true = np.array(all_test_true)
                    test_pred = np.array(all_test_pred)
                    
                    all_values = [train_true, train_pred, test_true, test_pred]
                    
                    # Calculate comprehensive statistics
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    
                    # Use pre-calculated stats for consistent fold-wise averaging
                    if config['data_type'] == 'original_voxel' and self.summary_stats:
                        # For deltaEint data, use raw summary stats
                        train_mae = self.summary_stats.get('train_mae', {}).get('mean', mean_absolute_error(train_true, train_pred))
                        test_mae = self.summary_stats.get('test_mae', {}).get('mean', mean_absolute_error(test_true, test_pred))
                    elif config['data_type'] == 'original_snapshot_avg' and hasattr(self, 'snapshot_avg_stats') and self.snapshot_avg_stats:
                        # For deltaEsol data, use snapshot-averaged stats
                        train_mae = self.snapshot_avg_stats.get('mae', {}).get('train_mean', mean_absolute_error(train_true, train_pred))
                        test_mae = self.snapshot_avg_stats.get('mae', {}).get('test_mean', mean_absolute_error(test_true, test_pred))
                    else:
                        train_mae = mean_absolute_error(train_true, train_pred)
                        test_mae = mean_absolute_error(test_true, test_pred)
                    
                    test_rmse = np.sqrt(mean_squared_error(test_true, test_pred))
                    test_r2 = r2_score(test_true, test_pred)
                    
                    # Set up the plot area with clean white background
                    ax.set_facecolor('white')
                    ax.set_axisbelow(True)
                    
                    # Plot data points with enhanced styling and integrated legend
                    ax.scatter(train_true, train_pred,
                             alpha=config['alpha_train'], 
                             s=config['marker_size'],
                             c=colors['train'], 
                             marker='^',
                             edgecolors='white', 
                             linewidths=0.3,
                             label=f'Train (MAE: {train_mae:.3f} eV)')
                    
                    ax.scatter(test_true, test_pred,
                            alpha=config['alpha_test'], 
                            s=config['marker_size'],
                            c=colors['test'], 
                            marker='o',
                            edgecolors='white', 
                            linewidths=0.3,
                            label=f'Test (MAE: {test_mae:.3f} eV)')
                    
                    # Create integrated legend
                    ax.legend(fontsize=self.font_size, 
                             loc='upper left', 
                             handlelength=1, 
                             handletextpad=0.3,
                             framealpha=0.9)
                    
                    # Add uncertainty bands for first two subplots only (a) and (b)
                    if confidence_band and config['data_type'] in ['original_voxel', 'original_snapshot_avg']:
                        from scipy import stats
                        
                        if len(test_true) > 3:  # Need sufficient data points
                            # Linear regression for test data
                            slope, intercept, r_value, p_value, std_err = stats.linregress(test_true, test_pred)
                            
                            # Calculate plot limits for uncertainty bands
                            data_min, data_max = np.min(test_true), np.max(test_true)
                            x_smooth = np.linspace(data_min, data_max, 100)
                            
                            # Calculate residuals and MSE
                            test_predicted = slope * test_true + intercept
                            residuals = test_pred - test_predicted
                            mse = np.sum(residuals**2) / max(len(test_true) - 2, 1)
                            
                            # Calculate appropriate uncertainty band based on shade parameter
                            t_val = stats.t.ppf(0.975, max(len(test_true) - 2, 1))
                            
                            if shade == 'confidence_interval':
                                # Confidence interval for regression line (narrower)
                                # Quantifies uncertainty in the mean predicted trend (regression parameters)
                                uncertainty_std = np.sqrt(mse * (1/len(test_true) + 
                                                                (x_smooth - np.mean(test_true))**2 / 
                                                                np.sum((test_true - np.mean(test_true))**2)))
                                band_label = '95% Confidence Band'
                                
                            elif shade == 'prediction_interval':
                                # Prediction interval for individual predictions (wider)
                                # Quantifies uncertainty for new individual observations
                                # KEY: Add "1 +" term to account for residual scatter
                                uncertainty_std = np.sqrt(mse * (1 + 1/len(test_true) + 
                                                                (x_smooth - np.mean(test_true))**2 / 
                                                                np.sum((test_true - np.mean(test_true))**2)))
                                band_label = '95% Prediction Interval'
                            
                            # Calculate bands
                            y_pred = slope * x_smooth + intercept
                            upper_band = y_pred + t_val * uncertainty_std
                            lower_band = y_pred - t_val * uncertainty_std
                            
                            # Fill uncertainty region for test data
                            ax.fill_between(x_smooth, lower_band, upper_band, 
                                          alpha=0.15, color=colors['test'], 
                                          label=band_label, zorder=1)
                            
                            # Optional: Print coverage statistics in verbose mode
                            if self.verbose and shade == 'prediction_interval':
                                # Calculate empirical coverage rate
                                from scipy.interpolate import interp1d
                                f_upper = interp1d(x_smooth, upper_band, bounds_error=False, fill_value='extrapolate')
                                f_lower = interp1d(x_smooth, lower_band, bounds_error=False, fill_value='extrapolate')
                                
                                upper_at_test = f_upper(test_true)
                                lower_at_test = f_lower(test_true)
                                
                                within_interval = (test_pred >= lower_at_test) & (test_pred <= upper_at_test)
                                coverage = np.mean(within_interval) * 100
                                
                                print(f"    {config['label']} Prediction Interval Coverage: {coverage:.1f}% (expected: ~95%)")
                    
            else:  # ML vs MD subplot
                if 'md_matched' in locals() and len(md_matched) > 0:
                    # Set up the plot area with clean white background
                    ax.set_facecolor('white')
                    ax.set_axisbelow(True)
                    
                    # Calculate metrics for both ML and MD
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    
                    # For ML, use pre-calculated snapshot-averaged fold-wise stats if available
                    if hasattr(self, 'snapshot_avg_stats') and self.snapshot_avg_stats:
                        ml_mae = self.snapshot_avg_stats.get('mae', {}).get('test_mean', mean_absolute_error(ml_dft_true, ml_dft_pred))
                        ml_rmse = self.snapshot_avg_stats.get('rmse', {}).get('test_mean', np.sqrt(mean_squared_error(ml_dft_true, ml_dft_pred)))
                        ml_r2 = self.snapshot_avg_stats.get('r2', {}).get('test_mean', r2_score(ml_dft_true, ml_dft_pred))
                    else:
                        ml_mae = mean_absolute_error(ml_dft_true, ml_dft_pred)
                        ml_rmse = np.sqrt(mean_squared_error(ml_dft_true, ml_dft_pred))
                        ml_r2 = r2_score(ml_dft_true, ml_dft_pred)
                    
                    # For MD, use real-time calculation (no pre-calculated fold stats for MD)
                    md_mae = mean_absolute_error(md_dft_true, md_dft_pred)
                    md_rmse = np.sqrt(mean_squared_error(md_dft_true, md_dft_pred))
                    md_r2 = r2_score(md_dft_true, md_dft_pred)
                    
                    # Print detailed MD performance metrics for ΔEsol prediction
                    if self.verbose:
                        print(f"\n--- MD vs DFT ΔEsol Prediction Performance ---")
                        print(f"    ML   - RMSE: {ml_rmse:.3f} eV, MAE: {ml_mae:.3f} eV, R²: {ml_r2:.3f}")
                        print(f"    MD   - RMSE: {md_rmse:.3f} eV, MAE: {md_mae:.3f} eV, R²: {md_r2:.3f}")
                    
                    # Plot ML Test data and MD data with unified alpha values
                    ax.scatter(ml_dft_true, ml_dft_pred,
                              alpha=config['alpha_test'],  # Use unified alpha for test
                              s=config['marker_size'],
                              c=colors['test'], 
                              marker='o',
                              edgecolors='white', 
                              linewidths=0.3,
                              label=f'ML (MAE: {ml_mae:.3f} eV)')
                    
                    ax.scatter(md_dft_true, md_dft_pred,
                              alpha=config['alpha_md'],    # Use unified alpha for MD (same as test)
                              s=config['marker_size'],
                              c=colors['md'], 
                              marker='^',
                              edgecolors='white', 
                              linewidths=0.3,
                              label=f'MD (MAE: {md_mae:.3f} eV)')
                    
                    if linear_fit:
                        # Add linear fitting for both ML and MD data
                        # Determine plot limits for fitting lines
                        all_x_values = np.concatenate([ml_dft_true, md_dft_true])
                        x_min, x_max = np.min(all_x_values), np.max(all_x_values)
                        lims = np.array([x_min, x_max])
                        
                        # ML linear fit
                        z_ml = np.polyfit(ml_dft_true, ml_dft_pred, 1)
                        p_ml = np.poly1d(z_ml)
                        ax.plot(lims, p_ml(lims), linestyle='--', color=colors['test'], 
                            alpha=self.alpha, linewidth=2)
                        ax.text(lims[0] + ml_slope_x, p_ml(lims[0] + ml_slope_y), 
                            f'ML slope={z_ml[0]:.2f}', 
                            color=colors['test'], fontsize=self.font_size)
                        
                        # MD linear fit
                        z_md = np.polyfit(md_dft_true, md_dft_pred, 1)
                        p_md = np.poly1d(z_md)
                        ax.plot(lims, p_md(lims), linestyle='--', color=colors['md'], 
                            alpha=self.alpha, linewidth=2)
                        ax.text(lims[0] + md_slope_x, p_md(lims[0] + md_slope_y), 
                            f'MD slope={z_md[0]:.2f}', 
                            color=colors['md'], fontsize=self.font_size)
                    
                    # Create integrated legend
                    ax.legend(fontsize=self.font_size, 
                             loc='upper left', 
                             handlelength=1, 
                             handletextpad=0.3,
                             framealpha=0.9)
                else:
                    # No MD data available
                    ax.text(0.5, 0.5, 'No MD data available',
                           transform=ax.transAxes,
                           fontsize=self.font_size,
                           ha='center', va='center')
                    all_values = [0, 1]  # Dummy values for plot limits
            
            # Calculate plot limits with padding
            if 'all_values' in locals() and all_values:
                all_values_concat = np.concatenate([np.array(v) for v in all_values if len(v) > 0])
                if len(all_values_concat) > 0:
                    data_min, data_max = np.min(all_values_concat), np.max(all_values_concat)
                    padding = (data_max - data_min) * 0.05
                    plot_min = data_min - padding
                    plot_max = data_max + padding
                    
                    # Perfect prediction line (unity line) with enhanced styling
                    unity_line = ax.plot([plot_min, plot_max], [plot_min, plot_max], 
                                       color=colors['unity'], 
                                       linewidth=2, 
                                       linestyle='--', 
                                       alpha=0.8,
                                       zorder=5)
                    
                    # Professional axis formatting
                    ax.set_xlim(plot_min, plot_max)
                    ax.set_ylim(plot_min, plot_max)
                    ax.set_aspect('equal', adjustable='box')
            
            # Enhanced labels with proper LaTeX formatting
            ax.set_xlabel(config['xlabel'], 
                         fontsize=self.font_size, fontweight='normal')
            ax.set_ylabel(config['ylabel'], 
                         fontsize=self.font_size, fontweight='normal')
            
            # Professional subplot label
            ax.text(num_pos_x, num_pos_y, config['label'], 
                   transform=ax.transAxes,
                   fontsize=self.font_size, 
                   fontweight='bold',
                   verticalalignment='top', 
                   horizontalalignment='left')
            
            # Enhanced tick formatting
            ax.tick_params(axis='both', which='major', labelsize=self.font_size)
            
            if self.verbose and config['data_type'] != 'ml_vs_md':
                if all_train_true and all_test_true:
                    print(f"    {config['label']} {config['data_type']}: Train={len(train_true)}, Test={len(test_true)}")
                    
                    # Print detailed metrics for ΔEsol prediction (snapshot-averaged data)
                    if config['data_type'] == 'original_snapshot_avg':
                        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                        
                        # Use pre-calculated snapshot-averaged fold-wise stats for consistency
                        if hasattr(self, 'snapshot_avg_stats') and self.snapshot_avg_stats:
                            train_rmse_sol = self.snapshot_avg_stats.get('rmse', {}).get('train_mean', np.sqrt(mean_squared_error(train_true, train_pred)))
                            train_mae_sol = self.snapshot_avg_stats.get('mae', {}).get('train_mean', mean_absolute_error(train_true, train_pred))
                            train_r2_sol = self.snapshot_avg_stats.get('r2', {}).get('train_mean', r2_score(train_true, train_pred))
                            
                            test_rmse_sol = self.snapshot_avg_stats.get('rmse', {}).get('test_mean', np.sqrt(mean_squared_error(test_true, test_pred)))
                            test_mae_sol = self.snapshot_avg_stats.get('mae', {}).get('test_mean', mean_absolute_error(test_true, test_pred))
                            test_r2_sol = self.snapshot_avg_stats.get('r2', {}).get('test_mean', r2_score(test_true, test_pred))
                        else:
                            # Fallback to real-time calculation if pre-calculated stats not available
                            train_rmse_sol = np.sqrt(mean_squared_error(train_true, train_pred))
                            train_mae_sol = mean_absolute_error(train_true, train_pred)
                            train_r2_sol = r2_score(train_true, train_pred)
                            
                            test_rmse_sol = np.sqrt(mean_squared_error(test_true, test_pred))
                            test_mae_sol = mean_absolute_error(test_true, test_pred)
                            test_r2_sol = r2_score(test_true, test_pred)
                        
                        print(f"\n--- ΔEsol Prediction Performance (Snapshot-averaged) ---")
                        print(f"    Train - RMSE: {train_rmse_sol:.3f} eV, MAE: {train_mae_sol:.3f} eV, R²: {train_r2_sol:.3f}")
                        print(f"    Test  - RMSE: {test_rmse_sol:.3f} eV, MAE: {test_mae_sol:.3f} eV, R²: {test_r2_sol:.3f}")
        
        plt.tight_layout()
        
        # Save with high quality settings
        if save_plot:
            parts = self.pkl_name.split('-')
            job_code = '-'.join(parts[:3])
            file_name = f"{job_code}-parity_plot_no_tta_with_md.png"
            save_path = os.path.join(self.output_figure_path, 'cnn_results', file_name)
            
            plt.savefig(save_path, dpi=1000, bbox_inches='tight')
            
            if self.verbose:
                print(f"No-TTA with MD parity plot saved to: {save_path}")
        
        # Reset matplotlib parameters to avoid affecting other plots
        plt.rcParams.update(original_rcParams)

        if show_plot:
            plt.show()


    def plot_parity_plot_no_test_augment_with_Train_with_MD_3_subfigures_vertical(self,
                                                                         figsize: Tuple[int, int] = (8, 24),
                                                                         linear_fit: bool = False,
                                                                         confidence_band: bool = False,
                                                                         shade: str = 'prediction_interval',
                                                                         ml_slope_x: float = 0.1,
                                                                         ml_slope_y: float = 0.02,
                                                                         md_slope_x: float = 0.015,
                                                                         md_slope_y: float = 0.5,
                                                                         show_plot: bool = False,
                                                                         save_plot: bool = False):
        """
        Professional-grade parity plots without test time augmentation - using only original voxels
        Enhanced version with MD data comparison in third subplot
        Creates three subplots (a), (b), (c) arranged vertically from top to bottom
        
        Args:
            figsize: Figure size for the overall plot
            linear_fit: Whether to add linear regression lines with slope values (default: False)
            confidence_band: Whether to add shaded uncertainty bands (default: False)
            shade: Type of uncertainty band to display (default: 'prediction_interval')
                - 'confidence_interval': 95% confidence band for regression line (narrower, shows parameter uncertainty)
                - 'prediction_interval': 95% prediction interval for individual points (wider, shows prediction uncertainty)
            ml_slope_x: X offset for ML slope text position (default: 0.1)
            ml_slope_y: Y offset for ML slope text position (default: 0.02)
            md_slope_x: X offset for MD slope text position (default: 0.015)
            md_slope_y: Y offset for MD slope text position (default: 0.5)
            show_plot: Whether to show the plot (default: False)
            save_plot: Whether to save the plot (default: False)
        """
        # Validate shade parameter
        valid_shade_types = ['confidence_interval', 'prediction_interval']
        if shade not in valid_shade_types:
            raise ValueError(f"shade must be one of {valid_shade_types}, got '{shade}'")
        
        if not self.model_storage:
            print("No results to plot")
            return

        # Professional color scheme (colorblind-friendly)
        colors = {
            'train': self.blue_color,    # Use class attribute
            'test': self.red_color,      # Use class attribute
            'md': self.green_color,      # Use class attribute
            'unity': '#2c2c2c',          # Dark gray for unity line
            'stats_bg': '#f8f9fa'        # Light background for stats box
        }
        
        # Keep original matplotlib settings (minimal changes)
        original_rcParams = plt.rcParams.copy()  # Save original settings

        # subplot number positions
        num_pos_x = -0.235
        num_pos_y = 1.05
        
        # Create figure with three subplots arranged vertically
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        if self.verbose:
            print(f"\n--- Professional No-TTA with MD Parity Plot Data Points (Vertical):")

        # Plot configurations for each subplot - different point sizes for deltaEint vs deltaEsol
        plot_configs = [
            {
                'data_type': 'original_voxel',
                'label': '(a)',
                'alpha_train': self.alpha_train,         # Use class attribute directly
                'alpha_test': self.alpha,                # Unified: test uses base alpha
                'marker_size': self.point_size_deltaEint, # Use deltaEint point size (more data points)
                'xlabel': r'DFT calculated $\Delta E^{\mathit{DFT}}_{\mathit{int}}$ (eV)',
                'ylabel': r'ML predicted $\Delta E^{\mathit{ML}}_{\mathit{int}}$ (eV)'
            },
            {
                'data_type': 'original_snapshot_avg',
                'label': '(b)',
                'alpha_train': self.alpha_train,     # Use class attribute directly
                'alpha_test': self.alpha,            # Unified: test uses base alpha
                'marker_size': self.point_size,      # Use regular point size for deltaEsol (fewer data points)
                'xlabel': r'DFT calculated $\Delta E^{\mathit{DFT}}_{\mathit{sol}}$ (eV)',
                'ylabel': r'ML predicted $\Delta E^{\mathit{ML}}_{\mathit{sol}}$ (eV)'
            },
            {
                'data_type': 'ml_vs_md',
                'label': '(c)',
                'alpha_train': self.alpha_train,     # Use class attribute directly (not used in this plot)
                'alpha_test': self.alpha,            # Unified: ML test uses base alpha
                'alpha_md': self.alpha,              # Unified: MD uses base alpha (same as test)
                'marker_size': self.point_size,      # Use regular point size for ML vs MD comparison
                'xlabel': r'DFT calculated $\Delta E^{\mathit{DFT}}_{\mathit{sol}}$ (eV)',
                'ylabel': r'ML/MD predicted $\Delta E_{\mathit{sol}}$ (eV)'
            }
        ]
        
        # Collect snapshot averaged data for MD comparison
        snapshot_avg_data = None
        
        for idx, config in enumerate(plot_configs):
            ax = axes[idx]
            
            # Collect data based on type
            all_train_true, all_train_pred = [], []
            all_test_true, all_test_pred = [], []
            
            for fold_idx, fold_results in self.model_storage.items():
                if 'df_train' in fold_results and 'df_test' in fold_results:
                    df_train = fold_results['df_train']
                    df_test = fold_results['df_test']
                    
                    if config['data_type'] == 'original_voxel':
                        if 'y_true' in df_train.columns and 'y_pred' in df_train.columns:
                            # Use original voxel data (first voxel of each snapshot, assuming it's original)
                            # This is a simplified approach - in practice you might need voxel identifiers
                            train_sample = df_train.groupby(['zeolite', 'environment', 'adsorbate', 'snapshot']).first().reset_index()
                            test_sample = df_test.groupby(['zeolite', 'environment', 'adsorbate', 'snapshot']).first().reset_index()
                            
                            all_train_true.extend(train_sample['y_true'].tolist())
                            all_train_pred.extend(train_sample['y_pred'].tolist())
                            all_test_true.extend(test_sample['y_true'].tolist())
                            all_test_pred.extend(test_sample['y_pred'].tolist())
                            
                    elif config['data_type'] == 'original_snapshot_avg':
                        if 'y_true' in df_train.columns and 'y_pred' in df_train.columns:
                            # Average across snapshots for each environment-adsorbate combination
                            train_avg = df_train.groupby(['zeolite', 'environment', 'adsorbate', 'snapshot']).first().reset_index()
                            train_avg = train_avg.groupby(['zeolite', 'environment', 'adsorbate']).agg({
                                'y_true': 'mean',
                                'y_pred': 'mean'
                            }).reset_index()
                            
                            test_avg = df_test.groupby(['zeolite', 'environment', 'adsorbate', 'snapshot']).first().reset_index()
                            test_avg = test_avg.groupby(['zeolite', 'environment', 'adsorbate']).agg({
                                'y_true': 'mean',
                                'y_pred': 'mean'
                            }).reset_index()
                            
                            # Store test data for MD comparison
                            if snapshot_avg_data is None:
                                snapshot_avg_data = test_avg.copy()
                            else:
                                snapshot_avg_data = pd.concat([snapshot_avg_data, test_avg], ignore_index=True)
                            
                            all_train_true.extend(train_avg['y_true'].tolist())
                            all_train_pred.extend(train_avg['y_pred'].tolist())
                            all_test_true.extend(test_avg['y_true'].tolist())
                            all_test_pred.extend(test_avg['y_pred'].tolist())
                            
                    elif config['data_type'] == 'ml_vs_md':
                        # This will be handled separately after the loop
                        continue
            
            # Handle ML vs MD comparison for third subplot
            if config['data_type'] == 'ml_vs_md' and self.md_data is not None and snapshot_avg_data is not None:
                # Remove duplicates and prepare for MD matching
                test_unique_clean = snapshot_avg_data.drop_duplicates()
                
                # Create comparison data for matching
                test_combinations = test_unique_clean[['zeolite', 'environment', 'adsorbate', 'y_true', 'y_pred']].copy()
                test_combinations.rename(columns={'y_true': 'dft_true', 'y_pred': 'ml_pred'}, inplace=True)
                
                # Merge with MD data
                md_matched = test_combinations.merge(
                    self.md_data, 
                    on=['zeolite', 'environment', 'adsorbate'], 
                    how='inner'
                )
                
                if len(md_matched) > 0:
                    # Plot ML Test vs DFT
                    ml_dft_true = md_matched['dft_true'].values
                    ml_dft_pred = md_matched['ml_pred'].values
                    
                    # Plot MD vs DFT  
                    md_dft_true = md_matched['dft_true'].values
                    md_dft_pred = md_matched['intE_MD'].values
                    
                    all_values = [ml_dft_true, ml_dft_pred, md_dft_true, md_dft_pred]
                    
                    if self.verbose:
                        print(f"    {config['label']} ML vs MD comparison: {len(md_matched)} data points")
                else:
                    if self.verbose:
                        print(f"    {config['label']} Warning: No MD data matches found")
                    continue
            
            # Plot regular data (first two subplots) or ML vs MD (third subplot)
            if config['data_type'] != 'ml_vs_md':
                if all_train_true and all_test_true:
                    # Convert to numpy arrays for easier manipulation
                    train_true = np.array(all_train_true)
                    train_pred = np.array(all_train_pred)
                    test_true = np.array(all_test_true)
                    test_pred = np.array(all_test_pred)
                    
                    all_values = [train_true, train_pred, test_true, test_pred]
                    
                    # Calculate comprehensive statistics
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    
                    # Use pre-calculated stats for consistent fold-wise averaging
                    if config['data_type'] == 'original_voxel' and self.summary_stats:
                        # For deltaEint data, use raw summary stats
                        train_mae = self.summary_stats.get('train_mae', {}).get('mean', mean_absolute_error(train_true, train_pred))
                        test_mae = self.summary_stats.get('test_mae', {}).get('mean', mean_absolute_error(test_true, test_pred))
                    elif config['data_type'] == 'original_snapshot_avg' and hasattr(self, 'snapshot_avg_stats') and self.snapshot_avg_stats:
                        # For deltaEsol data, use snapshot-averaged stats
                        train_mae = self.snapshot_avg_stats.get('mae', {}).get('train_mean', mean_absolute_error(train_true, train_pred))
                        test_mae = self.snapshot_avg_stats.get('mae', {}).get('test_mean', mean_absolute_error(test_true, test_pred))
                    else:
                        train_mae = mean_absolute_error(train_true, train_pred)
                        test_mae = mean_absolute_error(test_true, test_pred)
                    
                    test_rmse = np.sqrt(mean_squared_error(test_true, test_pred))
                    test_r2 = r2_score(test_true, test_pred)
                    
                    # Set up the plot area with clean white background
                    ax.set_facecolor('white')
                    ax.set_axisbelow(True)
                    
                    # Plot data points with enhanced styling and integrated legend
                    ax.scatter(train_true, train_pred,
                             alpha=config['alpha_train'], 
                             s=config['marker_size'],
                             c=colors['train'], 
                             marker='^',
                             edgecolors='white', 
                             linewidths=0.3,
                             label=f'Train (MAE: {train_mae:.3f} eV)')
                    
                    ax.scatter(test_true, test_pred,
                            alpha=config['alpha_test'], 
                            s=config['marker_size'],
                            c=colors['test'], 
                            marker='o',
                            edgecolors='white', 
                            linewidths=0.3,
                            label=f'Test (MAE: {test_mae:.3f} eV)')
                    
                    # Create integrated legend
                    ax.legend(fontsize=self.font_size, 
                             loc='upper left', 
                             handlelength=1, 
                             handletextpad=0.3,
                             framealpha=0.9)
                    
                    # Add uncertainty bands for first two subplots only (a) and (b)
                    if confidence_band and config['data_type'] in ['original_voxel', 'original_snapshot_avg']:
                        from scipy import stats
                        
                        if len(test_true) > 3:  # Need sufficient data points
                            # Linear regression for test data
                            slope, intercept, r_value, p_value, std_err = stats.linregress(test_true, test_pred)
                            
                            # Calculate plot limits for uncertainty bands
                            data_min, data_max = np.min(test_true), np.max(test_true)
                            x_smooth = np.linspace(data_min, data_max, 100)
                            
                            # Calculate residuals and MSE
                            test_predicted = slope * test_true + intercept
                            residuals = test_pred - test_predicted
                            mse = np.sum(residuals**2) / max(len(test_true) - 2, 1)
                            
                            # Calculate appropriate uncertainty band based on shade parameter
                            t_val = stats.t.ppf(0.975, max(len(test_true) - 2, 1))
                            
                            if shade == 'confidence_interval':
                                # Confidence interval for regression line (narrower)
                                # Quantifies uncertainty in the mean predicted trend (regression parameters)
                                uncertainty_std = np.sqrt(mse * (1/len(test_true) + 
                                                                (x_smooth - np.mean(test_true))**2 / 
                                                                np.sum((test_true - np.mean(test_true))**2)))
                                band_label = '95% Confidence Band'
                                
                            elif shade == 'prediction_interval':
                                # Prediction interval for individual predictions (wider)
                                # Quantifies uncertainty for new individual observations
                                # KEY: Add "1 +" term to account for residual scatter
                                uncertainty_std = np.sqrt(mse * (1 + 1/len(test_true) + 
                                                                (x_smooth - np.mean(test_true))**2 / 
                                                                np.sum((test_true - np.mean(test_true))**2)))
                                band_label = '95% Prediction Interval'
                            
                            # Calculate bands
                            y_pred = slope * x_smooth + intercept
                            upper_band = y_pred + t_val * uncertainty_std
                            lower_band = y_pred - t_val * uncertainty_std
                            
                            # Fill uncertainty region for test data
                            ax.fill_between(x_smooth, lower_band, upper_band, 
                                          alpha=0.15, color=colors['test'], 
                                          label=band_label, zorder=1)
                            
                            # Optional: Print coverage statistics in verbose mode
                            if self.verbose and shade == 'prediction_interval':
                                # Calculate empirical coverage rate
                                from scipy.interpolate import interp1d
                                f_upper = interp1d(x_smooth, upper_band, bounds_error=False, fill_value='extrapolate')
                                f_lower = interp1d(x_smooth, lower_band, bounds_error=False, fill_value='extrapolate')
                                
                                upper_at_test = f_upper(test_true)
                                lower_at_test = f_lower(test_true)
                                
                                within_interval = (test_pred >= lower_at_test) & (test_pred <= upper_at_test)
                                coverage = np.mean(within_interval) * 100
                                
                                print(f"    {config['label']} Prediction Interval Coverage: {coverage:.1f}% (expected: ~95%)")
                    
            else:  # ML vs MD subplot
                if 'md_matched' in locals() and len(md_matched) > 0:
                    # Set up the plot area with clean white background
                    ax.set_facecolor('white')
                    ax.set_axisbelow(True)
                    
                    # Calculate metrics for both ML and MD
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    
                    # For ML, use pre-calculated snapshot-averaged fold-wise stats if available
                    if hasattr(self, 'snapshot_avg_stats') and self.snapshot_avg_stats:
                        ml_mae = self.snapshot_avg_stats.get('mae', {}).get('test_mean', mean_absolute_error(ml_dft_true, ml_dft_pred))
                        ml_rmse = self.snapshot_avg_stats.get('rmse', {}).get('test_mean', np.sqrt(mean_squared_error(ml_dft_true, ml_dft_pred)))
                        ml_r2 = self.snapshot_avg_stats.get('r2', {}).get('test_mean', r2_score(ml_dft_true, ml_dft_pred))
                    else:
                        ml_mae = mean_absolute_error(ml_dft_true, ml_dft_pred)
                        ml_rmse = np.sqrt(mean_squared_error(ml_dft_true, ml_dft_pred))
                        ml_r2 = r2_score(ml_dft_true, ml_dft_pred)
                    
                    # For MD, use real-time calculation (no pre-calculated fold stats for MD)
                    md_mae = mean_absolute_error(md_dft_true, md_dft_pred)
                    md_rmse = np.sqrt(mean_squared_error(md_dft_true, md_dft_pred))
                    md_r2 = r2_score(md_dft_true, md_dft_pred)
                    
                    # Print detailed MD performance metrics for ΔEsol prediction
                    if self.verbose:
                        print(f"\n--- MD vs DFT ΔEsol Prediction Performance ---")
                        print(f"    ML   - RMSE: {ml_rmse:.3f} eV, MAE: {ml_mae:.3f} eV, R²: {ml_r2:.3f}")
                        print(f"    MD   - RMSE: {md_rmse:.3f} eV, MAE: {md_mae:.3f} eV, R²: {md_r2:.3f}")
                    
                    # Plot ML Test data and MD data with unified alpha values
                    ax.scatter(ml_dft_true, ml_dft_pred,
                              alpha=config['alpha_test'],  # Use unified alpha for test
                              s=config['marker_size'],
                              c=colors['test'], 
                              marker='o',
                              edgecolors='white', 
                              linewidths=0.3,
                              label=f'ML (MAE: {ml_mae:.3f} eV)')
                    
                    ax.scatter(md_dft_true, md_dft_pred,
                              alpha=config['alpha_md'],    # Use unified alpha for MD (same as test)
                              s=config['marker_size'],
                              c=colors['md'], 
                              marker='^',
                              edgecolors='white', 
                              linewidths=0.3,
                              label=f'MD (MAE: {md_mae:.3f} eV)')
                    
                    if linear_fit:
                        # Add linear fitting for both ML and MD data
                        # Determine plot limits for fitting lines
                        all_x_values = np.concatenate([ml_dft_true, md_dft_true])
                        x_min, x_max = np.min(all_x_values), np.max(all_x_values)
                        lims = np.array([x_min, x_max])
                        
                        # ML linear fit
                        z_ml = np.polyfit(ml_dft_true, ml_dft_pred, 1)
                        p_ml = np.poly1d(z_ml)
                        ax.plot(lims, p_ml(lims), linestyle='--', color=colors['test'], 
                            alpha=self.alpha, linewidth=2)
                        ax.text(lims[0] + ml_slope_x, p_ml(lims[0] + ml_slope_y), 
                            f'ML slope={z_ml[0]:.2f}', 
                            color=colors['test'], fontsize=self.font_size)
                        
                        # MD linear fit
                        z_md = np.polyfit(md_dft_true, md_dft_pred, 1)
                        p_md = np.poly1d(z_md)
                        ax.plot(lims, p_md(lims), linestyle='--', color=colors['md'], 
                            alpha=self.alpha, linewidth=2)
                        ax.text(lims[0] + md_slope_x, p_md(lims[0] + md_slope_y), 
                            f'MD slope={z_md[0]:.2f}', 
                            color=colors['md'], fontsize=self.font_size)
                    
                    # Create integrated legend
                    ax.legend(fontsize=self.font_size, 
                             loc='upper left', 
                             handlelength=1, 
                             handletextpad=0.3,
                             framealpha=0.9)
                else:
                    # No MD data available
                    ax.text(0.5, 0.5, 'No MD data available',
                           transform=ax.transAxes,
                           fontsize=self.font_size,
                           ha='center', va='center')
                    all_values = [0, 1]  # Dummy values for plot limits
            
            # Calculate plot limits with padding
            if 'all_values' in locals() and all_values:
                all_values_concat = np.concatenate([np.array(v) for v in all_values if len(v) > 0])
                if len(all_values_concat) > 0:
                    data_min, data_max = np.min(all_values_concat), np.max(all_values_concat)
                    padding = (data_max - data_min) * 0.05
                    plot_min = data_min - padding
                    plot_max = data_max + padding
                    
                    # Perfect prediction line (unity line) with enhanced styling
                    unity_line = ax.plot([plot_min, plot_max], [plot_min, plot_max], 
                                       color=colors['unity'], 
                                       linewidth=2, 
                                       linestyle='--', 
                                       alpha=0.8,
                                       zorder=5)
                    
                    # Professional axis formatting
                    ax.set_xlim(plot_min, plot_max)
                    ax.set_ylim(plot_min, plot_max)
                    ax.set_aspect('equal', adjustable='box')
            
            # Enhanced labels with proper LaTeX formatting
            ax.set_xlabel(config['xlabel'], 
                         fontsize=self.font_size, fontweight='normal')
            ax.set_ylabel(config['ylabel'], 
                         fontsize=self.font_size, fontweight='normal')
            
            # Professional subplot label
            ax.text(num_pos_x, num_pos_y, config['label'], 
                   transform=ax.transAxes,
                   fontsize=self.font_size, 
                   fontweight='bold',
                   verticalalignment='top', 
                   horizontalalignment='left')
            
            # Enhanced tick formatting
            ax.tick_params(axis='both', which='major', labelsize=self.font_size)
            
            if self.verbose and config['data_type'] != 'ml_vs_md':
                if all_train_true and all_test_true:
                    print(f"    {config['label']} {config['data_type']}: Train={len(train_true)}, Test={len(test_true)}")
                    
                    # Print detailed metrics for ΔEsol prediction (snapshot-averaged data)
                    if config['data_type'] == 'original_snapshot_avg':
                        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                        
                        # Use pre-calculated snapshot-averaged fold-wise stats for consistency
                        if hasattr(self, 'snapshot_avg_stats') and self.snapshot_avg_stats:
                            train_rmse_sol = self.snapshot_avg_stats.get('rmse', {}).get('train_mean', np.sqrt(mean_squared_error(train_true, train_pred)))
                            train_mae_sol = self.snapshot_avg_stats.get('mae', {}).get('train_mean', mean_absolute_error(train_true, train_pred))
                            train_r2_sol = self.snapshot_avg_stats.get('r2', {}).get('train_mean', r2_score(train_true, train_pred))
                            
                            test_rmse_sol = self.snapshot_avg_stats.get('rmse', {}).get('test_mean', np.sqrt(mean_squared_error(test_true, test_pred)))
                            test_mae_sol = self.snapshot_avg_stats.get('mae', {}).get('test_mean', mean_absolute_error(test_true, test_pred))
                            test_r2_sol = self.snapshot_avg_stats.get('r2', {}).get('test_mean', r2_score(test_true, test_pred))
                        else:
                            # Fallback to real-time calculation if pre-calculated stats not available
                            train_rmse_sol = np.sqrt(mean_squared_error(train_true, train_pred))
                            train_mae_sol = mean_absolute_error(train_true, train_pred)
                            train_r2_sol = r2_score(train_true, train_pred)
                            
                            test_rmse_sol = np.sqrt(mean_squared_error(test_true, test_pred))
                            test_mae_sol = mean_absolute_error(test_true, test_pred)
                            test_r2_sol = r2_score(test_true, test_pred)
                        
                        print(f"\n--- ΔEsol Prediction Performance (Snapshot-averaged) ---")
                        print(f"    Train - RMSE: {train_rmse_sol:.3f} eV, MAE: {train_mae_sol:.3f} eV, R²: {train_r2_sol:.3f}")
                        print(f"    Test  - RMSE: {test_rmse_sol:.3f} eV, MAE: {test_mae_sol:.3f} eV, R²: {test_r2_sol:.3f}")
        
        plt.tight_layout()
        
        # Save with high quality settings
        if save_plot:
            parts = self.pkl_name.split('-')
            job_code = '-'.join(parts[:3])
            file_name = f"{job_code}-parity_plot_no_tta_with_md_vertical.png"
            save_path = os.path.join(self.output_figure_path, 'cnn_results', file_name)
            
            plt.savefig(save_path, dpi=1000, bbox_inches='tight')
            
            if self.verbose:
                print(f"No-TTA with MD parity plot (vertical) saved to: {save_path}")
        
        # Reset matplotlib parameters to avoid affecting other plots
        plt.rcParams.update(original_rcParams)

        if show_plot:
            plt.show()


    def plot_graphic_abstract_parity_plot(self,
                                         figsize: Tuple[int, int] = (10, 10),
                                         shade: str = 'prediction_interval',
                                         show_plot: bool = False,
                                         save_plot: bool = False):
        """
        Create a simplified parity plot for graphic abstract - only snapshot averaged data
        No legend, no subplot label, larger fonts for better visibility in abstract
        
        Args:
            figsize: Figure size for the plot (default: 10x10 for square aspect)
            shade: Type of uncertainty band to display (default: 'prediction_interval')
                - 'confidence_interval': 95% confidence band for regression line
                - 'prediction_interval': 95% prediction interval for individual points
            show_plot: Whether to show the plot (default: False)
            save_plot: Whether to save the plot (default: False)
        """
        # Validate shade parameter
        valid_shade_types = ['confidence_interval', 'prediction_interval']
        if shade not in valid_shade_types:
            raise ValueError(f"shade must be one of {valid_shade_types}, got '{shade}'")
        
        if not self.model_storage:
            print("No results to plot")
            return

        # Professional color scheme
        colors = {
            'train': self.blue_color,
            'test': self.red_color,
            'unity': '#2c2c2c'
        }
        
        # Save original matplotlib settings
        original_rcParams = plt.rcParams.copy()
        
        # Create single figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Larger font size for graphic abstract
        abstract_font_size = self.font_size * 1.8
        
        if self.verbose:
            print(f"\n--- Graphic Abstract Parity Plot (Snapshot Averaged Data):")
        
        # Collect snapshot averaged data
        all_train_true, all_train_pred = [], []
        all_test_true, all_test_pred = [], []
        
        for fold_idx, fold_results in self.model_storage.items():
            if 'df_train' in fold_results and 'df_test' in fold_results:
                df_train = fold_results['df_train']
                df_test = fold_results['df_test']
                
                # Use only original voxel data (voxel_id == 1 or snapshot_id == 1)
                # Check which column exists
                if 'voxel_id' in df_train.columns:
                    df_train_original = df_train[df_train['voxel_id'] == 1].copy()
                    df_test_original = df_test[df_test['voxel_id'] == 1].copy()
                elif 'snapshot_id' in df_train.columns:
                    df_train_original = df_train[df_train['snapshot_id'] == 1].copy()
                    df_test_original = df_test[df_test['snapshot_id'] == 1].copy()
                else:
                    # If neither column exists, use all data
                    df_train_original = df_train.copy()
                    df_test_original = df_test.copy()
                
                # Group by env-adsorbate and average over 10 snapshots
                train_grouped = df_train_original.groupby(['zeolite', 'environment', 'adsorbate']).agg({
                    'y_true': 'mean',
                    'y_pred': 'mean'
                }).reset_index()
                
                test_grouped = df_test_original.groupby(['zeolite', 'environment', 'adsorbate']).agg({
                    'y_true': 'mean',
                    'y_pred': 'mean'
                }).reset_index()
                
                all_train_true.extend(train_grouped['y_true'].values)
                all_train_pred.extend(train_grouped['y_pred'].values)
                all_test_true.extend(test_grouped['y_true'].values)
                all_test_pred.extend(test_grouped['y_pred'].values)
        
        if all_train_true and all_test_true:
            # Convert to numpy arrays
            train_true = np.array(all_train_true)
            train_pred = np.array(all_train_pred)
            test_true = np.array(all_test_true)
            test_pred = np.array(all_test_pred)
            
            # Calculate statistics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            # Use pre-calculated stats if available
            if hasattr(self, 'snapshot_avg_stats') and self.snapshot_avg_stats:
                test_mae = self.snapshot_avg_stats.get('mae', {}).get('test_mean', mean_absolute_error(test_true, test_pred))
                test_rmse = self.snapshot_avg_stats.get('rmse', {}).get('test_mean', np.sqrt(mean_squared_error(test_true, test_pred)))
                test_r2 = self.snapshot_avg_stats.get('r2', {}).get('test_mean', r2_score(test_true, test_pred))
            else:
                test_mae = mean_absolute_error(test_true, test_pred)
                test_rmse = np.sqrt(mean_squared_error(test_true, test_pred))
                test_r2 = r2_score(test_true, test_pred)
            
            # Set up clean white background
            ax.set_facecolor('white')
            ax.set_axisbelow(True)
            
            # Plot data points (no labels for graphic abstract)
            ax.scatter(train_true, train_pred,
                      alpha=self.alpha_train,
                      s=self.point_size * 3.0,  # Larger points for visibility
                      c=colors['train'],
                      marker='^',
                      edgecolors='white',
                      linewidths=0.5)
            
            ax.scatter(test_true, test_pred,
                      alpha=self.alpha,
                      s=self.point_size * 3.0,  # Larger points for visibility
                      c=colors['test'],
                      marker='o',
                      edgecolors='white',
                      linewidths=0.5)
            
            # Calculate plot limits
            all_values = np.concatenate([train_true, train_pred, test_true, test_pred])
            data_min, data_max = np.min(all_values), np.max(all_values)
            padding = (data_max - data_min) * 0.05
            plot_min = data_min - padding
            plot_max = data_max + padding
            
            # Unity line
            ax.plot([plot_min, plot_max], [plot_min, plot_max],
                   color=colors['unity'],
                   linewidth=3,
                   linestyle='--',
                   alpha=0.8,
                   zorder=5)
            
            # Add uncertainty band for test data (consistent with main parity plots)
            from scipy import stats
            if len(test_true) > 3:
                # Linear regression for test data
                slope, intercept, r_value, p_value, std_err = stats.linregress(test_true, test_pred)
                
                # Calculate plot limits for uncertainty bands
                data_min_test, data_max_test = np.min(test_true), np.max(test_true)
                x_smooth = np.linspace(data_min_test, data_max_test, 100)
                
                # Calculate residuals and MSE
                test_predicted = slope * test_true + intercept
                residuals = test_pred - test_predicted
                mse = np.sum(residuals**2) / max(len(test_true) - 2, 1)
                
                # Calculate appropriate uncertainty band based on shade parameter
                t_val = stats.t.ppf(0.975, max(len(test_true) - 2, 1))
                
                if shade == 'confidence_interval':
                    # Confidence interval for regression line (narrower)
                    uncertainty_std = np.sqrt(mse * (1/len(test_true) + 
                                                    (x_smooth - np.mean(test_true))**2 / 
                                                    np.sum((test_true - np.mean(test_true))**2)))
                    
                elif shade == 'prediction_interval':
                    # Prediction interval for individual predictions (wider)
                    # KEY: Add "1 +" term to account for residual scatter
                    uncertainty_std = np.sqrt(mse * (1 + 1/len(test_true) + 
                                                    (x_smooth - np.mean(test_true))**2 / 
                                                    np.sum((test_true - np.mean(test_true))**2)))
                
                # Calculate bands
                y_pred = slope * x_smooth + intercept
                upper_band = y_pred + t_val * uncertainty_std
                lower_band = y_pred - t_val * uncertainty_std
                
                # Fill uncertainty region for test data
                ax.fill_between(x_smooth, lower_band, upper_band, 
                              alpha=0.15, color=colors['test'], zorder=1)
            
            # Set axis properties
            ax.set_xlim(plot_min, plot_max)
            ax.set_ylim(plot_min, plot_max)
            ax.set_aspect('equal', adjustable='box')
            
            # Larger axis labels (no bold)
            ax.set_xlabel(r'$\Delta E^{\mathit{DFT}}_{\mathit{sol}}$ (eV)',
                         fontsize=abstract_font_size)
            ax.set_ylabel(r'$\Delta E^{\mathit{ML}}_{\mathit{sol}}$ (eV)',
                         fontsize=abstract_font_size)
            
            # Set custom tick intervals (0.2 eV) and format to 1 decimal place
            from matplotlib.ticker import MultipleLocator, FormatStrFormatter
            ax.xaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            
            # Larger tick labels
            ax.tick_params(axis='both', which='major', labelsize=abstract_font_size)
            
            if self.verbose:
                print(f"    Train points: {len(train_true)}, Test points: {len(test_true)}")
                print(f"    Test MAE: {test_mae:.3f} eV, R²: {test_r2:.3f}")
        
        plt.tight_layout()
        
        # Save figure
        if save_plot:
            job_code = self.pkl_name.split('-')[0] if '-' in self.pkl_name else self.pkl_name.replace('.pkl', '')
            save_filename = f"{job_code}-graphic_abstract_parity.png"
            full_save_path = os.path.join(self.output_figure_path, 'cnn_results', save_filename)
            
            os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
            
            plt.savefig(full_save_path, dpi=500, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            
            if self.verbose:
                print(f"    Graphic abstract parity plot saved to: {full_save_path}")
        
        if show_plot:
            plt.show()
        
        # Restore original matplotlib settings
        plt.rcParams.update(original_rcParams)


    def plot_bar_plot_performance_no_test_augment(self,
                         figsize: Tuple[int, int] = (22, 7),
                         save_path: str = None,
                         bar_width: float = 0.2,
                         show_score_on_bar: bool = False,
                         red_color: str = None,           # Use class attribute as default
                         blue_color: str = None,          # Use class attribute as default
                         show_plot: bool = False,
                         save_plot: bool = False
                         ):
        """
        Plot comprehensive performance comparison without test time augmentation
        Shows only original voxel predictions and their snapshot averages (2 groups per metric)
        
        Args:
            figsize: Figure size for the plot
            save_path: Path to save the figure (optional)
            bar_width: Width of individual bars (default: 0.25)
            show_score_on_bar: Whether to show numerical scores on bars (default: False)
            red_color: Color for test bars (uses class attribute if None)
            blue_color: Color for train bars (uses class attribute if None)
            show_plot: Whether to show the plot (default: False)
            save_plot: Whether to save the plot (default: False)
        """
        # Use class attributes if no color parameters provided
        if red_color is None:
            red_color = self.red_color
        if blue_color is None:
            blue_color = self.blue_color
            
        if not self.model_storage:
            print("No results to plot")
            return
        
        # Use pre-calculated metrics for original voxel (no TTA) and snapshot averaged data
        raw_metrics = {}
        if self.summary_stats:
            for metric in ['rmse', 'mae', 'r2']:
                train_key = f'train_{metric}'
                test_key = f'test_{metric}'
                if train_key in self.summary_stats and test_key in self.summary_stats:
                    raw_metrics[metric] = {
                        'train_mean': self.summary_stats[train_key]['mean'],
                        'train_std': self.summary_stats[train_key]['std'],
                        'test_mean': self.summary_stats[test_key]['mean'],
                        'test_std': self.summary_stats[test_key]['std']
                    }
        
        # Use pre-calculated snapshot averaged metrics (no TTA)
        snapshot_avg_metrics = getattr(self, 'snapshot_avg_stats', {})
        
        if self.verbose:
            print(f"\n--- Performance Comparison (No TTA - using pre-calculated metrics):")
            print(f"    Original voxel metrics: {list(raw_metrics.keys())}")
            print(f"    Snapshot-averaged metrics: {list(snapshot_avg_metrics.keys())}")
        
        # Create comprehensive plot with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        metrics_info = [
            ('rmse', 'RMSE (eV)', 'lower_better'),
            ('mae', 'MAE (eV)', 'lower_better'), 
            ('r2', 'R² Score', 'higher_better')
        ]
        
        for idx, (metric, title, direction) in enumerate(metrics_info):
            ax = axes[idx]
            
            # Collect available data types (only 2 groups for no-TTA version)
            aggregation_types = []
            all_data = {}
            
            if metric in raw_metrics:
                aggregation_types.append('Original Voxel')
                all_data['Original Voxel'] = raw_metrics[metric]
            
            if metric in snapshot_avg_metrics:
                aggregation_types.append('Snapshot Avg')
                all_data['Snapshot Avg'] = snapshot_avg_metrics[metric]
            
            if not aggregation_types:
                ax.text(0.5, 0.5, f'No {metric.upper()} data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=self.font_size)
                ax.set_title(title, fontsize=self.font_size)
                continue
            
            # Set up bar positions for 2 groups
            n_groups = len(aggregation_types)
            group_width = bar_width * 2 + 0.15  # Space for train+test bars plus gap
            x_positions = np.arange(n_groups) * group_width
            
            # Plot bars for each aggregation type
            for i, agg_type in enumerate(aggregation_types):
                data = all_data[agg_type]
                
                # Train bar (blue)
                train_x = x_positions[i]
                train_bar = ax.bar(train_x, data['train_mean'], bar_width, 
                                  yerr=data['train_std'], capsize=5, 
                                  alpha=0.8, color=blue_color)
                
                # Test bar (red)
                test_x = x_positions[i] + bar_width
                test_bar = ax.bar(test_x, data['test_mean'], bar_width,
                                 yerr=data['test_std'], capsize=5,
                                 alpha=0.8, color=red_color)
                
                # Add value labels on bars (only if show_score_on_bar is True)
                if show_score_on_bar:
                    def add_value_label(bar, mean, std, offset_factor=1.1):
                        height = bar.get_height()
                        y_pos = height + std * offset_factor if height >= 0 else height - std * offset_factor
                        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                               f'{mean:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                               fontsize=self.font_size, fontweight='bold')
                    
                    add_value_label(train_bar[0], data['train_mean'], data['train_std'])
                    add_value_label(test_bar[0], data['test_mean'], data['test_std'])
            
            # Customize axis
            ax.set_ylabel(title, fontsize=self.font_size)
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='both', which='major', labelsize=self.font_size)
            
            # Set x-axis labels and positions (center each group)
            x_ticks = x_positions + bar_width / 2  # Center of each group
            ax.set_xticks(x_ticks)
            
            # Use descriptive chemical labels for no-TTA version
            descriptive_labels = []
            for agg_type in aggregation_types:
                if agg_type == 'Original Voxel':
                    descriptive_labels.append(r'$\mathit{\Delta E_{int}}$')
                elif agg_type == 'Snapshot Avg':
                    descriptive_labels.append(r'$\mathit{\Delta E_{sol}}$')
                else:
                    descriptive_labels.append(agg_type)
            
            ax.set_xticklabels(descriptive_labels, fontsize=self.font_size)
            
            # Add Train/Test legend in upper right corner of each subplot (horizontal layout)
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=blue_color, alpha=0.8, label='Train'),
                Patch(facecolor=red_color, alpha=0.8, label='Test')
            ]
            ax.legend(handles=legend_elements, fontsize=self.font_size,
                     loc='upper right', framealpha=0.9, ncol=2)
            
            # Set y-axis limits and tick configuration manually for each metric
            if metric == 'rmse':
                ax.set_ylim(0, 0.15)  # RMSE typically ranges from 0 to ~0.2
                ax.set_yticks(np.arange(0, 0.15, 0.03))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
            elif metric == 'mae':
                ax.set_ylim(0, 0.15)  # MAE typically ranges from 0 to ~0.16
                ax.set_yticks(np.arange(0, 0.15, 0.03))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
            elif metric == 'r2':
                ax.set_ylim(0, 1.2)  # R² typically ranges from 0 to 1.0
                ax.set_yticks(np.arange(0, 1.05, 0.2))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
        
        plt.tight_layout()
        
        if save_plot:
            # Extract job code from pkl_name
            parts = self.pkl_name.split('-')
            job_code = '-'.join(parts[:3])
            file_name = f"{job_code}-{self.split_type}-bar_plot_no_tta.png"
            save_path = os.path.join(self.output_figure_path, 'cnn_results', file_name)
            plt.savefig(save_path, dpi=1000, bbox_inches='tight')
            if self.verbose:
                print(f"    3D CNN performance comparison plot (No TTA) saved to: {save_path}")
          
        
        if show_plot:
            plt.show()
        
        # Print summary table for no-TTA version
        if self.verbose:
            print(f"\n--- Performance Summary Table (No TTA):")
            print(f"{'Metric':<12} {'Aggregation':<15} {'Train Mean':<12} {'Train Std':<12} {'Test Mean':<12} {'Test Std':<12}")
            print("-" * 85)
            
            for metric in ['rmse', 'mae', 'r2']:
                if metric in raw_metrics:
                    data = raw_metrics[metric]
                    print(f"{metric.upper():<12} {'Original Voxel':<15} {data['train_mean']:<12.3f} {data['train_std']:<12.3f} {data['test_mean']:<12.3f} {data['test_std']:<12.3f}")
                
                if metric in snapshot_avg_metrics:
                    data = snapshot_avg_metrics[metric]
                    print(f"{'':12} {'Snapshot Avg':<15} {data['train_mean']:<12.3f} {data['train_std']:<12.3f} {data['test_mean']:<12.3f} {data['test_std']:<12.3f}")
                
                if metric != 'r2':
                    print("-" * 85)


    def plot_bar_plot_performance_using_test_augment(self,
                         figsize: Tuple[int, int] = (24, 8),
                         save_path: str = None,
                         bar_width: float = 0.2,
                         show_score_on_bar: bool = False,
                         red_color: str = None,           # Use class attribute as default
                         blue_color: str = None,          # Use class attribute as default
                         show_plot: bool = False,
                         save_plot: bool = False
                         ):
        """
        Plot comprehensive performance comparison with improved visualization
        
        Args:
            figsize: Figure size for the plot
            save_path: Path to save the figure (optional)
            show_score_on_bar: Whether to show numerical scores on bars (default: False)
        """
        # Use class attributes if no color parameters provided
        if red_color is None:
            red_color = self.red_color
        if blue_color is None:
            blue_color = self.blue_color
            
        if not self.model_storage:
            print("No results to plot")
            return
        
        # Use pre-calculated metrics
        raw_metrics = {}
        if self.summary_stats:
            for metric in ['rmse', 'mae', 'r2']:
                train_key = f'train_{metric}'
                test_key = f'test_{metric}'
                if train_key in self.summary_stats and test_key in self.summary_stats:
                    raw_metrics[metric] = {
                        'train_mean': self.summary_stats[train_key]['mean'],
                        'train_std': self.summary_stats[train_key]['std'],
                        'test_mean': self.summary_stats[test_key]['mean'],
                        'test_std': self.summary_stats[test_key]['std']
                    }
        
        # Use pre-calculated averaged metrics
        voxel_avg_metrics = getattr(self, 'voxel_avg_stats', {})
        snapshot_avg_metrics = getattr(self, 'snapshot_avg_stats', {})
        
        if self.verbose:
            print(f"\n--- Performance Comparison (using pre-calculated metrics):")
            print(f"    Raw metrics: {list(raw_metrics.keys())}")
            print(f"    Voxel-averaged metrics: {list(voxel_avg_metrics.keys())}")
            print(f"    Snapshot-averaged metrics: {list(snapshot_avg_metrics.keys())}")
        
        # Create comprehensive plot
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        metrics_info = [
            ('rmse', 'RMSE (eV)', 'lower_better'),
            ('mae', 'MAE (eV)', 'lower_better'), 
            ('r2', 'R² Score', 'higher_better')
        ]
        
        for idx, (metric, title, direction) in enumerate(metrics_info):
            ax = axes[idx]
            
            # Collect available data types
            aggregation_types = []
            all_data = {}
            
            if metric in raw_metrics:
                aggregation_types.append('Raw')
                all_data['Raw'] = raw_metrics[metric]
            
            if metric in voxel_avg_metrics:
                aggregation_types.append('Voxel Avg')
                all_data['Voxel Avg'] = voxel_avg_metrics[metric]
            
            if metric in snapshot_avg_metrics:
                aggregation_types.append('Snapshot Avg')
                all_data['Snapshot Avg'] = snapshot_avg_metrics[metric]
            
            if not aggregation_types:
                ax.text(0.5, 0.5, f'No {metric.upper()} data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=self.font_size)
                ax.set_title(title, fontsize=self.font_size)
                continue
            
            # Set up bar positions
            n_groups = len(aggregation_types)
            bar_width = bar_width
            group_width = bar_width * 2 + 0.1  # Space for train+test bars plus gap
            x_positions = np.arange(n_groups) * group_width
            
            # Plot bars for each aggregation type
            for i, agg_type in enumerate(aggregation_types):
                data = all_data[agg_type]
                
                # Train bar (blue)
                train_x = x_positions[i]
                train_bar = ax.bar(train_x, data['train_mean'], bar_width, 
                                  yerr=data['train_std'], capsize=5, 
                                  alpha=0.8, color=blue_color)
                
                # Test bar (red)
                test_x = x_positions[i] + bar_width
                test_bar = ax.bar(test_x, data['test_mean'], bar_width,
                                 yerr=data['test_std'], capsize=5,
                                 alpha=0.8, color=red_color)
                
                # Add value labels on bars (only if show_score_on_bar is True)
                if show_score_on_bar:
                    def add_value_label(bar, mean, std, offset_factor=1.1):
                        height = bar.get_height()
                        y_pos = height + std * offset_factor if height >= 0 else height - std * offset_factor
                        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                               f'{mean:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                               fontsize=self.font_size, fontweight='bold')
                    
                    add_value_label(train_bar[0], data['train_mean'], data['train_std'])
                    add_value_label(test_bar[0], data['test_mean'], data['test_std'])
            
            # Customize axis
            ax.set_ylabel(title, fontsize=self.font_size)
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='both', which='major', labelsize=self.font_size)
            
            # Set x-axis labels and positions (center each group)
            x_ticks = x_positions + bar_width / 2  # Center of each group
            ax.set_xticks(x_ticks)
            
            # Use more descriptive chemical labels
            descriptive_labels = []
            for agg_type in aggregation_types:
                if agg_type == 'Raw':
                    descriptive_labels.append(r'$\mathit{\Delta E_{int,grid}}$')
                elif agg_type == 'Voxel Avg':
                    descriptive_labels.append(r'$\mathit{\Delta E_{int,snapshot}}$')
                elif agg_type == 'Snapshot Avg':
                    descriptive_labels.append(r'$\mathit{\Delta E_{sol,adsorbate}}$')
                else:
                    descriptive_labels.append(agg_type)
            
            ax.set_xticklabels(descriptive_labels, fontsize=self.font_size)
            
            # Add Train/Test legend in upper right corner of each subplot (horizontal layout)
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=blue_color, alpha=0.8, label='Train'),
                Patch(facecolor=red_color, alpha=0.8, label='Test')
            ]
            ax.legend(handles=legend_elements, fontsize=self.font_size,
                     loc='upper right', framealpha=0.9, ncol=2)
            
            # Set y-axis limits and tick configuration manually for each metric
            if metric == 'rmse':
                ax.set_ylim(0, 0.15)  # RMSE typically ranges from 0 to ~0.2
                ax.set_yticks(np.arange(0, 0.15, 0.03))  # Reduce upper bound to leave space
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))  # 3 decimal places
            elif metric == 'mae':
                ax.set_ylim(0, 0.15)  # MAE typically ranges from 0 to ~0.16
                ax.set_yticks(np.arange(0, 0.15, 0.03))  # Reduce upper bound to leave space
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))  # 3 decimal places
            elif metric == 'r2':
                ax.set_ylim(0, 1.2)  # R² typically ranges from 0 to 1.0
                ax.set_yticks(np.arange(0, 1.05, 0.2))  # Reduce upper bound to leave space
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))  # 2 decimal places
        
        plt.tight_layout()
        
        if save_plot:
            # Extract job code from pkl_name
            parts = self.pkl_name.split('-')
            job_code = '-'.join(parts[:3])
            file_name = f"{job_code}-{self.split_type}-bar_plot_using_tta.png"
            save_path = os.path.join(self.output_figure_path, 'cnn_results', file_name)
            plt.savefig(save_path, dpi=1000, bbox_inches='tight')
            if self.verbose:
                print(f"    3D CNN performance comparison plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
        
        # Print summary table
        if self.verbose:
            print(f"\n--- Performance Summary Table:")
            print(f"{'Metric':<12} {'Aggregation':<15} {'Train Mean':<12} {'Train Std':<12} {'Test Mean':<12} {'Test Std':<12}")
            print("-" * 85)
            
            for metric in ['rmse', 'mae', 'r2']:
                if metric in raw_metrics:
                    data = raw_metrics[metric]
                    print(f"{metric.upper():<12} {'Raw':<15} {data['train_mean']:<12.3f} {data['train_std']:<12.3f} {data['test_mean']:<12.3f} {data['test_std']:<12.3f}")
                
                if metric in voxel_avg_metrics:
                    data = voxel_avg_metrics[metric]
                    print(f"{'':12} {'Voxel Avg':<15} {data['train_mean']:<12.3f} {data['train_std']:<12.3f} {data['test_mean']:<12.3f} {data['test_std']:<12.3f}")
                
                if metric in snapshot_avg_metrics:
                    data = snapshot_avg_metrics[metric]
                    print(f"{'':12} {'Snapshot Avg':<15} {data['train_mean']:<12.3f} {data['train_std']:<12.3f} {data['test_mean']:<12.3f} {data['test_std']:<12.3f}")
                
                if metric != 'r2':
                    print("-" * 85)


if __name__ == "__main__":


    # Random split (5-fold CV): tests general molecular interactions
    # pkl_name = "model-random-2546193-epochs_200-bs_32-lr_0.0001-grid_16.0_0.8.pkl" # Test MAE 0.088
    # pkl_name = "model-random-2546194-epochs_200-bs_32-lr_0.0001-grid_16.0_0.8.pkl" # Test MAE 0.091
    # pkl_name = "model-random-2546195-epochs_200-bs_32-lr_0.0001-grid_16.0_0.8.pkl" # Test MAE 0.090
    # pkl_name = "model-random-2546197-epochs_200-bs_32-lr_0.0001-grid_16.0_0.8.pkl" # Test MAE 0.087
    # pkl_name = "model-random-2546199-epochs_200-bs_32-lr_0.0001-grid_16.0_0.8.pkl" # Test MAE 0.087
    # pkl_name = "model-random-2546201-epochs_200-bs_32-lr_0.0001-grid_16.0_0.8.pkl" # Test MAE 0.088
    # pkl_name = "model-random-2546213-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.088
    # pkl_name = "model-random-2546214-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.089
    # pkl_name = "model-random-2546215-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.088
    # pkl_name = "model-random-2546216-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.088
    # pkl_name = "model-random-2546217-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.085
    # pkl_name = "model-random-2546218-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.087
    # pkl_name = "model-random-2546220-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.088
    # pkl_name = "model-random-2546223-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.088
    # pkl_name = "model-random-2546226-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.088
    # pkl_name = "model-random-2546227-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.088
    # pkl_name = "model-random-2546228-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.087
    pkl_name = "model-random-2546229-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.083 # Best
    # pkl_name = "model-random-2546238-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.086
    # pkl_name = "model-random-2546239-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.087
    # pkl_name = "model-random-2546240-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.087
    # pkl_name = "model-random-2546241-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.088
    # pkl_name = "model-random-2546243-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.086
    # pkl_name = "model-random-2546244-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.087
    # pkl_name = "model-random-2546246-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.084
    # pkl_name = "model-random-2546247-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.089
        
    
    # # Pore type split (2-fold CV): tests cross-pore-type prediction ability
    # pkl_name = "model-pore_type-2518682-epochs_100-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.108
    # pkl_name = "model-pore_type-2518683-epochs_100-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.115
    # pkl_name = "model-pore_type-2518689-epochs_100-bs_32-lr_0.0005-grid_16.0_0.8.pkl" # Test MAE 0.112
    # pkl_name = "model-pore_type-2518692-epochs_100-bs_32-lr_0.0005-grid_16.0_0.8.pkl" # Test MAE 0.113
    # pkl_name = "model-pore_type-2518695-epochs_100-bs_32-lr_0.0005-grid_16.0_0.8.pkl" # Test MAE 0.106 # Best
    # pkl_name = "model-pore_type-2518732-epochs_100-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.112
    # pkl_name = "model-pore_type-2518733-epochs_100-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.115
    # pkl_name = "model-pore_type-2518734-epochs_100-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.116
    # pkl_name = "model-pore_type-2529419-epochs_200-bs_32-lr_0.0001-grid_16.0_0.8.pkl" # Test MAE 0.118
    # pkl_name = "model-pore_type-2532278-epochs_200-bs_32-lr_0.0001-grid_16.0_0.8.pkl" # Test MAE 0.116
    # pkl_name = "model-pore_type-2532283-epochs_200-bs_32-lr_0.0001-grid_16.0_0.8.pkl" # Test MAE 0.116
    
    
    # # Solvent split (4-fold CV): tests cross-solvent prediction ability
    # pkl_name = "model-solvent-2518713-epochs_100-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.090
    # pkl_name = "model-solvent-2518714-epochs_100-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.084
    # pkl_name = "model-solvent-2518741-epochs_100-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.089
    # pkl_name = "model-solvent-2518743-epochs_100-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.089
    # pkl_name = "model-solvent-2518745-epochs_100-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.088
    # pkl_name = "model-solvent-2529404-epochs_200-bs_32-lr_0.0001-grid_16.0_0.8.pkl" # Test MAE 0.085
    # pkl_name = "model-solvent-2529405-epochs_200-bs_32-lr_0.0001-grid_16.0_0.8.pkl" # Test MAE 0.083 # Best
    # pkl_name = "model-solvent-2529406-epochs_200-bs_32-lr_0.0001-grid_16.0_0.8.pkl" # Test MAE 0.090
    # pkl_name = "model-solvent-2529412-epochs_200-bs_32-lr_0.0001-grid_16.0_0.8.pkl" # Test MAE 0.088
    # pkl_name = "model-solvent-2529413-epochs_200-bs_32-lr_0.0001-grid_16.0_0.8.pkl" # Test MAE 0.090
    # pkl_name = "model-solvent-2529414-epochs_200-bs_32-lr_0.0001-grid_16.0_0.8.pkl" # Test MAE 0.087
    # pkl_name = "model-solvent-2529436-epochs_200-bs_32-lr_0.0001-grid_16.0_0.8.pkl" # Test MAE 0.087
    # pkl_name = "model-solvent-2529439-epochs_200-bs_32-lr_0.0001-grid_16.0_0.8.pkl" # Test MAE 0.087

    
    # Set to True to export results to CSV
    export_csv = False
    
    # Create extractor with all parameters - execution happens automatically in __init__
    extractor = CNN3DResultsExtractor(pkl_name=pkl_name,
                                      verbose=True,
                                      export_csv=export_csv,
                                      font_size=24,
                                      blue_color='#1f77b4',
                                      red_color='#d62728',
                                      green_color='#2ca02c',
                                      alpha=0.7,
                                      alpha_train=0.5,
                                      point_size=80,
                                      point_size_deltaEint=50,)
    
    
    # # Generate combined parity plot with three subplots (TTA method)
    # extractor.plot_parity_plots_using_test_augment(show_plot=False,
    #                                                save_plot=True)
    
    
    
    # # Generate no-TTA parity plot with two subplots (No-TTA method)
    # extractor.plot_parity_plot_no_test_augment_with_Train_2_subfigures(show_plot=False,
    #                                                                    save_plot=True)
    
    
    # Generate no-TTA parity plot with three subplots including MD comparison (No-TTA + MD method)
    extractor.plot_parity_plot_no_test_augment_with_Train_with_MD_3_subfigures(
        linear_fit=True,
        confidence_band=True,
        shade='confidence_interval',  # Options: 'confidence_interval' or 'prediction_interval'
        show_plot=False,
        save_plot=False,
        ml_slope_x=0.15,
        ml_slope_y=0.02,
        md_slope_x=0.04,
        md_slope_y=0.53
    )
    
    
    # Generate no-TTA parity plot with three subplots including MD comparison - Vertical layout
    extractor.plot_parity_plot_no_test_augment_with_Train_with_MD_3_subfigures_vertical(
        linear_fit=True,
        confidence_band=True,
        shade='confidence_interval',  # Options: 'confidence_interval' or 'prediction_interval'
        show_plot=False,
        save_plot=True,
        ml_slope_x=0.15,
        ml_slope_y=0.02,
        md_slope_x=0.04,
        md_slope_y=0.53
    )
    
    
    # # Generate comprehensive performance bar plot (with TTA)
    # extractor.plot_bar_plot_performance_using_test_augment(show_plot=False,
    #                                                        save_plot=True)
    
    
    # # Generate comprehensive performance bar plot (no TTA)
    # extractor.plot_bar_plot_performance_no_test_augment(show_plot=False,
    #                                                     save_plot=True)


    # extractor.plot_graphic_abstract_parity_plot(
    #     shade='prediction_interval',  # Options: 'confidence_interval' or 'prediction_interval'
    #     show_plot=False,
    #     save_plot=False
    # )