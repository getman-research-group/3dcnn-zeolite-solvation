"""
plot_attention_results.py

Visualize spatial attention maps from the trained 3D CNN model for one
representative zeolite-solvation sample. This streamlined script loads a saved
cross-validation results pickle, identifies which training folds contain the
requested sample, retrieves the corresponding voxel grid from the dataset
pickle, extracts spatial attention maps from the three CBAM-based modules, and
plots a single publication-style attention heatmap figure.

Current workflow
----------------
1. Load the saved model-results pickle from ``output_model_cnn``.
2. Find the requested sample in the stored training-set tables.
3. Load the matching voxel grid from ``dataset_cnn``.
4. Extract spatial attention maps from the adsorbate branch, solvent branch,
   and interaction-attention module.
5. Average the attention maps across the folds where the sample appears in the
   training set.
6. Save the final attention heatmap figure for the selected XY, XZ, or YZ
   plane.

Main output
-----------
- ``figure_7-attention_heatmaps.png``

Notes
-----
- The results pickle may have been created on GPU hardware, so this script
  loads model objects onto CPU when needed.
- The requested voxel-grid pickle must exist in ``dataset_cnn`` for the chosen
  zeolite / environment / adsorbate combination.
"""

import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

# Import your modules - Updated for dual-branch architecture
from core.path import get_paths






class SpatialAttentionVisualizer:
    """
    Simplified visualizer for spatial attention mechanisms in 3D CNN models
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def extract_spatial_attention_maps(self, input_data):
        """Extract spatial attention maps from dual-branch CBAM modules"""
        attention_maps = {}
        
        def spatial_attention_hook_fn(name):
            def hook(module, input, output):
                if hasattr(module, 'last_attention_weights') and module.last_attention_weights is not None:
                    spatial_weights = module.last_attention_weights.detach().cpu().numpy()
                    attention_maps[name] = spatial_weights
                    print(f"  ✅ Captured spatial attention from {name} with shape {spatial_weights.shape}")
            return hook
        
        # Register hooks for spatial attention modules in the dual-branch architecture
        hooks = []
        target_modules = [
            'adsorbate_processor', 
            'solvent_processor', 
            'interaction_attention'
        ]
        
        for name, module in self.model.named_modules():
            # Check if this is a spatial attention module within our target CBAM3D modules
            if 'spatial_attention' in name.lower() and hasattr(module, 'last_attention_weights'):
                # Check if it belongs to one of our target branches
                is_target_module = any(target in name for target in target_modules)
                
                if is_target_module:
                    hook = module.register_forward_hook(spatial_attention_hook_fn(name))
                    hooks.append(hook)
                    print(f"Registered hook for: {name}")
        
        print(f"Total hooks registered: {len(hooks)} (expecting 3 for dual-branch architecture)")
        
        # Forward pass to capture attention
        with torch.no_grad():
            _ = self.model(input_data)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        print(f"Captured attention maps: {list(attention_maps.keys())}")
        return attention_maps


class SpatialAttentionAnalyzer:
    """
    Simplified analyzer for spatial attention visualization
    """
    def __init__(self, results_filename, font_size=12):
        self.results_filename = results_filename
        self.results_pkl_path = os.path.join(get_paths("output_model_cnn"), self.results_filename)
        self.results = None
        self.best_model = None
        self.best_fold_idx = None
        self.font_size = font_size
        
        # Create output directory
        self.output_dir = self._create_output_directory()
        
        # Load results
        self.load_results()
    
    def _create_output_directory(self):
        """Create output directory for attention results"""
        output_dir = os.path.join(get_paths("output_figure_path"), "cnn_attention_results")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def load_results(self):
        """Load the saved training results with CPU mapping"""
        import io
        
        def cpu_unpickler(file):
            class CpuUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module == 'torch.storage' and name == '_load_from_bytes':
                        return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                    else:
                        return super().find_class(module, name)
                
                def persistent_load(self, pid):
                    if isinstance(pid, tuple) and len(pid) == 5:
                        storage_type, root_key, location, size, view_metadata = pid
                        if 'cuda' in str(location):
                            location = 'cpu'
                        return super().persistent_load((storage_type, root_key, location, size, view_metadata))
                    else:
                        return super().persistent_load(pid)
            return CpuUnpickler(file)
        
        try:
            self.results = torch.load(self.results_pkl_path, map_location='cpu', weights_only=False)
            print("✓ Successfully loaded with torch.load")
        except Exception as e:
            print(f"Direct load failed, using custom unpickler: {e}")
            with open(self.results_pkl_path, 'rb') as f:
                unpickler = cpu_unpickler(f)
                self.results = unpickler.load()
            print("✓ Successfully loaded with custom unpickler")
        
        self.model_storage = self.results['model_storage']
        
        # Find best model (lowest test RMSE)
        test_rmses = [self.model_storage[i]['test_rmse'] for i in range(len(self.model_storage))]
        self.best_fold_idx = np.argmin(test_rmses)
        self.best_model = self.model_storage[self.best_fold_idx]['model']
        
        print(f"Loaded results from: {self.results_pkl_path}")
        print(f"Best model is from fold {self.best_fold_idx + 1} with test RMSE: {test_rmses[self.best_fold_idx]:.4f}")
    
    def find_sample_in_results(self, zeolite, env_adsorbate, snapshot, voxel_id):
        """
        Find a specific sample in the results based on criteria
        Similar to plot_importance_maps.py find_sample method
        """
        print(f"\n--- Searching for sample in results:")
        print(f"    Zeolite: {zeolite}")
        print(f"    Env-Adsorbate: {env_adsorbate}")
        print(f"    Snapshot: {snapshot}")
        print(f"    Voxel ID: {voxel_id}")

        # Parse env_adsorbate into three components
        # Format: env-pore_type-adsorbate
        # Example: methanol_600_water_600-hydrophilic-11_01_propylene_glycol
        if '-' in env_adsorbate:
            parts = env_adsorbate.split('-')
            if len(parts) >= 3:
                env = parts[0]  # methanol_600_water_600
                pore_type = parts[1]  # hydrophilic
                adsorbate = parts[2]  # 11_01_propylene_glycol
                # Reconstruct environment for data lookup
                environment = f"{env}-{pore_type}"  # methanol_600_water_600-hydrophilic
            else:
                raise ValueError(f"Invalid env_adsorbate format: {env_adsorbate}. Expected format: env-pore_type-adsorbate")
        else:
            raise ValueError(f"Invalid env_adsorbate format: {env_adsorbate}. Expected format: env-pore_type-adsorbate")
        
        print(f"    Parsed - Env: {env}, Pore Type: {pore_type}, Adsorbate: {adsorbate}")
        print(f"    Environment (for lookup): {environment}")
        
        # Find which folds contain this sample as training data
        train_folds = []
        sample_info = None
        
        for fold_idx in sorted(self.model_storage.keys()):
            fold_data = self.model_storage[fold_idx]
            
            # Check train set for this sample
            if 'df_train' in fold_data:
                df_train = fold_data['df_train']
                mask = (
                    (df_train['zeolite'] == zeolite) & 
                    (df_train['environment'] == environment) & 
                    (df_train['adsorbate'] == adsorbate) & 
                    (df_train['snapshot'] == snapshot) & 
                    (df_train['voxel_id'] == voxel_id)
                )
                
                if len(df_train[mask]) > 0:
                    sample = df_train[mask].iloc[0]
                    train_folds.append(fold_idx)
                    if sample_info is None:  # Store info from first occurrence
                        sample_info = {
                            'zeolite': zeolite,
                            'env': env,
                            'pore_type': pore_type,
                            'environment': environment,
                            'adsorbate': adsorbate,
                            'snapshot': snapshot,
                            'voxel_id': voxel_id,
                            'y_true': sample['y_true'],
                            'train_folds': []  # Will be populated below
                        }
        
        if sample_info is None:
            print(f"❌ Sample not found in any training folds!")
            return None, None
            
        sample_info['train_folds'] = train_folds
        print(f"✅ Sample found in training folds: {[f+1 for f in train_folds]} ({len(train_folds)}/5 folds)")
        print(f"   True value: {sample_info['y_true']:.4f}")
        
        return sample_info, train_folds
    
    def load_actual_voxel_data_for_sample(self, zeolite, env_adsorbate, snapshot, voxel_id):
        """
        Load the actual voxel data for a specific sample using VoxelGridsLoader
        Same method as plot_importance_maps.py
        """
        # Parse env_adsorbate into three components
        # Format: env-pore_type-adsorbate
        if '-' in env_adsorbate:
            parts = env_adsorbate.split('-')
            if len(parts) >= 3:
                env = parts[0]  # methanol_600_water_600
                pore_type = parts[1]  # hydrophilic
                adsorbate = parts[2]  # 11_01_propylene_glycol
                # Reconstruct environment for data lookup
                environment = f"{env}-{pore_type}"  # methanol_600_water_600-hydrophilic
            else:
                raise ValueError(f"Invalid env_adsorbate format: {env_adsorbate}. Expected format: env-pore_type-adsorbate")
        else:
            raise ValueError(f"Invalid env_adsorbate format: {env_adsorbate}. Expected format: env-pore_type-adsorbate")
        
        print(f"Loading voxel data using VoxelGridsLoader...")
        print(f"  Zeolite: {zeolite}")
        print(f"  Env: {env}")
        print(f"  Pore Type: {pore_type}")
        print(f"  Environment: {environment}")  
        print(f"  Adsorbate: {adsorbate}")
        print(f"  Snapshot: {snapshot}, Voxel ID: {voxel_id}")
        
        try:
            from load_grids_pickle import VoxelGridsLoader
            
            # Create sample_info dict for compatibility with ImportanceMapAnalyzer method
            sample_info = {
                'zeolite': zeolite,
                'env': env,
                'pore_type': pore_type,
                'environment': environment,
                'adsorbate': adsorbate,
                'snapshot': snapshot,
                'voxel_id': voxel_id
            }
            
            data_loader = VoxelGridsLoader(
                zeolite_types=[sample_info['zeolite']],
                adsorbates_by_env={sample_info['environment']: [sample_info['adsorbate']]},
                box_grids_size=16.0,
                box_increment=0.8,
                num_features=28,  # 14 adsorbate + 14 solvent channels
                verbose=False
            )
            
            loaded_data = data_loader.load_all_pickle_files()
            
            for key, data in loaded_data.items():
                if (data['zeolite_type'] == sample_info['zeolite'] and 
                    data['environment'] == sample_info['environment'] and 
                    data['adsorbate'] == sample_info['adsorbate']):
                    
                    voxel_grids = data['voxel_grids']
                    snapshot_indices = data['snapshot_indices'] 
                    voxel_ids = data['voxel_ids']
                    
                    for i, (snap_idx, vox_id) in enumerate(zip(snapshot_indices, voxel_ids)):
                        if snap_idx == sample_info['snapshot'] and vox_id == sample_info['voxel_id']:
                            voxel_data = voxel_grids[i]
                            
                            print(f"✅ Successfully loaded voxel data:")
                            print(f"   Voxel data shape: {voxel_data.shape}")
                            print(f"   Voxel data range: [{voxel_data.min():.4f}, {voxel_data.max():.4f}]")
                            
                            # Convert to tensor format (1, C, D, H, W) for model input
                            X_tensor = torch.FloatTensor(voxel_data).permute(3, 0, 1, 2).unsqueeze(0)  # (1, 28, 20, 20, 20)
                            
                            # Since we don't have the y value from the loader, we'll get it from the results data
                            y_tensor = torch.FloatTensor([0.0])  # Placeholder, will be updated
                            
                            return X_tensor, y_tensor
            
            print(f"❌ Sample not found in loaded data")
            return None, None
            
        except Exception as e:
            print(f"❌ Error loading voxel data: {e}")
            return None, None

    def get_specific_sample_data(self, zeolite, env_adsorbate, snapshot, voxel_id):
        """Get data for a specific sample and determine which folds it appears in as training data"""
        # Find sample in results
        sample_info, train_folds = self.find_sample_in_results(zeolite, env_adsorbate, snapshot, voxel_id)
        
        if sample_info is None:
            print("Failed to find sample in results")
            return None, None, None
        
        # Load actual voxel data
        sample_data, sample_labels = self.load_actual_voxel_data_for_sample(zeolite, env_adsorbate, snapshot, voxel_id)
        
        if sample_data is None:
            print("Failed to load voxel data")
            return None, None, None
        
        # Update the y value with the actual value from results
        actual_y_value = sample_info['y_true']
        sample_labels = torch.FloatTensor([actual_y_value])
        
        # Create sample fold mapping (single sample)
        sample_fold_mapping = {0: train_folds}  # Single sample at index 0
        
        print(f"\n✅ Successfully loaded specific sample:")
        print(f"   Sample data shape: {sample_data.shape}")
        print(f"   Sample label (from results): {sample_labels.item():.4f}")
        print(f"   Appears as TRAINING data in folds: {[f+1 for f in train_folds]} ({len(train_folds)}/5 folds)")
        
        return sample_data, sample_labels, sample_fold_mapping


    def plot_comprehensive_attention_overlay_heatmap_only(self,
                                                          zeolite,
                                                          env,
                                                          pore_type,
                                                          adsorbate,
                                                          snapshot=0,
                                                          voxel_id=0,
                                                          view_angle='YZ',
                                                          show_plots=False,
                                                          save_fig=False):
        """
        Heatmap-only attention visualization - single plane view (YZ, XZ, or XY)
        
        Layout:
        - Single row: Specified plane attention heatmaps for all 3 CBAM modules (horizontal)
        
        Args:
            zeolite: zeolite type (e.g., 'FAU')
            env: environment (e.g., 'methanol_600_water_600')
            pore_type: pore type (e.g., 'hydrophilic')
            adsorbate: adsorbate (e.g., '11_01_propylene_glycol')
            snapshot: snapshot index (default: 0)
            voxel_id: voxel rotation ID (default: 0)
            view_angle: plane to visualize - 'YZ', 'XZ', or 'XY' (default: 'YZ')
            show_plots: whether to display plots (default: False)
            save_fig: whether to save figure (default: False)
        """
        # Construct env_adsorbate for compatibility
        env_adsorbate = f"{env}-{pore_type}-{adsorbate}"
        
        # Get sample data and fold mapping
        sample_data, sample_labels, sample_fold_mapping = self.get_specific_sample_data(
            zeolite, env_adsorbate, snapshot, voxel_id)
        if sample_data is None:
            print("Failed to get sample data")
            return
        
        print(f"\n🎨 HEATMAP-ONLY ATTENTION VISUALIZATION ({view_angle} Plane)")
        print("="*80)
        print(f"Sample: {zeolite} | {env} | {pore_type} | {adsorbate} | snapshot {snapshot} | voxel {voxel_id}")
        print(f"View: {view_angle} plane")
        
        # Extract voxel data for structure visualization
        voxel_data = sample_data[0].cpu().numpy()  # Shape: (28, 20, 20, 20)
        
        print(f"Voxel data shape: {voxel_data.shape}")
        
        # Extract attention maps from all models
        all_model_attention = {}
        valid_models = []
        
        for fold_idx in range(len(self.model_storage)):
            try:
                model = self.model_storage[fold_idx]['model']
                visualizer = SpatialAttentionVisualizer(model)
                attention_data = visualizer.extract_spatial_attention_maps(sample_data)
                
                if attention_data:
                    all_model_attention[fold_idx] = attention_data
                    valid_models.append(fold_idx)
            except Exception as e:
                print(f"Error with Model {fold_idx + 1}: {e}")
                continue
        
        if not valid_models:
            print("❌ No valid attention data found!")
            return
        
        # Find common modules
        common_modules = set()
        for fold_idx in valid_models:
            fold_modules = set()
            for key in all_model_attention[fold_idx].keys():
                if 'spatial_attention' in key:
                    module_name = key.split('.')[0]
                    fold_modules.add(module_name)
            
            if not common_modules:
                common_modules = fold_modules
            else:
                common_modules = common_modules.intersection(fold_modules)
        
        desired_order = ['adsorbate_processor', 'solvent_processor', 'interaction_attention']
        common_modules = [module for module in desired_order if module in common_modules]
        
        # Average attention across training folds
        sample_averaged_attention = {}
        
        for module_name in common_modules:
            actual_key = None
            for fold_idx in valid_models:
                if fold_idx in all_model_attention:
                    for key in all_model_attention[fold_idx].keys():
                        if module_name in key and 'spatial_attention' in key:
                            actual_key = key
                            break
                    if actual_key:
                        break
            
            if not actual_key:
                continue
            
            train_folds = sample_fold_mapping[0]
            sample_fold_attentions = []
            
            for fold_idx in train_folds:
                if fold_idx in all_model_attention and actual_key in all_model_attention[fold_idx]:
                    attention = all_model_attention[fold_idx][actual_key]
                    sample_attention = attention[0]
                    
                    if len(sample_attention.shape) == 4:
                        sample_attention = sample_attention.squeeze(0)
                    
                    if len(sample_attention.shape) == 3:
                        sample_fold_attentions.append(sample_attention)
            
            if sample_fold_attentions:
                sample_avg_attention = np.mean(sample_fold_attentions, axis=0)
                sample_averaged_attention[module_name] = sample_avg_attention
        
        # Helper function to extract plane slice based on view_angle
        def get_plane_slice(data_3d, plane):
            """Extract plane slice - average of middle two layers for even grids"""
            d, h, w = data_3d.shape
            # For even-sized grids (20x20x20), take average of two middle layers
            # Indices 9 and 10 are the true center for a grid of size 20
            if plane == 'YZ':
                mid1, mid2 = d//2 - 1, d//2  # indices 9, 10
                return (data_3d[mid1, :, :] + data_3d[mid2, :, :]) / 2.0
            elif plane == 'XZ':
                mid1, mid2 = h//2 - 1, h//2  # indices 9, 10
                return (data_3d[:, mid1, :] + data_3d[:, mid2, :]) / 2.0
            else:  # XY
                mid1, mid2 = w//2 - 1, w//2  # indices 9, 10
                return (data_3d[:, :, mid1] + data_3d[:, :, mid2]) / 2.0
        
        # Create heatmap-only visualization
        # Layout: 1 row × 3 columns (one column per module)
        fig = plt.figure(figsize=(18, 6))
        gs = fig.add_gridspec(1, 3,
                              wspace=0.38)  # Width padding between columns
        
        # Module display names
        module_display_names = {
            "adsorbate_processor": "Adsorbate Branch",
            "solvent_processor": "Solvent Branch",
            "interaction_attention": "Interaction Module"
        }
        
        # Single row: attention heatmaps
        for col_idx, module_name in enumerate(common_modules):
            avg_attention = sample_averaged_attention[module_name]
            display_name = module_display_names.get(module_name, module_name)
            
            ax = fig.add_subplot(gs[0, col_idx])
            
            # Get plane slice based on view_angle
            attn_slice = get_plane_slice(avg_attention, view_angle)
            
            # Display attention as heatmap with origin='lower' to match Cartesian coordinates
            # extent: [left, right, bottom, top] in data coordinates
            # For 20x20 grid with 0.8 Å resolution: total = 16 Å, centered at 0
            # Special handling for solvent processor (col_idx==1): set vmin=0.50 to better show high attention regions
            if col_idx == 1:  # Solvent processor
                vmin_val = 0.49  # Fixed lower bound for solvent processor
                vmax_val = avg_attention.max()
            else:  # Adsorbate processor and Interaction attention
                vmin_val = avg_attention.min()
                vmax_val = avg_attention.max()
            
            im = ax.imshow(attn_slice, cmap='jet', alpha=0.85, interpolation='bilinear',
                          vmin=vmin_val, vmax=vmax_val,
                          origin='lower',  # y-axis increases upward like ASE coordinates
                          extent=[-8, 8, -8, 8])  # Center at (0,0), range from -8 to +8 Å
            
            # Add column title
            if col_idx == 0:
                ax.set_title('Adsorbate Branch', fontsize=self.font_size+2, pad=15)
            elif col_idx == 1:
                ax.set_title('Solvent Branch', fontsize=self.font_size+2, pad=15)
            elif col_idx == 2:
                ax.set_title('Interaction Module', fontsize=self.font_size+2, pad=15)
            
            # Set axis labels based on view_angle with units (Angstrom)
            # Use labelpad to reduce distance between axis numbers and axis label
            if view_angle == 'YZ':
                ax.set_xlabel('z (Å)', fontsize=self.font_size, labelpad=0)
                ax.set_ylabel('y (Å)', fontsize=self.font_size, labelpad=0)
            elif view_angle == 'XZ':
                ax.set_xlabel('z (Å)', fontsize=self.font_size, labelpad=0)
                ax.set_ylabel('x (Å)', fontsize=self.font_size, labelpad=0)
            else:  # XY
                ax.set_xlabel('y (Å)', fontsize=self.font_size, labelpad=0)
                ax.set_ylabel('x (Å)', fontsize=self.font_size, labelpad=0)
            ax.tick_params(labelsize=self.font_size)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=self.font_size-2)
            
            # Add subplot label (a), (b), (c) outside the plot area, at top-left corner of each subplot
            subplot_labels = ['(a)', '(b)', '(c)']
            ax.text(-0.25, 1.05, subplot_labels[col_idx], 
                   transform=ax.transAxes,
                   fontsize=self.font_size+2,
                   fontweight='bold',
                   verticalalignment='bottom',
                   horizontalalignment='left')
        
        # Adjust layout
        plt.tight_layout(pad=2.0, w_pad=2.0)
        
        # Save figure if requested
        if save_fig:
            filename = "figure_7-attention_heatmaps.png"
            save_path = os.path.join(self.output_dir, filename)
            plt.savefig(save_path, dpi=1000, bbox_inches='tight', facecolor='white')
            print(f"\n✅ Saved heatmap-only {view_angle} plane attention plot: {save_path}")
        
        if show_plots:
            plt.show()
        else:
            pass


if __name__ == "__main__":
    
    # Model results file
    results_filename = "model.pkl"
    
    
    ## Define the simulation parameters
    zeolite_type = 'FAU'           # e.g. "FAU", "BEA" or "MFI"
    solvent_type = 'water_pure'    # e.g. "water_pure", "methanol_120_water_1080", "methanol_240_water_960", "methanol_600_water_600"
    pore_type = 'hydrophilic'      # e.g. "hydrophilic", "hydrophobic"
    adsorbate = '11_01_propylene_glycol'      # e.g. "01_methanol", "02_01_02_propanol"
    snapshot_index = 6
    voxel_id = 1
    
    
    # Create analyzer
    analyzer = SpatialAttentionAnalyzer(results_filename,
                                        font_size=19)
    
    
    # Plot comprehensive attention overlay heatmap only
    analyzer.plot_comprehensive_attention_overlay_heatmap_only(
        zeolite=zeolite_type,
        env=solvent_type,
        pore_type=pore_type,
        adsorbate=adsorbate,
        snapshot=snapshot_index,
        voxel_id=voxel_id,
        view_angle='XY', # 'XY', 'XZ', 'YZ'
        show_plots=False,
        save_fig=True, # True
    )
