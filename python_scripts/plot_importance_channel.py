# -*- coding: utf-8 -*-
"""
plot_importance_maps.py
Simplified script to generate 3D importance maps from trained PyTorch 3D CNN models.
Just specify your model file and sample parameters to generate all plots.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import pickle
import io
import torch
import torch
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
import io

from core.path import get_paths
from model_3d_cnn import AttentionCNN


feature_map = {
                'atom_type_C':          'Atom type (one-hot) Carbon',
                'atom_type_H':          'Atom type (one-hot) Hydrogen',
                'atom_type_O':          'Atom type (one-hot) Oxygen',
                'is_hydrophobic':       'Hydrophobic (binary)',
                'is_donor':             'H-bond Donor Potential (binary)',
                'is_acceptor':          'H-bond Acceptor Potential (binary)',
                'is_hbonded':           'H-bond Formation (binary)',
                'is_hbonded_donor':     'Active H-bond Donor (binary)',
                'is_hbonded_acceptor':  'Active H-bond Acceptor (binary)',
                'atom_mass':            'Atomic Mass',
                'partial_charge':       'Partial Charge',
                'valence':              'Atom Valence',
                'LJ_epsilon':           'Lennard-Jones parameter ε',
                'LJ_sigma':             'Lennard-Jones parameter σ'
                }

feature_map = {
                'atom_type_C':          'Atom type C (one-hot)',
                'atom_type_H':          'Atom type H (one-hot)',
                'atom_type_O':          'Atom type O (one-hot)',
                'is_hydrophobic':       'Hydrophobic Atom (binary)',
                'is_donor':             'H-bond Donor (binary)',
                'is_acceptor':          'H-bond Acceptor (binary)',
                'is_hbonded':           'Is H-bonded (binary)',
                'is_hbonded_donor':     'Is H-bonded Donor (binary)',
                'is_hbonded_acceptor':  'Is H-bonded Acceptor (binary)',
                'atom_mass':            'Atom Mass',
                'partial_charge':       'Partial Charge',
                'valence':              'Valence',
                'LJ_epsilon':           'Lennard-Jones parameter ε',
                'LJ_sigma':             'Lennard-Jones parameter σ'
                }


class IntegratedGradients3D:
    """3D Integrated Gradients implementation for PyTorch models with multi-level analysis"""
    
    def __init__(self,
                 model,
                 device=None,
                 ):
        
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Store intermediate activations for multi-level analysis
        self.intermediate_activations = {}
        self.hooks = []
        
    def register_hooks_for_multilevel_analysis(self):
        """Register forward hooks to capture intermediate activations"""
        self.intermediate_activations = {}
        
        def create_hook(name):
            def hook(module, input, output):
                # Keep gradient information for backpropagation
                self.intermediate_activations[name] = output
            return hook
        
        # Register hooks for different processing stages
        if hasattr(self.model, 'adsorbate_processor'):
            hook = self.model.adsorbate_processor.register_forward_hook(create_hook('adsorbate_features'))
            self.hooks.append(hook)
            
        if hasattr(self.model, 'solvent_processor'):  
            hook = self.model.solvent_processor.register_forward_hook(create_hook('solvent_features'))
            self.hooks.append(hook)
            
        if hasattr(self.model, 'interaction_conv'):
            hook = self.model.interaction_conv.register_forward_hook(create_hook('interaction_features'))
            self.hooks.append(hook)
            
        print(f"    Registered hooks for {len(self.hooks)} layers")
        
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.intermediate_activations = {}
    
    def generate_baseline(self, input_tensor, method='zero'):
        """Generate baseline for integrated gradients"""
        if method == 'zero':
            return torch.zeros_like(input_tensor)
        elif method == 'random':
            return torch.randn_like(input_tensor) * 0.1
        elif method == 'mean':
            baseline = torch.zeros_like(input_tensor)
            for c in range(input_tensor.shape[1]):
                baseline[:, c] = input_tensor[:, c].mean()
            return baseline
        else:
            return torch.zeros_like(input_tensor)
    
    def compute_integrated_gradients(self, input_tensor, target_class=None, steps=50, baseline_method='zero'):
        """
        Compute integrated gradients for 3D input with improved gradient handling
        
        Args:
            input_tensor: (1, C, D, H, W) tensor
            target_class: target class for attribution (None for regression)
            steps: number of integration steps
            baseline_method: method to generate baseline
        
        Returns:
            attributions: (C, D, H, W) attribution map
        """
        print(f"    Computing integrated gradients with {steps} steps...")
        print(f"    Input tensor shape: {input_tensor.shape}")
        print(f"    Input tensor range: [{input_tensor.min():.6f}, {input_tensor.max():.6f}]")
        
        baseline = self.generate_baseline(input_tensor, baseline_method)
        alphas = torch.linspace(0, 1, steps, device=self.device)
        integrated_grads = torch.zeros_like(input_tensor)
        
        # Get initial prediction for reference
        with torch.no_grad():
            baseline_pred = self.model(baseline)
            input_pred = self.model(input_tensor)
            pred_diff = input_pred - baseline_pred
            print(f"    Prediction difference (input - baseline): {pred_diff.item():.6f}")
        
        for i, alpha in enumerate(alphas):
            interpolated_input = baseline + alpha * (input_tensor - baseline)
            interpolated_input.requires_grad_(True)
            
            output = self.model(interpolated_input)
            target_output = output.sum() if target_class is None else output[0, target_class]
            
            # Compute gradients
            grads = torch.autograd.grad(
                outputs=target_output,
                inputs=interpolated_input,
                create_graph=False,
                retain_graph=False,
                allow_unused=False
            )[0]
            
            integrated_grads += grads
            
            # Print progress for long computations
            if (i + 1) % (steps // 4) == 0:
                print(f"    - Progress: {i+1}/{steps} steps completed")
        
        integrated_grads = integrated_grads / steps
        attributions = integrated_grads * (input_tensor - baseline)
        
        # Print attribution statistics
        print(f"    Attribution statistics:")
        print(f"    - Raw gradients range: [{integrated_grads.min():.8f}, {integrated_grads.max():.8f}]")
        print(f"    - Final attributions range: [{attributions.min():.8f}, {attributions.max():.8f}]")
        print(f"    - Attribution sum: {attributions.sum():.8f}")
        print(f"    - Prediction difference: {pred_diff.item():.8f}")

        return attributions.detach().squeeze(0)  # Return (C, D, H, W)
    
    def compute_multilevel_integrated_gradients(self, input_tensor, target_class=None, steps=50, baseline_method='zero'):
        """
        Compute integrated gradients at multiple levels: input, branches, and interaction
        
        Args:
            input_tensor: (1, C, D, H, W) tensor
            target_class: target class for attribution (None for regression)
            steps: number of integration steps
            baseline_method: method to generate baseline
        
        Returns:
            dict: multi-level attribution maps
        """
        print(f"    Computing multi-level integrated gradients with {steps} steps...")
        print(f"    Input tensor shape: {input_tensor.shape}")
        
        # Register hooks to capture intermediate features
        self.register_hooks_for_multilevel_analysis()
        
        baseline = self.generate_baseline(input_tensor, baseline_method)
        alphas = torch.linspace(0, 1, steps, device=self.device)
        
        # Initialize gradients for different levels
        input_integrated_grads = torch.zeros_like(input_tensor)
        adsorbate_integrated_grads = None
        solvent_integrated_grads = None  
        interaction_integrated_grads = None
        
        # Get initial prediction for reference
        with torch.no_grad():
            baseline_pred = self.model(baseline)
            input_pred = self.model(input_tensor)
            pred_diff = input_pred - baseline_pred
            
        print(f"    Baseline prediction: {baseline_pred.item():.6f}")
        print(f"    Input prediction: {input_pred.item():.6f}")
        print(f"    Prediction difference: {pred_diff.item():.6f}")
        
        for i, alpha in enumerate(alphas):
            # Interpolated input
            interpolated_input = baseline + alpha * (input_tensor - baseline)
            interpolated_input.requires_grad_(True)
            
            # Forward pass to capture intermediate activations
            output = self.model(interpolated_input)
            target_output = output.sum() if target_class is None else output[0, target_class]
            
            # Compute gradients w.r.t. input
            input_grad = torch.autograd.grad(
                outputs=target_output,
                inputs=interpolated_input,
                create_graph=True,
                retain_graph=True
            )[0]
            input_integrated_grads += input_grad
            
            # Compute gradients w.r.t. intermediate features if available
            if 'adsorbate_features' in self.intermediate_activations:
                adsorbate_features = self.intermediate_activations['adsorbate_features']
                # Ensure gradients are enabled
                if adsorbate_features.requires_grad:
                    try:
                        adsorbate_grad = torch.autograd.grad(
                            outputs=target_output,
                            inputs=adsorbate_features,
                            create_graph=False,
                            retain_graph=True,
                            allow_unused=True
                        )[0]
                        
                        if adsorbate_grad is not None:
                            if adsorbate_integrated_grads is None:
                                adsorbate_integrated_grads = torch.zeros_like(adsorbate_grad)
                            adsorbate_integrated_grads += adsorbate_grad
                    except RuntimeError as e:
                        if i == 0:  # Only print warning once
                            print(f"      Warning: Could not compute adsorbate gradients: {e}")
                
            if 'solvent_features' in self.intermediate_activations:
                solvent_features = self.intermediate_activations['solvent_features']
                if solvent_features.requires_grad:
                    try:
                        solvent_grad = torch.autograd.grad(
                            outputs=target_output,
                            inputs=solvent_features,
                            create_graph=False,
                            retain_graph=True,
                            allow_unused=True
                        )[0]
                        
                        if solvent_grad is not None:
                            if solvent_integrated_grads is None:
                                solvent_integrated_grads = torch.zeros_like(solvent_grad)
                            solvent_integrated_grads += solvent_grad
                    except RuntimeError as e:
                        if i == 0:  # Only print warning once
                            print(f"      Warning: Could not compute solvent gradients: {e}")
                
            if 'interaction_features' in self.intermediate_activations:
                interaction_features = self.intermediate_activations['interaction_features']
                if interaction_features.requires_grad:
                    try:
                        interaction_grad = torch.autograd.grad(
                            outputs=target_output,
                            inputs=interaction_features,
                            create_graph=False,
                            retain_graph=True,
                            allow_unused=True
                        )[0]
                        
                        if interaction_grad is not None:
                            if interaction_integrated_grads is None:
                                interaction_integrated_grads = torch.zeros_like(interaction_grad)
                            interaction_integrated_grads += interaction_grad
                    except RuntimeError as e:
                        if i == 0:  # Only print warning once
                            print(f"      Warning: Could not compute interaction gradients: {e}")
            
            if (i + 1) % (steps // 4) == 0:
                print(f"    Progress: {i+1}/{steps} steps completed")
        
        # Average and compute final attributions
        input_integrated_grads = input_integrated_grads / steps
        input_attributions = input_integrated_grads * (input_tensor - baseline)
        
        results = {
            'input_attributions': input_attributions.detach().squeeze(0),
            'prediction_diff': pred_diff.item()
        }
        
        # Process intermediate level attributions
        if adsorbate_integrated_grads is not None:
            adsorbate_integrated_grads = adsorbate_integrated_grads / steps
            # For intermediate features, we use the gradients directly as spatial importance
            results['adsorbate_spatial'] = adsorbate_integrated_grads.detach().squeeze(0)
            print(f"    Adsorbate features shape: {results['adsorbate_spatial'].shape}")
            
        if solvent_integrated_grads is not None:
            solvent_integrated_grads = solvent_integrated_grads / steps
            results['solvent_spatial'] = solvent_integrated_grads.detach().squeeze(0) 
            print(f"    Solvent features shape: {results['solvent_spatial'].shape}")
            
        if interaction_integrated_grads is not None:
            interaction_integrated_grads = interaction_integrated_grads / steps
            results['interaction_spatial'] = interaction_integrated_grads.detach().squeeze(0)
            print(f"    Interaction features shape: {results['interaction_spatial'].shape}")
        
        # Remove hooks
        self.remove_hooks()
        
        # Print attribution statistics
        print(f"    Multi-level attribution statistics:")
        print(f"    - Input attributions range: [{results['input_attributions'].min():.8f}, {results['input_attributions'].max():.8f}]")
        if 'interaction_spatial' in results:
            print(f"    - Interaction spatial range: [{results['interaction_spatial'].min():.8f}, {results['interaction_spatial'].max():.8f}]")

        return results
    
    def compute_layerwise_spatial_importance(self, input_tensor, target_class=None):
        """
        Compute spatial importance at different processor layers using gradient-based approach
        This computes gradients with respect to the output of each processor branch
        """
        print(f"    Computing layer-wise spatial importance using gradient-based analysis...")
        
        # Store intermediate activations that require gradients
        self.intermediate_activations = {}
        
        def create_hook(name):
            def hook(module, input, output):
                # Store output and enable gradient computation
                output.retain_grad()
                self.intermediate_activations[name] = output
            return hook
        
        # Register hooks for different processing stages
        hooks = []
        if hasattr(self.model, 'adsorbate_processor'):
            hook = self.model.adsorbate_processor.register_forward_hook(create_hook('adsorbate_features'))
            hooks.append(hook)
            
        if hasattr(self.model, 'solvent_processor'):  
            hook = self.model.solvent_processor.register_forward_hook(create_hook('solvent_features'))
            hooks.append(hook)
            
        if hasattr(self.model, 'interaction_conv'):
            hook = self.model.interaction_conv.register_forward_hook(create_hook('interaction_features'))
            hooks.append(hook)
        
        print(f"    Registered hooks for {len(hooks)} processor layers")
        
        # Forward pass with gradient computation
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        
        # Compute gradients for each processor layer
        multilevel_data = {}
        
        for layer_name, features in self.intermediate_activations.items():
            print(f"      Computing gradients for {layer_name}: shape {features.shape}")
            
            # Clear any existing gradients
            self.model.zero_grad()
            if input_tensor.grad is not None:
                input_tensor.grad.zero_()
            
            # Compute gradients of output w.r.t. this layer's features
            grad_outputs = torch.autograd.grad(
                outputs=output,
                inputs=features,
                grad_outputs=torch.ones_like(output),
                create_graph=False,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            # Compute spatial importance as mean absolute gradient across channels
            if len(grad_outputs.shape) == 5:  # (batch, channels, D, H, W)
                spatial_importance = torch.mean(torch.abs(grad_outputs.squeeze(0)), dim=0)  # Average across channels
                multilevel_data[layer_name.replace('_features', '_spatial')] = spatial_importance
                print(f"        Computed gradient-based spatial importance: {spatial_importance.shape}")
                print(f"        Gradient range: [{grad_outputs.min():.6f}, {grad_outputs.max():.6f}]")
                print(f"        Spatial importance range: [{spatial_importance.min():.6f}, {spatial_importance.max():.6f}]")
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Create results similar to integrated gradients format
        results = {
            'prediction_diff': 0.0,  # Not applicable for single-step gradient method
        }
        
        # Add spatial data
        for key, value in multilevel_data.items():
            results[key] = value.cpu()
            
        print(f"    Layer-wise gradient analysis completed. Available data: {list(results.keys())}")
        
        return results
    
    def compute_simple_gradients(self, input_tensor, target_class=None):
        """
        Compute simple input gradients (faster alternative)
        """
        print(f"  Computing simple gradients...")
        
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        target_output = output.sum() if target_class is None else output[0, target_class]
        
        grads = torch.autograd.grad(
            outputs=target_output,
            inputs=input_tensor,
            create_graph=False,
            retain_graph=False
        )[0]
        
        print(f"  Simple gradients range: [{grads.min():.8f}, {grads.max():.8f}]")
        
        return grads.detach().squeeze(0)
    
    def compute_guided_backprop(self, input_tensor, target_class=None):
        """
        Compute guided backpropagation for better visualization
        """
        print(f"  Computing guided backpropagation...")
        
        # Store original ReLU backward functions
        relu_functions = []
        
        def relu_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, set it to zero
            """
            if isinstance(module, torch.nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)
        
        # Register hooks for all ReLU layers
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                handle = module.register_backward_hook(relu_hook_function)
                relu_functions.append(handle)
        
        # Compute gradients
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        target_output = output.sum() if target_class is None else output[0, target_class]
        
        grads = torch.autograd.grad(
            outputs=target_output,
            inputs=input_tensor,
            create_graph=False,
            retain_graph=False
        )[0]
        
        # Remove hooks
        for handle in relu_functions:
            handle.remove()
        
        print(f"  Guided backprop range: [{grads.min():.8f}, {grads.max():.8f}]")
        
        return grads.detach().squeeze(0)


class CPUUnpickler(pickle.Unpickler):
    """Custom unpickler that maps CUDA tensors to CPU"""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


class ImportanceMapAnalyzer:
    """Simplified class for generating importance maps from trained 3D CNN models"""
    
    def __init__(self, results_filename, device=None, font_size=12):
        """
        Initialize the importance map analyzer
        
        Args:
            results_filename: name of the results pkl file
            device: compute device (cpu/cuda)
            font_size: font size for all plots
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.font_size = font_size
        print(f"Using device: {self.device}")
        print(f"Using font size: {self.font_size}")
        
        # Default names for the 14 adsorbate and 14 solvent feature channels.
        # First 14 channels: adsorbate features, Last 14 channels: solvent features
        self.atomic_features = [
            'atom_type_C', 'atom_type_H', 'atom_type_O',
            'is_hydrophobic', 'is_donor', 'is_acceptor', 'is_hbonded',
            'is_hbonded_donor', 'is_hbonded_acceptor', 'atom_mass',
            'partial_charge', 'valence', 'LJ_epsilon', 'LJ_sigma'
        ]
        
        # Create combined feature names for visualization
        self.feature_names = []
        # Add adsorbate channels (0-13)
        for feat in self.atomic_features:
            self.feature_names.append(f'adsorbate_{feat}')
        # Add solvent channels (14-27)
        for feat in self.atomic_features:
            self.feature_names.append(f'solvent_{feat}')
        
        # Set up paths
        self.results_path = os.path.join(get_paths("output_model_cnn"), results_filename)
        self.output_dir = os.path.join(get_paths("output_figure_path"), "cnn_importance")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"--- Output directory: {self.output_dir}")
        
        # Extract model name prefix from filename for plot naming
        self.model_prefix = self._extract_model_prefix(results_filename)
        print(f"--- Model prefix for plots: {self.model_prefix}")

    def _extract_model_prefix(self, results_filename):
        """
        Extract model prefix from results filename for consistent plot naming
        
        Args:
            results_filename: e.g., "model-random-2546229-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl"
        
        Returns:
            str: model prefix, e.g., "model-random-2546229"
        """
        # Remove .pkl extension
        base_name = results_filename.replace('.pkl', '')
        
        # Extract the part before '-epochs' (which contains model info + job number)
        if '-epochs' in base_name:
            model_prefix = base_name.split('-epochs')[0]
        else:
            # Fallback: extract the part before the first dash if no '-epochs' found
            if '-' in base_name:
                model_prefix = base_name.split('-')[0]
            else:
                model_prefix = base_name
        
        return model_prefix

    def analyze_sample(self, zeolite, env_adsorbate, snapshot=0, voxel_id=0, steps=50, 
                      ensemble_method='mean'):
        """
        Main function to analyze a specific sample and generate importance map
        
        Args:
            zeolite: zeolite type (e.g., 'FAU')
            env_adsorbate: environment-adsorbate combination (e.g., 'water_pure-hydrophilic-02_propanol')
            snapshot: snapshot index
            voxel_id: voxel rotation ID
            steps: integration steps for gradients
            ensemble_method: 'mean', 'median', or 'stack' for combining multiple fold results
                - 'mean': average importance maps across folds
                - 'median': median importance maps across folds  
                - 'stack': sum importance maps across folds (accumulative effect)
        """
        # Create cache filename based on analysis parameters
        cache_filename = f"{self.model_prefix}-{zeolite}-{env_adsorbate}-snap{snapshot}-vox{voxel_id}-steps{steps}.pkl"
        cache_path = os.path.join(self.output_dir, "cache", cache_filename)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        # Check if cached result exists
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_result = pickle.load(f)
                print(f"✓ Loaded cached analysis result from: {cache_filename}")
                return cached_result
            except Exception as e:
                print(f"⚠️ Failed to load cache, will recompute: {e}")
        
        print(f"\n--- Analyzing sample:")
        print(f"    Zeolite: {zeolite}")
        print(f"    Env-Adsorbate: {env_adsorbate}")
        print(f"    Snapshot: {snapshot}, Voxel ID: {voxel_id}")
        print(f"    Integration steps: {steps}")
        print(f"    Ensemble method: {ensemble_method}")
        
        print(f"\n=== Generating Importance Map ===")
        
        # Load results
        results = self.load_results(self.results_path)
        
        # Always use ensemble analysis of train folds
        analysis_result = self._analyze_sample_ensemble(results, zeolite, env_adsorbate, 
                                                       snapshot, voxel_id, steps, ensemble_method)
        
        # Save result to cache
        if analysis_result is not None:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(analysis_result, f)
                print(f"✓ Cached analysis result to: {cache_filename}")
            except Exception as e:
                print(f"⚠️ Failed to save cache: {e}")
        
        return analysis_result
    
    def _find_train_folds_for_sample(self, results, zeolite, env_adsorbate, snapshot, voxel_id):
        """Find all train folds that contain the target sample"""
        # Parse env_adsorbate
        if '-' in env_adsorbate:
            parts = env_adsorbate.split('-')
            if len(parts) >= 3:
                environment = f"{parts[0]}-{parts[1]}"
                adsorbate = parts[2]
            else:
                environment = parts[0]
                adsorbate = parts[1]
        else:
            raise ValueError(f"Invalid env_adsorbate format: {env_adsorbate}")
        
        train_folds = []
        
        for fold_idx in sorted(results['model_storage'].keys()):
            fold_data = results['model_storage'][fold_idx]
            
            # Only check train set
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
                    train_folds.append({
                        'fold_idx': fold_idx,
                        'split': 'train',
                        'zeolite': zeolite,
                        'environment': environment,
                        'adsorbate': adsorbate,
                        'snapshot': snapshot,
                        'voxel_id': voxel_id,
                        'y_true': sample.get('y_true', 'N/A'),
                        'y_pred': sample.get('y_pred', 'N/A')
                    })
        
        return train_folds
    
    def _analyze_sample_ensemble(self, results, zeolite, env_adsorbate, snapshot, voxel_id, steps, ensemble_method):
        """Ensemble analysis of multiple train folds"""
        print(f"\n--- Ensemble Analysis Mode ---")
        
        # Find all train folds
        train_folds = self._find_train_folds_for_sample(results, zeolite, env_adsorbate, snapshot, voxel_id)
        
        if len(train_folds) == 0:
            print(f"❌ No train folds found, falling back to single analysis")
            return self._analyze_sample_single(results, zeolite, env_adsorbate, snapshot, voxel_id, steps, None)
        
        print(f"✓ Found {len(train_folds)} train folds: {[f['fold_idx'] for f in train_folds]}")
        
        # Collect importance maps from each fold
        fold_importance_maps = []
        fold_predictions = []
        fold_multilevel_data = []  # New: collect multi-level data
        
        for fold_info in train_folds:
            fold_idx = fold_info['fold_idx']
            print(f"\n  Analyzing Fold {fold_idx}...")
            
            # Use single analysis to get results for this fold
            single_result = self._analyze_sample_single(results, zeolite, env_adsorbate, 
                                                      snapshot, voxel_id, steps, fold_idx)
            
            if single_result is not None:
                fold_importance_maps.append(single_result['importance_map'])
                fold_predictions.append({
                    'fold': fold_idx,
                    'y_true': fold_info['y_true'],
                    'y_pred': fold_info['y_pred']
                })
                
                # Collect multi-level data if available
                if single_result.get('multilevel_data') is not None:
                    fold_multilevel_data.append(single_result['multilevel_data'])
                    
                print(f"    ✓ Fold {fold_idx}: y_pred={fold_info['y_pred']:.4f}")
                if single_result.get('has_multilevel', False):
                    print(f"      Multi-level data available")
            else:
                print(f"    ❌ Fold {fold_idx}: analysis failed")
        
        if len(fold_importance_maps) == 0:
            print(f"❌ No successful fold analyses")
            return None
        
        # Ensemble importance maps
        print(f"\n--- Ensembling {len(fold_importance_maps)} fold results (method: {ensemble_method}) ---")
        
        if ensemble_method == 'mean':
            ensemble_importance = np.mean(fold_importance_maps, axis=0)
        elif ensemble_method == 'median':
            ensemble_importance = np.median(fold_importance_maps, axis=0)
        elif ensemble_method == 'stack':
            ensemble_importance = np.sum(fold_importance_maps, axis=0)
            print(f"    Using stack method: summing all {len(fold_importance_maps)} fold importance maps")
        else:
            print(f"⚠️ Unknown ensemble method {ensemble_method}, using mean")
            ensemble_importance = np.mean(fold_importance_maps, axis=0)
        
        # Calculate consistency metrics
        fold_consistency = np.std(fold_importance_maps, axis=0)
        mean_consistency = np.mean(fold_consistency)
        
        # Ensemble multi-level data if available
        ensemble_multilevel_data = None
        if len(fold_multilevel_data) > 0:
            print(f"    Ensembling multi-level data from {len(fold_multilevel_data)} folds...")
            ensemble_multilevel_data = {}
            
            # Get all available keys from multi-level data
            all_keys = set()
            for fold_data in fold_multilevel_data:
                all_keys.update(fold_data.keys())
            
            for key in all_keys:
                # Collect data for this key from all folds that have it
                key_data = []
                for fold_data in fold_multilevel_data:
                    if key in fold_data:
                        key_data.append(fold_data[key])
                
                if len(key_data) > 0:
                    if ensemble_method == 'mean':
                        ensemble_multilevel_data[key] = np.mean(key_data, axis=0)
                    elif ensemble_method == 'median':
                        ensemble_multilevel_data[key] = np.median(key_data, axis=0)
                    elif ensemble_method == 'stack':
                        ensemble_multilevel_data[key] = np.sum(key_data, axis=0)
                    
                    print(f"      {key}: ensembled from {len(key_data)} folds")
        
        # Create ensemble results
        ensemble_results = {
            'sample_info': {
                'zeolite': zeolite,
                'environment': train_folds[0]['environment'],
                'adsorbate': train_folds[0]['adsorbate'],
                'snapshot': snapshot,
                'voxel_id': voxel_id,
                'fold_idx': f"ensemble_{len(fold_importance_maps)}folds",
                'split': 'train_ensemble',
                'y_true': train_folds[0]['y_true'],
                'y_pred': np.mean([p['y_pred'] for p in fold_predictions if isinstance(p['y_pred'], (int, float))])
            },
            'importance_map': ensemble_importance,
            'multilevel_data': ensemble_multilevel_data,  # New: ensembled multi-level data
            'fold_importance_maps': fold_importance_maps,
            'fold_consistency': fold_consistency,
            'mean_consistency': mean_consistency,
            'fold_predictions': fold_predictions,
            'ensemble_method': ensemble_method,
            'num_folds': len(fold_importance_maps),
            'data_source': "ensemble_train"
        }
        
        print(f"✓ Ensemble completed:")
        print(f"    Number of folds used: {len(fold_importance_maps)}")
        print(f"    Mean consistency: {mean_consistency:.6f}")
        print(f"    Prediction range: {[p['y_pred'] for p in fold_predictions]}")
        
        return ensemble_results
    
    def _analyze_sample_single(self, results, zeolite, env_adsorbate, snapshot, voxel_id, steps, fold_num):
        """Single fold importance analysis (original logic)"""
        
        # 查找样本信息
        sample_info = self.find_sample(results, zeolite, env_adsorbate, snapshot, voxel_id, fold_num)
        
        if sample_info is None:
            print("❌ Failed to find sample")
            return None
        
        # Load model for the fold
        model, scaler_info = self.load_model_for_fold(self.results_path, sample_info['fold_idx'])
        
        # Load actual voxel data
        voxel_data = self.load_actual_voxel_data(sample_info)
        
        print(f"  ✓ Loaded actual voxel data")
        print(f"    Voxel data shape: {voxel_data.shape}")
        print(f"    Voxel data type: {type(voxel_data)}")
        print(f"    Voxel data range: [{voxel_data.min():.4f}, {voxel_data.max():.4f}]")
        
        input_tensor = torch.FloatTensor(voxel_data).unsqueeze(0).to(self.device)
        print(f"    Input tensor shape before permute: {input_tensor.shape}")
        input_tensor = input_tensor.permute(0, 4, 1, 2, 3)  # (1, 28, 20, 20, 20)
        print(f"    Input tensor shape after permute: {input_tensor.shape}")
        data_source = "actual"

        
        # Verify prediction
        with torch.no_grad():
            raw_prediction = model(input_tensor)
            
            if scaler_info:
                scaler_mean = scaler_info.get('mean', 0.0)
                scaler_std = scaler_info.get('std', 1.0)
                current_prediction = raw_prediction.item() * scaler_std + scaler_mean
            else:
                current_prediction = raw_prediction.item()
        
        prediction_diff = abs(current_prediction - sample_info['y_pred'])
        print(f"    Prediction - Current: {current_prediction:.4f}, Expected: {sample_info['y_pred']:.4f}")
        
        # Generate importance map using integrated gradients
        ig = IntegratedGradients3D(model, self.device)
        print(f"\n--- Computing integrated gradients...")
        
        # Check if model supports multi-level analysis
        has_branches = (hasattr(model, 'adsorbate_processor') and 
                       hasattr(model, 'solvent_processor') and 
                       hasattr(model, 'interaction_conv'))
        
        if has_branches:
            print(f"    Model supports multi-level analysis (adsorbate/solvent branches + interaction)")
            
            # First compute standard integrated gradients for input channels
            importance_map = ig.compute_integrated_gradients(input_tensor, steps=steps).cpu().numpy()
            
            # Then compute layer-wise spatial importance using activation analysis
            layerwise_results = ig.compute_layerwise_spatial_importance(input_tensor)
            
            # Store multi-level results
            multilevel_data = {}
            for key in ['adsorbate_spatial', 'solvent_spatial', 'interaction_spatial']:
                if key in layerwise_results:
                    multilevel_data[key] = layerwise_results[key].cpu().numpy()
        else:
            print(f"    Model uses standard analysis (no branch support)")
            importance_map = ig.compute_integrated_gradients(input_tensor, steps=steps).cpu().numpy()
            multilevel_data = None
        
        # Create analysis results
        analysis_results = {
            'sample_info': sample_info,
            'importance_map': importance_map,
            'multilevel_data': multilevel_data,  # New: multi-level importance data
            'prediction_diff': prediction_diff,
            'data_source': data_source,
            'input_tensor': input_tensor.detach().cpu().numpy(),
            'raw_prediction': raw_prediction.item(),
            'scaler_info': scaler_info,
            'has_multilevel': has_branches  # Flag to indicate multi-level support
        }
        
        # Show sample info
        print(f"\n--- Sample Analysis Summary:")
        print(f"    Sample: {sample_info['zeolite']}-{sample_info['environment']}-{sample_info['adsorbate']}")
        print(f"    Snapshot: {sample_info['snapshot']}, Voxel ID: {sample_info['voxel_id']}")
        print(f"    Fold: {sample_info['fold_idx']}, Split: {sample_info['split']}")
        print(f"    True value: {sample_info['y_true']:.4f}")
        print(f"    Predicted value: {sample_info['y_pred']:.4f}")
        print(f"    Data source: {analysis_results['data_source']}")
        print(f"    Prediction difference: {analysis_results['prediction_diff']:.6f}")

        return analysis_results
    
    def load_results(self, results_path):
        """Load trained models from results pickle file"""
        print(f"    Loading results from: {results_path}")
        
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Results file not found: {results_path}")
        
        try:
            # Try regular pickle loading first
            with open(results_path, 'rb') as f:
                results = pickle.load(f)
            print(f"  ✓ Loaded {len(results['model_storage'])} models")
            return results
        except Exception as e:
            print(f"    Regular pickle loading failed: {e}")
            # Try with custom CPU unpickler
            try:
                with open(results_path, 'rb') as f:
                    results = CPUUnpickler(f).load()
                print(f"  ✓ Loaded with CPU unpickler: {len(results['model_storage'])} models")
                return results
            except Exception as e2:
                print(f"    CPU unpickler failed: {e2}")
                # Last resort: try torch.load
                try:
                    results = torch.load(results_path, map_location='cpu', weights_only=False)
                    print(f"  ✓ Loaded with torch.load: {len(results['model_storage'])} models")
                    return results
                except Exception as e3:
                    print(f"All loading methods failed:")
                    print(f"  1. Regular pickle: {e}")
                    print(f"  2. CPU unpickler: {e2}")
                    print(f"  3. Torch load: {e3}")
                    raise RuntimeError("Cannot load results file. The file may contain CUDA tensors that cannot be mapped to CPU.")

    def find_sample(self, results, zeolite, env_adsorbate, snapshot=0, voxel_id=0, fold_num=None):
        """
        Find a specific sample in the results based on criteria
        
        Args:
            results: loaded results dictionary
            zeolite: zeolite type (e.g., 'FAU')
            env_adsorbate: environment-adsorbate combination (e.g., 'water_pure-hydrophilic-01_methanol')
            snapshot: snapshot index (default: 0, first snapshot)
            voxel_id: voxel rotation ID (default: 0, first rotation)
            fold_num: specific fold to search (0-4), if None will search all folds
        
        Returns:
            dict: sample information including fold, index, and metadata
        """
        print(f"\n--- Searching for sample:")
        print(f"    Zeolite: {zeolite}")
        print(f"    Env-Adsorbate: {env_adsorbate}")
        print(f"    Snapshot: {snapshot}")
        print(f"    Voxel ID: {voxel_id}")
        if fold_num is not None:
            print(f"    Target fold: {fold_num}")

        # Parse env_adsorbate
        if '-' in env_adsorbate:
            parts = env_adsorbate.split('-')
            if len(parts) >= 3:
                environment = f"{parts[0]}-{parts[1]}"
                adsorbate = parts[2]
            else:
                environment = parts[0]
                adsorbate = parts[1]
        else:
            raise ValueError(f"Invalid env_adsorbate format: {env_adsorbate}")
        
        print(f"    Parsed - Environment: {environment}, Adsorbate: {adsorbate}")
        
        # Determine which folds to search
        if fold_num is not None:
            # Search only the specified fold
            if fold_num not in results['model_storage']:
                print(f"❌ Fold {fold_num} not found in results")
                return None
            folds_to_search = [fold_num]
            print(f"    Searching only fold {fold_num}")
        else:
            # Search all folds (original behavior)
            folds_to_search = list(results['model_storage'].keys())
            print(f"    Searching all folds: {folds_to_search}")
        
        # Search through specified folds
        for fold_idx in folds_to_search:
            model_data = results['model_storage'][fold_idx]
            print(f"    Checking fold {fold_idx}...")
            
            # Check test set first (preferred for analysis), then train set
            for split_name in ['df_test', 'df_train']:
                if split_name not in model_data:
                    continue
                
                df = model_data[split_name]
                
                # Find matching samples
                mask = (
                    (df['zeolite'] == zeolite) & 
                    (df['environment'] == environment) & 
                    (df['adsorbate'] == adsorbate) & 
                    (df['snapshot'] == snapshot) & 
                    (df['voxel_id'] == voxel_id)
                )
                
                matching_samples = df[mask]
                
                if len(matching_samples) > 0:
                    sample_idx = matching_samples.index[0]
                    sample_row = matching_samples.iloc[0]
                    
                    sample_info = {
                        'fold_idx': fold_idx,
                        'sample_idx': sample_idx,
                        'split': split_name.replace('df_', ''),
                        'zeolite': sample_row['zeolite'],
                        'environment': sample_row['environment'],
                        'adsorbate': sample_row['adsorbate'],
                        'snapshot': sample_row['snapshot'],
                        'voxel_id': sample_row['voxel_id'],
                        'y_true': sample_row['y_true'],
                        'y_pred': sample_row['y_pred']
                    }
                    
                    print(f"  ✓ Found sample in fold {fold_idx} ({split_name.replace('df_', '')} set)")
                    print(f"    Sample index: {sample_idx}")
                    print(f"    True value: {sample_row['y_true']:.4f}")
                    print(f"    Predicted value: {sample_row['y_pred']:.4f}")
                    
                    return sample_info
        
        if fold_num is not None:
            print(f"❌ Sample not found in fold {fold_num}")
        else:
            print(f"❌ Sample not found in any fold")
        return None
    
    def load_actual_voxel_data(self, sample_info):
        """Load the sample's 28-channel separated-channel voxel data."""
        try:
            from load_grids_pickle import VoxelGridsLoader
            
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
                            return voxel_grids[i]
            
            return None
            
        except Exception as e:
            print(f"Error loading voxel data: {e}")
            return None
    
    def load_model_for_fold(self, results_path, fold_idx):
        """Load trained model for specific fold"""
        checkpoint_path = results_path.replace('.pkl', f'-fold_{fold_idx}.pth')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        print(f"\n--- Loading model from: {checkpoint_path}")
        
        try:
            # Always map to CPU first, then move to target device
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model = AttentionCNN(in_channels=28)
            
            # Handle potential state dict mismatches for dynamically created layers
            model_state_dict = checkpoint['model_state_dict']
            model_dict = model.state_dict()
            
            # Filter out unexpected keys and missing keys
            filtered_state_dict = {}
            unexpected_keys = []
            missing_keys = []
            
            for k, v in model_state_dict.items():
                if k in model_dict:
                    # Check if shapes match before loading
                    if model_dict[k].shape == v.shape:
                        filtered_state_dict[k] = v
                    else:
                        print(f"    Shape mismatch for {k}: checkpoint {v.shape} vs model {model_dict[k].shape}")
                        unexpected_keys.append(k)
                else:
                    unexpected_keys.append(k)
            
            for k in model_dict.keys():
                if k not in filtered_state_dict:
                    missing_keys.append(k)
            
            if unexpected_keys:
                print(f"    Ignoring unexpected keys: {unexpected_keys}")
            if missing_keys:
                print(f"    Missing keys (will use default initialization): {missing_keys}")
            
            # Load filtered state dict
            model.load_state_dict(filtered_state_dict, strict=False)
            model.to(self.device)
            model.eval()
            
            # Try to extract scaler info from checkpoint or results
            scaler_info = checkpoint.get('scaler_info', None)
            
            # If no scaler info in checkpoint, try to reconstruct from results
            if scaler_info is None:
                print(f"  ⚠️  No scaler info in checkpoint, attempting to reconstruct...")
                try:
                    results = self.load_results(results_path)
                    if fold_idx in results['model_storage']:
                        model_data = results['model_storage'][fold_idx]
                        if 'df_train' in model_data and 'df_test' in model_data:
                            # Reconstruct scaler from actual data
                            df_train = model_data['df_train']
                            df_test = model_data['df_test']
                            
                            if 'y_true' in df_train.columns and 'y_pred' in df_train.columns:
                                y_true_train = df_train['y_true'].values
                                y_pred_train = df_train['y_pred'].values
                                
                                # Estimate scaler parameters from the relationship between true and predicted values
                                from sklearn.preprocessing import StandardScaler
                                scaler = StandardScaler()
                                scaler.fit(y_true_train.reshape(-1, 1))
                                
                                scaler_info = {
                                    'mean': scaler.mean_[0],
                                    'std': scaler.scale_[0]
                                }
                                print(f"  ✓ Reconstructed scaler: mean={scaler_info['mean']:.4f}, std={scaler_info['std']:.4f}")
                            else:
                                print(f"  ❌ Cannot reconstruct scaler - missing y_true/y_pred columns")
                        else:
                            print(f"  ❌ Cannot reconstruct scaler - missing dataframes")
                    else:
                        print(f"  ❌ Cannot reconstruct scaler - fold {fold_idx} not in results")
                except Exception as e:
                    print(f"  ❌ Failed to reconstruct scaler: {e}")
            else:
                print(f"  ✓ Found scaler info: mean={scaler_info['mean']:.4f}, std={scaler_info['std']:.4f}")
            
            print(f"  ✓ Model loaded successfully")
            
            return model, scaler_info
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    
    def plot_channel_importance(self,
                                analysis_results,
                                show_plots=True,
                                save_plots=True,
                                ):
        """Plot channel importance separately for adsorbate and solvent features."""
        if analysis_results is None:
            print("No analysis results to plot")
            return None
            
        primary_importance = analysis_results['importance_map']
        sample_info = analysis_results['sample_info']
        
        print(f"\n--- Generating Separated-Channel Importance Plot")
        
        # Create filename with channel_importance prefix and hyphens instead of underscores for consistency
        filename_prefix = f"channel_importance-{self.model_prefix}-{sample_info['zeolite']}-{sample_info['environment']}-{sample_info['adsorbate']}-snap{sample_info['snapshot']}-vox{sample_info['voxel_id']}"
        
        # Generate channel importance plot directly
        print(f"    Computing channel importance with relative normalization...")
        
        # Calculate channel importance using different methods
        channel_importance_raw = np.mean(np.abs(primary_importance), axis=(1, 2, 3))
        
        # Alternative importance calculations for logging
        channel_importance_l2 = np.sqrt(np.mean(primary_importance**2, axis=(1, 2, 3)))
        channel_importance_max = np.max(np.abs(primary_importance), axis=(1, 2, 3))
        
        print(f"    Raw importance range: [{channel_importance_raw.min():.8f}, {channel_importance_raw.max():.8f}]")
        print(f"    L2 importance range: [{channel_importance_l2.min():.8f}, {channel_importance_l2.max():.8f}]")
        print(f"    Max importance range: [{channel_importance_max.min():.8f}, {channel_importance_max.max():.8f}]")
        
        # Choose importance method
        channel_importance = channel_importance_raw  # Default to mean absolute
        
        # Apply normalization
        channel_importance = channel_importance / channel_importance.max()
        
        # Split the attribution values into adsorbate and solvent channels.
        num_atomic_features = len(self.atomic_features)
        adsorbate_importance = channel_importance[:num_atomic_features]
        solvent_importance = channel_importance[num_atomic_features:]
        
        # Reverse the order (top to bottom)
        adsorbate_importance = adsorbate_importance[::-1]
        solvent_importance = solvent_importance[::-1]
        atomic_features_reversed = self.atomic_features[::-1]
        
        # Define feature groups (indices are now reversed)
        # Original indices for onehot features: [0,1,2,3,4,5,6,7,8] -> atom_type_C, atom_type_H, atom_type_O, is_hydrophobic, is_donor, is_acceptor, is_hbonded, is_hbonded_donor, is_hbonded_acceptor
        # Original indices for continuous features: [9,10,11,12,13] -> atom_mass, partial_charge, valence, LJ_epsilon, LJ_sigma
        # After reversal: onehot becomes [5,4,3,2,1,0] and continuous becomes [8,7,6] (in reversed array)
        onehot_indices_reversed = [13-i for i in [0,1,2,3,4,5,6,7,8]]  # [13,12,11,10,9,8,7,6,5]
        continuous_indices_reversed = [13-i for i in [9,10,11,12,13]]   # [4,3,2,1,0]
        
        # Create single subplot with centered y-axis
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        fig.patch.set_facecolor('white')
        
        # Set up y positions
        y_pos = np.arange(len(atomic_features_reversed))
        
        # Plot adsorbate (left side, negative values)
        bars_ads = ax.barh(y_pos, -adsorbate_importance, 
                          color=plt.cm.Reds(0.3 + 0.5 * (adsorbate_importance / adsorbate_importance.max()) if adsorbate_importance.max() > 0 else np.ones_like(adsorbate_importance) * 0.8),
                          alpha=0.8, label='Adsorbate')
        
        # Plot solvent (right side, positive values)
        bars_solv = ax.barh(y_pos, solvent_importance,
                           color=plt.cm.Blues(0.3 + 0.5 * (solvent_importance / solvent_importance.max()) if solvent_importance.max() > 0 else np.ones_like(solvent_importance) * 0.8),
                           alpha=0.8, label='Solvent')
        
        # Set y-axis in the middle with feature names
        ax.set_yticks(y_pos)
        # Map feature names to friendly descriptions
        friendly_names = [feature_map.get(feat, feat) for feat in atomic_features_reversed]
        ax.set_yticklabels(friendly_names, fontsize=self.font_size, ha='left', color='black')
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()
        ax.tick_params(axis='y', pad=20, labelcolor='black')
        
        # Set x-axis with custom range and labels
        # Since adsorbate importance is smaller (max ~0.5), use asymmetric range
        left_limit = -0.6  # For adsorbate (left side)
        right_limit = 1.1  # For solvent (right side)
        ax.set_xlim(left_limit, right_limit)
        
        # Set custom x-axis ticks and labels
        # Left side: 0.6, 0.2, 0 (showing positive values for adsorbate)
        # Right side: 0, 0.2, 0.4, 0.6, 0.8, 1.0 (for solvent)
        x_ticks = [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]
        x_labels = ['0.60', '0.40', '0.20', '0.00', '0.20', '0.40', '0.60', '0.80', '1.00']
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=self.font_size)
        ax.set_xlabel(f'Importance Score', fontsize=self.font_size, ha='center', color='black')
        
        # Add vertical line at center
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
        
        # Add background rectangles to group features with professional sci paper colors
        # Discrete features background (light green)
        onehot_y_min = min(onehot_indices_reversed) - 0.4
        onehot_y_max = max(onehot_indices_reversed) + 0.4
        onehot_rect = plt.Rectangle((left_limit, onehot_y_min), right_limit - left_limit, onehot_y_max - onehot_y_min,
                                   facecolor='#E8F5E8', alpha=0.6, zorder=0)  # Professional light green
        ax.add_patch(onehot_rect)
        
        # Continuous features background (light amber-beige)
        continuous_y_min = min(continuous_indices_reversed) - 0.4
        continuous_y_max = max(continuous_indices_reversed) + 0.4
        continuous_rect = plt.Rectangle((left_limit, continuous_y_min), right_limit - left_limit, continuous_y_max - continuous_y_min,
                                       facecolor='#FFF8E1', alpha=0.6, zorder=0)  # Professional light amber-beige
        ax.add_patch(continuous_rect)
        
        # Add dashed lines to separate feature groups
        group_boundary = (max(continuous_indices_reversed) + min(onehot_indices_reversed)) / 2
        ax.axhline(y=group_boundary, color='black', linestyle='--', linewidth=2, alpha=0.8)
        
        # Add group labels
        onehot_center = (onehot_y_min + onehot_y_max) / 2
        continuous_center = (continuous_y_min + continuous_y_max) / 2
        
        ax.text(left_limit*1.05, onehot_center, 'Discrete Features', fontsize=self.font_size, 
                ha='center', va='center', rotation=90, alpha=0.7, weight='bold', color='black')
        ax.text(left_limit*1.05, continuous_center, 'Continuous Features', fontsize=self.font_size, 
                ha='center', va='center', rotation=90, alpha=0.7, weight='bold', color='black')
        
        # Set background and clean up
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Add titles for each side
        ax.text(-0.65, len(atomic_features_reversed), 'Adsorbate Channel Importance', 
                fontsize=self.font_size, ha='left', va='bottom', weight='bold')
        ax.text(0.05, len(atomic_features_reversed), 'Solvent Channel Importance', 
                fontsize=self.font_size, ha='left', va='bottom', weight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Print summary statistics
        print(f"    Adsorbate channels summary:")
        print(f"    - Max importance: {adsorbate_importance.max():.6f}")
        print(f"    - Mean importance: {adsorbate_importance.mean():.6f}")
        print(f"    Solvent channels summary:")
        print(f"    - Max importance: {solvent_importance.max():.6f}")
        print(f"    - Mean importance: {solvent_importance.mean():.6f}")
        
        # Save plot
        if save_plots:
            save_path = os.path.join(self.output_dir, f"{filename_prefix}.png")
            fig.savefig(save_path, dpi=1000, bbox_inches='tight')
            print(f"  ✓ Enhanced channel importance plot saved: {save_path}")
        
        if show_plots:
            plt.show()
            
        return fig
    
    


## SIMPLIFIED MAIN FUNCTION
if __name__ == "__main__":
    
    # Model results file
    results_filename = "model.pkl" # Test MAE 0.083 # Best

    
    # Initialize analyzer and run analysis
    analyzer = ImportanceMapAnalyzer(results_filename=results_filename,
                                     font_size=18,
                                     )
    
    ## Define the simulation parameters
    zeolite_type = 'FAU'           # e.g. "FAU", "BEA" or "MFI"
    solvent_type = 'methanol_600_water_600'    # e.g. "water_pure", "methanol_120_water_1080", "methanol_240_water_960", "methanol_600_water_600"
    pore_type = 'hydrophilic'      # e.g. "hydrophilic", "hydrophobic"
    adsorbate = '11_01_propylene_glycol'      # e.g. "01_methanol", "02_01_02_propanol"
    snapshot = 4
    voxel_id = 1
    
    # Analyze the specified sample using ensemble of all train folds
    analysis_results = analyzer.analyze_sample(
                                               zeolite=zeolite_type,
                                               env_adsorbate=solvent_type + '-' + pore_type + '-' + adsorbate,
                                               snapshot=snapshot,
                                               voxel_id=voxel_id,
                                               steps=24,
                                               ensemble_method='mean'  # Average across train folds
                                               )
    
    
    # Generate enhanced channel importance plot (Separated adsorbate/solvent channels)
    print(f"\n--- Generating Enhanced Channel Importance Plot ---")
    analyzer.plot_channel_importance(analysis_results, show_plots=False, save_plots=True)
    