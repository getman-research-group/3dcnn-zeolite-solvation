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
    
    def plot_3d_spatial_importance_input_voxel_layer(self, analysis_results, show_plots=True, save_plots=True):
        """Generate 3D spatial importance plot with separated adsorbate/solvent/combined views"""
        if analysis_results is None:
            print("No analysis results to plot")
            return None
            
        # Use ensemble results if available, otherwise fall back to single results
        if 'fold_importance_maps' in analysis_results and len(analysis_results['fold_importance_maps']) > 1:
            print(f"    Using ensemble results from {analysis_results['num_folds']} folds")
            primary_importance = analysis_results['importance_map']  # This is already the ensemble average
            fold_consistency = analysis_results.get('fold_consistency', None)
        else:
            print(f"    Using single fold results")
            primary_importance = analysis_results['importance_map']
            fold_consistency = None
            
        sample_info = analysis_results['sample_info']
        
        print(f"\n=== Generating 3D Input Voxel Layer Spatial Importance Plot ===")
        
        # Add ensemble info to title if applicable
        if fold_consistency is not None:
            print(f"    Ensemble consistency (std): mean={np.mean(fold_consistency):.6f}")
        
        # Split importance into adsorbate and solvent channels
        num_atomic_features = len(self.atomic_features)
        adsorbate_importance = primary_importance[:num_atomic_features]  # Channels 0-13
        solvent_importance = primary_importance[num_atomic_features:]    # Channels 14-27
        
        # Generate spatial importance for each group
        print("    Computing spatial importance for adsorbate, solvent, and combined groups...")
        adsorbate_spatial = np.mean(np.abs(adsorbate_importance), axis=0)  # Average over adsorbate channels
        solvent_spatial = np.mean(np.abs(solvent_importance), axis=0)      # Average over solvent channels
        combined_spatial = np.mean(np.abs(primary_importance), axis=0)     # Average over all channels
        
        # Define spatial data and color schemes (only adsorbate and solvent)
        spatial_data = {
            'Adsorbate Channels': adsorbate_spatial,
            'Solvent Channels': solvent_spatial
        }
        
        color_schemes = {
            'Adsorbate Channels': {'cmap': 'Reds', 'name': 'Adsorbate'},
            'Solvent Channels': {'cmap': 'Blues', 'name': 'Solvent'}
        }
        
        # Create subplots (2 subplots side by side)
        fig = plt.figure(figsize=(20, 12))
        axes = [fig.add_subplot(121, projection='3d'), 
                fig.add_subplot(122, projection='3d')]
        
        # Adjust spacing between subplots - make them closer
        plt.subplots_adjust(wspace=0)  # Reduce horizontal spacing between subplots
        
        fig.patch.set_facecolor('white')
        
        # Print summary statistics
        print(f"    Adsorbate spatial: max={adsorbate_spatial.max():.6f}, mean={adsorbate_spatial.mean():.6f}")
        print(f"    Solvent spatial: max={solvent_spatial.max():.6f}, mean={solvent_spatial.mean():.6f}")
        print(f"    Combined spatial: max={combined_spatial.max():.6f}, mean={combined_spatial.mean():.6f}")
        
        for i, (group_name, spatial_importance) in enumerate(spatial_data.items()):
            color_scheme = color_schemes[group_name]
            ax = axes[i]
            
            print(f"\n    Processing {group_name}...")
            
            # Use percentile-based normalization to handle outliers
            p_min, p_max = np.percentile(spatial_importance, [1, 99])
            
            # Adjust display threshold based on channel type for better visibility
            if 'Adsorbate' in group_name:
                # Use higher threshold for adsorbate to show only most important voxels
                display_mask = spatial_importance > np.percentile(spatial_importance, 95)  # Top 5%
                print(f"      Using higher threshold (95th percentile) for adsorbate visibility")
            else:
                display_mask = spatial_importance > np.percentile(spatial_importance, 90)  # Top 10%
                
            print(f"      Using percentile normalization: [{p_min:.8f}, {p_max:.8f}]")
            
            if not np.any(display_mask):
                print(f"      No voxels above cutoff for {group_name}, skipping...")
                ax.text(0.5, 0.5, 0.5, f'No significant\nvoxels for\n{group_name}', 
                       ha='center', va='center', fontsize=self.font_size, transform=ax.transAxes)
                continue
            
            # Set white background for this subplot
            ax.xaxis.pane.fill = True
            ax.yaxis.pane.fill = True
            ax.zaxis.pane.fill = True
            ax.xaxis.pane.set_facecolor('white')
            ax.yaxis.pane.set_facecolor('white')
            ax.zaxis.pane.set_facecolor('white')
            ax.xaxis.pane.set_alpha(1.0)
            ax.yaxis.pane.set_alpha(1.0)
            ax.zaxis.pane.set_alpha(1.0)
            
            # Create coordinate grids for voxel plotting
            x, y, z = np.indices(np.array(spatial_importance.shape) + 1)
            
            # Create colormap instance
            try:
                colormap = plt.colormaps[color_scheme['cmap']]
            except (AttributeError, KeyError):
                try:
                    colormap = plt.colormaps.get_cmap(color_scheme['cmap'])
                except AttributeError:
                    colormap = cm.get_cmap(color_scheme['cmap'])
            
            # Normalize importance values to 0-1 range
            norm_min, norm_max = p_min, p_max
            
            if norm_max > norm_min:
                normalized_importance = (spatial_importance - norm_min) / (norm_max - norm_min)
                normalized_importance = np.clip(normalized_importance, 0, 1)
            else:
                normalized_importance = np.zeros_like(spatial_importance)
            
            display_values = normalized_importance[display_mask]
            
            print(f"      Original range: [{norm_min:.8f}, {norm_max:.8f}]")
            print(f"      Normalized range: [0.0, 1.0]")
            print(f"      Number of displayed voxels: {len(display_values)}")
            print(f"      Displayed normalized values range: [{display_values.min():.3f}, {display_values.max():.3f}]")
            
            # Add detailed statistics for displayed values
            print(f"      Displayed values statistics:")
            print(f"        - Min: {display_values.min():.6f}")
            print(f"        - Max: {display_values.max():.6f}")
            print(f"        - Mean: {display_values.mean():.6f}")
            print(f"        - Median: {np.median(display_values):.6f}")
            print(f"        - 90th percentile: {np.percentile(display_values, 90):.6f}")
            print(f"        - 95th percentile: {np.percentile(display_values, 95):.6f}")
            print(f"        - 99th percentile: {np.percentile(display_values, 99):.6f}")
            
            if norm_max > norm_min:
                # Apply color power transformation to enhance high-value saturation
                color_power = 0.7
                
                # Different alpha settings for different channel types
                if 'Adsorbate' in group_name:
                    # Much more opaque for adsorbate to improve visibility
                    alpha_min, alpha_max, alpha_power = 0.35, 0.99, 0.8  # Higher alpha_min, lower power for more uniform opacity
                else:
                    # Standard settings for other channels
                    alpha_min, alpha_max, alpha_power = 0.05, 0.95, 1.5
                
                color_normalized = normalized_importance ** color_power
                
                # Create alpha values based on normalized importance with power scaling
                alpha_values = np.zeros_like(spatial_importance)
                display_normalized = normalized_importance[display_mask]
                display_alpha = alpha_min + (alpha_max - alpha_min) * (display_normalized ** alpha_power)
                alpha_values[display_mask] = display_alpha
                
                # Print detailed color and alpha statistics
                color_display_values = color_normalized[display_mask]
                print(f"      Color-normalized values (power={color_power}):")
                print(f"        - Min: {color_display_values.min():.6f}")
                print(f"        - Max: {color_display_values.max():.6f}")
                print(f"        - Mean: {color_display_values.mean():.6f}")
                print(f"      Alpha values (range=[{alpha_min}, {alpha_max}], power={alpha_power}):")
                print(f"        - Min: {display_alpha.min():.6f}")
                print(f"        - Max: {display_alpha.max():.6f}")
                print(f"        - Mean: {display_alpha.mean():.6f}")
                
                # Create RGBA colors for each voxel with enhanced saturation
                colors = np.zeros(spatial_importance.shape + (4,))
                
                coords = np.where(display_mask)
                for i_coord, j_coord, k_coord in zip(*coords):
                    color_value = color_normalized[i_coord, j_coord, k_coord]
                    rgba_color = colormap(color_value)
                    
                    # Enhanced saturation for high-importance voxels with stronger effect for adsorbate
                    if 'Adsorbate' in group_name:
                        # More aggressive color enhancement for adsorbate
                        if color_value > 0.5:  # Lower threshold for enhancement
                            r, g, b = rgba_color[:3]
                            min_component = min(r, g, b)
                            enhancement_factor = 0.5  # Stronger enhancement
                            
                            r = min(1.0, r + enhancement_factor * (1 - min_component))
                            rgba_color = (r, g, b, rgba_color[3])
                    else:
                        # Standard enhancement for other channels
                        if color_value > 0.7:
                            r, g, b = rgba_color[:3]
                            min_component = min(r, g, b)
                            enhancement_factor = 0.3
                            
                            if color_scheme['cmap'] == 'Blues':
                                b = min(1.0, b + enhancement_factor * (1 - min_component))
                                rgba_color = (r, g, b, rgba_color[3])
                    
                    alpha = alpha_values[i_coord, j_coord, k_coord]
                    colors[i_coord, j_coord, k_coord] = [rgba_color[0], rgba_color[1], rgba_color[2], alpha]
                
                # Plot voxels with individual colors
                ax.voxels(x, y, z, display_mask, facecolors=colors, alpha=None, edgecolor=None)
                
                # Create colorbar
                sm = cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1))
                sm.set_array([])
                # shrink: length; aspect: thickness; pad: distance from plot
                cbar = plt.colorbar(sm, ax=ax, shrink=0.65, aspect=30, pad=0.02)
                
                # Set colorbar ticks to show 0-1 normalized values
                cbar_ticks = np.linspace(0, 1, 6)  # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                cbar_labels = [f"{tick:.1f}" for tick in cbar_ticks]  # Simple 0-1 labels
                cbar.set_ticks(cbar_ticks)
                cbar.set_ticklabels(cbar_labels)
                cbar.set_label(f'{color_scheme["name"]} Importance', fontsize=self.font_size, color='black')
                
                # Set colorbar tick label font size
                cbar.ax.tick_params(labelsize=self.font_size)
            
            # Set axis tick label font size
            ax.tick_params(axis='x', labelsize=self.font_size)
            ax.tick_params(axis='y', labelsize=self.font_size)
            ax.tick_params(axis='z', labelsize=self.font_size)
            
            # Set equal aspect ratio
            shape = np.array(spatial_importance.shape)
            aspect_ratio = shape / shape.max()
            ax.set_box_aspect(aspect_ratio)
            
            # Set axis limits
            ax.set_xlim([0, shape[2]])
            ax.set_ylim([0, shape[1]])
            ax.set_zlim([0, shape[0]])
            
            # Set integer ticks for all axes (0, 4, 8, 12, 16, 20)
            tick_positions = [0, 4, 8, 12, 16, 20]
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            ax.set_zticks(tick_positions)
            ax.set_xticklabels(tick_positions)
            ax.set_yticklabels(tick_positions)
            ax.set_zticklabels(tick_positions)
            
            # Set title for this subplot
            # ax.set_title(f'{group_name}\n3D Spatial Importance', fontsize=self.font_size, fontweight='bold')
            
            # Add subplot label (a) or (b)
            subplot_labels = ['(a)', '(b)']
            ax.text2D(0.00, 1.00, subplot_labels[i], fontsize=self.font_size+2, fontweight='bold',
                     transform=ax.transAxes, verticalalignment='top', horizontalalignment='left')
        
        # Add overall title - commented out to remove titles
        # fig.suptitle(f'3D Input Voxel Layer Spatial Importance\n'
        #             f'Sample: {sample_info["zeolite"]}-{sample_info["environment"]}-{sample_info["adsorbate"]} '
        #             f'(Snapshot {sample_info["snapshot"]}, Voxel {sample_info["voxel_id"]})',
        #             fontsize=self.font_size, fontweight='bold')
        
        plt.tight_layout()
        
        print(f"\n✓ 3D input voxel layer spatial importance plot generated with 3 subplots")
        
        # Save plot
        if save_plots:
            # Create filename with 3d_input_voxel prefix
            filename_prefix = f"3d_input_voxel-{self.model_prefix}-{sample_info['zeolite']}-{sample_info['environment']}-{sample_info['adsorbate']}-snap{sample_info['snapshot']}-vox{sample_info['voxel_id']}"
            save_path = os.path.join(self.output_dir, f"{filename_prefix}.png")
            fig.savefig(save_path, dpi=1000, bbox_inches='tight')
            print(f"  ✓ 3D input voxel layer plot saved: {save_path}")
        
        if show_plots:
            plt.show()
        else:
            # Keep figure open for Spyder plots panel when show_plots=False
            pass
            
        return fig
    
    def plot_2d_spatial_importance_input_voxel_layer(self, analysis_results, show_plots=True, save_plots=True):
        """Generate 2D spatial importance visualization from input voxel layer gradients (backpropagated to original 28-channel input)"""
        if analysis_results is None:
            print("No analysis results to plot")
            return None
            
        # Use ensemble results if available, otherwise fall back to single results
        if 'fold_importance_maps' in analysis_results and len(analysis_results['fold_importance_maps']) > 1:
            print(f"    Using ensemble results from {analysis_results['num_folds']} folds")
            primary_importance = analysis_results['importance_map']  # This is already the ensemble average
            fold_consistency = analysis_results.get('fold_consistency', None)
        else:
            print(f"    Using single fold results")
            primary_importance = analysis_results['importance_map']
            fold_consistency = None
            
        sample_info = analysis_results['sample_info']
        
        print(f"\n=== Generating Averaged 2D Projections Plot ===")
        
        # Add ensemble info
        if fold_consistency is not None:
            print(f"    Ensemble consistency (std): mean={np.mean(fold_consistency):.6f}")
        
        # Create sample-specific filename prefix with model name
        sample_prefix = f"{self.model_prefix}-{sample_info['zeolite']}-{sample_info['environment']}-{sample_info['adsorbate']}-snap{sample_info['snapshot']}-vox{sample_info['voxel_id']}"
        
        # Split importance into adsorbate and solvent channels
        num_atomic_features = len(self.atomic_features)
        adsorbate_importance = primary_importance[:num_atomic_features]  # Channels 0-13
        solvent_importance = primary_importance[num_atomic_features:]    # Channels 14-27
        
        # Generate spatial importance for each branch
        print("    Computing spatial importance for adsorbate and solvent branches...")
        adsorbate_spatial = np.mean(np.abs(adsorbate_importance), axis=0)  # Average over adsorbate channels
        solvent_spatial = np.mean(np.abs(solvent_importance), axis=0)      # Average over solvent channels
        
        # Get spatial dimensions
        depth, height, width = adsorbate_spatial.shape
        print(f"    Grid shape: {adsorbate_spatial.shape}")
        
        # Generate averaged projections for adsorbate and solvent branches
        print("    Computing averaged projections instead of center slices...")
        projection_data = {}
        for name, spatial_data in [('adsorbate', adsorbate_spatial), 
                                   ('solvent', solvent_spatial)]:
            projections = {}
            
            # Z-projection (XY plane) - average over all Z layers
            projections['z'] = np.mean(spatial_data, axis=0)  # Average over depth (Z-axis)
            print(f"      {name} Z-projection: averaged over {depth} layers")
            
            # Y-projection (XZ plane) - average over all Y layers  
            projections['y'] = np.mean(spatial_data, axis=1)  # Average over height (Y-axis)
            print(f"      {name} Y-projection: averaged over {height} layers")
                
            # X-projection (YZ plane) - average over all X layers
            projections['x'] = np.mean(spatial_data, axis=2)  # Average over width (X-axis)
            print(f"      {name} X-projection: averaged over {width} layers")
            
            projection_data[name] = projections
        
        # Create figure with 2 rows (adsorbate, solvent) and 3 columns (X, Y, Z slices)
        figsize = (18, 12)  # Original size
        fig = plt.figure(figsize=figsize, facecolor='white')
        fig.patch.set_facecolor('white')
        
        # Use GridSpec to control subplot layout more precisely
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 0.05], hspace=0.3, wspace=0.3)
        
        # Define row information (removed combined row)
        row_info = [
            ('adsorbate', 'Adsorbate Branch', 'Reds'),     # Red colormap
            ('solvent', 'Solvent Branch', 'Blues')         # Blue colormap
        ]
        
        for row_idx, (branch_name, branch_label, cmap_name) in enumerate(row_info):
            # Get averaged projections for three views
            spatial_data_current = projection_data[branch_name]
            
            # Z-projection (XY plane)
            ax_z = fig.add_subplot(gs[row_idx, 0])
            im_z = ax_z.imshow(spatial_data_current['z'], cmap=cmap_name, interpolation='nearest', origin='lower')
            ax_z.set_title(f'{branch_label}\nZ-projection (XY plane)', fontsize=self.font_size)
            ax_z.set_xlabel('X', fontsize=self.font_size)
            ax_z.set_ylabel('Y', fontsize=self.font_size)
            ax_z.set_aspect('equal')
            
            # Y-projection (XZ plane)
            ax_y = fig.add_subplot(gs[row_idx, 1])
            im_y = ax_y.imshow(spatial_data_current['y'], cmap=cmap_name, interpolation='nearest', origin='lower')
            ax_y.set_title(f'{branch_label}\nY-projection (XZ plane)', fontsize=self.font_size)
            ax_y.set_xlabel('X', fontsize=self.font_size)
            ax_y.set_ylabel('Z', fontsize=self.font_size)
            ax_y.set_aspect('equal')
            
            # X-projection (YZ plane)
            ax_x = fig.add_subplot(gs[row_idx, 2])
            im_x = ax_x.imshow(spatial_data_current['x'], cmap=cmap_name, interpolation='nearest', origin='lower')
            ax_x.set_title(f'{branch_label}\nX-projection (YZ plane)', fontsize=self.font_size)
            ax_x.set_xlabel('Y', fontsize=self.font_size)
            ax_x.set_ylabel('Z', fontsize=self.font_size)
            ax_x.set_aspect('equal')
            
            # Colorbar in separate column
            cax = fig.add_subplot(gs[row_idx, 3])
            cbar_x = plt.colorbar(im_x, cax=cax)
            cbar_x.set_label('Importance', fontsize=self.font_size)
            cbar_x.ax.tick_params(labelleft=False, labelright=False, labeltop=False, labelbottom=False)
        
        plt.tight_layout()
        
        # Print summary information
        print(f"    Created averaged projections:")
        print(f"    - Adsorbate branch: channels 0-{num_atomic_features-1}")
        print(f"    - Solvent branch: channels {num_atomic_features}-{2*num_atomic_features-1}")
        
        # Calculate and print branch comparison statistics
        ads_total = np.sum(adsorbate_spatial)
        solv_total = np.sum(solvent_spatial)
        print(f"    Branch importance comparison:")
        print(f"    - Adsorbate spatial total: {ads_total:.6f}")
        print(f"    - Solvent spatial total: {solv_total:.6f}")
        print(f"    - Solvent/Adsorbate ratio: {solv_total/ads_total:.3f}")
        
        # Save plot
        if save_plots:
            save_path = os.path.join(self.output_dir, f"2d_input_voxel-{sample_prefix}.png")
            fig.savefig(save_path, dpi=1000, bbox_inches='tight')
            print(f"  ✓ Averaged projections plot saved: {save_path}")
        
        if show_plots:
            plt.show()
            
        return fig

    
    def plot_gradient_distribution(self, analysis_results, show_plots=True, save_plots=True, num_top_features=5):
        """Generate diagnostic plots with ensemble support and distribution quality analysis"""
        if analysis_results is None:
            print("No analysis results to plot")
            return None
            
        # Use ensemble results if available, otherwise fall back to single results
        if 'fold_importance_maps' in analysis_results and len(analysis_results['fold_importance_maps']) > 1:
            print(f"    Using ensemble results from {analysis_results['num_folds']} folds")
            primary_importance = analysis_results['importance_map']  # This is already the ensemble average
            fold_consistency = analysis_results.get('fold_consistency', None)
        else:
            print(f"    Using single fold results")
            primary_importance = analysis_results['importance_map']
            fold_consistency = None
            
        sample_info = analysis_results['sample_info']
        
        print(f"\n=== Generating Diagnostic Plots ===")
        
        # Add ensemble info
        if fold_consistency is not None:
            print(f"    Ensemble consistency (std): mean={np.mean(fold_consistency):.6f}")
        
        # Create sample-specific filename prefix with model name
        sample_prefix = f"{self.model_prefix}-{sample_info['zeolite']}-{sample_info['environment']}-{sample_info['adsorbate']}-snap{sample_info['snapshot']}-vox{sample_info['voxel_id']}"
        
        # 🔍 DETAILED GRADIENT DISTRIBUTION ANALYSIS
        grad_magnitudes = np.abs(primary_importance).flatten()
        grad_magnitudes_nonzero = grad_magnitudes[grad_magnitudes > 0]
        
        print(f"\n📊 GRADIENT DISTRIBUTION QUALITY ANALYSIS:")
        print(f"    Total gradient values: {len(grad_magnitudes):,}")
        print(f"    Non-zero gradient values: {len(grad_magnitudes_nonzero):,}")
        print(f"    Zero gradient ratio: {(len(grad_magnitudes) - len(grad_magnitudes_nonzero))/len(grad_magnitudes):.4f}")
        
        # Basic statistics
        print(f"\n📈 BASIC STATISTICS:")
        print(f"    Range: [{grad_magnitudes.min():.2e}, {grad_magnitudes.max():.2e}]")
        print(f"    Mean: {grad_magnitudes.mean():.2e}")
        print(f"    Median: {np.median(grad_magnitudes):.2e}")
        print(f"    Std: {grad_magnitudes.std():.2e}")
        
        # 🎯 DISTRIBUTION QUALITY CHECKS
        print(f"\n🔬 DISTRIBUTION QUALITY ASSESSMENT:")
        
        # 1. Sparsity check (should have many near-zero values)
        zero_ratio = (len(grad_magnitudes) - len(grad_magnitudes_nonzero)) / len(grad_magnitudes)
        if len(grad_magnitudes_nonzero) > 0:
            # Among non-zero values, 90% should be relatively small
            percentile_90_nonzero = np.percentile(grad_magnitudes_nonzero, 90)
            small_nonzero_ratio = np.sum(grad_magnitudes_nonzero < percentile_90_nonzero) / len(grad_magnitudes_nonzero)
            effective_sparsity = zero_ratio + (1 - zero_ratio) * small_nonzero_ratio
        else:
            effective_sparsity = zero_ratio
        print(f"    1. Effective sparsity: {effective_sparsity:.3f} ({'✓ EXCELLENT' if effective_sparsity >= 0.9 else '✓ GOOD' if effective_sparsity >= 0.8 else '⚠ WARNING' if effective_sparsity >= 0.7 else '❌ POOR'})")
        print(f"       - Zero values: {zero_ratio:.3f}")
        if len(grad_magnitudes_nonzero) > 0:
            print(f"       - Small non-zero: {small_nonzero_ratio:.3f}")
        
        # 2. Dynamic range check (should be > 2 orders of magnitude)
        if len(grad_magnitudes_nonzero) > 0:
            dynamic_range = np.log10(grad_magnitudes.max()) - np.log10(np.median(grad_magnitudes_nonzero))
            print(f"    2. Dynamic range (log10): {dynamic_range:.2f} ({'✓ GOOD' if dynamic_range >= 2.0 else '⚠ WARNING' if dynamic_range >= 1.0 else '❌ POOR'})")
        else:
            print(f"    2. Dynamic range: Cannot compute (no non-zero values)")
            dynamic_range = 0
        
        # 3. Distribution skewness (should be highly right-skewed)
        from scipy import stats
        if grad_magnitudes.std() > 0:
            skewness = stats.skew(grad_magnitudes)
            print(f"    3. Skewness (right-tail): {skewness:.3f} ({'✓ GOOD' if skewness >= 2.0 else '⚠ WARNING' if skewness >= 1.0 else '❌ POOR'})")
        else:
            print(f"    3. Skewness: Cannot compute (zero variance)")
            skewness = 0
            
        # 4. Heavy tail check (top 1% should contain significant mass)
        percentile_99 = np.percentile(grad_magnitudes, 99)
        heavy_tail_ratio = np.sum(grad_magnitudes > percentile_99) / len(grad_magnitudes)
        tail_mass = np.sum(grad_magnitudes[grad_magnitudes > percentile_99]) / np.sum(grad_magnitudes) if np.sum(grad_magnitudes) > 0 else 0
        print(f"    4. Heavy tail mass (top 1%): {tail_mass:.3f} ({'✓ GOOD' if tail_mass >= 0.1 else '⚠ WARNING' if tail_mass >= 0.05 else '❌ POOR'})")
        
        # 5. Overall distribution quality score
        quality_scores = []
        if effective_sparsity >= 0.9: quality_scores.append(1)
        elif effective_sparsity >= 0.8: quality_scores.append(0.75)
        elif effective_sparsity >= 0.7: quality_scores.append(0.5)
        else: quality_scores.append(0)
        
        if dynamic_range >= 2.0: quality_scores.append(1)
        elif dynamic_range >= 1.0: quality_scores.append(0.5)
        else: quality_scores.append(0)
        
        if skewness >= 2.0: quality_scores.append(1)
        elif skewness >= 1.0: quality_scores.append(0.5)
        else: quality_scores.append(0)
        
        if tail_mass >= 0.1: quality_scores.append(1)
        elif tail_mass >= 0.05: quality_scores.append(0.5)
        else: quality_scores.append(0)
        
        overall_quality = np.mean(quality_scores)
        print(f"\n🎯 OVERALL DISTRIBUTION QUALITY: {overall_quality:.2f}/1.0 ", end="")
        if overall_quality >= 0.8:
            print("(✓ EXCELLENT - Ideal long-tail distribution)")
        elif overall_quality >= 0.6:
            print("(✓ GOOD - Acceptable distribution)")
        elif overall_quality >= 0.4:
            print("(⚠ WARNING - Distribution may need attention)")
        else:
            print("(❌ POOR - Distribution indicates potential issues)")
        
        # Generate diagnostic plots
        print(f"\nCreating diagnostic plots with top {num_top_features} features...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # For ensemble results, we may not have input_tensor, so skip that plot
        if 'input_tensor' in analysis_results:
            # Input tensor analysis (only if available)
            input_data = analysis_results['input_tensor'].squeeze()  # Remove batch dim
            input_spatial = np.mean(np.abs(input_data), axis=0)  # Average across channels
            
            im1 = axes[0].imshow(input_spatial[input_spatial.shape[0]//2, :, :], cmap='viridis')
            axes[0].set_title('Input Data (Center Slice)', fontsize=self.font_size)
            axes[0].tick_params(axis='both', labelsize=self.font_size)
            cbar1 = plt.colorbar(im1, ax=axes[0])
            cbar1.ax.tick_params(labelsize=self.font_size)
            
            # Adjust subplot indices for remaining plots
            grad_hist_ax = axes[1]
            channel_hist_ax = None  # Skip channel plot if we only have 2 subplots
        else:
            # Skip input tensor plot for ensemble results
            print("    Input tensor not available (ensemble mode), skipping input visualization")
            grad_hist_ax = axes[0]
            channel_hist_ax = axes[1]
        
        # Gradient magnitude distribution
        grad_hist_ax.hist(grad_magnitudes, bins=50, alpha=0.7, color='steelblue')
        grad_hist_ax.set_title('Gradient Magnitude Distribution', fontsize=self.font_size)
        grad_hist_ax.set_xlabel('|Gradient|', fontsize=self.font_size)
        grad_hist_ax.set_ylabel('Count', fontsize=self.font_size)
        grad_hist_ax.set_yscale('log')
        grad_hist_ax.tick_params(axis='both', labelsize=self.font_size)
        
        # Add subplot label (a) - outside the plot area
        grad_hist_ax.text(-0.15, 1.05, '(a)', fontsize=self.font_size+2, fontweight='bold',
                         transform=grad_hist_ax.transAxes, verticalalignment='bottom', horizontalalignment='center')
        
        # Channel-wise gradient distribution (only if we have separate subplot)
        if channel_hist_ax is not None:
            # Get top N features based on channel importance
            channel_importance = np.mean(np.abs(primary_importance), axis=(1, 2, 3))
            top_indices = np.argsort(channel_importance)[-num_top_features:][::-1]  # Top N in descending order
            
            print(f"    Top {num_top_features} channels by importance:")
            colors = plt.cm.Set1(np.linspace(0, 1, num_top_features))
            
            for i, idx in enumerate(top_indices):
                channel_name = self.feature_names[idx]
                channel_grads = np.abs(primary_importance[idx]).flatten()
                importance_score = channel_importance[idx]
                print(f"      {i+1}. Channel {idx}: {channel_name} (importance: {importance_score:.2e})")
                
                channel_hist_ax.hist(channel_grads, bins=30, alpha=0.6, 
                                   label=f"{channel_name}", color=colors[i])
            
            channel_hist_ax.set_title(f'Top {num_top_features} Channels - Gradient Distribution', fontsize=self.font_size)
            channel_hist_ax.set_xlabel('|Gradient|', fontsize=self.font_size)
            channel_hist_ax.set_ylabel('Count', fontsize=self.font_size)
            legend = channel_hist_ax.legend(fontsize=self.font_size)
            channel_hist_ax.set_yscale('log')
            channel_hist_ax.tick_params(axis='both', labelsize=self.font_size)
            
            # Add subplot label (b) - outside the plot area
            channel_hist_ax.text(-0.15, 1.05, '(b)', fontsize=self.font_size+2, fontweight='bold',
                                 transform=channel_hist_ax.transAxes, verticalalignment='bottom', horizontalalignment='center')
        
        # Remove main title (as requested)
        # if fold_consistency is not None:
        #     fig.suptitle(f'Diagnostic Plots - Ensemble from {analysis_results["num_folds"]} folds', fontsize=self.font_size)
        
        plt.tight_layout()
        
        # Save plot
        if save_plots:
            save_path = os.path.join(self.output_dir, f"gradient_distribution-{sample_prefix}.png")
            fig.savefig(save_path, dpi=1000, bbox_inches='tight')
            print(f"  ✓ Gradient distribution plots saved: {save_path}")
        
        if show_plots:
            plt.show()
            
        return fig


    def plot_2d_spatial_importance_processor_output_layer(self, analysis_results, show_plots=True, save_plots=True):
        """Generate 2D spatial importance visualization from processor branch output layers (adsorbate/solvent/interaction activations)"""
        if analysis_results is None:
            print("No analysis results provided")
            return None
            
        # Check if multi-level data is available
        multilevel_data = analysis_results.get('multilevel_data')
        if multilevel_data is None:
            print("Multi-level data not available, model may not support branch analysis")
            return None
            
        sample_info = analysis_results['sample_info']
        sample_prefix = f"{self.model_prefix}-{sample_info['zeolite']}-{sample_info['environment']}-{sample_info['adsorbate']}-snap{sample_info['snapshot']}-vox{sample_info['voxel_id']}"
        
        print(f"\n=== Generating Processor Output Layer 2D Slices Plot ===")
        print(f"Available multi-level data: {list(multilevel_data.keys())}")
        
        # Process available spatial importance data
        spatial_data = {}
        available_levels = []
        
        if 'adsorbate_spatial' in multilevel_data:
            adsorbate_spatial = multilevel_data['adsorbate_spatial']
            if len(adsorbate_spatial.shape) == 3:  # Ensure it's 3D (D, H, W)
                spatial_data['Adsorbate Branch'] = adsorbate_spatial
                available_levels.append('Adsorbate Branch')
            else:
                print(f"      Warning: adsorbate_spatial has unexpected shape {adsorbate_spatial.shape}")
            
        if 'solvent_spatial' in multilevel_data:
            solvent_spatial = multilevel_data['solvent_spatial']
            if len(solvent_spatial.shape) == 3:  # Ensure it's 3D (D, H, W)
                spatial_data['Solvent Branch'] = solvent_spatial
                available_levels.append('Solvent Branch')
            else:
                print(f"      Warning: solvent_spatial has unexpected shape {solvent_spatial.shape}")
            
        if 'interaction_spatial' in multilevel_data:
            interaction_spatial = multilevel_data['interaction_spatial']
            if len(interaction_spatial.shape) == 3:  # Ensure it's 3D (D, H, W)
                spatial_data['Interaction Layer'] = interaction_spatial
                available_levels.append('Interaction Layer')
            else:
                print(f"      Warning: interaction_spatial has unexpected shape {interaction_spatial.shape}")
            
        if len(available_levels) == 0:
            print("No spatial importance data found in multi-level results")
            return None
        
        # Define color schemes for different branches (原始着色方案)
        color_schemes = {
            'Adsorbate Branch': {'cmap': 'Reds', 'name': 'Adsorbate'},
            'Solvent Branch': {'cmap': 'Blues', 'name': 'Solvent'}, 
            'Interaction Layer': {'cmap': 'Greens', 'name': 'Interaction'}
        }
        
        # Create figure with 3 rows (branches) x 3 columns (views)
        n_levels = len(available_levels)
        fig = plt.figure(figsize=(18, 6*n_levels))
        
        # Use GridSpec to control subplot layout more precisely
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(n_levels, 4, figure=fig, width_ratios=[1, 1, 1, 0.05], hspace=0.3, wspace=0.3)
        
        for i, level_name in enumerate(available_levels):
            spatial_importance = spatial_data[level_name]
            color_scheme = color_schemes.get(level_name, color_schemes['Adsorbate Branch'])
            
            # Get center slices for three views
            depth, height, width = spatial_importance.shape
            
            # Z-slice (XY plane) - center slice
            if depth % 2 == 1:
                center_z = depth // 2
                z_slice = spatial_importance[center_z, :, :]
            else:
                center_z1, center_z2 = depth // 2 - 1, depth // 2
                z_slice = (spatial_importance[center_z1, :, :] + spatial_importance[center_z2, :, :]) / 2
                center_z = f"{center_z1}-{center_z2}"
            
            # Y-slice (XZ plane) - center slice
            if height % 2 == 1:
                center_y = height // 2
                y_slice = spatial_importance[:, center_y, :]
            else:
                center_y1, center_y2 = height // 2 - 1, height // 2
                y_slice = (spatial_importance[:, center_y1, :] + spatial_importance[:, center_y2, :]) / 2
                center_y = f"{center_y1}-{center_y2}"
            
            # X-slice (YZ plane) - center slice
            if width % 2 == 1:
                center_x = width // 2
                x_slice = spatial_importance[:, :, center_x]
            else:
                center_x1, center_x2 = width // 2 - 1, width // 2
                x_slice = (spatial_importance[:, :, center_x1] + spatial_importance[:, :, center_x2]) / 2
                center_x = f"{center_x1}-{center_x2}"
            
            # Plot Z-slice (XY plane)
            ax_z = fig.add_subplot(gs[i, 0])
            im_z = ax_z.imshow(z_slice, cmap=color_scheme['cmap'], interpolation='nearest', origin='lower')
            ax_z.set_title(f'{level_name}\nZ-slice (XY plane, Z={center_z})', fontsize=self.font_size)
            ax_z.set_xlabel('X', fontsize=self.font_size)
            ax_z.set_ylabel('Y', fontsize=self.font_size)
            ax_z.set_aspect('equal')
            
            # Plot Y-slice (XZ plane)
            ax_y = fig.add_subplot(gs[i, 1])
            im_y = ax_y.imshow(y_slice, cmap=color_scheme['cmap'], interpolation='nearest', origin='lower')
            ax_y.set_title(f'{level_name}\nY-slice (XZ plane, Y={center_y})', fontsize=self.font_size)
            ax_y.set_xlabel('X', fontsize=self.font_size)
            ax_y.set_ylabel('Z', fontsize=self.font_size)
            ax_y.set_aspect('equal')
            
            # Plot X-slice (YZ plane)
            ax_x = fig.add_subplot(gs[i, 2])
            im_x = ax_x.imshow(x_slice, cmap=color_scheme['cmap'], interpolation='nearest', origin='lower')
            ax_x.set_title(f'{level_name}\nX-slice (YZ plane, X={center_x})', fontsize=self.font_size)
            ax_x.set_xlabel('Y', fontsize=self.font_size)
            ax_x.set_ylabel('Z', fontsize=self.font_size)
            ax_x.set_aspect('equal')
            
            # Colorbar in separate column
            cax = fig.add_subplot(gs[i, 3])
            cbar_x = plt.colorbar(im_x, cax=cax)
            cbar_x.set_label('Importance', fontsize=self.font_size)
            cbar_x.ax.tick_params(labelleft=False, labelright=False, labeltop=False, labelbottom=False)
        
        plt.tight_layout()
        
        # Print summary
        print(f"✓ Processor output layer 2D slices plot generated")
        for level_name in available_levels:
            spatial_importance = spatial_data[level_name]
            print(f"    {level_name}: max={spatial_importance.max():.6f}, mean={spatial_importance.mean():.6f}")
        
        # Save plot
        if save_plots:
            save_path = os.path.join(self.output_dir, f"2d_processor_output_layer-{sample_prefix}.png")
            fig.savefig(save_path, dpi=1000, bbox_inches='tight', facecolor='white')
            print(f"  ✓ 2D processor output layer plot saved: {save_path}")
        
        if show_plots:
            plt.show()
            
        return fig

    def plot_multilevel_comparison(self, analysis_results, show_plots=True, save_plots=True):
        """
        Generate comparison plot showing relationship between input channels and processed features
        """
        if analysis_results is None:
            print("No analysis results provided")
            return None
            
        multilevel_data = analysis_results.get('multilevel_data')
        if multilevel_data is None:
            print("Multi-level data not available")
            return None
            
        primary_importance = analysis_results['importance_map']
        sample_info = analysis_results['sample_info']
        sample_prefix = f"{self.model_prefix}-{sample_info['zeolite']}-{sample_info['environment']}-{sample_info['adsorbate']}-snap{sample_info['snapshot']}-vox{sample_info['voxel_id']}"
        
        print(f"\n=== Generating Multi-Level Comparison Plot ===")
        print(f"🔍 DETAILED NUMERICAL ANALYSIS:")
        
        # Calculate channel importance from input
        print(f"\n--- INPUT LAYER ANALYSIS ---")
        print(f"Primary importance shape: {primary_importance.shape}")
        print(f"Primary importance raw range: [{primary_importance.min():.10f}, {primary_importance.max():.10f}]")
        
        channel_importance = np.mean(np.abs(primary_importance), axis=(1, 2, 3))
        print(f"Channel importance (28 channels): {channel_importance}")
        print(f"Channel importance range: [{channel_importance.min():.10f}, {channel_importance.max():.10f}]")
        
        adsorbate_channels = channel_importance[:14]  # First 14 channels
        solvent_channels = channel_importance[14:]   # Last 14 channels
        
        print(f"Adsorbate channels (0-13): {adsorbate_channels}")
        print(f"Adsorbate range: [{adsorbate_channels.min():.10f}, {adsorbate_channels.max():.10f}]")
        print(f"Adsorbate mean: {np.mean(adsorbate_channels):.10f}")
        
        print(f"Solvent channels (14-27): {solvent_channels}")
        print(f"Solvent range: [{solvent_channels.min():.10f}, {solvent_channels.max():.10f}]")
        print(f"Solvent mean: {np.mean(solvent_channels):.10f}")
        
        # Calculate spatial importance from processed features
        print(f"\n--- PROCESSOR LAYER ANALYSIS ---")
        spatial_summaries = {}
        if 'adsorbate_spatial' in multilevel_data:
            adsorbate_spatial_data = multilevel_data['adsorbate_spatial']
            print(f"Adsorbate spatial data shape: {adsorbate_spatial_data.shape}")
            print(f"Adsorbate spatial raw range: [{adsorbate_spatial_data.min():.10f}, {adsorbate_spatial_data.max():.10f}]")
            adsorbate_spatial = np.mean(np.abs(adsorbate_spatial_data))
            print(f"Adsorbate spatial mean: {adsorbate_spatial:.10f}")
            spatial_summaries['Adsorbate Branch'] = adsorbate_spatial
            
        if 'solvent_spatial' in multilevel_data:
            solvent_spatial_data = multilevel_data['solvent_spatial']
            print(f"Solvent spatial data shape: {solvent_spatial_data.shape}")
            print(f"Solvent spatial raw range: [{solvent_spatial_data.min():.10f}, {solvent_spatial_data.max():.10f}]")
            solvent_spatial = np.mean(np.abs(solvent_spatial_data))
            print(f"Solvent spatial mean: {solvent_spatial:.10f}")
            spatial_summaries['Solvent Branch'] = solvent_spatial
            
        if 'interaction_spatial' in multilevel_data:
            interaction_spatial_data = multilevel_data['interaction_spatial']
            print(f"Interaction spatial data shape: {interaction_spatial_data.shape}")
            print(f"Interaction spatial raw range: [{interaction_spatial_data.min():.10f}, {interaction_spatial_data.max():.10f}]")
            interaction_spatial = np.mean(np.abs(interaction_spatial_data))
            print(f"Interaction spatial mean: {interaction_spatial:.10f}")
            spatial_summaries['Interaction Layer'] = interaction_spatial
        
        # Create comparison plot with custom width ratios (left narrower, right wider)
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(16, 8))
        gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 2])  # Left:Right = 1:2 ratio
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Left subplot (a): Spatial importance summary
        if len(spatial_summaries) > 0:
            levels = list(spatial_summaries.keys())
            values = list(spatial_summaries.values())
            colors = ['red', 'blue', 'green'][:len(levels)]
            
            # Create two-line labels for better readability
            two_line_levels = []
            for level in levels:
                if 'Branch' in level:
                    parts = level.split(' Branch')
                    two_line_levels.append(f"{parts[0]}\nBranch")
                elif 'Layer' in level:
                    parts = level.split(' Layer')
                    two_line_levels.append(f"{parts[0]}\nLayer")
                else:
                    two_line_levels.append(level)
            
            bars = ax1.bar(range(len(levels)), values, color=colors, alpha=0.7, width=0.5)
            ax1.set_title('Integrated Gradients', fontsize=self.font_size, fontweight='bold')
            ax1.set_ylabel('Mean Spatial Importance', fontsize=self.font_size)
            ax1.set_xticks(range(len(levels)))
            ax1.set_xticklabels(two_line_levels, fontsize=self.font_size, ha='center')
            ax1.tick_params(axis='y', labelsize=self.font_size)  # Set y-axis tick font size
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.6f}', ha='center', va='bottom', fontsize=self.font_size)
        else:
            ax1.text(0.5, 0.5, 'No spatial data\navailable', ha='center', va='center',
                    transform=ax1.transAxes, fontsize=self.font_size)
            ax1.set_title('Integrated Gradients', fontsize=self.font_size, fontweight='bold')
        
        # Add subplot label (a) - outside the plot area
        ax1.text(-0.15, 1.05, '(a)', fontsize=self.font_size+2, fontweight='bold',
                transform=ax1.transAxes, verticalalignment='bottom', horizontalalignment='center')
        
        # Right subplot (b): Input vs processed comparison
        print(f"\n--- INPUT VS PROCESSED COMPARISON ---")
        input_summary = {
            'Adsorbate Voxels': np.mean(adsorbate_channels),
            'Solvent Voxels': np.mean(solvent_channels),
            'Combined Voxels': np.mean(channel_importance)
        }
        
        print(f"📊 INPUT IMPORTANCE VALUES:")
        for key, value in input_summary.items():
            print(f"  {key}: {value:.10f}")
        
        comparison_data = {**input_summary, **spatial_summaries}
        labels = list(comparison_data.keys())
        values = list(comparison_data.values())
        
        # Create two-line labels for better readability
        two_line_labels = []
        for label in labels:
            if 'Voxel' in label:
                parts = label.split(' Voxel')
                two_line_labels.append(f"{parts[0]}\nVoxel")
            elif 'Branch' in label:
                parts = label.split(' Branch')
                two_line_labels.append(f"{parts[0]}\nBranch")
            elif 'Layer' in label:
                parts = label.split(' Layer')
                two_line_labels.append(f"{parts[0]}\nLayer")
            else:
                two_line_labels.append(label)
        
        print(f"📊 ALL COMPARISON VALUES:")
        for i, (label, value) in enumerate(zip(labels, values)):
            print(f"  [{i}] {label}: {value:.10f}")
        
        print(f"📊 RATIO ANALYSIS:")
        if len(values) >= 6:  # Ensure we have both input and processed values
            input_values = values[:3]  # First 3 are input
            processed_values = values[3:]  # Rest are processed
            print(f"  Input values: {input_values}")
            print(f"  Processed values: {processed_values}")
            print(f"  Input max: {max(input_values):.10f}")
            print(f"  Processed max: {max(processed_values):.10f}")
            print(f"  Processed/Input ratio: {max(processed_values)/max(input_values) if max(input_values) > 0 else 'inf'}")
        
        # 🔧 SOLUTION: Use log scale for better visualization of different magnitude values
        # Color code: input (light) vs processed (dark)
        colors = ['lightcoral', 'lightblue', 'lightgray'] + ['darkred', 'darkblue', 'darkgreen'][:len(spatial_summaries)]
        
        bars = ax2.bar(range(len(labels)), values, color=colors[:len(labels)], alpha=0.8, width=0.5)
        ax2.set_title('Integrated Gradients Comparison', fontsize=self.font_size, fontweight='bold')
        ax2.set_ylabel('Mean Importance (Log Scale)', fontsize=self.font_size)
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(two_line_labels, fontsize=self.font_size, ha='center')
        ax2.tick_params(axis='y', labelsize=self.font_size)  # Set y-axis tick font size
        
        # Use log scale to handle large differences in magnitude
        ax2.set_yscale('log')
        
        # Add background regions and vertical divider
        if len(values) >= 6:  # Ensure we have both input and processed values
            # Extend y-axis upper limit to make room for text labels
            original_ylim = ax2.get_ylim()
            new_ylim_upper = original_ylim[1] * 2.0  # Double the upper limit for more space
            ax2.set_ylim(original_ylim[0], new_ylim_upper)
            ylim = ax2.get_ylim()
            
            # Add vertical divider line at position 2.5 (between index 2 and 3)
            ax2.axvline(x=2.5, color='black', linestyle='--', linewidth=2, alpha=0.8)
            
            # Add region labels at a safe position from the top
            label_y_position = ylim[1] * 0.7  # Position labels at 75% of max height
            ax2.text(1, label_y_position, 'Voxel Inputs', fontsize=self.font_size, 
                    ha='center', va='center', weight='bold', color='black')
            ax2.text(len(labels)-2, label_y_position, 'Processed Output', fontsize=self.font_size, 
                    ha='center', va='center', weight='bold', color='black')
        
        # Add value labels with scientific notation for small values
        for i, (label, value) in enumerate(zip(labels, values)):
            if value < 0.001:  # For very small values, use scientific notation
                ax2.text(i, value, f'{value:.2e}', ha='center', va='bottom', fontsize=self.font_size)
            else:
                ax2.text(i, value, f'{value:.4f}', ha='center', va='bottom', fontsize=self.font_size)
        
        # Add subplot label (b) - outside the plot area
        ax2.text(-0.085, 1.05, '(b)', fontsize=self.font_size+2, fontweight='bold',
                transform=ax2.transAxes, verticalalignment='bottom', horizontalalignment='center')
        
        plt.tight_layout()
        
        # Print summary
        print(f"✓ Multi-level comparison plot generated")
        print(f"    Input summary - Adsorbate: {np.mean(adsorbate_channels):.6f}, Solvent: {np.mean(solvent_channels):.6f}")
        for level, value in spatial_summaries.items():
            print(f"    Processed summary - {level}: {value:.6f}")
        
        # Save plot
        if save_plots:
            filename = f"multilevel_comparison-{sample_prefix}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=1000, bbox_inches='tight', facecolor='white')
            print(f"    ✓ Saved: {filepath}")
        
        if show_plots:
            plt.show()
        else:
            # Keep figure open for Spyder plots panel when show_plots=False
            pass
            
        return fig

    def plot_3d_spatial_importance_processor_output_layer(self, analysis_results, show_plots=True, save_plots=True):
        """Generate 3D spatial importance visualization from processor branch output layers (adsorbate/solvent/interaction activations)"""
        if analysis_results is None:
            print("No analysis results provided")
            return None
            
        # Check if multi-level data is available
        multilevel_data = analysis_results.get('multilevel_data')
        if multilevel_data is None:
            print("Multi-level data not available, model may not support branch analysis")
            return None
            
        sample_info = analysis_results['sample_info']
        sample_prefix = f"{self.model_prefix}-{sample_info['zeolite']}-{sample_info['environment']}-{sample_info['adsorbate']}-snap{sample_info['snapshot']}-vox{sample_info['voxel_id']}"
        
        print(f"\n=== Generating 3D Processor Output Layer Spatial Importance Plot ===")
        print(f"Available multi-level data: {list(multilevel_data.keys())}")
        
        # Process available spatial importance data
        spatial_data = {}
        available_levels = []
        
        if 'adsorbate_spatial' in multilevel_data:
            adsorbate_spatial = multilevel_data['adsorbate_spatial']
            if len(adsorbate_spatial.shape) == 3:  # Ensure it's 3D (D, H, W)
                spatial_data['Adsorbate Branch'] = adsorbate_spatial
                available_levels.append('Adsorbate Branch')
                print(f"    Adsorbate Branch: shape {adsorbate_spatial.shape}, range [{adsorbate_spatial.min():.6f}, {adsorbate_spatial.max():.6f}]")
            else:
                print(f"      Warning: adsorbate_spatial has unexpected shape {adsorbate_spatial.shape}")
            
        if 'solvent_spatial' in multilevel_data:
            solvent_spatial = multilevel_data['solvent_spatial']
            if len(solvent_spatial.shape) == 3:  # Ensure it's 3D (D, H, W)
                spatial_data['Solvent Branch'] = solvent_spatial
                available_levels.append('Solvent Branch')
                print(f"    Solvent Branch: shape {solvent_spatial.shape}, range [{solvent_spatial.min():.6f}, {solvent_spatial.max():.6f}]")
            else:
                print(f"      Warning: solvent_spatial has unexpected shape {solvent_spatial.shape}")
            
        if 'interaction_spatial' in multilevel_data:
            interaction_spatial = multilevel_data['interaction_spatial']
            if len(interaction_spatial.shape) == 3:  # Ensure it's 3D (D, H, W)
                spatial_data['Interaction Layer'] = interaction_spatial
                available_levels.append('Interaction Layer')
                print(f"    Interaction Layer: shape {interaction_spatial.shape}, range [{interaction_spatial.min():.6f}, {interaction_spatial.max():.6f}]")
            else:
                print(f"      Warning: interaction_spatial has unexpected shape {interaction_spatial.shape}")
            
        if len(available_levels) == 0:
            print("No spatial importance data found in multi-level results")
            return None
        
        # Define color schemes for different branches (make adsorbate red darker)
        color_schemes = {
            'Adsorbate Branch': {'cmap': 'Reds', 'name': 'Adsorbate'},
            'Solvent Branch': {'cmap': 'Blues', 'name': 'Solvent'}, 
            'Interaction Layer': {'cmap': 'Greens', 'name': 'Interaction'}
        }
        
        # Create subplots for each available level (more compact layout)
        n_levels = len(available_levels)
        
        if n_levels == 1:
            fig = plt.figure(figsize=(10, 8))
            axes = [fig.add_subplot(111, projection='3d')]
        elif n_levels == 2:
            fig = plt.figure(figsize=(20, 8))
            axes = [fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d')]
            plt.subplots_adjust(wspace=0.1)  # Closer spacing
        else:  # 3 levels
            fig = plt.figure(figsize=(24, 8))  # Reduced height from 12 to 8, width from 36 to 24
            axes = [fig.add_subplot(131, projection='3d'), fig.add_subplot(132, projection='3d'), fig.add_subplot(133, projection='3d')]
            plt.subplots_adjust(wspace=0.05)  # Much closer spacing between subplots
        
        fig.patch.set_facecolor('white')
        
        for i, level_name in enumerate(available_levels):
            spatial_importance = spatial_data[level_name]
            color_scheme = color_schemes.get(level_name, color_schemes['Adsorbate Branch'])
            ax = axes[i]
            
            print(f"\n    Processing {level_name}...")
            
            # Use percentile-based normalization to handle outliers
            p_min, p_max = np.percentile(spatial_importance, [1, 99])
            
            # Adjust display threshold based on branch type for better visibility
            if 'Adsorbate' in level_name:
                # Use higher threshold for adsorbate to show only most important voxels
                display_mask = spatial_importance > np.percentile(spatial_importance, 95)  # Top 5%
                print(f"      Using higher threshold (95th percentile) for adsorbate visibility")
            else:
                display_mask = spatial_importance > np.percentile(spatial_importance, 85)  # Top 20%

            print(f"      Using percentile normalization: [{p_min:.8f}, {p_max:.8f}]")
            
            if not np.any(display_mask):
                print(f"      No voxels above cutoff for {level_name}, skipping...")
                ax.text(0.5, 0.5, 0.5, f'No significant\nvoxels for\n{level_name}', 
                       ha='center', va='center', fontsize=self.font_size, transform=ax.transAxes)
                continue
            
            # Set white background for this subplot
            ax.xaxis.pane.fill = True
            ax.yaxis.pane.fill = True
            ax.zaxis.pane.fill = True
            ax.xaxis.pane.set_facecolor('white')
            ax.yaxis.pane.set_facecolor('white')
            ax.zaxis.pane.set_facecolor('white')
            ax.xaxis.pane.set_alpha(1.0)
            ax.yaxis.pane.set_alpha(1.0)
            ax.zaxis.pane.set_alpha(1.0)
            
            # Create coordinate grids for voxel plotting
            x, y, z = np.indices(np.array(spatial_importance.shape) + 1)
            
            # Create colormap instance
            try:
                colormap = plt.colormaps[color_scheme['cmap']]
            except (AttributeError, KeyError):
                try:
                    colormap = plt.colormaps.get_cmap(color_scheme['cmap'])
                except AttributeError:
                    colormap = cm.get_cmap(color_scheme['cmap'])
            
            # Normalize importance values to 0-1 range
            norm_min, norm_max = p_min, p_max
            
            if norm_max > norm_min:
                normalized_importance = (spatial_importance - norm_min) / (norm_max - norm_min)
                normalized_importance = np.clip(normalized_importance, 0, 1)
            else:
                normalized_importance = np.zeros_like(spatial_importance)
            
            display_values = normalized_importance[display_mask]
            
            print(f"      Original range: [{norm_min:.8f}, {norm_max:.8f}]")
            print(f"      Normalized range: [0.0, 1.0]")
            print(f"      Number of displayed voxels: {len(display_values)}")
            print(f"      Displayed normalized values range: [{display_values.min():.3f}, {display_values.max():.3f}]")
            
            if norm_max > norm_min:
                # Apply color power transformation
                color_power = 0.7
                
                # Set different alpha values for adsorbate branch (lower opacity)
                if level_name == 'Adsorbate Branch':
                    alpha_min, alpha_max, alpha_power = 0.15, 0.65, 1.5  # Lower alpha values for adsorbate
                else:
                    alpha_min, alpha_max, alpha_power = 0.05, 0.95, 1.5
                
                color_normalized = normalized_importance ** color_power
                
                # Create alpha values based on normalized importance
                alpha_values = np.zeros_like(spatial_importance)
                display_normalized = normalized_importance[display_mask]
                display_alpha = alpha_min + (alpha_max - alpha_min) * (display_normalized ** alpha_power)
                alpha_values[display_mask] = display_alpha
                
                # Create RGBA colors for each voxel
                colors = np.zeros(spatial_importance.shape + (4,))
                
                coords = np.where(display_mask)
                for i_coord, j_coord, k_coord in zip(*coords):
                    color_value = color_normalized[i_coord, j_coord, k_coord]
                    rgba_color = colormap(color_value)
                    
                    # Enhance saturation for high-importance voxels
                    if color_value > 0.7:
                        r, g, b = rgba_color[0], rgba_color[1], rgba_color[2]
                        min_component = min(r, g, b)
                        enhancement_factor = 0.3
                        
                        if color_scheme['cmap'] == 'Reds':
                            enhanced_r = min(1.0, r + 0.1)
                            enhanced_g = max(0.0, g - min_component * enhancement_factor)
                            enhanced_b = max(0.0, b - min_component * enhancement_factor)
                            rgba_color = (enhanced_r, enhanced_g, enhanced_b, rgba_color[3])
                        elif color_scheme['cmap'] == 'Blues':
                            enhanced_b = min(1.0, b + 0.1)
                            enhanced_r = max(0.0, r - min_component * enhancement_factor)
                            enhanced_g = max(0.0, g - min_component * enhancement_factor)
                            rgba_color = (enhanced_r, enhanced_g, enhanced_b, rgba_color[3])
                        elif color_scheme['cmap'] == 'Greens':
                            enhanced_g = min(1.0, g + 0.1)
                            enhanced_r = max(0.0, r - min_component * enhancement_factor)
                            enhanced_b = max(0.0, b - min_component * enhancement_factor)
                            rgba_color = (enhanced_r, enhanced_g, enhanced_b, rgba_color[3])
                    
                    alpha = alpha_values[i_coord, j_coord, k_coord]
                    colors[i_coord, j_coord, k_coord] = [rgba_color[0], rgba_color[1], rgba_color[2], alpha]
                
                # Plot voxels with individual colors
                ax.voxels(x, y, z, display_mask, facecolors=colors, alpha=None, edgecolor=None)
                
                # Create colorbar
                sm = cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.02)
                
                # Set colorbar ticks
                cbar_ticks = np.linspace(0, 1, 6)
                cbar_labels = [f"{tick:.1f}" for tick in cbar_ticks]
                cbar.set_ticks(cbar_ticks)
                cbar.set_ticklabels(cbar_labels)
                cbar.set_label(f'{level_name} Importance', fontsize=self.font_size, color='black')
                cbar.ax.tick_params(labelsize=self.font_size)
            
            # Set axis properties
            ax.tick_params(axis='x', labelsize=self.font_size)
            ax.tick_params(axis='y', labelsize=self.font_size)
            ax.tick_params(axis='z', labelsize=self.font_size)
            
            # Set equal aspect ratio
            shape = np.array(spatial_importance.shape)
            aspect_ratio = shape / shape.max()
            ax.set_box_aspect(aspect_ratio)
            
            # Set axis limits
            ax.set_xlim([0, shape[2]])
            ax.set_ylim([0, shape[1]])
            ax.set_zlim([0, shape[0]])
            
            # Set integer ticks for all axes
            tick_positions = [0, 4, 8, 12, 16, 20]
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            ax.set_zticks(tick_positions)
            ax.set_xticklabels(tick_positions)
            ax.set_yticklabels(tick_positions)
            ax.set_zticklabels(tick_positions)
            
            # Set title for this subplot
            # ax.set_title(f'{level_name}\n3D Spatial Importance', fontsize=self.font_size, fontweight='bold')
            
            # Add subplot label (a), (b), or (c)
            subplot_labels = ['(a)', '(b)', '(c)']
            ax.text2D(0.0, 1.0, subplot_labels[i], fontsize=self.font_size+2, fontweight='bold',
                     transform=ax.transAxes, verticalalignment='top', horizontalalignment='left',)
        
        # Add overall title - commented out to remove titles
        # fig.suptitle(f'3D Processor Output Layer Spatial Importance\n'
        #             f'Sample: {sample_info["zeolite"]}-{sample_info["environment"]}-{sample_info["adsorbate"]} '
        #             f'(Snapshot {sample_info["snapshot"]}, Voxel {sample_info["voxel_id"]})',
        #             fontsize=self.font_size, fontweight='bold')
        
        plt.tight_layout()
        
        # Print summary
        print(f"\n✓ 3D processor output layer spatial importance plot generated")
        for level_name in available_levels:
            spatial_importance = spatial_data[level_name]
            print(f"    {level_name}: max={spatial_importance.max():.6f}, mean={spatial_importance.mean():.6f}")
        
        # Save plot
        if save_plots:
            save_path = os.path.join(self.output_dir, f"3d_processor_output-{sample_prefix}.png")
            fig.savefig(save_path, dpi=1000, bbox_inches='tight', facecolor='white')
            print(f"  ✓ 3D processor output layer plot saved: {save_path}")
        
        if show_plots:
            plt.show()
        else:
            # Keep figure open for Spyder plots panel when show_plots=False
            pass
            
        return fig


## SIMPLIFIED MAIN FUNCTION
if __name__ == "__main__":
    
    # Model results file
    # results_filename = "model-random-2546193-epochs_200-bs_32-lr_0.0001-grid_16.0_0.8.pkl" # Test MAE 0.088
    # results_filename = "model-random-2546194-epochs_200-bs_32-lr_0.0001-grid_16.0_0.8.pkl" # Test MAE 0.091
    # results_filename = "model-random-2546195-epochs_200-bs_32-lr_0.0001-grid_16.0_0.8.pkl" # Test MAE 0.090
    # results_filename = "model-random-2546197-epochs_200-bs_32-lr_0.0001-grid_16.0_0.8.pkl" # Test MAE 0.087
    # results_filename = "model-random-2546199-epochs_200-bs_32-lr_0.0001-grid_16.0_0.8.pkl" # Test MAE 0.087
    # results_filename = "model-random-2546201-epochs_200-bs_32-lr_0.0001-grid_16.0_0.8.pkl" # Test MAE 0.088
    # results_filename = "model-random-2546213-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.088
    # results_filename = "model-random-2546214-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.089
    # results_filename = "model-random-2546215-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.088
    # results_filename = "model-random-2546216-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.088
    # results_filename = "model-random-2546217-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.085
    # results_filename = "model-random-2546218-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.087
    # results_filename = "model-random-2546220-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.088
    # results_filename = "model-random-2546223-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.088
    # results_filename = "model-random-2546226-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.088
    # results_filename = "model-random-2546227-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.088
    # results_filename = "model-random-2546228-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.087
    results_filename = "model-random-2546229-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.083 # Best
    # results_filename = "model-random-2546238-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.086
    # results_filename = "model-random-2546239-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.087
    # results_filename = "model-random-2546240-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.087
    # results_filename = "model-random-2546241-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.088
    # results_filename = "model-random-2546243-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.086
    # results_filename = "model-random-2546244-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.087
    # results_filename = "model-random-2546246-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.084
    # results_filename = "model-random-2546247-epochs_200-bs_32-lr_0.0002-grid_16.0_0.8.pkl" # Test MAE 0.089
    
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

    
    # # Generate 3D spatial importance plot from input voxel layer
    # print(f"\n--- Generating 3D Spatial Importance Plot ---")
    # analyzer.plot_3d_spatial_importance_input_voxel_layer(analysis_results, show_plots=False, save_plots=False)
    
    # # Generate 3D spatial importance plot from processor output layers
    # print(f"\n--- Generating 3D Spatial Importance from Processor Output Layers ---")
    # analyzer.plot_3d_spatial_importance_processor_output_layer(analysis_results, show_plots=False, save_plots=False)
    
    
    # # Generate 2D spatial importance from input voxel layer
    # print(f"\n--- Generating 2D Spatial Importance from Input Voxel Layer ---")
    # analyzer.plot_2d_spatial_importance_input_voxel_layer(analysis_results, show_plots=False, save_plots=False)
    
    # # Generate 3D spatial importance from processor output layers
    # print(f"\n--- Generating 3D Spatial Importance from Processor Output Layers ---")
    # analyzer.plot_2d_spatial_importance_processor_output_layer(analysis_results, show_plots=False, save_plots=False)
    
    
    # # Generate multi-level comparison plot (NEW)
    # print(f"\n--- Generating Multi-Level Comparison Plot ---")
    # analyzer.plot_multilevel_comparison(analysis_results, show_plots=False, save_plots=False)
    
    # # Generate gradient distribution plots
    # print(f"\n--- Generating Gradient Distribution Plots ---")
    # analyzer.plot_gradient_distribution(analysis_results, show_plots=False, save_plots=False, num_top_features=6)
