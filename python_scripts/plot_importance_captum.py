# -*- coding: utf-8 -*-
"""
plot_importance_captum.py
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

# Captum imports for model interpretation
from captum.attr import IntegratedGradients, GradientShap, Saliency, GuidedBackprop, DeepLift
from captum.attr import LayerConductance, LayerActivation, LayerGradientXActivation
from captum.attr._utils.attribution import Attribution

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

class CaptumAttributionAnalyzer:
    """Captum-based attribution analysis for PyTorch 3D CNN models with multi-level analysis"""
    
    def __init__(self,
                 model,
                 device=None,
                 ):
        
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize Captum attribution methods
        self.integrated_gradients = IntegratedGradients(self.model)
        self.gradient_shap = GradientShap(self.model)
        self.saliency = Saliency(self.model)
        self.guided_backprop = GuidedBackprop(self.model)
        self.deep_lift = DeepLift(self.model)
        
        # Store intermediate activations for multi-level analysis
        self.layer_conductance = {}
        self.layer_activation = {}
        self.layer_grad_x_act = {}
        
        print(f"    Initialized Captum attribution analyzer with {len(self._get_available_methods())} methods")
        
    def _get_available_methods(self):
        """Get list of available attribution methods"""
        return ['integrated_gradients', 'gradient_shap', 'saliency', 'guided_backprop', 'deep_lift']
        
    def _setup_layer_analysis(self):
        """Setup layer-wise attribution methods for multi-level analysis"""
        self.layer_conductance = {}
        self.layer_activation = {}
        self.layer_grad_x_act = {}
        
        # Setup layer attribution for different processing stages
        if hasattr(self.model, 'adsorbate_processor'):
            self.layer_conductance['adsorbate_processor'] = LayerConductance(self.model, self.model.adsorbate_processor)
            self.layer_activation['adsorbate_processor'] = LayerActivation(self.model, self.model.adsorbate_processor)
            self.layer_grad_x_act['adsorbate_processor'] = LayerGradientXActivation(self.model, self.model.adsorbate_processor)
            
        if hasattr(self.model, 'solvent_processor'):
            self.layer_conductance['solvent_processor'] = LayerConductance(self.model, self.model.solvent_processor)
            self.layer_activation['solvent_processor'] = LayerActivation(self.model, self.model.solvent_processor)
            self.layer_grad_x_act['solvent_processor'] = LayerGradientXActivation(self.model, self.model.solvent_processor)
            
        if hasattr(self.model, 'interaction_conv'):
            self.layer_conductance['interaction_conv'] = LayerConductance(self.model, self.model.interaction_conv)
            self.layer_activation['interaction_conv'] = LayerActivation(self.model, self.model.interaction_conv)
            self.layer_grad_x_act['interaction_conv'] = LayerGradientXActivation(self.model, self.model.interaction_conv)
            
        print(f"    Setup layer analysis for {len(self.layer_conductance)} layers")
    
    def generate_baseline(self, input_tensor, method='zero'):
        """Generate baseline for attribution methods"""
        if method == 'zero':
            return torch.zeros_like(input_tensor)
        elif method == 'random':
            return torch.randn_like(input_tensor) * 0.1
        elif method == 'gaussian_noise':
            return torch.randn_like(input_tensor) * input_tensor.std() * 0.1
        elif method == 'mean':
            baseline = torch.zeros_like(input_tensor)
            for c in range(input_tensor.shape[1]):
                baseline[:, c] = input_tensor[:, c].mean()
            return baseline
        else:
            return torch.zeros_like(input_tensor)
    
    def compute_integrated_gradients(self, input_tensor, target_class=None, steps=50, baseline_method='zero'):
        """
        Compute integrated gradients using Captum with progress display
        
        Args:
            input_tensor: (1, C, D, H, W) tensor
            target_class: target class for attribution (None for regression)
            steps: number of integration steps
            baseline_method: method to generate baseline
        
        Returns:
            attributions: (C, D, H, W) attribution map
        """
        print(f"    Computing Captum integrated gradients with {steps} steps...")
        print(f"    Input tensor shape: {input_tensor.shape}")
        print(f"    Input tensor range: [{input_tensor.min():.6f}, {input_tensor.max():.6f}]")
        
        baseline = self.generate_baseline(input_tensor, baseline_method)
        
        # Get initial prediction for reference
        with torch.no_grad():
            baseline_pred = self.model(baseline)
            input_pred = self.model(input_tensor)
            pred_diff = input_pred - baseline_pred
            print(f"    Prediction difference (input - baseline): {pred_diff.item():.6f}")
        
        # Compute integrated gradients using Captum with internal batch size to show progress
        try:
            # Use smaller internal_batch_size to show progress updates
            internal_batch_size = max(1, steps // 10)  # Show progress in ~10 updates
            print(f"    Using internal batch size: {internal_batch_size} (will show progress)")
            
            attributions = self.integrated_gradients.attribute(
                input_tensor,
                baselines=baseline,
                target=target_class,
                n_steps=steps,
                method='gausslegendre',  # Use Gauss-Legendre integration for better accuracy
                internal_batch_size=internal_batch_size,  # This will enable batch processing and progress
                return_convergence_delta=False
            )
            
            # Print attribution statistics
            print(f"    Attribution statistics:")
            print(f"    - Final attributions range: [{attributions.min():.8f}, {attributions.max():.8f}]")
            print(f"    - Attribution sum: {attributions.sum():.8f}")
            print(f"    - Prediction difference: {pred_diff.item():.8f}")
            
            return attributions.detach().squeeze(0)  # Return (C, D, H, W)
            
        except Exception as e:
            print(f"    Error computing integrated gradients: {e}")
            return None
    
    def compute_gradient_shap(self, input_tensor, target_class=None, n_samples=50, stdevs=0.1):
        """
        Compute GradientSHAP attributions using Captum with progress info
        """
        print(f"    Computing Captum GradientSHAP with {n_samples} samples...")
        print(f"    Noise standard deviation: {stdevs}")
        
        # Generate random baselines for GradientSHAP
        baselines = torch.randn(n_samples, *input_tensor.shape[1:]).to(self.device) * stdevs
        print(f"    Generated {baselines.shape[0]} random baselines")
        
        try:
            print(f"    Starting GradientSHAP computation...")
            attributions = self.gradient_shap.attribute(
                input_tensor,
                baselines=baselines,
                target=target_class,
                stdevs=stdevs,
                n_samples=n_samples
            )
            
            print(f"    ✓ GradientSHAP completed")
            print(f"    GradientSHAP attributions range: [{attributions.min():.8f}, {attributions.max():.8f}]")
            return attributions.detach().squeeze(0)
            
        except Exception as e:
            print(f"    Error computing GradientSHAP: {e}")
            return None
    
    def compute_saliency(self, input_tensor, target_class=None):
        """
        Compute Saliency (simple gradients) using Captum
        """
        print(f"    Computing Captum Saliency...")
        
        try:
            attributions = self.saliency.attribute(
                input_tensor,
                target=target_class,
                abs=False  # Don't take absolute value automatically
            )
            
            print(f"    Saliency attributions range: [{attributions.min():.8f}, {attributions.max():.8f}]")
            return attributions.detach().squeeze(0)
            
        except Exception as e:
            print(f"    Error computing Saliency: {e}")
            return None
    
    def compute_guided_backprop(self, input_tensor, target_class=None):
        """
        Compute Guided Backpropagation using Captum
        """
        print(f"    Computing Captum Guided Backpropagation...")
        
        try:
            attributions = self.guided_backprop.attribute(
                input_tensor,
                target=target_class
            )
            
            print(f"    Guided Backprop attributions range: [{attributions.min():.8f}, {attributions.max():.8f}]")
            return attributions.detach().squeeze(0)
            
        except Exception as e:
            print(f"    Error computing Guided Backpropagation: {e}")
            return None
    
    def compute_deep_lift(self, input_tensor, target_class=None, baseline_method='zero'):
        """
        Compute DeepLift attributions using Captum
        """
        print(f"    Computing Captum DeepLift...")
        
        baseline = self.generate_baseline(input_tensor, baseline_method)
        
        try:
            attributions = self.deep_lift.attribute(
                input_tensor,
                baselines=baseline,
                target=target_class
            )
            
            print(f"    DeepLift attributions range: [{attributions.min():.8f}, {attributions.max():.8f}]")
            return attributions.detach().squeeze(0)
            
        except Exception as e:
            print(f"    Error computing DeepLift: {e}")
            return None
    
    def compute_multilevel_layer_analysis(self, input_tensor, target_class=None, attribution_method='conductance'):
        """
        Compute layer-wise attributions at multiple levels using Captum with progress info
        
        Args:
            input_tensor: (1, C, D, H, W) tensor
            target_class: target class for attribution (None for regression)
            attribution_method: 'conductance', 'activation', or 'grad_x_activation'
        
        Returns:
            dict: multi-level attribution maps
        """
        print(f"    Computing multi-level layer analysis using {attribution_method}...")
        
        # Setup layer analysis if not already done
        if not self.layer_conductance:
            self._setup_layer_analysis()
        
        results = {}
        
        # Select the appropriate attribution method
        if attribution_method == 'conductance':
            layer_methods = self.layer_conductance
        elif attribution_method == 'activation':
            layer_methods = self.layer_activation
        elif attribution_method == 'grad_x_activation':
            layer_methods = self.layer_grad_x_act
        else:
            print(f"    Unknown attribution method: {attribution_method}, using conductance")
            layer_methods = self.layer_conductance
        
        print(f"    Processing {len(layer_methods)} layers...")
        
        for i, (layer_name, layer_method) in enumerate(layer_methods.items()):
            print(f"      [{i+1}/{len(layer_methods)}] Computing {attribution_method} for {layer_name}...")
            
            try:
                if attribution_method == 'activation':
                    # For activation, we don't need target
                    print(f"        Running layer activation analysis...")
                    attributions = layer_method.attribute(input_tensor)
                else:
                    # For conductance and grad_x_activation
                    print(f"        Running layer {attribution_method} analysis...")
                    attributions = layer_method.attribute(
                        input_tensor,
                        target=target_class
                    )
                
                # Convert to spatial importance by averaging across channels if needed
                if len(attributions.shape) == 5:  # (batch, channels, D, H, W)
                    spatial_importance = torch.mean(torch.abs(attributions.squeeze(0)), dim=0)  # Average across channels
                elif len(attributions.shape) == 4:  # (channels, D, H, W) - already squeezed
                    spatial_importance = torch.mean(torch.abs(attributions), dim=0)  # Average across channels
                else:
                    spatial_importance = attributions.squeeze()
                
                # Store results with appropriate naming
                if layer_name == 'adsorbate_processor':
                    results['adsorbate_spatial'] = spatial_importance.detach().cpu()
                elif layer_name == 'solvent_processor':
                    results['solvent_spatial'] = spatial_importance.detach().cpu()
                elif layer_name == 'interaction_conv':
                    results['interaction_spatial'] = spatial_importance.detach().cpu()
                else:
                    results[f'{layer_name}_spatial'] = spatial_importance.detach().cpu()
                
                print(f"        ✓ {layer_name}: shape {spatial_importance.shape}, range [{spatial_importance.min():.6f}, {spatial_importance.max():.6f}]")
                
            except Exception as e:
                print(f"        ❌ Error computing {attribution_method} for {layer_name}: {e}")
        
        print(f"    ✓ Multi-level layer analysis completed. Available data: {list(results.keys())}")
        return results
    
    def compute_all_attributions(self, input_tensor, target_class=None, steps=50, 
                                baseline_method='zero', include_layer_analysis=True):
        """
        Compute multiple attribution methods for comprehensive analysis
        
        Returns:
            dict: Results from multiple attribution methods
        """
        print(f"    Computing comprehensive attribution analysis...")
        
        results = {}
        
        # Compute different attribution methods
        methods = {
            'integrated_gradients': lambda: self.compute_integrated_gradients(
                input_tensor, target_class, steps, baseline_method),
            'gradient_shap': lambda: self.compute_gradient_shap(
                input_tensor, target_class, n_samples=25),
            'saliency': lambda: self.compute_saliency(input_tensor, target_class),
            'guided_backprop': lambda: self.compute_guided_backprop(input_tensor, target_class),
            'deep_lift': lambda: self.compute_deep_lift(input_tensor, target_class, baseline_method)
        }
        
        for method_name, method_func in methods.items():
            print(f"\n      [{list(methods.keys()).index(method_name)+1}/{len(methods)}] Computing {method_name}...")
            try:
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if start_time:
                    start_time.record()
                
                attribution = method_func()
                
                if end_time and start_time:
                    end_time.record()
                    torch.cuda.synchronize()
                    elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
                    
                if attribution is not None:
                    results[method_name] = attribution.cpu().numpy()
                    time_str = f" (took {elapsed_time:.2f}s)" if start_time and end_time else ""
                    print(f"        ✓ {method_name} completed{time_str}")
                    print(f"          Shape: {attribution.shape}, Range: [{attribution.min():.6f}, {attribution.max():.6f}]")
                else:
                    print(f"        ❌ {method_name} failed - returned None")
            except Exception as e:
                print(f"        ❌ {method_name} failed with error: {e}")
        
        # Compute layer-wise analysis if model supports it and requested
        if include_layer_analysis:
            print(f"\n    === Layer-wise Analysis ===")
            layer_methods = ['conductance', 'activation', 'grad_x_activation']
            for i, layer_method in enumerate(layer_methods):
                print(f"\n      [{i+1}/{len(layer_methods)}] Computing layer {layer_method}...")
                try:
                    layer_results = self.compute_multilevel_layer_analysis(
                        input_tensor, target_class, layer_method)
                    
                    if layer_results:
                        for key, value in layer_results.items():
                            results[f'{layer_method}_{key}'] = value.detach().numpy() if hasattr(value, 'detach') else value.numpy()
                        print(f"        ✓ Layer {layer_method} completed - added {len(layer_results)} spatial maps")
                    else:
                        print(f"        ❌ Layer {layer_method} failed - no results")
                        
                except Exception as e:
                    print(f"        ❌ Layer {layer_method} failed with error: {e}")
        
        print(f"    Comprehensive attribution analysis completed with {len(results)} methods")
        return results

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
        self.output_dir = os.path.join(get_paths("output_figure_path"), "cnn_saliency_maps_captum")
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
                      ensemble_method='mean', use_comprehensive_analysis=False):
        """
        Main function to analyze a specific sample and generate importance map using Captum
        
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
            use_comprehensive_analysis: if True, compute multiple Captum methods for comparison
        """
        # Create cache filename based on analysis parameters with 'captum' suffix
        cache_filename = f"{self.model_prefix}-{zeolite}-{env_adsorbate}-snap{snapshot}-vox{voxel_id}-captum.pkl"
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
                                                       snapshot, voxel_id, steps, ensemble_method, 
                                                       use_comprehensive_analysis)
        
        # Save result to cache
        if analysis_result is not None:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(analysis_result, f)
                print(f"✓ Cached analysis result to: {cache_filename}")
            except Exception as e:
                print(f"⚠️ Failed to save cache: {e}")
                
            # Print detailed analysis summary
            print(f"\n=== Captum Analysis Summary ===")
            print(f"Attribution method: {analysis_result.get('attribution_method', 'N/A')}")
            
            config = analysis_result.get('analysis_config', {})
            print(f"Analysis configuration:")
            print(f"  - Steps: {config.get('steps', 'N/A')}")
            print(f"  - Baseline method: {config.get('baseline_method', 'N/A')}")
            print(f"  - Multi-level analysis: {config.get('include_layer_analysis', 'N/A')}")
            print(f"  - Comprehensive analysis: {config.get('use_comprehensive_analysis', 'N/A')}")
            
            captum_methods = analysis_result.get('captum_methods', {})
            print(f"Available Captum methods: {list(captum_methods.keys())}")
            
            # Show primary importance map statistics
            importance_map = analysis_result.get('importance_map')
            if importance_map is not None:
                print(f"Primary importance map (integrated gradients):")
                print(f"  - Shape: {importance_map.shape}")
                print(f"  - Range: [{importance_map.min():.8f}, {importance_map.max():.8f}]")
                print(f"  - Mean: {importance_map.mean():.8f}")
            
            # Show multi-level data if available
            multilevel_data = analysis_result.get('multilevel_data')
            if multilevel_data:
                print(f"Multi-level spatial importance data:")
                for key, data in multilevel_data.items():
                    if isinstance(data, np.ndarray):
                        print(f"  - {key}: shape {data.shape}, range [{data.min():.6f}, {data.max():.6f}]")
            else:
                print(f"Multi-level spatial importance: Not available")
        
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
    
    def _analyze_sample_ensemble(self, results, zeolite, env_adsorbate, snapshot, voxel_id, steps, ensemble_method, use_comprehensive_analysis=False):
        """Ensemble analysis of multiple train folds"""
        print(f"\n--- Ensemble Analysis Mode ---")
        
        # Find all train folds
        train_folds = self._find_train_folds_for_sample(results, zeolite, env_adsorbate, snapshot, voxel_id)
        
        if len(train_folds) == 0:
            print(f"❌ No train folds found, falling back to single analysis")
            return self._analyze_sample_single(results, zeolite, env_adsorbate, snapshot, voxel_id, steps, None, use_comprehensive_analysis)
        
        print(f"✓ Found {len(train_folds)} train folds: {[f['fold_idx'] for f in train_folds]}")
        
        # Collect importance maps from each fold
        fold_importance_maps = []
        fold_predictions = []
        fold_multilevel_data = []  # New: collect multi-level data
        input_tensor = None  # Store input tensor from first successful fold
        
        for fold_info in train_folds:
            fold_idx = fold_info['fold_idx']
            print(f"\n  Analyzing Fold {fold_idx}...")
            
            # Use single analysis to get results for this fold
            single_result = self._analyze_sample_single(results, zeolite, env_adsorbate, 
                                                      snapshot, voxel_id, steps, fold_idx,
                                                      use_comprehensive_analysis)
            
            if single_result is not None:
                fold_importance_maps.append(single_result['importance_map'])
                fold_predictions.append({
                    'fold': fold_idx,
                    'y_true': fold_info['y_true'],
                    'y_pred': fold_info['y_pred']
                })
                
                # Store input tensor from first successful fold (same for all folds)
                if input_tensor is None and 'input_tensor' in single_result:
                    input_tensor = single_result['input_tensor']
                
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
            'input_tensor': input_tensor,  # Include input tensor for layer analysis
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
    
    def _analyze_sample_single(self, results, zeolite, env_adsorbate, snapshot, voxel_id, steps, fold_num, use_comprehensive_analysis=False):
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
        
        # Generate importance map using Captum attribution methods
        captum_analyzer = CaptumAttributionAnalyzer(model, self.device)
        print(f"\n--- Computing Captum attributions...")
        
        # Check if model supports multi-level analysis
        has_branches = (hasattr(model, 'adsorbate_processor') and 
                       hasattr(model, 'solvent_processor') and 
                       hasattr(model, 'interaction_conv'))
        
        if has_branches:
            print(f"    Model supports multi-level analysis (adsorbate/solvent branches + interaction)")
            
            if use_comprehensive_analysis:
                # Use comprehensive analysis with multiple Captum methods
                all_attributions = captum_analyzer.compute_all_attributions(
                    input_tensor, steps=steps, include_layer_analysis=True)
                
                # Use integrated gradients as primary method
                importance_map = all_attributions.get('integrated_gradients', 
                    captum_analyzer.compute_integrated_gradients(input_tensor, steps=steps).cpu().numpy())
                
                # Extract layer analysis results
                multilevel_data = {}
                for key in ['adsorbate_spatial', 'solvent_spatial', 'interaction_spatial']:
                    # Check different layer method results
                    for method in ['conductance', 'activation', 'grad_x_activation']:
                        method_key = f'{method}_{key}'
                        if method_key in all_attributions:
                            multilevel_data[f'{method}_{key}'] = all_attributions[method_key]
                            # Use conductance as primary method for compatibility
                            if method == 'conductance' and key not in multilevel_data:
                                multilevel_data[key] = all_attributions[method_key]
                
                # Store comprehensive results
                captum_methods = {k: v for k, v in all_attributions.items() 
                                if k not in ['conductance_adsorbate_spatial', 'conductance_solvent_spatial', 'conductance_interaction_spatial']}
            else:
                # Standard analysis - only integrated gradients and layer conductance
                importance_map = captum_analyzer.compute_integrated_gradients(input_tensor, steps=steps).cpu().numpy()
                
                # Compute layer-wise spatial importance using Captum layer methods
                layerwise_results = captum_analyzer.compute_multilevel_layer_analysis(
                    input_tensor, attribution_method='conductance')
                
                # Store multi-level results
                multilevel_data = {}
                for key in ['adsorbate_spatial', 'solvent_spatial', 'interaction_spatial']:
                    if key in layerwise_results:
                        multilevel_data[key] = layerwise_results[key].detach().cpu().numpy()
                
                captum_methods = {'integrated_gradients': importance_map}
        else:
            print(f"    Model uses standard analysis (no branch support)")
            if use_comprehensive_analysis:
                # Use multiple methods but no layer analysis
                all_attributions = captum_analyzer.compute_all_attributions(
                    input_tensor, steps=steps, include_layer_analysis=False)
                importance_map = all_attributions.get('integrated_gradients',
                    captum_analyzer.compute_integrated_gradients(input_tensor, steps=steps).cpu().numpy())
                captum_methods = all_attributions
            else:
                importance_map = captum_analyzer.compute_integrated_gradients(input_tensor, steps=steps).cpu().numpy()
                captum_methods = {'integrated_gradients': importance_map}
            multilevel_data = None
        
        # Create analysis results with Captum-based data structure
        analysis_results = {
            'sample_info': sample_info,
            'importance_map': importance_map,  # Primary method (integrated gradients)
            'multilevel_data': multilevel_data,  # Multi-level importance data
            'captum_methods': captum_methods,  # Store results from different Captum methods
            'prediction_diff': prediction_diff,
            'data_source': data_source,
            'input_tensor': input_tensor.detach().cpu().numpy(),
            'raw_prediction': raw_prediction.item(),
            'scaler_info': scaler_info,
            'has_multilevel': has_branches,  # Flag to indicate multi-level support
            'attribution_method': 'captum',  # Indicate this is Captum-based analysis
            'analysis_config': {
                'steps': steps,
                'baseline_method': 'zero',
                'include_layer_analysis': has_branches,
                'use_comprehensive_analysis': use_comprehensive_analysis,
                'available_methods': list(captum_methods.keys()) if captum_methods else []
            }
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


    def plot_2d_saliency_map_input_voxel_layer(self, analysis_results, show_plots=True, save_plots=True):
        """Generate 2D Captum saliency map visualization from input voxel layer (enhanced version focusing on saliency)"""
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
        
        print(f"\n=== Generating 2D Captum Saliency Map from Input Voxel Layer ===")
        
        # Add ensemble info
        if fold_consistency is not None:
            print(f"    Ensemble consistency (std): mean={np.mean(fold_consistency):.6f}")
        
        # Create sample-specific filename prefix with model name
        sample_prefix = f"{self.model_prefix}-{sample_info['zeolite']}-{sample_info['environment']}-{sample_info['adsorbate']}-snap{sample_info['snapshot']}-vox{sample_info['voxel_id']}"
        
        # Captum attribution analysis focuses on saliency rather than raw gradients
        print("    Computing Captum-based saliency visualization...")
        
        # Split importance into adsorbate and solvent channels
        num_atomic_features = len(self.atomic_features)
        adsorbate_importance = primary_importance[:num_atomic_features]  # Channels 0-13
        solvent_importance = primary_importance[num_atomic_features:]    # Channels 14-27
        
        # Generate saliency-based spatial importance (absolute values emphasize significance)
        print("    Computing saliency-based spatial importance for adsorbate and solvent branches...")
        adsorbate_saliency = np.mean(np.abs(adsorbate_importance), axis=0)  # Saliency: absolute importance
        solvent_saliency = np.mean(np.abs(solvent_importance), axis=0)      # Saliency: absolute importance
        
        # Get spatial dimensions
        depth, height, width = adsorbate_saliency.shape
        print(f"    Grid shape: {adsorbate_saliency.shape}")
        
        # Generate averaged projections for saliency visualization
        print("    Computing averaged saliency projections for comprehensive view...")
        projection_data = {}
        for name, saliency_data in [('adsorbate', adsorbate_saliency), 
                                   ('solvent', solvent_saliency)]:
            projections = {}
            
            # Z-projection (XY plane) - average saliency over all Z layers
            projections['z'] = np.mean(saliency_data, axis=0)  # Average over depth (Z-axis)
            print(f"      {name} Z-saliency projection: averaged over {depth} layers")
            
            # Y-projection (XZ plane) - average saliency over all Y layers  
            projections['y'] = np.mean(saliency_data, axis=1)  # Average over height (Y-axis)
            print(f"      {name} Y-saliency projection: averaged over {height} layers")
                
            # X-projection (YZ plane) - average saliency over all X layers
            projections['x'] = np.mean(saliency_data, axis=2)  # Average over width (X-axis)
            print(f"      {name} X-saliency projection: averaged over {width} layers")
            
            projection_data[name] = projections
        
        # Create figure with 2 rows (adsorbate, solvent) and 3 columns (X, Y, Z projections)
        figsize = (18, 12)  # Standard size
        fig = plt.figure(figsize=figsize, facecolor='white')
        fig.patch.set_facecolor('white')
        
        # Use GridSpec to control subplot layout more precisely
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 0.05], hspace=0.1, wspace=0.25)
        
        # Define row information for saliency visualization
        row_info = [
            ('adsorbate', 'Adsorbate Branch', 'Reds'),     # Red colormap for adsorbate saliency
            ('solvent', 'Solvent Branch', 'Blues')         # Blue colormap for solvent saliency
        ]
        
        for row_idx, (branch_name, branch_label, cmap_name) in enumerate(row_info):
            # Get averaged saliency projections for three views
            saliency_data_current = projection_data[branch_name]
            
            # Z-projection (XY plane)
            ax_z = fig.add_subplot(gs[row_idx, 0])
            im_z = ax_z.imshow(saliency_data_current['z'], cmap=cmap_name, interpolation='nearest', origin='lower')
            ax_z.set_title(f'{branch_label}', fontsize=self.font_size)
            ax_z.set_xlabel('X', fontsize=self.font_size)
            ax_z.set_ylabel('Y', fontsize=self.font_size)
            ax_z.set_aspect('equal')
            ax_z.tick_params(axis='both', labelsize=self.font_size)
            
            # Y-projection (XZ plane)
            ax_y = fig.add_subplot(gs[row_idx, 1])
            im_y = ax_y.imshow(saliency_data_current['y'], cmap=cmap_name, interpolation='nearest', origin='lower')
            ax_y.set_title(f'{branch_label}', fontsize=self.font_size)
            ax_y.set_xlabel('X', fontsize=self.font_size)
            ax_y.set_ylabel('Z', fontsize=self.font_size)
            ax_y.set_aspect('equal')
            ax_y.tick_params(axis='both', labelsize=self.font_size)
            
            # X-projection (YZ plane)
            ax_x = fig.add_subplot(gs[row_idx, 2])
            im_x = ax_x.imshow(saliency_data_current['x'], cmap=cmap_name, interpolation='nearest', origin='lower')
            ax_x.set_title(f'{branch_label}', fontsize=self.font_size)
            ax_x.set_xlabel('Y', fontsize=self.font_size)
            ax_x.set_ylabel('Z', fontsize=self.font_size)
            ax_x.set_aspect('equal')
            ax_x.tick_params(axis='both', labelsize=self.font_size)
            
            # Colorbar in separate column
            cax = fig.add_subplot(gs[row_idx, 3])
            cbar_x = plt.colorbar(im_x, cax=cax, shrink=0.5)
            cbar_x.set_label('Attribution', fontsize=self.font_size)
            # Hide colorbar numbers
            cbar_x.set_ticks([])
        
        plt.tight_layout()
        
        # Print saliency summary information
        print(f"    Created Captum saliency projections:")
        print(f"    - Adsorbate branch: channels 0-{num_atomic_features-1}")
        print(f"    - Solvent branch: channels {num_atomic_features}-{2*num_atomic_features-1}")
        
        # Calculate and print branch saliency comparison statistics
        ads_saliency_total = np.sum(adsorbate_saliency)
        solv_saliency_total = np.sum(solvent_saliency)
        print(f"    Branch saliency comparison:")
        print(f"    - Adsorbate spatial saliency total: {ads_saliency_total:.6f}")
        print(f"    - Solvent spatial saliency total: {solv_saliency_total:.6f}")
        print(f"    - Solvent/Adsorbate saliency ratio: {solv_saliency_total/ads_saliency_total:.3f}")
        
        # Save plot
        if save_plots:
            save_path = os.path.join(self.output_dir, f"2d_captum_saliency_input-{sample_prefix}.png")
            fig.savefig(save_path, dpi=1000, bbox_inches='tight')
            print(f"  ✓ 2D Captum saliency map (input layer) saved: {save_path}")
        
        if show_plots:
            plt.show()
            
        return fig

    
    def plot_2d_saliency_map_processor_output_layer(self, analysis_results, show_plots=True, save_plots=True):
        """Generate 2D Captum saliency map visualization from processor branch output layers (enhanced saliency focus)"""
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
        
        print(f"\n=== Generating 2D Captum Saliency Map from Processor Output Layers ===")
        print(f"Available multi-level data: {list(multilevel_data.keys())}")
        
        # Process available spatial saliency data
        saliency_data = {}
        available_levels = []
        
        if 'adsorbate_spatial' in multilevel_data:
            adsorbate_spatial = multilevel_data['adsorbate_spatial']
            if len(adsorbate_spatial.shape) == 3:  # Ensure it's 3D (D, H, W)
                # Convert to saliency (absolute values emphasize significance)
                saliency_data['Adsorbate Branch'] = np.abs(adsorbate_spatial)
                available_levels.append('Adsorbate Branch')
            else:
                print(f"      Warning: adsorbate_spatial has unexpected shape {adsorbate_spatial.shape}")
            
        if 'solvent_spatial' in multilevel_data:
            solvent_spatial = multilevel_data['solvent_spatial']
            if len(solvent_spatial.shape) == 3:  # Ensure it's 3D (D, H, W)
                # Convert to saliency (absolute values emphasize significance)
                saliency_data['Solvent Branch'] = np.abs(solvent_spatial)
                available_levels.append('Solvent Branch')
            else:
                print(f"      Warning: solvent_spatial has unexpected shape {solvent_spatial.shape}")
            
        if 'interaction_spatial' in multilevel_data:
            interaction_spatial = multilevel_data['interaction_spatial']
            if len(interaction_spatial.shape) == 3:  # Ensure it's 3D (D, H, W)
                # Convert to saliency (absolute values emphasize significance)
                saliency_data['Interaction Layer'] = np.abs(interaction_spatial)
                available_levels.append('Interaction Layer')
            else:
                print(f"      Warning: interaction_spatial has unexpected shape {interaction_spatial.shape}")
            
        if len(available_levels) == 0:
            print("No spatial saliency data found in multi-level results")
            return None
        
        # Define color schemes for different branches (saliency-focused)
        color_schemes = {
            'Adsorbate Branch': {'cmap': 'Reds', 'name': 'Adsorbate'},
            'Solvent Branch': {'cmap': 'Blues', 'name': 'Solvent'}, 
            'Interaction Layer': {'cmap': 'Greens', 'name': 'Interaction'}
        }
        
        # Create figure with n rows (branches) x 3 columns (views)
        n_levels = len(available_levels)
        fig = plt.figure(figsize=(18, 6*n_levels))
        
        # Use GridSpec to control subplot layout more precisely
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(n_levels, 4, figure=fig, width_ratios=[1, 1, 1, 0.05], hspace=0.1, wspace=0.25)
        
        for i, level_name in enumerate(available_levels):
            spatial_saliency = saliency_data[level_name]  # Already converted to saliency (absolute values)
            color_scheme = color_schemes.get(level_name, color_schemes['Adsorbate Branch'])
            
            # Get center slices for three saliency views
            depth, height, width = spatial_saliency.shape
            
            # Z-slice (XY plane) - center slice saliency
            if depth % 2 == 1:
                center_z = depth // 2
                z_slice = spatial_saliency[center_z, :, :]
            else:
                center_z1, center_z2 = depth // 2 - 1, depth // 2
                z_slice = (spatial_saliency[center_z1, :, :] + spatial_saliency[center_z2, :, :]) / 2
                center_z = f"{center_z1}-{center_z2}"
            
            # Y-slice (XZ plane) - center slice saliency
            if height % 2 == 1:
                center_y = height // 2
                y_slice = spatial_saliency[:, center_y, :]
            else:
                center_y1, center_y2 = height // 2 - 1, height // 2
                y_slice = (spatial_saliency[:, center_y1, :] + spatial_saliency[:, center_y2, :]) / 2
                center_y = f"{center_y1}-{center_y2}"
            
            # X-slice (YZ plane) - center slice saliency
            if width % 2 == 1:
                center_x = width // 2
                x_slice = spatial_saliency[:, :, center_x]
            else:
                center_x1, center_x2 = width // 2 - 1, width // 2
                x_slice = (spatial_saliency[:, :, center_x1] + spatial_saliency[:, :, center_x2]) / 2
                center_x = f"{center_x1}-{center_x2}"
            
            # Plot Z-slice (XY plane) saliency
            ax_z = fig.add_subplot(gs[i, 0])
            im_z = ax_z.imshow(z_slice, cmap=color_scheme['cmap'], interpolation='nearest', origin='lower')
            ax_z.set_title(f'{level_name}', fontsize=self.font_size)
            ax_z.set_xlabel('X', fontsize=self.font_size)
            ax_z.set_ylabel('Y', fontsize=self.font_size)
            ax_z.set_aspect('equal')
            ax_z.tick_params(axis='both', labelsize=self.font_size)
            
            # Plot Y-slice (XZ plane) saliency
            ax_y = fig.add_subplot(gs[i, 1])
            im_y = ax_y.imshow(y_slice, cmap=color_scheme['cmap'], interpolation='nearest', origin='lower')
            ax_y.set_title(f'{level_name}', fontsize=self.font_size)
            ax_y.set_xlabel('X', fontsize=self.font_size)
            ax_y.set_ylabel('Z', fontsize=self.font_size)
            ax_y.set_aspect('equal')
            ax_y.tick_params(axis='both', labelsize=self.font_size)
            
            # Plot X-slice (YZ plane) saliency
            ax_x = fig.add_subplot(gs[i, 2])
            im_x = ax_x.imshow(x_slice, cmap=color_scheme['cmap'], interpolation='nearest', origin='lower')
            ax_x.set_title(f'{level_name}', fontsize=self.font_size)
            ax_x.set_xlabel('Y', fontsize=self.font_size)
            ax_x.set_ylabel('Z', fontsize=self.font_size)
            ax_x.set_aspect('equal')
            ax_x.tick_params(axis='both', labelsize=self.font_size)
            
            # Colorbar in separate column
            cax = fig.add_subplot(gs[i, 3])
            cbar_x = plt.colorbar(im_x, cax=cax, shrink=0.5)
            cbar_x.set_label('Attribution', fontsize=self.font_size)
            # Hide colorbar numbers
            cbar_x.set_ticks([])
        
        plt.tight_layout()
        
        # Print saliency summary
        print(f"✓ 2D Captum processor output layer saliency plot generated")
        for level_name in available_levels:
            spatial_saliency = saliency_data[level_name]
            print(f"    {level_name}: max_saliency={spatial_saliency.max():.6f}, mean_saliency={spatial_saliency.mean():.6f}")
        
        # Save plot
        if save_plots:
            save_path = os.path.join(self.output_dir, f"2d_captum_saliency_processor-{sample_prefix}.png")
            fig.savefig(save_path, dpi=1000, bbox_inches='tight', facecolor='white')
            print(f"  ✓ 2D Captum saliency map (processor layer) saved: {save_path}")
        
        if show_plots:
            plt.show()
            
        return fig


    def plot_3d_saliency_map_input_voxel_layer(self, analysis_results, show_plots=True, save_plots=True):
        """Generate 3D Captum saliency map from input voxel layer with separated adsorbate/solvent views"""
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
        
        print(f"\n=== Generating 3D Captum Saliency Map from Input Voxel Layer ===")
        
        # Add ensemble info
        if fold_consistency is not None:
            print(f"    Ensemble consistency (std): mean={np.mean(fold_consistency):.6f}")
        
        # Split importance into adsorbate and solvent channels and convert to saliency
        num_atomic_features = len(self.atomic_features)
        adsorbate_importance = primary_importance[:num_atomic_features]  # Channels 0-13
        solvent_importance = primary_importance[num_atomic_features:]    # Channels 14-27
        
        # Generate saliency-based spatial importance (absolute values for significance)
        print("    Computing Captum saliency for adsorbate and solvent branches...")
        adsorbate_saliency = np.mean(np.abs(adsorbate_importance), axis=0)  # Saliency: absolute importance
        solvent_saliency = np.mean(np.abs(solvent_importance), axis=0)      # Saliency: absolute importance
        
        # Define saliency data and color schemes
        saliency_data = {
            'Adsorbate Branch': adsorbate_saliency,
            'Solvent Branch': solvent_saliency
        }
        
        color_schemes = {
            'Adsorbate Branch': {'cmap': 'Reds', 'name': 'Adsorbate'},
            'Solvent Branch': {'cmap': 'Blues', 'name': 'Solvent'}
        }
        
        # Create subplots (2 subplots side by side)
        fig = plt.figure(figsize=(22.5, 12))
        axes = [fig.add_subplot(121, projection='3d'),
                fig.add_subplot(122, projection='3d')]
        
        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0)
        
        fig.patch.set_facecolor('white')
        
        # Print summary statistics
        print(f"    Adsorbate saliency: max={adsorbate_saliency.max():.6f}, mean={adsorbate_saliency.mean():.6f}, std={adsorbate_saliency.std():.6f}")
        print(f"    Solvent saliency: max={solvent_saliency.max():.6f}, mean={solvent_saliency.mean():.6f}, std={solvent_saliency.std():.6f}")
        print(f"    Adsorbate/Solvent max ratio: {adsorbate_saliency.max()/solvent_saliency.max():.6f}")
        print(f"    Adsorbate percentiles (90%, 95%, 99%): {np.percentile(adsorbate_saliency, [90, 95, 99])}")
        print(f"    Solvent percentiles (90%, 95%, 99%): {np.percentile(solvent_saliency, [90, 95, 99])}")
        
        for i, (group_name, spatial_saliency) in enumerate(saliency_data.items()):
            color_scheme = color_schemes[group_name]
            ax = axes[i]
            
            print(f"\n    Processing {group_name}...")
            print(f"      Raw saliency stats: min={spatial_saliency.min():.8f}, max={spatial_saliency.max():.8f}, mean={spatial_saliency.mean():.8f}")
            
            # Use percentile-based normalization to handle outliers
            # Adjust display threshold based on saliency type
            if 'Adsorbate' in group_name:
                # Use higher percentile for adsorbate since many values are zero
                for percentile in [95, 96, 97, 98, 99]:
                    threshold = np.percentile(spatial_saliency, percentile)
                    if threshold > 0:
                        display_mask = spatial_saliency > threshold
                        print(f"      Using {percentile}th percentile threshold ({threshold:.8f}) for adsorbate saliency visibility")
                        break
                else:
                    # If all percentiles are 0, use top non-zero values
                    non_zero_values = spatial_saliency[spatial_saliency > 0]
                    if len(non_zero_values) > 0:
                        threshold = np.percentile(non_zero_values, 80)  # Top 20% of non-zero values
                        display_mask = spatial_saliency > threshold
                        print(f"      Using top 20% of non-zero values as threshold ({threshold:.8f}) for adsorbate saliency")
                    else:
                        display_mask = spatial_saliency > 0  # Show any non-zero
                        print(f"      Using any non-zero threshold for adsorbate saliency")
            else:
                display_mask = spatial_saliency > np.percentile(spatial_saliency, 88)  # Top 12%
                print(f"      Using standard threshold (88th percentile) for solvent saliency")
            
            # Use robust normalization based on displayed values to avoid outlier compression
            if np.any(display_mask):
                displayed_values = spatial_saliency[display_mask]
                # Use 90th percentile of displayed values as max to avoid outlier compression
                p_min = displayed_values.min()
                p_max = np.percentile(displayed_values, 90)  # Use 90% instead of max to avoid outliers
                print(f"      Using robust normalization based on displayed values: [{p_min:.8f}, {p_max:.8f}]")
                print(f"      Displayed values range: min={displayed_values.min():.8f}, max={displayed_values.max():.8f}")
                print(f"      Using 90th percentile ({p_max:.8f}) as normalization max to avoid outlier compression")
            else:
                p_min, p_max = np.percentile(spatial_saliency, [1, 99])
                print(f"      Using fallback percentile normalization: [{p_min:.8f}, {p_max:.8f}]")
            
            if not np.any(display_mask):
                print(f"      No significant saliency voxels for {group_name}, skipping...")
                ax.text(0.5, 0.5, 0.5, f'No significant\nsaliency for\n{group_name}', 
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
            x, y, z = np.indices(np.array(spatial_saliency.shape) + 1)
            
            # Create colormap instance
            try:
                colormap = plt.colormaps[color_scheme['cmap']]
            except (AttributeError, KeyError):
                try:
                    colormap = plt.colormaps.get_cmap(color_scheme['cmap'])
                except AttributeError:
                    colormap = cm.get_cmap(color_scheme['cmap'])
            
            # Normalize saliency values to 0-1 range
            norm_min, norm_max = p_min, p_max
            
            if norm_max > norm_min:
                normalized_saliency = (spatial_saliency - norm_min) / (norm_max - norm_min)
                normalized_saliency = np.clip(normalized_saliency, 0, 1)
            else:
                normalized_saliency = np.zeros_like(spatial_saliency)
            
            display_values = normalized_saliency[display_mask]
            
            print(f"      Original saliency range: [{norm_min:.8f}, {norm_max:.8f}]")
            print(f"      Normalized range: [0.0, 1.0]")
            print(f"      Number of displayed voxels: {len(display_values)}")
            print(f"      Displayed normalized saliency range: [{display_values.min():.3f}, {display_values.max():.3f}]")
            print(f"      88th percentile threshold: {np.percentile(spatial_saliency, 88):.8f}")
            print(f"      Max normalized saliency: {normalized_saliency.max():.6f}")
            print(f"      Voxels above threshold: {np.sum(display_mask)} out of {spatial_saliency.size}")
            if len(display_values) > 0:
                print(f"      Color values after power transform: [{(display_values.min() ** 0.7):.3f}, {(display_values.max() ** 0.7):.3f}]")
            
            if norm_max > norm_min:
                # Apply color power transformation for better visualization
                # Use different power for adsorbate to make it more visible
                if 'Adsorbate' in group_name:
                    color_power = 0.5  # Lower power makes colors darker/more saturated
                    print(f"      Using lower color_power={color_power} for adsorbate visibility")
                else:
                    color_power = 0.7  # Standard power for solvent
                color_normalized = normalized_saliency ** color_power
                
                # Set alpha values for saliency visualization
                if 'Adsorbate' in group_name:
                    alpha_min, alpha_max, alpha_power = 0.3, 0.9, 1.5  # Higher opacity for adsorbate saliency visibility
                else:
                    alpha_min, alpha_max, alpha_power = 0.1, 0.9, 1.5   # Standard alpha for solvent saliency
                
                # Create RGBA colors for each voxel
                colors = np.zeros(spatial_saliency.shape + (4,))
                coords = np.where(display_mask)
                
                for i_coord, j_coord, k_coord in zip(*coords):
                    color_value = color_normalized[i_coord, j_coord, k_coord]
                    rgba_color = colormap(color_value)
                    
                    # Enhanced saturation for high-saliency voxels (lower threshold for adsorbate)
                    color_threshold = 0.3 if color_scheme['cmap'] == 'Reds' else 0.7  # Even lower threshold for adsorbate
                    if color_value > color_threshold:
                        r, g, b = rgba_color[0], rgba_color[1], rgba_color[2]
                        min_component = min(r, g, b)
                        # Higher enhancement for adsorbate (red) to improve visibility
                        enhancement_factor = 0.7 if color_scheme['cmap'] == 'Reds' else 0.3  # Stronger enhancement for reds
                        
                        if color_scheme['cmap'] == 'Reds':
                            r = min(1.0, r + enhancement_factor * (1 - min_component))
                        elif color_scheme['cmap'] == 'Blues':
                            b = min(1.0, b + enhancement_factor * (1 - min_component))
                        
                        rgba_color = (r, g, b, rgba_color[3])
                    
                    # Additional boost for adsorbate colors to ensure visibility
                    if color_scheme['cmap'] == 'Reds':
                        # Apply minimum color intensity for adsorbate
                        r, g, b = rgba_color[0], rgba_color[1], rgba_color[2]
                        r = max(r, 0.3 + 0.7 * r)  # Ensure red component is at least 0.3
                        rgba_color = (r, g, b, rgba_color[3])
                    
                    # Calculate alpha based on normalized saliency
                    norm_val = normalized_saliency[i_coord, j_coord, k_coord]
                    alpha = alpha_min + (alpha_max - alpha_min) * (norm_val ** alpha_power)
                    
                    colors[i_coord, j_coord, k_coord] = [rgba_color[0], rgba_color[1], rgba_color[2], alpha]
                
                # Plot voxels with individual colors (edgecolors=colors to remove white edges)
                ax.voxels(x, y, z, display_mask, facecolors=colors, edgecolors=colors, linewidth=0)
                
                # Create colorbar
                sm = cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=30, pad=0.02)
                
                # Set colorbar ticks to show 0-1 normalized values
                cbar_ticks = np.linspace(0, 1, 6)
                cbar_labels = [f"{tick:.1f}" for tick in cbar_ticks]
                cbar.set_ticks(cbar_ticks)
                cbar.set_ticklabels(cbar_labels)
                cbar.set_label(f'{color_scheme["name"]} Attribution', fontsize=self.font_size, color='black')
                cbar.ax.tick_params(labelsize=self.font_size)
            
            # Set axis properties
            # Set axis properties - use negative pad to bring tick labels closer to axes
            ax.tick_params(axis='x', labelsize=self.font_size, pad=6)
            ax.tick_params(axis='y', labelsize=self.font_size, pad=6)
            ax.tick_params(axis='z', labelsize=self.font_size, pad=6)
            
            # Set equal aspect ratio
            shape = np.array(spatial_saliency.shape)
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
            
            # Add axis labels
            ax.set_xlabel('x (voxel)', fontsize=self.font_size, labelpad=12)
            ax.set_ylabel('y (voxel)', fontsize=self.font_size, labelpad=12)
            ax.set_zlabel('z (voxel)', fontsize=self.font_size, labelpad=12)
            
            # Add subplot label (a) or (b)
            subplot_labels = ['(a)', '(b)']
            ax.text2D(0.00, 1.00, subplot_labels[i], fontsize=self.font_size+2, fontweight='bold',
                     transform=ax.transAxes, verticalalignment='top', horizontalalignment='left')
        
        plt.tight_layout()
        
        print(f"\n✓ 3D Captum saliency map from input voxel layer generated")
        
        # Save plot
        if save_plots:
            sample_prefix = f"{self.model_prefix}-{sample_info['zeolite']}-{sample_info['environment']}-{sample_info['adsorbate']}-snap{sample_info['snapshot']}-vox{sample_info['voxel_id']}"
            save_path = os.path.join(self.output_dir, f"3d_captum_saliency_input-{sample_prefix}.png")
            fig.savefig(save_path, dpi=1000, bbox_inches='tight')
            print(f"  ✓ 3D Captum saliency map (input layer) saved: {save_path}")
        
        if show_plots:
            plt.show()
        else:
            # Keep figure open for Spyder plots panel when show_plots=False
            pass
            
        return fig

    def plot_3d_saliency_map_processor_output_layer(self, analysis_results, show_plots=True, save_plots=True):
        """Generate 3D Captum saliency map from processor branch output layers (adsorbate/solvent/interaction saliency)"""
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
        
        print(f"\n=== Generating 3D Captum Saliency Map from Processor Output Layers ===")
        print(f"Available multi-level data: {list(multilevel_data.keys())}")
        
        # Process available spatial saliency data
        saliency_data = {}
        available_levels = []
        
        if 'adsorbate_spatial' in multilevel_data:
            adsorbate_spatial = multilevel_data['adsorbate_spatial']
            if len(adsorbate_spatial.shape) == 3:  # Ensure it's 3D (D, H, W)
                # Convert to saliency (absolute values for significance)
                saliency_data['Adsorbate Branch'] = np.abs(adsorbate_spatial)
                available_levels.append('Adsorbate Branch')
                print(f"    Adsorbate Branch Saliency: shape {np.abs(adsorbate_spatial).shape}, range [{np.abs(adsorbate_spatial).min():.6f}, {np.abs(adsorbate_spatial).max():.6f}]")
            else:
                print(f"      Warning: adsorbate_spatial has unexpected shape {adsorbate_spatial.shape}")
            
        if 'solvent_spatial' in multilevel_data:
            solvent_spatial = multilevel_data['solvent_spatial']
            if len(solvent_spatial.shape) == 3:  # Ensure it's 3D (D, H, W)
                # Convert to saliency (absolute values for significance)
                saliency_data['Solvent Branch'] = np.abs(solvent_spatial)
                available_levels.append('Solvent Branch')
                print(f"    Solvent Branch Saliency: shape {np.abs(solvent_spatial).shape}, range [{np.abs(solvent_spatial).min():.6f}, {np.abs(solvent_spatial).max():.6f}]")
            else:
                print(f"      Warning: solvent_spatial has unexpected shape {solvent_spatial.shape}")
            
        if 'interaction_spatial' in multilevel_data:
            interaction_spatial = multilevel_data['interaction_spatial']
            if len(interaction_spatial.shape) == 3:  # Ensure it's 3D (D, H, W)
                # Convert to saliency (absolute values for significance)
                saliency_data['Interaction Layer'] = np.abs(interaction_spatial)
                available_levels.append('Interaction Layer')
                print(f"    Interaction Layer Saliency: shape {np.abs(interaction_spatial).shape}, range [{np.abs(interaction_spatial).min():.6f}, {np.abs(interaction_spatial).max():.6f}]")
            else:
                print(f"      Warning: interaction_spatial has unexpected shape {interaction_spatial.shape}")
            
        if len(available_levels) == 0:
            print("No spatial saliency data found in multi-level results")
            return None
        
        # Define color schemes for different branches (saliency-focused)
        color_schemes = {
            'Adsorbate Branch': {'cmap': 'Reds', 'name': 'Adsorbate'},
            'Solvent Branch': {'cmap': 'Blues', 'name': 'Solvent'}, 
            'Interaction Layer': {'cmap': 'Greens', 'name': 'Interaction'}
        }
        
        # Create subplots for each available level (same layout as original)
        n_levels = len(available_levels)
        
        if n_levels == 1:
            fig = plt.figure(figsize=(10, 8))
            axes = [fig.add_subplot(111, projection='3d')]
        elif n_levels == 2:
            fig = plt.figure(figsize=(20, 8))
            axes = [fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d')]
            plt.subplots_adjust(wspace=0.1)
        else:  # 3 levels
            fig = plt.figure(figsize=(24, 8))
            axes = [fig.add_subplot(131, projection='3d'), fig.add_subplot(132, projection='3d'), fig.add_subplot(133, projection='3d')]
            plt.subplots_adjust(wspace=0.05)
        
        fig.patch.set_facecolor('white')
        
        for i, level_name in enumerate(available_levels):
            spatial_saliency = saliency_data[level_name]  # Already converted to saliency (absolute values)
            color_scheme = color_schemes.get(level_name, color_schemes['Adsorbate Branch'])
            ax = axes[i]
            
            print(f"\n    Processing {level_name} Saliency...")
            
            # Use percentile-based normalization to handle outliers
            p_min, p_max = np.percentile(spatial_saliency, [1, 99])
            
            # Adjust display threshold based on branch type for saliency visualization
            if 'Adsorbate' in level_name:
                display_mask = spatial_saliency > np.percentile(spatial_saliency, 95)  # Top 5% for adsorbate saliency
                print(f"      Using higher threshold (95th percentile) for adsorbate saliency visibility")
            else:
                display_mask = spatial_saliency > np.percentile(spatial_saliency, 85)  # Top 15% for other saliency
                print(f"      Using standard threshold (85th percentile) for {level_name.lower()} saliency")

            print(f"      Using percentile normalization: [{p_min:.8f}, {p_max:.8f}]")
            
            if not np.any(display_mask):
                print(f"      No voxels above cutoff for {level_name} saliency, skipping...")
                ax.text(0.5, 0.5, 0.5, f'No significant\nsaliency for\n{level_name}', 
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
            x, y, z = np.indices(np.array(spatial_saliency.shape) + 1)
            
            # Create colormap instance
            try:
                colormap = plt.colormaps[color_scheme['cmap']]
            except (AttributeError, KeyError):
                try:
                    colormap = plt.colormaps.get_cmap(color_scheme['cmap'])
                except AttributeError:
                    colormap = cm.get_cmap(color_scheme['cmap'])
            
            # Normalize saliency values to 0-1 range
            norm_min, norm_max = p_min, p_max
            
            if norm_max > norm_min:
                normalized_saliency = (spatial_saliency - norm_min) / (norm_max - norm_min)
                normalized_saliency = np.clip(normalized_saliency, 0, 1)
            else:
                normalized_saliency = np.zeros_like(spatial_saliency)
            
            display_values = normalized_saliency[display_mask]
            
            print(f"      Original saliency range: [{norm_min:.8f}, {norm_max:.8f}]")
            print(f"      Normalized range: [0.0, 1.0]")
            print(f"      Number of displayed voxels: {len(display_values)}")
            print(f"      Displayed normalized saliency range: [{display_values.min():.3f}, {display_values.max():.3f}]")
            
            if norm_max > norm_min:
                # Apply color power transformation for saliency
                color_power = 0.7
                
                # Set different alpha values for adsorbate branch (higher opacity for saliency)
                if level_name == 'Adsorbate Branch':
                    alpha_min, alpha_max, alpha_power = 0.2, 0.7, 1.5  # Higher opacity for adsorbate saliency
                else:
                    alpha_min, alpha_max, alpha_power = 0.1, 0.9, 1.5   # Standard alpha for other saliency
                
                color_normalized = normalized_saliency ** color_power
                
                # Create RGBA colors for each voxel
                colors = np.zeros(spatial_saliency.shape + (4,))
                coords = np.where(display_mask)
                
                for i_coord, j_coord, k_coord in zip(*coords):
                    color_value = color_normalized[i_coord, j_coord, k_coord]
                    rgba_color = colormap(color_value)
                    
                    # Enhanced saturation for high-saliency voxels
                    if color_value > 0.7:
                        r, g, b = rgba_color[0], rgba_color[1], rgba_color[2]
                        min_component = min(r, g, b)
                        enhancement_factor = 0.3
                        
                        if color_scheme['cmap'] == 'Reds':
                            r = min(1.0, r + enhancement_factor * (1 - min_component))
                        elif color_scheme['cmap'] == 'Blues':
                            b = min(1.0, b + enhancement_factor * (1 - min_component))
                        elif color_scheme['cmap'] == 'Greens':
                            g = min(1.0, g + enhancement_factor * (1 - min_component))
                        
                        rgba_color = (r, g, b, rgba_color[3])
                    
                    # Calculate alpha based on normalized saliency
                    norm_val = normalized_saliency[i_coord, j_coord, k_coord]
                    alpha = alpha_min + (alpha_max - alpha_min) * (norm_val ** alpha_power)
                    
                    colors[i_coord, j_coord, k_coord] = [rgba_color[0], rgba_color[1], rgba_color[2], alpha]
                
                # Plot voxels with individual colors (edgecolors=colors to remove white edges)
                ax.voxels(x, y, z, display_mask, facecolors=colors, edgecolors=colors, linewidth=0)
                
                # Create colorbar
                sm = cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.02)
                
                # Set colorbar ticks
                cbar_ticks = np.linspace(0, 1, 6)
                cbar_labels = [f"{tick:.1f}" for tick in cbar_ticks]
                cbar.set_ticks(cbar_ticks)
                cbar.set_ticklabels(cbar_labels)
                cbar.set_label(f'{level_name} Attribution', fontsize=self.font_size, color='black')
                cbar.ax.tick_params(labelsize=self.font_size)
            
            # Set axis properties
            ax.tick_params(axis='x', labelsize=self.font_size)
            ax.tick_params(axis='y', labelsize=self.font_size)
            ax.tick_params(axis='z', labelsize=self.font_size)
            
            # Set equal aspect ratio
            shape = np.array(spatial_saliency.shape)
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
            
            # Add subplot label (a), (b), or (c)
            subplot_labels = ['(a)', '(b)', '(c)']
            ax.text2D(0.0, 1.0, subplot_labels[i], fontsize=self.font_size+2, fontweight='bold',
                     transform=ax.transAxes, verticalalignment='top', horizontalalignment='left')
        
        plt.tight_layout()
        
        # Print summary
        print(f"\n✓ 3D Captum saliency map from processor output layers generated")
        for level_name in available_levels:
            spatial_saliency = saliency_data[level_name]
            print(f"    {level_name}: max_saliency={spatial_saliency.max():.6f}, mean_saliency={spatial_saliency.mean():.6f}")
        
        # Save plot
        if save_plots:
            save_path = os.path.join(self.output_dir, f"3d_captum_saliency_processor-{sample_prefix}.png")
            fig.savefig(save_path, dpi=1000, bbox_inches='tight', facecolor='white')
            print(f"  ✓ 3D Captum saliency map (processor layer) saved: {save_path}")
        
        if show_plots:
            plt.show()
        else:
            # Keep figure open for Spyder plots panel when show_plots=False
            pass
            
        return fig

    
    def _get_layer_by_name(self, model, layer_name):
        """Helper function to get layer by name from model"""
        try:
            # Handle nested layer access
            if '.' in layer_name:
                parts = layer_name.split('.')
                layer = model
                for part in parts:
                    layer = getattr(layer, part)
                return layer
            else:
                return getattr(model, layer_name)
        except AttributeError:
            return None
    
    def _plot_input_stage_analysis(self, ax, layer_attributions, layer_config):
        """Plot input stage analysis (voxel importance distribution)"""
        if 'input_voxels' not in layer_attributions:
            ax.text(0.5, 0.5, 'Input data not available', ha='center', va='center', transform=ax.transAxes)
            return
            
        input_data = layer_attributions['input_voxels']['attribution']
        
        # Channel-wise importance analysis
        channel_importance = np.mean(np.abs(input_data), axis=(1, 2, 3))
        
        # Split into adsorbate (0-13) and solvent (14-27) channels
        adsorbate_importance = channel_importance[:14]
        solvent_importance = channel_importance[14:]
        
        x_ads = np.arange(14)
        x_solv = np.arange(14, 28)
        
        bars1 = ax.bar(x_ads, adsorbate_importance, alpha=0.7, color='red', label='Adsorbate Channels (0-13)')
        bars2 = ax.bar(x_solv, solvent_importance, alpha=0.7, color='blue', label='Solvent Channels (14-27)')
        
        ax.set_title('Input Stage: Channel Importance Distribution', fontsize=self.font_size, fontweight='bold')
        ax.set_xlabel('Channel Index', fontsize=self.font_size)
        ax.set_ylabel('Mean Importance', fontsize=self.font_size)
        ax.legend(fontsize=self.font_size)
        ax.tick_params(axis='both', labelsize=self.font_size)
        
        # Add statistics
        ads_total = np.sum(adsorbate_importance)
        solv_total = np.sum(solvent_importance)
        ax.text(0.02, 0.98, f'Adsorbate: {ads_total:.3f}\nSolvent: {solv_total:.3f}\nRatio: {solv_total/ads_total:.2f}', 
                transform=ax.transAxes, fontsize=self.font_size, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def _plot_processing_flow_analysis(self, ax, layer_attributions, layer_config):
        """Plot main processing flow with information preservation tracking"""
        
        # Define flow stages in processing order
        flow_stages = ['input_voxels', 'adsorbate_processor', 'solvent_processor', 
                      'interaction_conv', 'layer1', 'layer2', 'layer3', 'adaptive_pool', 'classifier']
        
        available_stages = [stage for stage in flow_stages if stage in layer_attributions]
        
        if len(available_stages) < 2:
            ax.text(0.5, 0.5, 'Insufficient layer data for flow analysis', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Extract flow data
        stage_names = []
        total_importance = []
        channel_counts = []
        
        for stage in available_stages:
            data = layer_attributions[stage]
            config = layer_config[stage]
            
            stage_names.append(stage.replace('_', '\n'))
            total_importance.append(data['total_importance'])
            channel_counts.append(data['channels'])
        
        x_positions = np.arange(len(stage_names))
        
        # Create dual y-axis plot
        ax2 = ax.twinx()
        
        # Plot total importance (line with markers)
        line1 = ax.plot(x_positions, total_importance, 'o-', color='darkgreen', linewidth=3, markersize=8, 
                       label='Total Importance Flow')
        
        # Plot channel count evolution (bar chart)
        bars = ax2.bar(x_positions, channel_counts, alpha=0.3, color='orange', label='Channel Count')
        
        # Styling
        ax.set_xlabel('CNN Processing Stages', fontsize=self.font_size)
        ax.set_ylabel('Total Importance', fontsize=self.font_size, color='darkgreen')
        ax2.set_ylabel('Number of Channels', fontsize=self.font_size, color='orange')
        
        ax.set_title('Complete CNN Information Flow: Input → Dual Processors → Fusion → Residual → Output', 
                    fontsize=self.font_size, fontweight='bold')
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(stage_names, fontsize=self.font_size, rotation=45, ha='right')
        
        ax.tick_params(axis='y', labelcolor='darkgreen', labelsize=self.font_size)
        ax2.tick_params(axis='y', labelcolor='orange', labelsize=self.font_size)
        
        # Add legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=self.font_size)
        
        # Add information preservation percentages
        if len(total_importance) > 1:
            baseline = total_importance[0]
            for i, (x_pos, importance) in enumerate(zip(x_positions[1:], total_importance[1:]), 1):
                preservation = (importance / baseline * 100) if baseline > 0 else 0
                ax.annotate(f'{preservation:.0f}%', (x_pos, importance), 
                           textcoords="offset points", xytext=(0,10), ha='center',
                           fontsize=self.font_size, color='darkgreen', weight='bold')
    
    def _plot_output_stage_analysis(self, ax, layer_attributions, layer_config):
        """Plot output stage analysis (final prediction composition)"""
        
        # Get the last few stages for output analysis
        output_stages = ['layer3', 'adaptive_pool', 'classifier']
        available_output_stages = [stage for stage in output_stages if stage in layer_attributions]
        
        if not available_output_stages:
            ax.text(0.5, 0.5, 'Output stage data not available', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Create output importance progression
        stage_names = []
        mean_importance = []
        
        for stage in available_output_stages:
            data = layer_attributions[stage]
            stage_names.append(stage.replace('_', ' ').title())
            mean_importance.append(data['mean_importance'])
        
        # Plot as horizontal bar chart for better readability
        y_positions = np.arange(len(stage_names))
        bars = ax.barh(y_positions, mean_importance, color=['lightcoral', 'lightblue', 'lightgreen'][:len(mean_importance)])
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(stage_names, fontsize=self.font_size)
        ax.set_xlabel('Mean Feature Importance', fontsize=self.font_size)
        ax.set_title('Output Stage: Feature Importance Evolution', fontsize=self.font_size, fontweight='bold')
        ax.tick_params(axis='both', labelsize=self.font_size)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, mean_importance)):
            ax.text(value, i, f'{value:.3f}', va='center', ha='left' if value > max(mean_importance)*0.1 else 'right',
                   fontsize=self.font_size, weight='bold')
    
    def _plot_flow_statistics(self, ax, layer_attributions, layer_config):
        """Plot comprehensive flow statistics and architecture summary"""
        
        ax.axis('off')  # Remove axes for text-based summary
        
        # Compile statistics
        stats_text = []
        stats_text.append("🏗️ ARCHITECTURE SUMMARY")
        stats_text.append("-" * 25)
        
        # Input stage
        if 'input_voxels' in layer_attributions:
            input_data = layer_attributions['input_voxels']
            stats_text.append(f"📥 INPUT STAGE")
            stats_text.append(f"   • Channels: {input_data['channels']}")
            stats_text.append(f"   • Total Importance: {input_data['total_importance']:.3f}")
            stats_text.append("")
        
        # Processing stages
        processing_stages = ['adsorbate_processor', 'solvent_processor', 'interaction_conv']
        available_processing = [s for s in processing_stages if s in layer_attributions]
        
        if available_processing:
            stats_text.append(f"🔄 PROCESSING STAGES")
            for stage in available_processing:
                data = layer_attributions[stage]
                config = layer_config[stage]
                stats_text.append(f"   • {stage.replace('_', ' ').title()}:")
                stats_text.append(f"     - Channels: {data['channels']}")
                stats_text.append(f"     - Importance: {data['total_importance']:.3f}")
            stats_text.append("")
        
        # Residual stages
        residual_stages = ['layer1', 'layer2', 'layer3']
        available_residual = [s for s in residual_stages if s in layer_attributions]
        
        if available_residual:
            stats_text.append(f"🔗 RESIDUAL BLOCKS")
            for stage in available_residual:
                data = layer_attributions[stage]
                stats_text.append(f"   • {stage.title()}: {data['channels']} ch")
                stats_text.append(f"     - Importance: {data['total_importance']:.3f}")
            stats_text.append("")
        
        # Final stages
        final_stages = ['adaptive_pool', 'classifier']
        available_final = [s for s in final_stages if s in layer_attributions]
        
        if available_final:
            stats_text.append(f"📤 OUTPUT STAGES")
            for stage in available_final:
                data = layer_attributions[stage]
                stats_text.append(f"   • {stage.replace('_', ' ').title()}:")
                stats_text.append(f"     - Channels: {data['channels']}")
                stats_text.append(f"     - Importance: {data['total_importance']:.3f}")
            stats_text.append("")
        
        # Overall statistics
        if len(layer_attributions) > 1:
            first_importance = list(layer_attributions.values())[0]['total_importance']
            last_importance = list(layer_attributions.values())[-1]['total_importance']
            preservation = (last_importance / first_importance * 100) if first_importance > 0 else 0
            
            stats_text.append(f"📊 FLOW ANALYSIS")
            stats_text.append(f"   • Stages Analyzed: {len(layer_attributions)}")
            stats_text.append(f"   • Info Preservation: {preservation:.1f}%")
            stats_text.append(f"   • Processing Path: Dual→Fusion→Residual")
        
        # Display all statistics
        full_text = "\n".join(stats_text)
        ax.text(0.05, 0.95, full_text, transform=ax.transAxes, fontsize=self.font_size,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))


## SIMPLIFIED MAIN FUNCTION
if __name__ == "__main__":
    
    # Model results file
    results_filename = "model.pkl" # Test MAE 0.083 # Best
    
    # Initialize analyzer and run analysis
    analyzer = ImportanceMapAnalyzer(results_filename=results_filename,
                                     font_size=22,
                                     )
    
    ## Define the simulation parameters
    zeolite_type = 'FAU'           # e.g. "FAU", "BEA" or "MFI"
    solvent_type = 'water_pure'    # e.g. "water_pure", "methanol_120_water_1080", "methanol_240_water_960", "methanol_600_water_600"
    pore_type = 'hydrophilic'      # e.g. "hydrophilic", "hydrophobic"
    adsorbate = '11_01_propylene_glycol'      # e.g. "01_methanol", "02_01_02_propanol"
    snapshot = 6
    voxel_id = 1
    
    # Analyze the specified sample using Captum with ensemble of all train folds
    analysis_results = analyzer.analyze_sample(
                                            zeolite=zeolite_type,
                                            env_adsorbate=solvent_type + '-' + pore_type + '-' + adsorbate,
                                            snapshot=snapshot,
                                            voxel_id=voxel_id,
                                            steps=100,
                                            ensemble_method='mean',  # Average across train folds
                                            use_comprehensive_analysis=False  # Set to True to use all Captum methods
                                        )


    # # Generate 2D Captum saliency map from input voxel layer
    # print(f"\n--- Generating 2D Captum Saliency Map from Input Voxel Layer ---")
    # analyzer.plot_2d_saliency_map_input_voxel_layer(analysis_results, show_plots=False, save_plots=True)
    
    # # Generate 2D Captum saliency map from processor output layers
    # print(f"\n--- Generating 2D Captum Saliency Map from Processor Output Layers ---")
    # analyzer.plot_2d_saliency_map_processor_output_layer(analysis_results, show_plots=False, save_plots=True)
    
    # # Generate 3D Captum saliency map from input voxel layer
    # print(f"\n--- Generating 3D Captum Saliency Map from Input Voxel Layer ---")
    analyzer.plot_3d_saliency_map_input_voxel_layer(analysis_results, show_plots=True, save_plots=False)
    
    # # # Generate 3D Captum saliency map from processor output layers
    # # print(f"\n--- Generating 3D Captum Saliency Map from Processor Output Layers ---")
    # analyzer.plot_3d_saliency_map_processor_output_layer(analysis_results, show_plots=False, save_plots=True)
