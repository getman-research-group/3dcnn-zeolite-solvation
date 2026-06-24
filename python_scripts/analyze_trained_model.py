
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_trained_model.py
Comprehensive analysis of trained 3D CNN models for optimization insights.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
from datetime import datetime
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
from copy import deepcopy

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.path import get_paths
from model_3d_cnn_2_7 import AttentionCNN_2_7
from model_3d_cnn_2_8 import AttentionCNN_2_8


class TrainedModelAnalyzer:
    """Comprehensive analyzer for trained 3D CNN models"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.checkpoint = None
        self.analysis_results = {}
        
        print(f"🔬 Initializing model analysis for: {os.path.basename(model_path)}")
        
    def load_model(self):
        """Load the trained model with error handling and compatibility fixes"""
        print(f"📂 Loading model from: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            print(f"❌ Model file not found: {self.model_path}")
            return False
        
        try:
            # Load checkpoint with weights_only=False for older PyTorch versions
            self.checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
            
            # Extract model version and analyze checkpoint structure
            model_state = self.checkpoint.get('model_state_dict', self.checkpoint)
            model_file = os.path.basename(self.model_path)
            
            if 'model_2_8' in model_file:
                print("🔧 Detected Model 2_8 version")
                success = self._load_model_2_8_compatible(model_state)
                self.model_version = "2_8"
            elif 'model_2_7' in model_file:
                print("🔧 Detected Model 2_7 version")  
                from model_3d_cnn_2_7 import AttentionCNN_2_7
                self.model = AttentionCNN_2_7(in_channels=28, dropout_rate=0.1)
                self.model.load_state_dict(model_state)
                self.model_version = "2_7"
                success = True
            else:
                print("⚠️ Unknown model version, attempting 2_8 with compatibility")
                success = self._load_model_2_8_compatible(model_state)
                self.model_version = "2_8"
            
            if success:
                self.model.eval()
                print("✅ Model loaded successfully")
                return True
            else:
                print("❌ Failed to load model")
                return False
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def _load_model_2_8_compatible(self, model_state):
        """Load model_2_8 with compatibility adjustments for different trained versions"""
        try:
            from model_3d_cnn_2_8 import AttentionCNN_2_8
            
            # Analyze the checkpoint structure to understand the trained architecture
            layer2_shape = model_state.get('layer2.conv1.weight', torch.tensor([])).shape
            regressor_keys = [k for k in model_state.keys() if k.startswith('regressor.')]
            
            print(f"🔍 Analyzing checkpoint structure:")
            if len(layer2_shape) > 0:
                print(f"   - layer2.conv1.weight shape: {layer2_shape}")
                trained_layer2_channels = layer2_shape[0]  # Output channels
            else:
                trained_layer2_channels = 56  # Default guess
                
            print(f"   - Regressor layers: {len(regressor_keys)}")
            print(f"   - Trained layer2 output channels: {trained_layer2_channels}")
            
            # Create model instance
            self.model = AttentionCNN_2_8(in_channels=28, dropout_rate=0.25)
            
            # Try direct loading first
            try:
                self.model.load_state_dict(model_state, strict=False)
                print("✅ Direct loading successful (with non-strict mode)")
                return True
            except Exception as direct_error:
                print(f"⚠️ Direct loading failed: {direct_error}")
                
            # If direct loading fails, try compatible loading with adjustments
            print("🔧 Attempting compatible loading with architecture adjustments...")
            
            # Load compatible layers only
            current_state = self.model.state_dict()
            updated_state = {}
            
            for key, param in model_state.items():
                if key in current_state:
                    current_param = current_state[key]
                    if param.shape == current_param.shape:
                        updated_state[key] = param
                        print(f"✓ Loaded: {key}")
                    else:
                        print(f"⚠️ Shape mismatch for {key}: {param.shape} vs {current_param.shape}")
                        # Try to adapt parameters for ALL mismatched layers
                        adapted_param = self._adapt_layer_parameters(param, current_param, key)
                        if adapted_param is not None:
                            updated_state[key] = adapted_param
                            print(f"✓ Adapted and loaded: {key}")
                        else:
                            print(f"✗ Could not adapt: {key}")
                else:
                    print(f"✗ Key not found in current model: {key}")
            
            # Load the compatible parameters
            self.model.load_state_dict(updated_state, strict=False)
            
            # Verify loading
            missing_keys = set(current_state.keys()) - set(updated_state.keys())
            if missing_keys:
                print(f"⚠️ Missing keys (will use random initialization): {len(missing_keys)} keys")
                for key in list(missing_keys)[:5]:  # Show first 5
                    print(f"   - {key}")
                if len(missing_keys) > 5:
                    print(f"   - ... and {len(missing_keys) - 5} more")
            
            print("✅ Compatible loading completed")
            return True
            
        except Exception as e:
            print(f"❌ Compatible loading failed: {e}")
            return False
    
    def _adapt_layer_parameters(self, trained_param, current_param, layer_name):
        """Adapt parameters between different layer configurations"""
        try:
            if trained_param.shape == current_param.shape:
                return trained_param
            
            # Handle different channel dimensions for conv layers
            if 'conv' in layer_name and len(trained_param.shape) >= 2:
                # For conv layers: (out_channels, in_channels, ...)
                trained_out, trained_in = trained_param.shape[0], trained_param.shape[1]
                current_out, current_in = current_param.shape[0], current_param.shape[1]
                
                print(f"   Adapting {layer_name}: ({trained_out}, {trained_in}) -> ({current_out}, {current_in})")
                
                # Handle both input and output channel mismatches
                adapted_param = trained_param.clone()
                
                # Adjust output channels
                if trained_out != current_out:
                    if trained_out < current_out:
                        # Pad output channels
                        pad_size = list(trained_param.shape)
                        pad_size[0] = current_out - trained_out
                        pad_tensor = torch.randn(pad_size) * trained_param.std()
                        adapted_param = torch.cat([adapted_param, pad_tensor], dim=0)
                        print(f"   ✓ Padded output channels: {trained_out} -> {current_out}")
                    else:
                        # Truncate output channels
                        adapted_param = adapted_param[:current_out]
                        print(f"   ✓ Truncated output channels: {trained_out} -> {current_out}")
                
                # Adjust input channels if still mismatch
                if adapted_param.shape[1] != current_in:
                    if adapted_param.shape[1] < current_in:
                        # Pad input channels
                        pad_size = list(adapted_param.shape)
                        pad_size[1] = current_in - adapted_param.shape[1]
                        pad_tensor = torch.randn(pad_size) * adapted_param.std()
                        adapted_param = torch.cat([adapted_param, pad_tensor], dim=1)
                        print(f"   ✓ Padded input channels: {adapted_param.shape[1] - pad_size[1]} -> {current_in}")
                    else:
                        # Truncate input channels
                        adapted_param = adapted_param[:, :current_in]
                        print(f"   ✓ Truncated input channels: {trained_param.shape[1]} -> {current_in}")
                
                return adapted_param
                        
            # Handle BatchNorm and other 1D parameters
            elif len(trained_param.shape) == 1:  # 1D parameter (weight, bias, running_mean, running_var)
                trained_size = trained_param.shape[0]
                current_size = current_param.shape[0]
                
                if trained_size != current_size:
                    if trained_size < current_size:
                        # Pad with appropriate values
                        if 'weight' in layer_name:
                            pad_tensor = torch.ones(current_size - trained_size)
                        elif 'running_var' in layer_name:
                            pad_tensor = torch.ones(current_size - trained_size)  # Variance should be 1
                        else:  # bias, running_mean
                            pad_tensor = torch.zeros(current_size - trained_size)
                        adapted_param = torch.cat([trained_param, pad_tensor])
                        print(f"   ✓ Padded BN parameter: {trained_size} -> {current_size}")
                        return adapted_param
                    else:
                        # Truncate
                        adapted_param = trained_param[:current_size]
                        print(f"   ✓ Truncated BN parameter: {trained_size} -> {current_size}")
                        return adapted_param
            
            return None  # Cannot adapt
            
        except Exception as e:
            print(f"   ❌ Parameter adaptation failed for {layer_name}: {e}")
            return None
    
    def extract_checkpoint_metadata(self):
        """Extract training metadata from checkpoint"""
        metadata = {}
        
        if isinstance(self.checkpoint, dict):
            # Standard checkpoint structure
            metadata_keys = [
                'fold_idx', 'epoch', 'train_rmse', 'train_r2', 'train_mae',
                'test_rmse', 'test_r2', 'test_mae', 'train_loss', 'test_loss',
                'learning_rate', 'batch_size', 'total_epochs', 'early_stopped'
            ]
            
            for key in metadata_keys:
                if key in self.checkpoint:
                    metadata[key] = self.checkpoint[key]
            
            # Training metrics
            if 'train_metrics' in self.checkpoint:
                metadata['train_metrics'] = self.checkpoint['train_metrics']
            if 'test_metrics' in self.checkpoint:
                metadata['test_metrics'] = self.checkpoint['test_metrics']
            
            # Model configuration
            if 'model_config' in self.checkpoint:
                metadata['model_config'] = self.checkpoint['model_config']
                
        return metadata
    
    def analyze_architecture_efficiency(self):
        """Analyze model architecture and parameter efficiency"""
        print("\n🏗️ Analyzing Architecture Efficiency...")
        
        analysis = {}
        
        # Parameter analysis
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        analysis['parameter_stats'] = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_efficiency': trainable_params / total_params
        }
        
        # Layer-wise parameter distribution
        layer_params = {}
        layer_categories = {
            'adsorbate_branch': 0,
            'solvent_branch': 0,
            'interaction_layers': 0,
            'cnn_backbone': 0,
            'cbam_attention': 0,
            'regressor': 0,
            'other': 0
        }
        
        for name, param in self.model.named_parameters():
            param_count = param.numel()
            layer_params[name] = param_count
            
            # Categorize parameters
            if 'adsorbate_processor' in name:
                layer_categories['adsorbate_branch'] += param_count
            elif 'solvent_processor' in name:
                layer_categories['solvent_branch'] += param_count
            elif 'interaction_attention' in name:
                layer_categories['interaction_layers'] += param_count
            elif 'layer' in name and 'cbam' not in name:
                layer_categories['cnn_backbone'] += param_count
            elif 'cbam' in name or 'attention' in name:
                layer_categories['cbam_attention'] += param_count
            elif 'regressor' in name:
                layer_categories['regressor'] += param_count
            else:
                layer_categories['other'] += param_count
        
        analysis['layer_distribution'] = layer_categories
        analysis['layer_percentages'] = {
            k: (v / total_params) * 100 for k, v in layer_categories.items()
        }
        
        return analysis
    
    def analyze_weight_patterns(self):
        """Analyze learned weight patterns and distributions"""
        print("\n⚖️ Analyzing Weight Patterns...")
        
        analysis = {}
        
        # Weight statistics per layer type
        weight_stats = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                weights = param.data.cpu().numpy()
                weights_flat = weights.flatten()
                
                weight_stats[name] = {
                    'mean': float(np.mean(weights_flat)),
                    'std': float(np.std(weights_flat)),
                    'min': float(np.min(weights_flat)),
                    'max': float(np.max(weights_flat)),
                    'abs_mean': float(np.mean(np.abs(weights_flat))),
                    'sparsity': float(np.sum(np.abs(weights_flat) < 1e-6) / len(weights_flat)),
                    'norm': float(np.linalg.norm(weights_flat)),
                    'shape': list(weights.shape),
                    'param_count': int(weights.size),
                    'effective_rank': self._compute_effective_rank(weights) if len(weights.shape) >= 2 else None,
                    'dead_neurons_ratio': self._compute_dead_neurons_ratio(weights, name)
                }
        
        analysis['weight_statistics'] = weight_stats
        analysis['layer_efficiency_analysis'] = self._analyze_layer_efficiency()
        analysis['channel_utilization'] = self._analyze_channel_utilization()
        analysis['weight_distribution_summary'] = self._compute_weight_summary()
        
        return analysis
    
    def _compute_effective_rank(self, weight_matrix):
        """Compute effective rank of weight matrix to assess information capacity"""
        if len(weight_matrix.shape) < 2:
            return None
            
        # Reshape to 2D if needed (for conv layers)
        if len(weight_matrix.shape) > 2:
            w_2d = weight_matrix.reshape(weight_matrix.shape[0], -1)
        else:
            w_2d = weight_matrix
            
        try:
            _, s, _ = np.linalg.svd(w_2d, full_matrices=False)
            # Compute effective rank using entropy-based measure
            s_norm = s / np.sum(s)
            effective_rank = np.exp(-np.sum(s_norm * np.log(s_norm + 1e-12)))
            return float(effective_rank / len(s))  # Normalize by actual rank
        except:
            return None
    
    def _compute_dead_neurons_ratio(self, weights, layer_name):
        """Compute ratio of dead/inactive neurons in layer"""
        if 'conv' in layer_name and 'weight' in layer_name:
            # For conv layers, check output channels
            if len(weights.shape) >= 4:  # (out_ch, in_ch, h, w)
                channel_norms = np.linalg.norm(weights.reshape(weights.shape[0], -1), axis=1)
                dead_ratio = np.sum(channel_norms < 1e-6) / len(channel_norms)
                return float(dead_ratio)
        elif 'linear' in layer_name and 'weight' in layer_name:
            # For linear layers, check output neurons
            if len(weights.shape) >= 2:
                neuron_norms = np.linalg.norm(weights, axis=1)
                dead_ratio = np.sum(neuron_norms < 1e-6) / len(neuron_norms)
                return float(dead_ratio)
        return 0.0
    
    def _analyze_layer_efficiency(self):
        """Analyze efficiency of each layer based on weight patterns"""
        efficiency_analysis = {}
        
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                weights = param.data.cpu().numpy()
                weights_flat = weights.flatten()
                
                # Basic efficiency metrics
                efficiency_metrics = {
                    'weight_magnitude': float(np.mean(np.abs(weights_flat))),
                    'weight_variance': float(np.var(weights_flat)),
                    'sparsity_level': float(np.sum(np.abs(weights_flat) < 1e-6) / len(weights_flat)),
                    'parameter_efficiency': float(np.std(weights_flat) / (np.mean(np.abs(weights_flat)) + 1e-12))
                }
                
                efficiency_analysis[name] = efficiency_metrics
        
        return efficiency_analysis
    
    def _analyze_channel_utilization(self):
        """Analyze channel utilization across convolutional layers"""
        channel_analysis = {}
        
        # Only analyze key layers for efficiency
        target_layers = ['adsorbate_processor.0', 'adsorbate_processor.3', 'solvent_processor.0', 'solvent_processor.3']
        
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv3d):
                weight = module.weight.data.cpu().numpy()
                # weight shape: (out_channels, in_channels, d, h, w)
                
                # Only show details for target layers
                show_details = any(target in name for target in target_layers)
                
                if show_details:
                    print(f"\n   🔍 Analyzing layer: {name}")
                    print(f"      Weight shape: {weight.shape}")
                
                # Analyze output channel utilization
                out_channel_norms = np.linalg.norm(weight.reshape(weight.shape[0], -1), axis=1)
                
                if show_details:
                    print(f"      Output channel weight norm statistics:")
                    print(f"        Mean: {np.mean(out_channel_norms):.4f}")
                    print(f"        Std: {np.std(out_channel_norms):.4f}")
                    print(f"        Max: {np.max(out_channel_norms):.4f}")
                    print(f"        Min: {np.min(out_channel_norms):.4f}")
                
                # Calculate basic utilization metrics
                mean_norm = np.mean(out_channel_norms)
                underutilized_threshold = 0.1 * mean_norm
                underutilized_count = np.sum(out_channel_norms < underutilized_threshold)
                underutilized_ratio = underutilized_count / len(out_channel_norms)
                
                if show_details:
                    print(f"        Underutilized channels: {underutilized_count}/{len(out_channel_norms)} ({underutilized_ratio:.1%})")
                
                out_channel_utilization = {
                    'mean_activation': float(np.mean(out_channel_norms)),
                    'std_activation': float(np.std(out_channel_norms)),
                    'min_activation': float(np.min(out_channel_norms)),
                    'max_activation': float(np.max(out_channel_norms)),
                    'underutilized_ratio': float(underutilized_ratio),
                    'channel_efficiency': float(np.std(out_channel_norms) / (np.mean(out_channel_norms) + 1e-12)),
                    'total_channels': len(out_channel_norms)
                }
                
                # Input channel analysis (simplified)
                if weight.shape[1] > 1:
                    in_channel_importance = np.mean(np.abs(weight), axis=(0, 2, 3, 4))
                    in_channel_analysis = {
                        'importance_variance': float(np.var(in_channel_importance)),
                        'redundant_channels': float(np.sum(in_channel_importance < 0.1 * np.mean(in_channel_importance)) / len(in_channel_importance))
                    }
                    
                    if show_details:
                        print(f"      Input channel sensitivity:")
                        print(f"        Importance variance: {in_channel_analysis['importance_variance']:.6f}")
                        print(f"        Redundant channels ratio: {in_channel_analysis['redundant_channels']:.1%}")
                else:
                    in_channel_analysis = {'importance_variance': 0.0, 'redundant_channels': 0.0}
                
                channel_analysis[name] = {
                    'output_channels': out_channel_utilization,
                    'input_channels': in_channel_analysis,
                    'total_channels': int(weight.shape[0])
                }
        
        return channel_analysis

    def _compute_weight_summary(self):
        """Compute summary weight statistics"""
        conv_weights = []
        linear_weights = []
        bn_weights = []
        
        for name, param in self.model.named_parameters():
            weights = param.data.cpu().numpy().flatten()
            if 'conv' in name and 'weight' in name:
                conv_weights.extend(weights)
            elif 'linear' in name or 'regressor' in name:
                if 'weight' in name:
                    linear_weights.extend(weights)
            elif 'bn' in name or 'norm' in name:
                bn_weights.extend(weights)
        
        return {
            'conv_layer_stats': {
                'mean': float(np.mean(conv_weights)) if conv_weights else 0,
                'std': float(np.std(conv_weights)) if conv_weights else 0,
                'count': len(conv_weights)
            },
            'linear_layer_stats': {
                'mean': float(np.mean(linear_weights)) if linear_weights else 0,
                'std': float(np.std(linear_weights)) if linear_weights else 0,
                'count': len(linear_weights)
            },
            'bn_layer_stats': {
                'mean': float(np.mean(bn_weights)) if bn_weights else 0,
                'std': float(np.std(bn_weights)) if bn_weights else 0,
                'count': len(bn_weights)
            }
        }
    
    def analyze_attention_mechanisms(self):
        """Analyze CBAM attention mechanism effectiveness"""
        print("\n👁️ Analyzing Attention Mechanisms...")
        
        analysis = {}
        
        # Find CBAM modules
        cbam_modules = {}
        for name, module in self.model.named_modules():
            if hasattr(module, 'channel_attention') or hasattr(module, 'spatial_attention'):
                cbam_modules[name] = module
        
        analysis['cbam_module_count'] = len(cbam_modules)
        analysis['cbam_modules'] = list(cbam_modules.keys())
        
        # Analyze attention weights if available
        attention_analysis = {}
        for name, module in cbam_modules.items():
            module_analysis = {}
            
            # Channel attention analysis
            if hasattr(module, 'channel_attention'):
                ch_attention = module.channel_attention
                if hasattr(ch_attention, 'fc1') and hasattr(ch_attention, 'fc2'):
                    fc1_weight = ch_attention.fc1.weight.data.cpu().numpy()
                    fc2_weight = ch_attention.fc2.weight.data.cpu().numpy()
                    
                    module_analysis['channel_attention'] = {
                        'reduction_ratio': float(fc1_weight.shape[0] / fc1_weight.shape[1]) if fc1_weight.shape[1] > 0 else 0,
                        'fc1_weight_norm': float(np.linalg.norm(fc1_weight)),
                        'fc2_weight_norm': float(np.linalg.norm(fc2_weight)),
                        'weight_diversity': float(np.std(fc1_weight.flatten())),
                    }
            
            # Spatial attention analysis
            if hasattr(module, 'spatial_attention'):
                sp_attention = module.spatial_attention
                if hasattr(sp_attention, 'conv'):
                    conv_weight = sp_attention.conv.weight.data.cpu().numpy()
                    module_analysis['spatial_attention'] = {
                        'conv_weight_norm': float(np.linalg.norm(conv_weight)),
                        'kernel_diversity': float(np.std(conv_weight)),
                    }
            
            attention_analysis[name] = module_analysis
        
        analysis['attention_weight_analysis'] = attention_analysis
        analysis['attention_effectiveness'] = self._evaluate_attention_effectiveness(cbam_modules)
        
        return analysis
    
    def _evaluate_attention_effectiveness(self, cbam_modules):
        """Evaluate the effectiveness of attention mechanisms"""
        effectiveness_metrics = {}
        
        for name, module in cbam_modules.items():
            metrics = {}
            
            # Analyze attention diversity
            if hasattr(module, 'channel_attention'):
                ch_attn = module.channel_attention
                if hasattr(ch_attn, 'fc1'):
                    fc1_weights = ch_attn.fc1.weight.data.cpu().numpy()
                    # Measure weight diversity as effectiveness indicator
                    weight_entropy = -np.sum(np.abs(fc1_weights.flatten()) * np.log(np.abs(fc1_weights.flatten()) + 1e-12))
                    metrics['channel_diversity'] = float(weight_entropy)
            
            # Parameter utilization
            total_attention_params = sum(p.numel() for p in module.parameters())
            metrics['parameter_count'] = total_attention_params
            
        return effectiveness_metrics
    
    def analyze_gradient_patterns(self):
        """Analyze gradient magnitudes and patterns for parameter importance"""
        print("\n📊 Analyzing Gradient Patterns...")
        
        analysis = {}
        
        try:
            # Create a sample input for gradient analysis
            sample_input = torch.randn(1, 28, 20, 20, 20)  # Batch size 1
            sample_target = torch.randn(1, 1)  # Single target value
            
            # Enable gradient computation
            self.model.train()
            for param in self.model.parameters():
                param.requires_grad = True
            
            # Forward pass
            output = self.model(sample_input)
            loss = F.mse_loss(output, sample_target)
            
            # Backward pass to compute gradients
            loss.backward()
            
            # Analyze gradients
            gradient_stats = {}
            layer_gradient_norms = {}
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad = param.grad.data.cpu().numpy()
                    grad_flat = grad.flatten()
                    
                    gradient_stats[name] = {
                        'mean': float(np.mean(grad_flat)),
                        'std': float(np.std(grad_flat)),
                        'max': float(np.max(np.abs(grad_flat))),
                        'norm': float(np.linalg.norm(grad_flat)),
                        'sparsity': float(np.sum(np.abs(grad_flat) < 1e-8) / len(grad_flat)),
                        'param_count': int(len(grad_flat))
                    }
                    
                    # Categorize by layer type
                    layer_type = self._get_layer_category(name)
                    if layer_type not in layer_gradient_norms:
                        layer_gradient_norms[layer_type] = []
                    layer_gradient_norms[layer_type].append(gradient_stats[name]['norm'])
            
            # Compute layer-level gradient statistics
            layer_grad_summary = {}
            for layer_type, norms in layer_gradient_norms.items():
                if norms:
                    layer_grad_summary[layer_type] = {
                        'mean_norm': float(np.mean(norms)),
                        'std_norm': float(np.std(norms)),
                        'max_norm': float(np.max(norms)),
                        'min_norm': float(np.min(norms)),
                        'layer_count': len(norms)
                    }
            
            analysis['gradient_statistics'] = gradient_stats
            analysis['layer_gradient_summary'] = layer_grad_summary
            analysis['gradient_flow_health'] = self._assess_gradient_flow(gradient_stats)
            
            # Reset model to eval mode
            self.model.eval()
            
        except Exception as e:
            print(f"⚠️ Gradient analysis failed: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _get_layer_category(self, layer_name):
        """Categorize layer by name for analysis"""
        if 'adsorbate_processor' in layer_name:
            return 'adsorbate_branch'
        elif 'solvent_processor' in layer_name:
            return 'solvent_branch'
        elif 'interaction' in layer_name:
            return 'interaction_layers'
        elif 'layer' in layer_name and 'cbam' not in layer_name:
            return 'cnn_backbone'
        elif 'cbam' in layer_name or 'attention' in layer_name:
            return 'attention_modules'
        elif 'regressor' in layer_name:
            return 'regressor'
        else:
            return 'other'
    
    def _assess_gradient_flow(self, gradient_stats):
        """Assess the health of gradient flow through the network"""
        gradient_norms = [stats['norm'] for stats in gradient_stats.values()]
        
        if not gradient_norms:
            return {'error': 'No gradients available'}
        
        # Identify problematic layers
        very_small_gradients = [name for name, stats in gradient_stats.items() 
                               if stats['norm'] < 1e-6]
        very_large_gradients = [name for name, stats in gradient_stats.items() 
                               if stats['norm'] > 10.0]
        
        return {
            'overall_gradient_norm': float(np.sqrt(np.sum([norm**2 for norm in gradient_norms]))),
            'mean_gradient_norm': float(np.mean(gradient_norms)),
            'gradient_norm_std': float(np.std(gradient_norms)),
            'vanishing_gradient_layers': very_small_gradients,
            'exploding_gradient_layers': very_large_gradients,
            'gradient_flow_ratio': float(np.max(gradient_norms) / (np.min(gradient_norms) + 1e-12)),
            'healthy_gradient_flow': len(very_small_gradients) == 0 and len(very_large_gradients) == 0
        }
    
    def analyze_processor_information_handling(self):
        """
        Deep analysis of how well adsorbate and solvent processors handle their respective voxel data.
        This function specifically examines whether each processor is suitable for its input data type.
        """
        print("\n🔬 Analyzing Processor Information Handling...")
        
        analysis = {}
        
        try:
            # Create realistic test inputs
            batch_size = 4
            # Simulate realistic adsorbate and solvent channel distributions
            
            # Adsorbate input: sparse central features (most voxels are zeros)
            adsorbate_input = torch.zeros(batch_size, 14, 20, 20, 20)
            # Add concentrated adsorbate features in center region (8x8x8 around center)
            center = 10
            for b in range(batch_size):
                for c in range(14):
                    # Simulate sparse adsorbate concentrated in center
                    adsorbate_input[b, c, center-4:center+4, center-4:center+4, center-4:center+4] = \
                        torch.randn(8, 8, 8) * 0.5 + 0.3  # Some realistic adsorbate feature values
            
            # Solvent input: dense distributed features (fills most space)
            solvent_input = torch.randn(batch_size, 14, 20, 20, 20) * 0.2 + 0.1
            # Make it more realistic - stronger around edges, weaker in center
            for b in range(batch_size):
                for c in range(14):
                    # Reduce solvent density in center where adsorbate is
                    solvent_input[b, c, center-4:center+4, center-4:center+4, center-4:center+4] *= 0.3
            
            # Combine to full input
            full_input = torch.zeros(batch_size, 28, 20, 20, 20)
            full_input[:, :14] = adsorbate_input  # First 14 channels: adsorbate
            full_input[:, 14:] = solvent_input    # Last 14 channels: solvent
            
            self.model.eval()
            
            # Hook both processors to capture intermediate outputs
            adsorbate_outputs = {}
            solvent_outputs = {}
            hooks = []
            
            def adsorbate_hook_fn(layer_idx):
                def hook(module, input, output):
                    adsorbate_outputs[f'layer_{layer_idx}'] = output.clone().detach()
                return hook
            
            def solvent_hook_fn(layer_idx):
                def hook(module, input, output):
                    solvent_outputs[f'layer_{layer_idx}'] = output.clone().detach()
                return hook
            
            # Register hooks for each layer in processors
            adsorbate_layers = list(self.model.adsorbate_processor.children())
            solvent_layers = list(self.model.solvent_processor.children())
            
            for i, layer in enumerate(adsorbate_layers):
                if isinstance(layer, (torch.nn.Conv3d, torch.nn.Sequential)):
                    hook = layer.register_forward_hook(adsorbate_hook_fn(i))
                    hooks.append(hook)
            
            for i, layer in enumerate(solvent_layers):
                if isinstance(layer, (torch.nn.Conv3d, torch.nn.Sequential)):
                    hook = layer.register_forward_hook(solvent_hook_fn(i))
                    hooks.append(hook)
            
            # Forward pass
            with torch.no_grad():
                _ = self.model(full_input)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Analyze adsorbate processor performance
            print("   🔍 Analyzing Adsorbate Processor:")
            adsorbate_analysis = self._analyze_sparse_data_processing(
                adsorbate_input, adsorbate_outputs, "adsorbate"
            )
            
            # Analyze solvent processor performance  
            print("   🔍 Analyzing Solvent Processor:")
            solvent_analysis = self._analyze_dense_data_processing(
                solvent_input, solvent_outputs, "solvent"
            )
            
            # Compare information preservation between processors
            information_comparison = self._compare_information_preservation(
                adsorbate_analysis, solvent_analysis
            )
            
            analysis['adsorbate_processor'] = adsorbate_analysis
            analysis['solvent_processor'] = solvent_analysis  
            analysis['information_comparison'] = information_comparison
            analysis['processor_suitability'] = self._assess_processor_suitability(
                adsorbate_analysis, solvent_analysis
            )
            
        except Exception as e:
            print(f"   ⚠️ Processor analysis failed: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _analyze_sparse_data_processing(self, input_data, layer_outputs, processor_name):
        """Analyze how well the processor handles sparse adsorbate data"""
        analysis = {}
        
        # Input characteristics
        input_sparsity = float(torch.sum(torch.abs(input_data) < 1e-6) / input_data.numel())
        input_magnitude = float(torch.mean(torch.abs(input_data[input_data.abs() > 1e-6])))  # Non-zero values only
        input_concentration = self._compute_spatial_concentration(input_data)
        
        print(f"      Input characteristics:")
        print(f"        Sparsity: {input_sparsity:.1%}")
        print(f"        Non-zero magnitude: {input_magnitude:.4f}")
        print(f"        Spatial concentration: {input_concentration:.4f}")
        
        # Analyze each layer's output
        layer_analysis = {}
        for layer_name, output in layer_outputs.items():
            if len(output.shape) >= 4:  # Conv layer outputs
                layer_sparsity = float(torch.sum(torch.abs(output) < 1e-6) / output.numel())
                layer_magnitude = float(torch.mean(torch.abs(output)))
                layer_concentration = self._compute_spatial_concentration(output)
                
                # Information preservation metrics
                info_density = layer_magnitude * (1 - layer_sparsity)
                concentration_change = layer_concentration - input_concentration
                
                layer_analysis[layer_name] = {
                    'sparsity': layer_sparsity,
                    'magnitude': layer_magnitude, 
                    'concentration': layer_concentration,
                    'concentration_change': concentration_change,
                    'information_density': info_density,
                    'output_shape': list(output.shape)
                }
                
                print(f"        {layer_name}: sparsity={layer_sparsity:.1%}, "
                      f"mag={layer_magnitude:.4f}, conc={layer_concentration:.4f}")
        
        analysis['input_stats'] = {
            'sparsity': input_sparsity,
            'magnitude': input_magnitude,
            'concentration': input_concentration
        }
        analysis['layer_outputs'] = layer_analysis
        analysis['information_flow'] = self._assess_sparse_information_flow(layer_analysis)
        
        return analysis
    
    def _analyze_dense_data_processing(self, input_data, layer_outputs, processor_name):
        """Analyze how well the processor handles dense solvent data"""
        analysis = {}
        
        # Input characteristics
        input_sparsity = float(torch.sum(torch.abs(input_data) < 1e-6) / input_data.numel())
        input_magnitude = float(torch.mean(torch.abs(input_data)))
        input_uniformity = self._compute_spatial_uniformity(input_data)
        
        print(f"      Input characteristics:")
        print(f"        Sparsity: {input_sparsity:.1%}")
        print(f"        Magnitude: {input_magnitude:.4f}")
        print(f"        Spatial uniformity: {input_uniformity:.4f}")
        
        # Analyze each layer's output
        layer_analysis = {}
        for layer_name, output in layer_outputs.items():
            if len(output.shape) >= 4:  # Conv layer outputs
                layer_sparsity = float(torch.sum(torch.abs(output) < 1e-6) / output.numel())
                layer_magnitude = float(torch.mean(torch.abs(output)))
                layer_uniformity = self._compute_spatial_uniformity(output)
                
                # Information preservation metrics
                pattern_diversity = float(torch.std(output))
                feature_richness = layer_magnitude * pattern_diversity
                
                layer_analysis[layer_name] = {
                    'sparsity': layer_sparsity,
                    'magnitude': layer_magnitude,
                    'uniformity': layer_uniformity, 
                    'pattern_diversity': pattern_diversity,
                    'feature_richness': feature_richness,
                    'output_shape': list(output.shape)
                }
                
                print(f"        {layer_name}: sparsity={layer_sparsity:.1%}, "
                      f"mag={layer_magnitude:.4f}, diversity={pattern_diversity:.4f}")
        
        analysis['input_stats'] = {
            'sparsity': input_sparsity,
            'magnitude': input_magnitude,
            'uniformity': input_uniformity
        }
        analysis['layer_outputs'] = layer_analysis
        analysis['information_flow'] = self._assess_dense_information_flow(layer_analysis)
        
        return analysis
    
    def _compute_spatial_concentration(self, tensor):
        """Compute how concentrated the features are spatially (for sparse data)"""
        if len(tensor.shape) < 4:
            return 0.0
        
        # Sum across batch and channel dimensions
        spatial_sum = torch.sum(torch.abs(tensor), dim=(0, 1))  # Shape: (H, W, D)
        
        # Compute center of mass
        coords = torch.stack(torch.meshgrid(
            torch.arange(spatial_sum.shape[0], dtype=torch.float),
            torch.arange(spatial_sum.shape[1], dtype=torch.float), 
            torch.arange(spatial_sum.shape[2], dtype=torch.float),
            indexing='ij'
        ), dim=-1)  # Shape: (H, W, D, 3)
        
        if torch.sum(spatial_sum) < 1e-6:
            return 0.0
        
        # Weighted center of mass
        center_of_mass = torch.sum(coords * spatial_sum.unsqueeze(-1), dim=(0, 1, 2)) / torch.sum(spatial_sum)
        
        # Compute concentration as inverse of spread around center of mass
        distances = torch.norm(coords - center_of_mass, dim=-1)
        weighted_distance = torch.sum(distances * spatial_sum) / torch.sum(spatial_sum)
        
        # Normalize by maximum possible distance
        max_distance = float(torch.norm(torch.tensor(spatial_sum.shape, dtype=torch.float)))
        concentration = 1.0 - (weighted_distance / max_distance)
        
        return float(concentration)
    
    def _compute_spatial_uniformity(self, tensor):
        """Compute how uniform the features are spatially (for dense data)"""
        if len(tensor.shape) < 4:
            return 0.0
        
        # Sum across batch and channel dimensions
        spatial_sum = torch.sum(torch.abs(tensor), dim=(0, 1))  # Shape: (H, W, D)
        
        if torch.sum(spatial_sum) < 1e-6:
            return 0.0
        
        # Compute coefficient of variation (std/mean) as measure of non-uniformity
        spatial_mean = torch.mean(spatial_sum)
        spatial_std = torch.std(spatial_sum)
        
        if spatial_mean < 1e-6:
            return 0.0
        
        # Uniformity = 1 - coefficient_of_variation (higher is more uniform)
        uniformity = 1.0 - (spatial_std / spatial_mean)
        return float(torch.clamp(uniformity, 0.0, 1.0))
    
    def _assess_sparse_information_flow(self, layer_analysis):
        """Assess how well information flows through layers for sparse input"""
        if not layer_analysis:
            return {'quality': 'unknown', 'issues': ['no_layer_data']}
        
        # Check for information preservation issues
        issues = []
        quality_scores = []
        
        layers = sorted(layer_analysis.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
        
        for i, layer_name in enumerate(layers):
            layer_data = layer_analysis[layer_name]
            
            # Check if sparsity is increasing too much (information loss)
            if layer_data['sparsity'] > 0.9:
                issues.append(f"{layer_name}_excessive_sparsity")
            
            # Check if magnitude is decreasing too much (vanishing activations)
            if layer_data['magnitude'] < 0.01:
                issues.append(f"{layer_name}_vanishing_activations")
            
            # Check if concentration is decreasing (spreading too much)
            if layer_data['concentration_change'] < -0.3:
                issues.append(f"{layer_name}_excessive_spreading")
            
            # Quality score based on information density and concentration
            info_quality = layer_data['information_density'] * (1 + layer_data['concentration'])
            quality_scores.append(info_quality)
        
        # Overall assessment
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        if avg_quality > 0.1:
            quality = 'good'
        elif avg_quality > 0.05:
            quality = 'moderate'  
        else:
            quality = 'poor'
        
        return {
            'quality': quality,
            'average_info_quality': avg_quality,
            'issues': issues,
            'layer_qualities': dict(zip(layers, quality_scores))
        }
    
    def _assess_dense_information_flow(self, layer_analysis):
        """Assess how well information flows through layers for dense input"""
        if not layer_analysis:
            return {'quality': 'unknown', 'issues': ['no_layer_data']}
        
        issues = []
        quality_scores = []
        
        layers = sorted(layer_analysis.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
        
        for layer_name in layers:
            layer_data = layer_analysis[layer_name]
            
            # Check for pattern diversity (important for dense data)
            if layer_data['pattern_diversity'] < 0.01:
                issues.append(f"{layer_name}_low_pattern_diversity")
            
            # Check for feature richness
            if layer_data['feature_richness'] < 0.001:
                issues.append(f"{layer_name}_low_feature_richness")
            
            # Check if becoming too sparse (losing distributed information)
            if layer_data['sparsity'] > 0.8:
                issues.append(f"{layer_name}_excessive_sparsification")
            
            # Quality score based on feature richness and pattern diversity
            richness_quality = layer_data['feature_richness'] * (1 - layer_data['sparsity'])
            quality_scores.append(richness_quality)
        
        # Overall assessment
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        if avg_quality > 0.05:
            quality = 'good'
        elif avg_quality > 0.02:
            quality = 'moderate'
        else:
            quality = 'poor'
        
        return {
            'quality': quality,
            'average_info_quality': avg_quality, 
            'issues': issues,
            'layer_qualities': dict(zip(layers, quality_scores))
        }
    
    def _compare_information_preservation(self, adsorbate_analysis, solvent_analysis):
        """Compare information preservation between the two processors"""
        comparison = {}
        
        # Get quality assessments
        ads_quality = adsorbate_analysis.get('information_flow', {}).get('quality', 'unknown')
        solv_quality = solvent_analysis.get('information_flow', {}).get('quality', 'unknown')
        
        ads_score = adsorbate_analysis.get('information_flow', {}).get('average_info_quality', 0)
        solv_score = solvent_analysis.get('information_flow', {}).get('average_info_quality', 0)
        
        comparison['quality_comparison'] = {
            'adsorbate': ads_quality,
            'solvent': solv_quality,
            'adsorbate_score': ads_score,
            'solvent_score': solv_score
        }
        
        # Identify which processor performs better
        if ads_score > solv_score * 1.2:
            better_processor = 'adsorbate'
        elif solv_score > ads_score * 1.2:
            better_processor = 'solvent'
        else:
            better_processor = 'similar'
        
        comparison['better_processor'] = better_processor
        comparison['score_ratio'] = solv_score / (ads_score + 1e-8)
        
        return comparison
    
    def _assess_processor_suitability(self, adsorbate_analysis, solvent_analysis):
        """Assess whether each processor is suitable for its input data type"""
        suitability = {}
        
        # Adsorbate processor suitability for sparse data
        ads_issues = adsorbate_analysis.get('information_flow', {}).get('issues', [])
        ads_quality = adsorbate_analysis.get('information_flow', {}).get('quality', 'unknown')
        
        # Key issues for sparse data processing
        sparse_critical_issues = [issue for issue in ads_issues if 
                                'excessive_sparsity' in issue or 'vanishing_activations' in issue]
        
        if ads_quality == 'good' and len(sparse_critical_issues) == 0:
            ads_suitability = 'well_suited'
        elif ads_quality == 'moderate' and len(sparse_critical_issues) <= 1:
            ads_suitability = 'moderately_suited'
        else:
            ads_suitability = 'poorly_suited'
        
        # Solvent processor suitability for dense data
        solv_issues = solvent_analysis.get('information_flow', {}).get('issues', [])
        solv_quality = solvent_analysis.get('information_flow', {}).get('quality', 'unknown')
        
        # Key issues for dense data processing
        dense_critical_issues = [issue for issue in solv_issues if
                               'low_pattern_diversity' in issue or 'excessive_sparsification' in issue]
        
        if solv_quality == 'good' and len(dense_critical_issues) == 0:
            solv_suitability = 'well_suited'
        elif solv_quality == 'moderate' and len(dense_critical_issues) <= 1:
            solv_suitability = 'moderately_suited'
        else:
            solv_suitability = 'poorly_suited'
        
        suitability['adsorbate_processor'] = {
            'suitability': ads_suitability,
            'critical_issues': sparse_critical_issues,
            'quality': ads_quality
        }
        
        suitability['solvent_processor'] = {
            'suitability': solv_suitability,
            'critical_issues': dense_critical_issues,
            'quality': solv_quality
        }
        
        return suitability

    def analyze_layer_contributions(self):
        """Analyze the contribution of each layer to the final prediction"""
        print("\n🎯 Analyzing Layer Contributions...")
        
        analysis = {}
        
        try:
            # Create sample input
            sample_input = torch.randn(2, 28, 20, 20, 20)  # Batch size 2 for better statistics
            
            self.model.eval()
            
            # Get baseline prediction
            with torch.no_grad():
                baseline_output = self.model(sample_input)
            
            # Hook functions to capture layer outputs
            layer_outputs = {}
            hooks = []
            
            def hook_fn(name):
                def hook(module, input, output):
                    if isinstance(output, torch.Tensor):
                        layer_outputs[name] = output.clone().detach()
                return hook
            
            # Register hooks for key layers
            key_modules = [
                'adsorbate_processor',
                'solvent_processor', 
                'interaction_conv',
                'layer1',
                'layer2', 
                'layer3',
                'regressor'
            ]
            
            for name, module in self.model.named_modules():
                if any(key in name for key in key_modules) and len(list(module.children())) == 0:
                    # Only hook leaf modules
                    hook = module.register_forward_hook(hook_fn(name))
                    hooks.append(hook)
            
            # Forward pass to collect layer outputs
            with torch.no_grad():
                _ = self.model(sample_input)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Analyze layer contributions
            layer_contributions = {}
            for layer_name, output in layer_outputs.items():
                if len(output.shape) >= 4:  # Conv layers
                    # Compute activation statistics
                    activation_mean = float(torch.mean(torch.abs(output)))
                    activation_std = float(torch.std(output))
                    sparsity = float(torch.sum(torch.abs(output) < 1e-6) / output.numel())
                    
                    layer_contributions[layer_name] = {
                        'activation_magnitude': activation_mean,
                        'activation_diversity': activation_std,
                        'sparsity_ratio': sparsity,
                        'output_shape': list(output.shape),
                        'total_activations': int(output.numel()),
                        'contribution_score': activation_mean * (1 - sparsity)  # Higher is better
                    }
            
            analysis['layer_contributions'] = layer_contributions
            analysis['contribution_ranking'] = self._rank_layer_contributions(layer_contributions)
            
        except Exception as e:
            print(f"⚠️ Layer contribution analysis failed: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _rank_layer_contributions(self, contributions):
        """Rank layers by their contribution to the model"""
        ranked_layers = sorted(
            contributions.items(), 
            key=lambda x: x[1]['contribution_score'], 
            reverse=True
        )
        
        ranking = {}
        for i, (layer_name, stats) in enumerate(ranked_layers):
            ranking[layer_name] = {
                'rank': i + 1,
                'contribution_score': stats['contribution_score'],
                'relative_importance': stats['contribution_score'] / ranked_layers[0][1]['contribution_score'] if ranked_layers else 0
            }
        
        return ranking
    
    def analyze_parameter_sensitivity(self):
        """Analyze parameter sensitivity and redundancy"""
        print("\n🔍 Analyzing Parameter Sensitivity...")
        
        analysis = {}
        
        try:
            # Create test input
            test_input = torch.randn(1, 28, 20, 20, 20)
            
            self.model.eval()
            
            # Get baseline prediction
            with torch.no_grad():
                baseline_pred = self.model(test_input)
            
            # Analyze parameter importance by perturbation
            param_sensitivity = {}
            
            for name, param in self.model.named_parameters():
                if param.numel() > 1000:  # Only analyze large parameter tensors
                    original_param = param.data.clone()
                    
                    # Perturb parameters slightly
                    perturbation = 0.01 * torch.randn_like(param.data)
                    param.data.add_(perturbation)
                    
                    # Get prediction with perturbed parameters
                    with torch.no_grad():
                        perturbed_pred = self.model(test_input)
                    
                    # Compute sensitivity
                    pred_change = float(torch.abs(perturbed_pred - baseline_pred).mean())
                    param_change = float(torch.norm(perturbation))
                    sensitivity = pred_change / (param_change + 1e-12)
                    
                    param_sensitivity[name] = {
                        'sensitivity': sensitivity,
                        'param_count': int(param.numel()),
                        'param_norm': float(torch.norm(original_param)),
                        'relative_sensitivity': sensitivity * param.numel()  # Account for parameter count
                    }
                    
                    # Restore original parameters
                    param.data.copy_(original_param)
            
            analysis['parameter_sensitivity'] = param_sensitivity
            analysis['sensitivity_ranking'] = self._rank_parameter_sensitivity(param_sensitivity)
            analysis['redundancy_analysis'] = self._analyze_parameter_redundancy()
            
        except Exception as e:
            print(f"⚠️ Parameter sensitivity analysis failed: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _rank_parameter_sensitivity(self, sensitivity_stats):
        """Rank parameters by sensitivity"""
        ranked_params = sorted(
            sensitivity_stats.items(),
            key=lambda x: x[1]['relative_sensitivity'],
            reverse=True
        )
        
        ranking = {}
        for i, (param_name, stats) in enumerate(ranked_params):
            ranking[param_name] = {
                'rank': i + 1,
                'sensitivity_score': stats['relative_sensitivity'],
                'importance_ratio': stats['relative_sensitivity'] / ranked_params[0][1]['relative_sensitivity'] if ranked_params else 0
            }
        
        return ranking
    
    def _analyze_parameter_redundancy(self):
        """Analyze parameter redundancy within layers"""
        redundancy_analysis = {}
        
        for name, param in self.model.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                param_np = param.data.cpu().numpy()
                
                if len(param_np.shape) == 4:  # Conv layer: (out_ch, in_ch, k, k)
                    # Reshape to (out_ch, in_ch*k*k)
                    reshaped = param_np.reshape(param_np.shape[0], -1)
                elif len(param_np.shape) == 2:  # Linear layer
                    reshaped = param_np
                else:
                    continue
                
                # Compute pairwise correlations between output channels/neurons
                try:
                    correlation_matrix = np.corrcoef(reshaped)
                    
                    # Find highly correlated pairs (excluding diagonal)
                    high_corr_threshold = 0.9
                    high_correlations = []
                    
                    for i in range(len(correlation_matrix)):
                        for j in range(i + 1, len(correlation_matrix)):
                            if abs(correlation_matrix[i, j]) > high_corr_threshold:
                                high_correlations.append((i, j, correlation_matrix[i, j]))
                    
                    redundancy_analysis[name] = {
                        'total_channels': reshaped.shape[0],
                        'highly_correlated_pairs': len(high_correlations),
                        'redundancy_ratio': len(high_correlations) / (reshaped.shape[0] * (reshaped.shape[0] - 1) / 2),
                        'mean_correlation': float(np.mean(np.abs(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])))
                    }
                    
                except Exception as e:
                    redundancy_analysis[name] = {'error': str(e)}
        
        return redundancy_analysis
    
    def analyze_computational_complexity(self):
        """Analyze computational complexity and memory usage"""
        print("\n⚡ Analyzing Computational Complexity...")
        
        analysis = {}
        
        try:
            # Sample input for FLOP counting
            sample_input = torch.randn(1, 28, 20, 20, 20)
            
            # Count parameters by module type
            module_params = {}
            module_memory = {}
            
            for name, module in self.model.named_modules():
                if len(list(module.children())) == 0:  # Leaf modules only
                    param_count = sum(p.numel() for p in module.parameters())
                    if param_count > 0:
                        module_params[name] = param_count
                        # Estimate memory usage (assuming float32)
                        module_memory[name] = param_count * 4  # bytes
            
            # Estimate FLOPs for different module types
            flop_estimates = {}
            
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv3d):
                    # For 3D conv: FLOPs ≈ output_size * kernel_size * in_channels
                    kernel_ops = np.prod(module.kernel_size) * module.in_channels
                    output_size = 20 * 20 * 20  # Assuming typical feature map size
                    flop_estimates[name] = kernel_ops * output_size * module.out_channels
                elif isinstance(module, torch.nn.Linear):
                    # For linear: FLOPs = input_size * output_size
                    flop_estimates[name] = module.in_features * module.out_features
            
            total_flops = sum(flop_estimates.values())
            total_params = sum(module_params.values())
            total_memory = sum(module_memory.values())
            
            analysis['parameter_breakdown'] = module_params
            analysis['memory_breakdown'] = module_memory
            analysis['flop_estimates'] = flop_estimates
            analysis['complexity_summary'] = {
                'total_parameters': total_params,
                'total_memory_mb': total_memory / (1024 * 1024),
                'estimated_flops': total_flops,
                'params_per_flop': total_params / (total_flops + 1e-12)
            }
            
        except Exception as e:
            print(f"⚠️ Complexity analysis failed: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def generate_optimization_recommendations(self):
        """Generate specific optimization recommendations based on analysis"""
        print("\n💡 Generating Optimization Recommendations...")
        
        recommendations = {
            'parameter_reduction': [],
            'architecture_changes': [],
            'training_improvements': [],
            'priority_actions': []
        }
        
        # Analyze current results to generate recommendations
        arch_analysis = self.analysis_results.get('architecture_efficiency', {})
        weight_patterns = self.analysis_results.get('weight_patterns', {})
        gradient_analysis = self.analysis_results.get('gradient_patterns', {})
        sensitivity_analysis = self.analysis_results.get('parameter_sensitivity', {})
        
        # Parameter reduction recommendations
        param_dist = arch_analysis.get('layer_percentages', {})
        if param_dist.get('cnn_backbone', 0) > 50:
            recommendations['parameter_reduction'].append({
                'action': 'Reduce CNN backbone channels',
                'reason': f"CNN backbone uses {param_dist.get('cnn_backbone', 0):.1f}% of parameters",
                'suggestion': 'Consider reducing ResNet block channels by 20-30%',
                'priority': 'High'
            })
        
        if param_dist.get('solvent_branch', 0) > 25:
            recommendations['parameter_reduction'].append({
                'action': 'Optimize solvent branch',
                'reason': f"Solvent branch uses {param_dist.get('solvent_branch', 0):.1f}% of parameters",
                'suggestion': 'Reduce solvent processor channels from 48 to 32',
                'priority': 'Medium'
            })
        
        # Architecture change recommendations
        channel_util = weight_patterns.get('channel_utilization', {})
        underutilized_layers = []
        for layer_name, info in channel_util.items():
            if isinstance(info, dict) and 'output_channels' in info:
                underutil_ratio = info['output_channels'].get('underutilized_ratio', 0)
                if underutil_ratio > 0.3:
                    underutilized_layers.append(layer_name)
        
        if underutilized_layers:
            recommendations['architecture_changes'].append({
                'action': 'Channel pruning',
                'reason': f"{len(underutilized_layers)} layers have >30% underutilized channels",
                'suggestion': f'Prune channels in: {", ".join(underutilized_layers[:3])}',
                'priority': 'High'
            })
        
        # Training improvement recommendations
        training_conv = self.analysis_results.get('training_convergence', {})
        performance = training_conv.get('final_performance', {})
        overfitting_ratio = performance.get('overfitting_ratio', 1)
        
        if overfitting_ratio > 5:
            recommendations['training_improvements'].append({
                'action': 'Reduce overfitting',
                'reason': f"Overfitting ratio: {overfitting_ratio:.2f} (Train RMSE much lower than Test RMSE)",
                'suggestion': 'Add dropout (0.2-0.3), reduce model complexity, or use stronger regularization',
                'priority': 'Critical'
            })
        
        # Priority actions based on analysis
        if overfitting_ratio > 10:
            recommendations['priority_actions'].append({
                'action': 'URGENT: Address severe overfitting',
                'steps': [
                    '1. Reduce model parameters by 30-50%',
                    '2. Increase dropout to 0.3-0.4',
                    '3. Add L2 regularization (weight_decay=1e-4)',
                    '4. Consider early stopping with patience=10'
                ]
            })
        
        if arch_analysis.get('parameter_stats', {}).get('total_parameters', 0) > 800000:
            recommendations['priority_actions'].append({
                'action': 'Reduce model size',
                'steps': [
                    '1. Reduce ResNet channels: 48→32, 64→48, 80→64',
                    '2. Reduce solvent branch: 32→24, 48→32',
                    '3. Consider depthwise separable convolutions'
                ]
            })
        
        return recommendations
    
    def analyze_feature_activation_patterns(self):
        """Analyze feature map activation patterns using synthetic data"""
        print("\n🔥 Analyzing Feature Activation Patterns...")
        
        analysis = {}
        
        # Generate synthetic input for activation analysis
        device = next(self.model.parameters()).device
        synthetic_input = torch.randn(1, 28, 20, 20, 20).to(device)  # Batch=1, Channels=28, D=H=W=20
        
        activation_stats = {}
        hooks = []
        
        def get_activation_stats(name):
            def hook(model, input, output):
                if isinstance(output, torch.Tensor):
                    activation = output.detach().cpu().numpy()
                    activation_stats[name] = {
                        'mean_activation': float(np.mean(activation)),
                        'std_activation': float(np.std(activation)),
                        'max_activation': float(np.max(activation)),
                        'min_activation': float(np.min(activation)),
                        'sparsity': float(np.sum(np.abs(activation) < 1e-6) / activation.size),
                        'effective_neurons': float(np.sum(np.max(np.abs(activation.reshape(activation.shape[0], activation.shape[1], -1)), axis=2) > 1e-6)),
                        'output_shape': list(activation.shape),
                        'dead_neuron_ratio': self._compute_dead_neuron_ratio(activation)
                    }
            return hook
        
        # Register hooks for key layers
        target_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv3d, torch.nn.Linear)):
                if any(keyword in name for keyword in ['layer1', 'layer2', 'layer3', 'adsorbate_processor', 'solvent_processor', 'regressor']):
                    target_layers.append((name, module))
        
        # Register hooks
        for name, module in target_layers:
            hook = module.register_forward_hook(get_activation_stats(name))
            hooks.append(hook)
        
        # Forward pass
        try:
            self.model.eval()
            with torch.no_grad():
                _ = self.model(synthetic_input, synthetic_input)  # Assuming dual input (adsorbate, solvent)
        except Exception as e:
            print(f"⚠️ Forward pass failed, using single input: {e}")
            try:
                _ = self.model(synthetic_input)
            except Exception as e2:
                print(f"⚠️ Forward pass completely failed: {e2}")
                activation_stats = {'error': str(e2)}
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        analysis['layer_activations'] = activation_stats
        analysis['activation_summary'] = self._summarize_activation_patterns(activation_stats)
        
        return analysis
    
    def _compute_dead_neuron_ratio(self, activation):
        """Compute ratio of dead neurons in activation"""
        if len(activation.shape) < 2:
            return 0.0
        
        # For conv layers: check channels, for linear: check neurons
        if len(activation.shape) >= 3:  # Conv layer (batch, channels, ...)
            channel_max = np.max(np.abs(activation.reshape(activation.shape[0], activation.shape[1], -1)), axis=(0, 2))
            dead_ratio = np.sum(channel_max < 1e-6) / len(channel_max)
        else:  # Linear layer (batch, features)
            neuron_max = np.max(np.abs(activation), axis=0)
            dead_ratio = np.sum(neuron_max < 1e-6) / len(neuron_max)
        
        return float(dead_ratio)
    
    def _summarize_activation_patterns(self, activation_stats):
        """Summarize activation patterns across layers"""
        if 'error' in activation_stats:
            return {'error': activation_stats['error']}
            
        summary = {
            'total_layers_analyzed': len(activation_stats),
            'overall_sparsity': np.mean([stats['sparsity'] for stats in activation_stats.values()]),
            'overall_dead_neuron_ratio': np.mean([stats['dead_neuron_ratio'] for stats in activation_stats.values()]),
            'layers_with_high_sparsity': [],
            'layers_with_dead_neurons': [],
            'underutilized_layers': []
        }
        
        for layer_name, stats in activation_stats.items():
            if stats['sparsity'] > 0.8:
                summary['layers_with_high_sparsity'].append(layer_name)
            if stats['dead_neuron_ratio'] > 0.3:
                summary['layers_with_dead_neurons'].append(layer_name)
            if stats['std_activation'] < 0.01:
                summary['underutilized_layers'].append(layer_name)
        
        return summary
    
    def analyze_training_convergence(self):
        """Analyze training convergence patterns from checkpoint"""
        print("\n📈 Analyzing Training Convergence...")
        
        analysis = {}
        metadata = self.extract_checkpoint_metadata()
        
        if metadata:
            # Training metrics analysis
            if 'train_metrics' in metadata and 'test_metrics' in metadata:
                train_metrics = metadata['train_metrics']
                test_metrics = metadata['test_metrics']
                
                analysis['final_performance'] = {
                    'train_rmse': train_metrics.get('rmse', 'N/A'),
                    'test_rmse': test_metrics.get('rmse', 'N/A'),
                    'train_r2': train_metrics.get('r2', 'N/A'),
                    'test_r2': test_metrics.get('r2', 'N/A'),
                    'overfitting_ratio': (test_metrics.get('rmse', 1) ** 2) / (train_metrics.get('rmse', 1) ** 2) if train_metrics.get('rmse', 0) > 0 else 'N/A'
                }
            
            # Training completion analysis
            if 'epoch' in metadata and 'total_epochs' in metadata:
                completion_ratio = metadata['epoch'] / metadata.get('total_epochs', 100)
                analysis['training_completion'] = {
                    'stopped_at_epoch': metadata['epoch'],
                    'total_epochs': metadata.get('total_epochs', 100),
                    'completion_ratio': completion_ratio,
                    'early_stopped': metadata.get('early_stopped', False)
                }
        
        analysis['checkpoint_metadata'] = metadata
        
        return analysis
    
    
    def print_detailed_channel_analysis(self):
        """Print basic channel utilization information"""
        print("\n🔍 BASIC CHANNEL UTILIZATION SUMMARY")
        print("=" * 50)
        
        channel_analysis = self.analysis_results.get('weight_patterns', {}).get('channel_utilization', {})
        
        if not channel_analysis:
            print("❌ No channel analysis available")
            return
        
        print("\nChannel utilization overview:")
        for layer_name, channel_info in channel_analysis.items():
            if isinstance(channel_info.get('output_channels'), dict):
                out_info = channel_info['output_channels']
                total_channels = channel_info.get('total_channels', 0)
                underutilized_ratio = out_info.get('underutilized_ratio', 0)
                channel_efficiency = out_info.get('channel_efficiency', 0)
                
                print(f"\n🔬 {layer_name}:")
                print(f"   Channels: {total_channels}")
                print(f"   Underutilized ratio: {underutilized_ratio:.1%}")
                print(f"   Channel efficiency: {channel_efficiency:.4f}")
                print(f"   Weight norm range: {out_info.get('min_activation', 0):.4f} - {out_info.get('max_activation', 0):.4f}")
        
        total_channels = sum(info.get('total_channels', 0) for info in channel_analysis.values())
        print(f"\n📊 SUMMARY:")
        print(f"   Total analyzed channels: {total_channels}")
        print(f"   Total analyzed layers: {len(channel_analysis)}")
        print("   Analysis complete.")
        
        return channel_analysis

    def run_comprehensive_analysis(self):
        """Run all analysis modules and compile results"""
        print("🚀 Starting Comprehensive Model Analysis...")
        print("=" * 60)
        
        # Load model
        if not self.load_model():
            return False
        
        # Run individual analyses
        self.analysis_results['architecture_efficiency'] = self.analyze_architecture_efficiency()
        self.analysis_results['weight_patterns'] = self.analyze_weight_patterns()
        self.analysis_results['attention_mechanisms'] = self.analyze_attention_mechanisms()
        self.analysis_results['feature_activation_patterns'] = self.analyze_feature_activation_patterns()
        self.analysis_results['training_convergence'] = self.analyze_training_convergence()
        
        # Run new processor-specific analysis
        self.analysis_results['processor_information_handling'] = self.analyze_processor_information_handling()
        
        # Run other advanced analyses
        self.analysis_results['gradient_patterns'] = self.analyze_gradient_patterns()
        self.analysis_results['layer_contributions'] = self.analyze_layer_contributions()
        self.analysis_results['parameter_sensitivity'] = self.analyze_parameter_sensitivity()
        self.analysis_results['computational_complexity'] = self.analyze_computational_complexity()
        self.analysis_results['optimization_recommendations'] = self.generate_optimization_recommendations()
        
        # Run NEW detailed activation analysis
        self.analysis_results['detailed_layer_activations'] = self.analyze_detailed_layer_activations()

        return True
    
    def print_analysis_summary(self):
        """Print a comprehensive analysis summary"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE MODEL ANALYSIS SUMMARY")
        print("="*80)
        
        # Model Overview
        print("\n🏗️ ARCHITECTURE OVERVIEW:")
        arch_stats = self.analysis_results['architecture_efficiency']['parameter_stats']
        print(f"   Total Parameters: {arch_stats['total_parameters']:,}")
        print(f"   Trainable Parameters: {arch_stats['trainable_parameters']:,}")
        
        # Parameter Distribution
        print("\n📊 PARAMETER DISTRIBUTION:")
        layer_percentages = self.analysis_results['architecture_efficiency']['layer_percentages']
        for layer_type, percentage in sorted(layer_percentages.items(), key=lambda x: x[1], reverse=True):
            if percentage > 0:
                print(f"   {layer_type.replace('_', ' ').title()}: {percentage:.1f}%")
        
        # Performance Analysis
        print("\n📈 PERFORMANCE ANALYSIS:")
        perf_analysis = self.analysis_results['training_convergence'].get('final_performance', {})
        if perf_analysis:
            print(f"   Train RMSE: {perf_analysis.get('train_rmse', 'N/A')}")
            print(f"   Test RMSE: {perf_analysis.get('test_rmse', 'N/A')}")
            print(f"   Train R²: {perf_analysis.get('train_r2', 'N/A')}")
            print(f"   Test R²: {perf_analysis.get('test_r2', 'N/A')}")
            overfitting_ratio = perf_analysis.get('overfitting_ratio', 'N/A')
            if isinstance(overfitting_ratio, (int, float)):
                print(f"   Overfitting Ratio: {overfitting_ratio:.2f}")
                if overfitting_ratio > 10:
                    print(f"   ⚠️  Severe overfitting warning!")
                elif overfitting_ratio > 5:
                    print(f"   ⚠️  Overfitting warning")
            else:
                print(f"   Overfitting Ratio: {overfitting_ratio}")
        
        # Layer Efficiency Analysis
        print("\n⚖️ LAYER EFFICIENCY ANALYSIS:")
        weight_patterns = self.analysis_results.get('weight_patterns', {})
        layer_efficiency = weight_patterns.get('layer_efficiency_analysis', {})
        
        inefficient_count = 0
        redundant_count = 0
        for layer_name, metrics in layer_efficiency.items():
            if metrics.get('sparsity_level', 0) > 0.8:
                inefficient_count += 1
            if metrics.get('redundancy_score', 0) > 0.7:
                redundant_count += 1
        
        print(f"   Total Layers Analyzed: {len(layer_efficiency)}")
        print(f"   High Sparsity Layers (>80%): {inefficient_count}")
        print(f"   Redundant Layers (>70% similarity): {redundant_count}")
        
        # Channel Utilization Analysis
        print("\n📊 CHANNEL UTILIZATION ANALYSIS:")
        channel_utilization = weight_patterns.get('channel_utilization', {})
        if channel_utilization:
            underutilized_layers = 0
            total_potential_reduction = 0
            
            for layer_name, channel_info in channel_utilization.items():
                current_channels = channel_info.get('total_channels', 0)
                recommended_channels = channel_info.get('recommended_channels', current_channels)
                underutilization = channel_info.get('output_channels', {}).get('underutilized_ratio', 0)
                
                if underutilization > 0.3:
                    underutilized_layers += 1
                    reduction = current_channels - recommended_channels
                    total_potential_reduction += reduction
            
            print(f"   Conv Layers Analyzed: {len(channel_utilization)}")
            print(f"   Underutilized Layers (>30%): {underutilized_layers}")
            if total_potential_reduction > 0:
                print(f"   Potential Channel Reduction: {total_potential_reduction} channels")
        
        # Feature Activation Analysis
        print("\n🔥 FEATURE ACTIVATION ANALYSIS:")
        activation_analysis = self.analysis_results.get('feature_activation_patterns', {})
        activation_summary = activation_analysis.get('activation_summary', {})
        
        if activation_summary and 'error' not in activation_summary:
            print(f"   Layers Analyzed: {activation_summary.get('total_layers_analyzed', 0)}")
            print(f"   Overall Sparsity: {activation_summary.get('overall_sparsity', 0):.3f}")
            print(f"   Dead Neuron Ratio: {activation_summary.get('overall_dead_neuron_ratio', 0):.3f}")
            
            dead_layers = len(activation_summary.get('layers_with_dead_neurons', []))
            if dead_layers > 0:
                print(f"   ⚠️  {dead_layers} layers have dead neurons (>30%)")
        else:
            print("   ⚠️  Activation analysis failed - model forward pass issue")
        
        # Attention Analysis
        print("\n👁️ ATTENTION MECHANISM ANALYSIS:")
        attention_analysis = self.analysis_results['attention_mechanisms']
        print(f"   CBAM Modules Found: {attention_analysis['cbam_module_count']}")
        if attention_analysis['cbam_modules']:
            print(f"   Module Locations: {', '.join(attention_analysis['cbam_modules'])}")
        
        attention_effectiveness = attention_analysis.get('attention_effectiveness', {})
        if attention_effectiveness:
            avg_param_count = np.mean([metrics.get('parameter_count', 0) for metrics in attention_effectiveness.values()])
            print(f"   Average Attention Parameters: {avg_param_count:.0f}")
        
        # Weight Analysis Summary
        print("\n⚖️ WEIGHT PATTERN SUMMARY:")
        weight_summary = self.analysis_results['weight_patterns']['weight_distribution_summary']
        for layer_type, stats in weight_summary.items():
            if stats['count'] > 0:
                print(f"   {layer_type.replace('_', ' ').title()}: μ={stats['mean']:.4f}, σ={stats['std']:.4f}")
        
        # Training Summary
        print("\n🎯 TRAINING SUMMARY:")
        training_completion = self.analysis_results['training_convergence'].get('training_completion', {})
        if training_completion:
            stopped_epoch = training_completion.get('stopped_at_epoch', 'N/A')
            total_epochs = training_completion.get('total_epochs', 'N/A')
            early_stopped = training_completion.get('early_stopped', False)
            print(f"   Stopped at Epoch: {stopped_epoch}/{total_epochs}")
            print(f"   Early Stopped: {'Yes' if early_stopped else 'No'}")
        
        print("\n" + "="*80)
        print("✅ Analysis Complete!")
        print("="*80)
    
    def print_processor_analysis_summary(self):
        """Print detailed processor information handling analysis"""
        print("\n" + "="*80)
        print("🔬 PROCESSOR INFORMATION HANDLING ANALYSIS")
        print("="*80)
        
        processor_analysis = self.analysis_results.get('processor_information_handling', {})
        
        if 'error' in processor_analysis:
            print(f"❌ Processor analysis failed: {processor_analysis['error']}")
            return
        
        # Adsorbate processor analysis
        ads_analysis = processor_analysis.get('adsorbate_processor', {})
        if ads_analysis:
            print("\n🧪 ADSORBATE PROCESSOR ANALYSIS:")
            ads_input = ads_analysis.get('input_stats', {})
            print(f"   Input Characteristics:")
            print(f"     Sparsity: {ads_input.get('sparsity', 0):.1%}")
            print(f"     Magnitude: {ads_input.get('magnitude', 0):.4f}")
            print(f"     Concentration: {ads_input.get('concentration', 0):.4f}")
            
            ads_flow = ads_analysis.get('information_flow', {})
            print(f"   Information Flow Quality: {ads_flow.get('quality', 'unknown').upper()}")
            print(f"   Average Info Quality Score: {ads_flow.get('average_info_quality', 0):.4f}")
            
            ads_issues = ads_flow.get('issues', [])
            if ads_issues:
                print(f"   ⚠️ Issues Found: {len(ads_issues)}")
                for issue in ads_issues[:3]:  # Show first 3 issues
                    print(f"     - {issue}")
            else:
                print(f"   ✅ No critical issues detected")
        
        # Solvent processor analysis
        solv_analysis = processor_analysis.get('solvent_processor', {})
        if solv_analysis:
            print("\n💧 SOLVENT PROCESSOR ANALYSIS:")
            solv_input = solv_analysis.get('input_stats', {})
            print(f"   Input Characteristics:")
            print(f"     Sparsity: {solv_input.get('sparsity', 0):.1%}")
            print(f"     Magnitude: {solv_input.get('magnitude', 0):.4f}")
            print(f"     Uniformity: {solv_input.get('uniformity', 0):.4f}")
            
            solv_flow = solv_analysis.get('information_flow', {})
            print(f"   Information Flow Quality: {solv_flow.get('quality', 'unknown').upper()}")
            print(f"   Average Info Quality Score: {solv_flow.get('average_info_quality', 0):.4f}")
            
            solv_issues = solv_flow.get('issues', [])
            if solv_issues:
                print(f"   ⚠️ Issues Found: {len(solv_issues)}")
                for issue in solv_issues[:3]:  # Show first 3 issues
                    print(f"     - {issue}")
            else:
                print(f"   ✅ No critical issues detected")
        
        # Comparison and suitability
        comparison = processor_analysis.get('information_comparison', {})
        suitability = processor_analysis.get('processor_suitability', {})
        
        if comparison:
            print("\n⚖️ PROCESSOR COMPARISON:")
            quality_comp = comparison.get('quality_comparison', {})
            print(f"   Adsorbate Quality: {quality_comp.get('adsorbate', 'unknown')}")
            print(f"   Solvent Quality: {quality_comp.get('solvent', 'unknown')}")
            print(f"   Better Processor: {comparison.get('better_processor', 'unknown')}")
            print(f"   Score Ratio (Solvent/Adsorbate): {comparison.get('score_ratio', 0):.2f}")
        
        if suitability:
            print("\n🎯 PROCESSOR SUITABILITY ASSESSMENT:")
            ads_suit = suitability.get('adsorbate_processor', {})
            solv_suit = suitability.get('solvent_processor', {})
            
            print(f"   Adsorbate Processor: {ads_suit.get('suitability', 'unknown').replace('_', ' ').upper()}")
            if ads_suit.get('critical_issues'):
                print(f"     Critical Issues: {len(ads_suit.get('critical_issues', []))}")
            
            print(f"   Solvent Processor: {solv_suit.get('suitability', 'unknown').replace('_', ' ').upper()}")
            if solv_suit.get('critical_issues'):
                print(f"     Critical Issues: {len(solv_suit.get('critical_issues', []))}")
        
        print("\n" + "="*80)

    def print_optimization_recommendations(self):
        """Print detailed optimization recommendations"""
        print("\n" + "="*80)
        print("💡 MODEL OPTIMIZATION RECOMMENDATIONS")
        print("="*80)
        
        recommendations = self.analysis_results.get('optimization_recommendations', {})
        
        if not recommendations:
            print("❌ No optimization recommendations available")
            return
        
        # Priority Actions
        priority_actions = recommendations.get('priority_actions', [])
        if priority_actions:
            print("\n🚨 PRIORITY ACTIONS (URGENT):")
            for i, action in enumerate(priority_actions, 1):
                print(f"\n{i}. {action['action']}:")
                for step in action.get('steps', []):
                    print(f"   {step}")
        
        # Parameter Reduction
        param_reductions = recommendations.get('parameter_reduction', [])
        if param_reductions:
            print("\n📉 PARAMETER REDUCTION RECOMMENDATIONS:")
            for i, rec in enumerate(param_reductions, 1):
                print(f"\n{i}. {rec['action']} ({rec['priority']} Priority)")
                print(f"   Reason: {rec['reason']}")
                print(f"   Suggestion: {rec['suggestion']}")
        
        # Architecture Changes
        arch_changes = recommendations.get('architecture_changes', [])
        if arch_changes:
            print("\n🏗️ ARCHITECTURE CHANGE RECOMMENDATIONS:")
            for i, rec in enumerate(arch_changes, 1):
                print(f"\n{i}. {rec['action']} ({rec['priority']} Priority)")
                print(f"   Reason: {rec['reason']}")
                print(f"   Suggestion: {rec['suggestion']}")
        
        # Training Improvements
        training_improvements = recommendations.get('training_improvements', [])
        if training_improvements:
            print("\n📚 TRAINING IMPROVEMENT RECOMMENDATIONS:")
            for i, rec in enumerate(training_improvements, 1):
                print(f"\n{i}. {rec['action']} ({rec['priority']} Priority)")
                print(f"   Reason: {rec['reason']}")
                print(f"   Suggestion: {rec['suggestion']}")
        
        print("\n" + "="*80)
        print("💡 OPTIMIZATION SUMMARY COMPLETE")
        print("="*80)
    
    def print_advanced_analysis_summary(self):
        """Print summary of advanced analyses"""
        print("\n" + "="*80)
        print("🔬 ADVANCED ANALYSIS SUMMARY")
        print("="*80)
        
        # Gradient Analysis
        grad_analysis = self.analysis_results.get('gradient_patterns', {})
        if grad_analysis and 'error' not in grad_analysis:
            grad_health = grad_analysis.get('gradient_flow_health', {})
            print("\n📊 GRADIENT FLOW ANALYSIS:")
            print(f"   Overall Gradient Norm: {grad_health.get('overall_gradient_norm', 0):.6f}")
            print(f"   Mean Gradient Norm: {grad_health.get('mean_gradient_norm', 0):.6f}")
            print(f"   Gradient Flow Ratio: {grad_health.get('gradient_flow_ratio', 0):.2f}")
            print(f"   Healthy Gradient Flow: {'✅ Yes' if grad_health.get('healthy_gradient_flow', False) else '⚠️ No'}")
            
            vanishing = grad_health.get('vanishing_gradient_layers', [])
            exploding = grad_health.get('exploding_gradient_layers', [])
            if vanishing:
                print(f"   ⚠️ Vanishing Gradients: {len(vanishing)} layers")
            if exploding:
                print(f"   ⚠️ Exploding Gradients: {len(exploding)} layers")
        
        # Layer Contributions
        contrib_analysis = self.analysis_results.get('layer_contributions', {})
        if contrib_analysis and 'error' not in contrib_analysis:
            contributions = contrib_analysis.get('layer_contributions', {})
            ranking = contrib_analysis.get('contribution_ranking', {})
            print(f"\n🎯 LAYER CONTRIBUTION ANALYSIS:")
            print(f"   Layers Analyzed: {len(contributions)}")
            
            # Show top 3 contributing layers
            top_layers = sorted(ranking.items(), key=lambda x: x[1]['rank'])[:3]
            if top_layers:
                print("   Top Contributing Layers:")
                for layer_name, rank_info in top_layers:
                    importance = rank_info['relative_importance']
                    print(f"     {rank_info['rank']}. {layer_name}: {importance:.3f}")
        
        # Parameter Sensitivity
        sens_analysis = self.analysis_results.get('parameter_sensitivity', {})
        if sens_analysis and 'error' not in sens_analysis:
            sensitivity = sens_analysis.get('parameter_sensitivity', {})
            ranking = sens_analysis.get('sensitivity_ranking', {})
            print(f"\n🔍 PARAMETER SENSITIVITY ANALYSIS:")
            print(f"   Parameters Analyzed: {len(sensitivity)}")
            
            # Show most sensitive parameters
            top_params = sorted(ranking.items(), key=lambda x: x[1]['rank'])[:3]
            if top_params:
                print("   Most Sensitive Parameters:")
                for param_name, rank_info in top_params:
                    importance = rank_info['importance_ratio']
                    print(f"     {rank_info['rank']}. {param_name}: {importance:.3f}")
        
        # Computational Complexity
        comp_analysis = self.analysis_results.get('computational_complexity', {})
        if comp_analysis and 'error' not in comp_analysis:
            summary = comp_analysis.get('complexity_summary', {})
            print(f"\n⚡ COMPUTATIONAL COMPLEXITY:")
            print(f"   Total Parameters: {summary.get('total_parameters', 0):,}")
            print(f"   Memory Usage: {summary.get('total_memory_mb', 0):.2f} MB")
            print(f"   Estimated FLOPs: {summary.get('estimated_flops', 0):,}")
            print(f"   Params/FLOP Ratio: {summary.get('params_per_flop', 0):.6f}")
        
        # NEW: Detailed Layer Activation Analysis
        detailed_act_analysis = self.analysis_results.get('detailed_layer_activations', {})
        if detailed_act_analysis and 'error' not in detailed_act_analysis:
            print(f"\n🔬 DETAILED LAYER ACTIVATION ANALYSIS:")
            
            # Activation health summary
            health = detailed_act_analysis.get('activation_health', {})
            if health:
                print(f"   Overall Activation Health: {health.get('overall_health', 'unknown').upper()}")
                print(f"   Total Layers Analyzed: {health.get('total_layers_analyzed', 0)}")
                print(f"   Healthy Layers: {health.get('healthy_layers', 0)}")
                print(f"   Warning Layers: {health.get('warning_layers', 0)}")
                print(f"   Critical Layers: {health.get('critical_layers', 0)}")
                print(f"   Health Ratio: {health.get('health_ratio', 0):.2%}")
                
                # Show major issues
                issues = health.get('issues', {})
                for issue_type, affected_layers in issues.items():
                    if len(affected_layers) > 0:
                        print(f"   ⚠️ {issue_type.replace('_', ' ').title()}: {len(affected_layers)} layers")
            
            # Bottleneck analysis
            bottlenecks = detailed_act_analysis.get('bottleneck_analysis', {})
            if bottlenecks and 'error' not in bottlenecks:
                print(f"\n🚧 BOTTLENECK IDENTIFICATION:")
                for scenario, layer_metrics in bottlenecks.items():
                    if scenario != 'error' and len(layer_metrics) > 0:
                        print(f"   {scenario.title()} Scenario - Top Bottlenecks:")
                        for i, (layer_name, score, _) in enumerate(layer_metrics[:3], 1):
                            print(f"     {i}. {layer_name}: {score:.3f}")
            
            # Layer behavior comparison
            layer_comparison = detailed_act_analysis.get('layer_comparison', {})
            if layer_comparison and 'error' not in layer_comparison:
                adaptable_layers = [(name, data.get('adaptability_score', 0)) 
                                  for name, data in layer_comparison.items() if isinstance(data, dict)]
                adaptable_layers.sort(key=lambda x: x[1], reverse=True)
                
                if adaptable_layers:
                    print(f"\n🔀 LAYER ADAPTABILITY ANALYSIS:")
                    print(f"   Most Adaptable Layers (top 3):")
                    for i, (layer_name, score) in enumerate(adaptable_layers[:3], 1):
                        print(f"     {i}. {layer_name}: {score:.4f}")
        
        print("\n" + "="*80)
    
    def analyze_detailed_layer_activations(self):
        """
        Comprehensive analysis of layer-by-layer activations with multiple test inputs.
        This provides deep insights into each layer's behavior and effectiveness.
        """
        print("\n🔬 Analyzing Detailed Layer Activations...")
        
        analysis = {}
        
        try:
            device = next(self.model.parameters()).device
            
            # Create multiple test scenarios to understand layer behavior
            test_scenarios = {
                'random_normal': torch.randn(2, 28, 20, 20, 20).to(device),
                'sparse_adsorbate': self._create_sparse_adsorbate_input().to(device),
                'dense_solvent': self._create_dense_solvent_input().to(device),
                'mixed_realistic': self._create_realistic_mixed_input().to(device)
            }
            
            layer_analysis = {}
            activation_hooks = []
            
            def create_hook(layer_name):
                def hook_fn(module, input, output):
                    if isinstance(output, torch.Tensor):
                        # Comprehensive activation analysis
                        act_np = output.detach().cpu().numpy()
                        
                        layer_analysis[layer_name] = {
                            'output_shape': list(act_np.shape),
                            'mean_activation': float(np.mean(act_np)),
                            'std_activation': float(np.std(act_np)),
                            'max_activation': float(np.max(act_np)),
                            'min_activation': float(np.min(act_np)),
                            'sparsity': float(np.sum(np.abs(act_np) < 1e-6) / act_np.size),
                            'effective_rank': self._compute_effective_rank(act_np),
                            'information_content': self._compute_information_content(act_np),
                            'gradient_readiness': self._assess_gradient_readiness(act_np),
                            'channel_diversity': self._compute_channel_diversity(act_np),
                            'spatial_concentration': self._compute_spatial_concentration_detailed(act_np),
                            'dead_neurons': float(np.sum(np.max(np.abs(act_np.reshape(act_np.shape[0], -1)), axis=0) < 1e-6)),
                            'saturation_ratio': self._compute_saturation_ratio(act_np)
                        }
                return hook_fn
            
            # Register hooks for all key layers
            key_layers = []
            for name, module in self.model.named_modules():
                if isinstance(module, (torch.nn.Conv3d, torch.nn.Linear)) and not any(x in name for x in ['shortcut', 'downsample']):
                    key_layers.append((name, module))
                    
            print(f"   📊 Analyzing {len(key_layers)} key layers across {len(test_scenarios)} scenarios...")
            
            # Analyze each scenario
            scenario_results = {}
            for scenario_name, test_input in test_scenarios.items():
                print(f"   🧪 Testing scenario: {scenario_name}")
                
                # Register hooks
                layer_analysis = {}
                activation_hooks = []
                for layer_name, module in key_layers:
                    hook = module.register_forward_hook(create_hook(f"{scenario_name}_{layer_name}"))
                    activation_hooks.append(hook)
                
                # Forward pass
                self.model.eval()
                with torch.no_grad():
                    try:
                        _ = self.model(test_input)
                        scenario_results[scenario_name] = dict(layer_analysis)
                    except Exception as e:
                        print(f"⚠️ Forward pass failed for {scenario_name}: {e}")
                        scenario_results[scenario_name] = {'error': str(e)}
                
                # Remove hooks
                for hook in activation_hooks:
                    hook.remove()
            
            # Consolidate and analyze results
            analysis['scenario_results'] = scenario_results
            analysis['layer_comparison'] = self._compare_layer_behaviors(scenario_results)
            analysis['bottleneck_analysis'] = self._identify_bottlenecks(scenario_results)
            analysis['activation_health'] = self._assess_overall_activation_health(scenario_results)
            
        except Exception as e:
            print(f"⚠️ Detailed activation analysis failed: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _create_sparse_adsorbate_input(self):
        """Create realistic sparse adsorbate-like input"""
        x = torch.zeros(2, 28, 20, 20, 20)
        # Fill adsorbate channels (first 14) with sparse central data
        center = 10
        for b in range(2):
            for c in range(14):
                # Small central region with values
                x[b, c, center-2:center+3, center-2:center+3, center-2:center+3] = torch.randn(5, 5, 5) * 0.5
        return x
    
    def _create_dense_solvent_input(self):
        """Create realistic dense solvent-like input"""
        x = torch.zeros(2, 28, 20, 20, 20)
        # Fill solvent channels (last 14) with dense distributed data
        for b in range(2):
            for c in range(14, 28):
                x[b, c] = torch.randn(20, 20, 20) * 0.2 + 0.1
        return x
    
    def _create_realistic_mixed_input(self):
        """Create realistic mixed adsorbate-solvent input"""
        x = torch.zeros(2, 28, 20, 20, 20)
        center = 10
        
        for b in range(2):
            # Adsorbate channels (sparse, central)
            for c in range(14):
                x[b, c, center-1:center+2, center-1:center+2, center-1:center+2] = torch.randn(3, 3, 3) * 0.3
            
            # Solvent channels (dense, distributed, avoiding adsorbate region)
            for c in range(14, 28):
                solvent_data = torch.randn(20, 20, 20) * 0.15
                # Reduce intensity near adsorbate
                distance_weight = self._create_distance_weight(20, center)
                x[b, c] = solvent_data * distance_weight
        
        return x
    
    def _create_distance_weight(self, size, center):
        """Create distance-based weight matrix"""
        coords = torch.stack(torch.meshgrid(
            torch.arange(size, dtype=torch.float),
            torch.arange(size, dtype=torch.float),
            torch.arange(size, dtype=torch.float),
            indexing='ij'
        ), dim=-1)
        
        center_coord = torch.tensor([center, center, center], dtype=torch.float)
        distances = torch.norm(coords - center_coord, dim=-1)
        # Weight increases with distance from center (solvent avoids adsorbate)
        weights = torch.clamp(distances / 5.0, 0.3, 1.0)
        return weights
    
    def _compute_effective_rank(self, activation):
        """Compute effective rank of activation matrix"""
        try:
            if len(activation.shape) < 2:
                return 0.0
            
            # Reshape to 2D: (batch*spatial, channels) or (batch, features)
            if len(activation.shape) > 2:
                reshaped = activation.reshape(-1, activation.shape[1])
            else:
                reshaped = activation
            
            # Compute SVD
            if reshaped.shape[0] >= reshaped.shape[1]:
                _, s, _ = np.linalg.svd(reshaped, full_matrices=False)
            else:
                _, s, _ = np.linalg.svd(reshaped.T, full_matrices=False)
            
            # Effective rank based on normalized singular values
            s_norm = s / np.sum(s)
            effective_rank = np.exp(-np.sum(s_norm * np.log(s_norm + 1e-12)))
            
            return float(effective_rank)
        except:
            return 0.0
    
    def _compute_information_content(self, activation):
        """Estimate information content using entropy"""
        try:
            # Flatten and compute histogram
            flat_act = activation.flatten()
            hist, _ = np.histogram(flat_act, bins=50, density=True)
            hist = hist + 1e-12  # Avoid log(0)
            hist = hist / np.sum(hist)
            
            # Compute entropy
            entropy = -np.sum(hist * np.log2(hist))
            return float(entropy)
        except:
            return 0.0
    
    def _assess_gradient_readiness(self, activation):
        """Assess how ready the activation is for gradient flow"""
        try:
            # Check for vanishing/exploding activations
            mean_abs = np.mean(np.abs(activation))
            max_abs = np.max(np.abs(activation))
            
            # Good gradient readiness: not too small, not too large, good variance
            if mean_abs < 1e-6:
                return 'vanishing'
            elif max_abs > 100:
                return 'exploding'
            elif 0.01 <= mean_abs <= 10 and np.std(activation) > 1e-3:
                return 'good'
            else:
                return 'moderate'
        except:
            return 'unknown'
    
    def _compute_channel_diversity(self, activation):
        """Compute diversity among channels"""
        try:
            if len(activation.shape) < 2:
                return 0.0
                
            # Get channel dimension (usually dim 1)
            if len(activation.shape) >= 4:  # Conv layer: (batch, channels, h, w, d)
                channel_means = np.mean(activation, axis=(0, 2, 3, 4))
            else:  # Linear layer: (batch, features)
                channel_means = np.mean(activation, axis=0)
            
            # Compute coefficient of variation
            if np.mean(channel_means) > 1e-6:
                diversity = np.std(channel_means) / np.mean(np.abs(channel_means))
            else:
                diversity = 0.0
                
            return float(diversity)
        except:
            return 0.0
    
    def _compute_spatial_concentration_detailed(self, activation):
        """Detailed spatial concentration analysis"""
        try:
            if len(activation.shape) < 4:
                return 0.0
                
            # Sum across batch and channel: (H, W, D)
            spatial_sum = np.sum(np.abs(activation), axis=(0, 1))
            
            if np.sum(spatial_sum) < 1e-6:
                return 0.0
            
            # Compute center of mass
            shape = spatial_sum.shape
            center = [s//2 for s in shape]
            
            total_sum = np.sum(spatial_sum)
            weighted_sum = 0.0
            
            for h in range(shape[0]):
                for w in range(shape[1]):
                    for d in range(shape[2]):
                        distance = np.sqrt((h-center[0])**2 + (w-center[1])**2 + (d-center[2])**2)
                        weighted_sum += spatial_sum[h, w, d] * distance
            
            # Normalize by maximum possible distance and total sum
            max_distance = np.sqrt(sum([(s-1)**2 for s in shape]))
            concentration = 1.0 - (weighted_sum / (total_sum * max_distance))
            
            return float(max(0.0, concentration))
        except:
            return 0.0
    
    def _compute_saturation_ratio(self, activation):
        """Compute ratio of saturated activations"""
        try:
            # Assume saturated if |activation| > 5.0 (for typical activations)
            saturated_count = np.sum(np.abs(activation) > 5.0)
            total_count = activation.size
            return float(saturated_count / total_count)
        except:
            return 0.0
    
    def _compare_layer_behaviors(self, scenario_results):
        """Compare layer behaviors across different scenarios"""
        comparison = {}
        
        try:
            # Extract layer names from first successful scenario
            layer_names = []
            for scenario_name, results in scenario_results.items():
                if 'error' not in results:
                    layer_names = [k.split('_', 1)[1] for k in results.keys() if '_' in k]
                    break
            
            # Compare each layer across scenarios
            for layer_name in layer_names:
                layer_comparison = {}
                scenario_metrics = {}
                
                for scenario_name in scenario_results.keys():
                    full_key = f"{scenario_name}_{layer_name}"
                    if full_key in scenario_results.get(scenario_name, {}):
                        scenario_metrics[scenario_name] = scenario_results[scenario_name][full_key]
                
                if len(scenario_metrics) >= 2:
                    # Compare key metrics across scenarios
                    metrics_to_compare = ['sparsity', 'information_content', 'effective_rank', 'channel_diversity']
                    
                    for metric in metrics_to_compare:
                        values = [data.get(metric, 0) for data in scenario_metrics.values()]
                        layer_comparison[f'{metric}_variance'] = float(np.var(values))
                        layer_comparison[f'{metric}_range'] = float(max(values) - min(values))
                    
                    # Overall adaptability score
                    adaptability = np.mean([layer_comparison.get(f'{m}_variance', 0) for m in metrics_to_compare])
                    layer_comparison['adaptability_score'] = adaptability
                    
                    comparison[layer_name] = layer_comparison
            
        except Exception as e:
            comparison['error'] = str(e)
        
        return comparison
    
    def _identify_bottlenecks(self, scenario_results):
        """Identify potential bottlenecks in the network"""
        bottlenecks = {}
        
        try:
            # Analyze information flow degradation
            for scenario_name, results in scenario_results.items():
                if 'error' in results:
                    continue
                    
                layer_metrics = []
                for key, data in results.items():
                    if '_' in key:
                        layer_name = key.split('_', 1)[1]
                        info_content = data.get('information_content', 0)
                        effective_rank = data.get('effective_rank', 0)
                        sparsity = data.get('sparsity', 1)
                        
                        # Bottleneck score: low information, low rank, high sparsity
                        bottleneck_score = (1 - info_content/10) + (1 - effective_rank/100) + sparsity
                        layer_metrics.append((layer_name, bottleneck_score, data))
                
                # Sort by bottleneck score
                layer_metrics.sort(key=lambda x: x[1], reverse=True)
                bottlenecks[scenario_name] = layer_metrics[:5]  # Top 5 bottlenecks
        
        except Exception as e:
            bottlenecks['error'] = str(e)
        
        return bottlenecks
    
    def _assess_overall_activation_health(self, scenario_results):
        """Assess overall health of activation patterns"""
        health_assessment = {}
        
        try:
            total_layers = 0
            healthy_layers = 0
            warning_layers = 0
            critical_layers = 0
            
            issues = {
                'high_sparsity': [],
                'low_information': [],
                'poor_gradient_readiness': [],
                'high_saturation': [],
                'dead_neurons': []
            }
            
            for scenario_name, results in scenario_results.items():
                if 'error' in results:
                    continue
                    
                for key, data in results.items():
                    if '_' in key:
                        total_layers += 1
                        layer_name = key.split('_', 1)[1]
                        
                        # Health criteria
                        sparsity = data.get('sparsity', 0)
                        info_content = data.get('information_content', 0)
                        gradient_readiness = data.get('gradient_readiness', 'unknown')
                        saturation = data.get('saturation_ratio', 0)
                        dead_neurons = data.get('dead_neurons', 0)
                        
                        # Classify layer health
                        is_healthy = True
                        
                        if sparsity > 0.9:
                            issues['high_sparsity'].append(f"{scenario_name}_{layer_name}")
                            is_healthy = False
                        
                        if info_content < 1.0:
                            issues['low_information'].append(f"{scenario_name}_{layer_name}")
                            is_healthy = False
                        
                        if gradient_readiness in ['vanishing', 'exploding']:
                            issues['poor_gradient_readiness'].append(f"{scenario_name}_{layer_name}")
                            is_healthy = False
                        
                        if saturation > 0.1:
                            issues['high_saturation'].append(f"{scenario_name}_{layer_name}")
                            is_healthy = False
                        
                        if dead_neurons > 0.3:
                            issues['dead_neurons'].append(f"{scenario_name}_{layer_name}")
                            is_healthy = False
                        
                        if is_healthy:
                            healthy_layers += 1
                        elif gradient_readiness == 'vanishing' or sparsity > 0.95:
                            critical_layers += 1
                        else:
                            warning_layers += 1
            
            health_assessment = {
                'total_layers_analyzed': total_layers,
                'healthy_layers': healthy_layers,
                'warning_layers': warning_layers,
                'critical_layers': critical_layers,
                'health_ratio': healthy_layers / (total_layers + 1e-6),
                'issues': issues,
                'overall_health': 'good' if healthy_layers / (total_layers + 1e-6) > 0.7 else ('moderate' if healthy_layers / (total_layers + 1e-6) > 0.4 else 'poor')
            }
            
        except Exception as e:
            health_assessment['error'] = str(e)
        
        return health_assessment
        
        # Gradient Analysis
        grad_analysis = self.analysis_results.get('gradient_patterns', {})
        if grad_analysis and 'error' not in grad_analysis:
            grad_health = grad_analysis.get('gradient_flow_health', {})
            print("\n📊 GRADIENT FLOW ANALYSIS:")
            print(f"   Overall Gradient Norm: {grad_health.get('overall_gradient_norm', 0):.6f}")
            print(f"   Mean Gradient Norm: {grad_health.get('mean_gradient_norm', 0):.6f}")
            print(f"   Gradient Flow Ratio: {grad_health.get('gradient_flow_ratio', 0):.2f}")
            print(f"   Healthy Gradient Flow: {'✅ Yes' if grad_health.get('healthy_gradient_flow', False) else '⚠️ No'}")
            
            vanishing = grad_health.get('vanishing_gradient_layers', [])
            exploding = grad_health.get('exploding_gradient_layers', [])
            if vanishing:
                print(f"   ⚠️ Vanishing Gradients: {len(vanishing)} layers")
            if exploding:
                print(f"   ⚠️ Exploding Gradients: {len(exploding)} layers")
        
        # Layer Contributions
        contrib_analysis = self.analysis_results.get('layer_contributions', {})
        if contrib_analysis and 'error' not in contrib_analysis:
            contributions = contrib_analysis.get('layer_contributions', {})
            ranking = contrib_analysis.get('contribution_ranking', {})
            print(f"\n🎯 LAYER CONTRIBUTION ANALYSIS:")
            print(f"   Layers Analyzed: {len(contributions)}")
            
            # Show top 3 contributing layers
            top_layers = sorted(ranking.items(), key=lambda x: x[1]['rank'])[:3]
            if top_layers:
                print("   Top Contributing Layers:")
                for layer_name, rank_info in top_layers:
                    importance = rank_info['relative_importance']
                    print(f"     {rank_info['rank']}. {layer_name}: {importance:.3f}")
        
        # Parameter Sensitivity
        sens_analysis = self.analysis_results.get('parameter_sensitivity', {})
        if sens_analysis and 'error' not in sens_analysis:
            sensitivity = sens_analysis.get('parameter_sensitivity', {})
            ranking = sens_analysis.get('sensitivity_ranking', {})
            print(f"\n🔍 PARAMETER SENSITIVITY ANALYSIS:")
            print(f"   Parameters Analyzed: {len(sensitivity)}")
            
            # Show most sensitive parameters
            top_params = sorted(ranking.items(), key=lambda x: x[1]['rank'])[:3]
            if top_params:
                print("   Most Sensitive Parameters:")
                for param_name, rank_info in top_params:
                    importance = rank_info['importance_ratio']
                    print(f"     {rank_info['rank']}. {param_name}: {importance:.3f}")
        
        # Computational Complexity
        comp_analysis = self.analysis_results.get('computational_complexity', {})
        if comp_analysis and 'error' not in comp_analysis:
            summary = comp_analysis.get('complexity_summary', {})
            print(f"\n⚡ COMPUTATIONAL COMPLEXITY:")
            print(f"   Total Parameters: {summary.get('total_parameters', 0):,}")
            print(f"   Memory Usage: {summary.get('total_memory_mb', 0):.2f} MB")
            print(f"   Estimated FLOPs: {summary.get('estimated_flops', 0):,}")
            print(f"   Params/FLOP Ratio: {summary.get('params_per_flop', 0):.6f}")
        
        print("\n" + "="*80)

if __name__ == "__main__":
    # Model to analyze
    # model_name = "model_2_7_2466635-epochs_100-bs_32-lr_0.0002-splits_5-grid_16.0_0.8-fold_0.pth"
    model_name = "model_2_8_2487596-epochs_100-bs_32-lr_0.0008-splits_5-grid_16.0_0.8-fold_0.pth"
    model_path = os.path.join(get_paths("output_model_cnn"), model_name)
    
    # Run analysis
    analyzer = TrainedModelAnalyzer(model_path)
    analyzer.run_comprehensive_analysis()
    
    # Print all analysis results
    analyzer.print_analysis_summary()
    analyzer.print_processor_analysis_summary()  # Add new processor analysis
    analyzer.print_advanced_analysis_summary()
    analyzer.print_optimization_recommendations()
    
    # Print detailed channel analysis with explanations
    analyzer.print_detailed_channel_analysis()
    