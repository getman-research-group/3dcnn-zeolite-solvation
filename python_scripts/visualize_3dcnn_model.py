# -*- coding: utf-8 -*-
"""
visualize_3dcnn_model.py
Professional 3D-CNN-Transformer hybrid model visualization tool.
Create publication-quality figures showing the dual-branch architecture with separated channel groups.

Features:
- 3D box visualization of feature maps at each layer
- Dual-branch processing visualization (adsorbate vs solvent channels)
- Separated channel groups analysis (14 adsorbate + 14 solvent features)
- Group interaction CBAM attention visualization
- Vision Transformer integration with cross-attention
- Adsorbate-solvent interaction modeling visualization
- Publication-ready diagrams for molecular interaction energy prediction
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import inspect
from collections import OrderedDict

# Set style for publication-quality figures
plt.style.use(['seaborn-v0_8-paper', 'seaborn-v0_8-whitegrid'])
sns.set_palette("husl")

# Import the hybrid 3D-CNN-Transformer model
from model_3d_cnn_2_5 import AttentionCNNTransformer_2_5
from core.path import get_paths

class ModelArchitectureExtractor:
    """
    Automatically extract model architecture from the actual model definition.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.layer_info = []
        self.feature_shapes = {}
        
    def extract_architecture(self, input_shape=(1, 28, 20, 20, 20)):
        """
        Extract architecture by analyzing the model structure and forward pass.
        
        Args:
            input_shape: Input tensor shape for tracing (28 channels = 14 adsorbate + 14 solvent)
        """
        print("🔍 Automatically extracting model architecture...")
        
        # Step 1: Extract static architecture from model definition
        self._extract_static_architecture()
        
        # Step 2: Trace through model to get dynamic shapes
        self._trace_forward_pass(input_shape)
        
        # Step 3: Detect CBAM locations
        self._detect_attention_mechanisms()
        
        # Step 4: Build visualization-ready architecture
        self._build_visualization_architecture()
        
        return self.architecture_flow
    
    def _extract_static_architecture(self):
        """Extract layer information from model structure."""
        print("  📋 Analyzing model structure...")
        
        # Analyze the model's named modules
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                layer_info = {
                    'name': name,
                    'module': module,
                    'type': self._classify_layer_type(module),
                    'params': sum(p.numel() for p in module.parameters())
                }
                self.layer_info.append(layer_info)
    
    def _classify_layer_type(self, module):
        """Classify the type of a layer module for hybrid CNN-Transformer."""
        if isinstance(module, nn.Conv3d):
            return 'conv3d'
        elif isinstance(module, nn.MaxPool3d):
            return 'maxpool3d'
        elif isinstance(module, nn.AvgPool3d):
            return 'avgpool3d'
        elif isinstance(module, nn.AdaptiveAvgPool3d):
            return 'adaptiveavgpool3d'
        elif isinstance(module, nn.Linear):
            return 'linear'
        elif isinstance(module, nn.BatchNorm3d):
            return 'batchnorm3d'
        elif isinstance(module, nn.BatchNorm1d):
            return 'batchnorm1d'
        elif isinstance(module, nn.LayerNorm):
            return 'layernorm'
        elif isinstance(module, nn.ReLU):
            return 'relu'
        elif isinstance(module, nn.LeakyReLU):
            return 'leakyrelu'
        elif isinstance(module, nn.SiLU):
            return 'silu'
        elif isinstance(module, nn.Sigmoid):
            return 'sigmoid'
        elif isinstance(module, nn.Dropout):
            return 'dropout'
        elif isinstance(module, nn.Dropout3d):
            return 'dropout3d'
        elif isinstance(module, nn.Flatten):
            return 'flatten'
        elif 'CBAM' in module.__class__.__name__:
            return 'cbam'
        elif 'Attention' in module.__class__.__name__:
            return 'attention'
        elif 'VisionTransformer' in module.__class__.__name__:
            return 'vision_transformer'
        elif 'PositionalEncoding' in module.__class__.__name__:
            return 'positional_encoding'
        elif 'MultiHeadAttention' in module.__class__.__name__:
            return 'multihead_attention'
        elif 'TransformerEncoder' in module.__class__.__name__:
            return 'transformer_encoder'
        elif 'Residual' in module.__class__.__name__:
            return 'resblock'
        else:
            return 'other'
    
    def _trace_forward_pass(self, input_shape):
        """Trace forward pass to extract feature shapes."""
        print("  🚀 Tracing forward pass to extract feature shapes...")
        
        # Create hooks to capture intermediate features
        hooks = []
        feature_shapes = {}
        
        def create_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    feature_shapes[name] = output.shape
                elif isinstance(output, (list, tuple)):
                    feature_shapes[name] = [o.shape if isinstance(o, torch.Tensor) else str(o) for o in output]
            return hook
        
        # Register hooks on key modules for hybrid model
        key_modules = [
            'input_attention', 'initial_conv', 'layer1', 'layer2', 'layer3', 'layer4', 
            'pool1', 'pool2', 'adaptive_pool', 'vision_transformer', 'classifier'
        ]
        for module_name in key_modules:
            if hasattr(self.model, module_name):
                module = getattr(self.model, module_name)
                hook = module.register_forward_hook(create_hook(module_name))
                hooks.append(hook)
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(input_shape)
            try:
                _ = self.model(dummy_input)
                self.feature_shapes = feature_shapes
                print(f"    ✅ Captured shapes for {len(feature_shapes)} layers")
            except Exception as e:
                print(f"    ⚠️ Forward pass failed: {e}")
                self.feature_shapes = {}
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    def _detect_attention_mechanisms(self):
        """Detect where attention mechanisms are used."""
        print("  🎯 Detecting attention mechanisms...")
        
        self.attention_locations = []
        for name, module in self.model.named_modules():
            if any(keyword in module.__class__.__name__.lower() for keyword in ['cbam', 'attention']):
                self.attention_locations.append(name)
                print(f"    Found attention: {name} ({module.__class__.__name__})")
    
    def _build_visualization_architecture(self):
        """Build architecture flow for visualization."""
        print("  🎨 Building visualization architecture...")
        
        self.architecture_flow = []
        position_x = 2  # UPDATED: Start further from edge
        
        # Input layer for format
        input_layer = {
            'name': 'Input Voxel Grids',
            'type': 'input',
            'shape': (20, 20, 20),
            'channels': 28,
            'description': 'Separated Channel Groups\n14 Adsorbate + 14 Solvent',
            'position': (position_x, 5),
            'auto_detected': True
        }
        self.architecture_flow.append(input_layer)
        position_x += 5  # UPDATED: Increase spacing from 3 to 5
        
        # Analyze main components
        components = self._analyze_main_components()
        
        for component in components:
            component['position'] = (position_x, 5)
            component['auto_detected'] = True
            self.architecture_flow.append(component)
            position_x += 5  # UPDATED: Increase spacing from 3 to 5
        
        # Output layer
        output_layer = {
            'name': 'Output',
            'type': 'output',
            'shape': None,
            'channels': 1,
            'description': 'Interaction Energy\n(eV)',
            'position': (position_x, 5),
            'auto_detected': True
        }
        self.architecture_flow.append(output_layer)
        
        print(f"    ✅ Built architecture with {len(self.architecture_flow)} components")
    
    def _analyze_main_components(self):
        """Analyze and group main model components for dual-branch CNN-Transformer."""
        components = []
        
        # Dual-branch processing
        # Adsorbate branch
        if hasattr(self.model, 'adsorbate_processor'):
            components.append({
                'name': 'Adsorbate Branch',
                'type': 'dual_branch_ads',
                'shape': (20, 20, 20),
                'channels': 24,
                'description': 'Adsorbate Processing\n14→24 channels\n3×3×3 kernels (sparse, central)'
            })
        
        # Solvent branch  
        if hasattr(self.model, 'solvent_processor'):
            components.append({
                'name': 'Solvent Branch',
                'type': 'dual_branch_solv',
                'shape': (20, 20, 20),
                'channels': 48,
                'description': 'Solvent Processing\n14→48 channels\n5×5×5 kernels (dense, distributed)'
            })
        
        # Interaction fusion
        if hasattr(self.model, 'interaction_conv'):
            components.append({
                'name': 'Interaction Fusion',
                'type': 'interaction_fusion',
                'shape': (20, 20, 20),
                'channels': 32,
                'description': 'Adsorbate-Solvent Fusion\n72→32 channels\nGroup Interaction CBAM'
            })
        
        # CNN backbone layers
        layer_configs = [
            ('layer1', 'pool1', 1, 48, (20, 20, 20), (10, 10, 10)),
            ('layer2', 'pool2', 2, 64, (10, 10, 10), (5, 5, 5)), 
            ('layer3', None, 3, 80, (5, 5, 5), (5, 5, 5)),
            ('layer4', None, 4, 96, (5, 5, 5), (5, 5, 5))
        ]
        
        for layer_name, pool_name, layer_num, channels, input_shape, output_shape in layer_configs:
            if hasattr(self.model, layer_name):
                components.append({
                    'name': f'ResBlock{layer_num} + CBAM',
                    'type': 'resblock',
                    'shape': output_shape,
                    'channels': channels,
                    'description': f'Residual Block + CBAM\n{channels} channels\nMolecular interactions'
                })
                
                if pool_name and hasattr(self.model, pool_name):
                    components.append({
                        'name': f'AvgPool3D',
                        'type': 'pool',
                        'shape': output_shape,
                        'channels': channels,
                        'description': f'3D Average Pooling\n{input_shape[0]}³→{output_shape[0]}³'
                    })
        
        # Adaptive pooling
        if hasattr(self.model, 'adaptive_pool'):
            components.append({
                'name': 'Adaptive Pool3D',
                'type': 'pool',
                'shape': (3, 3, 3),
                'channels': 96,
                'description': 'Adaptive Average Pooling\n→(3,3,3) = 27 patches'
            })
        
        # Vision Transformer
        if hasattr(self.model, 'use_transformer') and self.model.use_transformer:
            components.append({
                'name': 'Vision Transformer 3D',
                'type': 'vision_transformer',
                'shape': (27,),  # 27 spatial patches
                'channels': self.model.transformer_dim,
                'description': f'ViT with Cross-Attention\n{self.model.transformer_layers} layers, {self.model.transformer_heads} heads\ndim={self.model.transformer_dim}'
            })
        
        # Classifier
        if hasattr(self.model, 'classifier'):
            components.append({
                'name': 'Enhanced Classifier',
                'type': 'fc',
                'shape': None,
                'channels': 1,
                'description': 'Multi-layer Classifier\nGELU + BatchNorm + Dropout\n→Energy Prediction (eV)'
            })
        
        return components
    
    def _analyze_initial_conv(self):
        """Analyze initial convolution layer."""
        conv_module = self.model.initial_conv
        
        # Extract conv layer info
        conv_layer = None
        for module in conv_module.modules():
            if isinstance(module, nn.Conv3d):
                conv_layer = module
                break
        
        if conv_layer:
            in_channels = conv_layer.in_channels
            out_channels = conv_layer.out_channels
            kernel_size = conv_layer.kernel_size[0]
        else:
            in_channels, out_channels, kernel_size = 15, 32, 3
        
        shape = self.feature_shapes.get('initial_conv', (None, out_channels, 20, 20, 20))
        if isinstance(shape, torch.Size):
            spatial_shape = shape[2:5]
        else:
            spatial_shape = (20, 20, 20)
        
        return {
            'name': 'Initial Conv3D',
            'type': 'conv',
            'shape': spatial_shape,
            'channels': out_channels,
            'description': f'{kernel_size}×{kernel_size}×{kernel_size} Conv + BatchNorm + ReLU\n{in_channels}→{out_channels} channels'
        }
    
    def _analyze_resblock(self, layer_name, layer_num):
        """Analyze residual block."""
        resblock = getattr(self.model, layer_name)
        
        # Try to extract channel info
        out_channels = 32 * (2 ** (layer_num - 1))  # Default progression
        for module in resblock.modules():
            if isinstance(module, nn.Conv3d):
                out_channels = module.out_channels
                break
        
        # Get shape from traced forward pass
        shape = self.feature_shapes.get(layer_name, (None, out_channels, 20, 20, 20))
        if isinstance(shape, torch.Size):
            spatial_shape = shape[2:5]
            channels = shape[1]
        else:
            spatial_shape = (20, 20, 20)
            channels = out_channels
        
        # Check if CBAM is present
        has_cbam = any('cbam' in name.lower() for name, _ in resblock.named_modules())
        
        description = f'Residual Block\n{channels} channels'
        if has_cbam:
            description += '\nwith CBAM Attention'
        
        return {
            'name': f'ResBlock + {"CBAM" if has_cbam else "Conv"}',
            'type': 'resblock',
            'shape': spatial_shape,
            'channels': channels,
            'description': description,
            'has_attention': has_cbam
        }
    
    def _analyze_pooling(self, pool_name):
        """Analyze pooling layer."""
        pool_module = getattr(self.model, pool_name)
        
        # Get shape from traced forward pass
        shape = self.feature_shapes.get(pool_name, (None, 32, 10, 10, 10))
        if isinstance(shape, torch.Size):
            spatial_shape = shape[2:5]
            channels = shape[1]
        else:
            spatial_shape = (10, 10, 10)
            channels = 32
        
        # Determine pooling type
        if isinstance(pool_module, nn.MaxPool3d):
            pool_type = "Max Pooling"
            kernel_size = pool_module.kernel_size
        elif isinstance(pool_module, nn.AvgPool3d):
            pool_type = "Avg Pooling"
            kernel_size = pool_module.kernel_size
        else:
            pool_type = "Pooling"
            kernel_size = 2
        
        # Estimate input shape (rough approximation)
        input_shape = tuple(s * kernel_size for s in spatial_shape)
        
        return {
            'name': f'{"Max" if "Max" in pool_type else "Avg"}Pool3D',
            'type': 'pool',
            'shape': spatial_shape,
            'channels': channels,
            'description': f'{kernel_size}×{kernel_size}×{kernel_size} {pool_type}\n{input_shape[0]}³→{spatial_shape[0]}³'
        }
    
    def _analyze_global_pooling(self):
        """Analyze global pooling layer."""
        shape = self.feature_shapes.get('global_avg_pool', (None, 128, 1, 1, 1))
        if isinstance(shape, torch.Size):
            channels = shape[1]
        else:
            channels = 128
        
        return {
            'name': 'Global AvgPool3D',
            'type': 'global_pool',
            'shape': (1, 1, 1),
            'channels': channels,
            'description': 'Adaptive Average Pooling\n→(1,1,1)'
        }
    
    def _analyze_classifier(self):
        """Analyze classifier layers."""
        classifier = self.model.classifier
        
        components = []
        
        # Flatten layer
        components.append({
            'name': 'Flatten',
            'type': 'fc',
            'shape': None,
            'channels': 128,  # From global pooling
            'description': 'Flatten to Vector\n(128,)'
        })
        
        # Extract linear layer dimensions
        linear_dims = []
        for module in classifier.modules():
            if isinstance(module, nn.Linear):
                linear_dims.append((module.in_features, module.out_features))
        
        if linear_dims:
            # MLP layers
            components.append({
                'name': 'MLP Classifier',
                'type': 'fc',
                'shape': None,
                'channels': [dim[1] for dim in linear_dims],
                'description': f'Multi-Layer Perceptron\n{" → ".join(str(dim[1]) for dim in linear_dims)}'
            })
        else:
            # Fallback
            components.append({
                'name': 'MLP Classifier',
                'type': 'fc',
                'shape': None,
                'channels': [256, 128, 64, 1],
                'description': 'Multi-Layer Perceptron\n256→128→64→1'
            })
        
        return components
    
    def _analyze_input_attention(self):
        """Analyze input attention layer."""
        return {
            'name': 'Input Attention',
            'type': 'attention',
            'shape': (20, 20, 20),
            'channels': 15,
            'description': 'CBAM Channel Attention\nLearn Feature Importance'
        }
    
    def _analyze_adaptive_pooling(self):
        """Analyze adaptive pooling layer."""
        shape = self.feature_shapes.get('adaptive_pool', (None, 96, 2, 2, 2))
        if isinstance(shape, torch.Size):
            channels = shape[1]
        else:
            channels = 96
        
        return {
            'name': 'Adaptive Pool3D',
            'type': 'pool',
            'shape': (2, 2, 2),
            'channels': channels,
            'description': 'Adaptive Average Pooling\n→(2,2,2)'
        }
    
    def _analyze_vision_transformer(self):
        """Analyze Vision Transformer component."""
        transformer_dim = getattr(self.model, 'transformer_dim', 200)
        transformer_heads = getattr(self.model, 'transformer_heads', 8)
        transformer_layers = getattr(self.model, 'transformer_layers', 2)
        
        return {
            'name': 'Vision Transformer',
            'type': 'vision_transformer',
            'shape': (8,),  # 2x2x2 = 8 patches
            'channels': transformer_dim,
            'description': f'ViT 3D\n{transformer_layers} layers, {transformer_heads} heads\ndim={transformer_dim}'
        }

class CNN3DVisualizer:
    """
    Professional 3D-CNN visualization tool with automatic architecture extraction.
    """
    
    def __init__(self, model: AttentionCNNTransformer_2_5, figsize: Tuple[int, int] = (20, 12)):
        """
        Initialize the visualizer with a hybrid 3D-CNN-Transformer model.
        
        Args:
            model: AttentionCNNTransformer_2_5 instance
            figsize: Figure size for plots
        """
        self.model = model
        self.figsize = figsize
        self.colors = {
            'input': '#9370DB',                # Medium slate blue for input
            'dual_branch_ads': '#FF69B4',      # Hot pink for adsorbate branch
            'dual_branch_solv': '#00CED1',     # Dark turquoise for solvent branch  
            'interaction_fusion': '#FF6347',   # Tomato for interaction fusion
            'conv': '#4169E1',                 # Royal blue for conv layers
            'resblock': '#1E90FF',             # Dodger blue for residual blocks
            'pool': '#32CD32',                 # Lime green for pooling
            'attention': '#FFD700',            # Gold for attention
            'transformer': '#FF6347',          # Tomato for transformer
            'vision_transformer': '#FF4500',   # Orange red for ViT
            'global_pool': '#FF6347', # Tomato for global pooling
            'fc': '#FF69B4',         # Hot pink for fully connected
            'output': '#98FB98',     # Pale green for output
            'arrow': '#2F4F4F',      # Dark slate gray for arrows
            'hybrid': '#DA70D6'      # Orchid for hybrid components
        }
        
        # Automatically extract architecture
        self.extractor = ModelArchitectureExtractor(model)
        self.architecture_flow = self.extractor.extract_architecture()
        
        print(f"\n🎯 Auto-detected architecture with {len(self.architecture_flow)} components:")
        for i, layer in enumerate(self.architecture_flow):
            print(f"  {i+1}. {layer['name']} ({layer['type']}) - {layer.get('channels', 'N/A')} channels")
    
    def create_3d_box(self, ax, position, size, color, alpha=0.7, edge_color='black'):
        """
        Create a 3D box to represent a feature map.
        
        Args:
            ax: 3D matplotlib axis
            position: (x, y, z) position of the box center
            size: (width, height, depth) of the box
            color: Color of the box
            alpha: Transparency
            edge_color: Color of the edges
        """
        x, y, z = position
        w, h, d = size
        
        # Define the vertices of the box
        vertices = [
            [x-w/2, y-h/2, z-d/2], [x+w/2, y-h/2, z-d/2],
            [x+w/2, y+h/2, z-d/2], [x-w/2, y+h/2, z-d/2],
            [x-w/2, y-h/2, z+d/2], [x+w/2, y-h/2, z+d/2],
            [x+w/2, y+h/2, z+d/2], [x-w/2, y+h/2, z+d/2]
        ]
        
        # Define the 12 edges of the box
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
            [vertices[1], vertices[2], vertices[6], vertices[5]]   # right
        ]
        
        # Create the 3D polygon collection
        poly3d = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor=edge_color, linewidth=1)
        ax.add_collection3d(poly3d)
        
        return poly3d
    
    def draw_3d_arrow(self, ax, start, end, color='black', arrow_style='->', mutation_scale=20):
        """
        Draw a 3D arrow between two points.
        """
        from matplotlib.patches import FancyArrowPatch
        from mpl_toolkits.mplot3d import proj3d
        
        class Arrow3D(FancyArrowPatch):
            def __init__(self, xs, ys, zs, *args, **kwargs):
                FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
                self._verts3d = xs, ys, zs

            def do_3d_projection(self, renderer=None):
                xs3d, ys3d, zs3d = self._verts3d
                xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
                self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
                return np.min(zs)
        
        arrow = Arrow3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                       mutation_scale=mutation_scale, lw=2, arrowstyle=arrow_style, color=color)
        ax.add_artist(arrow)
        return arrow
    
    def create_3d_architecture_diagram(self, save_path: Optional[str] = None, dpi: int = 300):
        """
        Create a 3D architectural diagram showing the layer-by-layer flow.
        """
        fig = plt.figure(figsize=(36, 14))  # INCREASED: Much wider for better canvas usage
        
        # CRITICAL: Force the figure to use the full specified size and disable auto-adjustment
        fig.set_tight_layout(False)
        
        # FIXED: Create subplot with manual positioning to maximize space usage
        ax = fig.add_subplot(111, projection='3d')
        
        # FIXED: Calculate proper axis limits based on layer positions
        max_x_pos = max([layer['position'][0] for layer in self.architecture_flow])
        
        # IMPROVED: Set axis limits to properly fill the canvas
        ax.set_xlim(-3, max_x_pos + 10)  # EXPANDED: More padding on both sides
        ax.set_ylim(-2, 17)              # EXPANDED: More vertical space
        ax.set_zlim(-2, 12)              # EXPANDED: More depth space
        
        # CRITICAL: Force aspect ratio to utilize full canvas width
        ax.set_box_aspect([max_x_pos + 13, 19, 14])  # ADJUSTED: Better proportions
        
        # Title with better positioning
        fig.suptitle('Dual-Branch CNN-Transformer: Separated Channel Processing → Interaction Fusion → Global Context', 
                    fontsize=18, fontweight='bold', y=0.96)
        
        # Draw each layer as 3D boxes
        box_positions = []
        for i, layer in enumerate(self.architecture_flow):
            if layer['shape'] is not None:
                # REDUCED: Make boxes smaller to prevent overlapping
                base_size = 2.5  # REDUCED from 4.5 to 2.5 to prevent overlap
                if layer['type'] == 'input':
                    size_multiplier = 1.2  # REDUCED from 1.8 to 1.2
                else:
                    max_dim = max(layer['shape'])
                    size_multiplier = 0.8 + (max_dim / 20.0) * 0.8  # REDUCED scaling factor
                
                w = size_multiplier * base_size
                h = size_multiplier * base_size  
                d = size_multiplier * base_size
                
                # Position in 3D space - better distribution
                x_pos = layer['position'][0]
                y_pos = 8.5  # ADJUSTED: Better Y center
                z_pos = 5    # Center in Z
                
                # Create the 3D box
                color = self.colors[layer['type']]
                self.create_3d_box(ax, (x_pos, y_pos, z_pos), (w, h, d), color, alpha=0.8)
                
                box_positions.append((x_pos, y_pos, z_pos))
                
                # FIXED: Better text spacing relative to smaller boxes
                # Layer name - positioned higher relative to box size
                ax.text(x_pos, y_pos, z_pos + w*2 + 2.0, layer['name'], 
                       fontsize=15, fontweight='bold', ha='center', va='bottom')  # ADJUSTED: Dynamic spacing based on box size
                
                # Channel information - better spacing
                if isinstance(layer['channels'], int):
                    ax.text(x_pos, y_pos, z_pos + w*1.5 + 1.5, f'{layer["channels"]} channels', 
                           fontsize=13, ha='center', va='bottom', style='italic')  # ADJUSTED: Dynamic spacing
                
                # Shape information - better spacing
                if layer['shape']:
                    if len(layer['shape']) >= 3:
                        shape_str = f'{layer["shape"][0]}×{layer["shape"][1]}×{layer["shape"][2]}'
                    elif len(layer['shape']) == 1:
                        # For transformer (sequence length)
                        shape_str = f'seq={layer["shape"][0]}'
                    else:
                        shape_str = f'{layer["shape"]}'
                    ax.text(x_pos, y_pos, z_pos + w*1.0 + 1.0, shape_str, 
                           fontsize=12, ha='center', va='bottom', color='darkblue')  # ADJUSTED: Dynamic spacing
            
            else:
                # Handle FC layers (no 3D shape) - reduced sizing
                x_pos = layer['position'][0]
                y_pos = 8.5  # ADJUSTED: Match other layers
                z_pos = 5
                
                # REDUCED: Smaller FC layer boxes to prevent overlap
                if layer['name'] == 'Flatten':
                    box_size = (0.8, 2.5, 2.5)  # REDUCED from (1.0, 4.5, 4.5)
                    self.create_3d_box(ax, (x_pos, y_pos, z_pos), box_size, 
                                     self.colors['fc'], alpha=0.8)
                elif 'MLP' in layer['name']:
                    mlp_channels = layer['channels']
                    for j, channels in enumerate(mlp_channels):
                        offset_z = z_pos + (j - len(mlp_channels)/2) * 1.0  # REDUCED spacing from 1.5 to 1.0
                        size = 1.8 - j * 0.2  # REDUCED base size from 2.5 to 1.8
                        self.create_3d_box(ax, (x_pos, y_pos, offset_z), 
                                         (1.0, size, size), self.colors['fc'], alpha=0.8)  # REDUCED box size
                elif layer['type'] == 'output':
                    self.create_3d_box(ax, (x_pos, y_pos, z_pos), (1.0, 1.0, 1.0),  # REDUCED from (1.2, 1.2, 1.2)
                                     self.colors['output'], alpha=0.9)
                
                box_positions.append((x_pos, y_pos, z_pos))
                
                # Add labels with adjusted spacing for smaller boxes
                ax.text(x_pos, y_pos, z_pos + 4.5, layer['name'],  # REDUCED spacing from 6.5 to 4.5
                       fontsize=15, fontweight='bold', ha='center', va='bottom')
        
        # Draw arrows between consecutive layers - adjusted for smaller boxes
        for i in range(len(box_positions) - 1):
            start_pos = box_positions[i]
            end_pos = box_positions[i + 1]
            
            # ADJUSTED: Better arrow spacing for smaller boxes
            distance = end_pos[0] - start_pos[0]
            arrow_offset = min(2.0, distance * 0.3)  # REDUCED offset from 3.5 to 2.0
            arrow_start = (start_pos[0] + arrow_offset, start_pos[1], start_pos[2])
            arrow_end = (end_pos[0] - arrow_offset, end_pos[1], end_pos[2])
            
            self.draw_3d_arrow(ax, arrow_start, arrow_end, color=self.colors['arrow'])
        
        # Add CBAM attention annotations with adjusted positioning for smaller boxes
        cbam_layers = [2, 4, 6]  # ResBlock layers with CBAM
        for layer_idx in cbam_layers:
            if layer_idx < len(box_positions):
                x_pos, y_pos, z_pos = box_positions[layer_idx]
                
                # ADJUSTED: Better positioned attention indicators for smaller boxes
                self.create_3d_box(ax, (x_pos, y_pos - 4.5, z_pos), (2.5, 0.8, 2.5),  # REDUCED size and spacing
                                 self.colors['attention'], alpha=0.9)
                ax.text(x_pos, y_pos - 15.5, z_pos, 'CBAM\nAttention',  # ADJUSTED spacing
                       fontsize=13, fontweight='bold', ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8))
        
        # ENHANCED: Better axis configuration for maximum canvas usage
        ax.set_xlabel('Model Flow Direction →', fontsize=22, fontweight='bold', labelpad=30)
        ax.set_ylabel('Feature Height', fontsize=16, labelpad=20)
        ax.set_zlabel('Feature Depth', fontsize=16, labelpad=20)
        
        # OPTIMIZED: Set viewing angle to maximize canvas usage
        ax.view_init(elev=20, azim=-65)  # ADJUSTED: Better angle for wide canvas
        
        # ENHANCED: Make X-axis more prominent
        ax.xaxis.label.set_color('darkred')
        ax.xaxis.label.set_fontweight('bold')
        
        # IMPROVED: Better grid and pane settings
        ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        ax.grid(True, alpha=0.3)
        
        # Customize panes for better visibility
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('darkred')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.xaxis.pane.set_alpha(0.2)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
        
        # CRITICAL: Force exact canvas usage
        plt.rcParams['figure.autolayout'] = False
        
        if save_path:
            # CRITICAL: Save with exact dimensions
            fig.set_size_inches(36, 14, forward=True)
            plt.savefig(save_path, dpi=dpi, 
                       facecolor='white', edgecolor='none',
                       bbox_inches=None,
                       pad_inches=0)
            print(f"3D architecture diagram saved: {save_path}")
        
        # plt.show()  # Removed: only save, don't display
    
    def create_2d_detailed_flow(self, save_path: Optional[str] = None, dpi: int = 300):
        """
        Create a detailed 2D flow diagram with precise layer information.
        """
        fig, ax = plt.subplots(figsize=(32, 12))  # UPDATED: Increased width from 28 to 32
        ax.set_xlim(0, 65)  # UPDATED: Increased from 36 to 65 to match 3D spacing
        ax.set_ylim(0, 14)  # UPDATED: Slightly increased height
        ax.axis('off')
        
        # Title
        ax.text(32.5, 13, ' Dual-Branch CNN-Transformer: Separated Channels → Interaction Fusion → Energy',
                fontsize=24, fontweight='bold', ha='center')
        ax.text(32.5, 12.2, 'Adsorbate & Solvent Processing + Group Interaction CBAM + Vision Transformer', 
                fontsize=16, ha='center', style='italic', color='gray')
        
        # Draw detailed layers
        for i, layer in enumerate(self.architecture_flow):
            x, y = layer['position']
            
            # UPDATED: Larger boxes with better proportions
            if layer['type'] == 'input':
                width, height = 3.0, 3.0  # UPDATED: Increased from 2.5 to 3.0
                box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                                   boxstyle="round,pad=0.15", 
                                   facecolor=self.colors[layer['type']],
                                   edgecolor='black', linewidth=2, alpha=0.8)
            elif layer['shape'] is not None:
                # 3D layers
                width = 2.5  # UPDATED: Increased from 2.0 to 2.5
                height = 2.5  # UPDATED: Increased from 2.0 to 2.5
                box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                                   boxstyle="round,pad=0.1",
                                   facecolor=self.colors[layer['type']],
                                   edgecolor='black', linewidth=1.5, alpha=0.8)
            else:
                # FC layers
                width = 2.0  # UPDATED: Increased from 1.5 to 2.0
                height = 2.5  # UPDATED: Increased from 2.0 to 2.5
                box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                                   boxstyle="round,pad=0.1",
                                   facecolor=self.colors[layer['type']],
                                   edgecolor='black', linewidth=1.5, alpha=0.8)
            
            ax.add_patch(box)
            
            # Layer name
            ax.text(x, y + 0.5, layer['name'], fontsize=10, fontweight='bold', 
                   ha='center', va='center')
            
            # Layer details
            if layer['shape']:
                if len(layer['shape']) >= 3:
                    shape_text = f'{layer["shape"][0]}×{layer["shape"][1]}×{layer["shape"][2]}'
                elif len(layer['shape']) == 1:
                    # For transformer (sequence length)
                    shape_text = f'seq={layer["shape"][0]}'
                else:
                    shape_text = f'{layer["shape"]}'
                ax.text(x, y, shape_text, fontsize=9, ha='center', va='center',
                       fontweight='bold', color='darkblue')
                
                if isinstance(layer['channels'], int):
                    ax.text(x, y - 0.5, f'{layer["channels"]} channels', 
                           fontsize=8, ha='center', va='center', style='italic')
            else:
                # FC layer details
                if isinstance(layer['channels'], list):
                    channels_text = '→'.join(map(str, layer['channels']))
                    ax.text(x, y, channels_text, fontsize=9, ha='center', va='center',
                           fontweight='bold')
                else:
                    ax.text(x, y, f'{layer["channels"]}', fontsize=9, ha='center', va='center',
                           fontweight='bold')
            
            # Description below
            ax.text(x, y - 1.5, layer['description'], fontsize=8, ha='center', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcyan', alpha=0.8))
            
            # Add arrows
            if i < len(self.architecture_flow) - 1:
                next_x = self.architecture_flow[i + 1]['position'][0]
                arrow_start_x = x + width/2 + 0.1
                arrow_end_x = next_x - 1.0
                ax.arrow(arrow_start_x, y, arrow_end_x - arrow_start_x, 0,
                        head_width=0.2, head_length=0.15, fc='black', ec='black', linewidth=2)
        
        # Add CBAM detailed explanation
        cbam_explanation_box = FancyBboxPatch((2, 1), 32, 2,
                                            boxstyle="round,pad=0.15",
                                            facecolor='lightyellow',
                                            edgecolor='orange', linewidth=2, alpha=0.9)
        ax.add_patch(cbam_explanation_box)
        
        cbam_text = """CBAM (Convolutional Block Attention Module) Integration:
• Channel Attention: Learns "what" features are important using global average and max pooling
• Spatial Attention: Learns "where" to focus using spatial pooling across channels  
• Applied in each ResidualBlock: ResBlock → CBAM → Output
• Improves feature representation for zeolite-adsorbate interaction learning"""
        
        ax.text(18, 2, cbam_text, fontsize=11, ha='center', va='center', 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"2D detailed flow diagram saved: {save_path}")
        
        # plt.show()  # Removed: only save, don't display
    
    def create_cbam_attention_detail(self, save_path: Optional[str] = None, dpi: int = 300):
        """
        Create a detailed visualization of the CBAM attention mechanism.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('CBAM Attention Mechanism in 3D-CNN', fontsize=18, fontweight='bold')
        
        # 1. Channel Attention Mechanism
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 8)
        ax1.axis('off')
        ax1.set_title('Channel Attention Module', fontsize=14, fontweight='bold')
        
        # Input feature map
        input_box = FancyBboxPatch((1, 3), 2, 2, boxstyle="round,pad=0.1",
                                 facecolor='lightblue', edgecolor='black', linewidth=2)
        ax1.add_patch(input_box)
        ax1.text(2, 4, 'Input\nFeature Map\n(C×H×W×D)', fontsize=10, ha='center', va='center', fontweight='bold')
        
        # Global pooling
        pool_box1 = FancyBboxPatch((4, 4.5), 1.5, 0.8, boxstyle="round,pad=0.05",
                                 facecolor='lightgreen', edgecolor='black')
        ax1.add_patch(pool_box1)
        ax1.text(4.75, 4.9, 'Global\nAvgPool', fontsize=9, ha='center', va='center')
        
        pool_box2 = FancyBboxPatch((4, 2.7), 1.5, 0.8, boxstyle="round,pad=0.05",
                                 facecolor='lightgreen', edgecolor='black')
        ax1.add_patch(pool_box2)
        ax1.text(4.75, 3.1, 'Global\nMaxPool', fontsize=9, ha='center', va='center')
        
        # MLP
        mlp_box = FancyBboxPatch((6.5, 3), 1.5, 2, boxstyle="round,pad=0.1",
                               facecolor='orange', edgecolor='black', linewidth=2)
        ax1.add_patch(mlp_box)
        ax1.text(7.25, 4, 'Shared MLP\n(FC→ReLU→FC)', fontsize=10, ha='center', va='center', fontweight='bold')
        
        # Output weights
        output_box1 = FancyBboxPatch((8.5, 3), 1, 2, boxstyle="round,pad=0.1",
                                   facecolor='yellow', edgecolor='black', linewidth=2)
        ax1.add_patch(output_box1)
        ax1.text(9, 4, 'Channel\nWeights\n(C×1×1×1)', fontsize=9, ha='center', va='center', fontweight='bold')
        
        # Add arrows
        ax1.arrow(3.2, 4, 0.6, 0.7, head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax1.arrow(3.2, 4, 0.6, -0.7, head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax1.arrow(5.7, 4, 0.6, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax1.arrow(8.2, 4, 0.2, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # 2. Spatial Attention Mechanism  
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 8)
        ax2.axis('off')
        ax2.set_title('Spatial Attention Module', fontsize=14, fontweight='bold')
        
        # Input feature map
        input_box2 = FancyBboxPatch((1, 3), 2, 2, boxstyle="round,pad=0.1",
                                  facecolor='lightblue', edgecolor='black', linewidth=2)
        ax2.add_patch(input_box2)
        ax2.text(2, 4, 'Feature Map\n(C×H×W×D)', fontsize=10, ha='center', va='center', fontweight='bold')
        
        # Channel pooling
        pool_box3 = FancyBboxPatch((4, 4.5), 1.5, 0.8, boxstyle="round,pad=0.05",
                                 facecolor='lightcoral', edgecolor='black')
        ax2.add_patch(pool_box3)
        ax2.text(4.75, 4.9, 'Channel\nAvgPool', fontsize=9, ha='center', va='center')
        
        pool_box4 = FancyBboxPatch((4, 2.7), 1.5, 0.8, boxstyle="round,pad=0.05",
                                 facecolor='lightcoral', edgecolor='black')
        ax2.add_patch(pool_box4)
        ax2.text(4.75, 3.1, 'Channel\nMaxPool', fontsize=9, ha='center', va='center')
        
        # Conv layer
        conv_box = FancyBboxPatch((6.5, 3), 1.5, 2, boxstyle="round,pad=0.1",
                                facecolor='mediumpurple', edgecolor='black', linewidth=2)
        ax2.add_patch(conv_box)
        ax2.text(7.25, 4, 'Conv3D\n(7×7×7)', fontsize=10, ha='center', va='center', fontweight='bold')
        
        # Output weights
        output_box2 = FancyBboxPatch((8.5, 3), 1, 2, boxstyle="round,pad=0.1",
                                   facecolor='yellow', edgecolor='black', linewidth=2)
        ax2.add_patch(output_box2)
        ax2.text(9, 4, 'Spatial\nWeights\n(1×H×W×D)', fontsize=9, ha='center', va='center', fontweight='bold')
        
        # Add arrows
        ax2.arrow(3.2, 4, 0.6, 0.7, head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax2.arrow(3.2, 4, 0.6, -0.7, head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax2.arrow(5.7, 4, 0.6, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax2.arrow(8.2, 4, 0.2, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # 3. CBAM Integration in ResidualBlock
        ax3.set_xlim(0, 12)
        ax3.set_ylim(0, 8)
        ax3.axis('off')
        ax3.set_title('CBAM Integration in ResidualBlock', fontsize=14, fontweight='bold')
        
        # Draw the residual block flow
        components = [
            ('Input', 1, 4, 'lightblue'),
            ('Conv3D\n+BN+ReLU', 3, 4, 'lightgreen'),
            ('Conv3D\n+BN', 5.5, 4, 'lightgreen'),
            ('CBAM', 8, 4, 'gold'),
            ('Add', 10, 4, 'lightcoral'),
            ('Output', 11.5, 4, 'lightblue')
        ]
        
        for i, (name, x, y, color) in enumerate(components):
            if name == 'Add':
                # Special circle for addition
                circle = plt.Circle((x, y), 0.3, facecolor=color, edgecolor='black', linewidth=2)
                ax3.add_patch(circle)
                ax3.text(x, y, '+', fontsize=16, ha='center', va='center', fontweight='bold')
            else:
                box = FancyBboxPatch((x-0.6, y-0.4), 1.2, 0.8, boxstyle="round,pad=0.05",
                                   facecolor=color, edgecolor='black', linewidth=1)
                ax3.add_patch(box)
                ax3.text(x, y, name, fontsize=10, ha='center', va='center', fontweight='bold')
            
            # Add arrows
            if i < len(components) - 1:
                next_x = components[i+1][1]
                ax3.arrow(x + 0.7, y, next_x - x - 1.4, 0, 
                         head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # Residual connection
        ax3.arrow(1, 3.2, 8.3, 0, head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2)
        ax3.text(5, 2.8, 'Residual Connection', fontsize=10, ha='center', color='red', fontweight='bold')
        
        # 4. Feature Map Evolution
        ax4.set_xlim(0, 12)
        ax4.set_ylim(0, 8)
        ax4.axis('off')
        ax4.set_title('Feature Map Size Evolution', fontsize=14, fontweight='bold')
        
        # Show feature map size changes
        sizes = [(20, 20, 20), (10, 10, 10), (5, 5, 5), (2, 2, 2), (1, 1, 1)]
        channels = [32, 32, 64, 128, 128]
        x_positions = [2, 4, 6, 8, 10]
        
        for i, ((h, w, d), ch, x) in enumerate(zip(sizes, channels, x_positions)):
            # Calculate relative size for visualization
            rel_size = max(h, w, d) / 20.0  # Normalize to max size
            box_size = 0.3 + rel_size * 1.2
            
            box = FancyBboxPatch((x-box_size/2, 4-box_size/2), box_size, box_size,
                               boxstyle="round,pad=0.02", 
                               facecolor=plt.cm.viridis(i/4), edgecolor='black', linewidth=1)
            ax4.add_patch(box)
            
            ax4.text(x, 4, f'{h}×{w}×{d}', fontsize=9, ha='center', va='center', fontweight='bold', color='white')
            ax4.text(x, 3, f'{ch} ch', fontsize=8, ha='center', va='top', fontweight='bold')
            
            # Add arrows
            if i < len(sizes) - 1:
                ax4.arrow(x + box_size/2 + 0.1, 4, x_positions[i+1] - x - box_size/2 - 0.3, 0,
                         head_width=0.1, head_length=0.1, fc='gray', ec='gray')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"CBAM attention detail saved: {save_path}")
        
        # plt.show()  # Removed: only save, don't display
    
    def create_type2_dual_branch_diagram(self, save_path: Optional[str] = None, dpi: int = 300):
        """
        Create a detailed diagram showing the dual-branch architecture with shape transformations.
        """
        fig = plt.figure(figsize=(28, 24))  # Larger to accommodate ResidualBlock3D detail
        fig.suptitle('Type_2 Dual-Branch CNN-Transformer Architecture: Actual Shape Transformations', 
                    fontsize=20, fontweight='bold', y=0.96)
        
        # Create grid layout for comprehensive view - updated to 6 rows for ResidualBlock3D detail
        gs = fig.add_gridspec(6, 3, height_ratios=[0.5, 0.9, 0.9, 1.0, 1.0, 0.7], hspace=0.2, wspace=0.15)
        
        # 1. Input and Channel Split (top row)
        ax_input = fig.add_subplot(gs[0, :])
        ax_input.set_xlim(0, 24)
        ax_input.set_ylim(0, 4)
        ax_input.axis('off')
        ax_input.set_title('Input Voxel Channel Splitting', fontsize=16, fontweight='bold')
        
        # Input voxel
        input_box = FancyBboxPatch((2, 1), 6, 2, boxstyle="round,pad=0.2",
                                  facecolor='lightblue', edgecolor='navy', linewidth=2)
        ax_input.add_patch(input_box)
        ax_input.text(5, 2, 'Input Voxel Grid\n(B, 28, 20, 20, 20)', 
                     fontsize=12, ha='center', va='center', fontweight='bold')
        
        # Split arrow
        ax_input.arrow(8.5, 2, 2, 0, head_width=0.2, head_length=0.3, fc='black', ec='black', linewidth=2)
        ax_input.text(9.5, 2.8, 'Split', fontsize=11, ha='center', fontweight='bold')
        
        # Adsorbate channel group
        ads_box = FancyBboxPatch((12, 2.5), 4, 1, boxstyle="round,pad=0.1",
                                facecolor='#FF69B4', edgecolor='darkred', linewidth=2)
        ax_input.add_patch(ads_box)
        ax_input.text(14, 3, 'Adsorbate Channels\n(B, 14, 20, 20, 20)', 
                     fontsize=11, ha='center', va='center', fontweight='bold')
        
        # Solvent channel group
        solv_box = FancyBboxPatch((12, 0.5), 4, 1, boxstyle="round,pad=0.1",
                                 facecolor='#00CED1', edgecolor='darkblue', linewidth=2)
        ax_input.add_patch(solv_box)
        ax_input.text(14, 1, 'Solvent Channels\n(B, 14, 20, 20, 20)', 
                     fontsize=11, ha='center', va='center', fontweight='bold')
        
        # 2. Adsorbate Branch Processing (left column)
        ax_ads = fig.add_subplot(gs[1:3, 0])
        ax_ads.set_xlim(0, 10)
        ax_ads.set_ylim(0, 16)
        ax_ads.axis('off')
        ax_ads.set_title('Adsorbate Branch (3×3×3 kernels)', fontsize=14, fontweight='bold', color='darkred')
        
        # Adsorbate processing stages (actual architecture)
        ads_stages = [
            ('Input\n(B, 14, 20, 20, 20)', 5, 14.5, '#FF69B4'),
            ('Conv3d(14→16)\n3×3×3, pad=1\n(B, 16, 20, 20, 20)', 5, 12.5, '#FF1493'),
            ('Conv3d(16→20)\n1×1×1 densify\n(B, 20, 20, 20, 20)', 5, 10.5, '#DC143C'),
            ('Conv3d(20→24)\n3×3×3, pad=1\n(B, 24, 20, 20, 20)', 5, 8.5, '#B22222'),
            ('Output\n(B, 24, 20, 20, 20)', 5, 6.5, '#8B0000')
        ]
        
        for i, (name, x, y, color) in enumerate(ads_stages):
            # Calculate box size based on channel complexity
            if 'Input' in name:
                box_width, box_height = 3, 1.2
            elif 'densify' in name:
                box_width, box_height = 2.8, 1.4  # Slightly larger for 1x1x1 conv
            elif 'Output' in name:
                box_width, box_height = 2.6, 1.0
            else:
                box_width, box_height = 2.8, 1.3
                
            box = FancyBboxPatch((x-box_width/2, y-box_height/2), box_width, box_height,
                               boxstyle="round,pad=0.1", facecolor=color, edgecolor='darkred', linewidth=1)
            ax_ads.add_patch(box)
            ax_ads.text(x, y, name, fontsize=9, ha='center', va='center', fontweight='bold', color='white')
            
            # Add downward arrows
            if i < len(ads_stages) - 1:
                ax_ads.arrow(x, y - box_height/2 - 0.1, 0, -1.2, 
                           head_width=0.3, head_length=0.2, fc='darkred', ec='darkred', linewidth=2)
        
        # 3. Solvent Branch Processing (right column)
        ax_solv = fig.add_subplot(gs[1:3, 2])
        ax_solv.set_xlim(0, 10)
        ax_solv.set_ylim(0, 16)
        ax_solv.axis('off')
        ax_solv.set_title('Solvent Branch (5×5×5 kernels)', fontsize=14, fontweight='bold', color='darkblue')
        
        # Solvent processing stages (actual architecture)
        solv_stages = [
            ('Input\n(B, 14, 20, 20, 20)', 5, 14.5, '#00CED1'),
            ('Conv3d(14→32)\n5×5×5, pad=2\n(B, 32, 20, 20, 20)', 5, 12, '#1E90FF'),
            ('Conv3d(32→48)\n5×5×5, pad=2\n(B, 48, 20, 20, 20)', 5, 9.5, '#4169E1'),
            ('Output\n(B, 48, 20, 20, 20)', 5, 7, '#00008B')
        ]
        
        for i, (name, x, y, color) in enumerate(solv_stages):
            # Calculate box size based on channel complexity
            if 'Input' in name:
                box_width, box_height = 3, 1.2
            elif 'Output' in name:
                box_width, box_height = 3.2, 1.0
            else:
                box_width, box_height = 3.4, 1.4  # Larger boxes for 5x5x5 convs
                
            box = FancyBboxPatch((x-box_width/2, y-box_height/2), box_width, box_height,
                               boxstyle="round,pad=0.1", facecolor=color, edgecolor='darkblue', linewidth=1)
            ax_solv.add_patch(box)
            ax_solv.text(x, y, name, fontsize=9, ha='center', va='center', fontweight='bold', color='white')
            
            # Add downward arrows
            if i < len(solv_stages) - 1:
                ax_solv.arrow(x, y - box_height/2 - 0.1, 0, -1.8, 
                            head_width=0.3, head_length=0.2, fc='darkblue', ec='darkblue', linewidth=2)
        
        # 4. Feature Fusion (middle column)
        ax_fusion = fig.add_subplot(gs[1:3, 1])
        ax_fusion.set_xlim(0, 10)
        ax_fusion.set_ylim(0, 16)
        ax_fusion.axis('off')
        ax_fusion.set_title('Feature Fusion\n(Interaction Layer)', fontsize=14, fontweight='bold', color='purple')
        
        # Concatenation
        concat_box = FancyBboxPatch((3, 11), 4, 1.2, boxstyle="round,pad=0.1",
                                   facecolor='lightcoral', edgecolor='red', linewidth=2)
        ax_fusion.add_patch(concat_box)
        ax_fusion.text(5, 11.6, 'Concatenate\n(B, 72, 20, 20, 20)\n24+48=72', 
                      fontsize=9, ha='center', va='center', fontweight='bold')
        
        # Interaction conv stages (actual architecture)
        interaction_stages = [
            ('Conv3d(72→48)\n3×3×3, pad=1\n(B, 48, 20, 20, 20)', 5, 8.5, 'orange'),
            ('Conv3d(48→32)\n1×1×1\n(B, 32, 20, 20, 20)', 5, 6.5, 'darkorange'),
            ('CBAM3D\nInteraction Attention\n(B, 32, 20, 20, 20)', 5, 4.5, 'gold')
        ]
        
        for i, (name, x, y, color) in enumerate(interaction_stages):
            box = FancyBboxPatch((x-1.8, y-0.8), 3.6, 1.6, boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor='black', linewidth=1)
            ax_fusion.add_patch(box)
            ax_fusion.text(x, y, name, fontsize=9, ha='center', va='center', fontweight='bold')
            
            # Add downward arrows
            if i == 0:  # From concat to first conv
                ax_fusion.arrow(x, 10.3, 0, -1.0, head_width=0.2, head_length=0.15, fc='red', ec='red', linewidth=2)
            if i < len(interaction_stages) - 1:
                ax_fusion.arrow(x, y - 0.9, 0, -1.2, 
                              head_width=0.2, head_length=0.15, fc='orange', ec='orange', linewidth=2)
        
        # Cross-branch arrows to fusion (updated positions)
        ax_fusion.annotate('', xy=(3.2, 11), xytext=(0.5, 7), 
                          arrowprops=dict(arrowstyle='->', lw=2, color='darkred'))
        ax_fusion.annotate('', xy=(6.8, 11), xytext=(9.5, 8), 
                          arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
        
        # 5. CNN Backbone (full width, third row)
        ax_backbone = fig.add_subplot(gs[3, :])
        ax_backbone.set_xlim(0, 30)
        ax_backbone.set_ylim(0, 6)
        ax_backbone.axis('off')
        ax_backbone.set_title('CNN Backbone: ResidualBlock3D with CBAM Progression', fontsize=16, fontweight='bold')
        
        # CNN backbone stages (actual architecture with ResidualBlock3D)
        backbone_stages = [
            ('Input\n(B, 32, 20, 20, 20)', 2, 3, 'gold', 3.0),
            ('ResidualBlock3D\n32→48 + CBAM\n(B, 48, 20, 20, 20)', 6.5, 3, 'mediumpurple', 4.0),
            ('AvgPool3d(2)\n(B, 48, 10, 10, 10)', 11.5, 3, 'mediumorchid', 3.0),
            ('ResidualBlock3D\n48→64 + CBAM\n(B, 64, 10, 10, 10)', 16, 3, 'blueviolet', 4.0),
            ('AvgPool3d(2)\n(B, 64, 5, 5, 5)', 21, 3, 'darkviolet', 3.0),
            ('ResidualBlock3D\n64→80 + CBAM\n(B, 80, 5, 5, 5)', 25, 3, 'indigo', 4.0),
            ('ResidualBlock3D\n80→96 + CBAM\n(B, 96, 5, 5, 5)', 29, 3, 'navy', 4.0)
        ]
        
        for i, (name, x, y, color, box_width) in enumerate(backbone_stages):
            # Calculate box size based on spatial dimensions and complexity
            if 'ResidualBlock3D' in name:
                box_height = 1.8  # Larger for ResidualBlock3D
            elif '20, 20, 20' in name:
                box_height = 1.4
            elif '10, 10, 10' in name:
                box_height = 1.2
            elif '5, 5, 5' in name:
                box_height = 1.0
            else:
                box_height = 1.3
                
            box = FancyBboxPatch((x-box_width/2, y-box_height/2), box_width, box_height,
                               boxstyle="round,pad=0.1", facecolor=color, edgecolor='purple', linewidth=2)
            ax_backbone.add_patch(box)
            ax_backbone.text(x, y, name, fontsize=9, ha='center', va='center', fontweight='bold', color='white')
            
            # Add rightward arrows
            if i < len(backbone_stages) - 1:
                next_x = backbone_stages[i+1][1]
                ax_backbone.arrow(x + box_width/2 + 0.1, y, next_x - x - box_width/2 - 0.3, 0,
                                head_width=0.15, head_length=0.15, fc='purple', ec='purple', linewidth=2)
        
        # Arrow from fusion to backbone
        ax_backbone.annotate('', xy=(0.5, 3), xytext=(5, 4), 
                           arrowprops=dict(arrowstyle='->', lw=3, color='gold'))
        
        # 6. ResidualBlock3D Internal Structure Detail (fourth row)
        ax_resblock = fig.add_subplot(gs[4, :])
        ax_resblock.set_xlim(0, 24)
        ax_resblock.set_ylim(0, 8)
        ax_resblock.axis('off')
        ax_resblock.set_title('ResidualBlock3D Internal Structure (use_cbam=True)', fontsize=16, fontweight='bold', color='darkblue')
        
        # ResidualBlock3D components
        resblock_components = [
            # Main path
            ('Input\n(B, in_ch, H, W, D)', 2, 6, 'lightblue', 2.5),
            ('Conv3d + BN + ReLU\nin_ch→out_ch', 6, 6, 'lightgreen', 3.0),
            ('Conv3d + BN\nout_ch→out_ch', 10, 6, 'lightgreen', 3.0),
            ('CBAM3D\nChannel + Spatial\nAttention', 14, 6, 'gold', 3.5),
            ('Add\n(Residual)', 18, 6, 'lightcoral', 2.0),
            ('ReLU\nOutput', 21, 6, 'orange', 2.0),
            
            # Shortcut path
            ('Shortcut\n(Identity or\n1×1 Conv)', 6, 2.5, 'lightgray', 3.0),
        ]
        
        for i, (name, x, y, color, box_width) in enumerate(resblock_components):
            if name.startswith('Shortcut'):
                box_height = 1.0
                edge_style = '--'
                edge_width = 1
            elif 'Add' in name:
                # Special circle for addition
                circle = plt.Circle((x, y), 0.8, facecolor=color, edgecolor='red', linewidth=2)
                ax_resblock.add_patch(circle)
                ax_resblock.text(x, y, '+', fontsize=16, ha='center', va='center', fontweight='bold')
                continue
            else:
                box_height = 1.4
                edge_style = '-'
                edge_width = 2
                
            box = FancyBboxPatch((x-box_width/2, y-box_height/2), box_width, box_height,
                               boxstyle="round,pad=0.1", facecolor=color, 
                               edgecolor='darkblue', linewidth=edge_width, linestyle=edge_style)
            ax_resblock.add_patch(box)
            ax_resblock.text(x, y, name, fontsize=9, ha='center', va='center', fontweight='bold')
        
        # Main path arrows
        main_path_arrows = [(2, 6, 6, 6), (6, 6, 10, 6), (10, 6, 14, 6), (14, 6, 18, 6), (18, 6, 21, 6)]
        for x1, y1, x2, y2 in main_path_arrows:
            ax_resblock.arrow(x1 + 1.25, y1, x2 - x1 - 2.5, 0, 
                            head_width=0.2, head_length=0.2, fc='darkblue', ec='darkblue', linewidth=2)
        
        # Shortcut path arrows
        ax_resblock.arrow(2 + 1.25, 6 - 0.7, 2.5, -2.5, head_width=0.15, head_length=0.15, 
                        fc='gray', ec='gray', linewidth=2, linestyle='--')
        ax_resblock.arrow(6 + 1.5, 2.5, 9.5, 2.8, head_width=0.15, head_length=0.15, 
                        fc='gray', ec='gray', linewidth=2, linestyle='--')
        
        # Labels for paths
        ax_resblock.text(12, 7.2, 'Main Path', fontsize=11, ha='center', fontweight='bold', color='darkblue')
        ax_resblock.text(12, 1.2, 'Shortcut Path (Residual Connection)', fontsize=11, ha='center', 
                       fontweight='bold', color='gray', style='italic')
        
        # 7. Final Processing (bottom row)
        ax_final = fig.add_subplot(gs[5, :])
        ax_final.set_xlim(0, 24)
        ax_final.set_ylim(0, 4)
        ax_final.axis('off')
        ax_final.set_title('Final Processing & Output', fontsize=16, fontweight='bold')
        
        # Final stages (updated to start from layer4 output)
        final_stages = [
            ('AdaptiveAvgPool3d(3)\n(B, 96, 3, 3, 3)', 5, 2, 'darkslateblue', 4),
            ('Vision Transformer\nGlobal Context', 11, 2, 'lightgreen', 4),
            ('Final Prediction\nSolvation Energy', 17, 2, 'yellow', 4)
        ]
        
        for i, (name, x, y, color, box_width) in enumerate(final_stages):
            box = FancyBboxPatch((x-box_width/2, y-0.6), box_width, 1.2,
                               boxstyle="round,pad=0.1", facecolor=color, 
                               edgecolor='black' if color != 'yellow' else 'orange', linewidth=2)
            ax_final.add_patch(box)
            text_color = 'white' if color in ['darkslateblue'] else 'black'
            ax_final.text(x, y, name, fontsize=10, ha='center', va='center', fontweight='bold', color=text_color)
            
            # Add rightward arrows
            if i < len(final_stages) - 1:
                next_x = final_stages[i+1][1]
                arrow_color = 'purple' if i < 1 else 'green'
                ax_final.arrow(x + box_width/2 + 0.1, y, next_x - x - box_width/2 - 0.3, 0,
                             head_width=0.2, head_length=0.2, fc=arrow_color, ec=arrow_color, linewidth=2)
        
        # Arrow from backbone to final processing
        ax_final.annotate('', xy=(3, 2), xytext=(29, 3.5), 
                         arrowprops=dict(arrowstyle='->', lw=3, color='navy'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"Type_2 dual-branch diagram saved to {save_path}")

        # plt.show()  # Removed: only save, don't display
    
    def create_transformer_architecture_detail(self, save_path: Optional[str] = None, dpi: int = 300):
        """
        Create detailed Vision Transformer architecture diagram following standard 
        scientific journal format for Transformer visualization.
        Shows step-by-step data shape transformations through the 3D ViT pipeline.
        """
        fig, ax = plt.subplots(figsize=(20, 14))
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 14)
        ax.axis('off')
        
        # Title
        ax.text(10, 13.3, '3D Vision Transformer Architecture for Molecular Interaction Energy Prediction', 
                fontsize=20, fontweight='bold', ha='center')
        ax.text(10, 12.8, 'Standard Transformer Components with 3D Spatial Adaptations', 
                fontsize=14, ha='center', style='italic', color='gray')
        
        # Color scheme following journal standards
        colors = {
            'input': '#E8F4FD',
            'embedding': '#D1E7DD', 
            'positional': '#FFF2CC',
            'attention': '#F8D7DA',
            'feedforward': '#D4EDDA',
            'output': '#E2E3E5',
            'data_flow': '#6C757D'
        }
        
        # ==================== INPUT PROCESSING ====================
        y_level = 11.5
        
        # CNN Feature Maps Input
        input_box = FancyBboxPatch((0.5, y_level-0.4), 3, 0.8, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=colors['input'], 
                                   edgecolor='black', linewidth=2)
        ax.add_patch(input_box)
        ax.text(2, y_level, 'CNN Features\n(B, 80, 3, 3, 3)', 
                fontsize=11, ha='center', va='center', fontweight='bold')
        
        # Arrow 1
        ax.arrow(3.7, y_level, 0.6, 0, head_width=0.15, head_length=0.1, 
                fc=colors['data_flow'], ec=colors['data_flow'], linewidth=2)
        
        # Flatten to Patches
        flatten_box = FancyBboxPatch((4.5, y_level-0.4), 3, 0.8,
                                     boxstyle="round,pad=0.1",
                                     facecolor=colors['embedding'],
                                     edgecolor='black', linewidth=2)
        ax.add_patch(flatten_box)
        ax.text(6, y_level, 'Flatten to Patches\n(B, 27, 80)', 
                fontsize=11, ha='center', va='center', fontweight='bold')
        ax.text(6, y_level-0.8, '3×3×3 = 27 spatial patches', 
                fontsize=9, ha='center', va='center', style='italic', color='gray')
        
        # Arrow 2
        ax.arrow(7.7, y_level, 0.6, 0, head_width=0.15, head_length=0.1,
                fc=colors['data_flow'], ec=colors['data_flow'], linewidth=2)
        
        # Input Projection
        proj_box = FancyBboxPatch((8.5, y_level-0.4), 3, 0.8,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['embedding'],
                                  edgecolor='black', linewidth=2)
        ax.add_patch(proj_box)
        ax.text(10, y_level, 'Input Projection\n(B, 27, 256)', 
                fontsize=11, ha='center', va='center', fontweight='bold')
        ax.text(10, y_level-0.8, 'Linear: 80 → 256 dims', 
                fontsize=9, ha='center', va='center', style='italic', color='gray')
        
        # Arrow 3
        ax.arrow(11.7, y_level, 0.6, 0, head_width=0.15, head_length=0.1,
                fc=colors['data_flow'], ec=colors['data_flow'], linewidth=2)
        
        # Positional Encoding
        pos_box = FancyBboxPatch((12.5, y_level-0.4), 3.5, 0.8,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['positional'],
                                 edgecolor='black', linewidth=2)
        ax.add_patch(pos_box)
        ax.text(14.25, y_level, 'Add Positional Encoding\n(B, 27, 256)', 
                fontsize=11, ha='center', va='center', fontweight='bold')
        ax.text(14.25, y_level-0.8, 'Learnable 3D position embeddings', 
                fontsize=9, ha='center', va='center', style='italic', color='gray')
        
        # ==================== TRANSFORMER ENCODER BLOCK (Standard) ====================
        y_level = 9.5
        
        # Left side: Standard Transformer Encoder Block (like your image)
        ax.text(3, y_level+1.5, 'Standard Transformer Encoder Block (×2)', 
                fontsize=14, ha='center', va='center', fontweight='bold', color='darkred')
        ax.text(3, y_level+1.1, 'Following Academic Paper Style', 
                fontsize=11, ha='center', va='center', style='italic', color='gray')
        
        # Draw the standard Transformer block structure
        self.draw_standard_transformer_block(ax, 3, y_level-1, colors)
        
        # Connecting arrow from input processing to transformer
        ax.arrow(16.2, 11.5, -5, 2.5, head_width=0.15, head_length=0.2, 
                fc='purple', ec='purple', linewidth=3, alpha=0.7)
        ax.text(13, 12.5, 'Token\nEmbeddings', fontsize=10, ha='center', va='center', 
                fontweight='bold', color='purple')
        
        # Right side: Multi-Head Attention Detail
        ax.text(15, y_level+1.5, 'Multi-Head Self-Attention Detail', 
                fontsize=14, ha='center', va='center', fontweight='bold', color='darkblue')
        
        # Q, K, V projections (more compact)
        qkv_y = y_level + 0.3
        for i, (name, x_pos, color_qkv) in enumerate([('Q', 12.5, '#FFB6C1'), ('K', 14.5, '#98FB98'), ('V', 16.5, '#87CEEB')]):
            qkv_box = FancyBboxPatch((x_pos-0.4, qkv_y-0.25), 0.8, 0.5,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color_qkv,
                                     edgecolor='darkblue', linewidth=1.5)
            ax.add_patch(qkv_box)
            ax.text(x_pos, qkv_y, name, fontsize=12, ha='center', va='center', fontweight='bold')
        
        # Attention computation formula
        formula_box = FancyBboxPatch((11.5, qkv_y-1.2), 6, 0.6,
                                     boxstyle="round,pad=0.1",
                                     facecolor='lightyellow',
                                     edgecolor='orange', linewidth=2)
        ax.add_patch(formula_box)
        ax.text(14.5, qkv_y-0.9, 'Attention(Q,K,V) = softmax(QK^T/√d_k + RelPos)V', 
                fontsize=10, ha='center', va='center', fontweight='bold')
        
        # Shape transformation details
        shape_text = """Input: (B, 27, 256)
Linear Proj: Q,K,V → (B, 4, 27, 64)  
Attention Score: (B, 4, 27, 27)
Output: (B, 27, 256)"""
        
        ax.text(14.5, qkv_y-2.2, shape_text, 
                fontsize=9, ha='center', va='center', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcyan', alpha=0.8))
        
        # ==================== OUTPUT PROCESSING ====================
        y_level = 3.5
        
        # Final Layer Norm
        final_norm_box = FancyBboxPatch((2, y_level-0.3), 3, 0.6,
                                        boxstyle="round,pad=0.1",
                                        facecolor='#F0F8FF',
                                        edgecolor='navy', linewidth=2)
        ax.add_patch(final_norm_box)
        ax.text(3.5, y_level, 'Final LayerNorm\n(B, 27, 256)', 
                fontsize=11, ha='center', va='center', fontweight='bold')
        
        # Arrow
        ax.arrow(5.2, y_level, 0.6, 0, head_width=0.15, head_length=0.1,
                fc=colors['data_flow'], ec=colors['data_flow'], linewidth=2)
        
        # Global Pooling
        pool_box = FancyBboxPatch((6, y_level-0.4), 3.5, 0.8,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['output'],
                                  edgecolor='black', linewidth=2)
        ax.add_patch(pool_box)
        ax.text(7.75, y_level, 'Global Pooling\n(B, 256)', 
                fontsize=11, ha='center', va='center', fontweight='bold')
        ax.text(7.75, y_level-0.8, 'Weighted: α×AvgPool + β×MaxPool', 
                fontsize=9, ha='center', va='center', style='italic', color='gray')
        
        # Arrow
        ax.arrow(9.7, y_level, 0.6, 0, head_width=0.15, head_length=0.1,
                fc=colors['data_flow'], ec=colors['data_flow'], linewidth=2)
        
        # To Classifier
        classifier_box = FancyBboxPatch((10.5, y_level-0.3), 3, 0.6,
                                        boxstyle="round,pad=0.1",
                                        facecolor='lightgreen',
                                        edgecolor='darkgreen', linewidth=2)
        ax.add_patch(classifier_box)
        ax.text(12, y_level, 'To Classifier\n(B, 256)', 
                fontsize=11, ha='center', va='center', fontweight='bold')
        
        # ==================== KEY INNOVATIONS ====================
        y_level = 1.5
        
        innovations_box = FancyBboxPatch((1, y_level-0.8), 18, 1.6,
                                         boxstyle="round,pad=0.15",
                                         facecolor='#F0FFF0',
                                         edgecolor='green', linewidth=2, alpha=0.9)
        ax.add_patch(innovations_box)
        
        ax.text(10, y_level+0.3, 'Key 3D Vision Transformer Innovations for Molecular Interactions', 
                fontsize=14, ha='center', va='center', fontweight='bold', color='darkgreen')
        
        innovations_text = """• 3D Spatial Patches: 3×3×3 CNN features → 27 spatial tokens for molecular geometry awareness
• Learnable 3D Positional Encoding: Captures relative spatial relationships between molecular regions  
• Relative Position Bias: Learned spatial attention bias for H-bond and van der Waals interactions
• Multi-Scale Fusion: Optional cross-attention between high-level and low-level CNN features
• Temperature-Controlled Attention: Fixed temperature (0.03) for focused molecular interaction patterns"""
        
        ax.text(10, y_level-0.2, innovations_text, 
                fontsize=10, ha='center', va='center', 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Transformer architecture diagram saved: {save_path}")
        
        # Don't display, only save
    
    def draw_standard_transformer_block(self, ax, center_x, center_y, colors):
        """
        Draw the standard Transformer Encoder Block structure (like in academic papers).
        This follows the classic diagram style shown in Transformer papers.
        """
        # Block dimensions
        block_width = 2.5
        block_height = 0.6
        
        # 1. Input (bottom)
        input_y = center_y - 2.5
        input_box = FancyBboxPatch((center_x - block_width/2, input_y - block_height/2), 
                                   block_width, block_height,
                                   boxstyle="round,pad=0.05",
                                   facecolor='#E8F4FD',
                                   edgecolor='black', linewidth=1.5)
        ax.add_patch(input_box)
        ax.text(center_x, input_y, 'Embed\nPatches', 
                fontsize=10, ha='center', va='center', fontweight='bold')
        
        # 2. First LayerNorm
        norm1_y = center_y - 1.5
        norm1_box = FancyBboxPatch((center_x - block_width/2, norm1_y - block_height/2), 
                                   block_width, block_height,
                                   boxstyle="round,pad=0.05",
                                   facecolor='#D1FFD1',
                                   edgecolor='darkgreen', linewidth=1.5)
        ax.add_patch(norm1_box)
        ax.text(center_x, norm1_y, 'Norm', 
                fontsize=10, ha='center', va='center', fontweight='bold')
        
        # 3. Multi-Head Attention
        attn_y = center_y - 0.5
        attn_box = FancyBboxPatch((center_x - block_width/2, attn_y - block_height/2), 
                                  block_width, block_height,
                                  boxstyle="round,pad=0.05",
                                  facecolor='#FFE1E1',
                                  edgecolor='darkred', linewidth=2)
        ax.add_patch(attn_box)
        ax.text(center_x, attn_y, 'Multi-Head\nAttention', 
                fontsize=9, ha='center', va='center', fontweight='bold')
        
        # 4. First Add & Norm (residual connection symbol)
        add1_y = center_y + 0.5
        add1_circle = plt.Circle((center_x, add1_y), 0.2, 
                                facecolor='lightblue', edgecolor='blue', linewidth=2)
        ax.add_patch(add1_circle)
        ax.text(center_x, add1_y, '+', 
                fontsize=14, ha='center', va='center', fontweight='bold')
        
        # 5. Second LayerNorm
        norm2_y = center_y + 1.2
        norm2_box = FancyBboxPatch((center_x - block_width/2, norm2_y - block_height/2), 
                                   block_width, block_height,
                                   boxstyle="round,pad=0.05",
                                   facecolor='#D1FFD1',
                                   edgecolor='darkgreen', linewidth=1.5)
        ax.add_patch(norm2_box)
        ax.text(center_x, norm2_y, 'Norm', 
                fontsize=10, ha='center', va='center', fontweight='bold')
        
        # 6. MLP (Feed-Forward)
        mlp_y = center_y + 2.0
        mlp_box = FancyBboxPatch((center_x - block_width/2, mlp_y - block_height/2), 
                                 block_width, block_height,
                                 boxstyle="round,pad=0.05",
                                 facecolor='#E1F5E1',
                                 edgecolor='darkgreen', linewidth=2)
        ax.add_patch(mlp_box)
        ax.text(center_x, mlp_y, 'MLP', 
                fontsize=10, ha='center', va='center', fontweight='bold')
        
        # 7. Second Add (residual connection symbol)
        add2_y = center_y + 2.8
        add2_circle = plt.Circle((center_x, add2_y), 0.2, 
                                facecolor='lightblue', edgecolor='blue', linewidth=2)
        ax.add_patch(add2_circle)
        ax.text(center_x, add2_y, '+', 
                fontsize=14, ha='center', va='center', fontweight='bold')
        
        # Vertical arrows (main path)
        arrow_positions = [
            (center_x, input_y + block_height/2, center_x, norm1_y - block_height/2),
            (center_x, norm1_y + block_height/2, center_x, attn_y - block_height/2),
            (center_x, attn_y + block_height/2, center_x, add1_y - 0.2),
            (center_x, add1_y + 0.2, center_x, norm2_y - block_height/2),
            (center_x, norm2_y + block_height/2, center_x, mlp_y - block_height/2),
            (center_x, mlp_y + block_height/2, center_x, add2_y - 0.2)
        ]
        
        for x1, y1, x2, y2 in arrow_positions:
            ax.arrow(x1, y1, 0, y2-y1-0.05, head_width=0.08, head_length=0.05, 
                    fc='black', ec='black', linewidth=1.5)
        
        # Residual connection arrows (curved)
        # First residual connection (input to first add)
        ax.annotate('', xy=(center_x-0.2, add1_y), xytext=(center_x-1.5, input_y),
                   arrowprops=dict(arrowstyle='->', lw=2, color='blue', 
                                 connectionstyle="arc3,rad=0.3"))
        
        # Second residual connection (after first add to second add)
        ax.annotate('', xy=(center_x-0.2, add2_y), xytext=(center_x-1.5, add1_y),
                   arrowprops=dict(arrowstyle='->', lw=2, color='blue', 
                                 connectionstyle="arc3,rad=0.3"))
        
        # Add dimensional annotations
        ax.text(center_x + 1.5, center_y, '(B, 27, 256)', 
                fontsize=9, ha='left', va='center', fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))
    
    
    def create_hybrid_architecture_detail(self, save_path: Optional[str] = None, dpi: int = 300):
        """
        Create a detailed visualization of the hybrid CNN-Transformer architecture.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Hybrid CNN-Transformer Architecture Details', fontsize=20, fontweight='bold')
        
        # 1. CNN Backbone Flow
        ax1.set_xlim(0, 14)
        ax1.set_ylim(0, 10)
        ax1.axis('off')
        ax1.set_title('CNN Backbone with Strategic CBAM Placement', fontsize=14, fontweight='bold')
        
        # CNN backbone components
        cnn_components = [
            ('Input\n(15×20³)', 1, 5, 'lightblue', '15 channels'),
            ('Input\nAttention', 3, 5, 'gold', 'CBAM'),
            ('Initial\nConv', 5, 5, 'royalblue', '24 channels'),
            ('Layer1\n+CBAM', 7, 5, 'dodgerblue', '36 channels'),
            ('Layer2\n+CBAM+SA', 9, 5, 'mediumpurple', '52 channels'),
            ('Layer3\n+SA', 11, 5, 'orange', '72 channels'),
            ('Layer4\n+CBAM', 13, 5, 'tomato', '96 channels')
        ]
        
        for name, x, y, color, desc in cnn_components:
            box = FancyBboxPatch((x-0.7, y-1), 1.4, 2, boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.8)
            ax1.add_patch(box)
            ax1.text(x, y, name, fontsize=9, ha='center', va='center', fontweight='bold')
            ax1.text(x, y-2, desc, fontsize=7, ha='center', va='center', style='italic')
        
        # Add arrows
        for i in range(len(cnn_components) - 1):
            x1 = cnn_components[i][1] + 0.7
            x2 = cnn_components[i+1][1] - 0.7
            ax1.arrow(x1, 5, x2-x1, 0, head_width=0.2, head_length=0.1, fc='black', ec='black')
        
        # 2. Vision Transformer Detail
        ax2.set_xlim(0, 12)
        ax2.set_ylim(0, 10)
        ax2.axis('off')
        ax2.set_title('Vision Transformer 3D Processing', fontsize=14, fontweight='bold')
        
        # ViT components
        vit_components = [
            ('CNN\nFeatures\n(96×2³)', 2, 7, 'lightcoral', '2×2×2 spatial'),
            ('Patch\nEmbedding', 5, 7, 'lightyellow', 'Project to d_model'),
            ('Positional\nEncoding', 8, 7, 'lightgreen', '3D position aware'),
            ('Transformer\nLayers', 5, 4, 'mediumpurple', f'{self.model.transformer_layers} layers'),
            ('Global\nPooling', 8, 4, 'orange', 'Multi-type pooling'),
            ('Output\nFeatures', 10, 4, 'lightblue', f'{self.model.transformer_dim}D')
        ]
        
        for name, x, y, color, desc in vit_components:
            box = FancyBboxPatch((x-0.8, y-0.8), 1.6, 1.6, boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.8)
            ax2.add_patch(box)
            ax2.text(x, y, name, fontsize=9, ha='center', va='center', fontweight='bold')
            ax2.text(x, y-1.5, desc, fontsize=7, ha='center', va='center', style='italic')
        
        # ViT flow arrows
        flow_connections = [(0,1), (1,2), (1,3), (2,3), (3,4), (4,5)]
        for i, j in flow_connections:
            x1, y1 = vit_components[i][1], vit_components[i][2]
            x2, y2 = vit_components[j][1], vit_components[j][2]
            if i == 1 and j == 3:  # Downward arrow
                ax2.arrow(x1, y1-0.8, 0, y2-y1+1.6, head_width=0.2, head_length=0.1, fc='blue', ec='blue')
            elif i == 2 and j == 3:  # Diagonal arrow
                ax2.arrow(x1-0.8, y1, x2+0.8-x1, y2-y1, head_width=0.2, head_length=0.1, fc='blue', ec='blue')
            else:  # Horizontal arrows
                dx = x2 - x1
                if abs(dx) > 0.1:
                    import numpy as np
                    ax2.arrow(x1+0.8*np.sign(dx), y1, dx-1.6*np.sign(dx), 0, 
                             head_width=0.2, head_length=0.1, fc='blue', ec='blue')
        
        # 3. Attention Mechanism Comparison
        ax3.set_xlim(0, 12)
        ax3.set_ylim(0, 8)
        ax3.axis('off')
        ax3.set_title('Multi-Level Attention Strategy', fontsize=14, fontweight='bold')
        
        # Attention strategy diagram
        attention_levels = [
            ('Input Level', 2, 6.5, 'CBAM Channel\nFeature Selection', 'gold'),
            ('Early CNN', 2, 5, 'CBAM Full\nLocal Attention', 'orange'),
            ('Mid CNN', 2, 3.5, 'CBAM + Self-Attention\nLocal + Context', 'mediumpurple'),
            ('Late CNN', 2, 2, 'Self-Attention Only\nContext Focus', 'lightblue'),
            ('Transformer', 8, 4, 'Multi-Head Attention\nGlobal Dependencies', 'tomato')
        ]
        
        for name, x, y, desc, color in attention_levels:
            # Main box
            box = FancyBboxPatch((x-1, y-0.4), 5, 0.8, boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.7)
            ax3.add_patch(box)
            ax3.text(x+1.5, y, f'{name}: {desc}', fontsize=10, ha='center', va='center', fontweight='bold')
        
        # Connection lines showing progression
        for i in range(len(attention_levels)-2):
            y1 = attention_levels[i][2]
            y2 = attention_levels[i+1][2]
            ax3.arrow(1.8, y1-0.2, 0, y2-y1+0.4, head_width=0.1, head_length=0.1, fc='gray', ec='gray')
        
        # Transformer connection
        ax3.arrow(6, 3.5, 1.5, 0.3, head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2)
        ax3.text(6.7, 3.2, 'CNN→Transformer', fontsize=9, ha='center', color='red', fontweight='bold')
        
        # 4. Model Performance Comparison
        ax4.set_xlim(0, 10)
        ax4.set_ylim(0, 8)
        ax4.axis('off')
        ax4.set_title('Hybrid vs Pure CNN Benefits', fontsize=14, fontweight='bold')
        
        # Benefits comparison
        benefits_text = """
        🎯 CNN Backbone Benefits:
        • Efficient 3D voxel processing
        • Local feature extraction 
        • CBAM attention for important features
        • Residual connections for deep learning
        
        🚀 Transformer Addition:
        • Global dependency modeling
        • Long-range solvent interactions
        • Better context understanding
        • Enhanced feature integration
        
        ⚖️ Hybrid Advantage:
        • Best of both worlds
        • Efficient + Expressive
        • Local + Global features
        • ~{:.1f}M parameters""".format(sum(p.numel() for p in self.model.parameters()) / 1e6)
        
        ax4.text(5, 4, benefits_text, fontsize=11, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Hybrid architecture detail saved: {save_path}")
        
        # plt.show()  # Removed: only save, don't display
    
    def create_model_complexity_analysis(self, save_path: Optional[str] = None, dpi: int = 300):
        """
        Create model complexity analysis visualization.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('3D-CNN Model Complexity Analysis', fontsize=18, fontweight='bold')
        
        # Calculate model statistics
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Get layer-wise parameter counts
        layer_params = {}
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    layer_params[name] = params
        
        # 1. Parameter distribution pie chart
        layer_groups = {
            'Initial Conv': 0,
            'ResBlocks': 0,
            'CBAM Attention': 0,
            'Classifier': 0
        }
        
        for name, params in layer_params.items():
            if 'initial_conv' in name:
                layer_groups['Initial Conv'] += params
            elif any(x in name for x in ['layer1', 'layer2', 'layer3']):
                if any(x in name for x in ['channel_attention', 'spatial_attention']):
                    layer_groups['CBAM Attention'] += params
                else:
                    layer_groups['ResBlocks'] += params
            elif 'classifier' in name:
                layer_groups['Classifier'] += params
        
        # Remove zero entries
        layer_groups = {k: v for k, v in layer_groups.items() if v > 0}
        
        labels = list(layer_groups.keys())
        sizes = list(layer_groups.values())
        colors = [self.colors['conv'], self.colors['resblock'], self.colors['attention'], self.colors['fc']][:len(labels)]
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        ax1.set_title('Parameter Distribution by Component', fontsize=14, fontweight='bold')
        
        # 2. Layer-wise parameter counts
        if len(layer_groups) > 0:
            ax2.bar(labels, sizes, color=colors, alpha=0.8, edgecolor='black')
            ax2.set_title('Parameter Counts by Layer Group', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Number of Parameters')
            ax2.tick_params(axis='x', rotation=45)
            
            # Format y-axis
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}K' if x >= 1e3 else f'{x:.0f}'))
        
        # 3. Feature map memory usage
        ax3.set_title('Feature Map Memory Usage', fontsize=14, fontweight='bold')
        
        # Calculate memory for each layer (assuming float32)
        layer_memory = []
        layer_names = []
        
        for layer in self.architecture_flow:
            if layer['shape'] is not None:
                if len(layer['shape']) >= 3:
                    h, w, d = layer['shape'][:3]
                elif len(layer['shape']) == 1:
                    # For transformer (treat as 1D sequence)
                    h, w, d = layer['shape'][0], 1, 1
                else:
                    # Skip layers without proper 3D shape
                    continue
                    
                channels = layer['channels'] if isinstance(layer['channels'], int) else layer['channels'][0] if isinstance(layer['channels'], list) else 1
                # Memory in MB (4 bytes per float32)
                memory_mb = (h * w * d * channels * 4) / (1024 * 1024)
                layer_memory.append(memory_mb)
                layer_names.append(layer['name'].replace(' ', '\n'))
        
        if layer_memory:
            bars = ax3.bar(range(len(layer_memory)), layer_memory, 
                          color=plt.cm.plasma(np.linspace(0, 1, len(layer_memory))), 
                          alpha=0.8, edgecolor='black')
            ax3.set_xticks(range(len(layer_names)))
            ax3.set_xticklabels(layer_names, rotation=45, ha='right')
            ax3.set_ylabel('Memory Usage (MB)')
            
            # Add value labels
            for bar, mem in zip(bars, layer_memory):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{mem:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 4. Model efficiency metrics
        ax4.axis('off')
        ax4.set_title('Model Efficiency Metrics', fontsize=14, fontweight='bold')
        
        # Calculate FLOPs estimate (simplified)
        total_flops = 0
        for layer in self.architecture_flow:
            if layer['shape'] is not None and layer['type'] in ['conv', 'resblock']:
                h, w, d = layer['shape']
                channels = layer['channels'] if isinstance(layer['channels'], int) else layer['channels'][0]
                if layer['type'] == 'conv':
                    # Conv3D: kernel_size^3 * in_channels * out_channels * output_size
                    kernel_size = 3
                    in_channels = 15 if 'Initial' in layer['name'] else channels // 2
                    flops = kernel_size**3 * in_channels * channels * h * w * d
                elif layer['type'] == 'resblock':
                    # Simplified ResBlock FLOP estimate
                    flops = 2 * (3**3 * channels * channels * h * w * d)  # Two conv layers
                total_flops += flops
        
        metrics_text = f"""Model Efficiency Metrics:

Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Model Size: {total_params * 4 / (1024**2):.2f} MB

Estimated FLOPs: {total_flops / 1e9:.2f} GFLOPs
Parameters per Layer (avg): {total_params / len(self.architecture_flow):.0f}

Memory Efficiency:
- Peak Feature Map Memory: {max(layer_memory) if layer_memory else 0:.2f} MB
- Total Feature Memory: {sum(layer_memory) if layer_memory else 0:.2f} MB

Architecture Highlights:
- Residual Connections: ✓
- Attention Mechanism: CBAM
- Regularization: BatchNorm + Dropout
- Pooling Strategy: Progressive Max Pooling"""
        
        ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Model complexity analysis saved: {save_path}")
        
        # plt.show()  # Removed: only save, don't display
    
    def create_auto_architecture_summary(self, save_path: Optional[str] = None, dpi: int = 300):
        """
        Create a summary of the automatically detected architecture.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Auto-Detected 3D-CNN Architecture Analysis', fontsize=18, fontweight='bold')
        
        # 1. Layer type distribution
        layer_types = {}
        for layer in self.architecture_flow:
            layer_type = layer['type']
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
        
        if layer_types:
            ax1.pie(layer_types.values(), labels=layer_types.keys(), autopct='%1.1f%%', startangle=90)
            ax1.set_title('Layer Type Distribution', fontsize=14, fontweight='bold')
        
        # 2. Channel progression
        layers_with_channels = [layer for layer in self.architecture_flow if layer.get('channels') and isinstance(layer['channels'], int)]
        if layers_with_channels:
            layer_names = [layer['name'][:10] + '...' if len(layer['name']) > 10 else layer['name'] for layer in layers_with_channels]
            channels = [layer['channels'] for layer in layers_with_channels]
            
            ax2.bar(range(len(channels)), channels, color=plt.cm.viridis(np.linspace(0, 1, len(channels))))
            ax2.set_xticks(range(len(layer_names)))
            ax2.set_xticklabels(layer_names, rotation=45, ha='right')
            ax2.set_ylabel('Number of Channels')
            ax2.set_title('Channel Progression', fontsize=14, fontweight='bold')
        
        # 3. Spatial dimension progression
        layers_with_shape = [layer for layer in self.architecture_flow if layer.get('shape')]
        if layers_with_shape:
            layer_names = [layer['name'][:10] + '...' if len(layer['name']) > 10 else layer['name'] for layer in layers_with_shape]
            spatial_sizes = [np.prod(layer['shape']) for layer in layers_with_shape]
            
            ax3.plot(range(len(spatial_sizes)), spatial_sizes, 'o-', linewidth=2, markersize=8)
            ax3.set_xticks(range(len(layer_names)))
            ax3.set_xticklabels(layer_names, rotation=45, ha='right')
            ax3.set_ylabel('Spatial Size (H×W×D)')
            ax3.set_yscale('log')
            ax3.set_title('Spatial Dimension Reduction', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # 4. Architecture summary text
        ax4.axis('off')
        ax4.set_title('Auto-Detection Summary', fontsize=14, fontweight='bold')
        
        # Calculate statistics
        total_params = sum(p.numel() for p in self.model.parameters())
        attention_layers = [layer for layer in self.architecture_flow if layer.get('has_attention', False)]
        conv_layers = [layer for layer in self.architecture_flow if layer['type'] in ['conv', 'resblock']]
        
        summary_text = f"""Architecture Auto-Detection Results:

🔍 Detection Method: Dynamic Analysis
✅ Successfully detected: {len(self.architecture_flow)} components

📊 Architecture Statistics:
• Total Parameters: {total_params:,}
• Convolutional Layers: {len(conv_layers)}
• Attention Mechanisms: {len(attention_layers)}
• Layer Types: {len(layer_types)}

🧠 Detected Components:
"""
        
        # Add component list
        for i, layer in enumerate(self.architecture_flow[:8]):  # Show first 8
            shape_str = f"{layer['shape']}" if layer.get('shape') else "Variable"
            channels_str = f"{layer['channels']}" if layer.get('channels') else "N/A"
            summary_text += f"• {layer['name']}: {channels_str} ch, {shape_str}\n"
        
        if len(self.architecture_flow) > 8:
            summary_text += f"• ... and {len(self.architecture_flow) - 8} more layers\n"
        
        summary_text += f"""
🎯 Key Features Detected:
• CBAM Attention: {'✅' if attention_layers else '❌'}
• Residual Connections: {'✅' if any('resblock' in l['type'] for l in self.architecture_flow) else '❌'}
• Progressive Pooling: {'✅' if any('pool' in l['type'] for l in self.architecture_flow) else '❌'}
• Batch Normalization: ✅ (Assumed)

💡 This analysis was generated automatically from your model code!
"""
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Auto-architecture summary saved: {save_path}")
        
        # plt.show()  # Removed: only save, don't display
    
def create_3dcnn_visualizations():
    """
    Create all hybrid 3D-CNN-Transformer visualization figures with automatic architecture detection.
    """
    print("=== Creating Hybrid 3D-CNN-Transformer Model Visualizations (Auto-Detection) ===")
    
    # Create hybrid model instance for Type_2 format (28 channels: 14 adsorbate + 14 solvent)
    model = AttentionCNNTransformer_2_5(
        in_channels=28,  # Type_2 format: 14 adsorbate + 14 solvent channels
        dropout_rate=0.35,
        use_transformer=True,
        transformer_dim=200,
        transformer_heads=4,
        transformer_layers=2
    )
    print(f"Hybrid model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create visualizer (will auto-detect architecture)
    visualizer = CNN3DVisualizer(model)
    
    # Create output directory - use get_paths for proper path
    output_dir = os.path.join(get_paths('output_figure_path'), 'model_visualizations_3dcnn_transformer')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n--- Creating Visualizations (saved in {output_dir}/) ---")
    
    # 0. dual-branch architecture diagram (NEW)
    print("Creating dual-branch architecture diagram...")
    visualizer.create_type2_dual_branch_diagram(
        save_path=f"{output_dir}/hybrid_cnn_transformer_dual_branch_architecture.png"
    )

    # 1. Transformer architecture detail (NEW)
    print("Creating Transformer architecture detail diagram...")
    visualizer.create_transformer_architecture_detail(
        save_path=f"{output_dir}/transformer_architecture_detail.png"
    )
    
    # # Auto-detection summary
    # print("Creating auto-detection summary...")
    # visualizer.create_auto_architecture_summary(
    #     save_path=f"{output_dir}/hybrid_cnn_transformer_auto_detection_summary.png"
    # )
    
    # # 3D Architecture diagram
    # print("Creating 3D architecture diagram...")
    # visualizer.create_3d_architecture_diagram(
    #     save_path=f"{output_dir}/hybrid_cnn_transformer_3d_architecture.png"
    # )
    
    # # Detailed 2D flow diagram
    # print("Creating detailed 2D flow diagram...")
    # visualizer.create_2d_detailed_flow(
    #     save_path=f"{output_dir}/hybrid_cnn_transformer_detailed_flow.png"
    # )
    
    # # CBAM attention mechanism detail
    # print("Creating CBAM attention mechanism detail...")
    # visualizer.create_cbam_attention_detail(
    #     save_path=f"{output_dir}/hybrid_cnn_transformer_cbam_attention.png"
    # )
    
    # # Hybrid architecture detail
    # print("Creating hybrid architecture detail...")
    # visualizer.create_hybrid_architecture_detail(
    #     save_path=f"{output_dir}/hybrid_cnn_transformer_architecture_detail.png"
    # )
    
    # # Model complexity analysis
    # print("Creating model complexity analysis...")
    # visualizer.create_model_complexity_analysis(
    #     save_path=f"{output_dir}/hybrid_cnn_transformer_complexity_analysis.png"
    # )
    
if __name__ == "__main__":
    create_3dcnn_visualizations()
