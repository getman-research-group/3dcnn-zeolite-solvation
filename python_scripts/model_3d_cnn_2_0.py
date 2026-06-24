"""
model_3d_cnn_2_0.py
This file contains the definitions for the 3D-CNN model with CBAM attention mechanism 
enhanced with Vision Transformer (ViT) for global feature modeling, adapted for type_2 format.

Format specifications:
- 28 input features (14 atomic features × 2 groups: adsorbate + solvent)
- Separated channel groups for better feature interpretation
- Default input shape: (batch_size, 28, 20, 20, 20)

The hybrid CNN-Transformer architecture combines:
- CNN backbone with CBAM attention for local feature extraction
- Vision Transformer head for global dependency modeling

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math


class PositionalEncoding3D(nn.Module):
    """
    3D Positional Encoding for molecular voxel grids.
    Adds learnable position embeddings with physics-informed center bias.
    """
    def __init__(self, d_model, max_len=8, center_bias=True):
        super(PositionalEncoding3D, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.center_bias = center_bias
        
        # Create learnable positional embeddings for 3D positions
        # For a 2x2x2 grid after adaptive pooling, we have 8 positions
        self.pos_embedding = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        
        # Distance-aware encoding: learn different patterns for different distance shells
        # This helps capture the physics of interaction energy decay with distance
        self.distance_encoding = nn.Parameter(torch.randn(max_len, d_model) * 0.01)
        
        # Physics-informed center bias: positions closer to center (adsorbate) are more important
        if center_bias:
            # For 2x2x2 voxel arrangement after adaptive pooling, calculate actual 3D distances from center
            # Note: Original 20x20x20 voxel grid (16Å) is pooled down to 2x2x2 (8 spatial positions)
            # Each position in 2x2x2 represents an 8Å×8Å×8Å region of the original space
            # Positions in flattened 2x2x2: [0,1,2,3,4,5,6,7] map to 3D coordinates:
            # 0:(0,0,0), 1:(1,0,0), 2:(0,1,0), 3:(1,1,0), 4:(0,0,1), 5:(1,0,1), 6:(0,1,1), 7:(1,1,1)
            positions_3d = [(i%2, (i//2)%2, i//4) for i in range(max_len)]
            center_3d = (0.5, 0.5, 0.5)  # Center of 2x2x2 grid corresponds to adsorbate center
            
            with torch.no_grad():
                distance_weights = []
                for pos in positions_3d:
                    # Calculate 3D distance from center
                    dist = ((pos[0] - center_3d[0])**2 + (pos[1] - center_3d[1])**2 + (pos[2] - center_3d[2])**2)**0.5
                    # Use exponential decay similar to van der Waals interactions
                    weight = torch.exp(torch.tensor(-dist * 1.5))  # Convert to tensor first
                    distance_weights.append(weight)
                
                distance_weights = torch.tensor(distance_weights)
                self.pos_embedding.data *= distance_weights.unsqueeze(1)
                self.distance_encoding.data *= distance_weights.unsqueeze(1)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model) - flattened 3D patches
        Returns:
            x + enhanced positional encoding with distance awareness
        """
        seq_len = x.size(1)
        
        # Combine positional and distance encodings for richer spatial representation
        pos_enc = self.pos_embedding[:seq_len, :].unsqueeze(0)  # (1, seq_len, d_model)
        dist_enc = self.distance_encoding[:seq_len, :].unsqueeze(0)  # (1, seq_len, d_model)
        
        # Let the model learn the optimal combination of positional and distance information
        combined_encoding = pos_enc + dist_enc
        
        return x + combined_encoding


class MultiHeadAttention3D(nn.Module):
    """
    Enhanced Multi-Head Self-Attention optimized for molecular 3D voxel data.
    Includes improved scaling and activation strengthening.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention3D, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V with improved initialization
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        # Further reduced scaling to strengthen attention patterns - based on log analysis showing weak attention
        self.scale = math.sqrt(self.d_k) * 0.5  # Reduced from 0.8 to 0.5 to increase attention strength
        
        # Add learnable temperature parameter for dynamic attention scaling
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)  # Start with low temperature for sharper attention
        
        # Initialize weights for stronger attention
        self._init_attention_weights()
        
        # Store attention weights for analysis
        self.last_attention_weights = None
    
    def _init_attention_weights(self):
        """Initialize attention weights for stronger activation"""
        # Use stronger initialization for better head diversity - based on log showing 1/6 head diversity
        for module in [self.w_q, self.w_k, self.w_v]:
            nn.init.xavier_normal_(module.weight, gain=1.5)  # Increased gain from 1.2 to 1.5
        nn.init.xavier_normal_(self.w_o.weight, gain=0.8)
        if self.w_o.bias is not None:
            nn.init.constant_(self.w_o.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections and reshape for multi-head attention
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch, heads, seq, d_k)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention with learnable temperature for dynamic attention strength
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, heads, seq, seq)
        
        # Apply learnable temperature to enhance attention diversity and strength
        scores = scores / (self.temperature + 1e-8)  # Add small epsilon to avoid division by zero
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Store attention weights for analysis (detached from computation graph)
        self.last_attention_weights = attention_weights.detach()
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # (batch, heads, seq, d_k)
        
        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.w_o(context)
        
        return output


class CrossAttention3D(nn.Module):
    """
    Enhanced Cross-Attention layer optimized for adsorbate-solvent interactions.
    
    """
    def __init__(self, d_model, num_heads, dropout=0.1, enable_adsorbate_solvent_interaction=True):
        super(CrossAttention3D, self).__init__()
        assert d_model % num_heads == 0
        assert num_heads >= 3, "Need at least 3 heads for specialized molecular interactions"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.enable_adsorbate_solvent_interaction = enable_adsorbate_solvent_interaction
        
        # Standard linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Specialized projections for adsorbate-solvent interactions
        if enable_adsorbate_solvent_interaction:
            # Simple physics-informed enhancements
            # This allows us to understand molecular interactions without complex architectures
            pass  # Physics-informed bias will be computed dynamically
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # Feature fusion projection to match dimensions
        self.low_level_proj = nn.Linear(d_model, d_model)
        
        # Enhanced monitoring for cross-attention
        self.last_attention_weights = None
        self.attention_statistics = {
            'cross_fusion_strength': 0.0,
            'low_level_utilization': 0.0,
            'attention_entropy': 0.0,
            'attention_focus': 0.0
        }
        
    def forward(self, query, key_value, feature_channels=None):
        """
        Enhanced forward pass with adsorbate-solvent interaction modeling.
        
        Args:
            query: (batch_size, seq_len, d_model) - high-level features
            key_value: (batch_size, seq_len, d_model) - low-level features
            feature_channels: (batch_size, 28, H, W, D) - original feature channels for interaction analysis
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = query.size()
        
        # Project low-level features to match dimension
        key_value = self.low_level_proj(key_value)
        
        if self.enable_adsorbate_solvent_interaction and feature_channels is not None:
            # Enhanced adsorbate-solvent interaction modeling
            output = self._compute_adsorbate_solvent_attention(query, key_value, feature_channels)
        else:
            # Standard cross-attention fallback
            output = self._compute_standard_cross_attention(query, key_value)
        
        return output
    
    def _compute_adsorbate_solvent_attention(self, query, key_value, feature_channels):
        """
        Compute enhanced attention with explicit adsorbate-solvent interaction modeling.
        Note: feature_channels is a dict mapping feature names to channel indices.
        """
        batch_size, seq_len, d_model = query.size()
        
        # For now, use standard cross-attention but with physics-informed enhancement
        # This ensures compatibility while adding molecular interaction understanding
        
        # Standard multi-head attention computation
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key_value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(key_value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Add physics-informed bias based on feature channels
        physics_bias = torch.zeros_like(scores)
        if feature_channels:
            # Create simple physics-informed bias
            # Closer positions should have higher attention for molecular interactions
            for i in range(seq_len):
                for j in range(seq_len):
                    distance = abs(i - j)
                    if distance == 0:
                        physics_bias[:, :, i, j] += 0.1  # Self-interaction
                    elif distance == 1:
                        physics_bias[:, :, i, j] += 0.05  # Nearest neighbor
                    else:
                        physics_bias[:, :, i, j] += 0.01  # Distant interaction
        
        scores = scores + physics_bias
        
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Final output projection
        output = self.w_o(context)
        self.last_attention_weights = attention_weights.detach()
        
        # Calculate and update attention statistics (same as standard cross-attention)
        with torch.no_grad():
            cross_fusion_strength = attention_weights.mean().item()
            flat_attention = attention_weights.view(-1, attention_weights.size(-1))
            attention_entropy = -torch.sum(flat_attention * torch.log(flat_attention + 1e-8), dim=-1).mean().item()
            low_level_utilization = attention_weights.std().item()
            attention_focus = attention_weights.max().item()
            
            # Update attention statistics
            self.attention_statistics.update({
                'cross_fusion_strength': cross_fusion_strength,
                'low_level_utilization': low_level_utilization,
                'attention_entropy': attention_entropy,
                'attention_focus': attention_focus
            })
        
        return output
    
    def _compute_standard_cross_attention(self, query, key_value):
        """
        Standard multi-head cross-attention computation with monitoring.
        """
        batch_size, seq_len, d_model = query.size()
        
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key_value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(key_value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Store attention weights for monitoring
        self.last_attention_weights = attention_weights.detach()
        
        # Calculate standard cross-attention monitoring statistics
        with torch.no_grad():
            cross_fusion_strength = attention_weights.mean().item()
            flat_attention = attention_weights.view(-1, attention_weights.size(-1))
            attention_entropy = -torch.sum(flat_attention * torch.log(flat_attention + 1e-8), dim=-1).mean().item()
            low_level_utilization = attention_weights.std().item()
            attention_focus = attention_weights.max().item()
            
            # Update attention statistics with consistent field mapping for training logs
            self.attention_statistics.update({
                'cross_fusion_strength': cross_fusion_strength,
                'low_level_utilization': low_level_utilization,
                'attention_entropy': attention_entropy,
                'attention_focus': attention_focus
            })
            
            # Store cross-attention data for aggregate analysis by parent model
            if not hasattr(self, '_parent_model_ref'):
                # Try to find parent model reference for aggregate monitoring
                import gc
                for obj in gc.get_objects():
                    if hasattr(obj, 'transformer') and hasattr(obj.transformer, 'modules'):
                        for module in obj.transformer.modules():
                            if module is self:
                                self._parent_model_ref = obj
                                break
            
            # Register this layer's statistics for aggregate analysis
            if hasattr(self, '_parent_model_ref') and self._parent_model_ref is not None:
                if not hasattr(self._parent_model_ref, '_cross_attention_data'):
                    self._parent_model_ref._cross_attention_data = {}
                
                layer_id = f"cross_attention_{id(self)}"
                self._parent_model_ref._cross_attention_data[layer_id] = {
                    'cross_fusion_strength': cross_fusion_strength,
                    'low_level_utilization': low_level_utilization,
                    'attention_entropy': attention_entropy,
                    'attention_focus': attention_focus
                }
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        output = self.w_o(context)
        
        return output


class TransformerEncoderLayer3D(nn.Module):
    """
    Enhanced Transformer Encoder Layer with Cross-Attention for multi-level feature fusion.
    Uses pre-norm architecture and SiLU activation for better molecular interaction modeling.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_cross_attention=False):
        super(TransformerEncoderLayer3D, self).__init__()
        
        self.use_cross_attention = use_cross_attention
        self.self_attention = MultiHeadAttention3D(d_model, num_heads, dropout)
        
        # Add cross-attention if enabled
        if use_cross_attention:
            self.cross_attention = CrossAttention3D(d_model, num_heads, dropout)
            self.norm_cross = nn.LayerNorm(d_model)
        
        # Enhanced feed-forward network for molecular interaction modeling
        # Use gated mechanism to better capture non-linear molecular interactions
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),  # SiLU activation for enhanced non-linearity in molecular modeling
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Add a gated linear unit branch for better interaction modeling
        # This helps capture the complex nature of hydrogen bonding and van der Waals interactions
        self.interaction_gate = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Sigmoid(),  # Gate to control information flow
            nn.Linear(d_ff, d_model)
        )
        
        # Learnable mixing parameter for FFN and gated outputs
        self.gate_weight = nn.Parameter(torch.tensor(0.1))  # Start with small contribution
        
        # Layer normalization (pre-norm architecture)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, low_level_features=None, feature_channels=None):
        """
        Enhanced forward pass with support for adsorbate-solvent interaction analysis.
        
        Args:
            x: (batch_size, seq_len, d_model) - high-level features
            low_level_features: (batch_size, seq_len, d_model) - optional low-level features for cross-attention
            feature_channels: (batch_size, 28, H, W, D) - original feature channels for molecular interaction analysis
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # Pre-norm self-attention with residual connection
        norm_x = self.norm1(x)
        attention_output = self.self_attention(norm_x)
        x = x + self.dropout(attention_output)
        
        # Enhanced cross-attention for adsorbate-solvent feature fusion
        if self.use_cross_attention and low_level_features is not None:
            norm_x_cross = self.norm_cross(x)
            # Pass feature_channels for enhanced adsorbate-solvent interaction modeling
            cross_output = self.cross_attention(norm_x_cross, low_level_features, feature_channels)
            x = x + self.dropout(cross_output)
        
        # Enhanced feed-forward with gated interaction modeling
        norm_x = self.norm2(x)
        
        # Standard feed-forward path
        ff_output = self.feed_forward(norm_x)
        
        # Gated interaction path for better molecular interaction capture
        gate_output = self.interaction_gate(norm_x)
        
        # Learnable combination of standard FFN and gated interactions
        # This allows the model to learn the optimal balance for capturing adsorbate-solvent interactions
        combined_output = ff_output + self.gate_weight * gate_output
        
        x = x + combined_output
        
        return x


class VisionTransformer3D(nn.Module):
    """
    3D Vision Transformer module for processing CNN-extracted features.
    Converts 3D feature maps to sequence of patches and applies transformer layers.
    """
    def __init__(self, input_dim, d_model=256, num_heads=8, num_layers=2, dropout=0.1, use_cross_attention=True):
        super(VisionTransformer3D, self).__init__()
        
        self.input_dim = input_dim  # Input channels from CNN (e.g., 96)
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_cross_attention = use_cross_attention
        
        # Project input features to transformer dimension with stronger initialization
        self.input_projection = nn.Linear(input_dim, d_model)
        # Enhanced initialization for better feature projection
        nn.init.xavier_normal_(self.input_projection.weight, gain=1.4)
        
        # Low-level feature projection (for cross-attention)
        # Note: This will be dynamically adjusted based on actual low-level feature channels
        if use_cross_attention:
            self.low_level_projection = None  # Will be created dynamically
        
        # 3D Positional encoding with enhanced center bias for molecular interactions
        self.pos_encoding = PositionalEncoding3D(d_model, max_len=8, center_bias=True)
        
        # Enhanced transformer encoder layers with cross-attention capability
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer3D(
                d_model, 
                num_heads, 
                d_model*2, 
                dropout,
                use_cross_attention=use_cross_attention  # Enable cross-attention in all layers for better fusion
            )
            for i in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Enhanced global pooling with learnable weights including physics-aware pooling
        self.pooling_weights = nn.Parameter(torch.tensor([0.5, 0.2, 0.3]))  # avg, max, distance-weighted
        
        # Store attention information for analysis
        self.attention_maps = []
        self.cross_attention_maps = []
    
    def forward(self, x, low_level_features=None, return_attention=False, feature_channels=None):
        """
        Enhanced forward pass with cross-attention for multi-level feature fusion.
        
        Args:
            x: (batch_size, channels, D, H, W) - output from CNN backbone (high-level)
            low_level_features: (batch_size, channels, D, H, W) - optional low-level CNN features
            return_attention: bool - whether to return attention weights
            feature_channels: dict - mapping of feature channels for molecular interaction analysis
        Returns:
            output: (batch_size, d_model) - global feature representation
        """
        batch_size, channels, D, H, W = x.shape
        
        # Flatten spatial dimensions to create sequence of patches
        # (batch_size, channels, D*H*W) -> (batch_size, D*H*W, channels)
        x = x.view(batch_size, channels, -1).transpose(1, 2)
        seq_len = x.size(1)  # Should be D*H*W (e.g., 2*2*2 = 8)
        
        # Project to transformer dimension
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Process low-level features if provided for cross-attention
        low_level_projected = None
        if self.use_cross_attention and low_level_features is not None:
            # Create low-level projection dynamically if needed
            low_level_channels = low_level_features.size(1)
            if self.low_level_projection is None:
                self.low_level_projection = nn.Linear(low_level_channels, self.d_model).to(x.device)
            
            # Ensure low_level_features match the spatial dimensions
            if low_level_features.shape[-3:] != (D, H, W):
                # Adaptive pooling to match dimensions
                low_level_features = F.adaptive_avg_pool3d(low_level_features, (D, H, W))
            
            # Project low-level features
            low_level_flat = low_level_features.view(batch_size, low_level_channels, -1).transpose(1, 2)
            low_level_projected = self.low_level_projection(low_level_flat)
            low_level_projected = self.pos_encoding(low_level_projected)
        
        # Clear previous attention maps
        self.attention_maps = []
        self.cross_attention_maps = []
        
        # Apply transformer layers with cross-attention capability
        for i, layer in enumerate(self.transformer_layers):
            if i == 0 or not self.use_cross_attention or low_level_projected is None:
                # First layer or no cross-attention: standard self-attention only
                x = layer(x)
            else:
                # Subsequent layers: use cross-attention with low-level features
                x = layer(x, low_level_projected, feature_channels)
            
            # Store attention weights for analysis (check for None values)
            if hasattr(layer.self_attention, 'last_attention_weights') and layer.self_attention.last_attention_weights is not None:
                self.attention_maps.append(layer.self_attention.last_attention_weights.clone())
            
            # Store cross-attention weights if available (check for None values)
            if hasattr(layer, 'cross_attention') and hasattr(layer.cross_attention, 'last_attention_weights') and layer.cross_attention.last_attention_weights is not None:
                self.cross_attention_maps.append(layer.cross_attention.last_attention_weights.clone())
        
        # Final normalization
        x = self.final_norm(x)  # (batch_size, seq_len, d_model)
        
        # Physics-aware global pooling for adsorbate-solvent interactions
        # Standard global pooling
        avg_pooled = torch.mean(x, dim=1)  # (batch_size, d_model)
        max_pooled, _ = torch.max(x, dim=1)  # (batch_size, d_model)
        
        # Distance-weighted pooling: give more weight to positions closer to adsorbate center
        # This captures the physics that closer solvent molecules contribute more to interaction energy
        seq_len = x.size(1)
        if seq_len == 8:  # 2x2x2 grid case
            # Create distance weights from center (position-aware pooling)
            positions_3d = [(i%2, (i//2)%2, i//4) for i in range(seq_len)]
            center_3d = (0.5, 0.5, 0.5)
            
            distance_weights = []
            for pos in positions_3d:
                dist = ((pos[0] - center_3d[0])**2 + (pos[1] - center_3d[1])**2 + (pos[2] - center_3d[2])**2)**0.5
                # Use Gaussian-like weighting: closer positions get exponentially higher weights
                weight = torch.exp(torch.tensor(-dist * 2.0))  # Convert to tensor first
                distance_weights.append(weight)
            
            distance_weights = torch.tensor(distance_weights, device=x.device)
            distance_weights = distance_weights / distance_weights.sum()  # Normalize
            
            # Apply distance-weighted pooling
            weighted_pooled = torch.sum(x * distance_weights.unsqueeze(0).unsqueeze(2), dim=1)  # (batch_size, d_model)
            
            # Learnable combination of different pooling strategies
            global_features = (self.pooling_weights[0] * avg_pooled + 
                              self.pooling_weights[1] * max_pooled +
                              self.pooling_weights[2] * weighted_pooled)
        else:
            # Fallback for other sequence lengths
            global_features = (self.pooling_weights[0] * avg_pooled + 
                              self.pooling_weights[1] * max_pooled)
        
        if return_attention:
            return global_features, self.attention_maps, self.cross_attention_maps
        return global_features
    
    def get_attention_summary(self):
        """Get summary of attention patterns across transformer layers."""
        if not self.attention_maps:
            return "No attention data available"
        
        summary = {}
        for i, attention_map in enumerate(self.attention_maps):
            # attention_map shape: (batch_size, num_heads, seq_len, seq_len)
            # Average across batch and heads for summary
            avg_attention = attention_map.mean(dim=(0, 1))  # (seq_len, seq_len)
            
            summary[f'layer_{i+1}'] = {
                'mean_attention': float(avg_attention.mean().item()),
                'max_attention': float(avg_attention.max().item()),
                'attention_entropy': float(-torch.sum(avg_attention * torch.log(avg_attention + 1e-8)).item()),
                'diagonal_attention': float(torch.diag(avg_attention).mean().item()),  # Self-attention strength
            }
        
        return summary


class CBAMChannelAttention(nn.Module):
    """
    Standard CBAM Channel Attention Module with regularization and comprehensive monitoring.
    Enhanced with dropout for better generalization.
    """
    def __init__(self, in_channels, reduction_ratio=16, dropout=0.1):
        super(CBAMChannelAttention, self).__init__()
        
        # Standard CBAM channel attention design
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # Shared MLP with dropout for regularization
        reduced_channels = max(in_channels // reduction_ratio, 1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),  # Add dropout for regularization
            nn.Linear(reduced_channels, in_channels, bias=False)
        )
        
        # Enhanced initialization for better channel diversity - based on log showing low diversity
        for module in self.shared_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.3)  # Stronger initialization
        
        self.activation = nn.Sigmoid()
        
        # Enhanced monitoring for training analysis
        self.last_attention_weights = None
        self.attention_statistics = {
            'mean_attention': 0.0,
            'std_attention': 0.0,
            'max_attention': 0.0,
            'min_attention': 0.0,
            'attention_entropy': 0.0,
            'channel_diversity': 0.0,
            'most_important_channels': [],
            'least_important_channels': [],
            'attention_distribution': 'uniform'  # uniform, concentrated, bimodal
        }

    def forward(self, x):
        batch_size, channels, _, _, _ = x.size()
        
        # Global average pooling and max pooling
        avg_out = self.avg_pool(x).view(batch_size, channels)
        max_out = self.max_pool(x).view(batch_size, channels)
        
        # Process through shared MLP
        avg_weights = self.shared_mlp(avg_out)
        max_weights = self.shared_mlp(max_out)
        
        # Element-wise addition and sigmoid activation (standard CBAM)
        channel_weights = self.activation(avg_weights + max_weights)
        channel_weights = channel_weights.view(batch_size, channels, 1, 1, 1)
        
        # Store for analysis
        self.last_attention_weights = channel_weights.detach()
        
        # Calculate comprehensive attention statistics during training
        if self.training:
            self._update_attention_statistics(channel_weights)
        
        return x * channel_weights.expand_as(x)
    
    def _update_attention_statistics(self, channel_weights):
        """Calculate and update detailed attention statistics for monitoring"""
        with torch.no_grad():
            # Average across batch and spatial dimensions to get per-channel attention
            channel_attn = channel_weights.squeeze().mean(dim=0) if len(channel_weights.shape) > 2 else channel_weights.squeeze()
            
            if channel_attn.numel() == 1:  # Single channel case
                channel_attn = channel_attn.unsqueeze(0)
            
            # Basic statistics
            mean_attn = channel_attn.mean().item()
            std_attn = channel_attn.std().item()
            max_attn = channel_attn.max().item()
            min_attn = channel_attn.min().item()
            
            # Attention entropy (measure of diversity)
            # Add small epsilon to avoid log(0)
            normalized_attn = channel_attn / (channel_attn.sum() + 1e-8)
            entropy = -torch.sum(normalized_attn * torch.log(normalized_attn + 1e-8)).item()
            
            # Channel diversity (standard deviation as measure of how different channels are)
            diversity = std_attn / (mean_attn + 1e-8)
            
            # Most and least important channels
            num_channels = len(channel_attn)
            top_k = min(5, num_channels)
            most_important = torch.topk(channel_attn, top_k).indices.tolist()
            least_important = torch.topk(channel_attn, top_k, largest=False).indices.tolist()
            
            # Attention distribution pattern
            if diversity < 0.1:
                distribution = 'uniform'
            elif diversity > 0.5:
                distribution = 'concentrated'
            else:
                distribution = 'balanced'
            
            # Update statistics
            self.attention_statistics.update({
                'mean_attention': mean_attn,
                'std_attention': std_attn,
                'max_attention': max_attn,
                'min_attention': min_attn,
                'attention_entropy': entropy,
                'channel_diversity': diversity,
                'most_important_channels': most_important,
                'least_important_channels': least_important,
                'attention_distribution': distribution
            })


class CBAMSpatialAttention(nn.Module):
    """
    Standard CBAM Spatial Attention Module with comprehensive monitoring.
    Pure implementation following the original CBAM paper.
    """
    def __init__(self, kernel_size=7):
        super(CBAMSpatialAttention, self).__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        
        # Standard CBAM spatial attention: single convolution
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.activation = nn.Sigmoid()
        
        # Enhanced monitoring for training analysis
        self.last_attention_weights = None
        self.attention_statistics = {
            'mean_attention': 0.0,
            'std_attention': 0.0,
            'center_to_edge_ratio': 1.0,
            'spatial_focus_type': 'uniform',  # center, edge, uniform, clustered
            'attention_concentration': 0.0,  # How concentrated the attention is
            'spatial_entropy': 0.0,
            'hotspot_locations': [],  # Coordinates of attention hotspots
            'attention_spread': 0.0,  # How spread out the attention is
            'center_attention': 0.0,
            'edge_attention': 0.0,
            'gradient_magnitude': 0.0  # How much spatial variation exists
        }

    def forward(self, x):
        # Standard CBAM spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Channel-wise average pooling
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Channel-wise max pooling
        
        # Concatenate along channel dimension
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        
        # Generate spatial attention weights
        spatial_weights = self.conv(spatial_input)
        spatial_weights = self.activation(spatial_weights)
        
        # Store for visualization
        self.last_attention_weights = spatial_weights.detach()
        
        # Calculate comprehensive spatial statistics during training
        if self.training:
            self._update_spatial_statistics(spatial_weights)
        
        # Apply attention
        return x * spatial_weights.expand_as(x)
    
    def _update_spatial_statistics(self, spatial_weights):
        """Calculate and update detailed spatial attention statistics for monitoring"""
        with torch.no_grad():
            # Average across batch dimension to get representative spatial attention
            spatial_attn = spatial_weights.squeeze().mean(dim=0) if len(spatial_weights.shape) > 3 else spatial_weights.squeeze()
            
            if len(spatial_attn.shape) != 3:
                return  # Skip if unexpected shape
            
            # Basic statistics
            mean_attn = spatial_attn.mean().item()
            std_attn = spatial_attn.std().item()
            
            # Spatial concentration (normalized standard deviation)
            concentration = std_attn / (mean_attn + 1e-8)
            
            # Spatial entropy
            flat_attn = spatial_attn.view(-1)
            normalized_attn = flat_attn / (flat_attn.sum() + 1e-8)
            spatial_entropy = -torch.sum(normalized_attn * torch.log(normalized_attn + 1e-8)).item()
            
            # Center vs Edge analysis
            size = spatial_attn.shape[0]  # Assuming cubic
            center = size // 2
            
            # Center region (5x5x5 around center)
            center_start = max(0, center - 2)
            center_end = min(size, center + 3)
            center_region = spatial_attn[center_start:center_end, center_start:center_end, center_start:center_end]
            center_attention = center_region.mean().item()
            
            # Edge regions (corners)
            edge_values = []
            for i in [0, -1]:
                for j in [0, -1]:
                    for k in [0, -1]:
                        edge_slice_i = slice(0, 3) if i == 0 else slice(-3, None)
                        edge_slice_j = slice(0, 3) if j == 0 else slice(-3, None)
                        edge_slice_k = slice(0, 3) if k == 0 else slice(-3, None)
                        edge_region = spatial_attn[edge_slice_i, edge_slice_j, edge_slice_k]
                        edge_values.append(edge_region.mean().item())
            
            edge_attention = np.mean(edge_values)
            center_to_edge_ratio = center_attention / (edge_attention + 1e-8)
            
            # Determine focus type
            if center_to_edge_ratio > 1.2:
                focus_type = 'center'
            elif center_to_edge_ratio < 0.8:
                focus_type = 'edge'
            elif concentration > 0.5:
                focus_type = 'clustered'
            else:
                focus_type = 'uniform'
            
            # Find hotspots (top 5% of attention values)
            threshold = torch.quantile(flat_attn, 0.95)
            hotspot_mask = spatial_attn > threshold
            hotspot_coords = torch.nonzero(hotspot_mask).tolist()
            
            # Attention spread (average distance from center of mass)
            coords = torch.nonzero(torch.ones_like(spatial_attn)).float()
            center_of_mass = torch.sum(coords * spatial_attn.view(-1, 1), dim=0) / torch.sum(spatial_attn)
            distances = torch.norm(coords - center_of_mass, dim=1)
            attention_spread = torch.sum(distances * spatial_attn.view(-1)) / torch.sum(spatial_attn)
            
            # Gradient magnitude (how much spatial variation)
            grad_x = torch.abs(spatial_attn[1:] - spatial_attn[:-1]).mean()
            grad_y = torch.abs(spatial_attn[:, 1:] - spatial_attn[:, :-1]).mean()
            grad_z = torch.abs(spatial_attn[:, :, 1:] - spatial_attn[:, :, :-1]).mean()
            gradient_magnitude = (grad_x + grad_y + grad_z).item() / 3.0
            
            # Update statistics
            self.attention_statistics.update({
                'mean_attention': mean_attn,
                'std_attention': std_attn,
                'center_to_edge_ratio': center_to_edge_ratio,
                'spatial_focus_type': focus_type,
                'attention_concentration': concentration,
                'spatial_entropy': spatial_entropy,
                'hotspot_locations': hotspot_coords[:10],  # Limit to 10 hotspots
                'attention_spread': attention_spread.item(),
                'center_attention': center_attention,
                'edge_attention': edge_attention,
                'gradient_magnitude': gradient_magnitude
            })

class CBAM3D(nn.Module):
    """
    Enhanced Convolutional Block Attention Module (CBAM) for 3D molecular data.
    Uses physics-informed channel attention and improved spatial attention.
    Optimized kernel_size=7 for 0.8Å resolution to capture vdW interactions.
    """
    def __init__(self, in_channels, reduction_ratio=6, kernel_size=7, dropout=0.1):
        super(CBAM3D, self).__init__()
        self.channel_attention = CBAMChannelAttention(in_channels, reduction_ratio, dropout)
        self.spatial_attention = CBAMSpatialAttention(kernel_size)

    def forward(self, x):
        # Apply physics-informed channel attention first, then spatial attention
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class ResidualBlock3D(nn.Module):
    """
    Enhanced 3D Residual Block with:
    - Dual-path design for molecular interactions
    - SiLU activation for better molecular modeling
    - Light Self-Attention for local-global transition
    - Improved gradient flow
    """
    def __init__(self, in_channels, out_channels, stride=1, use_cbam=True, use_self_attention=False):
        super(ResidualBlock3D, self).__init__()
        
        self.use_self_attention = use_self_attention
        
        # Primary path: medium-range interactions (H-bond/electrostatic)
        self.conv1_primary = nn.Conv3d(in_channels, out_channels//2, kernel_size=5, 
                                     stride=stride, padding=2, bias=False)
        self.bn1_primary = nn.BatchNorm3d(out_channels//2)
        
        # Secondary path: local interactions (covalent)
        self.conv1_secondary = nn.Conv3d(in_channels, out_channels//2, kernel_size=3,
                                       stride=stride, padding=1, bias=False)
        self.bn1_secondary = nn.BatchNorm3d(out_channels//2)
        
        # Combine and refine features
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # CBAM attention with optimized kernel size for molecular interactions
        self.cbam = CBAM3D(out_channels, kernel_size=5, dropout=0.1) if use_cbam else nn.Identity()
        
        # Light Self-Attention for enhanced local-global transition (based on log analysis)
        if use_self_attention:
            self.self_attention = LightSelfAttention3D(out_channels, num_heads=4)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        
        # Optimized dropout and SiLU activation for molecular feature learning
        self.dropout = nn.Dropout3d(p=0.02)  # Reduced based on log analysis
        self.activation = nn.SiLU(inplace=True)  # SiLU for enhanced non-linearity
    
    def forward(self, x):
        residual = x
        
        # Dual-path processing for different interaction ranges
        primary_out = self.activation(self.bn1_primary(self.conv1_primary(x)))    # 5x5x5 path
        secondary_out = self.activation(self.bn1_secondary(self.conv1_secondary(x)))  # 3x3x3 path
        
        # Combine both paths
        combined_out = torch.cat([primary_out, secondary_out], dim=1)
        combined_out = self.dropout(combined_out)
        
        # Refine combined features
        out = self.bn2(self.conv2(combined_out))
        
        # Apply CBAM attention
        out = self.cbam(out)
        
        # Apply light self-attention if enabled
        if self.use_self_attention:
            out = self.self_attention(out)
        
        # Add shortcut and apply SiLU
        out += self.shortcut(residual)
        out = self.activation(out)
        
        return out

class LightSelfAttention3D(nn.Module):
    """
    Lightweight Self-Attention for 3D CNN layers to enhance local-global transitions.
    Based on training log analysis showing need for better feature integration.
    """
    def __init__(self, channels, num_heads=4, reduction_ratio=8):
        super(LightSelfAttention3D, self).__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        
        # Ensure reduced_dim is divisible by num_heads
        reduced_dim = max(num_heads, channels // reduction_ratio)
        reduced_dim = (reduced_dim // num_heads) * num_heads  # Make divisible by num_heads
        
        self.reduced_dim = reduced_dim
        self.head_dim = reduced_dim // num_heads
        
        # Lightweight projections
        self.query = nn.Conv3d(channels, reduced_dim, 1, bias=False)
        self.key = nn.Conv3d(channels, reduced_dim, 1, bias=False)
        self.value = nn.Conv3d(channels, reduced_dim, 1, bias=False)
        
        # Output projection back to original dimension
        self.out_proj = nn.Conv3d(reduced_dim, channels, 1, bias=False)
        
        # Normalization and dropout
        self.norm = nn.BatchNorm3d(channels)
        self.dropout = nn.Dropout3d(0.05)
        
        self.scale = self.head_dim ** -0.5
        
        # Store attention weights and statistics for monitoring
        self.last_attention_weights = None
        self.attention_statistics = {
            'self_attention_strength': 0.0,
            'spatial_focus_variance': 0.0,
            'attention_entropy': 0.0,
            'feature_diversity': 0.0
        }
    
    def forward(self, x):
        batch_size, channels, D, H, W = x.shape
        
        # Generate Q, K, V
        q = self.query(x)  # (batch, reduced_dim, D, H, W)
        k = self.key(x)
        v = self.value(x)
        
        # Reshape for multi-head attention
        spatial_size = D * H * W
        q = q.view(batch_size, self.num_heads, self.head_dim, spatial_size)  # (batch, heads, head_dim, spatial)
        k = k.view(batch_size, self.num_heads, self.head_dim, spatial_size)
        v = v.view(batch_size, self.num_heads, self.head_dim, spatial_size)
        
        # Attention computation
        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale  # (batch, heads, spatial, spatial)
        attn = F.softmax(attn, dim=-1)
        
        # Store attention weights and calculate monitoring statistics
        self.last_attention_weights = attn.detach()
        
        # Calculate self-attention monitoring statistics
        with torch.no_grad():
            # Self-attention strength: how much spatial correlation exists
            self_attention_strength = attn.mean().item()
            
            # Spatial focus variance: how uniform attention is across spatial locations
            spatial_variance = attn.var(dim=-1).mean().item()
            
            # Attention entropy: diversity of attention patterns
            flat_attn = attn.view(-1, attn.size(-1))
            attention_entropy = -torch.sum(flat_attn * torch.log(flat_attn + 1e-8), dim=-1).mean().item()
            
            # Feature diversity: how well different heads focus on different patterns
            # Calculate the standard deviation of attention patterns across heads
            try:
                if attn.size(1) > 1:  # Multi-head attention
                    # Get attention pattern for each head: (batch, heads, spatial, spatial) -> (heads, spatial*spatial)
                    head_patterns = attn.view(attn.size(0), attn.size(1), -1).mean(dim=0)  # Average across batch
                    # Calculate how different the attention patterns are across heads
                    head_std = torch.std(head_patterns, dim=0)  # Std across heads
                    head_differences = head_std.mean().item()  # Mean of std values
                    
                    # Check for valid result
                    if torch.isnan(torch.tensor(head_differences)) or torch.isinf(torch.tensor(head_differences)):
                        head_differences = 0.0
                else:
                    head_differences = 0.0  # Single head has no diversity
            except Exception:
                head_differences = 0.0  # Fallback on any error
            
            self.attention_statistics.update({
                'self_attention_strength': self_attention_strength,
                'spatial_focus_variance': spatial_variance,
                'attention_entropy': attention_entropy,
                'feature_diversity': head_differences
            })
        
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(v, attn)  # (batch, heads, head_dim, spatial)
        out = out.view(batch_size, self.reduced_dim, D, H, W)  # (batch, reduced_dim, D, H, W)
        
        # Project back to original dimension
        out = self.out_proj(out)
        
        # Residual connection and normalization
        out = self.norm(x + out)
        
        return out

class AttentionCNNTransformerType2(nn.Module):
    """
    Hybrid 3D CNN-Transformer model for adsorbate-solvent interaction energy prediction.
    Adapted for type_2 format with 28 input features (14 atomic features × 2 groups).
    
    Architecture:
    1. CNN Backbone with CBAM attention for local feature extraction
    2. Vision Transformer head for global dependency modeling
    3. Enhanced classifier for final prediction
    
    This hybrid approach combines:
    - CNN's efficiency in processing 3D voxel grids and extracting local patterns
    - Transformer's ability to capture long-range dependencies in solvent arrangements
    
    Type_2 format specifications:
    - Default 28 input channels (14 atomic features × 2 groups: adsorbate + solvent)
    - Supports dynamic channel numbers via in_channels parameter
    - Enhanced feature analysis for separated channel groups
    """
    def __init__(self, in_channels=28,  # Changed default from 15 to 28 for type_2 format
                 dropout_rate=0.25,  # Reduced from 0.4 - log shows good performance, don't over-regularize
                 use_transformer=True,
                 transformer_dim=200,  # Increased from 180 for better transformer capacity
                 transformer_heads=8,  # Increased from 6 for better head diversity
                 transformer_layers=2,
                 feature_names=None):  # Add feature_names parameter
        
        super(AttentionCNNTransformerType2, self).__init__()
        
        # Store configuration including input channels for dynamic feature analysis
        self.in_channels = in_channels  # Store input channel count for type_2 compatibility
        self.use_transformer = use_transformer
        self.transformer_dim = transformer_dim
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        
        # Store feature names for analysis methods
        self.feature_names = feature_names
        
        # Feature importance learning layer for input features (same as CNN model)
        self.input_attention = CBAMChannelAttention(in_channels, reduction_ratio=3)  # Same as CNN model
        
        # Initial feature extraction (same as CNN model)
        self.initial_conv = nn.Sequential(
            nn.Conv3d(in_channels, 24, kernel_size=3, padding=1, bias=False),  # Same as CNN model
            nn.BatchNorm3d(24),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout3d(0.05)
        )
        
        # Enhanced channel progression with self-attention integration
        self.layer1 = ResidualBlock3D(24, 36, stride=1, use_cbam=True, use_self_attention=False)
        self.pool1 = nn.AvgPool3d(2)  
        
        self.layer2 = ResidualBlock3D(36, 52, stride=1, use_cbam=True, use_self_attention=True)  # Add light self-attention
        self.pool2 = nn.AvgPool3d(2)  
        
        self.layer3 = ResidualBlock3D(52, 72, stride=1, use_cbam=True, use_self_attention=True)  # Add light self-attention
        
        # Final feature layer
        self.layer4 = ResidualBlock3D(72, 96, stride=1, use_cbam=True, use_self_attention=False)
        
        # Keep spatial information with less aggressive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool3d(2)  # 2x2x2 spatial preservation
        
        # Enhanced Vision Transformer for global feature modeling
        if self.use_transformer:
            self.vision_transformer = VisionTransformer3D(
                input_dim=96,
                d_model=transformer_dim,
                num_heads=transformer_heads,
                num_layers=transformer_layers,
                dropout=dropout_rate * 0.75,
                use_cross_attention=True  # Enable cross-attention for multi-level fusion
            )
            classifier_input_dim = transformer_dim
        else:
            self.vision_transformer = None
            classifier_input_dim = 96 * 2 * 2 * 2  # Same flattened features as CNN model = 768
        
        # Enhanced classifier with SiLU activation and better regularization
        if self.use_transformer:
            # Improved classifier for transformer features with better depth
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, 128),
                nn.BatchNorm1d(128),
                nn.SiLU(inplace=True),  # SiLU activation for better molecular modeling
                nn.Dropout(dropout_rate * 0.8),  # Adjusted dropout based on log analysis
                
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout_rate * 0.6),
                
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout_rate * 0.4),
                
                nn.Linear(32, 1)
            )
        else:
            # Enhanced classifier for CNN features with SiLU
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(classifier_input_dim, 192),  # Increased capacity
                nn.BatchNorm1d(192),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(dropout_rate),
                
                nn.Linear(192, 64),            # Intermediate layer
                nn.BatchNorm1d(64),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(dropout_rate * 0.7),
                
                nn.Linear(64, 16),             # Final hidden layer
                nn.BatchNorm1d(16),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(dropout_rate * 0.5),
                
                nn.Linear(16, 1)               # Output
            )
        
        # Initialize weights with specific attention to stability
        self._initialize_weights()
        
        # Add monitoring attributes
        self.training_stats = {
            'epoch_losses': {'train': [], 'test': []},
            'learning_rates': [],
            'gradient_norms': [],
            'weight_norms': {},
            'layer_activations': {},
            'overfitting_metrics': []
        }
        
        # For attention analysis
        self.attention_weights_history = []
    
    def _initialize_weights(self):
        """Enhanced weight initialization for better training stability and reduced overfitting"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # Reduced initialization variance to prevent gradient explosion
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu', a=0.05)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Smaller Xavier initialization to reduce weight magnitudes
                nn.init.xavier_normal_(m.weight, gain=0.3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, monitor_activations=False, detailed_analysis=None, feature_channels=None):
        """
        Enhanced forward pass with comprehensive monitoring (supports type_2 format)
        
        Args:
            x: Input voxel grid (batch_size, 28, 20, 20, 20) for type_2 format, or (batch_size, channels, 20, 20, 20) for dynamic channels
            monitor_activations: Whether to collect basic activation statistics
            detailed_analysis: Whether to collect detailed analysis (auto-enabled if monitor_activations=True)
            feature_channels: dict - mapping of feature channels for molecular interaction analysis
        
        Returns:
            For compatibility with CNN model:
            - If monitor_activations=False: returns x only
            - If monitor_activations=True: returns (x, activation_stats)
        """
        # Auto-enable detailed analysis if monitoring is requested
        if detailed_analysis is None:
            detailed_analysis = monitor_activations
            
        monitoring_data = {
            'activation_stats': {},
            'attention_info': {},
            'layer_contributions': {},
            'feature_analysis': {},
            'gradient_flow': {},
            'training_insights': {}
        }
        
        # 🔬 Enhanced input feature analysis
        if monitor_activations or detailed_analysis:
            # Analyze input features before any processing
            monitoring_data['feature_analysis']['input_features'] = self.analyze_input_features(x)
            
            # 🎯 Special analysis for type_2 format (28 channels with separated groups)
            if x.size(1) == 28:
                monitoring_data['feature_analysis']['type2_channel_groups'] = self.analyze_type2_channel_groups(x)
            
        # Apply learnable feature importance to input features first
        x = self.input_attention(x)
        if monitor_activations:
            monitoring_data['activation_stats'].update(self.analyze_activations(x, "input_attention"))
            monitoring_data['attention_info']['input_attention'] = self.extract_attention_weights_direct(self.input_attention)
            
        if detailed_analysis:
            # Analyze feature importance changes
            monitoring_data['feature_analysis']['learned_importance'] = self.get_learned_feature_importance()
            monitoring_data['layer_contributions']['input_attention'] = self.calculate_layer_contribution(x)
        
        # Initial feature extraction
        x = self.initial_conv(x)
        if monitor_activations:
            monitoring_data['activation_stats'].update(self.analyze_activations(x, "initial_conv"))
        if detailed_analysis:
            monitoring_data['layer_contributions']['initial_conv'] = self.calculate_layer_contribution(x)
        
        # Progressive feature learning with enhanced monitoring
        x = self.layer1(x)
        layer1_features = x.clone() if self.use_transformer else None  # Store for cross-attention
        if monitor_activations:
            monitoring_data['activation_stats'].update(self.analyze_activations(x, "layer1"))
            monitoring_data['attention_info']['layer1'] = self.extract_attention_weights(self.layer1)
        if detailed_analysis:
            monitoring_data['layer_contributions']['layer1'] = self.calculate_layer_contribution(x)
            monitoring_data['training_insights']['layer1_attention_strength'] = self.analyze_attention_strength(self.layer1)
        x = self.pool1(x)  # (batch, 36, 10, 10, 10)
        
        x = self.layer2(x)
        layer2_features = x.clone() if self.use_transformer else None  # Store for cross-attention
        if monitor_activations:
            monitoring_data['activation_stats'].update(self.analyze_activations(x, "layer2"))
            monitoring_data['attention_info']['layer2'] = self.extract_attention_weights(self.layer2)
            
            # 🎯 New: Monitor self-attention in layer2 (if enabled)
            if hasattr(self.layer2, 'self_attention') and self.layer2.self_attention is not None:
                monitoring_data['attention_info']['layer2_self_attention'] = self.extract_self_attention_stats(self.layer2.self_attention)
        if detailed_analysis:
            monitoring_data['layer_contributions']['layer2'] = self.calculate_layer_contribution(x)
            monitoring_data['training_insights']['layer2_attention_strength'] = self.analyze_attention_strength(self.layer2)
            
            # 🎯 New: Detailed self-attention analysis for layer2
            if hasattr(self.layer2, 'self_attention') and self.layer2.self_attention is not None:
                monitoring_data['training_insights']['layer2_self_attention_patterns'] = self.analyze_self_attention_patterns(self.layer2.self_attention)
        x = self.pool2(x)  # (batch, 52, 5, 5, 5)
        
        x = self.layer3(x)
        if monitor_activations:
            monitoring_data['activation_stats'].update(self.analyze_activations(x, "layer3"))
            monitoring_data['attention_info']['layer3'] = self.extract_attention_weights(self.layer3)
            
            # 🎯 New: Monitor self-attention in layer3 (if enabled)
            if hasattr(self.layer3, 'self_attention') and self.layer3.self_attention is not None:
                monitoring_data['attention_info']['layer3_self_attention'] = self.extract_self_attention_stats(self.layer3.self_attention)
        if detailed_analysis:
            monitoring_data['layer_contributions']['layer3'] = self.calculate_layer_contribution(x)
            monitoring_data['training_insights']['layer3_attention_strength'] = self.analyze_attention_strength(self.layer3)
            
            # 🎯 New: Detailed self-attention analysis for layer3
            if hasattr(self.layer3, 'self_attention') and self.layer3.self_attention is not None:
                monitoring_data['training_insights']['layer3_self_attention_patterns'] = self.analyze_self_attention_patterns(self.layer3.self_attention)
        
        # Additional layer for better feature abstraction
        x = self.layer4(x)
        if monitor_activations:
            monitoring_data['activation_stats'].update(self.analyze_activations(x, "layer4"))
            monitoring_data['attention_info']['layer4'] = self.extract_attention_weights(self.layer4)
        if detailed_analysis:
            monitoring_data['layer_contributions']['layer4'] = self.calculate_layer_contribution(x)
            monitoring_data['training_insights']['layer4_attention_strength'] = self.analyze_attention_strength(self.layer4)
        
        # Adaptive pooling to 2x2x2 for better feature preservation
        x = self.adaptive_pool(x)  # (batch, 96, 2, 2, 2)
        if monitor_activations:
            monitoring_data['activation_stats'].update(self.analyze_activations(x, "adaptive_pool"))
        
        # 🧠 Enhanced Vision Transformer with cross-attention
        if self.use_transformer:
            # Use transformer to process CNN features globally with cross-attention
            low_level_features = layer2_features
            
            # Always use simple mode - no complex branching
            x = self.vision_transformer(x, low_level_features=low_level_features, feature_channels=feature_channels)
            
            if monitor_activations:
                monitoring_data['activation_stats'].update(self.analyze_activations(x, "vision_transformer"))
                monitoring_data['attention_info']['vision_transformer'] = self.vision_transformer.get_attention_summary()
                
                # Always get cross-attention stats
                try:
                    cross_stats = self.get_cross_attention_stats()
                    if cross_stats:
                        monitoring_data['attention_info']['cross_attention'] = cross_stats
                except Exception as e:
                    monitoring_data['attention_info']['cross_attention'] = {'error': str(e)}
        else:
            # Traditional CNN path: flatten for classifier
            x = x.view(x.size(0), -1)  # (batch, 96*2*2*2 = 768)
        
        # Classification with gradient flow analysis
        x_before_classifier = x.clone() if detailed_analysis else None
        x = self.classifier(x)  # (batch, 1)
        
        if monitor_activations:
            monitoring_data['activation_stats'].update(self.analyze_activations(x, "output"))
            # Store attention information
            self.attention_weights_history.append(monitoring_data['attention_info'])
            
        if detailed_analysis:
            monitoring_data['training_insights']['classifier_contribution'] = self.analyze_classifier_contribution(x_before_classifier, x)
            monitoring_data['gradient_flow'] = self.analyze_gradient_flow()
            monitoring_data['training_insights']['model_architecture_efficiency'] = self.analyze_architecture_efficiency()
        
        # For compatibility with CNN model, return activation_stats directly
        if monitor_activations:
            # Return in the same format as CNN model: (x, activation_stats)
            return x, monitoring_data['activation_stats']
        return x
    

    
    def get_layer_weight_norms(self):
        """Get L2 norms of weights for each layer for monitoring"""
        weight_norms = {}
        for name, param in self.named_parameters():
            if param.requires_grad and 'weight' in name:
                weight_norms[name] = param.data.norm(2).item()
        return weight_norms
    
    def get_gradient_norms(self):
        """Get gradient norms for monitoring training dynamics"""
        grad_norms = {}
        total_norm = 0
        param_count = 0
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                grad_norms[name] = param_norm.item()
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        total_norm = total_norm ** (1. / 2)
        avg_norm = total_norm / param_count if param_count > 0 else 0
        
        return {
            'total_norm': total_norm,
            'avg_norm': avg_norm,
            'layer_norms': grad_norms,
            'param_count': param_count
        }
    
    def analyze_feature_utilization(self, x):
        """Analyze how each input feature is being utilized (supports type_2 format with dynamic channel count)"""
        if x.dim() != 5 or x.size(1) != self.in_channels:
            return {}
        
        feature_stats = {}
        # Use feature names passed to model - must be provided
        if self.feature_names is not None:
            feature_names = self.feature_names[:x.size(1)]
        else:
            raise ValueError("Feature names not provided to model. Ensure feature_names are passed during model initialization.")
        
        for i, feature_name in enumerate(feature_names):
            feature_data = x[:, i, :, :, :].detach().cpu().numpy()
            # Calculate information content as a combination of std and non_zero_ratio
            std_val = float(np.std(feature_data))
            non_zero_ratio = float(np.mean(feature_data != 0))
            information_content = std_val * non_zero_ratio  # Higher values indicate more informative features
            
            feature_stats[f'feature_{i:02d}_{feature_name}'] = {
                'mean': float(np.mean(feature_data)),
                'std': std_val,
                'non_zero_ratio': non_zero_ratio,
                'dynamic_range': float(np.max(feature_data) - np.min(feature_data)) if std_val > 1e-6 else 0.0,
                'information_content': information_content
            }
        
        return feature_stats
    
    def get_batchnorm_statistics(self):
        """Get BatchNorm layer statistics to monitor internal covariate shift"""
        bn_stats = {}
        for name, module in self.named_modules():
            if isinstance(module, (nn.BatchNorm3d, nn.BatchNorm1d)):
                if hasattr(module, 'running_mean') and hasattr(module, 'running_var'):
                    bn_stats[name] = {
                        'running_mean_norm': float(module.running_mean.norm().item()),
                        'running_var_mean': float(module.running_var.mean().item()),
                        'weight_norm': float(module.weight.norm().item()) if module.weight is not None else 0.0,
                        'bias_norm': float(module.bias.norm().item()) if module.bias is not None else 0.0
                    }
        return bn_stats
    
    def analyze_attention_evolution(self):
        """Analyze how attention weights evolve over time during training"""
        if not self.attention_weights_history:
            return {}
        
        evolution = {}
        if len(self.attention_weights_history) >= 2:
            latest = self.attention_weights_history[-1]
            previous = self.attention_weights_history[-2]
            
            for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
                if layer in latest and layer in previous:
                    layer_evolution = {}
                    
                    # Channel attention evolution
                    if 'channel_weights' in latest[layer] and 'channel_weights' in previous[layer]:
                        curr_ch = latest[layer]['channel_weights']
                        prev_ch = previous[layer]['channel_weights']
                        layer_evolution['channel_std_change'] = curr_ch['std'] - prev_ch['std']
                        layer_evolution['channel_mean_change'] = curr_ch['mean'] - prev_ch['mean']
                    
                    # Spatial attention evolution  
                    if 'spatial_weights' in latest[layer] and 'spatial_weights' in previous[layer]:
                        curr_sp = latest[layer]['spatial_weights']
                        prev_sp = previous[layer]['spatial_weights']
                        layer_evolution['spatial_std_change'] = curr_sp['std'] - prev_sp['std']
                        layer_evolution['spatial_mean_change'] = curr_sp['mean'] - prev_sp['mean']
                    
                    evolution[layer] = layer_evolution
        
        return evolution
    
    def analyze_activations(self, x, layer_name_prefix):
        """Enhanced activation analysis with more detailed statistics"""
        stats = {}
        
        # Convert to numpy for analysis
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
        
        # Basic statistics
        stats[f'{layer_name_prefix}_mean'] = np.mean(x_np)
        stats[f'{layer_name_prefix}_std'] = np.std(x_np)
        stats[f'{layer_name_prefix}_min'] = np.min(x_np)
        stats[f'{layer_name_prefix}_max'] = np.max(x_np)
        stats[f'{layer_name_prefix}_zeros_pct'] = np.mean(x_np == 0) * 100
        
        # Additional detailed statistics for better monitoring
        stats[f'{layer_name_prefix}_abs_mean'] = np.mean(np.abs(x_np))
        stats[f'{layer_name_prefix}_negative_pct'] = np.mean(x_np < 0) * 100
        stats[f'{layer_name_prefix}_saturation_pct'] = np.mean(np.abs(x_np) > 0.9) * 100
        
        # Channel-wise analysis for multi-channel tensors
        if len(x_np.shape) >= 2:
            channel_means = np.mean(x_np, axis=tuple(range(2, len(x_np.shape))))
            if len(channel_means.shape) > 1:  # Multi-channel
                stats[f'{layer_name_prefix}_channel_diversity'] = np.std(np.mean(channel_means, axis=0))
        
        return stats
    
    def analyze_input_features(self, x):
        """🔬 Comprehensive input feature analysis for model optimization (supports type_2 format with dynamic channel count)"""
        if x.dim() != 5 or x.size(1) != self.in_channels:
            return {}
        
        feature_analysis = {}
        # Use feature names passed to model - must be provided
        if self.feature_names is not None:
            feature_names = self.feature_names[:x.size(1)]
        else:
            raise ValueError("Feature names not provided to model. Ensure feature_names are passed during model initialization.")
        
        for i, feature_name in enumerate(feature_names):
            feature_data = x[:, i, :, :, :].detach().cpu()
            
            # Basic statistics
            feature_stats = {
                'mean': float(feature_data.mean().item()),
                'std': float(feature_data.std().item()),
                'min': float(feature_data.min().item()),
                'max': float(feature_data.max().item()),
                'non_zero_ratio': float((feature_data != 0).float().mean().item()),
                'sparsity': float((feature_data == 0).float().mean().item()),
                'dynamic_range': float((feature_data.max() - feature_data.min()).item())
            }
            
            # Spatial distribution analysis
            if len(feature_data.shape) == 5:  # (batch, 1, D, H, W)
                spatial_variance = torch.var(feature_data, dim=(2, 3, 4)).mean().item()
            elif len(feature_data.shape) == 4:  # (batch, D, H, W)
                spatial_variance = torch.var(feature_data, dim=(1, 2, 3)).mean().item()
            else:  # fallback
                spatial_variance = torch.var(feature_data).item()
            feature_stats['spatial_variance'] = float(spatial_variance)
            
            # Information content estimation
            if feature_stats['std'] > 1e-6:
                feature_stats['information_content'] = float(feature_stats['std'] * feature_stats['non_zero_ratio'])
            else:
                feature_stats['information_content'] = 0.0
            
            feature_analysis[f'feature_{i:02d}_{feature_name}'] = feature_stats
        
        # Feature correlation analysis (simplified)
        batch_flattened = x.view(x.size(0), x.size(1), -1).mean(dim=2)  # (batch, self.in_channels)
        correlation_matrix = torch.corrcoef(batch_flattened.T)
        feature_analysis['feature_correlations'] = {
            'max_correlation': float(correlation_matrix[correlation_matrix != 1].max().item()),
            'avg_correlation': float(correlation_matrix[correlation_matrix != 1].abs().mean().item())
        }
        
        return feature_analysis
    
    def analyze_type2_channel_groups(self, x):
        """🔬 Specialized analysis for type_2 format with separated adsorbate and solvent channel groups"""
        if x.dim() != 5 or x.size(1) != 28:
            return {}
        
        # Type_2 format: 28 channels = 14 adsorbate features + 14 solvent features
        adsorbate_channels = x[:, :14, :, :, :]  # First 14 channels for adsorbate
        solvent_channels = x[:, 14:, :, :, :]    # Last 14 channels for solvent
        
        type2_analysis = {}
        
        # Analyze adsorbate features (channels 0-13)
        adsorbate_stats = {}
        for i in range(14):
            channel_data = adsorbate_channels[:, i, :, :, :].detach().cpu()
            adsorbate_stats[f'adsorbate_feature_{i:02d}'] = {
                'mean': float(channel_data.mean().item()),
                'std': float(channel_data.std().item()),
                'non_zero_ratio': float((channel_data != 0).float().mean().item()),
                'information_content': float(channel_data.std().item() * (channel_data != 0).float().mean().item())
            }
        
        # Analyze solvent features (channels 14-27)
        solvent_stats = {}
        for i in range(14):
            channel_data = solvent_channels[:, i, :, :, :].detach().cpu()
            solvent_stats[f'solvent_feature_{i:02d}'] = {
                'mean': float(channel_data.mean().item()),
                'std': float(channel_data.std().item()),
                'non_zero_ratio': float((channel_data != 0).float().mean().item()),
                'information_content': float(channel_data.std().item() * (channel_data != 0).float().mean().item())
            }
        
        # Group-level analysis
        adsorbate_magnitude = torch.mean(torch.abs(adsorbate_channels)).item()
        solvent_magnitude = torch.mean(torch.abs(solvent_channels)).item()
        
        type2_analysis['adsorbate_group'] = {
            'features': adsorbate_stats,
            'group_magnitude': float(adsorbate_magnitude),
            'group_sparsity': float((adsorbate_channels == 0).float().mean().item()),
            'spatial_coverage': float((adsorbate_channels.sum(dim=1) != 0).float().mean().item())  # Non-zero voxels
        }
        
        type2_analysis['solvent_group'] = {
            'features': solvent_stats,
            'group_magnitude': float(solvent_magnitude),
            'group_sparsity': float((solvent_channels == 0).float().mean().item()),
            'spatial_coverage': float((solvent_channels.sum(dim=1) != 0).float().mean().item())  # Non-zero voxels
        }
        
        # Cross-group interaction analysis
        if adsorbate_magnitude > 0 and solvent_magnitude > 0:
            magnitude_ratio = adsorbate_magnitude / solvent_magnitude
            type2_analysis['group_interaction'] = {
                'adsorbate_to_solvent_ratio': float(magnitude_ratio),
                'interaction_balance': 'adsorbate_dominant' if magnitude_ratio > 2 else 'solvent_dominant' if magnitude_ratio < 0.5 else 'balanced'
            }
        
        return type2_analysis
    
    def calculate_layer_contribution(self, x):
        """📊 Calculate layer's contribution to overall feature representation"""
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
        
        contribution = {
            'feature_magnitude': float(np.mean(np.abs(x_np))),
            'feature_diversity': float(np.std(x_np)),
            'effective_channels': 0,
            'spatial_focus': 0.0
        }
        
        if len(x_np.shape) >= 3:  # Has spatial dimensions
            # Calculate effective channels (channels with significant variance)
            if len(x_np.shape) == 5:  # (batch, channel, D, H, W)
                channel_vars = np.var(x_np, axis=(0, 2, 3, 4))
                contribution['effective_channels'] = int(np.sum(channel_vars > 0.01))
                
                # Spatial focus analysis (center vs edge)
                center_region = x_np[:, :, x_np.shape[2]//4:3*x_np.shape[2]//4, 
                                         x_np.shape[3]//4:3*x_np.shape[3]//4,
                                         x_np.shape[4]//4:3*x_np.shape[4]//4]
                center_magnitude = np.mean(np.abs(center_region))
                total_magnitude = np.mean(np.abs(x_np))
                contribution['spatial_focus'] = float(center_magnitude / (total_magnitude + 1e-8))
        
        return contribution
    
    def analyze_attention_strength(self, layer):
        """📈 Analyze attention mechanism strength and effectiveness"""
        attention_analysis = {}
        
        if hasattr(layer, 'cbam'):
            # Channel attention analysis
            if hasattr(layer.cbam.channel_attention, 'last_attention_weights'):
                ch_weights = layer.cbam.channel_attention.last_attention_weights
                if ch_weights is not None:
                    attention_analysis['channel_attention'] = {
                        'uniformity': float(1.0 / (ch_weights.std().item() + 1e-8)),
                        'selectivity': float(ch_weights.max().item() / ch_weights.mean().item()),
                        'effective_channels': int((ch_weights > ch_weights.mean()).sum().item())
                    }
            
            # Spatial attention analysis
            if hasattr(layer.cbam.spatial_attention, 'last_attention_weights'):
                sp_weights = layer.cbam.spatial_attention.last_attention_weights
                if sp_weights is not None:
                    # Center focus analysis
                    center_idx = sp_weights.shape[-1] // 2
                    center_weights = sp_weights[:, :, center_idx-1:center_idx+2, 
                                                     center_idx-1:center_idx+2,
                                                     center_idx-1:center_idx+2]
                    center_focus = center_weights.mean() / sp_weights.mean()
                    
                    attention_analysis['spatial_attention'] = {
                        'center_focus_ratio': float(center_focus.item()),
                        'attention_spread': float(sp_weights.std().item()),
                        'peak_attention': float(sp_weights.max().item())
                    }
        
        return attention_analysis
    
    def analyze_transformer_attention_detailed(self, attention_maps):
        """🧠 Detailed analysis of transformer attention patterns"""
        if not attention_maps:
            return {}
        
        detailed_analysis = {}
        for layer_idx, attention_map in enumerate(attention_maps):
            # attention_map: (batch_size, num_heads, seq_len, seq_len)
            layer_analysis = {}
            
            # Average across batch for analysis
            avg_attention = attention_map.mean(dim=0)  # (num_heads, seq_len, seq_len)
            
            # Head specialization analysis
            head_specializations = []
            for head_idx in range(avg_attention.size(0)):
                head_attn = avg_attention[head_idx]
                
                # Analyze attention patterns
                diagonal_strength = torch.diag(head_attn).mean().item()
                off_diagonal_strength = (head_attn.sum() - torch.diag(head_attn).sum()).item() / (head_attn.numel() - head_attn.size(0))
                attention_entropy = -torch.sum(head_attn * torch.log(head_attn + 1e-8)).item()
                
                head_specializations.append({
                    'head_id': head_idx,
                    'self_attention_strength': diagonal_strength,
                    'cross_attention_strength': off_diagonal_strength,
                    'attention_entropy': attention_entropy,
                    'specialization_type': self.classify_attention_pattern(head_attn)
                })
            
            layer_analysis['head_specializations'] = head_specializations
            layer_analysis['layer_attention_diversity'] = float(torch.std(avg_attention.view(-1)).item())
            layer_analysis['global_attention_strength'] = float(avg_attention.mean().item())
            
            detailed_analysis[f'transformer_layer_{layer_idx+1}'] = layer_analysis
        
        return detailed_analysis
    
    def classify_attention_pattern(self, attention_matrix):
        """🔍 Classify attention pattern type for interpretability"""
        # Simple heuristic classification
        diagonal_strength = torch.diag(attention_matrix).mean()
        off_diagonal_std = attention_matrix[~torch.eye(attention_matrix.size(0), dtype=bool)].std()
        
        if diagonal_strength > 0.3:
            return "self_focused"
        elif off_diagonal_std > 0.1:
            return "global_interaction"
        else:
            return "uniform_attention"
    
    def compare_transformer_cnn_contribution(self, cnn_features, transformer_features):
        """⚖️ Compare CNN vs Transformer contribution"""
        if cnn_features is None or transformer_features is None:
            return {}
        
        # Flatten for comparison
        cnn_flat = cnn_features.view(cnn_features.size(0), -1)
        transformer_flat = transformer_features.view(transformer_features.size(0), -1)
        
        comparison = {
            'cnn_feature_magnitude': float(torch.mean(torch.abs(cnn_flat)).item()),
            'transformer_feature_magnitude': float(torch.mean(torch.abs(transformer_flat)).item()),
            'feature_compression_ratio': float(cnn_flat.numel() / transformer_flat.numel()),
            'information_preservation': float(torch.corrcoef(torch.stack([
                cnn_flat.mean(dim=1), transformer_flat.mean(dim=1)
            ]))[0, 1].item()) if cnn_flat.size(0) > 1 else 0.0
        }
        
        return comparison
    
    def analyze_classifier_contribution(self, classifier_input, classifier_output):
        """🎯 Analyze classifier's contribution to final prediction"""
        if classifier_input is None or classifier_output is None:
            return {}
        
        input_magnitude = torch.mean(torch.abs(classifier_input)).item()
        output_magnitude = torch.mean(torch.abs(classifier_output)).item()
        
        return {
            'input_magnitude': float(input_magnitude),
            'output_magnitude': float(output_magnitude),
            'amplification_factor': float(output_magnitude / (input_magnitude + 1e-8)),
            'prediction_confidence': float(torch.std(classifier_output).item())
        }
    
    def analyze_gradient_flow(self):
        """🌊 Analyze gradient flow through the network"""
        gradient_info = {}
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                param_norm = param.norm().item()
                
                gradient_info[name] = {
                    'gradient_norm': grad_norm,
                    'parameter_norm': param_norm,
                    'gradient_to_param_ratio': grad_norm / (param_norm + 1e-8)
                }
        
        return gradient_info
    
    def analyze_architecture_efficiency(self):
        """🏗️ Analyze overall architecture efficiency"""
        efficiency_metrics = {}
        
        # Parameter utilization analysis
        total_params = sum(p.numel() for p in self.parameters())
        active_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        efficiency_metrics['parameter_efficiency'] = {
            'total_params': total_params,
            'active_params': active_params,
            'param_utilization': active_params / total_params
        }
        
        # Component efficiency analysis
        component_analysis = self.get_model_complexity_analysis()
        efficiency_metrics['component_balance'] = component_analysis['parameter_percentages']
        
        return efficiency_metrics
    
    def print_training_insights(self, epoch=None, batch_idx=None):
        """📋 Print comprehensive training insights for log monitoring"""
        print(f"\n{'='*80}")
        print(f"🔬 TRAINING INSIGHTS - Epoch {epoch}, Batch {batch_idx}" if epoch is not None else "🔬 MODEL ANALYSIS")
        print(f"{'='*80}")
        
        # Create monitoring data from stored information
        monitoring_data = {}
        
        # Add self-attention data if available
        if hasattr(self, '_self_attention_data'):
            monitoring_data['self_attention'] = self._self_attention_data
        
        # Add cross-attention data if available  
        if hasattr(self, '_cross_attention_data'):
            monitoring_data['cross_attention'] = self._cross_attention_data
        
        # Add transformer attention data if available
        if hasattr(self, '_transformer_attention_data'):
            monitoring_data['transformer_attention'] = self._transformer_attention_data
        
        # 1. Feature Analysis
        if 'feature_analysis' in monitoring_data:
            print(f"\n📊 INPUT FEATURE ANALYSIS:")
            feature_analysis = monitoring_data['feature_analysis']
            
            if 'learned_importance' in feature_analysis:
                print(f"🎯 Top 5 Most Important Features:")
                importance = feature_analysis['learned_importance']
                sorted_features = sorted(importance.items(), 
                                       key=lambda x: x[1]['normalized_weight'], reverse=True)[:5]
                for i, (name, data) in enumerate(sorted_features, 1):
                    print(f"   {i}. {name}: {data['normalized_weight']:.4f} (rank {data['rank']})")
            
            if 'input_features' in feature_analysis:
                print(f"📈 Feature Information Content (Top 5):")
                input_features = feature_analysis['input_features']
                sorted_info = sorted([(k, v['information_content']) for k, v in input_features.items() 
                                    if isinstance(v, dict) and 'information_content' in v], 
                                   key=lambda x: x[1], reverse=True)[:5]
                for name, info_content in sorted_info:
                    print(f"   {name.split('_', 2)[-1]}: {info_content:.4f}")
            
            # 🎯 Type_2 format special analysis
            if 'type2_channel_groups' in feature_analysis:
                type2_analysis = feature_analysis['type2_channel_groups']
                print(f"\n🧪 TYPE_2 FORMAT CHANNEL GROUP ANALYSIS:")
                
                # Adsorbate group analysis
                adsorbate_group = type2_analysis.get('adsorbate_group', {})
                print(f"   🔗 Adsorbate Group (channels 0-13):")
                print(f"      Magnitude: {adsorbate_group.get('group_magnitude', 0):.4f}")
                print(f"      Sparsity: {adsorbate_group.get('group_sparsity', 0):.3f}")
                print(f"      Spatial coverage: {adsorbate_group.get('spatial_coverage', 0):.3f}")
                
                # Solvent group analysis  
                solvent_group = type2_analysis.get('solvent_group', {})
                print(f"   💧 Solvent Group (channels 14-27):")
                print(f"      Magnitude: {solvent_group.get('group_magnitude', 0):.4f}")
                print(f"      Sparsity: {solvent_group.get('group_sparsity', 0):.3f}")
                print(f"      Spatial coverage: {solvent_group.get('spatial_coverage', 0):.3f}")
                
                # Group interaction analysis
                if 'group_interaction' in type2_analysis:
                    interaction = type2_analysis['group_interaction']
                    ratio = interaction.get('adsorbate_to_solvent_ratio', 1.0)
                    balance = interaction.get('interaction_balance', 'unknown')
                    print(f"   ⚖️ Group Interaction: {balance} (ratio: {ratio:.2f})")
        
        # 2. Layer Contributions
        if 'layer_contributions' in monitoring_data:
            print(f"\n🏗️ LAYER CONTRIBUTION ANALYSIS:")
            contributions = monitoring_data['layer_contributions']
            for layer_name, contrib in contributions.items():
                print(f"   {layer_name}: magnitude={contrib['feature_magnitude']:.4f}, "
                      f"focus={contrib['spatial_focus']:.3f}, "
                      f"channels={contrib.get('effective_channels', 'N/A')}")
        
        # 3. Attention Analysis
        if 'attention_info' in monitoring_data:
            print(f"\n👁️ ATTENTION MECHANISM ANALYSIS:")
            attention_info = monitoring_data['attention_info']
            
            # CBAM attention summary
            for layer_name, attn_data in attention_info.items():
                if layer_name.startswith('layer') and 'channel_weights' in attn_data:
                    ch_weights = attn_data['channel_weights']
                    print(f"   {layer_name} CBAM: selectivity={ch_weights['max']/ch_weights['mean']:.2f}, "
                          f"diversity={ch_weights['std']:.4f}")
            
            # Transformer attention if available
            if 'vision_transformer_detailed' in attention_info:
                print(f"🧠 TRANSFORMER ATTENTION PATTERNS:")
                transformer_details = attention_info['vision_transformer_detailed']
                for layer_name, layer_data in transformer_details.items():
                    if 'head_specializations' in layer_data:
                        specializations = layer_data['head_specializations']
                        pattern_counts = {}
                        for head in specializations:
                            pattern = head['specialization_type']
                            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                        print(f"   {layer_name}: {dict(pattern_counts)}")
        
        # 4. Training Insights
        if 'training_insights' in monitoring_data:
            print(f"\n🎯 TRAINING OPTIMIZATION INSIGHTS:")
            insights = monitoring_data['training_insights']
            
            if 'transformer_vs_cnn_contribution' in insights:
                comparison = insights['transformer_vs_cnn_contribution']
                print(f"   CNN vs Transformer: {comparison['cnn_feature_magnitude']:.4f} vs "
                      f"{comparison['transformer_feature_magnitude']:.4f}")
            
            if 'model_architecture_efficiency' in insights:
                efficiency = insights['model_architecture_efficiency']
                balance = efficiency['component_balance']
                print(f"   Architecture balance: CNN={balance['convolutional']:.1f}%, "
                      f"Transformer={balance.get('vision_transformer', 0):.1f}%, "
                      f"Attention={balance['cbam_attention']:.1f}%")
            
            # 🎯 NEW: Cross-attention fusion monitoring
            if 'cross_attention_fusion_strength' in insights:
                fusion_eval = insights['cross_attention_fusion_strength']
                if 'information_preservation' in fusion_eval:
                    print(f"   Cross-attention fusion: preservation={fusion_eval['information_preservation']:.3f}, "
                          f"efficiency={fusion_eval['fusion_efficiency']:.3f}")
        
        # 5. Cross-Attention Analysis (NEW)
        if 'attention_info' in monitoring_data:
            attention_info = monitoring_data['attention_info']
            
            # Self-attention in CNN layers
            self_attention_layers = [key for key in attention_info.keys() if 'self_attention' in key]
            if self_attention_layers:
                print(f"\n🔄 SELF-ATTENTION IN CNN LAYERS:")
                for layer_key in self_attention_layers:
                    layer_data = attention_info[layer_key]
                    if 'self_attention_strength' in layer_data:
                        print(f"   {layer_key}: strength={layer_data['self_attention_strength']:.4f}, "
                              f"entropy={layer_data.get('attention_entropy', 0):.4f}")
            
            # Cross-attention analysis
            if 'cross_attention_analysis' in attention_info:
                cross_analysis = attention_info['cross_attention_analysis']
                print(f"\n🔀 CROSS-ATTENTION FUSION ANALYSIS:")
                if 'fusion_layers_count' in cross_analysis:
                    print(f"   Fusion layers: {cross_analysis['fusion_layers_count']}")
                    print(f"   Average fusion strength: {cross_analysis.get('average_fusion_strength', 0):.4f}")
                    print(f"   Fusion diversity: {cross_analysis.get('fusion_diversity', 0):.4f}")
            
            # Basic cross-attention stats
            if 'cross_attention_basic' in attention_info:
                basic_stats = attention_info['cross_attention_basic']
                if 'fusion_strength_distribution' in basic_stats:
                    print(f"\n🔀 CROSS-ATTENTION BASIC STATS:")
                    for layer_stat in basic_stats['fusion_strength_distribution'][:2]:  # Show first 2 layers
                        print(f"   Layer {layer_stat['layer_index']}: attention={layer_stat['mean_attention']:.4f}, "
                              f"sparsity={layer_stat['attention_sparsity']:.2f}")
        
        print(f"{'='*80}\n")
    
    def get_optimization_recommendations(self):
        """💡 Generate specific optimization recommendations based on available monitoring data"""
        recommendations = []
        
        # Create monitoring data from available sources
        monitoring_data = {}
        
        # Check for stored detailed monitoring data
        if hasattr(self, '_self_attention_data'):
            monitoring_data['self_attention'] = self._self_attention_data
        if hasattr(self, '_cross_attention_data'):
            monitoring_data['cross_attention'] = self._cross_attention_data
        if hasattr(self, '_transformer_attention_data'):
            monitoring_data['transformer_attention'] = self._transformer_attention_data
        
        # Fallback: Use attention history if detailed data not available
        if not monitoring_data and hasattr(self, 'attention_weights_history') and self.attention_weights_history:
            latest_attention = self.attention_weights_history[-1]
            
            # Analyze CBAM attention patterns from history
            cbam_analysis = {}
            for layer_name, attention_data in latest_attention.items():
                if layer_name.startswith('layer') and isinstance(attention_data, dict):
                    if 'channel_weights' in attention_data:
                        ch_weights = attention_data['channel_weights']
                        # Check for attention collapse
                        if ch_weights.get('std', 0) < 0.01:
                            recommendations.append(
                                f"⚠️ Attention collapse in {layer_name}: σ={ch_weights.get('std', 0):.4f}. "
                                f"Consider attention regularization or different initialization."
                            )
                        
                        # Check for extreme selectivity
                        mean_val = ch_weights.get('mean', 0)
                        max_val = ch_weights.get('max', 0)
                        if mean_val > 0 and max_val / mean_val > 5:
                            recommendations.append(
                                f"💡 High attention selectivity in {layer_name} (ratio={max_val/mean_val:.1f}). "
                                f"Model is learning focused feature selection."
                            )
            
            # Check for transformer attention patterns
            if 'vision_transformer_detailed' in latest_attention:
                transformer_details = latest_attention['vision_transformer_detailed']
                if isinstance(transformer_details, dict):
                    for layer_name, layer_data in transformer_details.items():
                        if 'head_specializations' in layer_data:
                            specializations = layer_data['head_specializations']
                            if isinstance(specializations, list) and len(specializations) > 0:
                                # Analyze head diversity
                                pattern_types = [head.get('specialization_type', 'unknown') for head in specializations]
                                unique_patterns = len(set(pattern_types))
                                total_heads = len(pattern_types)
                                
                                if unique_patterns < total_heads // 2:
                                    recommendations.append(
                                        f"⚠️ Low head diversity in {layer_name}: {unique_patterns}/{total_heads} unique patterns. "
                                        f"Consider adjusting attention temperature or head initialization."
                                    )
                                else:
                                    recommendations.append(
                                        f"✅ Good head diversity in {layer_name}: {unique_patterns}/{total_heads} unique patterns."
                                    )
            
            monitoring_data['fallback_attention'] = latest_attention
        
        # Fallback: Use current model state for basic recommendations
        if not monitoring_data:
            # Check for basic model properties
            total_params = sum(p.numel() for p in self.parameters())
            
            if total_params > 2_000_000:
                recommendations.append("💡 Large model detected. Monitor for overfitting and consider regularization.")
            elif total_params < 500_000:
                recommendations.append("💡 Compact model detected. May benefit from increased capacity if underfitting.")
            
            # Check dropout rate
            if hasattr(self, 'dropout_rate') and self.dropout_rate:
                if self.dropout_rate > 0.5:
                    recommendations.append(f"⚠️ High dropout rate ({self.dropout_rate}). May indicate overfitting issues.")
                elif self.dropout_rate < 0.1:
                    recommendations.append(f"💡 Low dropout rate ({self.dropout_rate}). Monitor for overfitting.")
            
            # Architecture-specific recommendations
            if self.use_transformer:
                recommendations.append("🧠 Transformer enabled: Monitor cross-attention fusion effectiveness.")
                if self.transformer_layers > 2:
                    recommendations.append("⚠️ Deep transformer: Watch for attention collapse in deeper layers.")
            else:
                recommendations.append("🔧 CNN-only mode: Consider enabling transformer for global dependency modeling.")
        
        # Feature importance analysis using current input attention weights
        try:
            if hasattr(self, 'input_attention') and hasattr(self.input_attention, 'last_attention_weights'):
                ch_weights = self.input_attention.last_attention_weights
                if ch_weights is not None:
                    weights_np = ch_weights.detach().cpu().numpy().flatten()
                    weight_std = weights_np.std()
                    
                    if weight_std < 0.05:
                        recommendations.append(
                            f"⚠️ Low input feature diversity (σ={weight_std:.3f}). "
                            f"All features weighted similarly - check feature engineering."
                        )
                    elif weight_std > 0.3:
                        recommendations.append(
                            f"💡 High input feature selectivity (σ={weight_std:.3f}). "
                            f"Model is learning strong feature preferences."
                        )
                    
                    # Check for dominant features
                    max_weight = weights_np.max()
                    mean_weight = weights_np.mean()
                    if max_weight > mean_weight * 3:
                        recommendations.append(
                            f"⚠️ Dominant feature detected (max/mean={max_weight/mean_weight:.1f}). "
                            f"Consider feature normalization or regularization."
                        )
        except Exception:
            pass  # Skip if input attention analysis fails
        
        # Detailed analysis from monitoring data (if available)
        if 'feature_analysis' in monitoring_data:
            feature_analysis = monitoring_data['feature_analysis']
            if 'learned_importance' in feature_analysis and feature_analysis['learned_importance']:
                importance = feature_analysis['learned_importance']
                top_features = sorted(importance.items(), key=lambda x: x[1]['normalized_weight'], reverse=True)
                
                # Check for feature imbalance
                if len(top_features) >= 2:
                    top_weight = top_features[0][1]['normalized_weight']
                    bottom_weight = top_features[-1][1]['normalized_weight']
                    if bottom_weight > 0 and top_weight / bottom_weight > 10:
                        recommendations.append(
                            f"⚠️ Feature imbalance: {top_features[0][0]} is {top_weight/bottom_weight:.1f}x more important. "
                            f"Consider feature engineering or regularization."
                        )
        
        # Layer contribution analysis (if available)
        if 'layer_contributions' in monitoring_data and monitoring_data['layer_contributions']:
            contributions = monitoring_data['layer_contributions']
            magnitudes = []
            for contrib in contributions.values():
                if isinstance(contrib, dict) and 'feature_magnitude' in contrib:
                    magnitudes.append(contrib['feature_magnitude'])
            
            if len(magnitudes) > 1:
                magnitude_ratio = max(magnitudes) / min(magnitudes) if min(magnitudes) > 0 else 0
                if magnitude_ratio > 5:
                    recommendations.append(
                        f"⚠️ Large magnitude variation between layers ({magnitude_ratio:.1f}x). "
                        f"Consider gradient clipping or layer normalization."
                    )
        
        # Default positive recommendations if no issues found
        if not recommendations:
            recommendations.extend([
                "✅ Model appears to be training normally",
                "💡 Continue monitoring attention patterns and gradient norms",
                "🔄 Current architecture shows balanced attention distribution"
            ])
        
        return recommendations
    
    def extract_attention_weights(self, layer):
        """Extract attention weights from a residual block with CBAM"""
        attention_info = {}
        
        if hasattr(layer, 'cbam') and hasattr(layer.cbam, 'channel_attention'):
            # Channel attention weights
            if hasattr(layer.cbam.channel_attention, 'last_attention_weights'):
                channel_weights = layer.cbam.channel_attention.last_attention_weights
                if channel_weights is not None:
                    attention_info['channel_weights'] = {
                        'mean': float(channel_weights.mean().item()),
                        'std': float(channel_weights.std().item()),
                        'min': float(channel_weights.min().item()),
                        'max': float(channel_weights.max().item()),
                        'shape': list(channel_weights.shape)
                    }
            
            # Spatial attention weights
            if hasattr(layer.cbam.spatial_attention, 'last_attention_weights'):
                spatial_weights = layer.cbam.spatial_attention.last_attention_weights
                if spatial_weights is not None:
                    attention_info['spatial_weights'] = {
                        'mean': float(spatial_weights.mean().item()),
                        'std': float(spatial_weights.std().item()),
                        'min': float(spatial_weights.min().item()),
                        'max': float(spatial_weights.max().item()),
                        'shape': list(spatial_weights.shape)
                    }
        
        return attention_info
    
    def extract_attention_weights_direct(self, attention_module):
        """Extract attention weights directly from a CBAMChannelAttention module"""
        attention_info = {}
        
        if hasattr(attention_module, 'last_attention_weights') and attention_module.last_attention_weights is not None:
            channel_weights = attention_module.last_attention_weights
            attention_info['channel_weights'] = {
                'mean': float(channel_weights.mean().item()),
                'std': float(channel_weights.std().item()),
                'min': float(channel_weights.min().item()),
                'max': float(channel_weights.max().item()),
                'shape': list(channel_weights.shape)
            }
        
        return attention_info
    
    def get_cbam_attention_summary(self):
        """Get summary of CBAM attention weights across all CNN layers"""
        if not self.attention_weights_history:
            return "No CBAM attention data available"
        
        # Use the latest attention weights
        latest_attention = self.attention_weights_history[-1]
        summary = {}
        
        for layer_name, attention_data in latest_attention.items():
            layer_summary = {}
            
            if 'channel_weights' in attention_data:
                ch_weights = attention_data['channel_weights']
                layer_summary['channel'] = f"μ={ch_weights['mean']:.3f}, σ={ch_weights['std']:.3f}, range=[{ch_weights['min']:.3f}, {ch_weights['max']:.3f}]"
            
            if 'spatial_weights' in attention_data:
                sp_weights = attention_data['spatial_weights']
                layer_summary['spatial'] = f"μ={sp_weights['mean']:.3f}, σ={sp_weights['std']:.3f}, range=[{sp_weights['min']:.3f}, {sp_weights['max']:.3f}]"
            
            summary[layer_name] = layer_summary
        
        return summary
    
    def get_learned_feature_importance(self):
        """Get current learned feature importance weights"""
        # Check the dedicated input attention module first
        if hasattr(self, 'input_attention') and hasattr(self.input_attention, 'feature_importance'):
            module = self.input_attention
            # Check for either apply_importance or importance_regularization
            if (hasattr(module, 'apply_importance') and module.apply_importance) or \
               (hasattr(module, 'importance_regularization') and module.importance_regularization):
                # Apply same transformation as in forward pass
                raw_weights = module.feature_importance.detach().cpu()
                # Use temperature of 2.0 to match forward pass transformation
                temperature = 2.0 if hasattr(module, 'importance_regularization') else 0.5
                normalized_weights = torch.softmax(raw_weights / temperature, dim=0)
                
                # Use feature names passed to model - must be provided
                if self.feature_names is not None:
                    feature_names = self.feature_names[:len(normalized_weights)]
                else:
                    raise ValueError("Feature names not provided to model. Ensure feature_names are passed during model initialization.")
                
                importance_dict = {}
                for i, (name, weight) in enumerate(zip(feature_names, normalized_weights)):
                    importance_dict[f'{name}'] = {
                        'raw_weight': float(raw_weights[i]),
                        'normalized_weight': float(weight),
                        'rank': int((normalized_weights >= weight).sum())
                    }
                
                return importance_dict
        
        # Fallback: find any CBAMChannelAttention module with learnable feature importance
        for name, module in self.named_modules():
            if isinstance(module, CBAMChannelAttention) and hasattr(module, 'feature_importance'):
                # Check for either apply_importance or importance_regularization
                if (hasattr(module, 'apply_importance') and module.apply_importance) or \
                   (hasattr(module, 'importance_regularization') and module.importance_regularization):
                    # Apply same transformation as in forward pass
                    raw_weights = module.feature_importance.detach().cpu()
                    # Use temperature of 2.0 to match forward pass transformation
                    temperature = 2.0 if hasattr(module, 'importance_regularization') else 0.5
                    normalized_weights = torch.softmax(raw_weights / temperature, dim=0)
                    
                    # Use feature names passed to model, with fallbacks
                    if self.feature_names is not None:
                        feature_names = self.feature_names[:len(normalized_weights)]
                    else:
                        raise ValueError("Feature names not provided to model. Ensure feature_names are passed during model initialization.")
                    
                    importance_dict = {}
                    for i, (name, weight) in enumerate(zip(feature_names, normalized_weights)):
                        importance_dict[f'{name}'] = {
                            'raw_weight': float(raw_weights[i]),
                            'normalized_weight': float(weight),
                            'rank': int((normalized_weights >= weight).sum())
                        }
                    
                    return importance_dict
        
        return {}
    
    def get_model_complexity_analysis(self):
        """Get detailed model complexity analysis including transformer components"""
        analysis = {}
        
        # Parameter count by component
        conv_params = 0
        attention_params = 0  # CBAM attention
        transformer_params = 0  # Vision Transformer
        classifier_params = 0
        bn_params = 0
        
        for name, param in self.named_parameters():
            param_count = param.numel()
            
            if 'vision_transformer' in name.lower():
                transformer_params += param_count
            elif 'conv' in name.lower():
                conv_params += param_count
            elif 'cbam' in name.lower() or 'attention' in name.lower():
                attention_params += param_count
            elif 'classifier' in name.lower() or 'fc' in name.lower():
                classifier_params += param_count
            elif 'bn' in name.lower() or 'norm' in name.lower():
                bn_params += param_count
        
        total_params = conv_params + attention_params + transformer_params + classifier_params + bn_params
        
        analysis['parameter_breakdown'] = {
            'total': total_params,
            'convolutional': conv_params,
            'cbam_attention': attention_params,
            'vision_transformer': transformer_params,
            'classifier': classifier_params,
            'batch_norm': bn_params
        }
        
        analysis['parameter_percentages'] = {
            'convolutional': (conv_params / total_params * 100) if total_params > 0 else 0,
            'cbam_attention': (attention_params / total_params * 100) if total_params > 0 else 0,
            'vision_transformer': (transformer_params / total_params * 100) if total_params > 0 else 0,
            'classifier': (classifier_params / total_params * 100) if total_params > 0 else 0,
            'batch_norm': (bn_params / total_params * 100) if total_params > 0 else 0
        }
        
        # Memory analysis (rough estimate)
        analysis['memory_estimate'] = {
            'parameters_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'approx_forward_pass_mb': total_params * 8 / (1024 * 1024)  # Rough estimate including activations
        }
        
        # Architecture information
        analysis['architecture_info'] = {
            'use_transformer': self.use_transformer,
            'transformer_dim': self.transformer_dim if self.use_transformer else 0,
            'transformer_heads': self.transformer_heads if self.use_transformer else 0,
            'transformer_layers': self.transformer_layers if self.use_transformer else 0
        }
        
        return analysis
    
    def print_model_analysis(self):
        """Print comprehensive model analysis including transformer components"""
        analysis = self.get_model_complexity_analysis()
        
        print(f"\n🔬 HYBRID CNN-TRANSFORMER MODEL ANALYSIS")
        print(f"{'='*60}")
        
        # Architecture info
        arch_info = analysis['architecture_info']
        print(f"🏗️ Architecture Configuration:")
        print(f"  Hybrid CNN-Transformer: {arch_info['use_transformer']}")
        if arch_info['use_transformer']:
            print(f"  Transformer Dimension: {arch_info['transformer_dim']}")
            print(f"  Attention Heads: {arch_info['transformer_heads']}")
            print(f"  Transformer Layers: {arch_info['transformer_layers']}")
        
        # Parameter breakdown
        params = analysis['parameter_breakdown']
        percentages = analysis['parameter_percentages']
        
        print(f"\n📊 Parameter Distribution:")
        print(f"  Total parameters: {params['total']:,}")
        print(f"  Convolutional:     {params['convolutional']:,} ({percentages['convolutional']:.1f}%)")
        print(f"  CBAM Attention:    {params['cbam_attention']:,} ({percentages['cbam_attention']:.1f}%)")
        if arch_info['use_transformer']:
            print(f"  Vision Transformer: {params['vision_transformer']:,} ({percentages['vision_transformer']:.1f}%)")
        print(f"  Classifier:        {params['classifier']:,} ({percentages['classifier']:.1f}%)")
        print(f"  Batch Norm:        {params['batch_norm']:,} ({percentages['batch_norm']:.1f}%)")
        
        # Memory estimates
        memory = analysis['memory_estimate']
        print(f"\n💾 Memory Estimates:")
        print(f"  Parameters: {memory['parameters_mb']:.2f} MB")
        print(f"  Forward pass (approx): {memory['approx_forward_pass_mb']:.2f} MB")
        
        # Model architecture insights
        print(f"\n🏗️  Architecture Insights:")
        total_attention = percentages['cbam_attention'] + percentages.get('vision_transformer', 0)
        if total_attention > 15:
            print(f"  ⚠️ High attention overhead ({total_attention:.1f}%) - monitor training efficiency")
        elif total_attention > 8:
            print(f"  ✓ Moderate attention usage ({total_attention:.1f}%)")
        else:
            print(f"  ⚡ Efficient attention usage ({total_attention:.1f}%)")
        
        if arch_info['use_transformer']:
            if percentages['vision_transformer'] > 20:
                print(f"  🧠 Transformer-heavy hybrid ({percentages['vision_transformer']:.1f}%) - good for global modeling")
            else:
                print(f"  ⚖️  Balanced CNN-Transformer hybrid ({percentages['vision_transformer']:.1f}%)")
        
        if percentages['classifier'] > 40:
            print(f"  ⚠️  Classifier-heavy model ({percentages['classifier']:.1f}%) - may overfit")
        elif percentages['convolutional'] > 50:
            print(f"  ✓ Feature extraction focused ({percentages['convolutional']:.1f}%)")
        
        print(f"{'='*60}")
    
    
    def get_transformer_attention_analysis(self):
        """Analyze transformer attention patterns for molecular interaction insights"""
        if not self.use_transformer or not hasattr(self.vision_transformer, 'attention_maps'):
            return "Transformer not enabled or no attention data available"
        
        if not self.vision_transformer.attention_maps:
            return "No transformer attention data collected yet"
        
        # Get the latest attention maps
        attention_maps = self.vision_transformer.attention_maps
        analysis = {}
        
        for layer_idx, attention_map in enumerate(attention_maps):
            # attention_map shape: (batch_size, num_heads, seq_len, seq_len)
            # Average across batch dimension for analysis
            avg_attention = attention_map.mean(dim=0)  # (num_heads, seq_len, seq_len)
            
            # Analysis across all heads
            head_analysis = {}
            for head_idx in range(avg_attention.size(0)):
                head_attention = avg_attention[head_idx]  # (seq_len, seq_len)
                
                # For 2x2x2 = 8 patches, center patches are indices 3, 4 (approximately)
                center_patches = [3, 4]  # Adjust based on your specific patch arrangement
                center_attention = head_attention[center_patches, :].mean(dim=0)
                
                head_analysis[f'head_{head_idx}'] = {
                    'center_focus': float(center_attention[center_patches].mean().item()),
                    'attention_spread': float(head_attention.std().item()),
                    'max_attention': float(head_attention.max().item()),
                    'uniformity': float(1.0 / (head_attention.std().item() + 1e-8))  # Higher = more uniform
                }
            
            analysis[f'transformer_layer_{layer_idx+1}'] = {
                'heads': head_analysis,
                'layer_summary': {
                    'avg_center_focus': np.mean([h['center_focus'] for h in head_analysis.values()]),
                    'avg_attention_spread': np.mean([h['attention_spread'] for h in head_analysis.values()]),
                    'head_diversity': np.std([h['attention_spread'] for h in head_analysis.values()])
                }
            }
        
        return analysis
    
    def get_attention_summary(self):
        """Get attention summary (compatibility method for train_3d_cnn.py)"""
        # Always use CBAM attention analysis for comprehensive summary
        if not self.attention_weights_history:
            return "No attention data available"
        
        # Use the latest attention weights
        latest_attention = self.attention_weights_history[-1]
        summary = {}
        
        for layer_name, attention_data in latest_attention.items():
            layer_summary = {}
            
            if 'channel_weights' in attention_data:
                ch_weights = attention_data['channel_weights']
                layer_summary['channel'] = f"μ={ch_weights['mean']:.3f}, σ={ch_weights['std']:.3f}, range=[{ch_weights['min']:.3f}, {ch_weights['max']:.3f}]"
            
            if 'spatial_weights' in attention_data:
                sp_weights = attention_data['spatial_weights']
                layer_summary['spatial'] = f"μ={sp_weights['mean']:.3f}, σ={sp_weights['std']:.3f}, range=[{sp_weights['min']:.3f}, {sp_weights['max']:.3f}]"
            
            # Handle transformer self-attention data
            if 'self_attention_strength' in attention_data:
                layer_summary['self_attention_strength'] = attention_data['self_attention_strength']
            
            if 'spatial_focus_variance' in attention_data:
                layer_summary['spatial_focus_variance'] = attention_data['spatial_focus_variance']
            
            if 'attention_entropy' in attention_data:
                layer_summary['attention_entropy'] = attention_data['attention_entropy']
            
            if 'feature_diversity' in attention_data:
                layer_summary['feature_diversity'] = attention_data['feature_diversity']
            
            # Handle Vision Transformer data (from VisionTransformer3D.get_attention_summary)
            if 'layer_1' in attention_data:
                vt_data = attention_data['layer_1']
                if 'mean_attention' in vt_data:
                    vision_transformer_strength = vt_data['mean_attention']
                    layer_summary['vision_transformer_detailed'] = f"global_strength={vision_transformer_strength:.3f}"
                    
                    # Add full transformer summary if entropy is available
                    if 'attention_entropy' in vt_data:
                        vision_transformer_entropy = vt_data['attention_entropy']
                        layer_summary['vision_transformer'] = f"mean={vision_transformer_strength:.3f}, entropy={vision_transformer_entropy:.2f}"
            
            # Only add to summary if there's any data
            if layer_summary:
                summary[layer_name] = layer_summary
        
        return summary
    
    def extract_self_attention_stats(self, self_attention_layer=None):
        """Extract monitoring statistics from LightSelfAttention3D layers"""
        stats = {}
        
        if self_attention_layer is not None:
            # Extract stats from specific layer
            if isinstance(self_attention_layer, LightSelfAttention3D):
                if hasattr(self_attention_layer, 'attention_statistics'):
                    layer_stats = self_attention_layer.attention_statistics.copy()
                    
                    # Add attention weights analysis if available
                    if hasattr(self_attention_layer, 'last_attention_weights') and self_attention_layer.last_attention_weights is not None:
                        attention_weights = self_attention_layer.last_attention_weights
                        layer_stats.update({
                            'attention_weights_mean': float(attention_weights.mean().item()),
                            'attention_weights_std': float(attention_weights.std().item()),
                            'attention_weights_max': float(attention_weights.max().item()),
                            'attention_weights_min': float(attention_weights.min().item())
                        })
                    
                    return layer_stats
            return {}
        
        # Find all LightSelfAttention3D layers and collect their statistics
        for name, module in self.named_modules():
            if isinstance(module, LightSelfAttention3D):
                if hasattr(module, 'attention_statistics'):
                    layer_stats = module.attention_statistics.copy()
                    
                    # Add attention weights analysis if available
                    if hasattr(module, 'last_attention_weights') and module.last_attention_weights is not None:
                        attention_weights = module.last_attention_weights
                        layer_stats.update({
                            'attention_weights_mean': float(attention_weights.mean().item()),
                            'attention_weights_std': float(attention_weights.std().item()),
                            'attention_weights_max': float(attention_weights.max().item()),
                            'attention_weights_min': float(attention_weights.min().item())
                        })
                    
                    stats[name] = layer_stats
        
        return stats
    
    def analyze_self_attention_patterns(self, self_attention_layer):
        """Detailed analysis of self-attention patterns for training insights"""
        if not hasattr(self_attention_layer, 'last_attention_weights') or self_attention_layer.last_attention_weights is None:
            return {'status': 'No attention weights available'}
        
        attention_weights = self_attention_layer.last_attention_weights  # (batch, heads, spatial, spatial)
        
        analysis = {}
        
        # Spatial attention patterns
        diagonal_attention = torch.diagonal(attention_weights, dim1=-2, dim2=-1).mean().item()
        off_diagonal_attention = (attention_weights.sum() - torch.diagonal(attention_weights, dim1=-2, dim2=-1).sum()) / (attention_weights.numel() - attention_weights.size(-1))
        
        analysis.update({
            'self_attention_ratio': float(diagonal_attention),
            'cross_spatial_attention': float(off_diagonal_attention.item()),
            'attention_locality': float(diagonal_attention / (diagonal_attention + off_diagonal_attention.item() + 1e-8)),
            'pattern_type': 'local' if diagonal_attention > off_diagonal_attention else 'global'
        })
        
        # Head diversity analysis
        if attention_weights.size(1) > 1:  # Multi-head
            head_similarities = []
            for i in range(attention_weights.size(1)):
                for j in range(i+1, attention_weights.size(1)):
                    similarity = F.cosine_similarity(
                        attention_weights[:, i].flatten(), 
                        attention_weights[:, j].flatten(), 
                        dim=0
                    ).item()
                    head_similarities.append(similarity)
            
            analysis['head_diversity'] = {
                'mean_similarity': float(np.mean(head_similarities)),
                'std_similarity': float(np.std(head_similarities)),
                'diversity_score': float(1.0 - np.mean(head_similarities))  # Higher = more diverse
            }
        
        return analysis
    
    def get_cross_attention_stats(self):
        """
        Unified cross-attention statistics - ALL FIELDS, EVERY TIME, SIMPLE!
        Returns ALL cross-attention metrics in one place.
        """
        # Collect from active cross-attention layers
        cross_attention_layers = []
        if hasattr(self, 'vision_transformer') and self.vision_transformer is not None:
            for module in self.vision_transformer.modules():
                if hasattr(module, 'cross_attention') and hasattr(module.cross_attention, 'attention_statistics'):
                    cross_attention_layers.append(module.cross_attention)
        
        if not cross_attention_layers:
            return {
                'fusion_layers_count': 0,
                'average_fusion_strength': 0.0,
                'fusion_diversity': 0.0,
                'feature_enhancement_ratio': 0.0,
                'average_attention_entropy': 0.0,
                'average_attention_focus': 0.0,
                'entropy_consistency': 0.0,
                'overall_fusion_quality': 0.0,
                'fusion_assessment': 'no_data',
                'note': 'No cross-attention layers found'
            }
        
        # Collect all values
        fusion_strengths = []
        utilization_values = []
        entropy_values = []
        focus_values = []
        
        for layer in cross_attention_layers:
            stats = layer.attention_statistics
            if 'cross_fusion_strength' in stats:
                fusion_strengths.append(stats['cross_fusion_strength'])
            if 'low_level_utilization' in stats:
                utilization_values.append(stats['low_level_utilization'])
            if 'attention_entropy' in stats:
                entropy_values.append(stats['attention_entropy'])
            if 'attention_focus' in stats:
                focus_values.append(stats['attention_focus'])
        
        # Calculate ALL metrics in one place
        fusion_layers_count = len(cross_attention_layers)
        average_fusion_strength = float(np.mean(fusion_strengths)) if fusion_strengths else 0.0
        fusion_diversity = float(np.std(fusion_strengths)) if len(fusion_strengths) > 1 else 0.0
        feature_enhancement_ratio = float(np.mean(utilization_values)) if utilization_values else 0.0
        average_attention_entropy = float(np.mean(entropy_values)) if entropy_values else 0.0
        average_attention_focus = float(np.mean(focus_values)) if focus_values else 0.0
        entropy_consistency = float(np.std(entropy_values)) if len(entropy_values) > 1 else 0.0
        
        # Overall quality assessment
        if fusion_strengths and utilization_values:
            overall_fusion_quality = float((average_fusion_strength + feature_enhancement_ratio) / 2.0)
        else:
            overall_fusion_quality = 0.0
        
        # Simple quality assessment
        if overall_fusion_quality > 0.8:
            fusion_assessment = 'excellent'
        elif overall_fusion_quality > 0.6:
            fusion_assessment = 'good'
        elif overall_fusion_quality > 0.4:
            fusion_assessment = 'moderate'
        elif overall_fusion_quality > 0.0:
            fusion_assessment = 'poor'
        else:
            fusion_assessment = 'no_data'
        
        return {
            'fusion_layers_count': fusion_layers_count,
            'average_fusion_strength': average_fusion_strength,
            'fusion_diversity': fusion_diversity,
            'feature_enhancement_ratio': feature_enhancement_ratio,
            'average_attention_entropy': average_attention_entropy,
            'average_attention_focus': average_attention_focus,
            'entropy_consistency': entropy_consistency,
            'overall_fusion_quality': overall_fusion_quality,
            'fusion_assessment': fusion_assessment,
            'note': f"{fusion_layers_count} cross-attention layers active"
        }

