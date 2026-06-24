"""
model_3d_cnn_2_4.py
This file contains the definitions for the 3D-CNN model with CBAM attention mechanism 
enhanced with Vision Transformer (ViT) for global feature modeling, adapted for type_2 format.

Format specifications:
- 28 input features (14 atomic features × 2 groups: adsorbate + solvent)
- Separated channel groups for better feature interpretation
- Default input shape: (batch_size, 28, 20, 20, 20)

The hybrid CNN-Transformer architecture combines:
- CNN backbone with CBAM attention for local feature extraction
- Vision Transformer head for global dependency modeling

Key optimizations in v2.4:
- Removed CNN self-attention layers (LightSelfAttention3D) based on training analysis
- Simplified architecture: CBAM (local) + Vision Transformer (global)
- Improved training efficiency and reduced parameter count

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math


class PositionalEncoding3D(nn.Module):
    """
    Simplified 3D Positional Encoding for 3x3x3 spatial patches.
    Designed for molecular voxel structure: 27 spatial positions in 3D grid.
    Let the model learn optimal position representations without complex scaling.
    """
    def __init__(self, d_model, max_len=27, dropout=0.1):
        super(PositionalEncoding3D, self).__init__()
        self.d_model = d_model
        
        # 🔥 SIMPLIFIED: Direct learnable position embeddings for 27 spatial patches
        # Each of the 27 positions (3x3x3 cube) gets its own embedding vector
        # No complex scaling - let the model learn what it needs
        self.pos_embedding = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        
        # Simple dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model) - flattened spatial patches (seq_len=27 for 3x3x3)
        Returns:
            x + position encoding
        """
        seq_len = x.size(1)
        
        # Direct position encoding - no learnable scaling
        # The 27 positions correspond to 3x3x3 spatial arrangement: better resolution for molecular interactions
        pos_enc = self.pos_embedding[:seq_len, :].unsqueeze(0)  # (1, seq_len, d_model)
        
        return self.dropout(x + pos_enc)


class MultiHeadAttention3D(nn.Module):
    """
    Enhanced Multi-Head Self-Attention with integrated relative position encoding.
    Optimized for molecular feature space patches after CNN processing.
    """
    def __init__(self, d_model, num_heads, dropout=0.1, max_len=8, use_relative_pos=True):
        super(MultiHeadAttention3D, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_len = max_len
        self.use_relative_pos = use_relative_pos
        
        # Linear projections for Q, K, V with improved initialization
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Relative position bias table for 27 spatial patches (3x3x3)
        if use_relative_pos:
            # Simple relative position bias - let model learn spatial relationships
            self.relative_bias_table = nn.Parameter(torch.zeros(num_heads, max_len, max_len))
            nn.init.trunc_normal_(self.relative_bias_table, std=0.02)  # Conservative initialization
        
        self.dropout = nn.Dropout(dropout)
        # Standard scaling
        self.scale = math.sqrt(self.d_k)
        
        # 🔥 CRITICAL FIX: Fixed temperature to prevent learning dispersed attention
        # Model was learning to increase temperature from 0.05 to 0.4448, causing dispersed attention
        # Use fixed temperature to force focused attention patterns
        self.temperature = 0.03  # Fixed value, not learnable parameter
        
        # Standard weight initialization
        self._init_weights()
        
        # Store attention weights for analysis
        self.last_attention_weights = None
    
    def _init_weights(self):
        """Standard weight initialization"""
        # Standard initialization for stable training
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
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
        
        # Content-based attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, heads, seq, seq)
        
        # Add relative position bias (learnable spatial relationships for feature patches)
        if self.use_relative_pos and seq_len <= self.max_len:
            relative_bias = self.relative_bias_table[:, :seq_len, :seq_len]  # (heads, seq, seq)
            scores = scores + relative_bias.unsqueeze(0)  # Broadcast over batch dimension
        
        # Apply fixed temperature for focused attention (no learnable temperature)
        scores = scores / self.temperature
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Store attention weights for analysis (detached from computation graph)
        self.last_attention_weights = attention_weights.detach()
        
        # Store relative position bias for monitoring (if enabled)
        if self.use_relative_pos and hasattr(self, 'relative_bias_table'):
            self.last_relative_bias = relative_bias.detach() if 'relative_bias' in locals() else None
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # (batch, heads, seq, d_k)
        
        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.w_o(context)
        
        return output

    def get_relative_position_stats(self):
        """Get statistics about relative position encoding and attention temperature."""
        stats = {}
        
        # 🔥 NEW: Temperature monitoring for attention focus analysis
        stats['temperature'] = {
            'current_value': float(self.temperature),  # Now it's a fixed value, not a parameter
            'status': 'focused' if self.temperature < 0.1 else 'dispersed' if self.temperature > 0.2 else 'moderate'
        }
        
        if not self.use_relative_pos or not hasattr(self, 'relative_bias_table'):
            stats['relative_position'] = {'enabled': False}
            return stats
        
        bias_table = self.relative_bias_table.detach()
        stats['relative_position'] = {
            'enabled': True,
            'table_shape': list(bias_table.shape),
            'mean_bias': float(bias_table.mean().item()),
            'std_bias': float(bias_table.std().item()),
            'max_bias': float(bias_table.max().item()),
            'min_bias': float(bias_table.min().item()),
            'head_diversity': float(bias_table.std(dim=0).mean().item()),  # How different heads learn different patterns
        }
        
        # Analyze relative position patterns
        if hasattr(self, 'last_relative_bias') and self.last_relative_bias is not None:
            rel_bias = self.last_relative_bias
            stats['relative_position'].update({
                'active_bias_mean': float(rel_bias.mean().item()),
                'active_bias_std': float(rel_bias.std().item()),
                'bias_contribution': 'active'  # Currently being used
            })
        else:
            stats['relative_position']['bias_contribution'] = 'inactive'  # Not currently active
        
        return stats


class CrossAttention3D(nn.Module):
    """
    Optimized Cross-Attention for adsorbate-solvent interaction energy prediction.
    Focus on effective multi-level feature fusion for molecular interactions.
    """
    def __init__(self, d_model, num_heads, dropout=0.1, enable_adsorbate_solvent_interaction=True):
        super(CrossAttention3D, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.enable_adsorbate_solvent_interaction = enable_adsorbate_solvent_interaction
        
        # Enhanced projections with better initialization for stronger attention
        self.w_q = nn.Linear(d_model, d_model, bias=True)  # Add bias for better expressivity
        self.w_k = nn.Linear(d_model, d_model, bias=True)
        self.w_v = nn.Linear(d_model, d_model, bias=True)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Molecular interaction enhancement
        if enable_adsorbate_solvent_interaction:
            # Learnable interaction bias for H-bond and van der Waals modeling
            self.interaction_bias = nn.Parameter(torch.zeros(num_heads, 27, 27))  # 3x3x3 spatial positions
            # Interaction strength scaling
            self.interaction_scale = nn.Parameter(torch.ones(1) * 0.1)
        
        self.dropout = nn.Dropout(dropout)
        # Reduced scale for stronger attention patterns
        self.scale = math.sqrt(self.d_k) * 0.7  # Stronger attention than standard
        
        # Improved feature projection with residual connection
        self.low_level_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),  # ReLU for stability (was SiLU)
            nn.Linear(d_model, d_model)
        )
        
        # Fusion gate to control cross-attention contribution
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # Initialize weights for stronger cross-attention
        self._init_cross_attention_weights()
        
        # Enhanced monitoring for cross-attention
        self.last_attention_weights = None
        self.attention_statistics = {
            'cross_fusion_strength': 0.0,
            'low_level_utilization': 0.0,
            'attention_entropy': 0.0,
            'attention_focus': 0.0
        }
    
    def _init_cross_attention_weights(self):
        """Initialize weights for stronger cross-attention patterns"""
        # Stronger initialization for better cross-attention
        for module in [self.w_q, self.w_k, self.w_v]:
            nn.init.xavier_normal_(module.weight, gain=1.6)  # Higher gain for stronger attention
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)  # Small positive bias
        
        nn.init.xavier_normal_(self.w_o.weight, gain=0.5)  # Conservative output projection
    def forward(self, query, key_value, feature_channels=None):
        """
        Optimized forward pass with enhanced adsorbate-solvent interaction modeling.
        
        Args:
            query: (batch_size, seq_len, d_model) - high-level features
            key_value: (batch_size, seq_len, d_model) - low-level features  
            feature_channels: Optional feature channel information
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = query.size()
        
        # Enhanced low-level feature projection with residual connection
        original_key_value = key_value
        key_value = self.low_level_proj(key_value) + original_key_value  # Residual connection
        
        # Multi-head attention computation
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key_value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(key_value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Enhanced attention scores with molecular interaction bias
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Add learnable molecular interaction bias for H-bond and vdW modeling
        if self.enable_adsorbate_solvent_interaction and hasattr(self, 'interaction_bias'):
            # Apply interaction bias (models distance-dependent interactions)
            if scores.size(-1) == 27:  # 3x3x3 spatial arrangement
                interaction_bias = self.interaction_bias * self.interaction_scale
                scores = scores + interaction_bias.unsqueeze(0)  # Broadcast over batch
        
        # Apply attention
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Gated fusion: learn how much cross-attention to use
        query_context_concat = torch.cat([query, context], dim=-1)
        fusion_gate = self.fusion_gate(query_context_concat)
        
        # Apply gated fusion
        fused_output = query + fusion_gate * context  # Residual with gated cross-attention
        
        # Final output projection
        output = self.w_o(fused_output)
        
        # Store attention weights for monitoring
        self.last_attention_weights = attention_weights.detach()
        
        # Enhanced attention statistics calculation
        with torch.no_grad():
            # Cross-fusion strength (how much cross-attention is being used)
            fusion_strength = fusion_gate.mean().item()
            
            # Attention diversity and focus
            flat_attention = attention_weights.view(-1, attention_weights.size(-1))
            attention_entropy = -torch.sum(flat_attention * torch.log(flat_attention + 1e-8), dim=-1).mean().item()
            attention_focus = attention_weights.max().item()
            
            # Low-level feature utilization (how much the low-level features contribute)
            context_magnitude = torch.norm(context, dim=-1).mean().item()
            query_magnitude = torch.norm(query, dim=-1).mean().item()
            low_level_utilization = context_magnitude / (query_magnitude + 1e-8)
            
            # Update statistics
            self.attention_statistics.update({
                'cross_fusion_strength': fusion_strength,
                'low_level_utilization': low_level_utilization,
                'attention_entropy': attention_entropy,
                'attention_focus': attention_focus
            })
        
        return output
    def get_cross_attention_stats(self):
        """Get cross-attention statistics for monitoring"""
        return self.attention_statistics.copy()


class TransformerEncoderLayer3D(nn.Module):
    """
    Simplified Transformer Encoder Layer for molecular interaction modeling.
    Clean design - let the model learn what interactions matter.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_cross_attention=False):
        super(TransformerEncoderLayer3D, self).__init__()
        
        self.use_cross_attention = use_cross_attention
        self.self_attention = MultiHeadAttention3D(d_model, num_heads, dropout, max_len=27, use_relative_pos=True)
        
        # Add cross-attention if enabled
        if use_cross_attention:
            self.cross_attention = CrossAttention3D(d_model, num_heads, dropout)
            self.norm_cross = nn.LayerNorm(d_model)
        
        # 🔥 SIMPLIFIED: Standard feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization (pre-norm architecture)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, low_level_features=None, feature_channels=None):
        """
        Simplified forward pass for molecular interaction modeling.
        
        Args:
            x: (batch_size, seq_len, d_model) - high-level features (8 spatial patches)
            low_level_features: (batch_size, seq_len, d_model) - optional low-level features for cross-attention
            feature_channels: Optional feature channel information
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # Pre-norm self-attention with residual connection
        norm_x = self.norm1(x)
        attention_output = self.self_attention(norm_x)
        x = x + self.dropout(attention_output)
        
        # Cross-attention for multi-level feature fusion (if enabled)
        if self.use_cross_attention and low_level_features is not None:
            norm_x_cross = self.norm_cross(x)
            cross_output = self.cross_attention(norm_x_cross, low_level_features, feature_channels)
            x = x + self.dropout(cross_output)
        
        # 🔥 SIMPLIFIED: Standard feed-forward with residual
        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + ff_output
        
        return x


class VisionTransformer3D(nn.Module):
    """
    3D Vision Transformer optimized for molecular interaction energy prediction.
    Enhanced for adsorbate-solvent interactions with better cross-attention fusion.
    """
    def __init__(self, input_dim, d_model=256, num_heads=4, num_layers=2, dropout=0.1, use_cross_attention=True):
        super(VisionTransformer3D, self).__init__()
        
        self.input_dim = input_dim  # Input channels from CNN (e.g., 96)
        self.d_model = d_model
        self.num_heads = num_heads  # Reduced from 8 to 4 for better head diversity
        self.num_layers = num_layers
        self.use_cross_attention = use_cross_attention
        
        # 🔥 SIMPLIFIED: Clean input projection - let model learn what it needs
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # Enhanced low-level feature projection with better capacity
        if use_cross_attention:
            self.low_level_projection = None  # Will be created dynamically
        
        # 3D Positional encoding for 27 spatial patches (3x3x3)
        self.pos_encoding = PositionalEncoding3D(d_model, max_len=27, dropout=dropout)
        
        # Simplified transformer encoder layers for molecular interaction modeling
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer3D(
                d_model, 
                num_heads, 
                d_model * 2,  # Reduced FFN size for simplicity
                dropout,
                use_cross_attention=(use_cross_attention and i > 0)  # Cross-attention in later layers only
            )
            for i in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # 🔥 SIMPLIFIED: Basic learnable pooling weights - let model decide
        self.pooling_weights = nn.Parameter(torch.tensor([0.5, 0.5, 0.0]))  # avg, max, unused
        
        # 🔥 SIMPLIFIED: Remove complex interaction enhancement - let attention handle it
        # Simple residual enhancement
        self.feature_enhancement = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Store attention information for analysis
        self.attention_maps = []
        self.cross_attention_maps = []
    
    def forward(self, x, low_level_features=None, return_attention=False, feature_channels=None):
        """
        Optimized forward pass with enhanced cross-attention for molecular interaction modeling.
        
        Args:
            x: (batch_size, channels, D, H, W) - output from CNN backbone (high-level)
            low_level_features: (batch_size, channels, D, H, W) - optional low-level CNN features
            return_attention: bool - whether to return attention weights
            feature_channels: dict - mapping of feature channels
        Returns:
            output: (batch_size, d_model) - global feature representation
        """
        batch_size, channels, D, H, W = x.shape
        
        # Flatten spatial dimensions to create sequence of patches
        # (batch_size, channels, D*H*W) -> (batch_size, D*H*W, channels)
        x = x.view(batch_size, channels, -1).transpose(1, 2)
        seq_len = x.size(1)  # Should be D*H*W (e.g., 3*3*3 = 27)
        
        # Enhanced input projection with residual connection
        x_projected = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x_projected)
        
        # Process low-level features if provided for cross-attention
        low_level_projected = None
        if self.use_cross_attention and low_level_features is not None:
            # Create low-level projection dynamically if needed
            low_level_channels = low_level_features.size(1)
            if self.low_level_projection is None:
                self.low_level_projection = nn.Sequential(
                    nn.Linear(low_level_channels, self.d_model),
                    nn.LayerNorm(self.d_model),
                    nn.ReLU(inplace=True),  # ReLU for stability (was SiLU)
                    nn.Linear(self.d_model, self.d_model)
                ).to(x.device)
            
            # Ensure low_level_features match the spatial dimensions
            if low_level_features.shape[-3:] != (D, H, W):
                # Adaptive pooling to match dimensions
                low_level_features = F.adaptive_avg_pool3d(low_level_features, (D, H, W))
            
            # Project low-level features with enhanced processing
            low_level_flat = low_level_features.view(batch_size, low_level_channels, -1).transpose(1, 2)
            low_level_projected = self.low_level_projection(low_level_flat)
            low_level_projected = self.pos_encoding(low_level_projected)
        
        # Clear previous attention maps
        self.attention_maps = []
        self.cross_attention_maps = []
        
        # Apply transformer layers with progressive cross-attention
        for i, layer in enumerate(self.transformer_layers):
            if i == 0:
                # First layer: self-attention only for feature refinement
                x = layer(x)
            else:
                # Later layers: use cross-attention for multi-level fusion
                if self.use_cross_attention and low_level_projected is not None:
                    x = layer(x, low_level_projected, feature_channels)
                else:
                    x = layer(x)
            
            # Store attention weights for analysis
            if hasattr(layer.self_attention, 'last_attention_weights') and layer.self_attention.last_attention_weights is not None:
                self.attention_maps.append(layer.self_attention.last_attention_weights.clone())
            
            # Store cross-attention weights if available
            if (hasattr(layer, 'cross_attention') and 
                hasattr(layer.cross_attention, 'last_attention_weights') and 
                layer.cross_attention.last_attention_weights is not None):
                self.cross_attention_maps.append(layer.cross_attention.last_attention_weights.clone())
        
        # Final normalization
        x = self.final_norm(x)  # (batch_size, seq_len, d_model)
        
        # 🔥 SIMPLIFIED: Basic feature enhancement with residual
        x_enhanced = x + self.feature_enhancement(x)
        
        # 🔥 SIMPLIFIED: Standard global pooling
        avg_pooled = torch.mean(x_enhanced, dim=1)  # (batch_size, d_model)
        max_pooled, _ = torch.max(x_enhanced, dim=1)  # (batch_size, d_model)
        
        # Learnable weighted pooling - model learns avg vs max preference
        global_features = (self.pooling_weights[0] * avg_pooled + 
                          self.pooling_weights[1] * max_pooled)
        
        if return_attention:
            return global_features, self.attention_maps, self.cross_attention_maps
        return global_features
    
    def get_attention_summary(self):
        """Get comprehensive summary of attention patterns for optimization guidance."""
        if not self.attention_maps:
            return "No attention data available"
        
        summary = {}
        pooling_analysis = {}
        
        # 🔥 NEW: Comprehensive attention analysis for each layer
        for i, attention_map in enumerate(self.attention_maps):
            # attention_map shape: (batch_size, num_heads, seq_len, seq_len)
            batch_size, num_heads, seq_len, seq_len = attention_map.shape
            
            # Average across batch for stable statistics
            avg_attention = attention_map.mean(dim=0)  # (num_heads, seq_len, seq_len)
            
            # Head diversity analysis (critical for optimization)
            head_similarities = []
            for h1 in range(num_heads):
                for h2 in range(h1+1, num_heads):
                    similarity = F.cosine_similarity(
                        avg_attention[h1].flatten(), 
                        avg_attention[h2].flatten(), 
                        dim=0
                    ).item()
                    head_similarities.append(similarity)
            
            head_diversity = 1.0 - (sum(head_similarities) / len(head_similarities) if head_similarities else 0)
            
            # Global attention statistics
            global_avg_attention = avg_attention.mean(dim=0)  # (seq_len, seq_len)
            
            # Spatial pattern analysis for molecular interactions
            diagonal_strength = torch.diag(global_avg_attention).mean().item()
            off_diagonal_strength = (global_avg_attention.sum() - torch.diag(global_avg_attention).sum()).item() / (seq_len * (seq_len - 1))
            
            # Attention entropy (diversity measure)
            attention_entropy = float(-torch.sum(global_avg_attention * torch.log(global_avg_attention + 1e-8)).item())
            
            # 🔧 FIXED: Corrected saturation detection for attention matrices
            # For seq_len=27, uniform attention = 1/27 ≈ 0.037
            # Strong attention should be significantly above uniform (2-3x)
            uniform_threshold = 1.0 / seq_len  # 0.037 for seq_len=27
            strong_attention_threshold = uniform_threshold * 2.5  # ~0.093 for seq_len=27
            very_weak_threshold = uniform_threshold * 0.5  # ~0.019 for seq_len=27
            
            # Saturation: percentage of attention weights significantly above uniform
            high_attention_ratio = (global_avg_attention > strong_attention_threshold).float().mean().item()
            # Dead attention: percentage of attention weights significantly below uniform  
            low_attention_ratio = (global_avg_attention < very_weak_threshold).float().mean().item()
            
            summary[f'layer_{i+1}'] = {
                'mean_attention': float(global_avg_attention.mean().item()),
                'max_attention': float(global_avg_attention.max().item()),
                'min_attention': float(global_avg_attention.min().item()),
                'attention_entropy': attention_entropy,
                'diagonal_attention': diagonal_strength,
                'cross_attention': off_diagonal_strength,
                'head_diversity': head_diversity,
                'saturation_ratio': high_attention_ratio,
                'dead_attention_ratio': low_attention_ratio,
                'attention_balance': diagonal_strength / (off_diagonal_strength + 1e-8),
                'relative_pos_enabled': True,
                'attention_type': 'relative_position_encoding',
                # 🔧 DEBUG: Add threshold info for verification
                'uniform_threshold': uniform_threshold,
                'strong_threshold': strong_attention_threshold,
                'weak_threshold': very_weak_threshold,
                'attention_range': f"[{global_avg_attention.min().item():.4f}, {global_avg_attention.max().item():.4f}]",
                # 🎯 Optimization indicators  
                'needs_temperature_increase': high_attention_ratio > 0.4,  # Adjusted for corrected threshold
                'needs_diversity_boost': head_diversity < 0.3,
                'attention_health': 'healthy' if 0.05 < high_attention_ratio < 0.3 and head_diversity > 0.3 else 'needs_attention'
            }
        
        # 🔥 SIMPLIFIED: Pooling weights analysis for avg vs max preference
        if hasattr(self, 'pooling_weights'):
            weights = self.pooling_weights.detach()
            total_weight = weights[0] + weights[1]  # Normalize since we only use first 2 weights
            
            # Ensure we don't divide by zero and handle edge cases
            if total_weight.item() < 1e-8:
                # If weights are essentially zero, set to default balanced ratios
                avg_ratio = torch.tensor(0.5)
                max_ratio = torch.tensor(0.5)
                actual_strategy = 'weights_collapsed'  # Indicate this is a problem
            else:
                avg_ratio = weights[0] / total_weight
                max_ratio = weights[1] / total_weight
                actual_strategy = 'avg_dominant' if avg_ratio > 0.6 else 'max_dominant' if max_ratio > 0.6 else 'balanced'
            
            pooling_analysis = {
                'avg_pooling_weight': float(weights[0].item()),
                'max_pooling_weight': float(weights[1].item()),
                'total_pooling_weight': float(total_weight.item()),  # Add for debugging
                'avg_ratio': float(avg_ratio.item()),
                'max_ratio': float(max_ratio.item()),
                'avg_dominance': float(avg_ratio.item()),  # Add for compatibility
                'max_dominance': float(max_ratio.item()),  # Add for compatibility
                'learned_strategy': actual_strategy
            }
        
        # 🔥 NEW: Cross-attention analysis (if available)
        cross_attention_analysis = {}
        if self.cross_attention_maps:
            for i, cross_attn in enumerate(self.cross_attention_maps):
                cross_avg = cross_attn.mean(dim=(0, 1))  # Average across batch and heads
                cross_attention_analysis[f'cross_layer_{i+1}'] = {
                    'cross_attention_strength': float(cross_avg.mean().item()),
                    'cross_attention_focus': float(cross_avg.max().item()),
                    'cross_entropy': float(-torch.sum(cross_avg * torch.log(cross_avg + 1e-8)).item())
                }
        
        return {
            'attention_layers': summary,
            'pooling_analysis': pooling_analysis,
            'cross_attention': cross_attention_analysis,
            'overall_health': self._assess_transformer_health(summary)
        }
    
    def _assess_transformer_health(self, layer_summary):
        """Assess overall transformer health and provide optimization guidance."""
        if not layer_summary:
            return {'status': 'no_data', 'recommendations': []}
        
        issues = []
        recommendations = []
        
        # Check for common optimization issues
        avg_head_diversity = sum(layer['head_diversity'] for layer in layer_summary.values()) / len(layer_summary)
        avg_saturation = sum(layer['saturation_ratio'] for layer in layer_summary.values()) / len(layer_summary)
        avg_entropy = sum(layer['attention_entropy'] for layer in layer_summary.values()) / len(layer_summary)
        
        # Head diversity issues
        if avg_head_diversity < 0.3:
            issues.append('low_head_diversity')
            recommendations.append('Consider reducing model capacity or increasing dropout')
        
        # Saturation issues  
        if avg_saturation > 0.4:  # Adjusted threshold for corrected calculation
            issues.append('high_saturation')
            recommendations.append('Increase temperature scaling or use softer activations')
        elif avg_saturation < 0.02:  # Adjusted threshold for corrected calculation
            issues.append('weak_attention')
            recommendations.append('Decrease temperature scaling or check input preprocessing')
        
        # Entropy issues
        if avg_entropy < 1.0:
            issues.append('low_attention_diversity')
            recommendations.append('Add attention dropout or reduce model constraints')
        elif avg_entropy > 3.0:
            issues.append('attention_too_dispersed')
            recommendations.append('Increase attention focus with temperature reduction')
        
        status = 'healthy' if not issues else 'needs_optimization'
        
        return {
            'status': status,
            'issues': issues,
            'recommendations': recommendations,
            'metrics': {
                'head_diversity': avg_head_diversity,
                'saturation_level': avg_saturation,
                'attention_entropy': avg_entropy
            }
        }


class CBAMChannelAttention(nn.Module):
    """
    Enhanced CBAM Channel Attention Module with group interaction capability.
    Supports adsorbate-solvent group interaction for molecular interaction modeling.
    """
    def __init__(self, in_channels, reduction_ratio=16, dropout=0.1, enable_group_interaction=False, group_split=None):
        super(CBAMChannelAttention, self).__init__()
        
        # Group interaction settings
        self.enable_group_interaction = enable_group_interaction
        self.group_split = group_split  # e.g., [24, 48] for adsorbate+solvent channels
        
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
        
        # Group-specific MLPs for interaction modeling (if enabled)
        if self.enable_group_interaction and self.group_split is not None:
            assert len(self.group_split) == 2, "Currently supports binary group split (adsorbate + solvent)"
            ads_channels, solv_channels = self.group_split
            assert ads_channels + solv_channels == in_channels, "Group split must sum to total channels"
            
            # Separate MLPs for each group
            ads_reduced = max(ads_channels // reduction_ratio, 1)
            solv_reduced = max(solv_channels // reduction_ratio, 1)
            
            self.ads_mlp = nn.Sequential(
                nn.Linear(ads_channels, ads_reduced, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(ads_reduced, ads_channels, bias=False)
            )
            
            self.solv_mlp = nn.Sequential(
                nn.Linear(solv_channels, solv_reduced, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(solv_reduced, solv_channels, bias=False)
            )
            
            # Cross-group interaction weight
            self.interaction_strength = nn.Parameter(torch.tensor(0.5))  # Learnable interaction strength
        
        # Enhanced initialization for better channel diversity
        for module in self.shared_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.3)
        
        if self.enable_group_interaction and self.group_split is not None:
            for module in self.ads_mlp:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight, gain=1.2)
            for module in self.solv_mlp:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight, gain=1.2)
        
        self.activation = nn.Sigmoid()
        
        # Enhanced monitoring for training analysis
        self.last_attention_weights = None
        self.group_interaction_stats = {
            'ads_attention_mean': 0.0,
            'solv_attention_mean': 0.0,
            'interaction_strength_value': 0.0,
            'cross_group_correlation': 0.0
        }
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
        
        if self.enable_group_interaction and self.group_split is not None:
            # Group-aware processing with adsorbate-solvent interaction
            ads_channels, solv_channels = self.group_split
            
            # Split features into adsorbate and solvent groups
            avg_ads = avg_out[:, :ads_channels]  # Adsorbate channels
            avg_solv = avg_out[:, ads_channels:] # Solvent channels
            max_ads = max_out[:, :ads_channels]
            max_solv = max_out[:, ads_channels:]
            
            # Process each group separately
            ads_avg_weights = self.ads_mlp(avg_ads)
            ads_max_weights = self.ads_mlp(max_ads)
            solv_avg_weights = self.solv_mlp(avg_solv)
            solv_max_weights = self.solv_mlp(max_solv)
            
            # Combine avg and max for each group
            ads_weights = self.activation(ads_avg_weights + ads_max_weights)
            solv_weights = self.activation(solv_avg_weights + solv_max_weights)
            
            # 🎯 Cross-group interaction: Adsorbate "guides" solvent attention
            # Model the physical intuition that adsorbate influences solvent arrangement
            interaction_factor = torch.sigmoid(self.interaction_strength)
            
            # Method 1: Multiplicative interaction (adsorbate gates solvent)
            ads_influence = ads_weights.mean(dim=1, keepdim=True)  # Global adsorbate signal
            enhanced_solv_weights = solv_weights * (1 + interaction_factor * ads_influence)
            
            # Method 2: Add residual connection to preserve individual group information
            final_ads_weights = ads_weights
            final_solv_weights = enhanced_solv_weights * interaction_factor + solv_weights * (1 - interaction_factor)
            
            # Combine group attentions
            channel_weights = torch.cat([final_ads_weights, final_solv_weights], dim=1)
            
            # Update group interaction statistics
            if self.training:
                with torch.no_grad():
                    self.group_interaction_stats.update({
                        'ads_attention_mean': final_ads_weights.mean().item(),
                        'solv_attention_mean': final_solv_weights.mean().item(),
                        'interaction_strength_value': interaction_factor.item(),
                        'cross_group_correlation': torch.corrcoef(torch.stack([
                            final_ads_weights.mean(dim=1), final_solv_weights.mean(dim=1)
                        ]))[0, 1].item() if batch_size > 1 else 0.0
                    })
        else:
            # Standard CBAM processing
            avg_weights = self.shared_mlp(avg_out)
            max_weights = self.shared_mlp(max_out)
            channel_weights = self.activation(avg_weights + max_weights)
        
        # Reshape for broadcasting
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
    Optimized for adsorbate-solvent interaction modeling with H-bond focus.
    Kernel_size=5 optimized for 0.8Å resolution to capture H-bond distances (2-3Å).
    """
    def __init__(self, in_channels, reduction_ratio=6, kernel_size=5, dropout=0.1, 
                 enable_group_interaction=False, group_split=None):
        super(CBAM3D, self).__init__()
        self.channel_attention = CBAMChannelAttention(
            in_channels, reduction_ratio, dropout, 
            enable_group_interaction=enable_group_interaction, 
            group_split=group_split
        )
        # Optimized spatial attention for molecular interactions (H-bond: ~3-4 voxels at 0.8Å)
        self.spatial_attention = CBAMSpatialAttention(kernel_size)
        
        # Store group interaction settings for analysis
        self.enable_group_interaction = enable_group_interaction
        self.group_split = group_split

    def forward(self, x):
        # Apply channel attention first (feature importance), then spatial attention (location importance)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
    
    def get_group_interaction_stats(self):
        """Get group interaction statistics for monitoring"""
        if self.enable_group_interaction and hasattr(self.channel_attention, 'group_interaction_stats'):
            return self.channel_attention.group_interaction_stats
        return {}

class ResidualBlock3D(nn.Module):
    """
    Enhanced 3D Residual Block with:
    - Dual-path design for molecular interactions
    - CBAM attention for feature importance weighting
    - Improved gradient flow
    """
    def __init__(self, in_channels, out_channels, stride=1, use_cbam=True):
        super(ResidualBlock3D, self).__init__()
        
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
        
        # CBAM attention optimized for molecular interactions (H-bond: ~3-4 voxels at 0.8Å)
        self.cbam = CBAM3D(out_channels, kernel_size=5, dropout=0.12) if use_cbam else nn.Identity()
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        
        # 🔥 STABILITY FIX: Use ReLU for complete BatchNorm stabilization
        self.dropout = nn.Dropout3d(p=0.02)  # Reduced based on log analysis
        self.activation = nn.ReLU(inplace=True)  # ReLU for BatchNorm stability (was SiLU)
    
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
        
        # Add shortcut and apply SiLU (no self-attention)
        out += self.shortcut(residual)
        out = self.activation(out)
        
        return out

class AttentionCNNTransformer_2_4(nn.Module):
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
                 dropout_rate=0.35,     # Increased from 0.25 based on overfitting analysis
                 use_transformer=True,
                 transformer_dim=200,  # Kept optimal size
                 transformer_heads=4,  # Reduced from 8 to 4 for better head diversity
                 transformer_layers=2,
                 feature_names=None):  # Add feature_names parameter
        
        super(AttentionCNNTransformer_2_4, self).__init__()
        
        # Store configuration including input channels for dynamic feature analysis
        self.in_channels = in_channels  # Store input channel count for type_2 compatibility
        self.use_transformer = use_transformer
        self.transformer_dim = transformer_dim
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        
        # Store feature names for analysis methods
        self.feature_names = feature_names
        
        # Enhanced channel group processing for Type_2 format (28 channels = 14 adsorbate + 14 solvent)
        assert in_channels == 28, "Type_2 format requires exactly 28 channels (14 adsorbate + 14 solvent)"
        
        # Adsorbate branch: Focus on static structural features (smaller kernels, reduced capacity)
        # 🔥 STABILITY FIX: Use ReLU to stabilize BatchNorm variance (from SiLU)
        # Adsorbate occupies central small volume, mostly zeros, so use enhanced processing
        self.adsorbate_processor = nn.Sequential(
            nn.Conv3d(14, 16, kernel_size=3, padding=1, bias=False),  # Reduced from 24 to 16
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),  # ReLU for BatchNorm stability (was SiLU)
            # Add 1x1 conv to enhance feature density and reduce sparsity
            nn.Conv3d(16, 20, kernel_size=1, bias=False),  # Feature densification
            nn.BatchNorm3d(20),
            nn.ReLU(inplace=True),  # Keep ReLU for consistency
            nn.Conv3d(20, 24, kernel_size=3, padding=1, bias=False),  # Final expansion
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),  # ReLU for final stability (was SiLU)
            nn.Dropout3d(0.08)  # Slightly increased dropout
        )
        
        # Solvent branch: Focus on dynamic density patterns (optimized capacity)
        # 🔥 STABILITY FIX: Use ReLU to stabilize BatchNorm variance (from SiLU)
        # Solvent fills most space and is more important for interaction energy
        self.solvent_processor = nn.Sequential(
            nn.Conv3d(14, 32, kernel_size=5, padding=2, bias=False),  # Reduced from 48 to 32
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),  # ReLU for BatchNorm stability (was SiLU)
            nn.Conv3d(32, 48, kernel_size=5, padding=2, bias=False),  # Reduced from 64 to 48, still 2x adsorbate
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True),  # ReLU for BatchNorm stability (was SiLU)
            nn.Dropout3d(0.05)
        )
        
        # Interaction fusion layer: Model adsorbate-solvent interactions with enhanced attention
        # 🔥 STABILITY FIX: Use ReLU to stabilize BatchNorm variance
        self.interaction_conv = nn.Sequential(
            nn.Conv3d(72, 48, kernel_size=3, padding=1, bias=False),  # 24+48=72 -> 48, reduced complexity
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True),  # ReLU for BatchNorm stability (was SiLU)
            nn.Conv3d(48, 32, kernel_size=1, bias=False),  # Further reduced to 32 for efficiency
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)  # ReLU for BatchNorm stability (was SiLU)
        )
        
        # Enhanced CBAM for adsorbate-solvent interaction weighting with group interaction
        self.interaction_attention = CBAM3D(
            32, reduction_ratio=4, kernel_size=5, dropout=0.15,  # Increased dropout for regularization
            enable_group_interaction=True, 
            group_split=[16, 16]  # Balanced split for 32 channels after interaction_conv
        )
        
        # No manual branch weighting - let the architecture naturally express importance:
        # - Solvent processor: 48 channels (2x capacity) + larger kernels (5x5x5)
        # - Adsorbate processor: 24 channels + smaller kernels (3x3x3)
        # This design naturally emphasizes solvent features without artificial weights
        
        # Monitoring attributes for branch analysis
        self.branch_statistics = {
            'adsorbate_magnitude': 0.0,
            'solvent_magnitude': 0.0,
            'magnitude_ratio': 0.0,  # Add missing field
            'interaction_strength': 0.0,
            'branch_balance': 'architecture_based',  # No manual weighting
            'sparsity_ratio': 0.0,
            'adsorbate_sparsity': 0.0,  # Add missing field
            'solvent_sparsity': 0.0     # Add missing field
        }
        
        # Enhanced channel progression adapted for dual-branch output (32 channels from interaction_conv)
        self.layer1 = ResidualBlock3D(32, 48, stride=1, use_cbam=True)
        self.pool1 = nn.AvgPool3d(2)
        
        self.layer2 = ResidualBlock3D(48, 64, stride=1, use_cbam=True)
        self.pool2 = nn.AvgPool3d(2)
        
        self.layer3 = ResidualBlock3D(64, 80, stride=1, use_cbam=True)
        
        # Final feature layer
        self.layer4 = ResidualBlock3D(80, 96, stride=1, use_cbam=True)
        
        # Improved spatial information preservation with 3x3x3 pooling
        self.adaptive_pool = nn.AdaptiveAvgPool3d(3)  # 3x3x3 spatial preservation (27 patches)
        
        # Enhanced Vision Transformer for global feature modeling (updated input_dim)
        if self.use_transformer:
            self.vision_transformer = VisionTransformer3D(
                input_dim=96,  # Updated from 128 to 96 due to reduced architecture
                d_model=transformer_dim,
                num_heads=transformer_heads,
                num_layers=transformer_layers,
                dropout=dropout_rate * 0.75,
                use_cross_attention=True  # Enable cross-attention for multi-level fusion
            )
            classifier_input_dim = transformer_dim
        else:
            self.vision_transformer = None
            classifier_input_dim = 96 * 3 * 3 * 3  # Updated flattened features = 2592
        
        # Enhanced classifier with SiLU activation and better regularization
        if self.use_transformer:
            # 🔥 OPTIMIZATION: Improved classifier for transformer features with anti-saturation measures
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, 128),
                nn.BatchNorm1d(128),
                nn.GELU(),  # Changed from SiLU to GELU for gentler activation
                nn.Dropout(dropout_rate * 1.0),  # Increased dropout for better regularization
                
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.GELU(),  # Gentler activation
                nn.Dropout(dropout_rate * 0.8),
                
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),  # Use ReLU for final layers to prevent negative saturation
                nn.Dropout(dropout_rate * 0.6),
                
                nn.Linear(32, 1)
            )
        else:
            # Enhanced classifier for CNN features with SiLU (updated input dim)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(classifier_input_dim, 512),  # Increased capacity due to larger feature space (2592)
                nn.BatchNorm1d(512),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(dropout_rate),
                
                nn.Linear(512, 128),             # Intermediate layer
                nn.BatchNorm1d(128),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(dropout_rate * 0.7),
                
                nn.Linear(128, 32),              # Final hidden layer
                nn.BatchNorm1d(32),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(dropout_rate * 0.5),
                
                nn.Linear(32, 1)               # Output layer
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
        """Enhanced weight initialization for better training stability and improved gradient flow"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # More conservative initialization to prevent activation explosion
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu', a=0.05)  # Reduced from 0.1 to 0.05
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                # 🔥 STABILITY FIX: Enhanced BatchNorm stabilization to prevent variance explosion
                # Log shows variance explosion (interact.1: var=733.594, cnn_layer1.bn2: var=832.203)
                nn.init.constant_(m.weight, 0.9)  # Even more conservative (was 0.95)
                nn.init.constant_(m.bias, 0)
                m.momentum = 0.001  # Much slower adaptation (was 0.005)
                m.eps = 5e-3  # Larger eps for better stability (was 1e-3)
                # Initialize running_var conservatively
                if hasattr(m, 'running_var'):
                    m.running_var.fill_(1.0)  # Initialize running_var to 1
                    # NOTE: Let ReLU stabilize BN naturally first
            elif isinstance(m, nn.Linear):
                # More conservative linear layer initialization
                if m.weight.size(0) > 1000:  # Large linear layers (classifier)
                    nn.init.normal_(m.weight, 0, 0.01)  # Very conservative for classifier
                else:
                    nn.init.xavier_normal_(m.weight, gain=0.5)  # Conservative gain
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
        
        # 🔬 Enhanced input feature analysis with dual-branch processing
        if monitor_activations or detailed_analysis:
            # Analyze input features before any processing
            monitoring_data['feature_analysis']['input_features'] = self.analyze_input_features(x)
            
            # 🎯 Special analysis for type_2 format (28 channels with separated groups)
            if x.size(1) == 28:
                monitoring_data['feature_analysis']['type2_channel_groups'] = self.analyze_type2_channel_groups(x)
            
        # 🏗️ Dual-branch processing for Type_2 format
        # Separate adsorbate and solvent channel groups
        adsorbate_channels = x[:, :14, :, :, :]   # First 14 channels: adsorbate features
        solvent_channels = x[:, 14:, :, :, :]     # Last 14 channels: solvent features
        
        # Process adsorbate branch (sparse, central features)
        adsorbate_features = self.adsorbate_processor(adsorbate_channels)  # (batch, 24, 20, 20, 20)
        if monitor_activations:
            monitoring_data['activation_stats'].update(self.analyze_activations(adsorbate_features, "adsorbate_branch"))
        
        # Apply adsorbate mask to focus on non-zero regions (efficiency optimization)
        # Create mask from adsorbate channels (assuming first channel contains mol_type info or non-zero indicates adsorbate presence)
        adsorbate_mask = (adsorbate_channels.sum(dim=1, keepdim=True) != 0).float()  # (batch, 1, 20, 20, 20)
        adsorbate_features = adsorbate_features * adsorbate_mask  # Apply spatial mask to focus computation
        
        # Process solvent branch (dense, distributed features)
        solvent_features = self.solvent_processor(solvent_channels)       # (batch, 48, 20, 20, 20)
        if monitor_activations:
            monitoring_data['activation_stats'].update(self.analyze_activations(solvent_features, "solvent_branch"))
        
        # Natural combination without artificial weighting
        # The architecture naturally emphasizes solvent (48 channels) over adsorbate (24 channels)
        combined_features = torch.cat([adsorbate_features, solvent_features], dim=1)  # (batch, 72, 20, 20, 20)
        if monitor_activations:
            monitoring_data['activation_stats'].update(self.analyze_activations(combined_features, "combined_features"))
        
        # Model adsorbate-solvent interactions
        interaction_features = self.interaction_conv(combined_features)   # (batch, 32, 20, 20, 20)
        interaction_features = self.interaction_attention(interaction_features)
        if monitor_activations:
            monitoring_data['activation_stats'].update(self.analyze_activations(interaction_features, "interaction_features"))
        
        # Update branch monitoring statistics
        if monitor_activations or detailed_analysis:
            self._update_branch_statistics(adsorbate_features, solvent_features, interaction_features)
            monitoring_data['feature_analysis']['branch_analysis'] = self.get_branch_analysis()
            
        # Continue with the enhanced CNN backbone
        x = interaction_features  # Start main CNN pipeline with interaction features
        
        # Progressive feature learning with enhanced monitoring
        x = self.layer1(x)  # (batch, 32, 20, 20, 20) -> (batch, 48, 20, 20, 20)
        layer1_features = x.clone() if self.use_transformer else None  # Store for cross-attention
        if monitor_activations:
            monitoring_data['activation_stats'].update(self.analyze_activations(x, "layer1"))
            monitoring_data['attention_info']['layer1'] = self.extract_attention_weights(self.layer1)
        if detailed_analysis:
            monitoring_data['layer_contributions']['layer1'] = self.calculate_layer_contribution(x)
            monitoring_data['training_insights']['layer1_attention_strength'] = self.analyze_attention_strength(self.layer1)
        x = self.pool1(x)  # (batch, 48, 10, 10, 10)
        
        x = self.layer2(x)  # (batch, 48, 10, 10, 10) -> (batch, 64, 10, 10, 10)
        layer2_features = x.clone() if self.use_transformer else None  # Store for cross-attention
        if monitor_activations:
            monitoring_data['activation_stats'].update(self.analyze_activations(x, "layer2"))
            monitoring_data['attention_info']['layer2'] = self.extract_attention_weights(self.layer2)
        if detailed_analysis:
            monitoring_data['layer_contributions']['layer2'] = self.calculate_layer_contribution(x)
            monitoring_data['training_insights']['layer2_attention_strength'] = self.analyze_attention_strength(self.layer2)
        x = self.pool2(x)  # (batch, 64, 5, 5, 5)
        
        x = self.layer3(x)  # (batch, 64, 5, 5, 5) -> (batch, 80, 5, 5, 5)
        if monitor_activations:
            monitoring_data['activation_stats'].update(self.analyze_activations(x, "layer3"))
            monitoring_data['attention_info']['layer3'] = self.extract_attention_weights(self.layer3)
        if detailed_analysis:
            monitoring_data['layer_contributions']['layer3'] = self.calculate_layer_contribution(x)
            monitoring_data['training_insights']['layer3_attention_strength'] = self.analyze_attention_strength(self.layer3)
        
        # Additional layer for better feature abstraction
        x = self.layer4(x)  # (batch, 80, 5, 5, 5) -> (batch, 96, 5, 5, 5)
        if monitor_activations:
            monitoring_data['activation_stats'].update(self.analyze_activations(x, "layer4"))
            monitoring_data['attention_info']['layer4'] = self.extract_attention_weights(self.layer4)
        if detailed_analysis:
            monitoring_data['layer_contributions']['layer4'] = self.calculate_layer_contribution(x)
            monitoring_data['training_insights']['layer4_attention_strength'] = self.analyze_attention_strength(self.layer4)
        
        # Adaptive pooling to 3x3x3 for better feature preservation
        x = self.adaptive_pool(x)  # (batch, 96, 3, 3, 3)
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
            x = x.view(x.size(0), -1)  # (batch, 96*3*3*3 = 2592)
        
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
        """Get L2 norms of weights for each layer with clear component naming"""
        weight_norms = {}
        
        # Dual-branch weight norms
        for name, param in self.named_parameters():
            if param.requires_grad and 'weight' in name:
                # Create more descriptive names for better understanding
                if 'adsorbate_processor' in name:
                    layer_num = name.split('.')[1]
                    weight_norms[f"adsorbate_branch.{layer_num}"] = param.data.norm(2).item()
                elif 'solvent_processor' in name:
                    layer_num = name.split('.')[1]
                    weight_norms[f"solvent_branch.{layer_num}"] = param.data.norm(2).item()
                elif 'interaction_conv' in name:
                    layer_num = name.split('.')[1]
                    weight_norms[f"interaction_fusion.{layer_num}"] = param.data.norm(2).item()
                elif 'interaction_attention' in name:
                    weight_norms[f"interaction_attention.cbam"] = param.data.norm(2).item()
                elif 'layer1' in name:
                    weight_norms[f"cnn_backbone.layer1"] = param.data.norm(2).item()
                elif 'layer2' in name:
                    weight_norms[f"cnn_backbone.layer2"] = param.data.norm(2).item()
                elif 'layer3' in name:
                    weight_norms[f"cnn_backbone.layer3"] = param.data.norm(2).item()
                elif 'layer4' in name:
                    weight_norms[f"cnn_backbone.layer4"] = param.data.norm(2).item()
                elif 'vision_transformer' in name:
                    weight_norms[f"transformer.component"] = param.data.norm(2).item()
                elif 'classifier' in name:
                    layer_idx = name.split('.')[1]
                    weight_norms[f"classifier.layer_{layer_idx}"] = param.data.norm(2).item()
                else:
                    # Fallback for any other weights
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
        """Enhanced activation analysis with correct statistics"""
        stats = {}
        
        # Convert to numpy for analysis
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
        
        # Ensure we have valid data
        if x_np.size == 0:
            return {f'{layer_name_prefix}_mean': 0.0, f'{layer_name_prefix}_std': 0.0}
        
        # Basic statistics with proper handling
        stats[f'{layer_name_prefix}_mean'] = float(np.mean(x_np))
        stats[f'{layer_name_prefix}_std'] = float(np.std(x_np))  # This should not be 0 unless all values are identical
        stats[f'{layer_name_prefix}_min'] = float(np.min(x_np))
        stats[f'{layer_name_prefix}_max'] = float(np.max(x_np))
        stats[f'{layer_name_prefix}_zeros_pct'] = float(np.mean(x_np == 0) * 100)
        
        # Additional detailed statistics for better monitoring
        stats[f'{layer_name_prefix}_abs_mean'] = float(np.mean(np.abs(x_np)))
        stats[f'{layer_name_prefix}_abs_std'] = float(np.std(np.abs(x_np)))
        stats[f'{layer_name_prefix}_negative_pct'] = float(np.mean(x_np < 0) * 100)
        stats[f'{layer_name_prefix}_saturation_pct'] = float(np.mean(np.abs(x_np) > 0.9) * 100)
        
        # Channel-wise analysis for multi-channel tensors
        if len(x_np.shape) >= 2:
            # Flatten all dimensions except batch and channel
            if len(x_np.shape) == 5:  # (batch, channel, D, H, W)
                channel_means = np.mean(x_np, axis=(0, 2, 3, 4))  # Average over batch and spatial dims
            elif len(x_np.shape) == 4:  # (batch, channel, H, W) 
                channel_means = np.mean(x_np, axis=(0, 2, 3))  # Average over batch and spatial dims
            elif len(x_np.shape) == 3:  # (batch, seq_len, features) for transformer
                channel_means = np.mean(x_np, axis=(0, 1))  # Average over batch and sequence
            else:
                channel_means = np.mean(x_np, axis=0)  # Just average over batch
                
            if len(channel_means) > 1:  # Multi-channel
                stats[f'{layer_name_prefix}_channel_diversity'] = float(np.std(channel_means))
            else:
                stats[f'{layer_name_prefix}_channel_diversity'] = 0.0
        
        return stats
        
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
    
    def _update_branch_statistics(self, adsorbate_features, solvent_features, interaction_features):
        """Update statistics for dual-branch processing monitoring"""
        with torch.no_grad():
            # Calculate branch magnitudes
            adsorbate_magnitude = torch.mean(torch.abs(adsorbate_features)).item()
            solvent_magnitude = torch.mean(torch.abs(solvent_features)).item()
            interaction_strength = torch.mean(torch.abs(interaction_features)).item()
            
            # Calculate sparsity (percentage of near-zero values)
            adsorbate_sparsity = (torch.abs(adsorbate_features) < 0.01).float().mean().item()
            solvent_sparsity = (torch.abs(solvent_features) < 0.01).float().mean().item()
            
            # Update statistics
            self.branch_statistics.update({
                'adsorbate_magnitude': adsorbate_magnitude,
                'solvent_magnitude': solvent_magnitude,
                'interaction_strength': interaction_strength,
                'adsorbate_sparsity': adsorbate_sparsity,
                'solvent_sparsity': solvent_sparsity,
                'sparsity_ratio': adsorbate_sparsity / (solvent_sparsity + 1e-8),
                'magnitude_ratio': adsorbate_magnitude / (solvent_magnitude + 1e-8),
                'capacity_ratio': 48 / 24,  # Solvent vs adsorbate channel capacity (updated to 48/24 = 2.0)
                'kernel_size_ratio': 5 / 3,  # Solvent vs adsorbate kernel size
                'branch_balance': 'adsorbate_dominant' if adsorbate_magnitude > solvent_magnitude * 1.5 
                                else 'solvent_dominant' if solvent_magnitude > adsorbate_magnitude * 1.5 
                                else 'balanced'
            })
    
    def get_branch_analysis(self):
        """Get comprehensive analysis of dual-branch processing"""
        return {
            'current_statistics': self.branch_statistics.copy(),
            'architecture_insights': {
                'adsorbate_processor_params': sum(p.numel() for p in self.adsorbate_processor.parameters()),
                'solvent_processor_params': sum(p.numel() for p in self.solvent_processor.parameters()),
                'interaction_conv_params': sum(p.numel() for p in self.interaction_conv.parameters()),
                'parameter_ratio': sum(p.numel() for p in self.solvent_processor.parameters()) / 
                                 sum(p.numel() for p in self.adsorbate_processor.parameters()),
                'architecture_design': {
                    'adsorbate_capacity': '24 channels, 3x3x3 kernels',  # Updated from 32 to 24
                    'solvent_capacity': '48 channels, 5x5x5 kernels',    # Updated from 64 to 48
                    'natural_emphasis': 'solvent (2x channels + larger receptive field)'
                }
            },
            'recommendations': self._get_branch_recommendations()
        }
    
    def _get_branch_recommendations(self):
        """Generate recommendations based on branch statistics"""
        recommendations = []
        
        # Handle case where no forward pass has occurred yet
        if self.branch_statistics.get('magnitude_ratio', 0) == 0.0:
            recommendations.append("No forward pass data available yet - run model with data to get recommendations")
            return recommendations
        
        # Check sparsity ratio
        if self.branch_statistics.get('sparsity_ratio', 0) > 5.0:
            recommendations.append("High adsorbate sparsity detected - consider reducing adsorbate processor capacity")
        
        # Check magnitude balance
        magnitude_ratio = self.branch_statistics.get('magnitude_ratio', 0)
        if magnitude_ratio < 0.1:
            recommendations.append("Very low adsorbate magnitude - consider increasing adsorbate branch weight")
        elif magnitude_ratio > 2.0:
            recommendations.append("High adsorbate magnitude - consider increasing solvent branch weight")
        
        # Check interaction strength
        if self.branch_statistics.get('interaction_strength', 0) < 0.01:
            recommendations.append("Weak interaction strength - consider adjusting interaction_conv architecture")
        
        # Check branch weight balance
        if 'capacity_ratio' in self.branch_statistics:
            capacity_ratio = self.branch_statistics.get('capacity_ratio', 0)
            if capacity_ratio < 1.5:
                recommendations.append("Consider increasing solvent processor capacity relative to adsorbate")
            elif capacity_ratio > 3.0:
                recommendations.append("Solvent processor may be over-parameterized relative to adsorbate")
        
        if not recommendations:
            recommendations.append("Branch processing appears well-balanced")
        
        return recommendations
    
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
        
        # Parameter count by component (updated for dual-branch architecture)
        conv_params = 0
        attention_params = 0  # CBAM attention
        transformer_params = 0  # Vision Transformer
        classifier_params = 0
        bn_params = 0
        branch_params = 0  # New: dual-branch processing parameters
        
        for name, param in self.named_parameters():
            param_count = param.numel()
            
            if any(branch in name for branch in ['adsorbate_processor', 'solvent_processor', 'interaction_conv', 'interaction_attention']):
                branch_params += param_count
            elif 'vision_transformer' in name.lower():
                transformer_params += param_count
            elif 'conv' in name.lower():
                conv_params += param_count
            elif 'cbam' in name.lower() or 'attention' in name.lower():
                attention_params += param_count
            elif 'classifier' in name.lower() or 'fc' in name.lower():
                classifier_params += param_count
            elif 'bn' in name.lower() or 'norm' in name.lower():
                bn_params += param_count
        
        total_params = conv_params + attention_params + transformer_params + classifier_params + bn_params + branch_params
        
        analysis['parameter_breakdown'] = {
            'total': total_params,
            'dual_branch_processing': branch_params,  # New category
            'convolutional': conv_params,
            'cbam_attention': attention_params,
            'vision_transformer': transformer_params,
            'classifier': classifier_params,
            'batch_norm': bn_params
        }
        
        analysis['parameter_percentages'] = {
            'dual_branch_processing': (branch_params / total_params * 100) if total_params > 0 else 0,
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
                
                # For 3x3x3 = 27 patches, center patches are indices 3, 4 (approximately)
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
                    
                    # Add relative position encoding information
                    if 'relative_pos_enabled' in vt_data and vt_data['relative_pos_enabled']:
                        layer_summary['vision_transformer_detailed'] += f", rel_pos=enabled"
                    
                    # Add full transformer summary if entropy is available
                    if 'attention_entropy' in vt_data:
                        vision_transformer_entropy = vt_data['attention_entropy']
                        layer_summary['vision_transformer'] = f"mean={vision_transformer_strength:.3f}, entropy={vision_transformer_entropy:.2f}"
            
            # Only add to summary if there's any data
            if layer_summary:
                summary[layer_name] = layer_summary
        
        return summary
    
    def get_cross_attention_stats(self):
        """
        Enhanced cross-attention statistics with better monitoring for molecular interactions.
        Returns ALL cross-attention metrics in one place.
        """
        # Collect from active cross-attention layers in transformer
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
                'fusion_assessment': 'no_cross_attention',
                'note': 'No cross-attention layers found or activated'
            }
        
        # Collect all values from active cross-attention layers
        fusion_strengths = []
        utilization_values = []
        entropy_values = []
        focus_values = []
        
        for layer in cross_attention_layers:
            stats = layer.attention_statistics
            if 'cross_fusion_strength' in stats and stats['cross_fusion_strength'] > 0:
                fusion_strengths.append(stats['cross_fusion_strength'])
            if 'low_level_utilization' in stats and stats['low_level_utilization'] > 0:
                utilization_values.append(stats['low_level_utilization'])
            if 'attention_entropy' in stats and stats['attention_entropy'] > 0:
                entropy_values.append(stats['attention_entropy'])
            if 'attention_focus' in stats and stats['attention_focus'] > 0:
                focus_values.append(stats['attention_focus'])
        
        # Calculate enhanced metrics
        fusion_layers_count = len(cross_attention_layers)
        average_fusion_strength = float(np.mean(fusion_strengths)) if fusion_strengths else 0.0
        fusion_diversity = float(np.std(fusion_strengths)) if len(fusion_strengths) > 1 else 0.0
        feature_enhancement_ratio = float(np.mean(utilization_values)) if utilization_values else 0.0
        average_attention_entropy = float(np.mean(entropy_values)) if entropy_values else 0.0
        average_attention_focus = float(np.mean(focus_values)) if focus_values else 0.0
        entropy_consistency = float(np.std(entropy_values)) if len(entropy_values) > 1 else 0.0
        
        # Enhanced quality assessment for molecular interactions
        if fusion_strengths and utilization_values:
            # Weigh fusion strength more heavily for molecular interaction modeling
            overall_fusion_quality = float((average_fusion_strength * 0.7 + feature_enhancement_ratio * 0.3))
        else:
            overall_fusion_quality = 0.0
        
        # More nuanced quality assessment
        if overall_fusion_quality > 0.6:
            fusion_assessment = 'excellent'
        elif overall_fusion_quality > 0.4:
            fusion_assessment = 'good'
        elif overall_fusion_quality > 0.2:
            fusion_assessment = 'moderate'
        elif overall_fusion_quality > 0.05:
            fusion_assessment = 'weak'
        else:
            fusion_assessment = 'poor'
        
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
            'note': f"{fusion_layers_count} cross-attention layers active, molecular interaction optimized"
        }
    
    def get_training_health_summary(self):
        """Simplified comprehensive training health assessment for molecular interaction models"""
        health_issues = []
        recommendations = []
        confidence_score = 1.0
        
        # Check dual-branch balance (critical for molecular interactions)
        branch_stats = getattr(self, 'branch_statistics', {})
        if branch_stats:
            magnitude_ratio = branch_stats.get('magnitude_ratio', 0)
            if magnitude_ratio < 0.2:  # Adsorbate too weak
                health_issues.append('adsorbate_underutilized')
                recommendations.append('enhance_adsorbate_processing')
                confidence_score *= 0.9
            elif magnitude_ratio > 0.5:  # Adsorbate too dominant
                health_issues.append('adsorbate_overdominant')
                recommendations.append('balance_branch_weights')
                confidence_score *= 0.95
        
        # Check CBAM group interactions (key for molecular modeling)
        cbam_stats = {}
        for name, module in self.named_modules():
            if hasattr(module, 'get_group_interaction_stats'):
                stats = module.get_group_interaction_stats()
                if stats:
                    cbam_stats.update(stats)
        
        if cbam_stats:
            interaction_strength = cbam_stats.get('interaction_strength_value', 0)
            if interaction_strength < 0.3:
                health_issues.append('weak_molecular_interaction')
                recommendations.append('strengthen_cbam_attention')
                confidence_score *= 0.85
        
        # Check Vision Transformer health (if enabled)
        if hasattr(self, 'vision_transformer') and self.use_transformer:
            try:
                vit_summary = self.vision_transformer.get_attention_summary()
                vit_health = vit_summary.get('overall_health', {})
                
                if vit_health.get('status') == 'needs_optimization':
                    health_issues.extend(['attention_issues'])
                    recommendations.extend(['optimize_attention_heads'])
                    confidence_score *= 0.9
                    
                # Check attention entropy specifically
                metrics = vit_health.get('metrics', {})
                if metrics.get('attention_entropy', 0) > 10:
                    health_issues.append('attention_too_dispersed')
                    recommendations.append('reduce_temperature_or_increase_focus')
                    confidence_score *= 0.92
                    
            except Exception:
                pass  # VIT analysis not available
        
        # Determine overall health status
        if len(health_issues) == 0:
            overall_status = 'healthy'
        elif len(health_issues) <= 2:
            overall_status = 'warning'
        else:
            overall_status = 'critical'
        
        # Generate actionable summary
        priority_actions = []
        if 'weak_molecular_interaction' in health_issues:
            priority_actions.append('🔬 Strengthen CBAM molecular interactions')
        if 'adsorbate_underutilized' in health_issues:
            priority_actions.append('⚡ Enhance adsorbate branch processing')
        if 'attention_issues' in health_issues:
            priority_actions.append('🧠 Optimize transformer attention patterns')
        
        return {
            'overall_status': overall_status,
            'confidence_score': round(confidence_score, 3),
            'primary_issues': health_issues[:3],  # Top 3 issues
            'priority_actions': priority_actions[:2],  # Top 2 actions
            'molecular_interaction_health': 'good' if cbam_stats.get('interaction_strength_value', 0) > 0.4 else 'needs_attention',
            'architecture_efficiency': round(confidence_score * 100, 1)
        }
    
    def get_learning_efficiency_stats(self, current_loss=None, previous_loss=None, gradient_norm=None):
        """Monitor learning efficiency and convergence patterns"""
        stats = {}
        
        # Calculate loss improvement rate
        if current_loss is not None and previous_loss is not None and previous_loss > 0:
            loss_improvement_rate = (previous_loss - current_loss) / previous_loss
            stats['loss_improvement_rate'] = round(loss_improvement_rate, 6)
            
            # Assess convergence speed
            if loss_improvement_rate > 0.05:
                stats['convergence_speed'] = 'fast'
            elif loss_improvement_rate > 0.01:
                stats['convergence_speed'] = 'moderate'
            elif loss_improvement_rate > 0.001:
                stats['convergence_speed'] = 'slow'
            else:
                stats['convergence_speed'] = 'stagnant'
        
        # Gradient efficiency
        if gradient_norm is not None and current_loss is not None and current_loss > 0:
            gradient_efficiency = gradient_norm / current_loss
            stats['gradient_efficiency'] = round(gradient_efficiency, 4)
            
            if gradient_efficiency > 10:
                stats['gradient_status'] = 'too_high'
            elif gradient_efficiency > 1:
                stats['gradient_status'] = 'normal'
            else:
                stats['gradient_status'] = 'weak'
        
        # Model parameter utilization
        total_params = sum(p.numel() for p in self.parameters())
        active_params = sum(p.numel() for p in self.parameters() if p.requires_grad and p.grad is not None)
        
        if total_params > 0:
            param_utilization = active_params / total_params
            stats['parameter_utilization'] = round(param_utilization, 3)
        
        return stats
    
    def get_molecular_interaction_insights(self):
        """Specialized monitoring for molecular interaction learning patterns"""
        insights = {}
        
        # Analyze CBAM group interactions across all layers
        cbam_layers = []
        for name, module in self.named_modules():
            if hasattr(module, 'get_group_interaction_stats'):
                stats = module.get_group_interaction_stats()
                if stats:
                    cbam_layers.append({
                        'layer_name': name,
                        'stats': stats
                    })
        
        if cbam_layers:
            # Calculate molecular interaction trends
            interaction_strengths = [layer['stats'].get('interaction_strength_value', 0) for layer in cbam_layers]
            adsorbate_attentions = [layer['stats'].get('ads_attention_mean', 0) for layer in cbam_layers]
            solvent_attentions = [layer['stats'].get('solv_attention_mean', 0) for layer in cbam_layers]
            
            insights['num_interaction_layers'] = len(cbam_layers)
            insights['avg_interaction_strength'] = round(np.mean(interaction_strengths), 4) if interaction_strengths else 0
            insights['interaction_consistency'] = round(1 - np.std(interaction_strengths), 4) if len(interaction_strengths) > 1 else 1
            insights['adsorbate_vs_solvent_balance'] = round(np.mean(adsorbate_attentions) / (np.mean(solvent_attentions) + 1e-8), 4) if solvent_attentions else 0
            
            # Determine interaction learning quality
            if insights['avg_interaction_strength'] > 0.6:
                insights['interaction_learning'] = 'excellent'
            elif insights['avg_interaction_strength'] > 0.4:
                insights['interaction_learning'] = 'good'
            elif insights['avg_interaction_strength'] > 0.2:
                insights['interaction_learning'] = 'moderate'
            else:
                insights['interaction_learning'] = 'poor'
        
        # Analyze dual-branch contributions
        branch_stats = getattr(self, 'branch_statistics', {})
        if branch_stats:
            insights['adsorbate_contribution'] = round(branch_stats.get('magnitude_ratio', 0), 3)
            insights['solvent_dominance'] = round(1 - branch_stats.get('magnitude_ratio', 0), 3)
            insights['spatial_coverage_ratio'] = round(branch_stats.get('sparsity_ratio', 0), 2)
        
        return insights

