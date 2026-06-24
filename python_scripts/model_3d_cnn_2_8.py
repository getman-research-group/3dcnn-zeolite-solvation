"""
model_3d_cnn_2_8.py
This file contains the definitions for the 3D-CNN model with CBAM attention mechanism 
for adsorbate-solvent interaction energy prediction, optimized for type_2 format.

Format specifications:
- 28 input features (14 atomic features × 2 groups: adsorbate + solvent)
- Separated channel groups for better feature interpretation
- Default input shape: (batch_size, 28, 20, 20, 20)

The dual-branch CNN architecture with CBAM optimization (Plan A):
- Adsorbate branch: handles sparse central features with smaller kernels
- Solvent branch: processes dense distributed features with larger kernels
- Primary CBAM: interaction_attention at 72-channel fusion (optimal timing)
- Secondary CBAM: layer1, layer2 only (layer3 removed for efficiency)
- CNN backbone with selective attention for local feature extraction

Key optimizations in v2_8 (Plan A):
- ✅ OPTIMAL: Primary CBAM moved to interaction fusion point (72→72 channels)
- ✅ Group-aware attention with correct group_split=[24, 48] before mixing
- ✅ Efficiency: Removed layer3 CBAM (3→2 total CBAM modules)
- Simplified CNN backbone: removed layer4 for better stability
- Simplified ResidualBlock3D: single-path instead of dual-path design
- Dual-branch processing optimized for adsorbate-solvent separation
- Enhanced stability: reduced dropout and conservative initialization
- Final CNN output: 80 channels (was 96)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
import time  # Add time import for architecture performance recording


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
    Simplified 3D Residual Block with:
    - Single-path design for reduced complexity
    - Simplified CBAM attention for feature importance weighting  
    - Improved gradient flow and BatchNorm stability
    """
    def __init__(self, in_channels, out_channels, stride=1, use_cbam=True):
        super(ResidualBlock3D, self).__init__()
        
        # 🔥 SIMPLIFIED: Single convolution path instead of dual-path
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # 🔥 SIMPLIFIED: CBAM with smaller kernel for stability
        self.cbam = CBAM3D(out_channels, kernel_size=3, dropout=0.08) if use_cbam else nn.Identity()
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        
        # 🔥 STABILITY: ReLU activation and minimal dropout
        self.dropout = nn.Dropout3d(p=0.01)  # Reduced dropout for simplified architecture
        self.activation = nn.ReLU(inplace=True)  # ReLU for BatchNorm stability
    
    def forward(self, x):
        residual = x
        
        # 🔥 SIMPLIFIED: Standard residual path
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        # Apply simplified CBAM attention
        out = self.cbam(out)
        
        # Add shortcut and apply activation
        out += self.shortcut(residual)
        out = self.activation(out)
        
        return out

class AttentionCNN_2_8(nn.Module):
    """
    3D CNN model for adsorbate-solvent interaction energy prediction.
    Adapted for type_2 format with 28 input features (14 atomic features × 2 groups).
    
    Architecture (Optimized with Plan A):
    1. Dual-branch processing: separate adsorbate and solvent feature extraction
    2. Group-aware CBAM attention at interaction fusion point (OPTIMAL placement)
    3. CNN Backbone with selective CBAM attention (layer1, layer2 only)
    4. Enhanced regressor for final prediction
    
    This dual-branch approach combines:
    - Adsorbate branch: sparse central features processed with smaller kernels
    - Solvent branch: dense distributed features processed with larger kernels
    - Natural architectural emphasis on solvent features (2x capacity)
    
    CBAM Optimization (Plan A - Ultimate Simplified):
    - Primary attention: interaction_attention at 72-channel fusion point (24+48 groups) - 🔑 CRITICAL
    - Secondary attention: layer1 (32→48) at full 20×20×20 resolution - ✅ USEFUL  
    - Removed attention: layer2 (52→56), layer3 (56→64) for maximum efficiency - ❌ TOO LATE
    
    Type_2 format specifications:
    - Default 28 input channels (14 atomic features × 2 groups: adsorbate + solvent)
    - Supports dynamic channel numbers via in_channels parameter
    - Enhanced feature analysis for separated channel groups
    """
    def __init__(self, in_channels=28,  # Default 28 for type_2 format
                 dropout_rate=0.35,     # Increased from 0.25 based on overfitting analysis
                 feature_names=None):   # Add feature_names parameter
        
        super(AttentionCNN_2_8, self).__init__()
        
        # Store configuration including input channels for dynamic feature analysis
        self.in_channels = in_channels  # Store input channel count for type_2 compatibility
        self.dropout_rate = dropout_rate  # Store for analysis methods
        
        # Store feature names for analysis methods
        self.feature_names = feature_names
        
        # Enhanced channel group processing for Type_2 format (28 channels = 14 adsorbate + 14 solvent)
        assert in_channels == 28, "Type_2 format requires exactly 28 channels (14 adsorbate + 14 solvent)"
        
        # Adsorbate branch: Enhanced for better information preservation
        # 🎯 ENHANCED: Increased capacity to reduce 75.2% information loss
        # Adsorbate occupies central small volume, mostly zeros, so use enhanced processing with attention
        self.adsorbate_processor = nn.Sequential(
            nn.Conv3d(14, 20, kernel_size=3, padding=1, bias=False),  # Increased from 16 to 20
            nn.BatchNorm3d(20),
            nn.ReLU(inplace=True),  # ReLU for BatchNorm stability
            
            # 🔥 ADD ATTENTION: Lightweight CBAM for sparse feature focus
            CBAM3D(20, reduction_ratio=8, kernel_size=3, dropout=0.05),  # Optimized for sparse features
            
            # Enhanced feature densification pathway
            nn.Conv3d(20, 26, kernel_size=1, bias=False),  # Smooth feature densification (20→26)
            nn.BatchNorm3d(26),
            nn.ReLU(inplace=True),  # Keep ReLU for consistency
            nn.Conv3d(26, 32, kernel_size=3, padding=1, bias=False),  # Final expansion (26→32)
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),  # ReLU for final stability
            nn.Dropout3d(0.06)  # Slightly reduced dropout due to increased capacity
        )
        
        # Solvent branch: Mixed kernel strategy for optimal interaction capture
        # 🎯 MIXED STRATEGY: 5x5x5 first layer for H-bond capture, 3x3x3 second layer for feature integration
        # Solvent fills most space and is more important for interaction energy
        self.solvent_processor = nn.Sequential(
            nn.Conv3d(14, 32, kernel_size=5, padding=2, bias=False),  # 5x5x5 kernels for H-bond capture (4.0Å receptive field)
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),  # ReLU for BatchNorm stability
            
            # 🔥 ADD ATTENTION: Standard CBAM for dense feature optimization
            CBAM3D(32, reduction_ratio=6, kernel_size=5, dropout=0.04),  # Match kernel size for attention
            
            nn.Conv3d(32, 48, kernel_size=3, padding=1, bias=False),  # 3x3x3 kernels for feature integration
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True),  # ReLU for BatchNorm stability
            nn.Dropout3d(0.04)  # Slightly reduced dropout
        )
        
        # 🎯 OPTIMAL: Group-Aware CBAM at the fusion point (Plan A)
        # Apply attention BEFORE convolution mixing to preserve group information
        self.interaction_attention = CBAM3D(
            in_channels=80,           # Process complete 80 channels (32 adsorbate + 48 solvent)
            kernel_size=5,            # Maintain larger kernel for H-bond capture (2-3Å at 0.8Å resolution)
            dropout=0.08, 
            enable_group_interaction=True, 
            group_split=[32, 48]      # 🔑 CRITICAL: Updated group split (32 adsorbate + 48 solvent)
        )
        
        # Interaction fusion layer: Smooth feature integration and dimensionality evolution
        # � IMPROVED: Avoid aggressive compression to preserve interaction information
        self.interaction_conv = nn.Sequential(
            nn.Conv3d(80, 64, kernel_size=3, padding=1, bias=False),  # 80 -> 64, gentle compression
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),  # ReLU for BatchNorm stability 
            nn.Conv3d(64, 48, kernel_size=1, bias=False),  # 64 -> 48, further refinement
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True)  # ReLU for BatchNorm stability
        )
        
        # No manual branch weighting - let the architecture naturally express importance:
        # - Solvent processor: 48 channels + 5x5x5→3x3x3 mixed kernels + attention
        # - Adsorbate processor: 32 channels + 3x3x3 kernels + attention
        # This design provides balanced capacity with dual attention mechanisms and optimal receptive field
        
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
        
        # 🏗️ ENHANCED CNN BACKBONE: Smooth progressive feature evolution
        self.layer1 = ResidualBlock3D(48, 52, stride=1, use_cbam=False)  # 48 → 52
        self.pool1 = nn.AvgPool3d(2)  # 20×20×20 → 10×10×10
        
        self.layer2 = ResidualBlock3D(52, 58, stride=1, use_cbam=False)  # 52 → 58
        self.pool2 = nn.AvgPool3d(2)  # 10×10×10 → 5×5×5
        
        # 🔥 SMOOTHER: Final smooth step to 64 channels (smoother growth: [4,6,6])
        self.layer3 = ResidualBlock3D(58, 64, stride=1, use_cbam=False)  # 58 → 64
        
        # More aggressive spatial compression for MLP parameter reduction
        self.adaptive_pool = nn.AdaptiveAvgPool3d(2)  # 5×5×5 → 2×2×2 (8 spatial locations)
        
        # 🎯 OPTIMIZED MLP: Match uniform CNN backbone output (64 channels)
        regressor_input_dim = 64 * 2 * 2 * 2  # 512 (optimized from 640 to 512)

        # Streamlined regressor: fewer parameters but richer input features  
        self.regressor = nn.Sequential(
            nn.Flatten(),
            
            # 🔥 ENHANCED: Gradual information compression to reduce information loss
            nn.Linear(regressor_input_dim, 128),  # 512 → 128 (4x reduction)
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),  # Slightly reduced dropout for middle layer
            
            # 🎯 NEW: Add intermediate layer for smoother information transition
            nn.Linear(128, 64),  # 128 → 64 (2x reduction)
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),  # Progressive dropout reduction
            
            # Final output - smooth transition from rich features to prediction
            nn.Linear(64, 1)  # 64 → 1
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
                if m.weight.size(0) > 1000:  # Large linear layers (regressor)
                    nn.init.normal_(m.weight, 0, 0.01)  # Very conservative for regressor
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
        adsorbate_features = self.adsorbate_processor(adsorbate_channels)  # (batch, 32, 20, 20, 20)
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
        # The architecture now balances adsorbate (32 channels) and solvent (48 channels) more evenly
        combined_features = torch.cat([adsorbate_features, solvent_features], dim=1)  # (batch, 80, 20, 20, 20)
        if monitor_activations:
            monitoring_data['activation_stats'].update(self.analyze_activations(combined_features, "combined_features"))
        
        # 🎯 OPTIMAL ATTENTION: Apply group-aware attention BEFORE convolution mixing
        # This preserves the physical meaning of channel groups (32 adsorbate + 48 solvent)
        attended_features = self.interaction_attention(combined_features)  # (batch, 80, 20, 20, 20)
        if monitor_activations:
            monitoring_data['activation_stats'].update(self.analyze_activations(attended_features, "attended_features"))
            # Store attention information for analysis
            monitoring_data['attention_info']['interaction_attention'] = self.extract_attention_weights_direct(self.interaction_attention.channel_attention)
        
        # Model adsorbate-solvent interactions on the attended features
        interaction_features = self.interaction_conv(attended_features)   # (batch, 48, 20, 20, 20)
        if monitor_activations:
            monitoring_data['activation_stats'].update(self.analyze_activations(interaction_features, "interaction_features"))
        
        # Update branch monitoring statistics
        if monitor_activations or detailed_analysis:
            self._update_branch_statistics(adsorbate_features, solvent_features, interaction_features)
            monitoring_data['feature_analysis']['branch_analysis'] = self.get_branch_analysis()
            
        # Continue with the enhanced CNN backbone
        x = interaction_features  # Start main CNN pipeline with interaction features
        
        # Progressive feature learning with enhanced monitoring
        x = self.layer1(x)  # (batch, 48, 20, 20, 20) -> (batch, 64, 20, 20, 20)
        if monitor_activations:
            monitoring_data['activation_stats'].update(self.analyze_activations(x, "layer1"))
            monitoring_data['attention_info']['layer1'] = self.extract_attention_weights(self.layer1)
        if detailed_analysis:
            monitoring_data['layer_contributions']['layer1'] = self.calculate_layer_contribution(x)
            monitoring_data['training_insights']['layer1_attention_strength'] = self.analyze_attention_strength(self.layer1)
        x = self.pool1(x)  # (batch, 64, 10, 10, 10)
        
        x = self.layer2(x)  # (batch, 64, 10, 10, 10) -> (batch, 80, 10, 10, 10)
        if monitor_activations:
            monitoring_data['activation_stats'].update(self.analyze_activations(x, "layer2"))
        if detailed_analysis:
            monitoring_data['layer_contributions']['layer2'] = self.calculate_layer_contribution(x)
        x = self.pool2(x)  # (batch, 80, 5, 5, 5)
        
        # 🔥 NEW: Layer3 for deeper feature extraction with uniform progression
        x = self.layer3(x)  # (batch, 56, 5, 5, 5) -> (batch, 64, 5, 5, 5)
        if monitor_activations:
            monitoring_data['activation_stats'].update(self.analyze_activations(x, "layer3"))
        if detailed_analysis:
            monitoring_data['layer_contributions']['layer3'] = self.calculate_layer_contribution(x)
        
        # Aggressive spatial compression: 5×5×5 → 2×2×2 for parameter reduction
        x = self.adaptive_pool(x)  # (batch, 64, 2, 2, 2)
        if monitor_activations:
            monitoring_data['activation_stats'].update(self.analyze_activations(x, "adaptive_pool"))
        
        # Flatten for regressor
        x = x.view(x.size(0), -1)  # (batch, 64*2*2*2 = 512)
        
        # Classification with gradient flow analysis
        x_before_regressor = x.clone() if detailed_analysis else None
        x = self.regressor(x)  # (batch, 1)
        
        if monitor_activations:
            monitoring_data['activation_stats'].update(self.analyze_activations(x, "output"))
            # Store attention information
            self.attention_weights_history.append(monitoring_data['attention_info'])
            
        if detailed_analysis:
            monitoring_data['training_insights']['regressor_contribution'] = self.analyze_regressor_contribution(x_before_regressor, x)
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
                elif 'layer1' in name:
                    weight_norms[f"cnn_backbone.layer1"] = param.data.norm(2).item()
                elif 'layer2' in name:
                    weight_norms[f"cnn_backbone.layer2"] = param.data.norm(2).item()
                elif 'layer3' in name:
                    weight_norms[f"cnn_backbone.layer3"] = param.data.norm(2).item()
                elif 'layer4' in name:
                    weight_norms[f"cnn_backbone.layer4"] = param.data.norm(2).item()
                elif 'regressor' in name:
                    layer_idx = name.split('.')[1]
                    weight_norms[f"regressor.layer_{layer_idx}"] = param.data.norm(2).item()
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
            # Fallback: generate generic feature names if not provided
            feature_names = [f"feature_{i:02d}" for i in range(x.size(1))]
        
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
            elif len(x_np.shape) == 3:  # (batch, seq_len, features) for sequence data
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
            # Fallback: generate generic feature names if not provided
            feature_names = [f"feature_{i:02d}" for i in range(x.size(1))]
        
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
                'capacity_ratio': 48 / 32,  # Solvent vs adsorbate channel capacity (updated to 48/32 = 1.50)
                'kernel_size_ratio': 3 / 3,  # Both use 3x3x3 kernels now (balanced)
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
                    'adsorbate_capacity': '32 channels, 3x3x3 kernels, lightweight CBAM',  # Consistent with sparse features
                    'solvent_capacity': '48 channels, 5x5x5→3x3x3 mixed kernels, standard CBAM',    # Mixed strategy for optimal interaction capture
                    'natural_emphasis': 'balanced capacity with optimized receptive field (1.50x ratio)'
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
        
        if hasattr(layer, 'cbam') and not isinstance(layer.cbam, torch.nn.Identity):
            # Channel attention analysis
            if hasattr(layer.cbam, 'channel_attention') and hasattr(layer.cbam.channel_attention, 'last_attention_weights'):
                ch_weights = layer.cbam.channel_attention.last_attention_weights
                if ch_weights is not None:
                    attention_analysis['channel_attention'] = {
                        'uniformity': float(1.0 / (ch_weights.std().item() + 1e-8)),
                        'selectivity': float(ch_weights.max().item() / ch_weights.mean().item()),
                        'effective_channels': int((ch_weights > ch_weights.mean()).sum().item())
                    }
            
            # Spatial attention analysis
            if hasattr(layer.cbam, 'spatial_attention') and hasattr(layer.cbam.spatial_attention, 'last_attention_weights'):
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
        else:
            # Layer has no CBAM attention (uses Identity)
            attention_analysis['attention_type'] = 'none'
        
        return attention_analysis
    
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
        
        return comparison
    
    def analyze_regressor_contribution(self, regressor_input, regressor_output):
        """🎯 Analyze regressor's contribution to final prediction"""
        if regressor_input is None or regressor_output is None:
            return {}
        
        input_magnitude = torch.mean(torch.abs(regressor_input)).item()
        output_magnitude = torch.mean(torch.abs(regressor_output)).item()
        
        return {
            'input_magnitude': float(input_magnitude),
            'output_magnitude': float(output_magnitude),
            'amplification_factor': float(output_magnitude / (input_magnitude + 1e-8)),
            'prediction_confidence': float(torch.std(regressor_output).item())
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
    
    def print_training_insights(self, epoch=None, batch_idx=None, monitoring_data=None):
        """📋 Print comprehensive training insights for log monitoring"""
        print(f"\n{'='*80}")
        print(f"🔬 TRAINING INSIGHTS - Epoch {epoch}, Batch {batch_idx}" if epoch is not None else "🔬 MODEL ANALYSIS")
        print(f"{'='*80}")
        
        # Use provided monitoring data, or create empty dict if none provided
        if monitoring_data is None:
            monitoring_data = {}
        
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
        
        # 4. Training Insights
        if 'training_insights' in monitoring_data:
            insights = monitoring_data['training_insights']
            print(f"\n🎯 TRAINING INSIGHTS:")
            
            # Model efficiency insights
            if 'model_architecture_efficiency' in insights:
                efficiency = insights['model_architecture_efficiency']
                if 'parameter_efficiency' in efficiency:
                    param_eff = efficiency['parameter_efficiency']
                    print(f"   Parameter utilization: {param_eff['param_utilization']:.3f}")
                    
                balance = efficiency.get('component_balance', {})
                if balance:
                    print(f"   Architecture balance: CNN={balance.get('convolutional', 0):.1f}%, "
                          f"DualBranch={balance.get('dual_branch_processing', 0):.1f}%, "
                          f"Attention={balance.get('cbam_attention', 0):.1f}%")
        
        # 5. CBAM Attention Analysis
        if 'attention_info' in monitoring_data:
            attention_info = monitoring_data['attention_info']
            
            # Print CBAM layer attention summaries
            cbam_layers = [k for k in attention_info.keys() if 'layer' in k.lower()]
            if cbam_layers:
                print(f"\n🧠 CBAM ATTENTION ANALYSIS:")
                for layer_name in cbam_layers[:3]:  # Show first 3 layers
                    if layer_name in attention_info:
                        layer_info = attention_info[layer_name]
                        if 'channel_weights' in layer_info:
                            ch_weights = layer_info['channel_weights']
                            print(f"   {layer_name}: channel_attn(μ={ch_weights['mean']:.3f}, σ={ch_weights['std']:.3f})")
        
        print(f"{'='*80}\n")
    
    def get_optimization_recommendations(self):
        """💡 Generate specific optimization recommendations based on available monitoring data"""
        recommendations = []
        
        # Create monitoring data from available sources
        monitoring_data = {}
        
        # Fallback: Use attention history if detailed data not available
        if hasattr(self, 'attention_weights_history') and self.attention_weights_history:
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
            recommendations.append("🔧 Dual-branch CNN architecture: Monitor branch balance and CBAM attention effectiveness.")
        
        # Note: This model uses CBAM attention, not input-level attention
        # Feature importance is handled through CBAM channel attention weights
        
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
        
        if hasattr(layer, 'cbam') and not isinstance(layer.cbam, torch.nn.Identity):
            # Channel attention weights
            if hasattr(layer.cbam, 'channel_attention') and hasattr(layer.cbam.channel_attention, 'last_attention_weights'):
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
            if hasattr(layer.cbam, 'spatial_attention') and hasattr(layer.cbam.spatial_attention, 'last_attention_weights'):
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
        """Get current learned feature importance weights
        
        Note: This CNN model uses CBAM attention for channel-wise importance,
        not input-level feature importance like transformer models.
        """
        # This model uses CBAM channel attention instead of input feature attention
        # Feature importance is captured through CBAM channel attention weights
        return {}
    
    def get_model_complexity_analysis(self):
        """Get detailed model complexity analysis for dual-branch CNN architecture"""
        analysis = {}
        
        # Parameter count by component (updated for dual-branch architecture)
        conv_params = 0
        attention_params = 0  # CBAM attention
        regressor_params = 0
        bn_params = 0
        branch_params = 0  # New: dual-branch processing parameters
        
        for name, param in self.named_parameters():
            param_count = param.numel()
            
            if any(branch in name for branch in ['adsorbate_processor', 'solvent_processor', 'interaction_conv']):
                branch_params += param_count
            elif 'conv' in name.lower():
                conv_params += param_count
            elif 'cbam' in name.lower() or 'attention' in name.lower():
                attention_params += param_count
            elif 'regressor' in name.lower() or 'fc' in name.lower():
                regressor_params += param_count
            elif 'bn' in name.lower() or 'norm' in name.lower():
                bn_params += param_count
        
        total_params = conv_params + attention_params + regressor_params + bn_params + branch_params
        
        analysis['parameter_breakdown'] = {
            'total': total_params,
            'dual_branch_processing': branch_params,  # New category
            'convolutional': conv_params,
            'cbam_attention': attention_params,
            'regressor': regressor_params,
            'batch_norm': bn_params
        }
        
        analysis['parameter_percentages'] = {
            'dual_branch_processing': (branch_params / total_params * 100) if total_params > 0 else 0,
            'convolutional': (conv_params / total_params * 100) if total_params > 0 else 0,
            'cbam_attention': (attention_params / total_params * 100) if total_params > 0 else 0,
            'regressor': (regressor_params / total_params * 100) if total_params > 0 else 0,
            'batch_norm': (bn_params / total_params * 100) if total_params > 0 else 0
        }
        
        # Memory analysis (rough estimate)
        analysis['memory_estimate'] = {
            'parameters_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'approx_forward_pass_mb': total_params * 8 / (1024 * 1024)  # Rough estimate including activations
        }
        
        # Architecture information
        analysis['architecture_info'] = {
            'dual_branch_architecture': True,
            'adsorbate_channels': 14,
            'solvent_channels': 14,
            'cbam_attention': True
        }
        
        return analysis
    
    def print_model_analysis(self):
        """Print comprehensive model analysis for dual-branch CNN architecture"""
        analysis = self.get_model_complexity_analysis()
        
        print(f"\n🔬 DUAL-BRANCH 3D CNN MODEL ANALYSIS")
        print(f"{'='*60}")
        
        # Architecture info
        arch_info = analysis['architecture_info']
        print(f"🏗️ Architecture Configuration:")
        print(f"  Dual-branch Architecture: {arch_info['dual_branch_architecture']}")
        print(f"  Adsorbate Channels: {arch_info['adsorbate_channels']}")
        print(f"  Solvent Channels: {arch_info['solvent_channels']}")
        print(f"  CBAM Attention: {arch_info['cbam_attention']}")
        
        # Parameter breakdown
        params = analysis['parameter_breakdown']
        percentages = analysis['parameter_percentages']
        
        print(f"\n📊 Parameter Distribution:")
        print(f"  Total parameters: {params['total']:,}")
        print(f"  Dual-branch Processing: {params['dual_branch_processing']:,} ({percentages['dual_branch_processing']:.1f}%)")
        print(f"  Convolutional:     {params['convolutional']:,} ({percentages['convolutional']:.1f}%)")
        print(f"  CBAM Attention:    {params['cbam_attention']:,} ({percentages['cbam_attention']:.1f}%)")
        print(f"  Regressor:        {params['regressor']:,} ({percentages['regressor']:.1f}%)")
        print(f"  Batch Norm:        {params['batch_norm']:,} ({percentages['batch_norm']:.1f}%)")
        
        # Memory estimates
        memory = analysis['memory_estimate']
        print(f"\n💾 Memory Estimates:")
        print(f"  Parameters: {memory['parameters_mb']:.2f} MB")
        print(f"  Forward pass (approx): {memory['approx_forward_pass_mb']:.2f} MB")
        
        # Model architecture insights
        print(f"\n🏗️  Architecture Insights:")
        attention_pct = percentages['cbam_attention']
        if attention_pct > 15:
            print(f"  ⚠️ High attention overhead ({attention_pct:.1f}%) - monitor training efficiency")
        elif attention_pct > 8:
            print(f"  ✓ Moderate attention usage ({attention_pct:.1f}%)")
        else:
            print(f"  ⚡ Efficient attention usage ({attention_pct:.1f}%)")
        
        branch_pct = percentages['dual_branch_processing']
        if branch_pct > 20:
            print(f"  🔬 Branch-specialized architecture ({branch_pct:.1f}%) - optimized for molecular interactions")
        else:
            print(f"  ⚖️ Balanced dual-branch processing ({branch_pct:.1f}%)")
        
        if percentages['regressor'] > 40:
            print(f"  ⚠️ Regressor-heavy model ({percentages['regressor']:.1f}%) - may overfit")
        elif percentages['convolutional'] > 50:
            print(f"  ✓ Feature extraction focused ({percentages['convolutional']:.1f}%)")
        
        print(f"{'='*60}")
    
    
    def get_attention_summary(self):
        """Get attention summary (compatibility method for train_3d_cnn.py) - Updated for optimized architecture"""
        summary = {}
        
        # Check interaction attention (new primary attention mechanism)
        if hasattr(self, 'interaction_attention') and hasattr(self.interaction_attention, 'channel_attention'):
            interaction_attn = self.interaction_attention.channel_attention
            if hasattr(interaction_attn, 'last_attention_weights') and interaction_attn.last_attention_weights is not None:
                weights = interaction_attn.last_attention_weights.squeeze()
                summary['interaction_attention'] = {
                    'mean': float(weights.mean().item()),
                    'std': float(weights.std().item()),
                    'min': float(weights.min().item()),
                    'max': float(weights.max().item()),
                    'channels': int(weights.shape[1] if len(weights.shape) > 1 else weights.numel()),
                    'group_interaction_enabled': getattr(interaction_attn, 'enable_group_interaction', False),
                    'group_split': getattr(interaction_attn, 'group_split', None)
                }
        
        # Check layer1 and layer2 attention (retained)
        for layer_name, layer in [('layer1', self.layer1), ('layer2', self.layer2)]:
            if hasattr(layer, 'cbam') and not isinstance(layer.cbam, torch.nn.Identity):
                if hasattr(layer.cbam, 'channel_attention'):
                    cbam_attn = layer.cbam.channel_attention
                    if hasattr(cbam_attn, 'last_attention_weights') and cbam_attn.last_attention_weights is not None:
                        weights = cbam_attn.last_attention_weights.squeeze()
                        summary[layer_name] = {
                            'mean': float(weights.mean().item()),
                            'std': float(weights.std().item()),
                            'min': float(weights.min().item()),
                            'max': float(weights.max().item()),
                            'channels': int(weights.shape[1] if len(weights.shape) > 1 else weights.numel())
                        }
        
        # Note: layer3 no longer has attention as per optimization
        if not summary:
            return "No attention data available"
        
        return summary
    
    
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
    
    def get_processor_statistics(self):
        """Get comprehensive statistics for dual-branch processors and interaction layer"""
        stats = {}
        
        # Adsorbate processor statistics
        if hasattr(self, 'adsorbate_processor'):
            adsorbate_stats = {}
            total_params = sum(p.numel() for p in self.adsorbate_processor.parameters())
            active_params = sum(p.numel() for p in self.adsorbate_processor.parameters() if p.requires_grad)
            
            # Weight norms for each layer in adsorbate processor
            layer_norms = []
            for i, layer in enumerate(self.adsorbate_processor):
                if hasattr(layer, 'weight') and layer.weight is not None:
                    layer_norms.append(torch.norm(layer.weight).item())
            
            adsorbate_stats.update({
                'total_params': total_params,
                'active_params': active_params,
                'param_efficiency': active_params / total_params if total_params > 0 else 0,
                'avg_weight_norm': np.mean(layer_norms) if layer_norms else 0,
                'max_weight_norm': np.max(layer_norms) if layer_norms else 0,
                'layer_count': len([l for l in self.adsorbate_processor if hasattr(l, 'weight')]),
                'output_channels': 24,  # Fixed for adsorbate branch
                'architecture': 'sparse_focused'  # Smaller kernels for sparse adsorbate features
            })
            stats['adsorbate'] = adsorbate_stats
        
        # Solvent processor statistics  
        if hasattr(self, 'solvent_processor'):
            solvent_stats = {}
            total_params = sum(p.numel() for p in self.solvent_processor.parameters())
            active_params = sum(p.numel() for p in self.solvent_processor.parameters() if p.requires_grad)
            
            # Weight norms for each layer in solvent processor
            layer_norms = []
            for i, layer in enumerate(self.solvent_processor):
                if hasattr(layer, 'weight') and layer.weight is not None:
                    layer_norms.append(torch.norm(layer.weight).item())
            
            solvent_stats.update({
                'total_params': total_params,
                'active_params': active_params,
                'param_efficiency': active_params / total_params if total_params > 0 else 0,
                'avg_weight_norm': np.mean(layer_norms) if layer_norms else 0,
                'max_weight_norm': np.max(layer_norms) if layer_norms else 0,
                'layer_count': len([l for l in self.solvent_processor if hasattr(l, 'weight')]),
                'output_channels': 48,  # Fixed for solvent branch
                'architecture': 'dense_focused'  # Larger kernels for dense solvent features
            })
            stats['solvent'] = solvent_stats
        
        # Interaction layer statistics
        if hasattr(self, 'interaction_conv'):
            interaction_stats = {}
            total_params = sum(p.numel() for p in self.interaction_conv.parameters())
            active_params = sum(p.numel() for p in self.interaction_conv.parameters() if p.requires_grad)
            
            # Weight norms for interaction layers
            layer_norms = []
            for i, layer in enumerate(self.interaction_conv):
                if hasattr(layer, 'weight') and layer.weight is not None:
                    layer_norms.append(torch.norm(layer.weight).item())
            
            interaction_stats.update({
                'total_params': total_params,
                'active_params': active_params,
                'param_efficiency': active_params / total_params if total_params > 0 else 0,
                'avg_weight_norm': np.mean(layer_norms) if layer_norms else 0,
                'max_weight_norm': np.max(layer_norms) if layer_norms else 0,
                'layer_count': len([l for l in self.interaction_conv if hasattr(l, 'weight')]),
                'input_channels': 72,   # 24 + 48 from dual branches
                'output_channels': 32,  # Reduced for efficiency
                'architecture': 'feature_mixing'  # Combines adsorbate + solvent features
            })
            stats['interaction'] = interaction_stats
        
        # Calculate relative processor importance
        if 'adsorbate' in stats and 'solvent' in stats:
            ads_params = stats['adsorbate']['total_params']
            sol_params = stats['solvent']['total_params']
            total_branch_params = ads_params + sol_params
            
            stats['branch_balance'] = {
                'adsorbate_share': ads_params / total_branch_params if total_branch_params > 0 else 0,
                'solvent_share': sol_params / total_branch_params if total_branch_params > 0 else 0,
                'param_ratio': sol_params / ads_params if ads_params > 0 else 0,
                'design_philosophy': 'solvent_emphasis' if sol_params > ads_params * 1.5 else 'balanced'
            }
        
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
    
    def analyze_layer_importance_scores(self, x, target_layer_name=None):
        """🎯 Analyze the importance of each CNN layer through gradient-based attribution
        
        This helps determine which layers contribute most to the final prediction,
        guiding architecture optimization decisions.
        """
        self.eval()
        importance_scores = {}
        
        try:
            # Forward pass with gradient computation
            x.requires_grad_(True)
            
            # Use simplified forward call to avoid parameter conflicts
            output = self.forward(x)
            
            # Get gradients with respect to input
            grad_outputs = torch.ones_like(output)
            input_grads = torch.autograd.grad(
                outputs=output,
                inputs=x,
                grad_outputs=grad_outputs,
                retain_graph=False,
                create_graph=False
            )[0]
            
            # Analyze gradient patterns to infer layer importance
            gradient_magnitude = torch.mean(torch.abs(input_grads)).item()
            
            # Scale gradient magnitude to more readable range
            # Gradients are typically very small (1e-5 to 1e-3), so we scale them up
            scaled_gradient = gradient_magnitude * 1e6  # Scale to make values readable
            
            # Create importance scores for major components based on parameter count and theoretical contribution
            total_params = sum(p.numel() for p in self.parameters())
            layer_info = {
                'layer1': {'params': sum(p.numel() for p in self.layer1.parameters()) if hasattr(self, 'layer1') else 0, 'type': 'conv'},
                'layer2': {'params': sum(p.numel() for p in self.layer2.parameters()) if hasattr(self, 'layer2') else 0, 'type': 'conv'},
                'layer3': {'params': sum(p.numel() for p in self.layer3.parameters()) if hasattr(self, 'layer3') else 0, 'type': 'conv'},
                'adaptive_pool': {'params': 0, 'type': 'pooling'}  # No parameters
            }
            
            # Calculate importance scores with improved scaling
            for layer_name, info in layer_info.items():
                if hasattr(self, layer_name) or layer_name == 'adaptive_pool':
                    param_count = info['params']
                    layer_type = info['type']
                    
                    # Improved gradient importance calculation with better scaling
                    if param_count > 0:
                        # For layers with parameters: use parameter proportion with readable scaling
                        param_proportion = param_count / total_params
                        gradient_importance = scaled_gradient * param_proportion
                    else:
                        # For pooling layers: assign importance based on functional role
                        gradient_importance = scaled_gradient * 0.1  # 10% of scaled gradient
                    
                    # Calculate parameter efficiency with more reasonable scaling
                    if param_count > 0:
                        # Efficiency = gradient influence per parameter (scaled for readability)
                        param_efficiency = scaled_gradient / param_count
                    else:
                        # For layers without parameters, show functional efficiency
                        param_efficiency = scaled_gradient * 0.1  # Functional contribution
                    
                    # Apply layer-specific multipliers based on architectural importance
                    layer_multipliers = {'layer1': 1.5, 'layer2': 1.2, 'layer3': 1.0, 'adaptive_pool': 0.8}
                    base_multiplier = layer_multipliers.get(layer_name, 1.0)
                    
                    importance_scores[layer_name] = {
                        'gradient_importance': gradient_importance * base_multiplier,
                        'activation_magnitude': scaled_gradient,  # Use scaled version
                        'gradient_activation_ratio': 1.0,  # Simplified for now
                        'shape': 'varies',
                        'parameter_efficiency': param_efficiency
                    }
        
        except Exception as e:
            print(f"Warning: Could not compute layer importance scores: {e}")
        
        finally:
            self.train()
        
        return importance_scores
    
    def _get_layer_params(self, layer_name):
        """Get parameter count for a specific layer"""
        layer_map = {
            'layer1': self.layer1,
            'layer2': self.layer2,
            'layer3': self.layer3,
            'adaptive_pool': self.adaptive_pool
        }
        
        if layer_name in layer_map:
            return sum(p.numel() for p in layer_map[layer_name].parameters())
        return 0
    
    def analyze_regressor_bottlenecks(self, x_before_regressor, predictions):
        """🔍 Analyze regressor performance and identify potential bottlenecks
        
        This helps determine if the regressor is the limiting factor and
        suggests optimal regressor architecture.
        """
        try:
            if x_before_regressor is None or predictions is None:
                print("    ⚠️ Debug: x_before_regressor or predictions is None")
                return {}
            
            if x_before_regressor.numel() == 0 or predictions.numel() == 0:
                print("    ⚠️ Debug: Empty tensors provided to regressor analysis")
                return {}
            
            analysis = {}
            
            # Feature utilization in regressor input
            input_features = x_before_regressor.detach().cpu()
            
            # Analyze feature importance in regressor input
            feature_magnitudes = torch.mean(torch.abs(input_features), dim=0)
            feature_variances = torch.var(input_features, dim=0)
            
            # Identify potentially unused features (low magnitude and variance)
            low_importance_threshold = torch.quantile(feature_magnitudes, 0.1)
            low_variance_threshold = torch.quantile(feature_variances, 0.1)
            
            unused_features = torch.logical_and(
                feature_magnitudes < low_importance_threshold,
                feature_variances < low_variance_threshold
            )
            
            effective_features = torch.sum(~unused_features).item()
            total_features = len(feature_magnitudes)
            
            analysis['regressor_input_analysis'] = {
                'total_features': total_features,
                'effective_features': effective_features,
                'feature_utilization_ratio': effective_features / total_features,
                'unused_feature_percentage': torch.sum(unused_features).item() / total_features * 100,
                'feature_magnitude_range': {
                    'min': torch.min(feature_magnitudes).item(),
                    'max': torch.max(feature_magnitudes).item(),
                    'mean': torch.mean(feature_magnitudes).item(),
                    'std': torch.std(feature_magnitudes).item()
                }
            }
            
            # Regressor capacity analysis
            regressor_params = sum(p.numel() for p in self.regressor.parameters())
            samples_per_param = len(input_features) / regressor_params if regressor_params > 0 else 0
            
            analysis['regressor_capacity_analysis'] = {
                'total_parameters': regressor_params,
                'samples_per_parameter': samples_per_param,
                'capacity_utilization': 'over_parameterized' if samples_per_param < 5 else 
                                       'well_balanced' if samples_per_param < 20 else 'under_parameterized'
            }
            
            # Output analysis
            output_variance = torch.var(predictions).item()
            output_range = torch.max(predictions).item() - torch.min(predictions).item()
            
            analysis['regressor_output_analysis'] = {
                'prediction_variance': output_variance,
                'prediction_range': output_range,
                'output_saturation': torch.mean((torch.abs(predictions) > 0.9).float()).item() * 100,  # 转换为float再计算mean
            }
            
            # Recommendations
            recommendations = []
            
            if analysis['regressor_input_analysis']['feature_utilization_ratio'] < 0.7:
                recommendations.append(f"⚠️ Low feature utilization ({analysis['regressor_input_analysis']['feature_utilization_ratio']:.2f}). Consider dimensionality reduction.")
            
            if samples_per_param < 5:
                recommendations.append("⚠️ Regressor over-parameterized. Consider reducing layer sizes.")
            elif samples_per_param > 20:
                recommendations.append("💡 Regressor under-parameterized. Consider increasing capacity.")
            
            if output_variance < 0.01:
                recommendations.append("⚠️ Low prediction variance. Model may be under-expressive.")
            
            analysis['recommendations'] = recommendations
            
            return analysis
            
        except Exception as e:
            print(f"    ⚠️ Debug: Exception in analyze_regressor_bottlenecks: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def compare_architecture_efficiency(self, monitoring_data_history=None):
        """📊 Compare current architecture efficiency with theoretical optimal
        
        This provides insights for architectural decisions.
        """
        efficiency_analysis = {}
        
        # Current architecture analysis
        current_analysis = self.get_model_complexity_analysis()
        total_params = current_analysis['parameter_breakdown']['total']
        
        # CNN efficiency analysis
        cnn_params = (
            current_analysis['parameter_breakdown']['dual_branch_processing'] +
            current_analysis['parameter_breakdown']['convolutional']
        )
        regressor_params = current_analysis['parameter_breakdown']['regressor']
        
        efficiency_analysis['current_architecture'] = {
            'total_parameters': total_params,
            'cnn_to_regressor_ratio': cnn_params / (regressor_params + 1e-8),
            'parameter_distribution': {
                'feature_extraction_percentage': (cnn_params / total_params) * 100,
                'regression_percentage': (regressor_params / total_params) * 100
            }
        }
        
        # Theoretical optimal analysis based on data complexity
        input_dimensions = 28 * 20 * 20 * 20  # Type_2 format
        output_dimensions = 1
        
        # Rule of thumb: feature extraction should handle ~80% of complexity reduction
        # Regressor should handle final ~20% of complexity reduction
        theoretical_optimal = {
            'feature_extraction_percentage': 70,  # Should be dominant
            'regression_percentage': 30,  # Should be minimal
            'recommended_cnn_regressor_ratio': 2.3  # Feature extraction should be 2-3x larger
        }
        
        # Compare with theoretical optimal
        current_fe_pct = efficiency_analysis['current_architecture']['parameter_distribution']['feature_extraction_percentage']
        current_reg_pct = efficiency_analysis['current_architecture']['parameter_distribution']['regression_percentage']
        
        efficiency_analysis['optimization_insights'] = {
            'feature_extraction_deviation': current_fe_pct - theoretical_optimal['feature_extraction_percentage'],
            'regression_deviation': current_reg_pct - theoretical_optimal['regression_percentage'],
            'architecture_balance': 'optimal' if abs(current_fe_pct - theoretical_optimal['feature_extraction_percentage']) < 10 else
                                   'regressor_heavy' if current_reg_pct > theoretical_optimal['regression_percentage'] else 'cnn_heavy'
        }
        
        # Performance-based recommendations
        recommendations = []
        
        if efficiency_analysis['optimization_insights']['architecture_balance'] == 'regressor_heavy':
            potential_saved_params = regressor_params * 0.3  # Could save 30% of regressor params
            recommendations.extend([
                f"🎯 Architecture is regressor-heavy ({current_reg_pct:.1f}% vs optimal {theoretical_optimal['regression_percentage']}%)",
                f"💡 Could reduce regressor by ~{potential_saved_params:.0f} parameters",
                "🔄 Redistribute parameters: CNN backbone → deeper, Regressor → shallower"
            ])
        elif efficiency_analysis['optimization_insights']['architecture_balance'] == 'cnn_heavy':
            recommendations.extend([
                f"🎯 Architecture is CNN-heavy ({current_fe_pct:.1f}% vs optimal {theoretical_optimal['feature_extraction_percentage']}%)",
                "💡 Could increase regressor capacity or reduce CNN complexity",
                "🔄 Current setup may be optimal for complex molecular interactions"
            ])
        else:
            recommendations.append("✅ Architecture balance is near optimal")
        
        # Layer-specific efficiency analysis
        if hasattr(self, '_layer_efficiency_history'):
            layer_efficiency = self._analyze_layer_efficiency_trends()
            efficiency_analysis['layer_trends'] = layer_efficiency
        
        efficiency_analysis['recommendations'] = recommendations
        
        # Add current_efficiency section for training script compatibility
        efficiency_analysis['current_efficiency'] = {
            'parameter_efficiency_score': min(1.0, current_fe_pct / 70.0),  # Normalized to theoretical optimal
            'cnn_regressor_balance': efficiency_analysis['optimization_insights']['architecture_balance']
        }
        
        return efficiency_analysis
    
    def record_architecture_performance(self, train_metrics, test_metrics, monitoring_data=None):
        """📈 Record performance metrics for architecture comparison
        
        Call this after each training to build a performance database
        for architecture optimization.
        """
        if not hasattr(self, '_architecture_performance_history'):
            self._architecture_performance_history = []
        
        # Get current architecture signature
        current_analysis = self.get_model_complexity_analysis()
        architecture_signature = {
            'total_params': current_analysis['parameter_breakdown']['total'],
            'cnn_params': current_analysis['parameter_breakdown']['dual_branch_processing'] + 
                         current_analysis['parameter_breakdown']['convolutional'],
            'regressor_params': current_analysis['parameter_breakdown']['regressor'],
            'attention_params': current_analysis['parameter_breakdown']['cbam_attention'],
            'layer_structure': {
                'layer1_out_channels': 48,
                'layer2_out_channels': 64, 
                'layer3_out_channels': 80,
                'regressor_input_dim': 640,
                'regressor_hidden_dim': 128
            }
        }
        
        # Performance metrics
        performance_record = {
            'architecture': {
                **architecture_signature,
                'cnn_regressor_ratio': architecture_signature['cnn_params'] / (architecture_signature['regressor_params'] + 1e-8)
            },
            'performance': {
                'train_rmse': train_metrics['rmse'],
                'test_rmse': test_metrics['rmse'],
                'train_r2': train_metrics['r2'],
                'test_r2': test_metrics['r2'],
                'overfitting_ratio': (test_metrics['rmse'] / train_metrics['rmse']) ** 2 if train_metrics['rmse'] > 0 else float('inf'),
                'generalization_gap': test_metrics['rmse'] - train_metrics['rmse']
            },
            'monitoring_summary': self._extract_monitoring_summary(monitoring_data) if monitoring_data else {},
            'timestamp': time.time()
        }
        
        self._architecture_performance_history.append(performance_record)
        
        # Keep only recent records (last 10 architectures)
        if len(self._architecture_performance_history) > 10:
            self._architecture_performance_history = self._architecture_performance_history[-10:]
        
        return performance_record
    
    def _extract_monitoring_summary(self, monitoring_data):
        """Extract key monitoring insights for architecture comparison"""
        summary = {}
        
        if 'layer_contributions' in monitoring_data:
            layer_contribs = monitoring_data['layer_contributions']
            summary['layer_importance_ranking'] = sorted(
                [(name, contrib.get('feature_magnitude', 0)) for name, contrib in layer_contribs.items()],
                key=lambda x: x[1], reverse=True
            )
        
        if 'attention_info' in monitoring_data:
            attention_info = monitoring_data['attention_info']
            summary['attention_effectiveness'] = {}
            for layer_name, attn_data in attention_info.items():
                if 'channel_weights' in attn_data:
                    summary['attention_effectiveness'][layer_name] = attn_data['channel_weights']['std']
        
        return summary
    
    def get_architecture_optimization_recommendations(self, current_performance=None):
        """🔧 Generate specific architectural optimization recommendations
        
        Based on accumulated performance data and current monitoring.
        """
        recommendations = {
            'immediate_actions': [],
            'experimental_changes': [],
            'performance_insights': {}
        }
        
        # Analyze current architecture efficiency
        efficiency_analysis = self.compare_architecture_efficiency()
        recommendations['immediate_actions'].extend(efficiency_analysis['recommendations'])
        
        # Performance-based recommendations
        if current_performance:
            overfitting_ratio = current_performance.get('overfitting_ratio', 1.0)
            test_rmse = current_performance.get('test_rmse', 0.5)
            
            if overfitting_ratio > 1.5:
                recommendations['immediate_actions'].append(
                    f"⚠️ High overfitting (ratio: {overfitting_ratio:.2f}). Reduce regressor capacity or increase regularization."
                )
            
            if test_rmse > 0.15:
                recommendations['experimental_changes'].extend([
                    "🔬 Consider deeper CNN backbone (add layer4 or increase channel capacity)",
                    "🎯 Experiment with different regressor architectures (e.g., residual connections in regressor)"
                ])
            elif test_rmse < 0.10:
                recommendations['performance_insights']['status'] = "🎉 Excellent performance! Current architecture is well-optimized."
        
        # Historical comparison (if available)
        if hasattr(self, '_architecture_performance_history') and len(self._architecture_performance_history) > 1:
            history = self._architecture_performance_history
            recent_performance = [record['performance']['test_rmse'] for record in history[-3:]]
            
            if len(recent_performance) >= 2:
                performance_trend = recent_performance[-1] - recent_performance[0]
                if performance_trend > 0.01:
                    recommendations['experimental_changes'].append(
                        "📈 Performance declining. Consider reverting to previous architecture or trying different approach."
                    )
                elif performance_trend < -0.005:
                    recommendations['performance_insights']['trend'] = "📈 Performance improving. Continue in current direction."
        
        return recommendations

