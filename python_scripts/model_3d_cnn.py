"""
model_3d_cnn_2_8.py


"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        else:
            # Standard CBAM processing
            avg_weights = self.shared_mlp(avg_out)
            max_weights = self.shared_mlp(max_out)
            channel_weights = self.activation(avg_weights + max_weights)
        
        # Reshape for broadcasting
        channel_weights = channel_weights.view(batch_size, channels, 1, 1, 1)
        
        return x * channel_weights.expand_as(x)
    



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

    def forward(self, x):
        # Standard CBAM spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Channel-wise average pooling
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Channel-wise max pooling
        
        # Concatenate along channel dimension
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        
        # Generate spatial attention weights
        spatial_weights = self.conv(spatial_input)
        spatial_weights = self.activation(spatial_weights)
        
        # Apply attention
        return x * spatial_weights.expand_as(x)
    


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

    def forward(self, x):
        # Apply channel attention first (feature importance), then spatial attention (location importance)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

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
    Accepts 28 input channels (14 atomic features × 2 molecular groups).
    
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
    
    Input representation:
    - Default 28 input channels (14 atomic features × 2 groups: adsorbate + solvent)
    - Supports dynamic channel numbers via in_channels parameter
    - Enhanced feature analysis for separated channel groups
    """
    def __init__(self, in_channels=28,  # 14 adsorbate + 14 solvent channels
                 dropout_rate=0.35,     # Increased from 0.25 based on overfitting analysis
                 feature_names=None):   # Add feature_names parameter
        
        super(AttentionCNN_2_8, self).__init__()
        
        # Store configuration including input channels for dynamic feature analysis
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate  # Store for analysis methods
        
        # Store feature names for analysis methods
        self.feature_names = feature_names
        
        # The architecture expects matching 14-channel adsorbate and solvent groups.
        assert in_channels == 28, (
            "The model requires 28 channels: 14 adsorbate + 14 solvent"
        )
        
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

    def forward(self, x):
        """
        Forward pass for adsorbate-solvent interaction energy prediction
        
        Args:
            x: Input voxel grid with shape (batch_size, 28, 20, 20, 20)
        
        Returns:
            Predicted interaction energy (batch_size, 1)
        """
            
        # Dual-branch processing for separated adsorbate and solvent channels.
        # Separate adsorbate and solvent channel groups
        adsorbate_channels = x[:, :14, :, :, :]   # First 14 channels: adsorbate features
        solvent_channels = x[:, 14:, :, :, :]     # Last 14 channels: solvent features
        
        # Process adsorbate branch (sparse, central features)
        adsorbate_features = self.adsorbate_processor(adsorbate_channels)  # (batch, 32, 20, 20, 20)
        
        # Process solvent branch (dense, distributed features)
        solvent_features = self.solvent_processor(solvent_channels)       # (batch, 48, 20, 20, 20)
        
        # Natural combination without artificial weighting
        # The architecture now balances adsorbate (32 channels) and solvent (48 channels) more evenly
        combined_features = torch.cat([adsorbate_features, solvent_features], dim=1)  # (batch, 80, 20, 20, 20)
        
        # 🎯 OPTIMAL ATTENTION: Apply group-aware attention BEFORE convolution mixing
        # This preserves the physical meaning of channel groups (32 adsorbate + 48 solvent)
        attended_features = self.interaction_attention(combined_features)  # (batch, 80, 20, 20, 20)
        
        # Model adsorbate-solvent interactions on the attended features
        interaction_features = self.interaction_conv(attended_features)   # (batch, 48, 20, 20, 20)
            
        # Continue with the enhanced CNN backbone
        x = interaction_features  # Start main CNN pipeline with interaction features
        
        # Progressive feature learning
        x = self.layer1(x)  # (batch, 48, 20, 20, 20) -> (batch, 64, 20, 20, 20)
        x = self.pool1(x)  # (batch, 64, 10, 10, 10)
        
        x = self.layer2(x)  # (batch, 64, 10, 10, 10) -> (batch, 80, 10, 10, 10)
        x = self.pool2(x)  # (batch, 80, 5, 5, 5)
        
        # Layer3 for deeper feature extraction
        x = self.layer3(x)  # (batch, 56, 5, 5, 5) -> (batch, 64, 5, 5, 5)
        
        # Aggressive spatial compression: 5×5×5 → 2×2×2 for parameter reduction
        x = self.adaptive_pool(x)  # (batch, 64, 2, 2, 2)
        
        # Flatten for regressor
        x = x.view(x.size(0), -1)  # (batch, 64*2*2*2 = 512)
        
        # Final prediction
        x = self.regressor(x)  # (batch, 1)
        
        return x
    
