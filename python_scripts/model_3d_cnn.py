"""3D-CNN architecture for voxel-based interaction-energy prediction.

This module defines the attention blocks, residual blocks, and final network
used to predict adsorbate-solvent interaction energies from 28-channel voxel
representations. The input is organized as 14 adsorbate channels followed by
14 solvent channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAMChannelAttention(nn.Module):
    """
    Channel-attention block for 3D feature maps.

    The module supports the standard CBAM formulation and an optional
    adsorbate/solvent group-aware mode that applies separate channel MLPs to
    the two channel groups before combining them.
    """
    def __init__(self, in_channels, reduction_ratio=16, dropout=0.1, enable_group_interaction=False, group_split=None):
        super(CBAMChannelAttention, self).__init__()
        
        # Optional adsorbate/solvent group-aware attention.
        self.enable_group_interaction = enable_group_interaction
        self.group_split = group_split
        
        # Shared pooling operators used by CBAM channel attention.
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # Shared MLP used in the standard channel-attention path.
        reduced_channels = max(in_channels // reduction_ratio, 1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(reduced_channels, in_channels, bias=False)
        )
        
        # Group-specific MLPs used only in the group-aware attention path.
        if self.enable_group_interaction and self.group_split is not None:
            assert len(self.group_split) == 2, "Currently supports binary group split (adsorbate + solvent)"
            ads_channels, solv_channels = self.group_split
            assert ads_channels + solv_channels == in_channels, "Group split must sum to total channels"
            
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
            
            # Learnable coupling between the two channel groups.
            self.interaction_strength = nn.Parameter(torch.tensor(0.5))
        
        # Initialize linear layers conservatively for stable attention weights.
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
        """Apply channel attention to a 5D tensor ``(N, C, D, H, W)``."""
        batch_size, channels, _, _, _ = x.size()
        
        # Pool the spatial dimensions into channel descriptors.
        avg_out = self.avg_pool(x).view(batch_size, channels)
        max_out = self.max_pool(x).view(batch_size, channels)
        
        if self.enable_group_interaction and self.group_split is not None:
            ads_channels, solv_channels = self.group_split
            
            # Split pooled descriptors into adsorbate and solvent groups.
            avg_ads = avg_out[:, :ads_channels]
            avg_solv = avg_out[:, ads_channels:]
            max_ads = max_out[:, :ads_channels]
            max_solv = max_out[:, ads_channels:]
            
            # Compute group-specific attention weights.
            ads_avg_weights = self.ads_mlp(avg_ads)
            ads_max_weights = self.ads_mlp(max_ads)
            solv_avg_weights = self.solv_mlp(avg_solv)
            solv_max_weights = self.solv_mlp(max_solv)
            
            ads_weights = self.activation(ads_avg_weights + ads_max_weights)
            solv_weights = self.activation(solv_avg_weights + solv_max_weights)
            
            interaction_factor = torch.sigmoid(self.interaction_strength)
            
            # Use the adsorbate signal to modulate solvent-channel weights.
            ads_influence = ads_weights.mean(dim=1, keepdim=True)
            enhanced_solv_weights = solv_weights * (1 + interaction_factor * ads_influence)
            
            final_ads_weights = ads_weights
            final_solv_weights = enhanced_solv_weights * interaction_factor + solv_weights * (1 - interaction_factor)
            
            channel_weights = torch.cat([final_ads_weights, final_solv_weights], dim=1)
        else:
            avg_weights = self.shared_mlp(avg_out)
            max_weights = self.shared_mlp(max_out)
            channel_weights = self.activation(avg_weights + max_weights)
        
        channel_weights = channel_weights.view(batch_size, channels, 1, 1, 1)
        
        return x * channel_weights.expand_as(x)
    



class CBAMSpatialAttention(nn.Module):
    """
    Spatial-attention block for 3D feature maps.

    The implementation follows the standard CBAM design by combining
    channel-wise mean and max projections and then predicting a spatial mask.
    """
    def __init__(self, kernel_size=7):
        super(CBAMSpatialAttention, self).__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        
        # Standard CBAM spatial attention: single convolution
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """Apply spatial attention to a 5D tensor ``(N, C, D, H, W)``."""
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        
        spatial_weights = self.conv(spatial_input)
        spatial_weights = self.activation(spatial_weights)
        
        return x * spatial_weights.expand_as(x)
    


class CBAM3D(nn.Module):
    """
    Combined 3D CBAM block.

    Channel attention is applied first, followed by spatial attention. The
    block can optionally use group-aware channel attention when adsorbate and
    solvent channels should be treated separately.
    """
    def __init__(self, in_channels, reduction_ratio=6, kernel_size=5, dropout=0.1, 
                 enable_group_interaction=False, group_split=None):
        super(CBAM3D, self).__init__()
        self.channel_attention = CBAMChannelAttention(
            in_channels, reduction_ratio, dropout, 
            enable_group_interaction=enable_group_interaction, 
            group_split=group_split
        )
        self.spatial_attention = CBAMSpatialAttention(kernel_size)

    def forward(self, x):
        """Apply channel attention followed by spatial attention."""
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class ResidualBlock3D(nn.Module):
    """
    Residual 3D convolutional block used in the CNN backbone.

    Each block contains two 3D convolutions, an optional CBAM block, and a
    shortcut projection when the input and output channel counts differ.
    """
    def __init__(self, in_channels, out_channels, stride=1, use_cbam=True):
        super(ResidualBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.cbam = CBAM3D(out_channels, kernel_size=3, dropout=0.08) if use_cbam else nn.Identity()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        
        self.dropout = nn.Dropout3d(p=0.01)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """Apply the residual block to a 5D tensor ``(N, C, D, H, W)``."""
        residual = x
        
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        out = self.cbam(out)
        
        out += self.shortcut(residual)
        out = self.activation(out)
        
        return out

class AttentionCNN(nn.Module):
    """
    3D CNN for voxel-based interaction-energy prediction.

    The network uses a dual-branch front end to process adsorbate and solvent
    channels separately, merges the two streams with group-aware attention, and
    then applies a residual CNN backbone followed by a regression head.

    Expected input shape:
        ``(batch_size, 28, 20, 20, 20)``

    Channel layout:
        - channels 0-13: adsorbate features
        - channels 14-27: solvent features
    """
    def __init__(self, in_channels=28,  # 14 adsorbate + 14 solvent channels
                 dropout_rate=0.35,
                 feature_names=None):
        
        super(AttentionCNN, self).__init__()
        
        # Store configuration used by downstream analysis utilities.
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate
        
        self.feature_names = feature_names
        
        # The architecture expects matching 14-channel adsorbate and solvent groups.
        assert in_channels == 28, (
            "The model requires 28 channels: 14 adsorbate + 14 solvent"
        )
        
        # Adsorbate branch for sparse local features near the adsorption site.
        self.adsorbate_processor = nn.Sequential(
            nn.Conv3d(14, 20, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(20),
            nn.ReLU(inplace=True),
            
            CBAM3D(20, reduction_ratio=8, kernel_size=3, dropout=0.05),
            
            nn.Conv3d(20, 26, kernel_size=1, bias=False),
            nn.BatchNorm3d(26),
            nn.ReLU(inplace=True),
            nn.Conv3d(26, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.06)
        )
        
        # Solvent branch for denser distributed features across the voxel box.
        self.solvent_processor = nn.Sequential(
            nn.Conv3d(14, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            CBAM3D(32, reduction_ratio=6, kernel_size=5, dropout=0.04),
            
            nn.Conv3d(32, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.04)
        )
        
        # Group-aware attention at the adsorbate/solvent fusion point.
        self.interaction_attention = CBAM3D(
            in_channels=80,
            kernel_size=5,
            dropout=0.08, 
            enable_group_interaction=True, 
            group_split=[32, 48]
        )
        
        # Interaction fusion after attention-guided branch combination.
        self.interaction_conv = nn.Sequential(
            nn.Conv3d(80, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 48, kernel_size=1, bias=False),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True)
        )
        
        # Residual CNN backbone.
        self.layer1 = ResidualBlock3D(48, 52, stride=1, use_cbam=False)
        self.pool1 = nn.AvgPool3d(2)
        
        self.layer2 = ResidualBlock3D(52, 58, stride=1, use_cbam=False)
        self.pool2 = nn.AvgPool3d(2)
        
        self.layer3 = ResidualBlock3D(58, 64, stride=1, use_cbam=False)
        
        # Final spatial compression before the regression head.
        self.adaptive_pool = nn.AdaptiveAvgPool3d(2)
        
        regressor_input_dim = 64 * 2 * 2 * 2

        # Regression head for the final scalar energy prediction.
        self.regressor = nn.Sequential(
            nn.Flatten(),
            
            nn.Linear(regressor_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            
            nn.Linear(64, 1)
        )
        
        # Initialize convolutional, normalization, and linear layers.
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable 3D-CNN training."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu', a=0.05)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 0.9)
                nn.init.constant_(m.bias, 0)
                m.momentum = 0.001
                m.eps = 5e-3
                if hasattr(m, 'running_var'):
                    m.running_var.fill_(1.0)
            elif isinstance(m, nn.Linear):
                if m.weight.size(0) > 1000:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Predict interaction energy from a 28-channel voxel tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(batch_size, 28, 20, 20, 20)``.

        Returns
        -------
        torch.Tensor
            Predicted interaction energies with shape ``(batch_size, 1)``.
        """
            
        # Split the input into adsorbate and solvent channel groups.
        adsorbate_channels = x[:, :14, :, :, :]
        solvent_channels = x[:, 14:, :, :, :]
        
        # Process the two molecular groups with separate front-end branches.
        adsorbate_features = self.adsorbate_processor(adsorbate_channels)
        
        solvent_features = self.solvent_processor(solvent_channels)
        
        combined_features = torch.cat([adsorbate_features, solvent_features], dim=1)
        
        # Apply group-aware attention before branch fusion.
        attended_features = self.interaction_attention(combined_features)
        
        interaction_features = self.interaction_conv(attended_features)
            
        x = interaction_features
        
        x = self.layer1(x)
        x = self.pool1(x)
        
        x = self.layer2(x)
        x = self.pool2(x)
        
        x = self.layer3(x)
        
        x = self.adaptive_pool(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.regressor(x)
        
        return x


# Backward-compatible alias retained for scripts that still import the legacy name.
AttentionCNN_2_8 = AttentionCNN
    
