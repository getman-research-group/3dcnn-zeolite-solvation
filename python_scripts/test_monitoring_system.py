#!/usr/bin/env python3
"""
Simple Monitoring System Test Script
Direct test of _print_monitoring_summary with real data
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pickle
from core.path import get_paths

# Import the model and training classes
from model_3d_cnn_2_4 import AttentionCNNTransformer_2_4
from train_3d_cnn_2_4 import CNN3DTrainer

def test_vit_attention_collection(model, X_sample):
    """Test VIT attention data collection"""
    print("\n🔬 Testing VIT attention collection...")
    
    # Forward pass to collect attention data
    model.eval()
    with torch.no_grad():
        result = model.forward(X_sample, monitor_activations=True, detailed_analysis=True)
        if isinstance(result, tuple):
            outputs, activation_stats = result
        else:
            outputs = result
            activation_stats = {}
    
    # Get VIT attention summary
    try:
        if hasattr(model, 'vision_transformer'):
            vit_summary = model.vision_transformer.get_attention_summary()
            if isinstance(vit_summary, dict) and 'attention_layers' in vit_summary:
                print(f"✅ VIT attention data collected successfully")
                return vit_summary
            else:
                print(f"⚠️ VIT summary format unexpected: {type(vit_summary)}")
                return vit_summary
        else:
            print("⚠️ No vision_transformer in model")
            return None
    except Exception as e:
        print(f"❌ Error getting VIT summary: {e}")
        return None
    
    return vit_summary if 'vit_summary' in locals() else None

def main():
    """Simple test: load real data, create model, run training step, print monitoring."""
    print("🧪 SIMPLE MONITORING TEST")
    print("="*50)
    
    # 1. Load real data
    print("📂 Loading real voxel data...")
    dataset_dir = os.path.join(get_paths('dataset_cnn'), 'size_16.0-box_0.8-shape_20_20_20_28')
    pkl_file = 'FAU-methanol_120_water_1080-hydrophilic-01_methanol-size_16.0-box_0.8-shape_20_20_20_28.pkl'
    pkl_path = os.path.join(dataset_dir, pkl_file)
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract 4 real samples
    snapshots = data['snapshots']
    X_list, y_list = [], []
    
    for i, key in enumerate(list(snapshots.keys())[:4]):
        snapshot = snapshots[key]
        original_grid = snapshot['original_grid']  # (20, 20, 20, 28)
        interaction_energy = snapshot['target_interaction_energy']
        
        # Transpose to (28, 20, 20, 20)
        voxel_grid = np.transpose(original_grid, (3, 0, 1, 2))
        X_list.append(voxel_grid.astype(np.float32))
        y_list.append(float(interaction_energy))
    
    X = torch.stack([torch.from_numpy(x) for x in X_list])
    y = torch.tensor(y_list, dtype=torch.float32)
    
    print(f"✅ Loaded real data: X={X.shape}, y={y.shape}")
    print(f"   Energies: {y.tolist()}")
    
    # 2. Create model
    print("\n🤖 Creating model...")
    feature_names = [f'ads_feat_{i:02d}' for i in range(14)] + [f'solv_feat_{i:02d}' for i in range(14)]
    
    model = AttentionCNNTransformer_2_4(
        in_channels=28,
        feature_names=feature_names,
        use_transformer=True,
        dropout_rate=0.35,
        transformer_dim=128,
        transformer_heads=4,
        transformer_layers=2
    )
    print("✅ Model created")
    
    # 3. Create trainer and test enhanced monitoring
    print("\n🔧 Creating trainer...")
    trainer = CNN3DTrainer(
        zeolite_types=['FAU'],
        adsorbates_by_env={'water_pure-hydrophilic': ['methanol']},
        epochs=1,
        batch_size=2,
        learning_rate=0.001,
        verbose=True,
        auto_run=False,
        model_type='transformer'  # 🎯 明确指定使用transformer模型
    )
    trainer.device = 'cpu'
    print("✅ Trainer created")
    
    # 🔥 NEW: Test model's attention capabilities before training
    print("\n🔍 Testing model attention capabilities...")
    print(f"  Model has Vision Transformer: {hasattr(model, 'vision_transformer')}")
    if hasattr(model, 'vision_transformer'):
        print(f"  VIT transformer layers: {len(model.vision_transformer.transformer_layers)}")
        print(f"  VIT has attention maps: {hasattr(model.vision_transformer, 'attention_maps')}")
        
        # Initialize attention maps if needed
        if not hasattr(model.vision_transformer, 'attention_maps'):
            model.vision_transformer.attention_maps = []
        if not hasattr(model.vision_transformer, 'cross_attention_maps'):
            model.vision_transformer.cross_attention_maps = []
    
    print(f"  Model get_attention_summary method: {hasattr(model, 'get_attention_summary')}")
    print(f"  Model attention_weights_history: {hasattr(model, 'attention_weights_history')}")
    
    # Initialize attention history if needed
    if not hasattr(model, 'attention_weights_history'):
        model.attention_weights_history = []
    
    # 4. Simulate multiple training steps to build attention history
    print("\n🔥 Simulating multiple training steps to build attention history...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # Step 1: Initial forward pass to establish baseline
    optimizer.zero_grad()
    result1 = model.forward(X[:2], monitor_activations=True, detailed_analysis=True)
    if isinstance(result1, tuple):
        outputs1, activation_stats1 = result1
    else:
        outputs1 = result1
        activation_stats1 = {}
    
    targets = y[:2].unsqueeze(1)
    loss1 = criterion(outputs1, targets)
    loss1.backward()
    optimizer.step()
    
    print(f"  ✅ Step 1 completed: loss={loss1.item():.4f}")
    
    # Step 2: Second forward pass to create evolution data
    optimizer.zero_grad()
    result2 = model.forward(X[2:4], monitor_activations=True, detailed_analysis=True)
    if isinstance(result2, tuple):
        outputs2, activation_stats2 = result2
    else:
        outputs2 = result2
        activation_stats2 = {}
    
    targets2 = y[2:4].unsqueeze(1)
    loss2 = criterion(outputs2, targets2)
    loss2.backward()
    
    print(f"  ✅ Step 2 completed: loss={loss2.item():.4f}")
    
    # Use the final activation stats and loss
    final_loss = loss2
    activation_stats = activation_stats2
    
    # Calculate gradients
    gradient_norms = {}
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            gradient_norms[name] = param_norm.item()
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    # Get weight norms
    layer_weight_norms = model.get_layer_weight_norms() if hasattr(model, 'get_layer_weight_norms') else {}
    
    print(f"✅ Training steps completed: final_loss={final_loss.item():.4f}")
    
    # Detailed VIT attention testing
    vit_summary = test_vit_attention_collection(model, X[:1])
    
    # 5. Prepare comprehensive monitoring data with VIT attention analysis
    print("\n🔍 Preparing monitoring data...")
    
    # Use the VIT summary from detailed testing if available
    if vit_summary is None:
        # Force collection of VIT attention data
        if hasattr(model, 'vision_transformer') and model.vision_transformer:
            try:
                vit_attention_summary = model.vision_transformer.get_attention_summary()
                print(f"✅ VIT attention summary collected")
            except Exception as e:
                print(f"⚠️ VIT attention collection error: {e}")
                vit_attention_summary = "No attention data available"
        else:
            vit_attention_summary = "No transformer available"
    else:
        vit_attention_summary = vit_summary
        print(f"✅ Using VIT summary from detailed testing")
    
    monitoring_data = {
        'epoch': 2,  # Use epoch 2 to enable attention evolution analysis
        'train_loss': final_loss.item(),
        'test_loss': final_loss.item() * 1.2,
        'overfitting_ratio': 1.2,
        'avg_grad_norm': total_norm,
        'gradient_norms': gradient_norms,
        'current_weight_norms': layer_weight_norms,
        'vision_transformer_temperature': getattr(model.vision_transformer, 'temperature', 0.2) if hasattr(model, 'vision_transformer') else 0.2,
        
        # Add VIT attention data in the correct location
        'attention_info': {
            'vision_transformer': vit_attention_summary
        },
        
        # Also add as "Attention Analysis" (the format that's currently being displayed)
        'Attention Analysis': vit_attention_summary,
        
        **activation_stats,  # Real activation data
    }
    
    print(f"✅ Monitoring data prepared with {len(monitoring_data)} fields")
    
    # 6. Print monitoring summary
    print("\n" + "="*80)
    print("📊 MONITORING OUTPUT:")
    print("="*80)
    
    trainer._print_monitoring_summary(monitoring_data, epoch=2, batch_idx=0, model=model)
    
    print("="*80)
    print("✅ Test completed!")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()


