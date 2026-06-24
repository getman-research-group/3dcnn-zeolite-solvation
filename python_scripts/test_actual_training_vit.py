#!/usr/bin/env python3
"""
测试实际训练场景下的VIT数据显示
模拟真实训练过程中的监控数据结构
"""

import torch
import numpy as np
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_3d_cnn_2_4 import CNN3DTrainer

def test_actual_training_vit_display():
    """模拟实际训练过程中VIT数据的监控显示"""
    print("🧪 TESTING ACTUAL TRAINING VIT DATA DISPLAY")
    print("="*60)
    
    # 创建trainer实例用于测试监控功能
    trainer = CNN3DTrainer(
        zeolite_types=['FAU'],
        adsorbates_by_env={'water_pure-hydrophilic': ['methanol']},
        epochs=1,
        batch_size=2,
        learning_rate=0.001,
        verbose=True,
        auto_run=False
    )
    
    # 模拟实际训练中的监控数据结构（如line 959中构建的）
    print("📊 Creating monitoring data with actual training structure...")
    
    # 这是实际训练时的数据结构
    monitoring_data = {
        'epoch': 2,
        'train_loss': 0.156,
        'test_loss': 0.187,
        'overfitting_ratio': 1.2,
        'avg_grad_norm': 0.45,
        'gradient_norms': {'layer1.weight': 0.23, 'layer2.weight': 0.67},
        'current_weight_norms': {'adsorbate_branch.0': 5.2, 'solvent_branch.0': 6.1},
        
        # 🔥 关键：实际训练时VIT数据存储在这里 (line 959)
        'vit_attention_analysis': {
            'attention_layers': {
                'layer_1': {
                    'mean_attention': 0.037037,
                    'attention_entropy': 88.688,
                    'head_diversity': 0.0680,
                    'saturation_ratio': 0.0000,
                    'attention_health': 'needs_attention'
                },
                'layer_2': {
                    'mean_attention': 0.037037,
                    'attention_entropy': 88.754,
                    'head_diversity': 0.0894,
                    'saturation_ratio': 0.0000,
                    'attention_health': 'needs_attention'
                }
            },
            'pooling_analysis': {
                'learned_strategy': 'avg_dominant',
                'avg_ratio': 0.72,
                'max_ratio': 0.28
            },
            'overall_health': {
                'status': 'needs_optimization',
                'issues': ['low_head_diversity', 'weak_attention'],
                'recommendations': ['Increase temperature', 'Add attention dropout']
            }
        },
        
        # 其他实际训练中的激活统计
        'adsorbate_branch_mean': 0.065,
        'adsorbate_branch_std': 0.526,
        'adsorbate_branch_zeros_pct': 8.3,
        'solvent_branch_mean': 0.133,
        'solvent_branch_std': 0.478,
        'solvent_branch_zeros_pct': 4.2,
        'vision_transformer_mean': 0.936,
        'vision_transformer_std': 0.526,
        'vision_transformer_zeros_pct': 0.0,
        'vision_transformer_temperature': 0.2,
        
        # 其他VIT指标
        'vit_head_diversity': 0.077,
        'vit_saturation_ratio': 0.0,
        'vit_attention_entropy': 88.7,
    }
    
    print("✅ Monitoring data created with actual training structure")
    print(f"   - vit_attention_analysis included: {len(monitoring_data['vit_attention_analysis'])} fields")
    print(f"   - attention_layers: {list(monitoring_data['vit_attention_analysis']['attention_layers'].keys())}")
    
    # 测试监控输出
    print("\n" + "="*80)
    print("📊 ACTUAL TRAINING MONITORING OUTPUT:")
    print("="*80)
    
    # 调用监控函数 - 应该能够正确显示VIT数据
    trainer._print_monitoring_summary(monitoring_data, epoch=2, batch_idx=0, model=None)
    
    print("="*80)
    print("✅ Test completed!")
    print("\n💡 Expected result: Vision Transformer Self-Attention data should be displayed")
    print("   from the 'vit_attention_analysis' field (actual training structure)")

if __name__ == "__main__":
    test_actual_training_vit_display()
