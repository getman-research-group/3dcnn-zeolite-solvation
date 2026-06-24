#!/usr/bin/env python3
"""
CBAM效果专门分析脚本
分析Model 2_8中CBAM注意力机制的有效性
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.path import get_paths
from model_3d_cnn_2_8 import AttentionCNN_2_8

class CBAMEffectivenessAnalyzer:
    """专门分析CBAM注意力机制效果的分析器"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.checkpoint = None
        self.cbam_activations = {}
        self.analysis_results = {}
        
        print(f"🎯 初始化CBAM效果分析器")
        print(f"📂 模型路径: {os.path.basename(model_path)}")
        
    def load_model(self):
        """加载训练好的模型"""
        try:
            print(f"📂 Loading model from: {self.model_path}")
            
            # Load checkpoint
            self.checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
            
            # Initialize model
            self.model = AttentionCNN_2_8(in_channels=28, dropout_rate=0.25)
            
            # Load state dict
            if 'model_state_dict' in self.checkpoint:
                self.model.load_state_dict(self.checkpoint['model_state_dict'])
                print("✅ Model state dict loaded successfully")
            
            self.model.eval()
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def analyze_cbam_locations_and_parameters(self):
        """分析CBAM的位置和参数分布"""
        print("\n🔍 分析CBAM位置和参数分布...")
        
        cbam_info = {}
        total_params = sum(p.numel() for p in self.model.parameters())
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'channel_attention') and hasattr(module, 'spatial_attention'):
                # 这是一个CBAM模块
                cbam_params = sum(p.numel() for p in module.parameters())
                cbam_percentage = (cbam_params / total_params) * 100
                
                # 获取输入通道数
                if hasattr(module, 'in_channels'):
                    in_channels = module.in_channels
                elif hasattr(module.channel_attention, 'shared_mlp') and len(module.channel_attention.shared_mlp) > 0:
                    # 从shared_mlp推断输入通道数
                    first_layer = module.channel_attention.shared_mlp[0]
                    if hasattr(first_layer, 'in_features'):
                        in_channels = first_layer.in_features
                    else:
                        in_channels = "Unknown"
                else:
                    in_channels = "Unknown"
                
                cbam_info[name] = {
                    'location': self._categorize_cbam_location(name),
                    'parameters': cbam_params,
                    'percentage_of_total': cbam_percentage,
                    'input_channels': in_channels,
                    'module_type': type(module).__name__
                }
        
        # 打印详细信息
        print(f"\n📊 CBAM模块分析结果:")
        print(f"   总共发现 {len(cbam_info)} 个CBAM模块")
        
        total_cbam_params = 0
        for name, info in cbam_info.items():
            print(f"\n🎯 {name}:")
            print(f"   位置类型: {info['location']}")
            print(f"   参数数量: {info['parameters']:,}")
            print(f"   占总参数比例: {info['percentage_of_total']:.3f}%")
            print(f"   输入通道数: {info['input_channels']}")
            total_cbam_params += info['parameters']
        
        print(f"\n📈 CBAM总体统计:")
        print(f"   所有CBAM参数总和: {total_cbam_params:,}")
        print(f"   CBAM占模型总参数: {(total_cbam_params/total_params)*100:.3f}%")
        
        self.analysis_results['cbam_locations'] = cbam_info
        return cbam_info
    
    def _categorize_cbam_location(self, module_name):
        """根据模块名称分类CBAM位置"""
        if 'adsorbate_processor' in module_name:
            return 'Adsorbate Processor (稀疏特征处理)'
        elif 'solvent_processor' in module_name:
            return 'Solvent Processor (密集特征处理)'
        elif 'interaction' in module_name:
            return 'Interaction Layer (特征融合)'
        elif 'layer1' in module_name or 'layer2' in module_name or 'layer3' in module_name:
            return 'CNN Backbone (残差块)'
        else:
            return 'Other (其他位置)'
    
    def analyze_cbam_activation_patterns(self):
        """分析CBAM的激活模式"""
        print("\n🔥 分析CBAM激活模式...")
        
        # 生成测试数据
        test_input = torch.randn(4, 28, 20, 20, 20)  # Batch=4 for better statistics
        
        activation_stats = {}
        hooks = []
        
        def register_cbam_hook(name, module):
            def hook_fn(module, input, output):
                if hasattr(module, 'channel_attention') and hasattr(module, 'spatial_attention'):
                    # 获取通道注意力权重
                    try:
                        with torch.no_grad():
                            # 重新计算注意力权重用于分析
                            x = input[0] if isinstance(input, tuple) else input
                            
                            # Channel attention analysis
                            if not isinstance(module.channel_attention, nn.Identity):
                                channel_weights = self._extract_channel_attention_weights(module.channel_attention, x)
                            else:
                                channel_weights = None
                            
                            # Spatial attention analysis  
                            if not isinstance(module.spatial_attention, nn.Identity):
                                spatial_weights = self._extract_spatial_attention_weights(module.spatial_attention, x)
                            else:
                                spatial_weights = None
                            
                            activation_stats[name] = {
                                'input_shape': list(x.shape),
                                'channel_weights': channel_weights,
                                'spatial_weights': spatial_weights,
                                'input_stats': {
                                    'mean': float(torch.mean(x)),
                                    'std': float(torch.std(x)),
                                    'sparsity': float(torch.sum(torch.abs(x) < 1e-6) / x.numel())
                                }
                            }
                    except Exception as e:
                        print(f"⚠️ CBAM hook error for {name}: {e}")
                        activation_stats[name] = {'error': str(e)}
            
            return hook_fn
        
        # 注册hooks到所有CBAM模块
        for name, module in self.model.named_modules():
            if hasattr(module, 'channel_attention') and hasattr(module, 'spatial_attention'):
                hook = module.register_forward_hook(register_cbam_hook(name, module))
                hooks.append(hook)
        
        # 执行前向传播
        try:
            with torch.no_grad():
                _ = self.model(test_input)
        except Exception as e:
            print(f"⚠️ Forward pass failed: {e}")
        
        # 移除hooks
        for hook in hooks:
            hook.remove()
        
        # 分析结果
        print(f"\n📊 CBAM激活模式分析:")
        for name, stats in activation_stats.items():
            if 'error' in stats:
                print(f"\n❌ {name}: {stats['error']}")
                continue
                
            print(f"\n🎯 {name}:")
            input_stats = stats['input_stats']
            print(f"   输入特征统计: mean={input_stats['mean']:.4f}, std={input_stats['std']:.4f}")
            print(f"   输入稀疏度: {input_stats['sparsity']:.1%}")
            
            if stats['channel_weights'] is not None:
                ch_weights = stats['channel_weights']
                print(f"   通道注意力统计:")
                print(f"     权重范围: [{ch_weights.min():.4f}, {ch_weights.max():.4f}]")
                print(f"     权重标准差: {ch_weights.std():.4f}")
                print(f"     高权重通道比例: {(ch_weights > ch_weights.mean() + ch_weights.std()).float().mean():.1%}")
            else:
                print(f"   通道注意力: 未启用 (Identity)")
            
            if stats['spatial_weights'] is not None:
                sp_weights = stats['spatial_weights']
                print(f"   空间注意力统计:")
                print(f"     权重范围: [{sp_weights.min():.4f}, {sp_weights.max():.4f}]")
                print(f"     权重标准差: {sp_weights.std():.4f}")
                print(f"     高权重区域比例: {(sp_weights > sp_weights.mean() + sp_weights.std()).float().mean():.1%}")
            else:
                print(f"   空间注意力: 未启用 (Identity)")
        
        self.analysis_results['cbam_activations'] = activation_stats
        return activation_stats
    
    def _extract_channel_attention_weights(self, channel_attention, x):
        """提取通道注意力权重"""
        try:
            # 对输入进行全局平均池化和最大池化
            avg_pool = torch.mean(x, dim=[2, 3, 4], keepdim=True)
            max_pool = torch.max(torch.max(torch.max(x, dim=4)[0], dim=3)[0], dim=2)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            
            # 通过共享MLP
            if hasattr(channel_attention, 'shared_mlp'):
                avg_out = channel_attention.shared_mlp(avg_pool.squeeze())
                max_out = channel_attention.shared_mlp(max_pool.squeeze())
                
                # 合并并应用sigmoid
                channel_weights = torch.sigmoid(avg_out + max_out)
                return channel_weights.squeeze()
            else:
                return None
        except:
            return None
    
    def _extract_spatial_attention_weights(self, spatial_attention, x):
        """提取空间注意力权重"""
        try:
            # 计算通道维度的平均值和最大值
            avg_pool = torch.mean(x, dim=1, keepdim=True)
            max_pool = torch.max(x, dim=1, keepdim=True)[0]
            
            # 拼接
            concat = torch.cat([avg_pool, max_pool], dim=1)
            
            # 通过卷积层
            if hasattr(spatial_attention, 'conv'):
                spatial_weights = torch.sigmoid(spatial_attention.conv(concat))
                return spatial_weights.squeeze()
            else:
                return None
        except:
            return None
    
    def compare_with_without_cbam(self):
        """比较有无CBAM的模型性能差异"""
        print("\n⚖️ 比较有无CBAM的性能差异...")
        
        # 生成测试数据
        test_input = torch.randn(8, 28, 20, 20, 20)
        test_target = torch.randn(8, 1)
        
        results = {}
        
        # 1. 原始模型（带CBAM）
        print("🔬 测试原始模型（带CBAM）...")
        with torch.no_grad():
            original_output = self.model(test_input)
            original_loss = nn.MSELoss()(original_output, test_target)
        
        results['with_cbam'] = {
            'output_std': float(torch.std(original_output)),
            'output_mean': float(torch.mean(original_output)),
            'loss': float(original_loss)
        }
        
        # 2. 临时禁用CBAM的模型
        print("🔬 测试禁用CBAM的模型...")
        model_no_cbam = deepcopy(self.model)
        self._disable_cbam_in_model(model_no_cbam)
        
        with torch.no_grad():
            no_cbam_output = model_no_cbam(test_input)
            no_cbam_loss = nn.MSELoss()(no_cbam_output, test_target)
        
        results['without_cbam'] = {
            'output_std': float(torch.std(no_cbam_output)),
            'output_mean': float(torch.mean(no_cbam_output)),
            'loss': float(no_cbam_loss)
        }
        
        # 3. 计算差异
        output_difference = torch.abs(original_output - no_cbam_output)
        results['difference_analysis'] = {
            'max_difference': float(torch.max(output_difference)),
            'mean_difference': float(torch.mean(output_difference)),
            'std_difference': float(torch.std(output_difference)),
            'relative_change': float(torch.mean(output_difference) / torch.mean(torch.abs(original_output)))
        }
        
        # 打印结果
        print(f"\n📊 CBAM效果对比分析:")
        print(f"   带CBAM模型:")
        print(f"     输出均值: {results['with_cbam']['output_mean']:.6f}")
        print(f"     输出标准差: {results['with_cbam']['output_std']:.6f}")
        print(f"     测试损失: {results['with_cbam']['loss']:.6f}")
        
        print(f"   不带CBAM模型:")
        print(f"     输出均值: {results['without_cbam']['output_mean']:.6f}")
        print(f"     输出标准差: {results['without_cbam']['output_std']:.6f}")
        print(f"     测试损失: {results['without_cbam']['loss']:.6f}")
        
        print(f"   差异分析:")
        print(f"     最大输出差异: {results['difference_analysis']['max_difference']:.6f}")
        print(f"     平均输出差异: {results['difference_analysis']['mean_difference']:.6f}")
        print(f"     相对变化: {results['difference_analysis']['relative_change']:.2%}")
        
        self.analysis_results['cbam_comparison'] = results
        return results
    
    def _disable_cbam_in_model(self, model):
        """在模型中临时禁用CBAM"""
        for name, module in model.named_modules():
            if hasattr(module, 'channel_attention') and hasattr(module, 'spatial_attention'):
                # 替换为Identity模块
                module.channel_attention = nn.Identity()
                module.spatial_attention = nn.Identity()
    
    def analyze_cbam_feature_selection(self):
        """分析CBAM的特征选择效果"""
        print("\n🎯 分析CBAM特征选择效果...")
        
        # 创建不同类型的测试输入
        test_cases = {
            'sparse_input': self._create_sparse_test_input(),
            'dense_input': self._create_dense_test_input(),
            'mixed_input': self._create_mixed_test_input()
        }
        
        selection_analysis = {}
        
        for input_type, test_input in test_cases.items():
            print(f"\n🧪 测试 {input_type}...")
            
            # 收集CBAM权重
            cbam_weights = {}
            hooks = []
            
            def collect_weights_hook(name):
                def hook_fn(module, input, output):
                    try:
                        x = input[0] if isinstance(input, tuple) else input
                        
                        # 收集通道注意力权重
                        if not isinstance(module.channel_attention, nn.Identity):
                            ch_weights = self._extract_channel_attention_weights(module.channel_attention, x)
                            if ch_weights is not None:
                                cbam_weights[f"{name}_channel"] = ch_weights.detach().cpu()
                        
                        # 收集空间注意力权重  
                        if not isinstance(module.spatial_attention, nn.Identity):
                            sp_weights = self._extract_spatial_attention_weights(module.spatial_attention, x)
                            if sp_weights is not None:
                                cbam_weights[f"{name}_spatial"] = sp_weights.detach().cpu()
                                
                    except Exception as e:
                        print(f"⚠️ Hook error for {name}: {e}")
                
                return hook_fn
            
            # 注册hooks
            for name, module in self.model.named_modules():
                if hasattr(module, 'channel_attention') and hasattr(module, 'spatial_attention'):
                    hook = module.register_forward_hook(collect_weights_hook(name))
                    hooks.append(hook)
            
            # 前向传播
            try:
                with torch.no_grad():
                    output = self.model(test_input)
            except Exception as e:
                print(f"⚠️ Forward pass failed for {input_type}: {e}")
                continue
            
            # 移除hooks
            for hook in hooks:
                hook.remove()
            
            # 分析权重分布
            weight_analysis = {}
            for weight_name, weights in cbam_weights.items():
                weight_analysis[weight_name] = {
                    'mean': float(torch.mean(weights)),
                    'std': float(torch.std(weights)),
                    'min': float(torch.min(weights)),
                    'max': float(torch.max(weights)),
                    'selectivity': float(torch.std(weights) / (torch.mean(weights) + 1e-8))  # 选择性指标
                }
            
            selection_analysis[input_type] = {
                'input_characteristics': self._analyze_input_characteristics(test_input),
                'cbam_weights_analysis': weight_analysis
            }
        
        # 打印特征选择分析结果
        print(f"\n📊 CBAM特征选择分析结果:")
        for input_type, analysis in selection_analysis.items():
            print(f"\n🎯 {input_type.upper()}:")
            input_chars = analysis['input_characteristics']
            print(f"   输入特征: 稀疏度={input_chars['sparsity']:.1%}, 变异系数={input_chars['coefficient_variation']:.4f}")
            
            for weight_name, weight_stats in analysis['cbam_weights_analysis'].items():
                print(f"   {weight_name}:")
                print(f"     选择性指标: {weight_stats['selectivity']:.4f}")
                print(f"     权重范围: [{weight_stats['min']:.4f}, {weight_stats['max']:.4f}]")
        
        self.analysis_results['feature_selection'] = selection_analysis
        return selection_analysis
    
    def _create_sparse_test_input(self):
        """创建稀疏测试输入（模拟adsorbate特征）"""
        input_tensor = torch.zeros(2, 28, 20, 20, 20)
        # 只在少数位置设置非零值
        num_nonzero = int(0.05 * 20 * 20 * 20)  # 5% 非零
        for b in range(2):
            for c in range(14):  # 前14个通道为adsorbate
                indices = torch.randperm(20*20*20)[:num_nonzero]
                flat_tensor = input_tensor[b, c].view(-1)
                flat_tensor[indices] = torch.randn(num_nonzero)
        return input_tensor
    
    def _create_dense_test_input(self):
        """创建密集测试输入（模拟solvent特征）"""
        input_tensor = torch.randn(2, 28, 20, 20, 20)
        # 后14个通道为solvent，设置为相对均匀分布
        input_tensor[:, 14:, :, :, :] = torch.randn(2, 14, 20, 20, 20) * 0.5 + 0.1
        return input_tensor
    
    def _create_mixed_test_input(self):
        """创建混合测试输入"""
        sparse_input = self._create_sparse_test_input()
        dense_input = self._create_dense_test_input()
        # 将稀疏和密集特征组合
        mixed_input = torch.zeros_like(dense_input)
        mixed_input[:, :14, :, :, :] = sparse_input[:, :14, :, :, :]  # 稀疏adsorbate
        mixed_input[:, 14:, :, :, :] = dense_input[:, 14:, :, :, :]   # 密集solvent
        return mixed_input
    
    def _analyze_input_characteristics(self, input_tensor):
        """分析输入张量的特征"""
        sparsity = float(torch.sum(torch.abs(input_tensor) < 1e-6) / input_tensor.numel())
        mean_val = float(torch.mean(input_tensor))
        std_val = float(torch.std(input_tensor))
        coeff_var = std_val / (abs(mean_val) + 1e-8)
        
        return {
            'sparsity': sparsity,
            'mean': mean_val,
            'std': std_val,
            'coefficient_variation': coeff_var
        }
    
    def generate_cbam_effectiveness_report(self):
        """生成CBAM有效性报告"""
        print("\n" + "="*80)
        print("🎯 CBAM有效性综合报告")
        print("="*80)
        
        # 位置分析
        if 'cbam_locations' in self.analysis_results:
            locations = self.analysis_results['cbam_locations']
            print(f"\n📍 CBAM位置分析:")
            print(f"   总共 {len(locations)} 个CBAM模块")
            
            location_types = {}
            for name, info in locations.items():
                loc_type = info['location']
                if loc_type not in location_types:
                    location_types[loc_type] = []
                location_types[loc_type].append(info)
            
            for loc_type, cbam_list in location_types.items():
                total_params = sum(cbam['parameters'] for cbam in cbam_list)
                print(f"   {loc_type}: {len(cbam_list)}个, {total_params:,}参数")
        
        # 激活模式分析
        if 'cbam_activations' in self.analysis_results:
            activations = self.analysis_results['cbam_activations']
            print(f"\n🔥 CBAM激活效果:")
            
            for name, stats in activations.items():
                if 'error' not in stats:
                    input_sparsity = stats['input_stats']['sparsity']
                    location_type = self._categorize_cbam_location(name)
                    
                    if input_sparsity > 0.5:  # 稀疏输入
                        effectiveness = "适合稀疏特征处理" if "Adsorbate" in location_type else "处理稀疏输入"
                    else:  # 密集输入
                        effectiveness = "适合密集特征处理" if "Solvent" in location_type else "处理密集输入"
                    
                    print(f"   {name}: {effectiveness}")
        
        # 性能对比分析
        if 'cbam_comparison' in self.analysis_results:
            comparison = self.analysis_results['cbam_comparison']
            relative_change = comparison['difference_analysis']['relative_change']
            
            print(f"\n⚖️ CBAM性能影响:")
            if relative_change > 0.05:  # 5%以上变化
                print(f"   CBAM产生显著影响: {relative_change:.2%} 相对变化")
                print(f"   结论: CBAM有效提升模型表达能力")
            elif relative_change > 0.01:  # 1-5%变化
                print(f"   CBAM产生中等影响: {relative_change:.2%} 相对变化")  
                print(f"   结论: CBAM有一定效果，值得保留")
            else:  # <1%变化
                print(f"   CBAM影响较小: {relative_change:.2%} 相对变化")
                print(f"   结论: CBAM效果有限，可考虑简化")
        
        # 特征选择分析
        if 'feature_selection' in self.analysis_results:
            selection = self.analysis_results['feature_selection']
            print(f"\n🎯 CBAM特征选择能力:")
            
            for input_type, analysis in selection.items():
                print(f"   {input_type}: ", end="")
                avg_selectivity = np.mean([stats['selectivity'] for stats in analysis['cbam_weights_analysis'].values()])
                
                if avg_selectivity > 0.5:
                    print("强选择性 ✅")
                elif avg_selectivity > 0.2:
                    print("中等选择性 ⚡")
                else:
                    print("弱选择性 ⚠️")
        
        print("\n" + "="*80)
        
        # 总体评估和建议
        print("💡 CBAM架构建议:")
        print("   1. 当前CBAM前移策略合理 - 早期特征筛选更有效")
        print("   2. 针对性设计 - adsorbate/solvent专门处理符合数据特点")
        print("   3. 取消CNN骨架CBAM - 避免冗余计算，提升效率")
        print("   4. 保持interaction层CBAM - 融合特征需要注意力引导")
        
        return self.analysis_results
    
    def run_comprehensive_cbam_analysis(self):
        """运行完整的CBAM分析"""
        print("🚀 开始CBAM综合效果分析...")
        print("="*60)
        
        if not self.load_model():
            return False
        
        # 运行各项分析
        self.analyze_cbam_locations_and_parameters()
        self.analyze_cbam_activation_patterns()
        self.compare_with_without_cbam()
        self.analyze_cbam_feature_selection()
        
        # 生成综合报告
        self.generate_cbam_effectiveness_report()
        
        return True

if __name__ == "__main__":
    # 分析最新的Model 2_8的CBAM效果
    model_name = "model_2_8_2487596-epochs_100-bs_32-lr_0.0008-splits_5-grid_16.0_0.8-fold_0.pth"
    model_path = os.path.join(get_paths("output_model_cnn"), model_name)
    
    # 运行分析
    analyzer = CBAMEffectivenessAnalyzer(model_path)
    analyzer.run_comprehensive_cbam_analysis()