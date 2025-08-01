#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP分析工具 - 生成类似SHAP摘要图的可视化
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
import sys
from datetime import datetime
from tqdm import tqdm

# 添加模型路径
sys.path.insert(0, 'model_zoo')

class SHAPAnalyzer:
    """
    SHAP风格的分析器
    通过计算每个样本中每个特征的贡献来生成类似SHAP摘要图的可视化
    """
    
    def __init__(self, model_path: str, model_name: str, seq_len: int, pred_len: int, 
                 input_channels: int = 39, device: str = 'cuda'):
        """
        初始化SHAP分析器
        """
        self.model_path = model_path
        self.model_name = model_name
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_channels = input_channels
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.model = self._load_model()
        self.model.eval()
        
        # 驱动因素名称（根据实际数据顺序）
        self.driver_names = [
            'Firms_Fire', 'Temperatu', 'U_Wind_10m', 'V_Wind_10m', 'Snow_Cover',
            'Total_Percipitation', 'Surface_Latent_heat', 'Downpoint', 'Surface_Pressure', 'Volumetric_soil_moisture1',
            'Volumetric_soil_moisture2', 'Volumetric_soil_moisture3', 'Volumetric_soil_moisture4', 'Aspect', 'DEM',
            'Hillshade', 'Slope', 'D2Infrastre', 'D2Water', 'LULC',
            'NDVI', 'EVI', 'Band1', 'Band2', 'Band3', 'Band7',
            'Band20_D', 'Band21_D', 'Band20_N', 'Band21_N', 'LST_Day',
            'Band29_D', 'Band31_D', 'Band32_D', 'LST_N', 'Band29_N',
            'Band31_N', 'Band32_N', 'LAI'
        ]
        
        # 确保驱动因素名称数量匹配
        if len(self.driver_names) < input_channels:
            self.driver_names.extend([f'Driver_{i}' for i in range(len(self.driver_names), input_channels)])
        elif len(self.driver_names) > input_channels:
            self.driver_names = self.driver_names[:input_channels]
    
    def _load_model(self) -> torch.nn.Module:
        """加载训练好的模型"""
        try:
            # 创建配置
            class Config:
                def __init__(self, model_name, seq_len, pred_len, input_channels):
                    self.task_name = 'long_term_forecast'
                    self.seq_len = seq_len
                    self.label_len = 0
                    self.pred_len = pred_len
                    self.enc_in = input_channels
                    self.dec_in = input_channels
                    self.c_out = 39
                    
                    # 根据模型名称调整配置参数
                    if model_name == 's_mamba':
                        self.d_model = 1024
                        self.n_heads = 16
                        self.e_layers = 2
                        self.d_layers = 1
                        self.d_ff = 2048  # 修复：使用2048而不是4096
                    else:
                        self.d_model = 512
                        self.n_heads = 8
                        self.e_layers = 2
                        self.d_layers = 1
                        self.d_ff = 2048
                    
                    self.dropout = 0.1
                    self.activation = 'gelu'
                    self.output_attention = False
                    self.factor = 5
                    self.moving_avg = 25
                    self.individual = False
                    self.features = 'M'
                    self.patch_len = 16
                    self.alpha = 0.5
                    self.beta = 0.5
                    self.top_k = 5
                    self.num_kernels = 6
                    self.d_state = 16
                    self.d_conv = 4
                    self.expand = 2
                    self.use_norm = True
                    self.distil = True
                    self.embed = 'timeF'
                    self.freq = 'd'
            
            config = Config(self.model_name, self.seq_len, self.pred_len, self.input_channels)
            
            # 动态导入模型
            module = __import__(f'model_zoo.{self.model_name}', fromlist=['Model'])
            Model = getattr(module, 'Model')
            
            # 创建模型实例
            model = Model(config)
            
            # 加载权重
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device)
            print(f"✅ 成功加载模型: {self.model_name}")
            return model
            
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            raise
    
    def generate_sample_data(self, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成样本数据用于SHAP分析（优化版本）"""
        print(f"📊 生成{num_samples}个样本数据...")
        
        # 使用更大的批量生成数据
        batch_size = 128
        all_input_data = []
        all_predictions = []
        
        with tqdm(total=num_samples, desc="生成样本数据") as pbar:
            for i in range(0, num_samples, batch_size):
                current_batch_size = min(batch_size, num_samples - i)
                
                # 生成随机输入数据
                batch_data = torch.randn(current_batch_size, self.seq_len, self.input_channels, device=self.device)
                
                # 归一化到0-1范围（模拟真实数据）
                batch_data = torch.sigmoid(batch_data)
                
                # 生成对应的预测结果
                with torch.no_grad():
                    batch_predictions = self.model(batch_data, None, None, None)
                
                all_input_data.append(batch_data)
                all_predictions.append(batch_predictions)
                
                pbar.update(current_batch_size)
        
        # 合并所有批次
        input_data = torch.cat(all_input_data, dim=0)
        predictions = torch.cat(all_predictions, dim=0)
        
        return input_data, predictions
    
    def calculate_shap_values(self, input_data: torch.Tensor, 
                            num_background: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算SHAP值（优化版本，使用向量化和并行化）
        
        Args:
            input_data: 输入数据 [num_samples, seq_len, channels]
            num_background: 背景样本数量
            
        Returns:
            shap_values: SHAP值 [num_samples, seq_len, channels]
            feature_values: 特征值 [num_samples, seq_len, channels]
        """
        print("🔍 计算SHAP值（优化版本）...")
        
        num_samples = input_data.shape[0]
        device = input_data.device
        
        # 计算背景平均值
        background_samples = input_data[:num_background]
        background_mean = background_samples.mean(dim=0, keepdim=True)  # [1, seq_len, channels]
        
        # 计算基线预测
        with torch.no_grad():
            baseline_pred = self.model(background_samples, None, None, None).mean(dim=0, keepdim=True)
        
        # 使用更高效的SHAP计算方法
        shap_values = self._calculate_shap_values_vectorized(
            input_data, background_mean, baseline_pred, num_samples
        )
        
        return shap_values, input_data.cpu().numpy()
    
    def _calculate_shap_values_vectorized(self, input_data: torch.Tensor, 
                                        background_mean: torch.Tensor,
                                        baseline_pred: torch.Tensor,
                                        num_samples: int) -> np.ndarray:
        """向量化计算SHAP值"""
        
        # 根据样本数量选择最优方法
        if num_samples <= 50:
            # 小样本：使用梯度近似（最快）
            print("🚀 使用梯度近似方法（最快）")
            shap_values = self._calculate_shap_gradient_approximation(
                input_data, background_mean, baseline_pred
            )
        elif num_samples <= 500:
            # 中等样本：使用快速扰动方法
            print("⚡ 使用快速扰动方法（中等速度）")
            shap_values = self._calculate_shap_fast_perturbation(
                input_data, background_mean, baseline_pred, num_samples
            )
        else:
            # 大样本：使用批量扰动（最稳定）
            print("📦 使用批量扰动方法（最稳定）")
            shap_values = self._calculate_shap_batch_perturbation(
                input_data, background_mean, baseline_pred, num_samples
            )
        
        return shap_values
    
    def _calculate_shap_fast_perturbation(self, input_data: torch.Tensor,
                                        background_mean: torch.Tensor,
                                        baseline_pred: torch.Tensor,
                                        num_samples: int) -> np.ndarray:
        """快速扰动计算SHAP值（优化版本）"""
        
        # 使用更大的批量大小
        batch_size = 128
        shap_values = np.zeros((num_samples, self.seq_len, self.input_channels))
        
        # 计算总批次数
        total_batches = (num_samples + batch_size - 1) // batch_size
        
        print(f"⚡ 快速扰动方法: {total_batches}个批次")
        
        with tqdm(total=total_batches, desc="快速SHAP计算") as pbar:
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                batch_data = input_data[i:end_idx]
                batch_size_actual = batch_data.shape[0]
                
                # 使用更高效的扰动策略
                shap_batch = self._compute_fast_shap_batch(
                    batch_data, background_mean, baseline_pred, batch_size_actual
                )
                
                shap_values[i:i+batch_size_actual] = shap_batch
                pbar.update(1)
        
        return shap_values
    
    def _compute_fast_shap_batch(self, batch_data: torch.Tensor,
                                background_mean: torch.Tensor,
                                baseline_pred: torch.Tensor,
                                batch_size: int) -> np.ndarray:
        """快速计算单个批次的SHAP值"""
        
        shap_batch = np.zeros((batch_size, self.seq_len, self.input_channels))
        
        # 原始预测
        with torch.no_grad():
            original_preds = self.model(batch_data, None, None, None)
        
        # 对每个特征位置进行快速扰动
        for t in range(self.seq_len):
            for c in range(self.input_channels):
                # 创建扰动版本（只扰动一个特征）
                perturbed = batch_data.clone()
                perturbed[:, t, c] = background_mean[0, t, c]
                
                # 计算扰动预测
                with torch.no_grad():
                    perturbed_preds = self.model(perturbed, None, None, None)
                
                # 计算SHAP值
                pred_diff = original_preds[:, 0, 0] - perturbed_preds[:, 0, 0]
                feature_diff = batch_data[:, t, c] - background_mean[0, t, c]
                
                shap_batch[:, t, c] = (feature_diff * pred_diff).cpu().numpy()
        
        return shap_batch
    
    def _smart_sample_indices(self, data: np.ndarray, max_points: int) -> np.ndarray:
        """智能采样索引，保持数据分布"""
        
        if len(data) <= max_points:
            return np.arange(len(data))
        
        # 计算分位数
        quantiles = np.percentile(data, np.linspace(0, 100, 10))
        
        # 在每个分位数区间内采样
        sampled_indices = []
        points_per_quantile = max_points // 10
        
        for i in range(len(quantiles) - 1):
            mask = (data >= quantiles[i]) & (data < quantiles[i + 1])
            if i == len(quantiles) - 2:  # 最后一个区间包含边界
                mask = (data >= quantiles[i]) & (data <= quantiles[i + 1])
            
            quantile_indices = np.where(mask)[0]
            if len(quantile_indices) > 0:
                if len(quantile_indices) <= points_per_quantile:
                    sampled_indices.extend(quantile_indices)
                else:
                    # 随机采样
                    sampled_indices.extend(
                        np.random.choice(quantile_indices, points_per_quantile, replace=False)
                    )
        
        return np.array(sampled_indices)
    
    def _calculate_shap_gradient_approximation(self, input_data: torch.Tensor,
                                             background_mean: torch.Tensor,
                                             baseline_pred: torch.Tensor) -> np.ndarray:
        """使用梯度近似计算SHAP值（最快方法）"""
        
        input_data.requires_grad_(True)
        
        # 前向传播
        with torch.enable_grad():
            predictions = self.model(input_data, None, None, None)
            loss = torch.mean((predictions - baseline_pred) ** 2)
        
        # 计算梯度
        gradients = torch.autograd.grad(loss, input_data, create_graph=False)[0]
        
        # SHAP值近似 = 特征值 * 梯度
        shap_values = (input_data - background_mean) * gradients
        
        return shap_values.detach().cpu().numpy()
    
    def _calculate_shap_batch_perturbation(self, input_data: torch.Tensor,
                                         background_mean: torch.Tensor,
                                         baseline_pred: torch.Tensor,
                                         num_samples: int) -> np.ndarray:
        """批量扰动计算SHAP值（优化版本）"""
        
        batch_size = 64  # 增大批量大小
        shap_values = np.zeros((num_samples, self.seq_len, self.input_channels))
        
        # 计算总批次数
        total_batches = (num_samples + batch_size - 1) // batch_size
        total_features = self.seq_len * self.input_channels
        
        print(f"📊 使用批量扰动方法: {total_batches}个批次, {total_features}个特征位置")
        
        # 使用进度条
        with tqdm(total=total_batches, desc="计算SHAP值") as pbar:
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                batch_data = input_data[i:end_idx]
                batch_size_actual = batch_data.shape[0]
                
                # 向量化创建扰动数据
                perturbed_data = self._create_perturbed_batch_vectorized(
                    batch_data, background_mean, batch_size_actual
                )
                
                # 批量预测
                with torch.no_grad():
                    # 原始预测
                    original_preds = self.model(batch_data, None, None, None)
                    
                    # 扰动预测
                    perturbed_preds = self.model(perturbed_data, None, None, None)
                
                # 向量化计算SHAP值
                self._compute_shap_values_vectorized(
                    shap_values, batch_data, background_mean, 
                    original_preds, perturbed_preds, 
                    i, batch_size_actual
                )
                
                pbar.update(1)
        
        return shap_values
    
    def _create_perturbed_batch_vectorized(self, batch_data: torch.Tensor,
                                         background_mean: torch.Tensor,
                                         batch_size: int) -> torch.Tensor:
        """向量化创建扰动批次数据"""
        
        # 预分配内存
        total_perturbations = self.seq_len * self.input_channels
        perturbed_batch = torch.zeros(
            batch_size * total_perturbations, 
            self.seq_len, 
            self.input_channels, 
            device=batch_data.device
        )
        
        # 复制原始数据到所有扰动位置
        for i in range(total_perturbations):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            perturbed_batch[start_idx:end_idx] = batch_data
        
        # 向量化替换特征值
        idx = 0
        for t in range(self.seq_len):
            for c in range(self.input_channels):
                start_idx = idx * batch_size
                end_idx = start_idx + batch_size
                perturbed_batch[start_idx:end_idx, t, c] = background_mean[0, t, c]
                idx += 1
        
        return perturbed_batch
    
    def _compute_shap_values_vectorized(self, shap_values: np.ndarray,
                                      batch_data: torch.Tensor,
                                      background_mean: torch.Tensor,
                                      original_preds: torch.Tensor,
                                      perturbed_preds: torch.Tensor,
                                      batch_start: int,
                                      batch_size: int):
        """向量化计算SHAP值"""
        
        idx = 0
        for t in range(self.seq_len):
            for c in range(self.input_channels):
                # 计算预测差异
                pred_diff = (original_preds[:, 0, 0] - 
                           perturbed_preds[idx:idx+batch_size, 0, 0])
                
                # 计算特征差异
                feature_diff = batch_data[:, t, c] - background_mean[0, t, c]
                
                # SHAP值
                shap_values[batch_start:batch_start+batch_size, t, c] = \
                    (feature_diff * pred_diff).cpu().numpy()
                
                idx += batch_size
    
    def create_shap_summary_plot(self, shap_values: np.ndarray, feature_values: np.ndarray,
                               save_path: str = None):
        """创建SHAP摘要图"""
        print("📊 创建SHAP摘要图...")
        
        # 计算每个特征的平均绝对SHAP值
        feature_importance = np.mean(np.abs(shap_values), axis=(0, 1))  # [channels]
        
        # 选择最重要的特征（Top 20）
        top_k = min(20, self.input_channels)
        top_indices = np.argsort(feature_importance)[-top_k:][::-1]
        top_features = [self.driver_names[i] for i in top_indices]
        
        # 准备数据（优化版本）
        plot_data = []
        max_points_per_feature = 1000  # 限制每个特征的点数
        
        print(f"📈 处理{top_k}个重要特征的可视化数据...")
        
        for i, feat_idx in enumerate(tqdm(top_indices, desc="处理特征数据")):
            # 获取该特征的所有SHAP值和特征值
            feat_shap = shap_values[:, :, feat_idx].flatten()  # [num_samples * seq_len]
            feat_values = feature_values[:, :, feat_idx].flatten()  # [num_samples * seq_len]
            
            # 智能采样以减少点的数量
            if len(feat_shap) > max_points_per_feature:
                indices = self._smart_sample_indices(feat_shap, max_points_per_feature)
                feat_shap = feat_shap[indices]
                feat_values = feat_values[indices]
            
            # 向量化添加到绘图数据
            plot_data.extend([
                {
                    'Feature': top_features[i],
                    'SHAP': float(shap_val),
                    'Feature_Value': float(feat_val)
                }
                for shap_val, feat_val in zip(feat_shap, feat_values)
            ])
        
        # 创建DataFrame
        import pandas as pd
        df = pd.DataFrame(plot_data)
        
        # 创建SHAP摘要图
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 为每个特征创建水平点图
        y_positions = range(len(top_features))
        
        for i, feature in enumerate(top_features):
            feature_data = df[df['Feature'] == feature]
            
            # 绘制点
            scatter = ax.scatter(feature_data['SHAP'], [i] * len(feature_data), 
                               c=feature_data['Feature_Value'], 
                               cmap='viridis', alpha=0.6, s=20)
        
        # 添加零线
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # 设置坐标轴
        ax.set_yticks(y_positions)
        ax.set_yticklabels(top_features)
        ax.set_xlabel('SHAP Value', fontsize=12)
        ax.set_title(f'SHAP Summary Plot - {self.model_name}', fontsize=14, fontweight='bold')
        
        # 添加颜色条
        norm = plt.Normalize(0, 1)
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Feature Value', fontsize=10)
        
        # 调整布局
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 SHAP摘要图保存至: {save_path}")
        
        plt.show()
    
    def create_feature_importance_plot(self, shap_values: np.ndarray, save_path: str = None):
        """创建特征重要性图"""
        print("📈 创建特征重要性图...")
        
        # 计算每个特征的平均绝对SHAP值
        feature_importance = np.mean(np.abs(shap_values), axis=(0, 1))
        
        # 选择最重要的特征（Top 15）
        top_k = min(15, self.input_channels)
        top_indices = np.argsort(feature_importance)[-top_k:][::-1]
        top_features = [self.driver_names[i] for i in top_indices]
        top_importance = feature_importance[top_indices]
        
        # 创建水平条形图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(range(len(top_features)), top_importance, color='steelblue', alpha=0.7)
        
        # 设置坐标轴
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
        ax.set_title(f'Feature Importance - {self.model_name}', fontsize=14, fontweight='bold')
        
        # 添加数值标签
        for i, (bar, importance) in enumerate(zip(bars, top_importance)):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{importance:.4f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 特征重要性图保存至: {save_path}")
        
        plt.show()
    
    def generate_shap_report(self, shap_values: np.ndarray, feature_values: np.ndarray,
                           predictions: np.ndarray, save_path: str = None) -> str:
        """生成SHAP分析报告"""
        
        print("📄 生成SHAP分析报告...")
        
        # 计算每个特征的平均绝对SHAP值（重要性）
        feature_importance = np.mean(np.abs(shap_values), axis=(0, 1))  # [channels]
        
        # 计算每个特征的正向和负向影响分别
        positive_shap = np.where(shap_values > 0, shap_values, 0)
        negative_shap = np.where(shap_values < 0, shap_values, 0)
        
        # 正向影响：所有正SHAP值的平均
        positive_impact = np.mean(positive_shap, axis=(0, 1))  # [channels]
        # 负向影响：所有负SHAP值的平均（取绝对值）
        negative_impact = np.abs(np.mean(negative_shap, axis=(0, 1)))  # [channels]
        
        # 计算正负样本比例
        positive_ratio = np.sum(shap_values > 0, axis=(0, 1)) / shap_values.shape[0] / shap_values.shape[1]  # [channels]
        negative_ratio = np.sum(shap_values < 0, axis=(0, 1)) / shap_values.shape[0] / shap_values.shape[1]  # [channels]
        
        # 综合影响方向（用于排序和显示）
        feature_impact = positive_impact - negative_impact  # 正值表示正向影响，负值表示负向影响
        
        # 获取最重要的特征（Top 20）
        top_k = min(20, self.input_channels)
        top_indices = np.argsort(feature_importance)[-top_k:][::-1]
        top_features = [self.driver_names[i] for i in top_indices]
        top_importance = feature_importance[top_indices]
        top_impact = feature_impact[top_indices]
        
        # 计算SHAP值统计信息
        shap_stats = {
            'mean': np.mean(shap_values),
            'std': np.std(shap_values),
            'min': np.min(shap_values),
            'max': np.max(shap_values),
            'positive_ratio': np.sum(shap_values > 0) / shap_values.size,
            'negative_ratio': np.sum(shap_values < 0) / shap_values.size
        }
        
        # 生成报告
        report = f"""# SHAP分析报告 - {self.model_name}

## 📊 分析概述

- **模型名称**: {self.model_name}
- **模型路径**: {self.model_path}
- **输入序列长度**: {self.seq_len}
- **预测长度**: {self.pred_len}
- **输入通道数**: {self.input_channels}
- **样本数量**: {shap_values.shape[0]}
- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🔍 SHAP值统计信息

### 整体统计
- **平均SHAP值**: {shap_stats['mean']:.6f}
- **SHAP值标准差**: {shap_stats['std']:.6f}
- **SHAP值范围**: [{shap_stats['min']:.6f}, {shap_stats['max']:.6f}]
- **正向影响比例**: {shap_stats['positive_ratio']:.2%}
- **负向影响比例**: {shap_stats['negative_ratio']:.2%}

## 🏆 特征重要性排名 (Top 20)

| 排名 | 特征名称 | 平均绝对SHAP值 | 正向/负向影响 | 影响方向 (正样本比例/负样本比例) |
|------|----------|----------------|---------------|----------------------------------|
"""
        
        # 添加特征排名表格
        for i, (feature, importance, impact) in enumerate(zip(top_features, top_importance, top_impact)):
            feat_idx = top_indices[i]
            pos_impact = positive_impact[feat_idx]
            neg_impact = negative_impact[feat_idx]
            pos_ratio = positive_ratio[feat_idx]
            neg_ratio = negative_ratio[feat_idx]
            
            direction = "🟢 正向" if impact > 0 else "🔴 负向" if impact < 0 else "⚪ 中性"
            report += f"| {i+1} | {feature} | {importance:.6f} | {pos_impact:.6f}/{neg_impact:.6f} | {direction} ({pos_ratio:.1%}/{neg_ratio:.1%}) |\n"
        
        report += f"""
## 📈 详细特征分析

### 最重要的特征分析

"""
        
        # 分析前5个最重要的特征
        for i in range(min(5, len(top_features))):
            feat_idx = top_indices[i]
            feature = top_features[i]
            importance = top_importance[i]
            impact = top_impact[i]
            
            # 获取该特征的详细统计
            feat_shap = shap_values[:, :, feat_idx]
            feat_values = feature_values[:, :, feat_idx]
            
            # 计算特征值与SHAP值的关系
            high_val_mask = feat_values > np.median(feat_values)
            low_val_mask = feat_values <= np.median(feat_values)
            
            high_shap_mean = np.mean(feat_shap[high_val_mask])
            low_shap_mean = np.mean(feat_shap[low_val_mask])
            
            # 获取该特征的正负影响详细信息
            pos_impact = positive_impact[feat_idx]
            neg_impact = negative_impact[feat_idx]
            pos_ratio = positive_ratio[feat_idx]
            neg_ratio = negative_ratio[feat_idx]
            
            report += f"""#### {i+1}. {feature}

- **重要性排名**: 第{i+1}位
- **平均绝对SHAP值**: {importance:.6f}
- **正向影响强度**: {pos_impact:.6f} (占{pos_ratio:.1%}样本)
- **负向影响强度**: {neg_impact:.6f} (占{neg_ratio:.1%}样本)
- **综合影响方向**: {'正向' if impact > 0 else '负向' if impact < 0 else '中性'}
- **高值样本平均SHAP**: {high_shap_mean:.6f}
- **低值样本平均SHAP**: {low_shap_mean:.6f}
- **特征值范围**: [{np.min(feat_values):.4f}, {np.max(feat_values):.4f}]

**分析解读**: 
"""
            
            if impact > 0:
                report += f"{feature}对模型预测有正向影响。当{feature}值较高时，模型倾向于增加预测值。"
            else:
                report += f"{feature}对模型预测有负向影响。当{feature}值较高时，模型倾向于减少预测值。"
            
            if abs(high_shap_mean - low_shap_mean) > 0.01:
                report += f" 特征值的高低对SHAP值有显著影响。"
            else:
                report += f" 特征值的高低对SHAP值影响较小。"
            
            report += "\n\n"
        
        # 按类别分析特征
        report += """## 🗂️ 按类别分析特征

### 气象数据特征
"""
        
        weather_features = ['Temperatu', 'U_Wind_10r', 'V_Wind_10m', 'Snow_Cove', 'Total_Perci', 
                          'Surface_La', 'Downpoint', 'Surface_Pro']
        
        for feature in weather_features:
            if feature in self.driver_names:
                idx = self.driver_names.index(feature)
                importance = feature_importance[idx]
                impact = feature_impact[idx]
                pos_impact = positive_impact[idx]
                neg_impact = negative_impact[idx]
                pos_ratio = positive_ratio[idx]
                neg_ratio = negative_ratio[idx]
                rank = np.where(np.argsort(feature_importance)[::-1] == idx)[0][0] + 1
                report += f"- **{feature}**: 排名第{rank}位，重要性{importance:.6f}，正向{pos_impact:.6f}({pos_ratio:.1%})，负向{neg_impact:.6f}({neg_ratio:.1%})\n"
        
        report += """
### 土壤数据特征
"""
        
        soil_features = ['Volumetric_1', 'Volumetric_2', 'Volumetric_3', 'Volumetric_4']
        
        for feature in soil_features:
            if feature in self.driver_names:
                idx = self.driver_names.index(feature)
                importance = feature_importance[idx]
                impact = feature_impact[idx]
                pos_impact = positive_impact[idx]
                neg_impact = negative_impact[idx]
                pos_ratio = positive_ratio[idx]
                neg_ratio = negative_ratio[idx]
                rank = np.where(np.argsort(feature_importance)[::-1] == idx)[0][0] + 1
                report += f"- **{feature}**: 排名第{rank}位，重要性{importance:.6f}，正向{pos_impact:.6f}({pos_ratio:.1%})，负向{neg_impact:.6f}({neg_ratio:.1%})\n"
        
        report += """
### 地形数据特征
"""
        
        terrain_features = ['Aspect', 'DEM', 'Hillshade', 'Slope', 'D2Infrastre', 'D2Water']
        
        for feature in terrain_features:
            if feature in self.driver_names:
                idx = self.driver_names.index(feature)
                importance = feature_importance[idx]
                impact = feature_impact[idx]
                pos_impact = positive_impact[idx]
                neg_impact = negative_impact[idx]
                pos_ratio = positive_ratio[idx]
                neg_ratio = negative_ratio[idx]
                rank = np.where(np.argsort(feature_importance)[::-1] == idx)[0][0] + 1
                report += f"- **{feature}**: 排名第{rank}位，重要性{importance:.6f}，正向{pos_impact:.6f}({pos_ratio:.1%})，负向{neg_impact:.6f}({neg_ratio:.1%})\n"
        
        report += """
### 卫星数据特征
"""
        
        satellite_features = ['NDVI', 'EVI', 'Band1', 'Band2', 'Band3', 'Band7', 'LST_D', 'LST_N', 'LAI']
        
        for feature in satellite_features:
            if feature in self.driver_names:
                idx = self.driver_names.index(feature)
                importance = feature_importance[idx]
                impact = feature_impact[idx]
                pos_impact = positive_impact[idx]
                neg_impact = negative_impact[idx]
                pos_ratio = positive_ratio[idx]
                neg_ratio = negative_ratio[idx]
                rank = np.where(np.argsort(feature_importance)[::-1] == idx)[0][0] + 1
                report += f"- **{feature}**: 排名第{rank}位，重要性{importance:.6f}，正向{pos_impact:.6f}({pos_ratio:.1%})，负向{neg_impact:.6f}({neg_ratio:.1%})\n"
        
        report += f"""
## 📊 预测结果分析

### 预测值统计
- **平均预测值**: {np.mean(predictions):.6f}
- **预测值标准差**: {np.std(predictions):.6f}
- **预测值范围**: [{np.min(predictions):.6f}, {np.max(predictions):.6f}]

### 预测分布
- **高预测值样本比例**: {np.sum(predictions > np.median(predictions)) / len(predictions):.2%}
- **低预测值样本比例**: {np.sum(predictions <= np.median(predictions)) / len(predictions):.2%}

## 🎯 关键发现

### 1. 最重要的驱动因素
根据SHAP分析，以下特征对模型预测影响最大：
"""
        
        for i in range(min(3, len(top_features))):
            report += f"- **{top_features[i]}**: {top_importance[i]:.6f}\n"
        
        report += """
### 2. 影响方向分析
- **正向影响特征**: 这些特征值越高，模型预测值越大
- **负向影响特征**: 这些特征值越高，模型预测值越小

### 3. 特征交互模式
通过SHAP值分布可以看出不同特征之间的交互关系：
- 某些特征组合可能产生协同效应
- 不同特征值范围对预测的影响程度不同

## 💡 建议与改进

### 1. 数据收集建议
- 重点关注高重要性特征的测量精度
- 考虑增加对重要特征的时间分辨率
- 优化数据预处理流程

### 2. 模型优化建议
- 可以针对高重要性特征进行特征工程
- 考虑使用特征选择方法减少噪声特征
- 优化模型架构以更好地利用重要特征

### 3. 应用建议
- 在实际应用中优先关注高重要性特征
- 建立基于SHAP值的异常检测机制
- 定期更新SHAP分析以监控模型性能变化

## 📋 技术说明

### SHAP值计算原理
本分析使用基于扰动的SHAP值计算方法：
1. 使用背景样本建立基线预测
2. 对每个特征进行扰动（用背景平均值替换）
3. 计算预测差异作为SHAP值
4. 统计所有样本的SHAP值分布

### 正负影响计算方法
为了避免正负影响抵消的问题，本分析采用以下方法：
1. **正向影响**: 计算所有正SHAP值的平均值，表示该特征对预测的正向贡献强度
2. **负向影响**: 计算所有负SHAP值的平均值的绝对值，表示该特征对预测的负向贡献强度
3. **正负样本比例**: 统计正负SHAP值在总样本中的占比
4. **综合影响方向**: 正向影响 - 负向影响，正值表示整体正向，负值表示整体负向

### 可视化说明
- **SHAP摘要图**: 显示每个特征的SHAP值分布和特征值关系
- **特征重要性图**: 按重要性排序显示各特征的平均绝对SHAP值
- **颜色编码**: 蓝色表示低特征值，红色表示高特征值

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📄 SHAP分析报告保存至: {save_path}")
        
        return report
    
    def run_shap_analysis(self, output_dir: str = './shap_results', num_samples: int = 100):
        """运行完整的SHAP分析"""
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🚀 开始 {self.model_name} 的SHAP分析...")
        
        # 生成样本数据
        input_data, predictions = self.generate_sample_data(num_samples)
        
        # 计算SHAP值
        shap_values, feature_values = self.calculate_shap_values(input_data)
        
        # 创建可视化
        self.create_shap_summary_plot(
            shap_values, feature_values,
            os.path.join(output_dir, f'{self.model_name}_shap_summary.png')
        )
        
        self.create_feature_importance_plot(
            shap_values,
            os.path.join(output_dir, f'{self.model_name}_feature_importance.png')
        )
        
        # 生成分析报告
        report = self.generate_shap_report(
            shap_values, feature_values, predictions.cpu().numpy(),
            os.path.join(output_dir, f'{self.model_name}_shap_analysis_report.md')
        )
        
        print(f"✅ SHAP分析完成！结果保存在: {output_dir}")
        print(f"📄 分析报告: {os.path.join(output_dir, f'{self.model_name}_shap_analysis_report.md')}")
        
        return {
            'shap_values': shap_values,
            'feature_values': feature_values,
            'predictions': predictions.cpu().numpy(),
            'report': report
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SHAP分析工具')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径(.pth)')
    parser.add_argument('--model_name', type=str, required=True, help='模型名称')
    parser.add_argument('--seq_len', type=int, default=365, help='输入序列长度')
    parser.add_argument('--pred_len', type=int, default=1, help='预测长度')
    parser.add_argument('--input_channels', type=int, default=40, help='输入通道数')
    parser.add_argument('--output_dir', type=str, default='./shap_results', help='输出目录')
    parser.add_argument('--num_samples', type=int, default=100, help='样本数量')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    
    args = parser.parse_args()
    
    # 创建SHAP分析器
    analyzer = SHAPAnalyzer(
        model_path=args.model_path,
        model_name=args.model_name,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        input_channels=args.input_channels,
        device=args.device
    )
    
    # 运行分析
    results = analyzer.run_shap_analysis(args.output_dir, args.num_samples)
    
    print("🎉 SHAP分析完成！")


if __name__ == "__main__":
    main() 