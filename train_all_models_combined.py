#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
野火预测模型全面对比实验 - 统一版本
先训练测试model_zoo中的所有模型，再训练测试model_zoo_10x中的所有10倍参数模型
支持early stopping、F1评价指标、最佳模型测试和CSV结果导出
"""

from dataload_year import TimeSeriesDataLoader, TimeSeriesPixelDataset, FullDatasetLoader
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import os
import random
import importlib
from datetime import datetime, timedelta
import glob
import torch.nn.functional as F
import warnings
import pandas as pd
import sys
import time
import argparse

# 动态导入wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ wandb未安装，将跳过wandb监控功能")

warnings.filterwarnings("ignore")

# =============================================================================
# 配置参数
# =============================================================================

# 全局训练配置 - 统一管理所有训练相关参数
ENABLE_10X_TRAINING = True        # 是否训练10x模型的全局开关
WANDB_ENABLED = True               # 是否启用WandB监控
GLOBAL_SEED = 42                   # 全局随机种子
DEFAULT_PATIENCE = 20              # 默认early stopping patience
DEFAULT_MAX_PARALLEL_PER_GPU = 2   # 默认每GPU最大并行任务数

# 多任务学习配置
MULTITASK_CONFIG = {
    'firms_weight': 1,           # FIRMS预测的损失权重。典型的loss结合（other drivers loss*weight之后）：FIRMS loss: 0.3112890124320984, Other drivers loss: 0.0020517727825790644
    'other_drivers_weight': 1.0,   # 其他驱动因素预测的损失权重
    'ignore_zero_values': True,    # 是否忽略其他驱动因素中的0值
    'loss_function': 'mse',       # 损失函数类型：'huber', 'mse', 'mae'
    'loss_type': 'focal'          # 损失函数类型选择：'focal'(MultiTaskFocalLoss) 或 'kldiv'(MultiTaskKLDivLoss) 或 'multitask'(MultiTaskLoss)
}

# 数据集年份配置
DEFAULT_TRAIN_YEARS = [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 
                      2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
DEFAULT_VAL_YEARS = [2021, 2022]
DEFAULT_TEST_YEARS = [2023, 2024]

# 模型目录配置
# target_all_channels = target_all_channels.clone()
# target_all_channels[:, :, 0] = (target_all_channels[:, :, 0] > 10).float() 别忘了把这2行消掉
STANDARD_MODEL_DIR = '/mnt/raid/zhengsen/pths/7to1_Focal_woFirms_onlyFirmsLoss_newloadertest'  
MODEL_10X_DIR = '/mnt/raid/zhengsen/pths/model_pth_20epoch_MSE_10x'

def print_config_status():
    """打印当前配置状态"""
    print("📋 当前训练配置:")
    print(f"   10x模型训练: {'✅ 启用' if ENABLE_10X_TRAINING else '❌ 禁用'}")
    print(f"   WandB监控: {'✅ 启用' if WANDB_ENABLED else '❌ 禁用'}")
    print(f"   随机种子: {GLOBAL_SEED}")
    print(f"   默认并行数: {DEFAULT_MAX_PARALLEL_PER_GPU}/GPU")
    print(f"   Early Stopping patience: {DEFAULT_PATIENCE}")
    print(f"   多任务Loss类型: {MULTITASK_CONFIG['loss_type'].upper()}")
    print(f"   FIRMS权重: {MULTITASK_CONFIG['firms_weight']}")
    print(f"   其他驱动因素权重: {MULTITASK_CONFIG['other_drivers_weight']}")
    print(f"   忽略0值: {'✅' if MULTITASK_CONFIG['ignore_zero_values'] else '❌'}")
    print(f"   回归损失函数: {MULTITASK_CONFIG['loss_function']}")
    if MULTITASK_CONFIG['loss_type'] == 'focal':
        print(f"   Focal Loss α: {TRAINING_CONFIG['focal_alpha']}")
        print(f"   Focal Loss γ: {TRAINING_CONFIG['focal_gamma']}")
    elif MULTITASK_CONFIG['loss_type'] == 'kldiv':
        print(f"   KL散度温度参数: 1.0")
    elif MULTITASK_CONFIG['loss_type'] == 'multitask':
        print(f"   统一损失函数: {MULTITASK_CONFIG['loss_function']}")
    
    # 🔥 新增：位置信息和气象数据特征状态
    print(f"\n🔧 数据特征配置:")
    print(f"   位置信息特征: {'✅ 启用' if DATA_CONFIG['enable_position_features'] else '❌ 禁用'}")
    print(f"   未来气象数据: {'✅ 启用' if DATA_CONFIG['enable_future_weather'] else '❌ 禁用'}")
    if DATA_CONFIG['enable_future_weather']:
        channels_str = ','.join(map(str, DATA_CONFIG['weather_channels']))
        print(f"   气象通道: [{channels_str}] (共{len(DATA_CONFIG['weather_channels'])}个)")
    
    # 计算总输入通道数
    base_channels = 39
    additional_channels = 0
    if DATA_CONFIG['enable_position_features']:
        additional_channels += 1
    if DATA_CONFIG['enable_future_weather']:
        additional_channels += len(DATA_CONFIG['weather_channels'])
    
    total_channels = base_channels + additional_channels
    if additional_channels > 0:
        print(f"   输入通道数: {base_channels} (基础) + {additional_channels} (特征) = {total_channels} (总计)")
    else:
        print(f"   输入通道数: {total_channels} (标准)")

def is_model_trained(model_name, model_type='standard'):
    """
    检查模型是否已经训练完成
    通过检查final_epoch模型文件是否存在来判断
    """
    model_save_dir = TRAINING_CONFIG[model_type]['model_save_dir']
    final_model_path = os.path.join(model_save_dir, f'{model_name}_final_epoch.pth')
    return os.path.exists(final_model_path)

def get_trained_model_paths(model_name, model_type='standard'):
    """
    获取已训练模型的所有保存路径
    返回包含metric_name和path的字典
    """
    model_save_dir = TRAINING_CONFIG[model_type]['model_save_dir']
    metric_types = ['f1', 'recall', 'pr_auc', 'mae', 'mse', 'final_epoch']
    
    trained_paths = {}
    for metric_type in metric_types:
        if metric_type == 'final_epoch':
            path = os.path.join(model_save_dir, f'{model_name}_final_epoch.pth')
        else:
            path = os.path.join(model_save_dir, f'{model_name}_best_{metric_type}.pth')
        
        if os.path.exists(path):
            trained_paths[metric_type] = {'path': path, 'score': 0.0}  # score会在测试时更新
    
    return trained_paths

def filter_trained_models(model_list, model_type='standard', force_retrain=False):
    """
    过滤已训练的模型
    返回 (需要训练的模型列表, 已训练的模型字典)
    """
    if force_retrain:
        print(f"🔄 强制重新训练模式：将训练所有 {len(model_list)} 个{model_type}模型")
        return model_list, {}
    
    models_to_train = []
    trained_models = {}
    
    print(f"🔍 检查{model_type}模型训练状态...")
    
    for model_name in model_list:
        if is_model_trained(model_name, model_type):
            trained_paths = get_trained_model_paths(model_name, model_type)
            trained_models[model_name] = trained_paths
            print(f"✅ {model_name}: 已训练完成 ({len(trained_paths)}个保存版本)")
        else:
            models_to_train.append(model_name)
            print(f"❌ {model_name}: 需要训练")
    
    print(f"\n📊 {model_type}模型状态统计:")
    print(f"   需要训练: {len(models_to_train)} 个")
    print(f"   已训练完成: {len(trained_models)} 个")
    
    if models_to_train:
        print(f"   将训练: {', '.join(models_to_train)}")
    if trained_models:
        print(f"   跳过训练: {', '.join(trained_models.keys())}")
    
    return models_to_train, trained_models

def get_all_models(model_zoo_path):
    """获取指定model_zoo中所有可用的模型"""
    model_files = []
    if os.path.exists(model_zoo_path):
        for file in os.listdir(model_zoo_path):
            if file.endswith('.py') and not file.startswith('__') and file != 'trash':
                model_name = file[:-3]  # 去掉.py后缀
                model_files.append(model_name)
    return sorted(model_files)

# 获取标准模型和10x模型列表
MODEL_LIST_STANDARD = get_all_models('model_zoo')

if ENABLE_10X_TRAINING:
    MODEL_LIST_10X = get_all_models('model_zoo_10x')
else:
    MODEL_LIST_10X = []  # 空列表，跳过10x模型训练

print(f"发现标准模型: {MODEL_LIST_STANDARD}")
print(f"发现10x模型: {MODEL_LIST_10X}")
if not ENABLE_10X_TRAINING:
    print("⚠️  10x模型训练已禁用")

# 训练配置
TRAINING_CONFIG = {
    'use_wandb': WANDB_ENABLED,         # 使用WandB配置
    'seed': GLOBAL_SEED,                # 使用随机种子
    'patience': DEFAULT_PATIENCE,       # 使用patience配置
    'seq_len': 7,                      # 输入序列长度
    'pred_len': 1,                      # 预测序列长度
    'focal_alpha': 0.5,                 # 使用最佳的Focal Loss正样本权重
    'focal_gamma': 2.0,                 # Focal Loss聚焦参数
    
    # 标准模型配置
    'standard': {
        'epochs': 20,
        'batch_size': 128,
        'learning_rate': 5e-5,          # 降低学习率，与10x模型一致
        'weight_decay': 1e-4,
        'T_0': 20,
        'T_mult': 2,
        'eta_min':1e-5,
        'max_grad_norm': 0.0,           # 启用梯度裁剪防止梯度爆炸; 0.0表示不裁剪
        'model_save_dir': STANDARD_MODEL_DIR,
    },
    
    # 10x模型配置（考虑显存限制）
    '10x': {
        'epochs': 20,
        'batch_size': 128,
        'learning_rate': 5e-5,
        'weight_decay': 1e-4,
        'T_0': 20,
        'T_mult': 2,
        'eta_min': 1e-5,
        'max_grad_norm': 0.0,           # 启用梯度裁剪防止梯度爆炸
        'model_save_dir': MODEL_10X_DIR,
    }
}

# 数据配置
DATA_CONFIG = {
    'train_years': DEFAULT_TRAIN_YEARS,
    'val_years': DEFAULT_VAL_YEARS,
    'test_years': DEFAULT_TEST_YEARS,
    
    # 底层数据集配置（加载完整数据）
    'positive_ratio': 1.0,           # 底层加载所有正样本
    'pos_neg_ratio': 2.0,            # 底层正负样本比例1:1
    'resample_each_epoch': False,    # 底层禁用重新抽样，所以需要一直设为False
    'firms_min': 0,                  # FIRMS数据最小值（跳过统计）
    'firms_max': 100,                # FIRMS数据最大值（跳过统计）
    
    # 动态抽样配置（每epoch抽样）
    'enable_dynamic_sampling': True,   # 是否启用训练集的动态抽样
    'sampling_ratio': 0.3,            # 每epoch随机抽样的数据比例
    
    # 🔥 新增：位置信息特征配置
    'enable_position_features': False,  # 是否启用位置信息特征（默认禁用）
    'raster_size': (278, 130),         # 图像尺寸 (height, width)，用于位置归一化
    
    # 🔥 新增：未来气象数据特征配置  
    'enable_future_weather': False,    # 是否启用未来气象数据特征（默认禁用）
    'weather_channels': list(range(1, 13)),  # 气象数据通道索引：第2-13波段（索引1-12）
}

# =============================================================================
# 自定义动态抽样数据集类
# =============================================================================

class DynamicSamplingSubset(Dataset):
    """
    支持动态抽样的数据集子集（简化版）
    每个epoch从平衡的数据集中随机抽样指定比例的数据
    由于底层数据集已经是1:1平衡的，随机抽样会保持大致相同的比例
    """
    def __init__(self, dataset, full_indices, sampling_ratio=1.0, enable_dynamic_sampling=False):
        """
        Args:
            dataset: 原始数据集（已经是1:1平衡的）
            full_indices: 完整的索引列表
            sampling_ratio: 每epoch使用的数据比例 (0.0-1.0)
            enable_dynamic_sampling: 是否启用动态抽样
        """
        self.dataset = dataset
        self.full_indices = full_indices
        self.sampling_ratio = sampling_ratio
        self.enable_dynamic_sampling = enable_dynamic_sampling
        
        # 当前使用的索引
        if enable_dynamic_sampling and sampling_ratio < 1.0:
            self.current_indices = self._sample_indices(epoch_seed=42)
        else:
            self.current_indices = full_indices
            
        print(f"📊 DynamicSamplingSubset初始化:")
        print(f"   总索引: {len(full_indices)}")
        print(f"   当前使用: {len(self.current_indices)}")
        print(f"   抽样比例: {sampling_ratio:.1%}")
        print(f"   动态抽样: {'启用' if enable_dynamic_sampling else '禁用'}")
    
    def _sample_indices(self, epoch_seed):
        """根据epoch种子随机抽样索引"""
        if not self.enable_dynamic_sampling or self.sampling_ratio >= 1.0:
            return self.full_indices
            
        # 设置随机种子确保可重复性
        np.random.seed(epoch_seed)
        random.seed(epoch_seed)
        
        # 计算抽样数量
        sample_size = int(len(self.full_indices) * self.sampling_ratio)
        sample_size = max(1, sample_size)  # 至少保证1个样本
        sample_size = min(sample_size, len(self.full_indices))  # 不超过可用数量
        
        # 随机抽样
        sampled_indices = np.random.choice(self.full_indices, size=sample_size, replace=False)
        return sampled_indices.tolist()
    
    def resample_for_epoch(self, epoch):
        """为新epoch重新抽样"""
        if not self.enable_dynamic_sampling:
            return
            
        # old_size = len(self.current_indices)
        self.current_indices = self._sample_indices(epoch_seed=42 + epoch)
        # new_size = len(self.current_indices)
        
        # print(f"🔄 Epoch {epoch+1}: 重新抽样完成 {old_size} → {new_size} 样本 (比例: {self.sampling_ratio:.1%})")
    
    def __len__(self):
        return len(self.current_indices)
    
    def __getitem__(self, idx):
        # 将当前索引映射到原始数据集的实际索引
        actual_idx = self.current_indices[idx]
        return self.dataset[actual_idx]

# =============================================================================
# 工具函数
# =============================================================================

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def worker_init_fn(worker_id):
    """DataLoader worker初始化函数，确保多进程的可重复性"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class FIRMSNormalizer:
    """FIRMS数据归一化器"""
    
    def __init__(self, method='divide_by_100', firms_min=None, firms_max=None):
        self.method = method
        self.firms_min = firms_min
        self.firms_max = firms_max
        self.fitted = False
        
    def fit(self, data_loader):
        """拟合归一化器"""
        if self.firms_min is not None and self.firms_max is not None:
            print(f"🚀 使用指定的FIRMS数据范围: [{self.firms_min}, {self.firms_max}]")
            if self.method == 'log1p_minmax':
                self.global_min = np.log1p(self.firms_min)
                self.global_max = np.log1p(self.firms_max)
            elif self.method == 'divide_by_100':
                self.global_min = self.firms_min / 100.0
                self.global_max = self.firms_max / 100.0
            else:
                self.global_min = self.firms_min
                self.global_max = self.firms_max
            self.fitted = True
            print(f"✅ 归一化器初始化完成 (变换后范围: {self.global_min:.2f}-{self.global_max:.2f})")
            return
            
        print("🔧 收集FIRMS数据进行归一化拟合...")
        firms_values = []
        
        # 简化数据收集过程，减少频繁的进度显示以提高性能
        # progress = SimpleProgressTracker()
        for i, batch in enumerate(data_loader):
            # 只在关键节点显示进度，而不是每个batch
            if i % max(1, len(data_loader) // 10) == 0:  # 每10%显示一次
                print(f"📊 收集FIRMS数据进度: {i+1}/{len(data_loader)} ({100*(i+1)/len(data_loader):.0f}%)", end='\r')
            # progress.update(i+1, len(data_loader), "📊 收集FIRMS数据")
            past, future, _ = batch
            firms_data = past[:, 0, :]  # FIRMS通道 (B, T)
            firms_values.append(firms_data.numpy())
        
        print()  # 换行
        
        all_firms = np.concatenate(firms_values, axis=0).flatten()
        valid_firms = all_firms[all_firms != 255]  # 过滤掉NoData值(255)
        
        if self.method == 'log1p_minmax':
            log_firms = np.log1p(valid_firms)
            self.global_min = log_firms.min()
            self.global_max = log_firms.max()
        elif self.method == 'divide_by_100':
            divide_firms = valid_firms / 100.0
            self.global_min = divide_firms.min()
            self.global_max = divide_firms.max()
        else:
            self.global_min = valid_firms.min()
            self.global_max = valid_firms.max()
            
        self.fitted = True
        print(f"✅ {self.method.upper()}归一化完成 (范围: {self.global_min:.2f}-{self.global_max:.2f})")
        
    def normalize(self, firms_data):
        """归一化FIRMS数据"""
        if not self.fitted:
            raise ValueError("归一化器尚未拟合，请先调用fit()方法")
            
        if self.method == 'log1p_minmax':
            log1p_data = torch.log1p(firms_data)
            if self.global_max > self.global_min:
                return (log1p_data - self.global_min) / (self.global_max - self.global_min)
            else:
                return log1p_data
        elif self.method == 'divide_by_100':
            return firms_data / 100.0
        else:
            raise ValueError(f"不支持的归一化方法: {self.method}")
    
    def transform_tensor(self, tensor_data):
        """为tensor数据应用归一化变换（兼容方法）"""
        return self.normalize(tensor_data)
    
    def inverse_transform_numpy(self, normalized_data):
        """对归一化后的numpy数据进行反变换"""
        if not self.fitted:
            raise ValueError("归一化器尚未拟合，请先调用fit()方法")
        
        if isinstance(normalized_data, torch.Tensor):
            normalized_data = normalized_data.cpu().numpy()
        
        if self.method == 'log1p_minmax':
            if self.global_max > self.global_min:
                # 反归一化: y = x * (max - min) + min
                log_data = normalized_data * (self.global_max - self.global_min) + self.global_min
            else:
                log_data = normalized_data
            # 反log1p变换: expm1(log_data)
            return np.expm1(log_data)
        elif self.method == 'divide_by_100':
            return normalized_data * 100.0
        else:
            raise ValueError(f"不支持的归一化方法: {self.method}")

def add_position_features(data, metadata_list, raster_size):
    """
    为数据添加位置信息特征
    
    Args:
        data: 输入数据 (batch_size, channels, time_steps)
        metadata_list: 元数据列表，包含位置信息
        raster_size: 图像尺寸 (height, width)
    
    Returns:
        添加位置特征后的数据 (batch_size, channels+1, time_steps)
    """
    batch_size, channels, time_steps = data.shape
    height, width = raster_size
    
    # 创建位置特征张量
    position_features = torch.zeros(batch_size, 1, time_steps, device=data.device)
    
    for i, metadata in enumerate(metadata_list):
        # 🔥 修复：正确从metadata中提取位置信息
        try:
            if isinstance(metadata, dict):
                # 如果metadata是字典格式（从dataload.py的_parse_dataset_key返回）
                row = metadata.get('row', 0)
                col = metadata.get('col', 0)
            elif hasattr(metadata, '__len__') and len(metadata) >= 3:
                # 如果metadata是列表/元组格式
                if len(metadata) >= 3:
                    # 尝试不同的metadata格式
                    # 格式1: [date_int, row, col, ...]
                    try:
                        row, col = int(metadata[1]), int(metadata[2])
                    except (ValueError, IndexError):
                        # 格式2: [date_int, firms_value, row, col, ...]
                        try:
                            row, col = int(metadata[2]), int(metadata[3])
                        except (ValueError, IndexError):
                            row, col = 0, 0
                else:
                    row, col = 0, 0
            else:
                # 如果metadata是单个值（可能是date_int）
                row, col = 0, 0
        except Exception as e:
            print(f"⚠️ 位置信息提取失败: {e}, metadata: {metadata}")
            row, col = 0, 0
        
        # 归一化位置坐标到0-1范围
        norm_row = row / (height - 1) if height > 1 else 0.0
        norm_col = col / (width - 1) if width > 1 else 0.0
        
        # 将归一化的位置信息编码为单一值 (可以使用不同的编码方式)
        # 这里使用简单的线性组合：row_weight * norm_row + col_weight * norm_col
        position_value = 0.5 * norm_row + 0.5 * norm_col
        
        # 将位置特征应用到所有时间步
        position_features[i, 0, :] = position_value
    
    # 将位置特征拼接到原始数据
    enhanced_data = torch.cat([data, position_features], dim=1)
    return enhanced_data

def add_weather_features(past_data, future_data, weather_channels):
    """
    从future数据中提取气象特征并添加到past数据
    
    Args:
        past_data: 过去数据 (batch_size, channels, past_time_steps)
        future_data: 未来数据 (batch_size, channels, future_time_steps)  
        weather_channels: 气象数据通道索引列表
    
    Returns:
        添加气象特征后的past数据 (batch_size, channels+len(weather_channels), past_time_steps)
    """
    batch_size, channels, past_time_steps = past_data.shape
    future_time_steps = future_data.shape[2]
    
    # 提取未来的气象数据 (batch_size, len(weather_channels), future_time_steps)
    future_weather = future_data[:, weather_channels, :]
    
    # 将未来气象数据重复或插值到past时间步长度
    if future_time_steps != past_time_steps:
        # 使用线性插值调整时间维度
        future_weather = F.interpolate(
            future_weather, 
            size=past_time_steps, 
            mode='linear', 
            align_corners=False
        )
    
    # 将气象特征拼接到past数据
    enhanced_past = torch.cat([past_data, future_weather], dim=1)
    return enhanced_past

def normalize_batch(past, future, firms_normalizer=None, metadata_list=None):
    """
    对批次数据进行归一化处理，并可选地添加位置信息和气象数据特征
    
    Args:
        past: 过去数据 (batch_size, channels, past_time_steps)
        future: 未来数据 (batch_size, channels, future_time_steps)
        firms_normalizer: FIRMS数据归一化器
        metadata_list: 元数据列表，用于提取位置信息
    
    Returns:
        处理后的 (past, future) 数据元组
    """
    # 🔥 关键：先处理所有通道的NaN值，将其替换为0
    nan_mask_past = torch.isnan(past)
    past[nan_mask_past] = 0.0
    nan_mask_future = torch.isnan(future)
    future[nan_mask_future] = 0.0
    
    # 对第0个通道（FIRMS）进行归一化（past和future都要）
    if firms_normalizer is not None:
        past[:, 0, :] = firms_normalizer.normalize(past[:, 0, :])
        future[:, 0, :] = firms_normalizer.normalize(future[:, 0, :])
    
    # 🔥 新增：添加位置信息特征
    if DATA_CONFIG['enable_position_features'] and metadata_list is not None:
        past = add_position_features(past, metadata_list, DATA_CONFIG['raster_size'])
        # 注意：future数据通常不需要添加位置特征，因为位置信息主要用于输入
        
    # 🔥 新增：添加未来气象数据特征
    if DATA_CONFIG['enable_future_weather']:
        past = add_weather_features(past, future, DATA_CONFIG['weather_channels'])
    
    return past, future

def load_model(model_name, configs, model_type='standard'):
    """动态加载模型（统一使用model_zoo）"""
    try:
        # 检查特殊依赖
        if model_name in ['Mamba', 'Reformer', 'Transformer', 'iTransformer', 's_mamba']:
            try:
                import mamba_ssm
            except ImportError:
                print(f"⚠️ 模型 {model_name} 需要 mamba_ssm 库")
                print(f"💡 建议使用 mamba_env 环境: conda activate mamba_env")
                raise ImportError(f"模型 {model_name} 需要 mamba_ssm 库，请在 mamba_env 环境中运行")
        
        # 统一使用model_zoo，通过configs中的参数区分标准/10x模型
        model_zoo_path = os.path.join(os.getcwd(), 'model_zoo')
        module_name = f'model_zoo.{model_name}'
        
        if model_zoo_path not in sys.path:
            sys.path.insert(0, model_zoo_path)
        
        module = importlib.import_module(module_name)
        Model = getattr(module, 'Model')
        
        return Model(configs), model_type
    except Exception as e:
        print(f"加载{model_type}模型 {model_name} 失败: {e}")
        raise

def calculate_detailed_metrics(output, target):
    """计算详细的回归和二分类指标，包括MSE、MAE、PR-AUC"""
    # 原始输出值用于回归指标
    output_raw = output.view(-1).cpu().numpy()
    target_np = target.view(-1).cpu().numpy()
    
    # 计算MSE和MAE（回归指标，使用原始输出值）
    mse = np.mean((output_raw - target_np) ** 2)
    mae = np.mean(np.abs(output_raw - target_np))
    
    # Sigmoid处理后的概率值用于分类指标
    pred_probs = torch.sigmoid(output).view(-1).cpu().numpy()
    pred_binary = (pred_probs > 0.5).astype(int)
    target_binary = (target_np > 0).astype(int)
    
    unique_targets = np.unique(target_binary)
    if len(unique_targets) < 2:
        return 0.0, 0.0, 0.0, 0.0, mse, mae
    
    try:
        precision = precision_score(target_binary, pred_binary, average='binary', zero_division=0)
        recall = recall_score(target_binary, pred_binary, average='binary', zero_division=0)
        f1 = f1_score(target_binary, pred_binary, average='binary', zero_division=0)
        pr_auc = average_precision_score(target_binary, pred_probs)
    except Exception as e:
        print(f"计算指标时出错: {e}")
        return 0.0, 0.0, 0.0, 0.0, mse, mae
    
    return precision, recall, f1, pr_auc, mse, mae

def calculate_optimal_f1_metrics(output, target):
    """计算F1最优阈值下的详细指标，用于测试阶段 - 调试版本"""
    # 原始输出值用于回归指标
    output_raw = output.view(-1).cpu().numpy()
    target_np = target.view(-1).cpu().numpy()
    
    # 计算MSE和MAE（回归指标，使用原始输出值）
    mse = np.mean((output_raw - target_np) ** 2)
    mae = np.mean(np.abs(output_raw - target_np))
    
    # Sigmoid处理后的概率值用于分类指标
    pred_probs = torch.sigmoid(output).view(-1).cpu().numpy()
    target_binary = (target_np > 0).astype(int)
    
    # 🔍 调试信息：分析输入数据特性
    print(f"   🔍 数据统计:")
    print(f"      预测样本数: {len(pred_probs)}")
    print(f"      真实阳性样本数: {np.sum(target_binary)}")
    print(f"      真实阳性比例: {np.sum(target_binary) / len(target_binary):.4f}")
    print(f"      预测概率范围: [{np.min(pred_probs):.4f}, {np.max(pred_probs):.4f}]")
    print(f"      预测概率均值: {np.mean(pred_probs):.4f}")
    print(f"      预测概率std: {np.std(pred_probs):.4f}")
    
    unique_targets = np.unique(target_binary)
    if len(unique_targets) < 2:
        return 0.0, 0.0, 0.0, 0.0, mse, mae
    
    try:
        # 计算PR-AUC
        pr_auc = average_precision_score(target_binary, pred_probs)
        
        # 寻找F1最优阈值
        thresholds = np.linspace(0, 1, 100)  # 使用1000个阈值点进行搜索
        best_f1 = 0.0
        best_precision = 0.0
        best_recall = 0.0
        best_threshold = 0.5
        
        # 🔍 调试：记录所有阈值的指标
        all_recalls = []
        all_precisions = []
        all_f1s = []
        
        for threshold in thresholds:
            pred_binary_thresh = (pred_probs > threshold).astype(int)
            
            # 🔍 防止除零错误，添加更详细的检查
            tp = np.sum((pred_binary_thresh == 1) & (target_binary == 1))
            fp = np.sum((pred_binary_thresh == 1) & (target_binary == 0))
            fn = np.sum((pred_binary_thresh == 0) & (target_binary == 1))
            tn = np.sum((pred_binary_thresh == 0) & (target_binary == 0))
            
            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0.0
                
            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0.0
                
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            
            all_recalls.append(recall)
            all_precisions.append(precision)
            all_f1s.append(f1)
            
            if f1 > best_f1:
                best_f1 = f1
                best_precision = precision
                best_recall = recall
                best_threshold = threshold
        
        # 🔍 调试信息：分析recall的分布
        all_recalls = np.array(all_recalls)
        unique_recalls = np.unique(all_recalls)
        print(f"      发现{len(unique_recalls)}个不同的recall值")
        print(f"      Recall范围: [{np.min(all_recalls):.4f}, {np.max(all_recalls):.4f}]")
        print(f"      最高recall: {np.max(all_recalls):.6f}")
        print(f"      最优F1阈值: {best_threshold:.3f} (F1={best_f1:.4f})")
        
        # 🔍 如果所有recall都相同，说明有问题
        if len(unique_recalls) == 1:
            print(f"      ⚠️  警告：所有阈值的recall都相同 = {unique_recalls[0]:.6f}")
            print(f"      可能的原因：模型预测过于集中或数据分布异常")
            
        # 🔍 分析阈值分布
        recall_counts = {}
        for r in all_recalls:
            r_rounded = round(r, 6)
            recall_counts[r_rounded] = recall_counts.get(r_rounded, 0) + 1
        
        print(f"      Top 5 recall值出现频率:")
        for r, count in sorted(recall_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"         {r:.6f}: {count}次")
        
    except Exception as e:
        print(f"计算最优F1指标时出错: {e}")
        return 0.0, 0.0, 0.0, 0.0, mse, mae
    
    return best_precision, best_recall, best_f1, pr_auc, mse, mae

class Config:
    """配置类 - 修复类型安全问题"""
    def __init__(self, model_name, model_type='standard'):
        self.model_name = model_name  # 添加模型名称属性
        self.model_type = model_type
        config = TRAINING_CONFIG[model_type]
        
        # 基本训练参数 - 确保类型安全
        self.epochs = int(config['epochs'])
        self.batch_size = int(config['batch_size'])
        self.learning_rate = float(config['learning_rate'])
        self.weight_decay = float(config['weight_decay'])
        self.T_0 = int(config['T_0'])
        self.T_mult = int(config['T_mult'])
        self.eta_min = float(config['eta_min'])
        self.max_grad_norm = float(config['max_grad_norm'])
        
        # 序列参数 - 确保是整数类型，避免Config对象问题
        self.seq_len = int(TRAINING_CONFIG['seq_len'])
        self.pred_len = int(TRAINING_CONFIG['pred_len'])
        self.label_len = 0  # 默认标签长度
        
        # 根据模型类型获取配置（使用统一适配器）
        try:
            from model_adapter_unified import get_unified_model_configs
            model_configs = get_unified_model_configs(model_name, model_type)
            
            # 安全地设置配置，确保数值类型正确
            for key, value in model_configs.items():
                if key in ['seq_len', 'pred_len']:
                    continue  # 跳过，使用我们已经设置的固定值
                elif isinstance(value, (int, float, str, bool)):
                    setattr(self, key, value)
                elif value is None:
                    setattr(self, key, None)
                else:
                    # 对于复杂类型，尝试转换为基本类型
                    try:
                        if isinstance(value, list):
                            setattr(self, key, value)
                        else:
                            setattr(self, key, value)
                    except:
                        print(f"⚠️  跳过配置 {key}={value} (类型: {type(value)})")
                        
        except Exception as e:
            print(f"⚠️  动态配置导入失败: {e}，使用默认配置")
            # 使用默认配置
            if model_type == 'standard':
                self.d_model = 512
                self.n_heads = 8
                self.d_ff = 2048
                self.e_layers = 2
                self.d_layers = 2
                self.d_state = 16
                self.d_conv = 4
                self.expand = 2
            else:  # 10x
                self.d_model = 2048
                self.n_heads = 32
                self.d_ff = 2048
                self.e_layers = 4
                self.d_layers = 4
                self.d_state = 32
                self.d_conv = 8
                self.expand = 4
            
            # 通用模型参数
            self.dropout = 0.1
            self.activation = 'gelu'
            self.output_attention = False
            self.enc_in = 39
            self.dec_in = 39
            self.c_out = 39
            self.embed = 'timeF'
            self.freq = 'd'
            self.factor = 1
            self.moving_avg = 25
            self.channel_independence = False
            self.use_norm = True
            self.distil = True
            self.label_len = 3 if model_name in ['Autoformer', 'Autoformer_M'] else 0
        
        # 添加新模型需要的特殊配置
        self.task_name = 'long_term_forecast'  # 新模型普遍需要这个参数
        
        # 为特定模型添加特殊配置
        if model_name == 'DLinear':
            self.moving_avg = 25  # DLinear需要moving_avg用于series_decomp
            self.individual = False  # DLinear的individual参数
            
        elif model_name == 'CrossLinear':
            self.features = 'M'  # CrossLinear需要features参数
            self.patch_len = 16  # CrossLinear需要patch相关参数
            self.alpha = 0.5
            self.beta = 0.5
            
        elif model_name == 'TimesNet':
            self.top_k = 5  # TimesNet需要的参数
            self.num_kernels = 6
            
        elif model_name == 'Mamba':
            # Mamba需要的特殊参数已经在基础配置中设置了
            pass
        
        # FIRMS数据归一化参数
        self.normalize_firms = True
        self.firms_normalization_method = 'divide_by_100'
        self.binarization_threshold = 0.0
        self.firms_min = int(DATA_CONFIG['firms_min'])
        self.firms_max = int(DATA_CONFIG['firms_max'])
        
        # Focal Loss参数  
        self.focal_alpha = float(TRAINING_CONFIG['focal_alpha'])
        self.focal_gamma = float(TRAINING_CONFIG['focal_gamma'])
        
        # 多任务学习参数
        self.firms_weight = float(MULTITASK_CONFIG['firms_weight'])
        self.other_drivers_weight = float(MULTITASK_CONFIG['other_drivers_weight'])
        self.ignore_zero_values = MULTITASK_CONFIG['ignore_zero_values']
        self.loss_function = MULTITASK_CONFIG['loss_function']
        self.loss_type = MULTITASK_CONFIG['loss_type']  # 新增：损失函数类型选择
        
        # 数据集划分
        self.train_years = DATA_CONFIG['train_years']
        self.val_years = DATA_CONFIG['val_years']
        self.test_years = DATA_CONFIG['test_years']
        
        # 🔥 新增：动态更新模型通道配置
        self.update_model_channels()
    
    # 🔥 新增：动态计算输入通道数
    def calculate_input_channels(self):
        """
        根据配置动态计算输入通道数
        基础通道数 + 位置特征通道数 + 气象数据通道数
        """
        base_channels = 39  # 基础通道数
        additional_channels = 0
        
        # 位置信息特征 (+1 通道) - 优先使用config对象属性，否则使用全局配置
        enable_position = getattr(self, 'enable_position_features', DATA_CONFIG['enable_position_features'])
        if enable_position:
            additional_channels += 1
            
        # 未来气象数据特征 - 优先使用config对象属性，否则使用全局配置
        enable_weather = getattr(self, 'enable_future_weather', DATA_CONFIG['enable_future_weather'])
        if enable_weather:
            weather_channels = getattr(self, 'weather_channels', DATA_CONFIG['weather_channels'])
            additional_channels += len(weather_channels)
            
        return base_channels + additional_channels
    
    def update_model_channels(self):
        """更新模型的输入/输出通道配置"""
        # 动态计算输入通道数
        dynamic_enc_in = self.calculate_input_channels()
        
        # 更新编码器输入通道数
        self.enc_in = dynamic_enc_in
        
        # 解码器输入通道数通常与编码器一致
        self.dec_in = dynamic_enc_in
        
        # 输出通道数保持为39（预测所有原始通道）
        self.c_out = 39
        
        # 打印通道信息以便调试 - 使用config对象属性而不是全局配置
        features_info = []
        enable_position = getattr(self, 'enable_position_features', DATA_CONFIG['enable_position_features'])
        enable_weather = getattr(self, 'enable_future_weather', DATA_CONFIG['enable_future_weather'])
        
        if enable_position:
            features_info.append("位置信息(+1)")
        if enable_weather:
            weather_channels = getattr(self, 'weather_channels', DATA_CONFIG['weather_channels'])
            features_info.append(f"气象数据(+{len(weather_channels)})")
        
        if features_info:
            print(f"🔧 {self.model_name} 动态通道配置: {self.enc_in}输入 -> {self.c_out}输出 (额外特征: {', '.join(features_info)})")
        else:
            print(f"🔧 {self.model_name} 标准通道配置: {self.enc_in}输入 -> {self.c_out}输出")

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MultiTaskFocalLoss(nn.Module):
    """
    多任务Focal Loss：
    - 对FIRMS通道（第0通道）使用Focal Loss进行二分类
    - 对其他驱动因素使用回归损失（MSE/Huber/MAE）
    - 支持权重调节和忽略0值功能
    """
    def __init__(self, firms_weight=1.0, other_drivers_weight=0.1, 
                 focal_alpha=0.25, focal_gamma=2.0,
                 ignore_zero_values=True, regression_loss='mse'):
        super(MultiTaskFocalLoss, self).__init__()
        self.firms_weight = firms_weight
        self.other_drivers_weight = other_drivers_weight
        self.ignore_zero_values = ignore_zero_values
        
        # FIRMS的Focal Loss
        # self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')
        self.focal_loss = nn.BCELoss()
        # 其他驱动因素的回归损失函数
        if regression_loss == 'huber':
            self.regression_loss_fn = nn.HuberLoss(reduction='none')
        elif regression_loss == 'mse':
            self.regression_loss_fn = nn.MSELoss(reduction='none')
        elif regression_loss == 'mae':
            self.regression_loss_fn = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f"不支持的回归损失函数类型: {regression_loss}")
        
        self.regression_loss_type = regression_loss
    
    def forward(self, predictions, targets):
        """
        计算多任务损失
        
        Args:
            predictions: (B, T, C) 模型预测结果，C可能大于39（如果有额外特征）
            targets: (B, T, 39) 真实标签，始终是39个通道
            
        Returns:
            total_loss: 总损失
            loss_components: 各组件损失字典
        """
        batch_size, seq_len, pred_channels = predictions.shape
        _, _, target_channels = targets.shape
        
        # 🔥 关键修复：如果预测通道数大于目标通道数，只取前target_channels个通道
        # 这是因为额外的通道（如气象数据）已经作为输入特征，不应该计算损失
        if pred_channels > target_channels:
            predictions = predictions[:, :, :target_channels]
        #   print(f"🔧 损失计算：预测通道数({pred_channels}) > 目标通道数({target_channels})，只计算前{target_channels}个通道的损失")
        
        # 分离FIRMS和其他驱动因素
        firms_pred = predictions[:, :, 0]      # (B, T) - FIRMS通道用于二分类
        firms_target = targets[:, :, 0]        # (B, T)
        other_pred = predictions[:, :, 1:]     # (B, T, 38) - 其他通道用于回归
        other_target = targets[:, :, 1:]       # (B, T, 38)
        
        # 1. 计算FIRMS的Focal Loss（二分类）
        # 将FIRMS目标转换为二分类标签（>0为1，=0为0）
        firms_binary_target = (firms_target > 0).float()
        firms_pred = torch.sigmoid(firms_pred)  # 使用focal loss的时候不需要sigmoid，因为sigmoid已经内置到focal loss中
        firms_loss = self.focal_loss(firms_pred, firms_binary_target) * self.firms_weight
        
        # 2. 计算其他驱动因素的回归损失
        other_loss = self.regression_loss_fn(other_pred, other_target)  # (B, T, 38)
        
        if self.ignore_zero_values:
            # 创建非零值掩码，忽略0值
            non_zero_mask = (other_target != 0.0).float()  # (B, T, 38)
            
            # 计算有效样本数
            valid_samples = non_zero_mask.sum()
            
            if valid_samples > 0:
                # 只对非零值计算损失
                masked_loss = other_loss * non_zero_mask
                other_loss = masked_loss.sum() / valid_samples
            else:
                # 如果没有有效样本，损失为0
                other_loss = torch.tensor(0.0, device=predictions.device)
        else:
            # 不忽略0值，直接计算平均损失
            other_loss = other_loss.mean()
        
        other_loss = other_loss * self.other_drivers_weight
        
        # 总损失
        total_loss = firms_loss + other_loss
        
        # 返回损失组件信息
        loss_components = {
            'total_loss': total_loss.item(),
            'firms_loss': firms_loss.item(),
            'other_drivers_loss': other_loss.item(),
            'firms_weight': self.firms_weight,
            'other_drivers_weight': self.other_drivers_weight,
            # 'focal_alpha': self.focal_loss.alpha,  # 使用BCELoss的时候不需要focal_alpha和focal_gamma
            # 'focal_gamma': self.focal_loss.gamma,
            'regression_loss_type': self.regression_loss_type,
            'loss_type': 'focal'  # 新增：损失函数类型标识
        }
        # print(firms_loss, other_loss)
        return firms_loss, loss_components  # total_loss, loss_components

class MultiTaskKLDivLoss(nn.Module):
    """
    多任务KL散度Loss：
    - 对FIRMS通道（第0通道）使用KL散度进行分类
    - 对其他驱动因素使用KL散度进行回归
    - 支持权重调节和忽略0值功能
    """
    def __init__(self, firms_weight=1.0, other_drivers_weight=0.1, 
                 ignore_zero_values=True, temperature=1.0, epsilon=1e-8):
        super(MultiTaskKLDivLoss, self).__init__()
        self.firms_weight = firms_weight
        self.other_drivers_weight = other_drivers_weight
        self.ignore_zero_values = ignore_zero_values
        self.temperature = temperature  # 温度参数，用于控制分布的平滑度
        self.epsilon = epsilon  # 防止数值不稳定的小常数
        
        # KL散度损失函数（reduction='none'以便手动处理）
        self.kldiv_loss = nn.KLDivLoss(reduction='none')
    
    def _to_probability_distribution(self, x, is_classification=False):
        """
        将输入转换为概率分布
        
        Args:
            x: 输入张量
            is_classification: 是否为分类任务（FIRMS通道）
            
        Returns:
            概率分布张量
        """
        if is_classification:
            # 对于分类任务，使用sigmoid+归一化
            # x shape: (...,) 或 (..., 1)
            if x.dim() > 0 and x.shape[-1] == 1:
                x = x.squeeze(-1)  # 移除最后一维如果是1
            
            prob = torch.sigmoid(x / self.temperature)
            # 创建二项分布：[1-p, p]
            prob_neg = 1 - prob
            prob_dist = torch.stack([prob_neg, prob], dim=-1)  # (..., 2)
            # 归一化确保是概率分布
            prob_dist = prob_dist / (prob_dist.sum(dim=-1, keepdim=True) + self.epsilon)
        else:
            # 对于回归任务，将值转换为正值然后归一化
            # 使用softplus确保正值：softplus(x) = log(1 + exp(x))
            positive_vals = F.softplus(x / self.temperature)
            # 归一化为概率分布
            prob_dist = positive_vals / (positive_vals.sum(dim=-1, keepdim=True) + self.epsilon)
        
        # 添加小常数防止log(0)
        prob_dist = prob_dist + self.epsilon
        prob_dist = prob_dist / prob_dist.sum(dim=-1, keepdim=True)
        
        return prob_dist
    
    def forward(self, predictions, targets):
        """
        计算多任务KL散度损失
        
        Args:
            predictions: (B, T, C) 模型预测结果，C可能大于39（如果有额外特征）
            targets: (B, T, 39) 真实标签，始终是39个通道
            
        Returns:
            total_loss: 总损失
            loss_components: 各组件损失字典
        """
        batch_size, seq_len, pred_channels = predictions.shape
        _, _, target_channels = targets.shape
        
        # 🔥 关键修复：如果预测通道数大于目标通道数，只取前target_channels个通道
        # 这是因为额外的通道（如气象数据）已经作为输入特征，不应该计算损失
        if pred_channels > target_channels:
            predictions = predictions[:, :, :target_channels]
            print(f"🔧 KL散度损失计算：预测通道数({pred_channels}) > 目标通道数({target_channels})，只计算前{target_channels}个通道的损失")
        
        # 分离FIRMS和其他驱动因素
        firms_pred = predictions[:, :, 0]      # (B, T) - FIRMS通道
        firms_target = targets[:, :, 0]        # (B, T)
        other_pred = predictions[:, :, 1:]     # (B, T, 38) - 其他通道
        other_target = targets[:, :, 1:]       # (B, T, 38)
        
        # 1. 计算FIRMS的KL散度损失（分类任务）
        # 将FIRMS目标转换为二分类标签（>0为1，=0为0）
        firms_binary_target = (firms_target > 0).float()
        
        # 转换为概率分布
        firms_pred_dist = self._to_probability_distribution(firms_pred, is_classification=True)  # (B, T, 2)
        firms_target_dist = self._to_probability_distribution(firms_binary_target, is_classification=True)  # (B, T, 2)
        
        # 计算KL散度：KL(target || pred)
        firms_kl = self.kldiv_loss(firms_pred_dist.log(), firms_target_dist)  # (B, T, 2)
        firms_loss = firms_kl.sum(dim=-1).mean() * self.firms_weight  # 对分布维度求和，然后平均
        
        # 2. 计算其他驱动因素的KL散度损失（回归任务）
        # 转换为概率分布
        other_pred_dist = self._to_probability_distribution(other_pred, is_classification=False)  # (B, T, 38)
        other_target_dist = self._to_probability_distribution(other_target, is_classification=False)  # (B, T, 38)
        
        # 计算KL散度
        other_kl = self.kldiv_loss(other_pred_dist.log(), other_target_dist)  # (B, T, 38)
        
        if self.ignore_zero_values:
            # 创建非零值掩码，忽略0值
            non_zero_mask = (other_target != 0.0).float()  # (B, T, 38)
            
            # 计算有效样本数
            valid_samples = non_zero_mask.sum()
            
            if valid_samples > 0:
                # 只对非零值计算损失
                masked_kl = other_kl * non_zero_mask
                other_loss = masked_kl.sum() / valid_samples
            else:
                # 如果没有有效样本，损失为0
                other_loss = torch.tensor(0.0, device=predictions.device)
        else:
            # 不忽略0值，直接计算平均损失
            other_loss = other_kl.mean()
        
        other_loss = other_loss * self.other_drivers_weight
        
        # 总损失
        total_loss = firms_loss + other_loss
        
        # 返回损失组件信息
        loss_components = {
            'total_loss': total_loss.item(),
            'firms_loss': firms_loss.item(),
            'other_drivers_loss': other_loss.item(),
            'firms_weight': self.firms_weight,
            'other_drivers_weight': self.other_drivers_weight,
            'temperature': self.temperature,
            'loss_type': 'kldiv'
        }
        
        return total_loss, loss_components

class MultiMetricEarlyStopping:
    """
    多指标Early Stopping：同时监控F1、Recall、PR-AUC
    任何一个指标提升都会重置计数器
    """
    def __init__(self, patience=7, min_delta=0.0001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_metrics = {
            'f1': 0.0,
            'recall': 0.0,
            'pr_auc': 0.0,
            'mae': float('inf')  # MAE越小越好
        }
        self.best_weights = None
        self.should_stop = False
    
    def __call__(self, metrics, model):
        """
        检查是否应该停止训练
        Args:
            metrics: dict包含'f1', 'recall', 'pr_auc', 'mae'
            model: 模型实例
        Returns:
            bool: 是否应该停止训练
        """
        f1_improved = metrics['f1'] > (self.best_metrics['f1'] + self.min_delta)
        recall_improved = metrics['recall'] > (self.best_metrics['recall'] + self.min_delta)
        pr_auc_improved = metrics['pr_auc'] > (self.best_metrics['pr_auc'] + self.min_delta)
        mae_improved = metrics['mae'] < (self.best_metrics['mae'] - self.min_delta)  # MAE越小越好
        
        # 任何一个指标提升就重置计数器
        if f1_improved or recall_improved or pr_auc_improved or mae_improved:
            # 更新最佳指标
            if f1_improved:
                self.best_metrics['f1'] = metrics['f1']
            if recall_improved:
                self.best_metrics['recall'] = metrics['recall']
            if pr_auc_improved:
                self.best_metrics['pr_auc'] = metrics['pr_auc']
            if mae_improved:
                self.best_metrics['mae'] = metrics['mae']
                
            self.counter = 0
            if self.restore_best_weights:
                self.save_checkpoint(model)
            print(f"📈 指标提升! F1: {metrics['f1']:.4f}, Recall: {metrics['recall']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}, MAE: {metrics['mae']:.6f}")
        else:
            self.counter += 1
            print(f"⏳ 无改善 ({self.counter}/{self.patience}): F1: {metrics['f1']:.4f}, Recall: {metrics['recall']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}, MAE: {metrics['mae']:.6f}")
        
        if self.counter >= self.patience:
            self.should_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                print("🔄 恢复最佳权重")
        
        return self.should_stop
    
    def save_checkpoint(self, model):
        """保存最佳权重"""
        self.best_weights = model.state_dict().copy()

class MultiTaskLoss(nn.Module):
    """多任务损失函数，支持对不同通道的预测结果进行加权损失计算"""
    
    def __init__(self, firms_weight=1.0, other_drivers_weight=0.1, 
                 ignore_zero_values=True, loss_function='huber'):
        super(MultiTaskLoss, self).__init__()
        self.firms_weight = firms_weight
        self.other_drivers_weight = other_drivers_weight
        self.ignore_zero_values = ignore_zero_values
        
        # 选择损失函数
        if loss_function == 'huber':
            self.loss_fn = nn.HuberLoss(reduction='none')
        elif loss_function == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif loss_function == 'mae':
            self.loss_fn = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f"不支持的损失函数类型: {loss_function}")
    
    def forward(self, predictions, targets):
        """
        计算多任务损失
        
        Args:
            predictions: (B, T, C) 模型预测结果，C可能大于39（如果有额外特征）
            targets: (B, T, 39) 真实标签，始终是39个通道
            
        Returns:
            total_loss: 总损失
            loss_components: 各组件损失字典
        """
        batch_size, seq_len, pred_channels = predictions.shape
        _, _, target_channels = targets.shape
        
        # 🔥 关键修复：如果预测通道数大于目标通道数，只取前target_channels个通道
        # 这是因为额外的通道（如气象数据）已经作为输入特征，不应该计算损失
        if pred_channels > target_channels:
            predictions = predictions[:, :, :target_channels]
            print(f"🔧 多任务损失计算：预测通道数({pred_channels}) > 目标通道数({target_channels})，只计算前{target_channels}个通道的损失")
        
        # 分离FIRMS和其他驱动因素
        firms_pred = predictions[:, :, 0:1]  # (B, T, 1)
        firms_target = targets[:, :, 0:1]    # (B, T, 1)
        other_pred = predictions[:, :, 1:]   # (B, T, 38)
        other_target = targets[:, :, 1:]     # (B, T, 38)
        
        # 计算FIRMS损失
        firms_loss = self.loss_fn(firms_pred, firms_target)  # (B, T, 1)
        firms_loss = firms_loss.mean() * self.firms_weight
        
        # 计算其他驱动因素损失
        other_loss = self.loss_fn(other_pred, other_target)  # (B, T, 38)
        
        if self.ignore_zero_values:
            # 创建非零值掩码，忽略0值
            non_zero_mask = (other_target != 0.0).float()  # (B, T, 38)
            
            # 计算有效样本数
            valid_samples = non_zero_mask.sum()
            
            if valid_samples > 0:
                # 只对非零值计算损失
                masked_loss = other_loss * non_zero_mask
                other_loss = masked_loss.sum() / valid_samples
            else:
                # 如果没有有效样本，损失为0
                other_loss = torch.tensor(0.0, device=predictions.device)
        else:
            # 不忽略0值，直接计算平均损失
            other_loss = other_loss.mean()
        
        other_loss = other_loss * self.other_drivers_weight
        
        # 总损失
        total_loss = firms_loss + other_loss
        
        # 返回损失组件信息
        loss_components = {
            'total_loss': total_loss.item(),
            'firms_loss': firms_loss.item(),
            'other_drivers_loss': other_loss.item(),
            'firms_weight': self.firms_weight,
            'other_drivers_weight': self.other_drivers_weight,
            'loss_type': 'multitask'  # 新增：损失函数类型标识
        }
        # print(firms_loss, other_loss)
        return total_loss, loss_components

# =============================================================================
# 进度显示工具函数
# =============================================================================

class SimpleProgressTracker:
    """简化的进度跟踪器，模仿tqdm默认效果但去掉进度条"""
    def __init__(self):
        self.start_time = None
        
    def update(self, current, total, prefix="Progress", clear_on_complete=True):
        """
        更新进度显示 - tqdm风格但无进度条
        """
        if self.start_time is None:
            self.start_time = time.time()
            
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # 计算速度 (items/second)
        speed = current / elapsed_time if elapsed_time > 0 else 0
        
        # 计算百分比
        percent = int((current / total) * 100)
        
        # tqdm风格的显示格式
        if current == total:
            # 完成时的格式
            progress_text = f"\r{prefix}: {percent:3d}%|{current}/{total} [{self._format_time(elapsed_time)}, {speed:.2f}it/s]"
        else:
            # 进行中的格式，计算预估剩余时间
            if speed > 0:
                remaining_time = (total - current) / speed
                progress_text = f"\r{prefix}: {percent:3d}%|{current}/{total} [{self._format_time(elapsed_time)}<{self._format_time(remaining_time)}, {speed:.2f}it/s]"
            else:
                # 如果速度为0，使用简化格式
                progress_text = f"\r{prefix}: {percent:3d}%|{current}/{total} [{self._format_time(elapsed_time)}<?, ?it/s]"
        
        print(progress_text, end='', flush=True)
        
        # 完成后处理
        if current == total:
            if clear_on_complete:
                # 清除进度条
                print('\r' + ' ' * len(progress_text) + '\r', end='', flush=True)
            else:
                print()  # 保留最终状态并换行
    
    def _format_time(self, seconds):
        """格式化时间显示 - tqdm风格"""
        if seconds < 0:
            return "00s"
        elif seconds < 60:
            return f"{int(seconds):02d}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}:{minutes:02d}:00"

def print_dynamic_progress(current, total, prefix="Progress", show_percent=True):
    """
    兼容性函数 - 保持简单的动态进度显示
    """
    if show_percent:
        percent = (current / total) * 100
        progress_text = f"\r{prefix}: {current}/{total} ({percent:.1f}%)"
    else:
        progress_text = f"\r{prefix}: {current}/{total}"
    
    print(progress_text, end='', flush=True)
    
    # 完成后清除进度条
    if current == total:
        print('\r' + ' ' * len(progress_text) + '\r', end='', flush=True)

def save_epoch_metrics_to_log(epoch_metrics, log_file, model_name, model_type):
    """
    将每个epoch的训练和验证指标保存到日志文件
    """
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"详细训练日志 - {model_name} ({model_type})\n")
            f.write(f"{'='*80}\n")
            f.write(f"记录时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 写入表头
            f.write(f"{'Epoch':<6} {'Train_Loss':<11} {'Train_P':<8} {'Train_R':<8} {'Train_F1':<9} {'Train_PRAUC':<11} {'Train_MSE':<10} {'Train_MAE':<10} ")
            f.write(f"{'Val_Loss':<9} {'Val_P':<6} {'Val_R':<6} {'Val_F1':<7} {'Val_PRAUC':<9} {'Val_MSE':<8} {'Val_MAE':<8} {'LR':<10}\n")
            f.write("-" * 150 + "\n")
            
            # 写入每个epoch的数据
            for metrics in epoch_metrics:
                f.write(f"{metrics['epoch']:<6} ")
                f.write(f"{metrics['train_loss']:<11.6f} ")
                f.write(f"{metrics['train_precision']:<8.4f} ")
                f.write(f"{metrics['train_recall']:<8.4f} ")
                f.write(f"{metrics['train_f1']:<9.4f} ")
                f.write(f"{metrics['train_pr_auc']:<11.4f} ")
                f.write(f"{metrics['train_mse']:<10.6f} ")
                f.write(f"{metrics['train_mae']:<10.6f} ")
                f.write(f"{metrics['val_loss']:<9.6f} ")
                f.write(f"{metrics['val_precision']:<6.4f} ")
                f.write(f"{metrics['val_recall']:<6.4f} ")
                f.write(f"{metrics['val_f1']:<7.4f} ")
                f.write(f"{metrics['val_pr_auc']:<9.4f} ")
                f.write(f"{metrics['val_mse']:<8.6f} ")
                f.write(f"{metrics['val_mae']:<8.6f} ")
                f.write(f"{metrics['learning_rate']:<10.2e}\n")
            
            f.write("\n")
            
        print(f"📝 详细epoch日志已保存到: {log_file}")
        
    except Exception as e:
        print(f"⚠️ 保存epoch日志失败: {e}")

def save_structured_results_to_csv(structured_results, model_type):
    """
    将结构化测试结果保存为分类的CSV文件
    分别保存：best_f1.csv, best_recall.csv, final_epoch.csv
    每个CSV包含：Model, precision, recall, f1, pr_auc
    """
    if not structured_results:
        print("⚠️  没有结果可以保存")
        return
    
    # 确定保存目录
    if model_type == 'standard':
        save_dir = STANDARD_MODEL_DIR
    else:
        save_dir = MODEL_10X_DIR
    
    # 创建保存目录（如果不存在）
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 要保存的3种模型类型
    model_categories = ['f1', 'recall', 'final_epoch']
    classification_metrics = ['precision', 'recall', 'f1', 'pr_auc']  # 只保存分类指标
    
    saved_files = []
    
    for category in model_categories:
        # 准备CSV数据
        csv_data = []
        columns = ['Model'] + classification_metrics
        
        # 添加数据行
        for model_name, model_results in structured_results.items():
            if category in model_results and model_results[category]['precision'] is not None:
                row = [model_name]
                
                for metric_name in classification_metrics:
                    value = model_results[category][metric_name]
                    if value is not None:
                        row.append(f"{value:.6f}")
                    else:
                        row.append("N/A")
                
                csv_data.append(row)
        
        # 生成文件名并保存
        if category == 'final_epoch':
            filename = f"final_epoch.csv"
        else:
            filename = f"best_{category}.csv"
        
        csv_filepath = os.path.join(save_dir, filename)
        
        if csv_data:  # 只在有数据时保存
            df = pd.DataFrame(csv_data, columns=columns)
            df.to_csv(csv_filepath, index=False)
            saved_files.append(csv_filepath)
            
            print(f"📊 {filename}: {len(csv_data)} 个模型结果已保存")
        else:
            print(f"⚠️  {filename}: 没有可用数据")
    
    # 总结保存情况
    print(f"\n✅ 共保存 {len(saved_files)} 个CSV文件到: {save_dir}")
    for filepath in saved_files:
        print(f"   📄 {os.path.basename(filepath)}")
    
    print(f"\n📋 CSV文件结构说明:")
    print(f"   best_f1.csv: F1最佳模型的性能评价")
    print(f"   best_recall.csv: Recall最佳模型的性能评价")
    print(f"   final_epoch.csv: 最后epoch模型的性能评价")
    print(f"   每个文件包含: Model, precision, recall, f1, pr_auc")

# =============================================================================
# 核心训练和测试函数
# =============================================================================

def train_single_model(model_name, device, train_loader, val_loader, test_loader, firms_normalizer, model_type='standard', log_file=None):
    """训练单个模型"""
    print(f"\n🔥 训练{model_type}模型: {model_name}")
    
    config = Config(model_name, model_type)
    
    # 创建详细日志记录器
    epoch_metrics = []  # 记录每个epoch的指标
    
    # 初始化wandb (如果启用)
    wandb_run = None
    if TRAINING_CONFIG['use_wandb'] and WANDB_AVAILABLE:
        wandb_run = wandb.init(
            project="wildfire-forecasting-0708",
            name=f"{model_name}_{model_type}",
            config={
                "model_name": model_name,
                "model_type": model_type,
                "seq_len": config.seq_len,
                "pred_len": config.pred_len,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "focal_alpha": config.focal_alpha,
                "focal_gamma": config.focal_gamma,
                # 多任务学习配置
                "multitask_enabled": True,
                "firms_weight": config.firms_weight,
                "other_drivers_weight": config.other_drivers_weight,
                "ignore_zero_values": config.ignore_zero_values,
                "loss_function": config.loss_function,
            },
            reinit=True
        )
        print(f"✅ WandB初始化完成: {wandb_run.name}")
    
    # 使用统一适配器
    from model_adapter_unified import UnifiedModelAdapter
    adapter = UnifiedModelAdapter(config)
    
    try:
        model, _ = load_model(model_name, config, model_type)
        model = model.to(device)
    except Exception as e:
        print(f"❌ {model_type}模型 {model_name} 加载失败: {e}")
        if wandb_run:
            wandb_run.finish()
        return None
    
    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # 根据配置选择损失函数类型
    if config.loss_type == 'focal':
        # 使用多任务Focal Loss
        criterion = MultiTaskFocalLoss(
            firms_weight=config.firms_weight,
            other_drivers_weight=config.other_drivers_weight,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
            ignore_zero_values=config.ignore_zero_values,
            regression_loss=config.loss_function  # 'mse', 'huber', 'mae'
        )
        
        print(f"🔍 多任务Focal Loss配置:")
        print(f"   FIRMS权重: {config.firms_weight}, 其他驱动因素权重: {config.other_drivers_weight}")
        print(f"   Focal α: {config.focal_alpha}, Focal γ: {config.focal_gamma}")
        print(f"   回归损失: {config.loss_function}, 忽略0值: {config.ignore_zero_values}")
        
    elif config.loss_type == 'kldiv':
        # 使用多任务KL散度Loss
        criterion = MultiTaskKLDivLoss(
            firms_weight=config.firms_weight,
            other_drivers_weight=config.other_drivers_weight,
            ignore_zero_values=config.ignore_zero_values,
            temperature=1.0,  # 可以后续添加到配置中
            epsilon=1e-8
        )
        
        print(f"🔍 多任务KL散度Loss配置:")
        print(f"   FIRMS权重: {config.firms_weight}, 其他驱动因素权重: {config.other_drivers_weight}")
        print(f"   温度参数: 1.0, 忽略0值: {config.ignore_zero_values}")
        
    elif config.loss_type == 'multitask':
        # 使用多任务损失函数
        criterion = MultiTaskLoss(
            firms_weight=config.firms_weight,
            other_drivers_weight=config.other_drivers_weight,
            ignore_zero_values=config.ignore_zero_values,
            loss_function=config.loss_function
        )
        
        print(f"🔍 多任务损失函数配置:")
        print(f"   FIRMS权重: {config.firms_weight}, 其他驱动因素权重: {config.other_drivers_weight}")
        print(f"   忽略0值: {config.ignore_zero_values}")
        print(f"   损失函数: {config.loss_function}")
    
    else:
        raise ValueError(f"不支持的损失函数类型: {config.loss_type}。支持的类型: 'focal', 'kldiv', 'multitask'")
    
    print(f"🎯 当前使用损失函数: {config.loss_type.upper()}")
    
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.T_0, T_mult=config.T_mult, eta_min=config.eta_min
    )
    
    # Early stopping
    early_stopping = MultiMetricEarlyStopping(patience=TRAINING_CONFIG['patience'], min_delta=0.0001, restore_best_weights=True)
    
    # 追踪各指标的最佳值和模型路径
    best_metrics = {
        'f1': {'score': 0.0, 'path': None},
        'recall': {'score': 0.0, 'path': None},
        'pr_auc': {'score': 0.0, 'path': None},
        'mae': {'score': float('inf'), 'path': None},  # MAE越小越好，初始化为无穷大
        'mse': {'score': float('inf'), 'path': None}   # MSE越小越好，初始化为无穷大
    }
    
    print(f"🚀 开始训练 {config.epochs} 个epochs...")
    
    for epoch in range(config.epochs):
        # 每epoch重新抽样训练集（如果启用）
        if hasattr(train_loader.dataset, 'resample_for_epoch'):
            train_loader.dataset.resample_for_epoch(epoch)
        
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        # 训练阶段 - 简化进度显示以提高性能
        # train_progress = SimpleProgressTracker()
        for i, batch in enumerate(train_loader):
            # 注释掉详细的训练进度显示以减少CPU开销
            # train_progress.update(i+1, len(train_loader), f"🔥 Epoch {epoch+1}/{config.epochs} Training")
            
            past, future, metadata_list = batch
            past, future = past.to(device), future.to(device)
            
            # 🔥 修改：不删除第0个通道，而是将其数据置零，保持39个通道的完整性
            # past[:, 0, :] = 0.0  # 将第0个通道（FIRMS）置零，而不是删除
            
            if firms_normalizer is not None:
                past, future = normalize_batch(past, future, firms_normalizer, metadata_list)
            
            date_strings = [str(int(metadata[0])) for metadata in metadata_list]
            
            future_truncated = future[:, :, :config.pred_len].transpose(1, 2)
            target = future_truncated[:, :, 0]  # 如果是focal loss的单通道预测，则使用[:, :, 0]
            # target = (target > config.binarization_threshold).float()
            
            # 前向传播
            if model_name == 's_mamba':
                past_transposed = past.transpose(1, 2)
                past_truncated = past_transposed[:, -config.seq_len:, :]
                
                output = model(past_truncated, date_strings)
            else:
                x_enc, x_mark_enc, x_dec, x_mark_dec = adapter.adapt_inputs(past, future, date_strings)
                x_enc, x_mark_enc, x_dec, x_mark_dec = x_enc.to(device), x_mark_enc.to(device), x_dec.to(device), x_mark_dec.to(device)
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)  # B T C
                        
            # 多任务学习：预测所有39个通道
            # output shape: (B, T, C) where C=39
            # target shape: (B, T, C) where C=39
            target_all_channels = future_truncated  # 使用所有通道作为目标

            # 计算多任务Focal损失
            # 设置高置信度阈值，避免低置信度样本对损失函数的影响
            # 只对FIRMS通道二值化，其它通道保持原值
            # target_all_channels = target_all_channels.clone()
            # target_all_channels[:, :, 0] = (target_all_channels[:, :, 0] > 10).float()
            loss, loss_components = criterion(output, target_all_channels)
            # print(f"FIRMS loss: {loss_components['firms_loss']}, Other drivers loss: {loss_components['other_drivers_loss']}")
            optimizer.zero_grad()
            loss.backward()
            
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            
            train_loss += loss.item()
            # 只保存FIRMS通道的预测结果用于指标计算
            train_preds.append(output[:, :, 0].detach())
            train_targets.append(target_all_channels[:, :, 0].detach())
        
        # 计算训练指标
        train_loss /= len(train_loader)
        train_preds = torch.cat(train_preds, dim=0)
        train_targets = torch.cat(train_targets, dim=0)
        train_precision, train_recall, train_f1, train_pr_auc, train_mse, train_mae = calculate_detailed_metrics(train_preds, train_targets)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            # 验证阶段 - 简化进度显示以提高性能
            # val_progress = SimpleProgressTracker()
            for i, batch in enumerate(val_loader):
                # 注释掉详细的验证进度显示以减少CPU开销
                # val_progress.update(i+1, len(val_loader), f"📊 Epoch {epoch+1}/{config.epochs} Validation")
                
                past, future, metadata_list = batch
                past, future = past.to(device), future.to(device)
                
                # 🔥 修改：不删除第0个通道，而是将其数据置零，保持39个通道的完整性
                # past[:, 0, :] = 0.0  # 将第0个通道（FIRMS）置零，而不是删除
                
                # 为什么要对未来数据也归一化！！！
                if firms_normalizer is not None:
                    past, future = normalize_batch(past, future, firms_normalizer, metadata_list)
                
                date_strings = [str(int(metadata[0])) for metadata in metadata_list]
                
                future_truncated = future[:, :, :config.pred_len].transpose(1, 2)
                target = future_truncated[:, :, 0]  # 如果是focal loss的单通道预测，则使用[:, :, 0]
                # target = (target > config.binarization_threshold).float()
                
                if model_name == 's_mamba':
                    past_transposed = past.transpose(1, 2)
                    past_truncated = past_transposed[:, -config.seq_len:, :]
                    output = model(past_truncated, date_strings)
                else:
                    x_enc, x_mark_enc, x_dec, x_mark_dec = adapter.adapt_inputs(past, future, date_strings)
                    x_enc, x_mark_enc, x_dec, x_mark_dec = x_enc.to(device), x_mark_enc.to(device), x_dec.to(device), x_mark_dec.to(device)
                    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                
                # 多任务学习：预测所有39个通道
                target_all_channels = future_truncated  # 使用所有通道作为目标
                
                # 计算多任务Focal损失
                # target_all_channels = target_all_channels.clone()
                # target_all_channels[:, :, 0] = (target_all_channels[:, :, 0] > 10).float()
                loss, loss_components = criterion(output, target_all_channels)
                val_loss += loss.item()
                
                # 只保存FIRMS通道的预测结果用于指标计算
                val_preds.append(output[:, :, 0].detach())
                val_targets.append(target_all_channels[:, :, 0].detach())
        
        # 计算验证指标
        val_loss /= len(val_loader)
        val_preds = torch.cat(val_preds, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
        val_precision, val_recall, val_f1, val_pr_auc, val_mse, val_mae = calculate_detailed_metrics(val_preds, val_targets)
        
        # 记录当前epoch的指标
        epoch_data = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_precision": train_precision,
                "train_recall": train_recall,
                "train_f1": train_f1,
                "train_pr_auc": train_pr_auc,
                "train_mse": train_mse,
                "train_mae": train_mae,
                "val_loss": val_loss,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
                "val_pr_auc": val_pr_auc,
                "val_mse": val_mse,
                "val_mae": val_mae,
                "learning_rate": optimizer.param_groups[0]["lr"],
                # 添加多任务损失组件信息
                "firms_weight": config.firms_weight,
                "other_drivers_weight": config.other_drivers_weight,
                "loss_function": config.loss_function,
                "ignore_zero_values": config.ignore_zero_values
        }
        epoch_metrics.append(epoch_data)
        
        # 记录到wandb
        if wandb_run:
            wandb.log(epoch_data)
        
        # 显示训练进度
        print(f"Epoch {epoch+1}/{config.epochs} - "
              f"Train Loss: {train_loss:.4f} (F1: {train_f1:.4f}) - "
              f"Val Loss: {val_loss:.4f} (F1: {val_f1:.4f}) - "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # 显示多任务损失组件信息（每5个epoch显示一次）
        if (epoch + 1) % 5 == 0:
            print(f"   多任务损失组件 - FIRMS: {config.firms_weight:.1f}, "
                  f"其他驱动因素: {config.other_drivers_weight:.1f}, "
                  f"损失函数: {config.loss_function}")
        
        # 保存各指标的最佳模型
        model_save_dir = TRAINING_CONFIG[model_type]['model_save_dir']
        os.makedirs(model_save_dir, exist_ok=True)
        
        metrics_to_save = {
            'f1': val_f1,
            'recall': val_recall,
            'pr_auc': val_pr_auc,
            'mae': val_mae,
            'mse': val_mse
        }
        
        for metric_name, score in metrics_to_save.items():
            # MAE和MSE是越小越好，其他指标是越大越好
            if metric_name in ['mae', 'mse']:
                if score <= best_metrics[metric_name]['score']:
                    best_metrics[metric_name]['score'] = score
                    model_path = os.path.join(model_save_dir, f'{model_name}_best_{metric_name}.pth')
                    torch.save(model.state_dict(), model_path)
                    best_metrics[metric_name]['path'] = model_path
            else:
                if score >= best_metrics[metric_name]['score']:
                    best_metrics[metric_name]['score'] = score
                    model_path = os.path.join(model_save_dir, f'{model_name}_best_{metric_name}.pth')
                    torch.save(model.state_dict(), model_path)
                    best_metrics[metric_name]['path'] = model_path
        
        # 打印epoch总结
        print(f'Epoch {epoch+1:3d}/{config.epochs} | Train: Loss={train_loss:.4f}, F1={train_f1:.4f}, MSE={train_mse:.6f}, MAE={train_mae:.6f} | '
              f'Val: Loss={val_loss:.4f}, P={val_precision:.4f}, R={val_recall:.4f}, F1={val_f1:.4f}, PR-AUC={val_pr_auc:.4f}, MSE={val_mse:.6f}, MAE={val_mae:.6f} | '
              f'LR={optimizer.param_groups[0]["lr"]:.2e}')
        
        # Early stopping检查 (使用多指标)
        if early_stopping({'f1': val_f1, 'recall': val_recall, 'pr_auc': val_pr_auc, 'mae': val_mae, 'mse': val_mse}, model):
            print(f"⏹️  Early stopping triggered at epoch {epoch+1} (patience={TRAINING_CONFIG['patience']}, counter={early_stopping.counter})")
            break
        
        lr_scheduler.step()
    
    # 保存最后一个epoch的模型参数
    final_model_path = os.path.join(model_save_dir, f'{model_name}_final_epoch.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"💾 最后一个epoch模型已保存: {final_model_path}")
    
    # 将最后epoch路径添加到返回结果中
    best_metrics['final_epoch'] = {
        'score': epoch + 1,  # 记录最终epoch数
        'path': final_model_path
    }
    
    # 保存详细的epoch训练日志
    if log_file:
        save_epoch_metrics_to_log(epoch_metrics, log_file, model_name, model_type)
    
    # 关闭wandb
    if wandb_run:
        wandb.finish()
    
    return best_metrics

def test_model(model_name, model_path, device, test_loader, firms_normalizer, model_type='standard'):
    """测试模型"""
    print(f"\n📊 测试{model_type}模型: {model_name}")
    
    config = Config(model_name, model_type)
    
    # 使用统一适配器
    from model_adapter_unified import UnifiedModelAdapter
    adapter = UnifiedModelAdapter(config)
    
    # 根据配置选择损失函数类型
    if config.loss_type == 'focal':
        # 创建多任务Focal损失函数用于测试
        criterion = MultiTaskFocalLoss(
            firms_weight=config.firms_weight,
            other_drivers_weight=config.other_drivers_weight,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
            ignore_zero_values=config.ignore_zero_values,
            regression_loss=config.loss_function
        )
        
        print(f"🔍 测试阶段多任务Focal Loss配置:")
        print(f"   FIRMS权重: {config.firms_weight}, 其他驱动因素权重: {config.other_drivers_weight}")
        print(f"   Focal α: {config.focal_alpha}, Focal γ: {config.focal_gamma}")
        print(f"   回归损失: {config.loss_function}, 忽略0值: {config.ignore_zero_values}")
        
    elif config.loss_type == 'kldiv':
        # 创建多任务KL散度损失函数用于测试
        criterion = MultiTaskKLDivLoss(
            firms_weight=config.firms_weight,
            other_drivers_weight=config.other_drivers_weight,
            ignore_zero_values=config.ignore_zero_values,
            temperature=1.0,
            epsilon=1e-8
        )
        
        print(f"🔍 测试阶段多任务KL散度Loss配置:")
        print(f"   FIRMS权重: {config.firms_weight}, 其他驱动因素权重: {config.other_drivers_weight}")
        print(f"   温度参数: 1.0, 忽略0值: {config.ignore_zero_values}")
        
    elif config.loss_type == 'multitask':
        # 创建多任务损失函数用于测试
        criterion = MultiTaskLoss(
            firms_weight=config.firms_weight,
            other_drivers_weight=config.other_drivers_weight,
            ignore_zero_values=config.ignore_zero_values,
            loss_function=config.loss_function
        )
        
        print(f"🔍 测试阶段多任务损失函数配置:")
        print(f"   FIRMS权重: {config.firms_weight}, 其他驱动因素权重: {config.other_drivers_weight}")
        print(f"   忽略0值: {config.ignore_zero_values}")
        print(f"   损失函数: {config.loss_function}")
    
    else:
        raise ValueError(f"不支持的损失函数类型: {config.loss_type}。支持的类型: 'focal', 'kldiv', 'multitask'")
    
    print(f"🎯 测试阶段使用损失函数: {config.loss_type.upper()}")
    
    try:
        model, _ = load_model(model_name, config, model_type)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"❌ {model_type}模型 {model_name} 测试加载失败: {e}")
        return None
    
    test_preds = []
    test_targets = []
    total_test_loss = 0.0
    
    with torch.no_grad():
        # 简化测试进度显示以提高性能
        # test_progress = SimpleProgressTracker()
        for i, batch in enumerate(test_loader):
            # 注释掉详细的测试进度显示以减少CPU开销
            # test_progress.update(i+1, len(test_loader), f"🧪 Testing {model_name}")
            past, future, metadata_list = batch
            past, future = past.to(device), future.to(device)
            
            # 🔥 修改：不删除第0个通道，而是将其数据置零，保持39个通道的完整性
            # past[:, 0, :] = 0.0  # 将第0个通道（FIRMS）置零，而不是删除
            
            if firms_normalizer is not None:
                past, future = normalize_batch(past, future, firms_normalizer, metadata_list)
            
            date_strings = [str(int(metadata[0])) for metadata in metadata_list]
            
            future_truncated = future[:, :, :config.pred_len].transpose(1, 2)
            target = future_truncated[:, :, 0]  # 如果是focal loss的单通道预测，则使用[:, :, 0]
            # target = (target > config.binarization_threshold).float()
            
            if model_name == 's_mamba':
                past_transposed = past.transpose(1, 2)
                past_truncated = past_transposed[:, -config.seq_len:, :]
                output = model(past_truncated, date_strings)
            else:
                x_enc, x_mark_enc, x_dec, x_mark_dec = adapter.adapt_inputs(past, future, date_strings)
                x_enc, x_mark_enc, x_dec, x_mark_dec = x_enc.to(device), x_mark_enc.to(device), x_dec.to(device), x_mark_dec.to(device)
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # 多任务学习：预测所有39个通道
            target_all_channels = future_truncated  # 使用所有通道作为目标
            
            # 计算多任务Focal损失
            # target_all_channels = target_all_channels.clone()
            # target_all_channels[:, :, 0] = (target_all_channels[:, :, 0] > 10).float()
            loss, loss_components = criterion(output, target_all_channels)
            total_test_loss += loss.item()
            
            # 只保存FIRMS通道的预测结果用于指标计算
            test_preds.append(output[:, :, 0].detach())
            test_targets.append((target_all_channels[:, :, 0].detach()).float())
    
    # 计算测试指标 - 使用F1最优阈值
    test_preds = torch.cat(test_preds, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    precision, recall, f1, pr_auc, mse, mae = calculate_optimal_f1_metrics(test_preds, test_targets)
    
    avg_test_loss = total_test_loss / len(test_loader)
    
    print(f"✅ {model_name} {model_type}模型测试结果: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, PR-AUC={pr_auc:.4f}, MSE={mse:.6f}, MAE={mae:.6f}")
    print(f"   多任务损失: {avg_test_loss:.6f} (FIRMS权重: {config.firms_weight:.1f}, 其他驱动因素权重: {config.other_drivers_weight:.1f})")
    
    return {
        'model': model_name,
        'model_type': model_type,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pr_auc': pr_auc,
        'mse': mse,
        'mae': mae,
        'test_loss': avg_test_loss,
        'firms_weight': config.firms_weight,
        'other_drivers_weight': config.other_drivers_weight,
        'loss_function': config.loss_function,
        'loss_type': config.loss_type,  # 新增：损失函数类型信息
        'ignore_zero_values': config.ignore_zero_values
    }

def train_and_test_models(model_list, model_type, device, train_loader, val_loader, test_loader, firms_normalizer, force_retrain=False):
    """训练和测试一类模型"""
    print(f"\n🔥 开始训练{model_type}模型组")
    print(f"📋 原始模型列表: {len(model_list)} 个{model_type}模型")
    print(f"📊 {model_type}模型列表: {', '.join(model_list)}")
    
    # 过滤已训练的模型
    models_to_train, trained_models = filter_trained_models(model_list, model_type, force_retrain)
    
    # 训练新模型
    model_results = []
    failed_models = []
    
    if models_to_train:
        print(f"\n🚀 开始训练 {len(models_to_train)} 个需要训练的{model_type}模型...")
        for i, model_name in enumerate(models_to_train):
            print(f"\n🔄 {model_type}训练进度: {i+1}/{len(models_to_train)} (总体: {i+1+len(trained_models)}/{len(model_list)})")
        try:
            result = train_single_model(
                model_name, device, train_loader, val_loader, test_loader, firms_normalizer, model_type
            )
            if result is not None:
                best_metrics = result
                print(f"✅ {model_name} {model_type}模型训练完成，保存的模型:")
                for metric_name, metric_info in best_metrics.items():
                        if metric_name == 'final_epoch':
                            print(f"  最后epoch模型 (epoch {metric_info['score']}): {metric_info['path']}")
                        else:
                            print(f"  最佳{metric_name}模型 ({metric_info['score']:.4f}): {metric_info['path']}")
                model_results.append((model_name, best_metrics))
            else:
                failed_models.append(model_name)
        except Exception as e:
            print(f"❌ {model_name} {model_type}模型训练失败: {e}")
            failed_models.append(model_name)
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        print(f"\n✅ 所有{model_type}模型都已训练完成，跳过训练阶段")
    
    # 将已训练的模型添加到结果中
    for model_name, trained_paths in trained_models.items():
        model_results.append((model_name, trained_paths))
        print(f"📋 加载已训练模型: {model_name} ({len(trained_paths)}个保存版本)")
    
    print(f"\n📈 {model_type}模型准备完成!")
    print(f"   新训练: {len(models_to_train)} 个")
    print(f"   已训练: {len(trained_models)} 个") 
    print(f"   训练失败: {len(failed_models)} 个")
    print(f"   总可用: {len(model_results)} 个")
    
    if failed_models:
        print(f"❌ 失败的{model_type}模型: {', '.join(failed_models)}")
    
    # 测试阶段
    print("\n" + "="*60)
    print("🧪 测试阶段 - 评估训练好的模型")
    print("="*60)
    
    # 用于存储结构化测试结果的字典
    structured_results = {}
    
    for model_name, metrics in model_results:
        print(f"\n📋 测试模型: {model_name}")
        print("-" * 40)
        
        # 初始化该模型的结果字典
        structured_results[model_name] = {
            'f1': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'mse': None, 'mae': None},
            'recall': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'mse': None, 'mae': None},
            'pr_auc': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'mse': None, 'mae': None},
            'mae': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'mse': None, 'mae': None},
            'final_epoch': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'mse': None, 'mae': None}
        }
        
        # 分别测试所有保存的模型（包括最佳模型和最后epoch模型）
        for metric_name, metric_info in metrics.items():
            if metric_info['path'] is not None:
                if metric_name == 'final_epoch':
                    print(f"\n🎯 测试最后一个epoch的模型 (epoch: {metric_info['score']})")
                else:
                    print(f"\n🎯 测试基于 {metric_name.upper()} 的最佳模型 (分数: {metric_info['score']:.4f})")
                try:
                    result = test_model(model_name, metric_info['path'], device, test_loader, firms_normalizer, model_type)
                    if result:
                        # 保存到结构化结果中
                        structured_results[model_name][metric_name] = {
                            'precision': result['precision'],
                            'recall': result['recall'],
                            'f1': result['f1'],
                            'pr_auc': result['pr_auc'],
                            'mse': result['mse'],
                            'mae': result['mae']
                        }
                        print(f"✅ {model_name} ({metric_name}) 测试完成")
                except Exception as e:
                    print(f"❌ {model_name} ({metric_name}) 测试失败: {str(e)}")
    
    if not structured_results:
        print("⚠️  没有模型通过测试！")
        return None
    
    # 保存结构化结果到CSV
    save_structured_results_to_csv(structured_results, model_type)
    
    # 输出最终结果总结
    print("\n" + "="*80)
    print("📊 最终测试结果总结")
    print("="*80)
    
    # 显示表格形式的结果
    for model_name, model_results in structured_results.items():
        print(f"\n🔥 模型: {model_name}")
        print("-" * 80)
        print(f"{'指标类型':<12} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'PR-AUC':<8} {'MSE':<10} {'MAE':<10}")
        print("-" * 80)
        for metric_type, metrics in model_results.items():
            if metrics['precision'] is not None:
                display_type = "FINAL" if metric_type == 'final_epoch' else metric_type.upper()
                print(f"{display_type:<12} {metrics['precision']:<8.4f} {metrics['recall']:<8.4f} {metrics['f1']:<8.4f} {metrics['pr_auc']:<8.4f} {metrics['mse']:<10.6f} {metrics['mae']:<10.6f}")
    
    print(f"\n🎉 训练和测试完成！共训练了 {len(model_results)} 个模型")
    
    if failed_models:
        print(f"\n⚠️  失败的模型: {failed_models}")
    
    print("\n📁 所有模型已保存到相应目录中")
    save_dir = STANDARD_MODEL_DIR if model_type == 'standard' else MODEL_10X_DIR
    print(f"📊 测试结果已保存到目录: {save_dir}")
    
    return structured_results

def prepare_data_loaders():
    """准备数据加载器"""
    print("📂 加载数据...")
    data_loader = TimeSeriesDataLoader(
        # h5_dir='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/pixel_samples_merged',
        h5_dir='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/full_datasets',
        positive_ratio=DATA_CONFIG['positive_ratio'],
        pos_neg_ratio=DATA_CONFIG['pos_neg_ratio'],
        resample_each_epoch=False  # 在底层禁用，改用动态抽样
    )
    
    # 数据集划分
    train_indices, val_indices, test_indices = data_loader.get_year_based_split(
        train_years=DATA_CONFIG['train_years'],
        val_years=DATA_CONFIG['val_years'],
        test_years=DATA_CONFIG['test_years']
    )
    
    # 使用自定义的动态抽样数据集替代标准Subset
    train_dataset = DynamicSamplingSubset(
        dataset=data_loader.dataset,
        full_indices=train_indices,
        sampling_ratio=DATA_CONFIG['sampling_ratio'],
        enable_dynamic_sampling=DATA_CONFIG['enable_dynamic_sampling']
    )
    
    # 验证集和测试集使用完整数据，不进行动态抽样
    val_dataset = Subset(data_loader.dataset, val_indices)
    test_dataset = Subset(data_loader.dataset, test_indices)
    
    print(f"📊 数据集大小:")
    print(f"   训练集: {len(train_dataset)} (完整: {len(train_indices)})")
    print(f"   验证集: {len(val_dataset)} (完整数据)")
    print(f"   测试集: {len(test_dataset)} (完整数据)")
    print(f"   动态抽样: {'启用' if DATA_CONFIG['enable_dynamic_sampling'] else '禁用'}")
    if DATA_CONFIG['enable_dynamic_sampling']:
        print(f"   抽样配置: 每epoch随机使用 {DATA_CONFIG['sampling_ratio']:.1%} 的训练数据")
    
    return train_dataset, val_dataset, test_dataset, data_loader

def main():
    """主函数 - 依次训练标准模型和10x模型"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='野火预测模型训练脚本')
    parser.add_argument('--skip-10x', action='store_true', 
                       help='跳过10x模型训练，只训练标准模型')
    parser.add_argument('--only-10x', action='store_true',
                       help='只训练10x模型，跳过标准模型')
    parser.add_argument('--force-retrain', action='store_true',
                       help='强制重新训练所有模型，忽略已存在的模型文件')
    
    # 多任务学习参数
    parser.add_argument('--firms-weight', type=float, default=1,  # 0.005 for focal loss, 0.1 for multitask 
                       help='FIRMS预测的损失权重 (默认: 1.0)')
    parser.add_argument('--other-drivers-weight', type=float, default=1.0,
                       help='其他驱动因素预测的损失权重 (默认: 1.0)')
    parser.add_argument('--loss-function', type=str, default='mse',
                       choices=['huber', 'mse', 'mae'],
                       help='其他驱动因素的回归损失函数类型 (默认: mse)')
    parser.add_argument('--no-ignore-zero', action='store_true',
                       help='不忽略其他驱动因素中的0值')
    
    # Focal Loss参数
    parser.add_argument('--focal-alpha', type=float, default=0.5,
                       help='Focal Loss的alpha参数 (默认: 0.5)')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                       help='Focal Loss的gamma参数 (默认: 2.0)')
    
    # 损失函数类型选择参数
    parser.add_argument('--loss-type', type=str, default='focal',  #######################################
                       choices=['focal', 'kldiv', 'multitask'],
                       help='损失函数类型选择 (默认: focal)')
    
    # 🔥 新增：位置信息和气象数据特征参数
    parser.add_argument('--enable-position-features', action='store_true',
                       help='启用位置信息特征 (默认: 禁用)')
    parser.add_argument('--enable-future-weather', action='store_true', 
                       help='启用未来气象数据特征 (默认: 禁用)')
    parser.add_argument('--weather-channels', type=str, default='1-12',
                       help='气象数据通道范围，格式如"1-12"或"1,3,5-8" (默认: 1-12)')
    
    args = parser.parse_args()
    
    # 🔥 新增：解析气象数据通道范围
    def parse_channel_range(channel_str):
        """解析通道范围字符串，返回通道索引列表"""
        channels = []
        for part in channel_str.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                channels.extend(range(start, end + 1))
            else:
                channels.append(int(part))
        return channels
    
    # 更新数据配置
    global DATA_CONFIG
    DATA_CONFIG['enable_position_features'] = args.enable_position_features
    DATA_CONFIG['enable_future_weather'] = args.enable_future_weather
    
    if args.enable_future_weather:
        try:
            DATA_CONFIG['weather_channels'] = parse_channel_range(args.weather_channels)
        except ValueError as e:
            print(f"❌ 气象通道范围格式错误: {args.weather_channels}")
            print(f"   错误信息: {e}")
            print(f"   正确格式示例: '1-12' 或 '1,3,5-8'")
            return
    
    # 更新多任务学习配置
    global MULTITASK_CONFIG, TRAINING_CONFIG
    MULTITASK_CONFIG['firms_weight'] = args.firms_weight
    MULTITASK_CONFIG['other_drivers_weight'] = args.other_drivers_weight
    MULTITASK_CONFIG['loss_function'] = args.loss_function
    MULTITASK_CONFIG['ignore_zero_values'] = not args.no_ignore_zero
    MULTITASK_CONFIG['loss_type'] = args.loss_type  # 新增：损失函数类型配置
    
    # 更新Focal Loss配置
    TRAINING_CONFIG['focal_alpha'] = args.focal_alpha
    TRAINING_CONFIG['focal_gamma'] = args.focal_gamma
    
    print("🔥 野火预测模型全面对比实验 - 统一版本")
    
    # 显示共享配置状态
    print_config_status()
    print()
    
    # 根据命令行参数决定训练哪些模型
    train_standard = not args.only_10x
    train_10x = ENABLE_10X_TRAINING and not args.skip_10x
    
    if args.skip_10x:
        print("📋 已选择跳过10x模型训练")
        train_10x = False
    elif args.only_10x:
        print("📋 已选择只训练10x模型")
        train_standard = False
    
    print(f"📋 训练计划: 标准模型={'✅' if train_standard else '❌'}, 10x模型={'✅' if train_10x else '❌'}")
    if args.force_retrain:
        print("🔄 强制重新训练模式已启用，将忽略已存在的模型文件")
    
    # 初始化
    set_seed(TRAINING_CONFIG['seed'])
    # 使用当前可见的第一个CUDA设备（通过CUDA_VISIBLE_DEVICES控制）
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # 显示实际使用的GPU信息
    if torch.cuda.is_available():
        actual_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(actual_gpu)
        print(f"🖥️  使用设备: cuda:0 (实际GPU: {gpu_name})")
        print(f"🔍 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
    else:
        print(f"🖥️  使用设备: {device}")
    
    # WandB配置检查
    if TRAINING_CONFIG['use_wandb']:
        if WANDB_AVAILABLE:
            print("✅ WandB监控已启用")
        else:
            print("⚠️ WandB监控已配置但wandb未安装，将跳过监控功能")
    else:
        print("ℹ️ WandB监控已禁用")
    
    # 检查GPU内存
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"💾 GPU内存: {gpu_memory:.1f} GB")
    
    # 准备数据
    train_dataset, val_dataset, test_dataset, data_loader_obj = prepare_data_loaders()
    
    # 初始化FIRMS归一化器
    print("🔧 初始化FIRMS归一化器...")
    firms_normalizer = FIRMSNormalizer(
        method='divide_by_100',
        firms_min=DATA_CONFIG['firms_min'],
        firms_max=DATA_CONFIG['firms_max']
    )
    
    # 为归一化拟合创建临时数据加载器
    temp_loader = DataLoader(
        train_dataset, batch_size=512, shuffle=False, 
        num_workers=2, collate_fn=data_loader_obj.dataset.custom_collate_fn
    )
    firms_normalizer.fit(temp_loader)
    
    all_results = {}
    
    # ========== 第一阶段：训练标准模型 ==========
    if train_standard and MODEL_LIST_STANDARD:
        print(f"\n{'='*80}")
        print("🚀 第一阶段：训练标准model_zoo模型")
        print(f"{'='*80}")
        
        # 创建标准模型数据加载器
        standard_config = TRAINING_CONFIG['standard']
        train_loader = DataLoader(
            train_dataset, batch_size=standard_config['batch_size'], shuffle=True, 
            num_workers=4, collate_fn=data_loader_obj.dataset.custom_collate_fn, worker_init_fn=worker_init_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=standard_config['batch_size'], shuffle=False,
            num_workers=4, collate_fn=data_loader_obj.dataset.custom_collate_fn, worker_init_fn=worker_init_fn
        )
        test_loader = DataLoader(
            test_dataset, batch_size=standard_config['batch_size'], shuffle=False,
            num_workers=4, collate_fn=data_loader_obj.dataset.custom_collate_fn, worker_init_fn=worker_init_fn
        )
        
        standard_results = train_and_test_models(
            MODEL_LIST_STANDARD, 'standard', device, train_loader, val_loader, test_loader, firms_normalizer, args.force_retrain
        )
        all_results['standard'] = standard_results
    
    # ========== 第二阶段：训练10x模型 ==========
    if train_10x and MODEL_LIST_10X:
        print(f"\n{'='*80}")
        print("🚀 第二阶段：训练10x参数model_zoo_10x模型")
        print(f"{'='*80}")
        
        # 创建10x模型数据加载器（较小batch size）
        config_10x = TRAINING_CONFIG['10x']
        train_loader_10x = DataLoader(
            train_dataset, batch_size=config_10x['batch_size'], shuffle=True, 
            num_workers=6, collate_fn=data_loader_obj.dataset.custom_collate_fn, worker_init_fn=worker_init_fn
        )
        val_loader_10x = DataLoader(
            val_dataset, batch_size=config_10x['batch_size'], shuffle=False,
            num_workers=4, collate_fn=data_loader_obj.dataset.custom_collate_fn, worker_init_fn=worker_init_fn
        )
        test_loader_10x = DataLoader(
            test_dataset, batch_size=config_10x['batch_size'], shuffle=False,
            num_workers=4, collate_fn=data_loader_obj.dataset.custom_collate_fn, worker_init_fn=worker_init_fn
        )
        
        results_10x = train_and_test_models(
            MODEL_LIST_10X, '10x', device, train_loader_10x, val_loader_10x, test_loader_10x, firms_normalizer, args.force_retrain
        )
        all_results['10x'] = results_10x
    
    # ========== 最终总结 ==========
    print(f"\n{'='*80}")
    print("🎉 所有模型实验完成！")
    print(f"{'='*80}")
    
    for model_type, results in all_results.items():
        if results:
            df = pd.DataFrame(results)
            df = df.sort_values('f1', ascending=False)
            best_model = df.iloc[0]
            print(f"\n🏆 最佳{model_type}模型: {best_model['model']}")
            print(f"   F1-Score: {best_model['f1']:.4f}")
            print(f"   Precision: {best_model['precision']:.4f}")
            print(f"   Recall: {best_model['recall']:.4f}")
            print(f"   PR-AUC: {best_model['pr_auc']:.4f}")
    
    print("\n📊 所有结果已保存到相应的CSV文件中！")

if __name__ == "__main__":
    main() 