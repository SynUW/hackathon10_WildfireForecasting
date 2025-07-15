#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
野火预测模型全面对比实验 - 10倍参数模型版本
支持model_zoo_10x中所有模型的训练、early stopping、F1评价指标、最佳模型测试和CSV结果导出
"""

from dataload import TimeSeriesDataLoader, TimeSeriesPixelDataset, FullDatasetLoader
from torch.utils.data import Dataset, DataLoader, Subset
from model_adapter_10x import ModelAdapter, get_model_configs  # 使用10x版本的适配器
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
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
warnings.filterwarnings("ignore")

# =============================================================================
# 配置参数
# =============================================================================

# 从model_zoo_10x文件夹获取所有可用模型
def get_all_models_10x():
    """获取model_zoo_10x中所有可用的模型"""
    model_files = []
    model_zoo_path = 'model_zoo_10x'
    if os.path.exists(model_zoo_path):
        for file in os.listdir(model_zoo_path):
            if file.endswith('.py') and not file.startswith('__') and file != 'trash':
                model_name = file[:-3]  # 去掉.py后缀
                model_files.append(model_name)
    return sorted(model_files)

# 所有可用10x模型列表
MODEL_LIST_10X = get_all_models_10x()
print(f"发现可用10x模型: {MODEL_LIST_10X}")

# 训练配置 - 针对10x模型调整
TRAINING_CONFIG = {
    'models': MODEL_LIST_10X,        # 使用所有可用10x模型
    'use_wandb': False,              # 是否使用wandb
    'seed': 42,                      # 随机种子
    'epochs': 50,                    # 10x模型参数多，减少训练轮数
    'patience': 15,                  # Early stopping patience，稍微减少
    'batch_size': 256,               # 10x模型显存占用大，减小批次大小
    'learning_rate': 5e-4,          # 大模型使用较小学习率
    'seq_len': 30,                   # 输入序列长度
    'pred_len': 7,                   # 预测序列长度
    'weight_decay': 3e-4,           # 权重衰减
    'T_0': 10,                      # 余弦退火周期
    'T_mult': 1,                    # 余弦退火周期倍增因子
    'eta_min': 1e-6,                # 最小学习率
    'max_grad_norm': 1.0,           # 大模型使用梯度裁剪
    'focal_alpha': 0.5,             # 使用最佳的Focal Loss正样本权重
    'focal_gamma': 2.0,             # Focal Loss聚焦参数
    'model_save_dir': '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/model_pth_10x',
    'results_save_path': 'model_comparison_results_10x.csv',
}

# 数据配置
DATA_CONFIG = {
    'train_years': [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 
                   2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'val_years': [2021, 2022],
    'test_years': [2023, 2024],
    'positive_ratio': 1.0,          # 使用全部正样本
    'pos_neg_ratio': 1.0,           # 正负样本比例
    'resample_each_epoch': False,    # 是否每epoch重新抽样
    'firms_min': 0,                 # FIRMS数据最小值（跳过统计）
    'firms_max': 100,               # FIRMS数据最大值（跳过统计）
}

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
    
    def __init__(self, method='log1p_minmax', firms_min=None, firms_max=None):
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
            else:
                self.global_min = self.firms_min
                self.global_max = self.firms_max
            self.fitted = True
            print(f"✅ 归一化器初始化完成 (变换后范围: {self.global_min:.2f}-{self.global_max:.2f})")
            return
            
        print("🔧 收集FIRMS数据进行归一化拟合...")
        firms_values = []
        
        for batch in tqdm(data_loader, desc="收集FIRMS数据"):
            past, future, _ = batch
            firms_data = past[:, 0, :]  # FIRMS通道 (B, T)
            firms_values.append(firms_data.numpy())
        
        all_firms = np.concatenate(firms_values, axis=0).flatten()
        valid_firms = all_firms[all_firms != 255]  # 过滤掉NoData值(255)
        
        if self.method == 'log1p_minmax':
            log_firms = np.log1p(valid_firms)
            self.global_min = log_firms.min()
            self.global_max = log_firms.max()
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
        else:
            raise ValueError(f"不支持的归一化方法: {self.method}")

def normalize_batch(past, future, firms_normalizer=None):
    """对批次数据进行归一化处理"""
    if firms_normalizer is not None:
        past[:, 0, :] = firms_normalizer.normalize(past[:, 0, :])
    return past, future

def load_model_10x(model_name, configs):
    """动态加载10x模型"""
    try:
        import sys
        model_zoo_path = os.path.join(os.getcwd(), 'model_zoo_10x')
        if model_zoo_path not in sys.path:
            sys.path.insert(0, model_zoo_path)
        
        module = importlib.import_module(f'model_zoo_10x.{model_name}')
        Model = getattr(module, 'Model')
        
        return Model(configs), 'standard'
    except Exception as e:
        print(f"加载10x模型 {model_name} 失败: {e}")
        raise

def calculate_detailed_metrics(output, target):
    """计算详细的二分类指标，包括PR-AUC"""
    pred_probs = torch.sigmoid(output).view(-1).cpu().numpy()
    pred_binary = (pred_probs > 0.5).astype(int)
    target_np = target.view(-1).cpu().numpy()
    
    unique_targets = np.unique(target_np)
    if len(unique_targets) < 2:
        return 0.0, 0.0, 0.0, 0.0
    
    try:
        precision = precision_score(target_np, pred_binary, average='binary', zero_division=0)
        recall = recall_score(target_np, pred_binary, average='binary', zero_division=0)
        f1 = f1_score(target_np, pred_binary, average='binary', zero_division=0)
        pr_auc = average_precision_score(target_np, pred_probs)
    except Exception as e:
        print(f"计算指标时出错: {e}")
        return 0.0, 0.0, 0.0, 0.0
    
    return precision, recall, f1, pr_auc

class Config:
    """配置类 - 使用10x配置"""
    def __init__(self, model_name):
        self.epochs = TRAINING_CONFIG['epochs']
        self.batch_size = TRAINING_CONFIG['batch_size']
        self.learning_rate = TRAINING_CONFIG['learning_rate']
        self.seq_len = TRAINING_CONFIG['seq_len']
        self.pred_len = TRAINING_CONFIG['pred_len']
        
        # 获取10x模型配置
        model_configs = get_model_configs(model_name)
        for key, value in model_configs.items():
            setattr(self, key, value)
        
        # 强制使用统一配置
        self.seq_len = TRAINING_CONFIG['seq_len']
        self.pred_len = TRAINING_CONFIG['pred_len']
        
        # 训练参数
        self.weight_decay = TRAINING_CONFIG['weight_decay']
        self.T_0 = TRAINING_CONFIG['T_0']
        self.T_mult = TRAINING_CONFIG['T_mult']
        self.eta_min = TRAINING_CONFIG['eta_min']
        self.max_grad_norm = TRAINING_CONFIG['max_grad_norm']
        
        # FIRMS数据归一化参数
        self.normalize_firms = True
        self.firms_normalization_method = 'log1p_minmax'
        self.binarization_threshold = 0.0
        self.firms_min = DATA_CONFIG['firms_min']
        self.firms_max = DATA_CONFIG['firms_max']
        
        # Focal Loss参数  
        self.focal_alpha = TRAINING_CONFIG['focal_alpha']
        self.focal_gamma = TRAINING_CONFIG['focal_gamma']
        
        # 数据集划分
        self.train_years = DATA_CONFIG['train_years']
        self.val_years = DATA_CONFIG['val_years']
        self.test_years = DATA_CONFIG['test_years']

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
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

# =============================================================================
# 核心训练和测试函数
# =============================================================================

def train_single_model_10x(model_name, device, train_loader, val_loader, test_loader, firms_normalizer):
    """训练单个10x模型"""
    print(f"\n{'='*60}")
    print(f"🔥 开始训练10x模型: {model_name}")
    print(f"{'='*60}")
    
    # 创建配置和模型
    config = Config(model_name)
    adapter = ModelAdapter(config)
    
    try:
        model, model_type = load_model_10x(model_name, config)
        model = model.to(device)
    except Exception as e:
        print(f"❌ 10x模型 {model_name} 加载失败: {e}")
        return None
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 10x模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
    
    # 初始化训练组件
    criterion = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.T_0, T_mult=config.T_mult, eta_min=config.eta_min
    )
    early_stopping = EarlyStopping(patience=TRAINING_CONFIG['patience'], min_delta=0.001)
    
    best_f1 = 0.0
    best_model_path = None
    
    print("🚀 开始训练...")
    
    # 训练循环
    for epoch in range(config.epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        for batch in tqdm(train_loader, desc=f'Train {epoch+1}/{config.epochs}', leave=False):
            past, future, metadata_list = batch
            past, future = past.to(device), future.to(device)
            
            if firms_normalizer is not None:
                past, future = normalize_batch(past, future, firms_normalizer)
            
            date_strings = [str(int(metadata[0])) for metadata in metadata_list]
            
            future_truncated = future[:, :, :config.pred_len].transpose(1, 2)
            target = future_truncated[:, :, 0]
            target = (target > config.binarization_threshold).float()
            
            # 前向传播 - 10x模型支持
            if model_name == 's_mamba':
                past_transposed = past.transpose(1, 2)
                past_truncated = past_transposed[:, -config.seq_len:, :]
                output = model(past_truncated, date_strings)
            else:
                x_enc, x_mark_enc, x_dec, x_mark_dec = adapter.adapt_inputs(past, future, date_strings)
                x_enc, x_mark_enc, x_dec, x_mark_dec = x_enc.to(device), x_mark_enc.to(device), x_dec.to(device), x_mark_dec.to(device)
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            output_channel_0 = output[:, :, 0]
            loss = criterion(output_channel_0, target)
            
            optimizer.zero_grad()
            loss.backward()
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.append(output_channel_0.detach())
            train_targets.append(target.detach())
        
        # 计算训练指标
        train_loss /= len(train_loader)
        train_preds = torch.cat(train_preds, dim=0)
        train_targets = torch.cat(train_targets, dim=0)
        train_precision, train_recall, train_f1, train_pr_auc = calculate_detailed_metrics(train_preds, train_targets)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Val {epoch+1}/{config.epochs}', leave=False):
                past, future, metadata_list = batch
                past, future = past.to(device), future.to(device)
                
                if firms_normalizer is not None:
                    past, future = normalize_batch(past, future, firms_normalizer)
                
                date_strings = [str(int(metadata[0])) for metadata in metadata_list]
                
                future_truncated = future[:, :, :config.pred_len].transpose(1, 2)
                target = future_truncated[:, :, 0]
                target = (target > config.binarization_threshold).float()
                
                if model_name == 's_mamba':
                    past_transposed = past.transpose(1, 2)
                    past_truncated = past_transposed[:, -config.seq_len:, :]
                    output = model(past_truncated, date_strings)
                else:
                    x_enc, x_mark_enc, x_dec, x_mark_dec = adapter.adapt_inputs(past, future, date_strings)
                    x_enc, x_mark_enc, x_dec, x_mark_dec = x_enc.to(device), x_mark_enc.to(device), x_dec.to(device), x_mark_dec.to(device)
                    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                
                output_channel_0 = output[:, :, 0]
                loss = criterion(output_channel_0, target)
                val_loss += loss.item()
                
                val_preds.append(output_channel_0.detach())
                val_targets.append(target.detach())
        
        # 计算验证指标
        val_loss /= len(val_loader)
        val_preds = torch.cat(val_preds, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
        val_precision, val_recall, val_f1, val_pr_auc = calculate_detailed_metrics(val_preds, val_targets)
        
        # 保存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            model_path = os.path.join(TRAINING_CONFIG['model_save_dir'], f'{model_name}_best_f1.pth')
            os.makedirs(TRAINING_CONFIG['model_save_dir'], exist_ok=True)
            torch.save(model.state_dict(), model_path)
            best_model_path = model_path
        
        # 打印进度
        print(f'Epoch {epoch+1:3d}/{config.epochs} | Train: Loss={train_loss:.4f}, F1={train_f1:.4f} | '
              f'Val: Loss={val_loss:.4f}, P={val_precision:.4f}, R={val_recall:.4f}, F1={val_f1:.4f}, PR-AUC={val_pr_auc:.4f} | '
              f'LR={optimizer.param_groups[0]["lr"]:.2e}')
        
        # Early stopping检查
        if early_stopping(val_f1, model):
            print(f"⏹️  Early stopping triggered at epoch {epoch+1}")
            break
        
        lr_scheduler.step()
    
    return best_model_path, best_f1

def test_model_10x(model_name, model_path, device, test_loader, firms_normalizer):
    """测试10x模型"""
    print(f"\n📊 测试10x模型: {model_name}")
    
    config = Config(model_name)
    adapter = ModelAdapter(config)
    
    try:
        model, _ = load_model_10x(model_name, config)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"❌ 10x模型 {model_name} 测试加载失败: {e}")
        return None
    
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f'Testing {model_name}', leave=False):
            past, future, metadata_list = batch
            past, future = past.to(device), future.to(device)
            
            if firms_normalizer is not None:
                past, future = normalize_batch(past, future, firms_normalizer)
            
            date_strings = [str(int(metadata[0])) for metadata in metadata_list]
            
            future_truncated = future[:, :, :config.pred_len].transpose(1, 2)
            target = future_truncated[:, :, 0]
            target = (target > config.binarization_threshold).float()
            
            if model_name == 's_mamba':
                past_transposed = past.transpose(1, 2)
                past_truncated = past_transposed[:, -config.seq_len:, :]
                output = model(past_truncated, date_strings)
            else:
                x_enc, x_mark_enc, x_dec, x_mark_dec = adapter.adapt_inputs(past, future, date_strings)
                x_enc, x_mark_enc, x_dec, x_mark_dec = x_enc.to(device), x_mark_enc.to(device), x_dec.to(device), x_mark_dec.to(device)
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            output_channel_0 = output[:, :, 0]
            test_preds.append(output_channel_0.detach())
            test_targets.append(target.detach())
    
    # 计算测试指标
    test_preds = torch.cat(test_preds, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    precision, recall, f1, pr_auc = calculate_detailed_metrics(test_preds, test_targets)
    
    print(f"✅ {model_name} 10x模型测试结果: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, PR-AUC={pr_auc:.4f}")
    
    return {
        'model': model_name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pr_auc': pr_auc
    }

def main():
    """主函数 - 训练所有10x模型并测试"""
    print("🔥 野火预测10x模型全面对比实验")
    print(f"📋 将训练 {len(MODEL_LIST_10X)} 个10x模型")
    print(f"📊 10x模型列表: {', '.join(MODEL_LIST_10X)}")
    
    # 初始化
    set_seed(TRAINING_CONFIG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    # 检查GPU内存
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"💾 GPU内存: {gpu_memory:.1f} GB (10x模型需要更多内存)")
    
    # 数据加载
    print("📂 加载数据...")
    data_loader = TimeSeriesDataLoader(
        h5_dir='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/pixel_samples_merged',
        positive_ratio=DATA_CONFIG['positive_ratio'],
        pos_neg_ratio=DATA_CONFIG['pos_neg_ratio'],
        resample_each_epoch=DATA_CONFIG['resample_each_epoch']
    )
    
    # 数据集划分
    train_indices, val_indices, test_indices = data_loader.get_year_based_split(
        train_years=DATA_CONFIG['train_years'],
        val_years=DATA_CONFIG['val_years'],
        test_years=DATA_CONFIG['test_years']
    )
    
    train_dataset = Subset(data_loader.dataset, train_indices)
    val_dataset = Subset(data_loader.dataset, val_indices)
    test_dataset = Subset(data_loader.dataset, test_indices)
    
    # 创建数据加载器 - 10x模型使用较少的worker
    train_loader = DataLoader(
        train_dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=True, 
        num_workers=8, collate_fn=data_loader.dataset.custom_collate_fn, worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=False,
        num_workers=4, collate_fn=data_loader.dataset.custom_collate_fn, worker_init_fn=worker_init_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=False,
        num_workers=4, collate_fn=data_loader.dataset.custom_collate_fn, worker_init_fn=worker_init_fn
    )
    
    print(f"📊 数据集大小: 训练集 {len(train_dataset)}, 验证集 {len(val_dataset)}, 测试集 {len(test_dataset)}")
    
    # 初始化FIRMS归一化器
    print("🔧 初始化FIRMS归一化器...")
    firms_normalizer = FIRMSNormalizer(
        method='log1p_minmax',
        firms_min=DATA_CONFIG['firms_min'],
        firms_max=DATA_CONFIG['firms_max']
    )
    firms_normalizer.fit(train_loader)
    
    # 训练所有10x模型
    model_results = []
    failed_models = []
    
    for i, model_name in enumerate(MODEL_LIST_10X):
        print(f"\n🔄 进度: {i+1}/{len(MODEL_LIST_10X)}")
        try:
            result = train_single_model_10x(
                model_name, device, train_loader, val_loader, test_loader, firms_normalizer
            )
            if result is not None:
                best_model_path, best_f1 = result
                print(f"✅ {model_name} 10x模型训练完成，最佳验证F1: {best_f1:.4f}")
                model_results.append((model_name, best_model_path))
            else:
                failed_models.append(model_name)
        except Exception as e:
            print(f"❌ {model_name} 10x模型训练失败: {e}")
            failed_models.append(model_name)
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    print(f"\n📈 10x模型训练完成! 成功: {len(model_results)}, 失败: {len(failed_models)}")
    if failed_models:
        print(f"❌ 失败的10x模型: {', '.join(failed_models)}")
    
    # 测试所有成功训练的10x模型
    print(f"\n🧪 开始测试 {len(model_results)} 个10x模型...")
    test_results = []
    
    for model_name, model_path in model_results:
        try:
            result = test_model_10x(model_name, model_path, device, test_loader, firms_normalizer)
            if result:
                test_results.append(result)
        except Exception as e:
            print(f"❌ {model_name} 10x模型测试失败: {e}")
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 保存结果到CSV
    if test_results:
        df = pd.DataFrame(test_results)
        df = df.sort_values('f1', ascending=False)  # 按F1分数排序
        df.to_csv(TRAINING_CONFIG['results_save_path'], index=False)
        
        print(f"\n📊 10x模型最终测试结果已保存到: {TRAINING_CONFIG['results_save_path']}")
        print("\n🏆 10x模型排行榜 (按F1分数排序):")
        print(df.to_string(index=False, float_format='%.4f'))
        
        # 显示最佳10x模型
        best_model = df.iloc[0]
        print(f"\n🥇 最佳10x模型: {best_model['model']}")
        print(f"   F1-Score: {best_model['f1']:.4f}")
        print(f"   Precision: {best_model['precision']:.4f}")
        print(f"   Recall: {best_model['recall']:.4f}")
        print(f"   PR-AUC: {best_model['pr_auc']:.4f}")
    
    print("\n🎉 所有10x模型实验完成!")

if __name__ == "__main__":
    main() 