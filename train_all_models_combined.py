#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Wildfire Forecasting Model Benchmark - Unified Version
Supports early stopping, F1 evaluation metric, best model testing, and CSV result export
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

# Dynamically import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ wandb is not installed, will skip wandb monitoring")

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration Parameters
# =============================================================================

# Global training configuration - unified management of all training-related parameters
# Whether to enable WandB monitoring
WANDB_ENABLED = False               # Whether to enable WandB monitoring
GLOBAL_SEED = 42                   # Global random seed
DEFAULT_PATIENCE = 5              # Default early stopping patience
DEFAULT_MAX_PARALLEL_PER_GPU = 2   # Default maximum parallel tasks per GPU

# Multi-task learning configuration
MULTITASK_CONFIG = {
    'firms_weight': 1,           # Loss weight for FIRMS prediction. Typical loss combination (other drivers loss*weight): FIRMS loss: 0.3112890124320984, Other drivers loss: 0.0020517727825790644
    'other_drivers_weight': 1.0,   # Loss weight for other drivers prediction
    'ignore_zero_values': True,    # Whether to ignore zero values in other drivers
    'loss_function': 'mse',       # Loss function type: 'huber', 'mse', 'mae'
    'loss_type': 'focal'          # Loss type selection: 'focal'(MultiTaskFocalLoss), 'kldiv'(MultiTaskKLDivLoss), or 'multitask'(MultiTaskLoss)
}

# Dataset year configuration
DEFAULT_TRAIN_YEARS = [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 
                      2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
DEFAULT_VAL_YEARS = [2021, 2022]
DEFAULT_TEST_YEARS = [2023, 2024]

# Model directory configuration
# target_all_channels = target_all_channels.clone()
# target_all_channels[:, :, 0] = (target_all_channels[:, :, 0] > 10).float() Don't forget to remove these 2 lines
STANDARD_MODEL_DIR = '/mnt/raid/zhengsen/pths/7to1_focal_withRegressionLoss_itransformer'

def print_config_status():
    """Print current configuration status"""
    print("📋 Current training configuration:")
    print(f"   WandB monitoring: {'✅ Enabled' if WANDB_ENABLED else '❌ Disabled'}")
    print(f"   Random seed: {GLOBAL_SEED}")
    print(f"   Default parallelism: {DEFAULT_MAX_PARALLEL_PER_GPU}/GPU")
    print(f"   Early Stopping patience: {DEFAULT_PATIENCE}")
    print(f"   Multi-task Loss type: {MULTITASK_CONFIG['loss_type'].upper()}")
    print(f"   FIRMS weight: {MULTITASK_CONFIG['firms_weight']}")
    print(f"   Other drivers weight: {MULTITASK_CONFIG['other_drivers_weight']}")
    print(f"   Ignore zero values: {'✅' if MULTITASK_CONFIG['ignore_zero_values'] else '❌'}")
    print(f"   Regression loss function: {MULTITASK_CONFIG['loss_function']}")
    if MULTITASK_CONFIG['loss_type'] == 'focal':
        print(f"   Focal Loss α: {TRAINING_CONFIG['focal_alpha']}")
        print(f"   Focal Loss γ: {TRAINING_CONFIG['focal_gamma']}")
    elif MULTITASK_CONFIG['loss_type'] == 'kldiv':
        print(f"   KL divergence temperature parameter: 1.0")
    elif MULTITASK_CONFIG['loss_type'] == 'multitask':
        print(f"   Unified loss function: {MULTITASK_CONFIG['loss_function']}")
    
    # 🔥 New: Position and weather feature status
    print(f"\n🔧 Data feature configuration:")
    print(f"   Position features: {'✅ Enabled' if DATA_CONFIG['enable_position_features'] else '❌ Disabled'}")
    print(f"   Future weather data: {'✅ Enabled' if DATA_CONFIG['enable_future_weather'] else '❌ Disabled'}")
    if DATA_CONFIG['enable_future_weather']:
        channels_str = ','.join(map(str, DATA_CONFIG['weather_channels']))
        print(f"   Weather channels: [{channels_str}] (Total {len(DATA_CONFIG['weather_channels'])})")
    
    # Calculate total input channels
    base_channels = 39
    additional_channels = 0
    if DATA_CONFIG['enable_position_features']:
        additional_channels += 1
    if DATA_CONFIG['enable_future_weather']:
        additional_channels += len(DATA_CONFIG['weather_channels'])
    
    total_channels = base_channels + additional_channels
    if additional_channels > 0:
        print(f"   Input channels: {base_channels} (base) + {additional_channels} (features) = {total_channels} (total)")
    else:
        print(f"   Input channels: {total_channels} (standard)")

def is_model_trained(model_name, model_type='standard'):
    """
    Check if the model has been trained
    Determine by checking if the final_epoch model file exists
    """
    model_save_dir = TRAINING_CONFIG[model_type]['model_save_dir']
    final_model_path = os.path.join(model_save_dir, f'{model_name}_final_epoch.pth')
    return os.path.exists(final_model_path)

def get_trained_model_paths(model_name, model_type='standard'):
    """
    Get all saved paths for trained models
    Returns a dictionary containing metric_name and path
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
            trained_paths[metric_type] = {'path': path, 'score': 0.0}  # score will be updated during testing
    
    return trained_paths

def filter_trained_models(model_list, model_type='standard', force_retrain=False):
    """
    Filter trained models
    Returns (list of models to train, dictionary of trained models)
    """
    if force_retrain:
        print(f"🔄 Force retrain mode: will train all {len(model_list)} {model_type} models")
        return model_list, {}
    
    models_to_train = []
    trained_models = {}
    
    print(f"🔍 Checking {model_type} model training status...")
    
    for model_name in model_list:
        if is_model_trained(model_name, model_type):
            trained_paths = get_trained_model_paths(model_name, model_type)
            trained_models[model_name] = trained_paths
            print(f"✅ {model_name}: Training completed ({len(trained_paths)} saved versions)")
        else:
            models_to_train.append(model_name)
            print(f"❌ {model_name}: Needs training")
    
    print(f"\n📊 {model_type} model status statistics:")
    print(f"   Need training: {len(models_to_train)} models")
    print(f"   Training completed: {len(trained_models)} models")
    
    if models_to_train:
        print(f"   Will train: {', '.join(models_to_train)}")
    if trained_models:
        print(f"   Skip training: {', '.join(trained_models.keys())}")
    
    return models_to_train, trained_models

def get_all_models(model_zoo_path):
    """Get all available models in the specified model_zoo"""
    model_files = []
    if os.path.exists(model_zoo_path):
        for file in os.listdir(model_zoo_path):
            if file.endswith('.py') and not file.startswith('__') and file != 'trash':
                model_name = file[:-3]  # Remove .py extension
                model_files.append(model_name)
    return sorted(model_files)

# Get standard model list
MODEL_LIST_STANDARD = get_all_models('model_zoo')

print(f"Found standard models: {MODEL_LIST_STANDARD}")

# Training configuration
TRAINING_CONFIG = {
    'use_wandb': WANDB_ENABLED,         # Use WandB configuration
    'seed': GLOBAL_SEED,                # Use random seed
    'patience': DEFAULT_PATIENCE,       # Use patience configuration
    'seq_len': 7,                      # Input sequence length
    'pred_len': 1,                      # Prediction sequence length
    'focal_alpha': 0.5,                 # Use optimal Focal Loss positive sample weight
    'focal_gamma': 2.0,                 # Focal Loss focus parameter
    
    # Standard model configuration
    'standard': {
        'epochs': 20,
        'batch_size': 128,
        'learning_rate': 5e-5,          # Lower learning rate
        'weight_decay': 1e-4,
        'T_0': 20,
        'T_mult': 2,
        'eta_min':1e-5,
        'max_grad_norm': 0.0,           # Enable gradient clipping to prevent gradient explosion; 0.0 means no clipping
        'model_save_dir': STANDARD_MODEL_DIR,
    },
}

# Data configuration
DATA_CONFIG = {
    'train_years': DEFAULT_TRAIN_YEARS,
    'val_years': DEFAULT_VAL_YEARS,
    'test_years': DEFAULT_TEST_YEARS,
    
    # Underlying dataset configuration (load full data)
    'positive_ratio': 1.0,           # Load all positive samples at the bottom layer
    'pos_neg_ratio': 2.0,            # Positive to negative sample ratio 1:1 at the bottom layer
    'resample_each_epoch': False,    # Disable resampling at the bottom layer, so always set to False
    'firms_min': 0,                  # Minimum value of FIRMS data (skip statistics)
    'firms_max': 100,                # Maximum value of FIRMS data (skip statistics)
    
    # Dynamic sampling configuration (sample per epoch)
    'enable_dynamic_sampling': True,   # Whether to enable dynamic sampling for training set
    'sampling_ratio': 0.3,            # Proportion of data to sample per epoch (0.0-1.0)
    
    # 🔥 New: Position information feature configuration
    'enable_position_features': False,  # Whether to enable position information feature (default enabled)
    'raster_size': (278, 130),         # Image size (height, width), used for normalization of position 278, 130 or 130, 278 ????
    
    # 🔥 New: Future weather data feature configuration  
    'enable_future_weather': False,    # Whether to enable future weather data feature (default disabled)
    'weather_channels': list(range(1, 13)),  # Weather data channel indices: 2-13 bands (indices 1-12)
}

# =============================================================================
# Custom dynamic sampling dataset class
# =============================================================================

class DynamicSamplingSubset(Dataset):
    """
    Support dynamic sampling of subset (simplified version)
    Each epoch randomly samples a specified proportion of data from a balanced dataset
    Since the underlying dataset is already 1:1 balanced, random sampling will maintain a similar proportion
    """
    def __init__(self, dataset, full_indices, sampling_ratio=1.0, enable_dynamic_sampling=False):
        """
        Args:
            dataset: Original dataset (already 1:1 balanced)
            full_indices: List of complete indices
            sampling_ratio: Proportion of data to use per epoch (0.0-1.0)
            enable_dynamic_sampling: Whether to enable dynamic sampling
        """
        self.dataset = dataset
        self.full_indices = full_indices
        self.sampling_ratio = sampling_ratio
        self.enable_dynamic_sampling = enable_dynamic_sampling
        
        # Current indices being used
        if enable_dynamic_sampling and sampling_ratio < 1.0:
            self.current_indices = self._sample_indices(epoch_seed=42)
        else:
            self.current_indices = full_indices
            
        print(f"📊 DynamicSamplingSubset initialization:")
        print(f"   Total indices: {len(full_indices)}")
        print(f"   Current use: {len(self.current_indices)}")
        print(f"   Sampling ratio: {sampling_ratio:.1%}")
        print(f"   Dynamic sampling: {'Enabled' if enable_dynamic_sampling else 'Disabled'}")
    
    def _sample_indices(self, epoch_seed):
        """Randomly sample indices based on epoch seed"""
        if not self.enable_dynamic_sampling or self.sampling_ratio >= 1.0:
            return self.full_indices
            
        # Set random seed for reproducibility
        np.random.seed(epoch_seed)
        random.seed(epoch_seed)
        
        # Calculate sampling quantity
        sample_size = int(len(self.full_indices) * self.sampling_ratio)
        sample_size = max(1, sample_size)  # At least ensure 1 sample
        sample_size = min(sample_size, len(self.full_indices))  # No more than available quantity
        
        # Random sampling
        sampled_indices = np.random.choice(self.full_indices, size=sample_size, replace=False)
        return sampled_indices.tolist()
    
    def resample_for_epoch(self, epoch):
        """Resample for new epoch"""
        if not self.enable_dynamic_sampling:
            return
            
        # old_size = len(self.current_indices)
        self.current_indices = self._sample_indices(epoch_seed=42 + epoch)
        # new_size = len(self.current_indices)
        
        # print(f"🔄 Epoch {epoch+1}: Re-sampling completed {old_size} → {new_size} samples (ratio: {self.sampling_ratio:.1%})")
    
    def __len__(self):
        return len(self.current_indices)
    
    def __getitem__(self, idx):
        # Map current indices to actual indices in the original dataset
        actual_idx = self.current_indices[idx]
        return self.dataset[actual_idx]

# =============================================================================
# Utility functions
# =============================================================================

def set_seed(seed):
    """Set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def worker_init_fn(worker_id):
    """DataLoader worker initialization function, ensuring reproducibility across multiple processes"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class FIRMSNormalizer:
    """FIRMS data normalizer"""
    
    def __init__(self, method='divide_by_100', firms_min=None, firms_max=None):
        self.method = method
        self.firms_min = firms_min
        self.firms_max = firms_max
        self.fitted = False
        
    def fit(self, data_loader):
        """Fit normalizer"""
        if self.firms_min is not None and self.firms_max is not None:
            print(f"🚀 Using specified FIRMS data range: [{self.firms_min}, {self.firms_max}]")
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
            print(f"✅ Normalizer initialization completed (transformed range: {self.global_min:.2f}-{self.global_max:.2f})")
            return
            
        print("🔧 Collecting FIRMS data for normalization fitting...")
        firms_values = []
        
        # Simplified data collection process to improve performance
        # progress = SimpleProgressTracker()
        for i, batch in enumerate(data_loader):
            # Only show progress at key nodes instead of every batch
            if i % max(1, len(data_loader) // 10) == 0:  # Show progress every 10%
                print(f"📊 Progress of collecting FIRMS data: {i+1}/{len(data_loader)} ({100*(i+1)/len(data_loader):.0f}%)", end='\r')
            # progress.update(i+1, len(data_loader), "📊 Collecting FIRMS data")
            past, future, _ = batch
            firms_data = past[:, 0, :]  # FIRMS channel (B, T)
            firms_values.append(firms_data.numpy())
        
        print()  # Newline
        
        all_firms = np.concatenate(firms_values, axis=0).flatten()
        valid_firms = all_firms[all_firms != 255]  # Filter out NoData values (255)
        
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
        print(f"✅ {self.method.upper()} normalization completed (range: {self.global_min:.2f}-{self.global_max:.2f})")
        
    def normalize(self, firms_data):
        """Normalize FIRMS data"""
        if not self.fitted:
            raise ValueError("Normalizer not fitted, please call fit() method")
            
        if self.method == 'log1p_minmax':
            log1p_data = torch.log1p(firms_data)
            if self.global_max > self.global_min:
                return (log1p_data - self.global_min) / (self.global_max - self.global_min)
            else:
                return log1p_data
        elif self.method == 'divide_by_100':
            return firms_data / 100.0
        else:
            raise ValueError(f"Unsupported normalization method: {self.method}")
    
    def transform_tensor(self, tensor_data):
        """Apply normalization transformation to tensor data (compatible method)"""
        return self.normalize(tensor_data)
    
    def inverse_transform_numpy(self, normalized_data):
        """Inverse transform normalized numpy data"""
        if not self.fitted:
            raise ValueError("Normalizer not fitted, please call fit() method")
        
        if isinstance(normalized_data, torch.Tensor):
            normalized_data = normalized_data.cpu().numpy()
        
        if self.method == 'log1p_minmax':
            if self.global_max > self.global_min:
                # Inverse normalization: y = x * (max - min) + min
                log_data = normalized_data * (self.global_max - self.global_min) + self.global_min
            else:
                log_data = normalized_data
            # Inverse log1p transformation: expm1(log_data)
            return np.expm1(log_data)
        elif self.method == 'divide_by_100':
            return normalized_data * 100.0
        else:
            raise ValueError(f"Unsupported normalization method: {self.method}")

def add_position_features(data, metadata_list, raster_size):
    """
    Add position information feature to data
    
    Args:
        data: Input data (batch_size, channels, time_steps)
        metadata_list: List of metadata, containing position information
        raster_size: Image size (height, width)
    
    Returns:
        Enhanced data with position features (batch_size, channels+1, time_steps)
    """
    batch_size, channels, time_steps = data.shape
    height, width = raster_size
    
    # Create position feature tensor
    position_features = torch.zeros(batch_size, 1, time_steps, device=data.device)
    
    for i, metadata in enumerate(metadata_list):
        # 🔥 Fix: Correctly extract position information from metadata
        try:
            if isinstance(metadata, dict):
                # If metadata is in dictionary format (returned from _parse_dataset_key in dataload.py)
                row = metadata.get('row', 0)
                col = metadata.get('col', 0)
            elif hasattr(metadata, '__len__') and len(metadata) >= 3:
                # If metadata is in list/tuple format
                if len(metadata) >= 3:
                    # Try different metadata formats
                    # Format 1: [date_int, row, col, ...]
                    try:
                        row, col = int(metadata[1]), int(metadata[2])
                    except (ValueError, IndexError):
                        # Format 2: [date_int, firms_value, row, col, ...]
                        try:
                            row, col = int(metadata[2]), int(metadata[3])
                        except (ValueError, IndexError):
                            row, col = 0, 0
                else:
                    row, col = 0, 0
            else:
                # If metadata is a single value (possibly date_int)
                row, col = 0, 0
        except Exception as e:
            print(f"⚠️ Failed to extract position information: {e}, metadata: {metadata}")
            row, col = 0, 0
        
        # Normalize position coordinates to 0-1 range
        norm_row = row / (height - 1) if height > 1 else 0.0
        norm_col = col / (width - 1) if width > 1 else 0.0
        
        # Encode normalized position information as a single value (can use different encoding methods)
        # Here we use simple linear combination: row_weight * norm_row + col_weight * norm_col
        position_value = 0.5 * norm_row + 0.5 * norm_col
        
        # Apply position features to all time steps
        position_features[i, 0, :] = position_value
    
    # Concatenate position features to original data
    enhanced_data = torch.cat([data, position_features], dim=1)
    return enhanced_data

def add_weather_features(past_data, future_data, weather_channels):
    """
    Extract weather features from future data and add to past data
    
    Args:
        past_data: Past data (batch_size, channels, past_time_steps)
        future_data: Future data (batch_size, channels, future_time_steps)  
        weather_channels: List of weather data channel indices
    
    Returns:
        Enhanced past data with weather features (batch_size, channels+len(weather_channels), past_time_steps)
    """
    batch_size, channels, past_time_steps = past_data.shape
    future_time_steps = future_data.shape[2]
    
    # Extract future weather data (batch_size, len(weather_channels), future_time_steps)
    future_weather = future_data[:, weather_channels, :]
    
    # Repeat or interpolate future weather data to match past time step length
    if future_time_steps != past_time_steps:
        # Use linear interpolation to adjust time dimension
        future_weather = F.interpolate(
            future_weather, 
            size=past_time_steps, 
            mode='linear', 
            align_corners=False
        )
    
    # Concatenate weather features to past data
    enhanced_past = torch.cat([past_data, future_weather], dim=1)
    return enhanced_past

def normalize_batch(past, future, firms_normalizer=None, metadata_list=None):
    """
    Normalize batch data and optionally add position information and weather data features
    
    Args:
        past: Past data (batch_size, channels, past_time_steps)
        future: Future data (batch_size, channels, future_time_steps)
        firms_normalizer: FIRMS data normalizer
        metadata_list: List of metadata, used for extracting position information
    
    Returns:
        Processed (past, future) data tuple
    """
    # 🔥 Key: First handle all NaN values in all channels, replace them with 0
    nan_mask_past = torch.isnan(past)
    past[nan_mask_past] = 0.0
    nan_mask_future = torch.isnan(future)
    future[nan_mask_future] = 0.0
    
    # Normalize the 0th channel (FIRMS) for both past and future
    if firms_normalizer is not None:
        past[:, 0, :] = firms_normalizer.normalize(past[:, 0, :])
        future[:, 0, :] = firms_normalizer.normalize(future[:, 0, :])
    
    # 🔥 New: Add position information feature
    if DATA_CONFIG['enable_position_features'] and metadata_list is not None:
        past = add_position_features(past, metadata_list, DATA_CONFIG['raster_size'])
        # Note: Future data usually doesn't need position feature addition, as position information is mainly used as input
        
    # 🔥 New: Add future weather data feature
    if DATA_CONFIG['enable_future_weather']:
        past = add_weather_features(past, future, DATA_CONFIG['weather_channels'])
    
    return past, future

def load_model(model_name, configs, model_type='standard'):
    """Load model dynamically (using model_zoo)"""
    try:
        # Check for special dependencies
        if model_name in ['Mamba', 'Reformer', 'Transformer', 'iTransformer', 's_mamba']:
            try:
                import mamba_ssm
            except ImportError:
                print(f"⚠️ Model {model_name} requires mamba_ssm library")
                print(f"💡 Suggest using mamba_env environment: conda activate mamba_env")
                raise ImportError(f"Model {model_name} requires mamba_ssm library, please run in mamba_env environment")
        
        # Use model_zoo uniformly
        model_zoo_path = os.path.join(os.getcwd(), 'model_zoo')
        module_name = f'model_zoo.{model_name}'
        
        if model_zoo_path not in sys.path:
            sys.path.insert(0, model_zoo_path)
        
        module = importlib.import_module(module_name)
        Model = getattr(module, 'Model')
        
        return Model(configs), model_type
    except Exception as e:
        print(f"Failed to load {model_type} model {model_name}: {e}")
        raise

def calculate_detailed_metrics(output, target):
    """Calculate detailed regression and binary classification metrics, including MSE, MAE, PR-AUC"""
    # Original output values used for regression metrics
    output_raw = output.view(-1).cpu().numpy()
    target_np = target.view(-1).cpu().numpy()
    
    # Calculate MSE and MAE (regression metrics, using original output values)
    mse = np.mean((output_raw - target_np) ** 2)
    mae = np.mean(np.abs(output_raw - target_np))
    
    # Probability values from Sigmoid processing used for classification metrics
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
        print(f"Error calculating metrics: {e}")
        return 0.0, 0.0, 0.0, 0.0, mse, mae
    
    return precision, recall, f1, pr_auc, mse, mae

def calculate_optimal_f1_metrics(output, target):
    """Calculate detailed metrics at optimal F1 threshold for testing - debugging version"""
    # Original output values used for regression metrics
    output_raw = output.view(-1).cpu().numpy()
    target_np = target.view(-1).cpu().numpy()
    
    # Calculate MSE and MAE (regression metrics, using original output values)
    mse = np.mean((output_raw - target_np) ** 2)
    mae = np.mean(np.abs(output_raw - target_np))
    
    # Probability values from Sigmoid processing used for classification metrics
    pred_probs = torch.sigmoid(output).view(-1).cpu().numpy()
    target_binary = (target_np > 0).astype(int)
    
    # 🔍 Debugging information: Analyze input data characteristics
    print(f"   🔍 Data statistics:")
    print(f"       Number of prediction samples: {len(pred_probs)}")
    print(f"       Number of true positive samples: {np.sum(target_binary)}")
    print(f"       Proportion of true positives: {np.sum(target_binary) / len(target_binary):.4f}")
    print(f"       Range of predicted probabilities: [{np.min(pred_probs):.4f}, {np.max(pred_probs):.4f}]")
    print(f"       Mean of predicted probabilities: {np.mean(pred_probs):.4f}")
    print(f"       Standard deviation of predicted probabilities: {np.std(pred_probs):.4f}")
    
    unique_targets = np.unique(target_binary)
    if len(unique_targets) < 2:
        return 0.0, 0.0, 0.0, 0.0, mse, mae
    
    try:
        # Calculate PR-AUC
        pr_auc = average_precision_score(target_binary, pred_probs)
        
        # Find optimal F1 threshold
        thresholds = np.linspace(0, 1, 100)  # Search using 1000 threshold points
        best_f1 = 0.0
        best_precision = 0.0
        best_recall = 0.0
        best_threshold = 0.5
        
        # 🔍 Debugging: Record metrics for all thresholds
        all_recalls = []
        all_precisions = []
        all_f1s = []
        
        for threshold in thresholds:
            pred_binary_thresh = (pred_probs > threshold).astype(int)
            
            # 🔍 Prevent division by zero error, add more detailed check
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
        
        # 🔍 Debugging information: Analyze distribution of recalls
        all_recalls = np.array(all_recalls)
        unique_recalls = np.unique(all_recalls)
        print(f"       Found {len(unique_recalls)} different recall values")
        print(f"      Range of recalls: [{np.min(all_recalls):.4f}, {np.max(all_recalls):.4f}]")
        print(f"       Highest recall: {np.max(all_recalls):.6f}")
        print(f"       Best F1 threshold: {best_threshold:.3f} (F1={best_f1:.4f})")
        
        # 🔍 If all recalls are the same, there's a problem
        if len(unique_recalls) == 1:
            print(f"      ⚠️ Warning: All thresholds have the same recall = {unique_recalls[0]:.6f}")
            print(f"      Possible cause: Model predictions are too concentrated or data distribution is abnormal")
            
        # 🔍 Analyze threshold distribution
        recall_counts = {}
        for r in all_recalls:
            r_rounded = round(r, 6)
            recall_counts[r_rounded] = recall_counts.get(r_rounded, 0) + 1
        
        print(f"      Top 5 recall values frequency:")
        for r, count in sorted(recall_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"         {r:.6f}: {count} times")
        
    except Exception as e:
        print(f"Error calculating optimal F1 metrics: {e}")
        return 0.0, 0.0, 0.0, 0.0, mse, mae
    
    return best_precision, best_recall, best_f1, pr_auc, mse, mae

class Config:
    """Configuration class - fixing type safety issues"""
    def __init__(self, model_name, model_type='standard'):
        self.model_name = model_name  # Add model name attribute
        self.model_type = model_type
        config = TRAINING_CONFIG[model_type]
        
        # Basic training parameters - ensure type safety
        self.epochs = int(config['epochs'])
        self.batch_size = int(config['batch_size'])
        self.learning_rate = float(config['learning_rate'])
        self.weight_decay = float(config['weight_decay'])
        self.T_0 = int(config['T_0'])
        self.T_mult = int(config['T_mult'])
        self.eta_min = float(config['eta_min'])
        self.max_grad_norm = float(config['max_grad_norm'])
        
        # Sequence parameters - ensure they are integers, avoiding Config object issues
        self.seq_len = int(TRAINING_CONFIG['seq_len'])
        self.pred_len = int(TRAINING_CONFIG['pred_len'])
        self.label_len = 0  # Default label length
        
        # Get configuration based on model type (using unified adapter)
        try:
            from model_adapter_unified import get_unified_model_configs
            model_configs = get_unified_model_configs(model_name, model_type)
            
            # Safely set configuration, ensuring correct numerical types
            for key, value in model_configs.items():
                if key in ['seq_len', 'pred_len']:
                    continue  # Skip, using fixed values we've set
                elif isinstance(value, (int, float, str, bool)):
                    setattr(self, key, value)
                elif value is None:
                    setattr(self, key, None)
                else:
                    # For complex types, try converting to basic type
                    try:
                        if isinstance(value, list):
                            setattr(self, key, value)
                        else:
                            setattr(self, key, value)
                    except:
                        print(f"⚠️ Skipping configuration {key}={value} (type: {type(value)})")
                        
        except Exception as e:
            print(f"⚠️ Dynamic configuration import failed: {e}, using default configuration")
            # Use default configuration
            self.d_model = 512
            self.n_heads = 8
            self.d_ff = 2048
            self.e_layers = 2
            self.d_layers = 2
            self.d_state = 16
            self.d_conv = 4
            self.expand = 2
            
            # General model parameters
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
            self.use_norm = True  # True by default
            self.distil = True
            self.label_len = 3 if model_name in ['Autoformer', 'Autoformer_M'] else 0
        
        # Add special configuration needed for new models
        self.task_name = 'long_term_forecast'  # New models generally need this parameter
        
        # Add special configuration for specific models
        if model_name == 'DLinear':
            self.moving_avg = 25  # DLinear needs moving_avg for series_decomp
            self.individual = False  # DLinear's individual parameter
            
        elif model_name == 'CrossLinear':
            self.features = 'M'  # CrossLinear needs features parameter
            self.patch_len = 16  # CrossLinear needs patch-related parameters
            self.alpha = 0.5
            self.beta = 0.5
            
        elif model_name == 'TimesNet':
            self.top_k = 5  # TimesNet needs parameter
            self.num_kernels = 6
            
        elif model_name == 'Mamba':
            # Special parameters for Mamba are already set in the base configuration
            pass
        
        # FIRMS data normalization parameters
        self.normalize_firms = True
        self.firms_normalization_method = 'divide_by_100'
        self.binarization_threshold = 0.0
        self.firms_min = int(DATA_CONFIG['firms_min'])
        self.firms_max = int(DATA_CONFIG['firms_max'])
        
        # Focal Loss parameters  
        self.focal_alpha = float(TRAINING_CONFIG['focal_alpha'])
        self.focal_gamma = float(TRAINING_CONFIG['focal_gamma'])
        
        # Multi-task learning parameters
        self.firms_weight = float(MULTITASK_CONFIG['firms_weight'])
        self.other_drivers_weight = float(MULTITASK_CONFIG['other_drivers_weight'])
        self.ignore_zero_values = MULTITASK_CONFIG['ignore_zero_values']
        self.loss_function = MULTITASK_CONFIG['loss_function']
        self.loss_type = MULTITASK_CONFIG['loss_type']  # New: Loss function type selection
        
        # Dataset split
        self.train_years = DATA_CONFIG['train_years']
        self.val_years = DATA_CONFIG['val_years']
        self.test_years = DATA_CONFIG['test_years']
        
        # 🔥 New: Dynamic update of model channel configuration
        self.update_model_channels()
    
    # 🔥 New: Dynamic calculation of input channel number
    def calculate_input_channels(self):
        """
        Calculate input channel number dynamically based on configuration
        Base channel number + position feature channel number + weather data channel number
        """
        base_channels = 39  # Base channel number
        additional_channels = 0
        
        # Position information feature (+1 channel) - use config object attributes first, otherwise use global configuration
        enable_position = getattr(self, 'enable_position_features', DATA_CONFIG['enable_position_features'])
        if enable_position:
            additional_channels += 1
            
        # Weather data feature - use config object attributes first, otherwise use global configuration
        enable_weather = getattr(self, 'enable_future_weather', DATA_CONFIG['enable_future_weather'])
        if enable_weather:
            weather_channels = getattr(self, 'weather_channels', DATA_CONFIG['weather_channels'])
            additional_channels += len(weather_channels)
            
        return base_channels + additional_channels
    
    def update_model_channels(self):
        """Update model's input/output channel configuration"""
        # Dynamically calculate input channel number
        dynamic_enc_in = self.calculate_input_channels()
        
        # Update encoder input channel number
        self.enc_in = dynamic_enc_in
        
        # Decoder input channel number usually matches encoder
        self.dec_in = dynamic_enc_in
        
        # Output channel number remains 39 (predict all original channels)
        self.c_out = 39
        
        # Print channel information for debugging - use config object attributes instead of global configuration
        features_info = []
        enable_position = getattr(self, 'enable_position_features', DATA_CONFIG['enable_position_features'])
        enable_weather = getattr(self, 'enable_future_weather', DATA_CONFIG['enable_future_weather'])
        
        if enable_position:
            features_info.append("Position information(+1)")
        if enable_weather:
            weather_channels = getattr(self, 'weather_channels', DATA_CONFIG['weather_channels'])
            features_info.append(f"Weather data(+{len(weather_channels)})")
        
        if features_info:
            print(f"🔧 {self.model_name} Dynamic channel configuration: {self.enc_in} input -> {self.c_out} output (additional features: {', '.join(features_info)})")
        else:
            print(f"🔧 {self.model_name} Standard channel configuration: {self.enc_in} input -> {self.c_out} output")

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
    Multi-task Focal Loss:
    - Use Focal Loss for FIRMS channel (0th channel)
    - Use regression loss for other drivers (MSE/Huber/MAE)
    - Support weight adjustment and ignore0 value functionality
    """
    def __init__(self, firms_weight=1.0, other_drivers_weight=0.1, 
                 focal_alpha=0.25, focal_gamma=2.0,
                 ignore_zero_values=True, regression_loss='mse'):
        super(MultiTaskFocalLoss, self).__init__()
        self.firms_weight = firms_weight
        self.other_drivers_weight = other_drivers_weight
        self.ignore_zero_values = ignore_zero_values
        
        # Focal Loss for FIRMS
        # self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')
        self.focal_loss = nn.BCELoss()
        # Regression loss function for other drivers
        if regression_loss == 'huber':
            self.regression_loss_fn = nn.HuberLoss(reduction='none')
        elif regression_loss == 'mse':
            self.regression_loss_fn = nn.MSELoss(reduction='none')
        elif regression_loss == 'mae':
            self.regression_loss_fn = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f"Unsupported regression loss function type: {regression_loss}")
        
        self.regression_loss_type = regression_loss
    
    def forward(self, predictions, targets):
        """
        Calculate multi-task loss
        
        Args:
            predictions: (B, T, C) Model prediction results, C may be greater than 39 (if there are additional features)
            targets: (B, T, 39) True labels, always 39 channels
            
        Returns:
            total_loss: Total loss
            loss_components: Dictionary of loss components
        """
        batch_size, seq_len, pred_channels = predictions.shape
        _, _, target_channels = targets.shape
        
        # 🔥 Key fix: If predicted channel number is greater than target channel number, only take the first target_channels channels
        # This is because the extra channels (e.g., weather data) have already been used as input features, so we shouldn't calculate loss for them
        if pred_channels > target_channels:
            predictions = predictions[:, :, :target_channels]
        #   print(f"🔧 Loss calculation: Predicted channel number ({pred_channels}) > Target channel number ({target_channels}), only calculating loss for the first {target_channels} channels")
        
        # Separate FIRMS and other drivers
        firms_pred = predictions[:, :, 0]      # (B, T) - FIRMS channel used for binary classification
        firms_target = targets[:, :, 0]        # (B, T)
        other_pred = predictions[:, :, 1:]     # (B, T, 38) - Other channels used for regression
        other_target = targets[:, :, 1:]       # (B, T, 38)
        
        # 1. Calculate Focal Loss for FIRMS (binary classification)
        # Convert FIRMS target to binary classification labels (1 if >0, 0 if =0)
        firms_binary_target = (firms_target > 0).float()
        firms_pred = torch.sigmoid(firms_pred)  # Focal loss doesn't need sigmoid because it's already included in the focal loss
        firms_loss = self.focal_loss(firms_pred, firms_binary_target) * self.firms_weight
        
        # 2. Calculate regression loss for other drivers
        other_loss = self.regression_loss_fn(other_pred, other_target)  # (B, T, 38)
        
        if self.ignore_zero_values:
            # Create non-zero mask to ignore0 values
            non_zero_mask = (other_target != 0.0).float()  # (B, T, 38)
            
            # Calculate effective number of samples
            valid_samples = non_zero_mask.sum()
            
            if valid_samples > 0:
                # Only calculate loss for non-zero values
                masked_loss = other_loss * non_zero_mask
                other_loss = masked_loss.sum() / valid_samples
            else:
                # If there are no valid samples, loss is 0
                other_loss = torch.tensor(0.0, device=predictions.device)
        else:
            # Don't ignore0 values, just calculate average loss
            other_loss = other_loss.mean()
        
        other_loss = other_loss * self.other_drivers_weight
        
        # Total loss
        total_loss = firms_loss + other_loss
        
        # Return loss component information
        loss_components = {
            'total_loss': total_loss.item(),
            'firms_loss': firms_loss.item(),
            'other_drivers_loss': other_loss.item(),
            'firms_weight': self.firms_weight,
            'other_drivers_weight': self.other_drivers_weight,
            # 'focal_alpha': self.focal_loss.alpha,  # Not needed when using BCELoss
            # 'focal_gamma': self.focal_loss.gamma,
            'regression_loss_type': self.regression_loss_type,
            'loss_type': 'focal'  # New: Loss function type identifier
        }
        # print(firms_loss, other_loss)
        return total_loss, loss_components  # total_loss, loss_components

class MultiTaskKLDivLoss(nn.Module):
    """
    Multi-task KL divergence Loss:
    - Use KL divergence for FIRMS channel (0th channel)
    - Use KL divergence for regression of other drivers
    - Support weight adjustment and ignore0 value functionality
    """
    def __init__(self, firms_weight=1.0, other_drivers_weight=0.1, 
                 ignore_zero_values=True, temperature=1.0, epsilon=1e-8):
        super(MultiTaskKLDivLoss, self).__init__()
        self.firms_weight = firms_weight
        self.other_drivers_weight = other_drivers_weight
        self.ignore_zero_values = ignore_zero_values
        self.temperature = temperature  # Temperature parameter, used to control smoothness of distribution
        self.epsilon = epsilon  # Small constant to prevent numerical instability
        
        # KL divergence loss function (reduction='none' for manual handling)
        self.kldiv_loss = nn.KLDivLoss(reduction='none')
    
    def _to_probability_distribution(self, x, is_classification=False):
        """
        Convert input to probability distribution
        
        Args:
            x: Input tensor
            is_classification: Whether this is a classification task (FIRMS channel)
            
        Returns:
            Probability distribution tensor
        """
        if is_classification:
            # For classification tasks, use sigmoid+normalization
            # x shape: (...,) or (..., 1)
            if x.dim() > 0 and x.shape[-1] == 1:
                x = x.squeeze(-1)  # Remove last dimension if it's 1
            
            prob = torch.sigmoid(x / self.temperature)
            # Create binomial distribution: [1-p, p]
            prob_neg = 1 - prob
            prob_dist = torch.stack([prob_neg, prob], dim=-1)  # (..., 2)
            # Normalize to ensure it's a probability distribution
            prob_dist = prob_dist / (prob_dist.sum(dim=-1, keepdim=True) + self.epsilon)
        else:
            # For regression tasks, convert values to positive then normalize
            # Use softplus to ensure positive values: softplus(x) = log(1 + exp(x))
            positive_vals = F.softplus(x / self.temperature)
            # Normalize to probability distribution
            prob_dist = positive_vals / (positive_vals.sum(dim=-1, keepdim=True) + self.epsilon)
        
        # Add small constant to prevent log(0)
        prob_dist = prob_dist + self.epsilon
        prob_dist = prob_dist / prob_dist.sum(dim=-1, keepdim=True)
        
        return prob_dist
    
    def forward(self, predictions, targets):
        """
        Calculate multi-task KL divergence loss
        
        Args:
            predictions: (B, T, C) Model prediction results, C may be greater than 39 (if there are additional features)
            targets: (B, T, 39) True labels, always 39 channels
            
        Returns:
            total_loss: Total loss
            loss_components: Dictionary of loss components
        """
        batch_size, seq_len, pred_channels = predictions.shape
        _, _, target_channels = targets.shape
        
        # 🔥 Key fix: If predicted channel number is greater than target channel number, only take the first target_channels channels
        # This is because the extra channels (e.g., weather data) have already been used as input features, so we shouldn't calculate loss for them
        if pred_channels > target_channels:
            predictions = predictions[:, :, :target_channels]
            print(f"🔧 KL divergence loss calculation: Predicted channel number ({pred_channels}) > Target channel number ({target_channels}), only calculating loss for the first {target_channels} channels")
        
        # Separate FIRMS and other drivers
        firms_pred = predictions[:, :, 0]      # (B, T) - FIRMS channel
        firms_target = targets[:, :, 0]        # (B, T)
        other_pred = predictions[:, :, 1:]     # (B, T, 38) - Other channels
        other_target = targets[:, :, 1:]       # (B, T, 38)
        
        # 1. Calculate KL divergence loss for FIRMS (classification task)
        # Convert FIRMS target to binary classification labels (1 if >0, 0 if =0)
        firms_binary_target = (firms_target > 0).float()
        
        # Convert to probability distribution
        firms_pred_dist = self._to_probability_distribution(firms_pred, is_classification=True)  # (B, T, 2)
        firms_target_dist = self._to_probability_distribution(firms_binary_target, is_classification=True)  # (B, T, 2)
        
        # Calculate KL divergence: KL(target || pred)
        firms_kl = self.kldiv_loss(firms_pred_dist.log(), firms_target_dist)  # (B, T, 2)
        firms_loss = firms_kl.sum(dim=-1).mean() * self.firms_weight  # Sum over distribution dimensions then average
        
        # 2. Calculate KL divergence loss for other drivers (regression task)
        # Convert to probability distribution
        other_pred_dist = self._to_probability_distribution(other_pred, is_classification=False)  # (B, T, 38)
        other_target_dist = self._to_probability_distribution(other_target, is_classification=False)  # (B, T, 38)
        
        # Calculate KL divergence
        other_kl = self.kldiv_loss(other_pred_dist.log(), other_target_dist)  # (B, T, 38)
        
        if self.ignore_zero_values:
            # Create non-zero mask to ignore0 values
            non_zero_mask = (other_target != 0.0).float()  # (B, T, 38)
            
            # Calculate effective number of samples
            valid_samples = non_zero_mask.sum()
            
            if valid_samples > 0:
                # Only calculate loss for non-zero values
                masked_kl = other_kl * non_zero_mask
                other_loss = masked_kl.sum() / valid_samples
            else:
                # If there are no valid samples, loss is 0
                other_loss = torch.tensor(0.0, device=predictions.device)
        else:
            # Don't ignore0 values, just calculate average loss
            other_loss = other_kl.mean()
        
        other_loss = other_loss * self.other_drivers_weight
        
        # Total loss
        total_loss = firms_loss + other_loss
        
        # Return loss component information
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
    Multi-metric Early Stopping: Monitor F1, Recall, PR-AUC simultaneously
    Any improvement in any metric resets the counter
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
            'mae': float('inf')  # Lower MAE is better
        }
        self.best_weights = None
        self.should_stop = False
    
    def __call__(self, metrics, model):
        """
        Check if training should stop
        Args:
            metrics: dict containing 'f1', 'recall', 'pr_auc', 'mae'
            model: Model instance
        Returns:
            bool: Whether training should stop
        """
        f1_improved = metrics['f1'] > (self.best_metrics['f1'] + self.min_delta)
        recall_improved = metrics['recall'] > (self.best_metrics['recall'] + self.min_delta)
        pr_auc_improved = metrics['pr_auc'] > (self.best_metrics['pr_auc'] + self.min_delta)
        mae_improved = metrics['mae'] < (self.best_metrics['mae'] - self.min_delta)  # Lower MAE is better
        
        # Any improvement resets the counter
        if f1_improved or recall_improved or pr_auc_improved or mae_improved:
            # Update best metrics
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
            print(f"📈 Metrics improved! F1: {metrics['f1']:.4f}, Recall: {metrics['recall']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}, MAE: {metrics['mae']:.6f}")
        else:
            self.counter += 1
            print(f"⏳ No improvement ({self.counter}/{self.patience}): F1: {metrics['f1']:.4f}, Recall: {metrics['recall']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}, MAE: {metrics['mae']:.6f}")
        
        if self.counter >= self.patience:
            self.should_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                print("🔄 Restored best weights")
        
        return self.should_stop
    
    def save_checkpoint(self, model):
        """Save best weights"""
        self.best_weights = model.state_dict().copy()

class MultiTaskLoss(nn.Module):
    """Multi-task loss function, supports weighted loss calculation for different channels"""
    
    def __init__(self, firms_weight=1.0, other_drivers_weight=0.1, 
                 ignore_zero_values=True, loss_function='huber'):
        super(MultiTaskLoss, self).__init__()
        self.firms_weight = firms_weight
        self.other_drivers_weight = other_drivers_weight
        self.ignore_zero_values = ignore_zero_values
        
        # Select loss function
        if loss_function == 'huber':
            self.loss_fn = nn.HuberLoss(reduction='none')
        elif loss_function == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif loss_function == 'mae':
            self.loss_fn = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f"Unsupported loss function type: {loss_function}")
    
    def forward(self, predictions, targets):
        """
        Calculate multi-task loss
        
        Args:
            predictions: (B, T, C) Model prediction results, C may be greater than 39 (if there are additional features)
            targets: (B, T, 39) True labels, always 39 channels
            
        Returns:
            total_loss: Total loss
            loss_components: Dictionary of loss components
        """
        batch_size, seq_len, pred_channels = predictions.shape
        _, _, target_channels = targets.shape
        
        # 🔥 Key fix: If predicted channel number is greater than target channel number, only take the first target_channels channels
        # This is because the extra channels (e.g., weather data) have already been used as input features, so we shouldn't calculate loss for them
        if pred_channels > target_channels:
            predictions = predictions[:, :, :target_channels]
            print(f"🔧 Multi-task loss calculation: Predicted channel number ({pred_channels}) > Target channel number ({target_channels}), only calculating loss for the first {target_channels} channels")
        
        # Separate FIRMS and other drivers
        firms_pred = predictions[:, :, 0:1]  # (B, T, 1)
        firms_target = targets[:, :, 0:1]    # (B, T, 1)
        other_pred = predictions[:, :, 1:]   # (B, T, 38)
        other_target = targets[:, :, 1:]     # (B, T, 38)
        
        # Calculate FIRMS loss
        firms_loss = self.loss_fn(firms_pred, firms_target)  # (B, T, 1)
        firms_loss = firms_loss.mean() * self.firms_weight
        
        # Calculate other drivers loss
        other_loss = self.loss_fn(other_pred, other_target)  # (B, T, 38)
        
        if self.ignore_zero_values:
            # Create non-zero mask to ignore0 values
            non_zero_mask = (other_target != 0.0).float()  # (B, T, 38)
            
            # Calculate effective number of samples
            valid_samples = non_zero_mask.sum()
            
            if valid_samples > 0:
                # Only calculate loss for non-zero values
                masked_loss = other_loss * non_zero_mask
                other_loss = masked_loss.sum() / valid_samples
            else:
                # If there are no valid samples, loss is 0
                other_loss = torch.tensor(0.0, device=predictions.device)
        else:
            # Don't ignore0 values, just calculate average loss
            other_loss = other_loss.mean()
        
        other_loss = other_loss * self.other_drivers_weight
        
        # Total loss
        total_loss = firms_loss   # + other_loss
        
        # Return loss component information
        loss_components = {
            'total_loss': total_loss.item(),
            'firms_loss': firms_loss.item(),
            'other_drivers_loss': other_loss.item(),
            'firms_weight': self.firms_weight,
            'other_drivers_weight': self.other_drivers_weight,
            'loss_type': 'multitask'  # New: Loss function type identifier
        }
        # print(firms_loss, other_loss)
        return total_loss, loss_components

# =============================================================================
# Progress display utility functions
# =============================================================================

class SimpleProgressTracker:
    """Simplified progress tracker, mimicking tqdm default effect but without progress bar"""
    def __init__(self):
        self.start_time = None
        
    def update(self, current, total, prefix="Progress", clear_on_complete=True):
        """
        Update progress display - tqdm style but without progress bar
        """
        if self.start_time is None:
            self.start_time = time.time()
            
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Calculate speed (items/second)
        speed = current / elapsed_time if elapsed_time > 0 else 0
        
        # Calculate percentage
        percent = int((current / total) * 100)
        
        # tqdm style display format
        if current == total:
            # Format when complete
            progress_text = f"\r{prefix}: {percent:3d}%|{current}/{total} [{self._format_time(elapsed_time)}, {speed:.2f}it/s]"
        else:
            # Format while in progress, calculate estimated remaining time
            if speed > 0:
                remaining_time = (total - current) / speed
                progress_text = f"\r{prefix}: {percent:3d}%|{current}/{total} [{self._format_time(elapsed_time)}<{self._format_time(remaining_time)}, {speed:.2f}it/s]"
            else:
                # If speed is 0, use simplified format
                progress_text = f"\r{prefix}: {percent:3d}%|{current}/{total} [{self._format_time(elapsed_time)}<?, ?it/s]"
        
        print(progress_text, end='', flush=True)
        
        # Handle completion
        if current == total:
            if clear_on_complete:
                # Clear progress bar
                print('\r' + ' ' * len(progress_text) + '\r', end='', flush=True)
            else:
                print()  # Keep final state and add newline
    
    def _format_time(self, seconds):
        """Format time display - tqdm style"""
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
    Compatibility function - maintain simple dynamic progress display
    """
    if show_percent:
        percent = (current / total) * 100
        progress_text = f"\r{prefix}: {current}/{total} ({percent:.1f}%)"
    else:
        progress_text = f"\r{prefix}: {current}/{total}"
    
    print(progress_text, end='', flush=True)
    
    # Clear progress bar after completion
    if current == total:
        print('\r' + ' ' * len(progress_text) + '\r', end='', flush=True)

def save_epoch_metrics_to_log(epoch_metrics, log_file, model_name, model_type):
    """
    Save training and validation metrics for each epoch to log file
    """
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Detailed training log - {model_name} ({model_type})\n")
            f.write(f"{'='*80}\n")
            f.write(f"Record time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write header
            f.write(f"{'Epoch':<6} {'Train_Loss':<11} {'Train_P':<8} {'Train_R':<8} {'Train_F1':<9} {'Train_PRAUC':<11} {'Train_MSE':<10} {'Train_MAE':<10} ")
            f.write(f"{'Val_Loss':<9} {'Val_P':<6} {'Val_R':<6} {'Val_F1':<7} {'Val_PRAUC':<9} {'Val_MSE':<8} {'Val_MAE':<8} {'LR':<10}\n")
            f.write("-" * 150 + "\n")
            
            # Write data for each epoch
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
            
        print(f"📝 Detailed epoch log saved to: {log_file}")
        
    except Exception as e:
        print(f"⚠️ Failed to save epoch log: {e}")

def save_structured_results_to_csv(structured_results, model_type):
    """
    Save structured test results as CSV files for classification
    Save separately: best_f1.csv, best_recall.csv, final_epoch.csv
    Each CSV contains: Model, precision, recall, f1, pr_auc
    """
    if not structured_results:
        print("⚠️ No results to save")
        return
    
    # Determine save directory
    save_dir = STANDARD_MODEL_DIR
    
    # Create save directory if it doesn't exist
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 3 types of models to save
    model_categories = ['f1', 'recall', 'final_epoch']
    classification_metrics = ['precision', 'recall', 'f1', 'pr_auc']  # Only save classification metrics
    
    saved_files = []
    
    for category in model_categories:
        # Prepare CSV data
        csv_data = []
        columns = ['Model'] + classification_metrics
        
        # Add data rows
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
        
        # Generate filename and save
        if category == 'final_epoch':
            filename = f"final_epoch.csv"
        else:
            filename = f"best_{category}.csv"
        
        csv_filepath = os.path.join(save_dir, filename)
        
        if csv_data:  # Only save if there's data
            df = pd.DataFrame(csv_data, columns=columns)
            df.to_csv(csv_filepath, index=False)
            saved_files.append(csv_filepath)
            
            print(f"📊 {filename}: {len(csv_data)} model results saved")
        else:
            print(f"⚠️  {filename}: No data available")
    
    # Summarize save situation
    print(f"\n✅ Total {len(saved_files)} CSV files saved to: {save_dir}")
    for filepath in saved_files:
        print(f"   📄 {os.path.basename(filepath)}")
    
    print(f"\n📋 CSV file structure explanation:")
    print(f"   best_f1.csv: Performance evaluation of the best F1 model")
    print(f"   best_recall.csv: Performance evaluation of the best Recall model")
    print(f"   final_epoch.csv: Performance evaluation of the final epoch model")
    print(f"   Each file contains: Model, precision, recall, f1, pr_auc")

# =============================================================================
# Core training and testing functions
# =============================================================================

def train_single_model(model_name, device, train_loader, val_loader, test_loader, firms_normalizer, model_type='standard', log_file=None):
    """Train a single model"""
    print(f"\n🔥 Training {model_type} model: {model_name}")
    
    config = Config(model_name, model_type)
    
    # Create detailed logger
    epoch_metrics = []  # Record metrics for each epoch
    
    # Initialize wandb (if enabled)
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
                # Multi-task learning configuration
                "multitask_enabled": True,
                "firms_weight": config.firms_weight,
                "other_drivers_weight": config.other_drivers_weight,
                "ignore_zero_values": config.ignore_zero_values,
                "loss_function": config.loss_function,
            },
            reinit=True
        )
        print(f"✅ WandB initialization completed: {wandb_run.name}")
    
    # Use unified adapter
    from model_adapter_unified import UnifiedModelAdapter
    adapter = UnifiedModelAdapter(config)
    
    try:
        model, _ = load_model(model_name, config, model_type)
        model = model.to(device)
    except Exception as e:
        print(f"❌ {model_type} model {model_name} failed to load: {e}")
        if wandb_run:
            wandb_run.finish()
        return None
    
    # Optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Select loss function type based on configuration
    if config.loss_type == 'focal':
        # Use multi-task Focal Loss
        criterion = MultiTaskFocalLoss(
            firms_weight=config.firms_weight,
            other_drivers_weight=config.other_drivers_weight,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
            ignore_zero_values=config.ignore_zero_values,
            regression_loss=config.loss_function  # 'mse', 'huber', 'mae'
        )
        
        print(f"🔍 Multi-task Focal Loss configuration:")
        print(f"   FIRMS weight: {config.firms_weight}, other drivers weight: {config.other_drivers_weight}")
        print(f"   Focal α: {config.focal_alpha}, Focal γ: {config.focal_gamma}")
        print(f"   Regression loss: {config.loss_function}, Ignore zero values: {config.ignore_zero_values}")
        
    elif config.loss_type == 'kldiv':
        # Use multi-task KL divergence Loss
        criterion = MultiTaskKLDivLoss(
            firms_weight=config.firms_weight,
            other_drivers_weight=config.other_drivers_weight,
            ignore_zero_values=config.ignore_zero_values,
            temperature=1.0,  # Can be added to configuration later
            epsilon=1e-8
        )
        
        print(f"🔍 Multi-task KL divergence Loss configuration:")
        print(f"   FIRMS weight: {config.firms_weight}, other drivers weight: {config.other_drivers_weight}")
        print(f"   温度参数: 1.0, 忽略0值: {config.ignore_zero_values}")
        
    elif config.loss_type == 'multitask':
        # Use multi-task loss function
        criterion = MultiTaskLoss(
            firms_weight=config.firms_weight,
            other_drivers_weight=config.other_drivers_weight,
            ignore_zero_values=config.ignore_zero_values,
            loss_function=config.loss_function
        )
        
        print(f"🔍 Multi-task loss function configuration:")
        print(f"   FIRMS weight: {config.firms_weight}, other drivers weight: {config.other_drivers_weight}")
        print(f"   忽略0值: {config.ignore_zero_values}")
        print(f"   损失函数: {config.loss_function}")
    
    else:
        raise ValueError(f"Unsupported loss function type: {config.loss_type}. Supported types: 'focal', 'kldiv', 'multitask'")
    
    print(f"🎯 Current loss function being used: {config.loss_type.upper()}")
    
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.T_0, T_mult=config.T_mult, eta_min=config.eta_min
    )
    
    # Early stopping
    early_stopping = MultiMetricEarlyStopping(patience=TRAINING_CONFIG['patience'], min_delta=0.0001, restore_best_weights=True)
    
    # Track best metrics and model paths
    best_metrics = {
        'f1': {'score': 0.0, 'path': None},
        'recall': {'score': 0.0, 'path': None},
        'pr_auc': {'score': 0.0, 'path': None},
        'mae': {'score': float('inf'), 'path': None},  # Lower MAE is better, initialized to infinity
        'mse': {'score': float('inf'), 'path': None}   # Lower MSE is better, initialized to infinity
    }
    
    print(f"🚀 Starting training {config.epochs} epochs...")
    
    for epoch in range(config.epochs):
        # Resample training set for each epoch if enabled
        if hasattr(train_loader.dataset, 'resample_for_epoch'):
            train_loader.dataset.resample_for_epoch(epoch)
        
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        # Training phase - simplify progress display for performance
        # train_progress = SimpleProgressTracker()
        for i, batch in enumerate(train_loader):
            # Comment out detailed training progress display to reduce CPU overhead
            # train_progress.update(i+1, len(train_loader), f"🔥 Epoch {epoch+1}/{config.epochs} Training")
            
            past, future, metadata_list = batch
            past, future = past.to(device), future.to(device)
            
            # 🔥 Fix: Don't delete the 0th channel, just set its data to 0, keeping the completeness of 39 channels
            # past[:, 0, :] = 0.0  # Set the 0th channel (FIRMS) to 0 instead of deleting
                        
            if firms_normalizer is not None:
                past, future = normalize_batch(past, future, firms_normalizer, metadata_list)
            
            date_strings = [str(int(metadata[0])) for metadata in metadata_list]  # B 1, yyyymmdd
            
            future_truncated = future[:, :, :config.pred_len].transpose(1, 2)
            target = future_truncated[:, :, 0]  # If this is single-channel prediction for focal loss, use [:, :, 0]
            # target = (target > config.binarization_threshold).float()
            
            # Forward propagation
            # if model_name == 's_mamba':
            #     past_transposed = past.transpose(1, 2)
            #     past_truncated = past_transposed[:, -config.seq_len:, :]
                
            #     output = model(past_truncated, date_strings)
            # else:
            # x_mark_enc: B T 7 (year_norm, month_sin, month_cos, day_sin, day_cos, weekday_sin, weekday_cos)
            x_enc, x_mark_enc, x_dec, x_mark_dec = adapter.adapt_inputs(past, future, date_strings)
            x_enc, x_mark_enc, x_dec, x_mark_dec = x_enc.to(device), x_mark_enc.to(device), x_dec.to(device), x_mark_dec.to(device)
            
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)  # B T C
                        
            # Multi-task learning: Predict all 39 channels
            # output shape: (B, T, C) where C=39
            # target shape: (B, T, C) where C=39
            target_all_channels = future_truncated  # Use all channels as target

            # Calculate multi-task Focal loss
            # Set high confidence threshold to avoid low confidence samples affecting loss function
            # Only binaryize FIRMS channel, keep other channels as is
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
            # Only save prediction for FIRMS channel for metric calculation
            train_preds.append(output[:, :, 0].detach())
            train_targets.append(target_all_channels[:, :, 0].detach())
        
        # Calculate training metrics
        train_loss /= len(train_loader)
        train_preds = torch.cat(train_preds, dim=0)
        train_targets = torch.cat(train_targets, dim=0)
        train_precision, train_recall, train_f1, train_pr_auc, train_mse, train_mae = calculate_detailed_metrics(train_preds, train_targets)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            # Validation phase - simplify progress display for performance
            # val_progress = SimpleProgressTracker()
            for i, batch in enumerate(val_loader):
                # Comment out detailed validation progress display to reduce CPU overhead
                # val_progress.update(i+1, len(val_loader), f"📊 Epoch {epoch+1}/{config.epochs} Validation")
                
                past, future, metadata_list = batch
                past, future = past.to(device), future.to(device)
                
                # 🔥 Fix: Don't delete the 0th channel, just set its data to 0, keeping the completeness of 39 channels
                # past[:, 0, :] = 0.0  # Set the 0th channel (FIRMS) to 0 instead of deleting
                
                # Why normalize future data!?!
                if firms_normalizer is not None:
                    past, future = normalize_batch(past, future, firms_normalizer, metadata_list)
                
                date_strings = [str(int(metadata[0])) for metadata in metadata_list]
                
                future_truncated = future[:, :, :config.pred_len].transpose(1, 2)
                target = future_truncated[:, :, 0]  # If this is single-channel prediction for focal loss, use [:, :, 0]
                # target = (target > config.binarization_threshold).float()
                
                # if model_name == 's_mamba':
                #     past_transposed = past.transpose(1, 2)
                #     past_truncated = past_transposed[:, -config.seq_len:, :]
                #     output = model(past_truncated, date_strings)
                # else:
                
                # x_mark_enc: B T 5 (year_norm, month_sin, month_cos, day_sin, day_cos, weekday_norm)
                x_enc, x_mark_enc, x_dec, x_mark_dec = adapter.adapt_inputs(past, future, date_strings)
                x_enc, x_mark_enc, x_dec, x_mark_dec = x_enc.to(device), x_mark_enc.to(device), x_dec.to(device), x_mark_dec.to(device)
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                
                # Multi-task learning: Predict all 39 channels
                target_all_channels = future_truncated  # Use all channels as target
                
                # Calculate multi-task Focal loss
                # target_all_channels = target_all_channels.clone()
                # target_all_channels[:, :, 0] = (target_all_channels[:, :, 0] > 10).float()
                loss, loss_components = criterion(output, target_all_channels)
                val_loss += loss.item()
                
                # Only save prediction for FIRMS channel for metric calculation
                val_preds.append(output[:, :, 0].detach())
                val_targets.append(target_all_channels[:, :, 0].detach())
        
        # Calculate validation metrics
        val_loss /= len(val_loader)
        val_preds = torch.cat(val_preds, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
        val_precision, val_recall, val_f1, val_pr_auc, val_mse, val_mae = calculate_detailed_metrics(val_preds, val_targets)
        
        # Record metrics for current epoch
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
                # Add multi-task loss component information
                "firms_weight": config.firms_weight,
                "other_drivers_weight": config.other_drivers_weight,
                "loss_function": config.loss_function,
                "ignore_zero_values": config.ignore_zero_values
        }
        epoch_metrics.append(epoch_data)
        
        # Record to wandb
        if wandb_run:
            wandb.log(epoch_data)
        
        # Display training progress
        print(f"Epoch {epoch+1}/{config.epochs} - "
              f"Train Loss: {train_loss:.4f} (F1: {train_f1:.4f}) - "
              f"Val Loss: {val_loss:.4f} (F1: {val_f1:.4f}) - "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Display multi-task loss component information (every 5 epochs)
        if (epoch + 1) % 5 == 0:
            print(f"    Multi-task loss components - FIRMS: {config.firms_weight:.1f}, "
                  f"other drivers: {config.other_drivers_weight:.1f}, "
                  f"loss function: {config.loss_function}")
        
        # Save best model for each metric
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
            # Lower MAE and MSE are better, other metrics are better if higher
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
        
        # Print epoch summary
        print(f'Epoch {epoch+1:3d}/{config.epochs} | Train: Loss={train_loss:.4f}, F1={train_f1:.4f}, MSE={train_mse:.6f}, MAE={train_mae:.6f} | '
              f'Val: Loss={val_loss:.4f}, P={val_precision:.4f}, R={val_recall:.4f}, F1={val_f1:.4f}, PR-AUC={val_pr_auc:.4f}, MSE={val_mse:.6f}, MAE={val_mae:.6f} | '
              f'LR={optimizer.param_groups[0]["lr"]:.2e}')
        
        # Early stopping check (using multiple metrics)
        if early_stopping({'f1': val_f1, 'recall': val_recall, 'pr_auc': val_pr_auc, 'mae': val_mae, 'mse': val_mse}, model):
            print(f"⏹️  Early stopping triggered at epoch {epoch+1} (patience={TRAINING_CONFIG['patience']}, counter={early_stopping.counter})")
            break
        
        lr_scheduler.step()
    
    # Save model parameters for last epoch
    final_model_path = os.path.join(model_save_dir, f'{model_name}_final_epoch.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"💾 Last epoch model saved: {final_model_path}")
    
    # Add last epoch path to return result
    best_metrics['final_epoch'] = {
        'score': epoch + 1,  # Record final epoch number
        'path': final_model_path
    }
    
    # Save detailed epoch training log
    if log_file:
        save_epoch_metrics_to_log(epoch_metrics, log_file, model_name, model_type)
    
    # Close wandb
    if wandb_run:
        wandb.finish()
    
    return best_metrics

def test_model(model_name, model_path, device, test_loader, firms_normalizer, model_type='standard'):
    """Test model"""
    print(f"\n📊 Testing {model_type} model: {model_name}")
    
    config = Config(model_name, model_type)
    
    # Use unified adapter
    from model_adapter_unified import UnifiedModelAdapter
    adapter = UnifiedModelAdapter(config)
    
    # Select loss function type based on configuration
    if config.loss_type == 'focal':
        # Create multi-task Focal loss function for testing
        criterion = MultiTaskFocalLoss(
            firms_weight=config.firms_weight,
            other_drivers_weight=config.other_drivers_weight,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
            ignore_zero_values=config.ignore_zero_values,
            regression_loss=config.loss_function
        )
        
        print(f"🔍 Multi-task Focal Loss configuration for testing:")
        print(f"   FIRMS weight: {config.firms_weight}, other drivers weight: {config.other_drivers_weight}")
        print(f"   Focal α: {config.focal_alpha}, Focal γ: {config.focal_gamma}")
        print(f"   Regression loss: {config.loss_function}, Ignore zero values: {config.ignore_zero_values}")
        
    elif config.loss_type == 'kldiv':
        # Create multi-task KL divergence loss function for testing
        criterion = MultiTaskKLDivLoss(
            firms_weight=config.firms_weight,
            other_drivers_weight=config.other_drivers_weight,
            ignore_zero_values=config.ignore_zero_values,
            temperature=1.0,
            epsilon=1e-8
        )
        
        print(f"🔍 Multi-task KL divergence Loss configuration for testing:")
        print(f"   FIRMS weight: {config.firms_weight}, other drivers weight: {config.other_drivers_weight}")
        print(f"   温度参数: 1.0, 忽略0值: {config.ignore_zero_values}")
        
    elif config.loss_type == 'multitask':
        # Create multi-task loss function for testing
        criterion = MultiTaskLoss(
            firms_weight=config.firms_weight,
            other_drivers_weight=config.other_drivers_weight,
            ignore_zero_values=config.ignore_zero_values,
            loss_function=config.loss_function
        )
        
        print(f"🔍 Multi-task loss function configuration for testing:")
        print(f"   FIRMS weight: {config.firms_weight}, other drivers weight: {config.other_drivers_weight}")
        print(f"   忽略0值: {config.ignore_zero_values}")
        print(f"   损失函数: {config.loss_function}")
    
    else:
        raise ValueError(f"Unsupported loss function type: {config.loss_type}. Supported types: 'focal', 'kldiv', 'multitask'")
    
    print(f"🎯 Loss function being used for testing: {config.loss_type.upper()}")
    
    try:
        model, _ = load_model(model_name, config, model_type)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"❌ {model_type} model {model_name} failed to load for testing: {e}")
        return None
    
    test_preds = []
    test_targets = []
    total_test_loss = 0.0
    
    with torch.no_grad():
        # Simplify test progress display for performance
        # test_progress = SimpleProgressTracker()
        for i, batch in enumerate(test_loader):
            # Comment out detailed test progress display to reduce CPU overhead
            # test_progress.update(i+1, len(test_loader), f"🧪 Testing {model_name}")
            past, future, metadata_list = batch
            past, future = past.to(device), future.to(device)
            
            # 🔥 Fix: Don't delete the 0th channel, just set its data to 0, keeping the completeness of 39 channels
            # past[:, 0, :] = 0.0  # Set the 0th channel (FIRMS) to 0 instead of deleting
            
            if firms_normalizer is not None:
                past, future = normalize_batch(past, future, firms_normalizer, metadata_list)
            
            date_strings = [str(int(metadata[0])) for metadata in metadata_list]
            
            future_truncated = future[:, :, :config.pred_len].transpose(1, 2)
            target = future_truncated[:, :, 0]  # If this is single-channel prediction for focal loss, use [:, :, 0]
            # target = (target > config.binarization_threshold).float()
            
            # if model_name == 's_mamba':
            #     past_transposed = past.transpose(1, 2)
            #     past_truncated = past_transposed[:, -config.seq_len:, :]
            #     output = model(past_truncated, date_strings)
            # else:
            # x_mark_enc: B T 5 (year_norm, month_sin, month_cos, day_sin, day_cos, weekday_norm)
            x_enc, x_mark_enc, x_dec, x_mark_dec = adapter.adapt_inputs(past, future, date_strings)
            x_enc, x_mark_enc, x_dec, x_mark_dec = x_enc.to(device), x_mark_enc.to(device), x_dec.to(device), x_mark_dec.to(device)
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Multi-task learning: Predict all 39 channels
            target_all_channels = future_truncated  # Use all channels as target
            
            # Calculate multi-task Focal loss
            # target_all_channels = target_all_channels.clone()
            # target_all_channels[:, :, 0] = (target_all_channels[:, :, 0] > 10).float()
            loss, loss_components = criterion(output, target_all_channels)
            total_test_loss += loss.item()
            
            # Only save prediction for FIRMS channel for metric calculation
            test_preds.append(output[:, :, 0].detach())
            test_targets.append((target_all_channels[:, :, 0].detach()).float())
    
    # Calculate test metrics - using F1 optimal threshold
    test_preds = torch.cat(test_preds, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    precision, recall, f1, pr_auc, mse, mae = calculate_optimal_f1_metrics(test_preds, test_targets)
    
    avg_test_loss = total_test_loss / len(test_loader)
    
    print(f"✅ {model_name} {model_type} model test results: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, PR-AUC={pr_auc:.4f}, MSE={mse:.6f}, MAE={mae:.6f}")
    print(f"    Multi-task loss: {avg_test_loss:.6f} (FIRMS weight: {config.firms_weight:.1f}, other drivers weight: {config.other_drivers_weight:.1f})")
    
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
        'loss_type': config.loss_type,  # New: Loss function type information
        'ignore_zero_values': config.ignore_zero_values
    }

def train_and_test_models(model_list, model_type, device, train_loader, val_loader, test_loader, firms_normalizer, force_retrain=False):
    """Train and test a group of models"""
    print(f"\n🔥 Starting training {model_type} model group")
    print(f"📋 Original model list: {len(model_list)} {model_type} models")
    print(f"📊 {model_type} model list: {', '.join(model_list)}")
    
    # Filter trained models
    models_to_train, trained_models = filter_trained_models(model_list, model_type, force_retrain)
    
    # Train new models
    model_results = []
    failed_models = []
    
    if models_to_train:
        print(f"\n🚀 Starting training {len(models_to_train)} {model_type} models that need training...")
        for i, model_name in enumerate(models_to_train):
            print(f"\n🔄 {model_type} training progress: {i+1}/{len(models_to_train)} (overall: {i+1+len(trained_models)}/{len(model_list)})")
        try:
            result = train_single_model(
                model_name, device, train_loader, val_loader, test_loader, firms_normalizer, model_type
            )
            if result is not None:
                best_metrics = result
                print(f"✅ {model_name} {model_type} model training completed, saved model:")
                for metric_name, metric_info in best_metrics.items():
                        if metric_name == 'final_epoch':
                            print(f"  Final epoch model (epoch {metric_info['score']}): {metric_info['path']}")
                        else:
                            print(f"  Best {metric_name} model ({metric_info['score']:.4f}): {metric_info['path']}")
                model_results.append((model_name, best_metrics))
            else:
                failed_models.append(model_name)
        except Exception as e:
            print(f"❌ {model_name} {model_type} model training failed: {e}")
            failed_models.append(model_name)
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        print(f"\n✅ All {model_type} models have already been trained, skipping training phase")
    
    # Add trained models to results
    for model_name, trained_paths in trained_models.items():
        model_results.append((model_name, trained_paths))
        print(f"📋 Loaded trained model: {model_name} ({len(trained_paths)} saved versions)")
    
    print(f"\n📈 {model_type} model preparation completed!")
    print(f"    New training: {len(models_to_train)} models")
    print(f"    Trained: {len(trained_models)} models") 
    print(f"    Training failed: {len(failed_models)} models")
    print(f"    Total available: {len(model_results)} models")
    
    if failed_models:
        print(f"❌ Failed {model_type} models: {', '.join(failed_models)}")
    
    # Testing phase
    print("\n" + "="*60)
    print("🧪 Testing phase - evaluate trained models")
    print("="*60)
    
    # Dictionary to store structured test results
    structured_results = {}
    
    for model_name, metrics in model_results:
        print(f"\n📋 Testing model: {model_name}")
        print("-" * 40)
        
        # Initialize dictionary for model's results
        structured_results[model_name] = {
            'f1': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'mse': None, 'mae': None},
            'recall': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'mse': None, 'mae': None},
            'pr_auc': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'mse': None, 'mae': None},
            'mae': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'mse': None, 'mae': None},
            'final_epoch': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'mse': None, 'mae': None}
        }
        
        # Test all saved models (including best model and final epoch model)
        for metric_name, metric_info in metrics.items():
            if metric_info['path'] is not None:
                if metric_name == 'final_epoch':
                    print(f"\n🎯 Testing final epoch model (epoch: {metric_info['score']})")
                else:
                    print(f"\n🎯 Testing best {metric_name.upper()} model (score: {metric_info['score']:.4f})")
                try:
                    result = test_model(model_name, metric_info['path'], device, test_loader, firms_normalizer, model_type)
                    if result:
                        # Save to structured results
                        structured_results[model_name][metric_name] = {
                            'precision': result['precision'],
                            'recall': result['recall'],
                            'f1': result['f1'],
                            'pr_auc': result['pr_auc'],
                            'mse': result['mse'],
                            'mae': result['mae']
                        }
                        print(f"✅ {model_name} ({metric_name}) test completed")
                except Exception as e:
                    print(f"❌ {model_name} ({metric_name}) test failed: {str(e)}")
    
    if not structured_results:
        print("⚠️ No models passed testing!")
        return None
    
    # Save structured results to CSV
    save_structured_results_to_csv(structured_results, model_type)
    
    # Output final summary of results
    print("\n" + "="*80)
    print("📊 Final test results summary")
    print("="*80)
    
    # Display results in tabular format
    for model_name, model_results in structured_results.items():
        print(f"\n🔥 Model: {model_name}")
        print("-" * 80)
        print(f"{'Metric type':<12} {'Precision':<8} {'Recall':<8} {'F1 score':<8} {'PR-AUC':<8} {'MSE':<10} {'MAE':<10}")
        print("-" * 80)
        for metric_type, metrics in model_results.items():
            if metrics['precision'] is not None:
                display_type = "FINAL" if metric_type == 'final_epoch' else metric_type.upper()
                print(f"{display_type:<12} {metrics['precision']:<8.4f} {metrics['recall']:<8.4f} {metrics['f1']:<8.4f} {metrics['pr_auc']:<8.4f} {metrics['mse']:<10.6f} {metrics['mae']:<10.6f}")
    
    print(f"\n🎉 Training and testing completed! Total {len(model_results)} models trained")
    
    if failed_models:
        print(f"\n⚠️ Failed models: {failed_models}")
    
    print("\n📁 All models saved to corresponding directories")
    save_dir = STANDARD_MODEL_DIR
    print(f"Test results saved to directory: {save_dir}")
    
    return structured_results

def prepare_data_loaders():
    """Prepare data loaders"""
    print("📂 Loading data...")
    data_loader = TimeSeriesDataLoader(
        # h5_dir='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/pixel_samples_merged',
        h5_dir='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/full_datasets',
        positive_ratio=DATA_CONFIG['positive_ratio'],
        pos_neg_ratio=DATA_CONFIG['pos_neg_ratio'],
        resample_each_epoch=False  # Disable resampling at the bottom layer, use dynamic sampling instead
    )
    
    # Dataset split
    train_indices, val_indices, test_indices = data_loader.get_year_based_split(
        train_years=DATA_CONFIG['train_years'],
        val_years=DATA_CONFIG['val_years'],
        test_years=DATA_CONFIG['test_years']
    )
    
    # Use custom dynamic sampling dataset instead of standard Subset
    train_dataset = DynamicSamplingSubset(
        dataset=data_loader.dataset,
        full_indices=train_indices,
        sampling_ratio=DATA_CONFIG['sampling_ratio'],
        enable_dynamic_sampling=DATA_CONFIG['enable_dynamic_sampling']
    )
    
    # Validation set and test set use full data, no dynamic sampling
    val_dataset = Subset(data_loader.dataset, val_indices)
    test_dataset = Subset(data_loader.dataset, test_indices)
    
    print(f"📊 Dataset size:")
    print(f"    Training set: {len(train_dataset)} (full: {len(train_indices)})")
    print(f"    Validation set: {len(val_dataset)} (full data)")
    print(f"    Test set: {len(test_dataset)} (full data)")
    print(f"    Dynamic sampling: {'Enabled' if DATA_CONFIG['enable_dynamic_sampling'] else 'Disabled'}")
    if DATA_CONFIG['enable_dynamic_sampling']:
        print(f"    Sampling configuration: Randomly use {DATA_CONFIG['sampling_ratio']:.1%} of training data per epoch")
    
    return train_dataset, val_dataset, test_dataset, data_loader

def main():
    """Main function - train standard models"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Script for training wildfire prediction models')
    
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force retrain all models, ignoring existing model files')
    
    # Multi-task learning parameters
    parser.add_argument('--firms-weight', type=float, default=0.005,  # 0.005 for focal loss, 0.1 for multitask 
                       help='Loss weight for FIRMS prediction (default: 1.0)')
    parser.add_argument('--other-drivers-weight', type=float, default=1.0,
                       help='Loss weight for other drivers prediction (default: 1.0)')
    parser.add_argument('--loss-function', type=str, default='mse',
                       choices=['huber', 'mse', 'mae'],
                       help='Type of regression loss function for other drivers (default: mse)')
    parser.add_argument('--no-ignore-zero', action='store_true',
                       help='Do not ignore0 values in other drivers')
    
    # Focal Loss parameters
    parser.add_argument('--focal-alpha', type=float, default=0.5,
                       help='Alpha parameter for Focal Loss (default: 0.5)')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                       help='Gamma parameter for Focal Loss (default: 2.0)')
    
    # Loss function type selection parameter
    parser.add_argument('--loss-type', type=str, default='focal',  ####################################### use focal by default
                       choices=['focal', 'kldiv', 'multitask'],
                       help='Type of loss function to use (default: focal)')
    
    # 🔥 New: Position information and weather data feature parameters
    parser.add_argument('--enable-position-features', action='store_true',
                       help='Enable position information feature (default: disabled)')
    parser.add_argument('--enable-future-weather', action='store_true', 
                       help='Enable future weather data feature (default: disabled)')
    parser.add_argument('--weather-channels', type=str, default='1-12',
                       help='Range of weather data channels, format like "1-12" or "1,3,5-8" (default: 1-12)')
    
    args = parser.parse_args()
    
    # 🔥 New: Parse range of weather data channels
    def parse_channel_range(channel_str):
        """Parse channel range string, return list of channel indices"""
        channels = []
        for part in channel_str.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                channels.extend(range(start, end + 1))
            else:
                channels.append(int(part))
        return channels
    
    # Update data configuration
    global DATA_CONFIG
    DATA_CONFIG['enable_position_features'] = args.enable_position_features
    DATA_CONFIG['enable_future_weather'] = args.enable_future_weather
    
    if args.enable_future_weather:
        try:
            DATA_CONFIG['weather_channels'] = parse_channel_range(args.weather_channels)
        except ValueError as e:
            print(f"❌ Error: Invalid format for weather channel range: {args.weather_channels}")
            print(f"    Error message: {e}")
            print(f"    Correct format example: '1-12' or '1,3,5-8'")
            return
    
    # Update multi-task learning configuration
    global MULTITASK_CONFIG, TRAINING_CONFIG
    MULTITASK_CONFIG['firms_weight'] = args.firms_weight
    MULTITASK_CONFIG['other_drivers_weight'] = args.other_drivers_weight
    MULTITASK_CONFIG['loss_function'] = args.loss_function
    MULTITASK_CONFIG['ignore_zero_values'] = not args.no_ignore_zero
    MULTITASK_CONFIG['loss_type'] = args.loss_type  # New: Loss function type configuration
    
    # Update Focal Loss configuration
    TRAINING_CONFIG['focal_alpha'] = args.focal_alpha
    TRAINING_CONFIG['focal_gamma'] = args.focal_gamma
    
    print("🔥 Comprehensive wildfire forecasting model benchmark - unified version")
    
    # Display shared configuration status
    print_config_status()
    print()
    
    # Based on command line arguments, decide which models to train
    train_standard = True
    
    print("📋 Training plan: Standard models ✅")
    if args.force_retrain:
        print("🔄 Force retrain mode enabled, will ignore existing model files")
    
    # Initialize
    set_seed(TRAINING_CONFIG['seed'])
    # Use first visible CUDA device (controlled by CUDA_VISIBLE_DEVICES)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # Display actual GPU being used
    if torch.cuda.is_available():
        actual_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(actual_gpu)
        print(f"🖥️  Using device: cuda:0 (actual GPU: {gpu_name})")
        print(f"🔍 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    else:
        print(f"🖥️  Using device: {device}")
    
    # WandB configuration check
    if TRAINING_CONFIG['use_wandb']:
        if WANDB_AVAILABLE:
            print("✅ WandB monitoring enabled")
        else:
            print("⚠️ WandB monitoring configured but wandb not installed, will skip monitoring functionality")
    else:
        print("ℹ️ WandB monitoring disabled")
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"💾 GPU memory: {gpu_memory:.1f} GB")
    
    # Prepare data
    train_dataset, val_dataset, test_dataset, data_loader_obj = prepare_data_loaders()
    
    # Initialize FIRMS normalizer
    print("🔧 Initializing FIRMS normalizer...")
    firms_normalizer = FIRMSNormalizer(
        method='divide_by_100',
        firms_min=DATA_CONFIG['firms_min'],
        firms_max=DATA_CONFIG['firms_max']
    )
    
    # Create temporary data loader for fitting normalizer
    temp_loader = DataLoader(
        train_dataset, batch_size=512, shuffle=False, 
        num_workers=2, collate_fn=data_loader_obj.dataset.custom_collate_fn
    )
    firms_normalizer.fit(temp_loader)
    
    all_results = {}
    
    # ========== First stage: Train standard model_zoo models ==========
    if train_standard and MODEL_LIST_STANDARD:
        print(f"\n{'='*80}")
        print("🚀 First stage: Train standard model_zoo models")
        print(f"{'='*80}")
        
        # Create standard model data loader
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
    

    
    # ========== Final summary ==========
    print(f"\n{'='*80}")
    print("🎉 All models experiment completed!")
    print(f"{'='*80}")
    
    for model_type, results in all_results.items():
        if results:
            df = pd.DataFrame(results)
            df = df.sort_values('f1', ascending=False)
            best_model = df.iloc[0]
            print(f"\n🏆 Best {model_type} model: {best_model['model']}")
            print(f"   F1-Score: {best_model['f1']:.4f}")
            print(f"   Precision: {best_model['precision']:.4f}")
            print(f"   Recall: {best_model['recall']:.4f}")
            print(f"   PR-AUC: {best_model['pr_auc']:.4f}")
    
    print("\n📊 All results saved to corresponding CSV files!")

if __name__ == "__main__":
    main() 