#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single model training script - smart_parallel.py adapter
Supports command line arguments: --model, --type, --gpu, --log-dir
"""

import os
import sys

# Fix MKL conflict before importing any other modules
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

# Import numpy to initialize MKL first
import numpy as np

import argparse
import torch
import pandas as pd
from datetime import datetime

# added by Saeid
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from model_zoo.FLDmamba  import FLDMamba, WildfireConfigs
    print("✅ FLDmamba model loaded successfully")
except ImportError as e:
    print(f"⚠️ Could not import FLDmamba: {e}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Single model training script')
    parser.add_argument('--model', type=str, required=True, default= 's_mamba', help='Model name, s_mamba, s_mamba_full, ...')
    parser.add_argument('--type', type=str, default='standard', choices=['standard'], help='Model type')
    parser.add_argument('--gpu', type=int, default=1, help='GPU device number')
    parser.add_argument('--log-dir', type=str, default='./trash/smart_parallel_logs_single_model', help='Log directory')
    return parser.parse_args()

def setup_environment(gpu_id):
    """Set up training environment"""
    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Wait a moment to ensure environment variables take effect
    import time
    time.sleep(0.1)
    
    return True

def train_single_model_task(model_name, model_type, gpu_id, log_dir):
    """Train a single model task"""
    print(f"🚀 Starting single model training: {model_name} ({model_type}) on GPU {gpu_id}")
    
    # Set environment (including CUDA_VISIBLE_DEVICES)
    setup_environment(gpu_id)
    
    # Import training related modules
    from train_all_models_combined import (
        set_seed, TRAINING_CONFIG, prepare_data_loaders, FIRMSNormalizer,
        DATA_CONFIG, train_single_model, test_model, save_structured_results_to_csv,
        worker_init_fn
    )
    from torch.utils.data import DataLoader
    
    # Initialize
    set_seed(TRAINING_CONFIG['seed'])
    
    # Verify GPU settings
    if torch.cuda.is_available():
        device = torch.device('cuda:0')  # Since CUDA_VISIBLE_DEVICES is set, it will always be 0
        actual_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(actual_gpu)
        print(f"🖥️  Using device: Physical GPU {gpu_id} -> cuda:0 ({gpu_name})")
        
        # Verify GPU memory
        gpu_memory = torch.cuda.get_device_properties(actual_gpu).total_memory / 1024**3
        print(f"💾 GPU memory: {gpu_memory:.1f} GB")
    else:
        device = torch.device('cpu')
        print(f"⚠️  CUDA not available, using CPU")
        return False
    
    try:
        # Prepare data
        print(" Preparing data...")
        train_dataset, val_dataset, test_dataset, data_loader_obj, data_loader_test = prepare_data_loaders()
        
        # Initialize FIRMS normalizer
        print("🔧 Initializing FIRMS normalizer...")
        firms_normalizer = FIRMSNormalizer(
            method='log1p_minmax',
            firms_min=DATA_CONFIG['firms_min'],
            firms_max=DATA_CONFIG['firms_max']
        )
        
        # Create a temporary data loader for quick fitting (reduce worker count)
        temp_loader = DataLoader(
            train_dataset, batch_size=1024, shuffle=False,  # Increase batch size for faster fitting
            num_workers=1, collate_fn=data_loader_obj.dataset.custom_collate_fn  # Reduce workers to avoid memory conflicts
        )
        firms_normalizer.fit(temp_loader)
        
        # Create data loaders (optimized performance settings)
        config_key = model_type
        train_config = TRAINING_CONFIG[config_key]
        
        # Standard model configuration
        train_workers = 6
        val_workers = 4
        test_workers = 4
        
        train_loader = DataLoader(
            train_dataset, batch_size=train_config['batch_size'], shuffle=True, 
            num_workers=train_workers, collate_fn=data_loader_obj.dataset.custom_collate_fn, 
            worker_init_fn=worker_init_fn, pin_memory=True, persistent_workers=True  # Optimized settings
        )
        val_loader = DataLoader(
            val_dataset, batch_size=train_config['batch_size'], shuffle=False,
            num_workers=val_workers, collate_fn=data_loader_obj.dataset.custom_collate_fn, 
            worker_init_fn=worker_init_fn, pin_memory=True, persistent_workers=True  # Optimized settings
        )
        # Use appropriate collate_fn for test dataset
        if data_loader_test is not None:
            test_collate_fn = data_loader_test.dataset.custom_collate_fn
        else:
            test_collate_fn = data_loader_obj.dataset.custom_collate_fn
            
        test_loader = DataLoader(
            test_dataset, batch_size=train_config['batch_size'], shuffle=False,
            num_workers=test_workers, collate_fn=test_collate_fn, 
            worker_init_fn=worker_init_fn, pin_memory=True, persistent_workers=True  # Optimized settings
        )
        
        # Train model
        print(f"🔥 Starting training {model_name}...")
        result = train_single_model(
            model_name, device, train_loader, val_loader, test_loader, firms_normalizer, model_type
        )
        
        if result is None:
            print(f"❌ Training failed for {model_name} ({model_type})")
            return False
        
        print(f"✅ Training completed for {model_name} ({model_type})")
        
        # Test all saved models
        print(f"🧪 Starting to test all saved models for {model_name}...")
        
        # Dictionary to store structured test results
        structured_results = {model_name: {
            'f1': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'mse': None, 'mae': None},
            'recall': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'mse': None, 'mae': None},
            'pr_auc': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'mse': None, 'mae': None},
            'mae': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'mse': None, 'mae': None},
            'mse': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'mse': None, 'mae': None},
            'final_epoch': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'mse': None, 'mae': None}
        }}
        
        # Test best models and final epoch models
        for metric_name, metric_info in result.items():
            if metric_info['path'] is not None:
                if metric_name == 'final_epoch':
                    print(f"📊 Testing final_epoch model...")
                else:
                    print(f"📊 Testing {metric_name} model...")
                
                try:
                    test_result = test_model(model_name, metric_info['path'], device, test_loader, firms_normalizer, model_type)
                    if test_result:
                        # Save to structured results
                        structured_results[model_name][metric_name] = {
                            'precision': test_result['precision'],
                            'recall': test_result['recall'],
                            'f1': test_result['f1'],
                            'pr_auc': test_result['pr_auc'],
                            'mse': test_result['mse'],
                            'mae': test_result['mae']
                        }
                        print(f"   P={test_result['precision']:.4f}, R={test_result['recall']:.4f}, F1={test_result['f1']:.4f}, PR-AUC={test_result['pr_auc']:.4f}, MSE={test_result['mse']:.6f}, MAE={test_result['mae']:.6f}")
                except Exception as e:
                    print(f"❌ Testing failed for {model_name} ({metric_name}): {str(e)}")
        
        # Save results to CSV file
        print(f"💾 Saving test results...")
        
        # Prepare CSV data
        csv_data = []
        columns = ['Model']
        metric_types = ['f1', 'recall', 'pr_auc', 'mae', 'mse', 'final_epoch']
        metric_names = ['precision', 'recall', 'f1', 'pr_auc', 'mse', 'mae']
        
        for metric_type in metric_types:
            for metric_name in metric_names:
                display_type = "final_epoch" if metric_type == 'final_epoch' else f"best_{metric_type}"
                columns.append(f"{display_type}_{metric_name}")
        
        # Add data rows
        row = [model_name]
        for metric_type in metric_types:
            for metric_name in metric_names:
                value = structured_results[model_name][metric_type][metric_name]
                if value is not None:
                    row.append(f"{value:.6f}")
                else:
                    row.append("N/A")
        csv_data.append(row)
        
        # Save to CSV file
        df = pd.DataFrame(csv_data, columns=columns)
        csv_filename = os.path.join(log_dir, f"{model_name}_{model_type}_results.csv")
        df.to_csv(csv_filename, index=False)
        
        # Save summary file
        summary_filename = os.path.join(log_dir, f"{model_name}_{model_type}_summary.txt")
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write(f"Model training and testing summary - {model_name} ({model_type})\n")
            f.write(f"{'='*60}\n")
            f.write(f"Training time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"GPU device: {gpu_id}\n\n")
            
            f.write("Test results:\n")
            f.write("-" * 40 + "\n")
            for metric_type in metric_types:
                metrics = structured_results[model_name][metric_type]
                if metrics['precision'] is not None:
                    display_type = "FINAL" if metric_type == 'final_epoch' else metric_type.upper()
                    f.write(f"{display_type:<12} P={metrics['precision']:<8.4f} R={metrics['recall']:<8.4f} F1={metrics['f1']:<8.4f} PR-AUC={metrics['pr_auc']:<8.4f} MSE={metrics['mse']:<10.6f} MAE={metrics['mae']:<10.6f}\n")
        
        print(f"📄 Summary saved: {summary_filename}")
        print(f"🎉 Training and testing completed for {model_name} ({model_type})!")
        
        return True
        
    except Exception as e:
        print(f"💥 An exception occurred during training: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    args = parse_args()
    
    print(f"🔥 Single model trainer")
    print(f"📋 Model: {args.model}")
    print(f"📋 Type: {args.type}")
    print(f"📋 GPU: {args.gpu}")
    print(f"📋 Log directory: {args.log_dir}")
    print("=" * 50)
    
    # Ensure log directory exists
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set environment and train model
    success = train_single_model_task(args.model, args.type, args.gpu, args.log_dir)
    
    if success:
        print("🎉 Training completed successfully!")
        sys.exit(0)
    else:
        print("❌ Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 