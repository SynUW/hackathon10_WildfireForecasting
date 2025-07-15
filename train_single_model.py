#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单模型训练脚本 - smart_parallel.py适配器
支持命令行参数：--model, --type, --gpu, --log-dir
"""

import os
import sys

# 在导入任何其他模块之前修复MKL冲突
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

# 先导入numpy来初始化MKL
import numpy as np

import argparse
import torch
import pandas as pd
from datetime import datetime

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='单模型训练脚本')
    parser.add_argument('--model', type=str, required=True, help='模型名称')
    parser.add_argument('--type', type=str, default='standard', choices=['standard', '10x'], help='模型类型')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备号')
    parser.add_argument('--log-dir', type=str, default='./trash/smart_parallel_logs_single_model', help='日志目录')
    return parser.parse_args()

def setup_environment(gpu_id):
    """设置训练环境"""
    # 设置CUDA设备
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 等待一下确保环境变量生效
    import time
    time.sleep(0.1)
    
    return True

def train_single_model_task(model_name, model_type, gpu_id, log_dir):
    """训练单个模型任务"""
    print(f"🚀 开始训练单个模型: {model_name} ({model_type}) on GPU {gpu_id}")
    
    # 设置环境（包括CUDA_VISIBLE_DEVICES）
    setup_environment(gpu_id)
    
    # 导入训练相关模块
    from train_all_models_combined import (
        set_seed, TRAINING_CONFIG, prepare_data_loaders, FIRMSNormalizer,
        DATA_CONFIG, train_single_model, test_model, save_structured_results_to_csv,
        worker_init_fn
    )
    from torch.utils.data import DataLoader
    
    # 初始化
    set_seed(TRAINING_CONFIG['seed'])
    
    # 验证GPU设置
    if torch.cuda.is_available():
        device = torch.device('cuda:0')  # 由于设置了CUDA_VISIBLE_DEVICES，这里总是0
        actual_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(actual_gpu)
        print(f"🖥️  使用设备: Physical GPU {gpu_id} -> cuda:0 ({gpu_name})")
        
        # 验证GPU内存
        gpu_memory = torch.cuda.get_device_properties(actual_gpu).total_memory / 1024**3
        print(f"💾 GPU内存: {gpu_memory:.1f} GB")
    else:
        device = torch.device('cpu')
        print(f"⚠️  CUDA不可用，使用CPU")
        return False
    
    try:
        # 准备数据
        print("📂 准备数据...")
        train_dataset, val_dataset, test_dataset, data_loader_obj = prepare_data_loaders()
        
        # 初始化FIRMS归一化器
        print("🔧 初始化FIRMS归一化器...")
        firms_normalizer = FIRMSNormalizer(
            method='log1p_minmax',
            firms_min=DATA_CONFIG['firms_min'],
            firms_max=DATA_CONFIG['firms_max']
        )
        
        # 为归一化拟合创建临时数据加载器（减少worker数量，只用于快速拟合）
        temp_loader = DataLoader(
            train_dataset, batch_size=1024, shuffle=False,  # 增大batch_size加快拟合
            num_workers=1, collate_fn=data_loader_obj.dataset.custom_collate_fn  # 减少worker避免内存冲突
        )
        firms_normalizer.fit(temp_loader)
        
        # 创建数据加载器（优化性能设置）
        config_key = model_type
        train_config = TRAINING_CONFIG[config_key]
        
        # 根据模型类型调整worker数量，避免过多进程竞争
        if model_type == '10x':
            train_workers = 4  # 10x模型使用更多workers
            val_workers = 2
            test_workers = 2
        else:
            train_workers = 6  # 标准模型可以使用更多workers
            val_workers = 4
            test_workers = 4
        
        train_loader = DataLoader(
            train_dataset, batch_size=train_config['batch_size'], shuffle=True, 
            num_workers=train_workers, collate_fn=data_loader_obj.dataset.custom_collate_fn, 
            worker_init_fn=worker_init_fn, pin_memory=True, persistent_workers=True  # 优化设置
        )
        val_loader = DataLoader(
            val_dataset, batch_size=train_config['batch_size'], shuffle=False,
            num_workers=val_workers, collate_fn=data_loader_obj.dataset.custom_collate_fn, 
            worker_init_fn=worker_init_fn, pin_memory=True, persistent_workers=True  # 优化设置
        )
        test_loader = DataLoader(
            test_dataset, batch_size=train_config['batch_size'], shuffle=False,
            num_workers=test_workers, collate_fn=data_loader_obj.dataset.custom_collate_fn, 
            worker_init_fn=worker_init_fn, pin_memory=True, persistent_workers=True  # 优化设置
        )
        
        # 训练模型
        print(f"🔥 开始训练 {model_name}...")
        result = train_single_model(
            model_name, device, train_loader, val_loader, test_loader, firms_normalizer, model_type
        )
        
        if result is None:
            print(f"❌ {model_name} ({model_type}) 训练失败")
            return False
        
        print(f"✅ {model_name} ({model_type}) 训练完成")
        
        # 测试所有保存的模型
        print(f"🧪 开始测试 {model_name} 的所有保存模型...")
        
        # 用于存储结构化测试结果的字典
        structured_results = {model_name: {
            'f1': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'mse': None, 'mae': None},
            'recall': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'mse': None, 'mae': None},
            'pr_auc': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'mse': None, 'mae': None},
            'mae': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'mse': None, 'mae': None},
            'mse': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'mse': None, 'mae': None},
            'final_epoch': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'mse': None, 'mae': None}
        }}
        
        # 测试各个最佳模型和最后epoch模型
        for metric_name, metric_info in result.items():
            if metric_info['path'] is not None:
                if metric_name == 'final_epoch':
                    print(f"📊 测试 final_epoch 模型...")
                else:
                    print(f"📊 测试 {metric_name} 模型...")
                
                try:
                    test_result = test_model(model_name, metric_info['path'], device, test_loader, firms_normalizer, model_type)
                    if test_result:
                        # 保存到结构化结果中
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
                    print(f"❌ {model_name} ({metric_name}) 测试失败: {str(e)}")
        
        # 保存结果到CSV文件
        print(f"💾 保存测试结果...")
        
        # 准备CSV数据
        csv_data = []
        columns = ['Model']
        metric_types = ['f1', 'recall', 'pr_auc', 'mae', 'mse', 'final_epoch']
        metric_names = ['precision', 'recall', 'f1', 'pr_auc', 'mse', 'mae']
        
        for metric_type in metric_types:
            for metric_name in metric_names:
                display_type = "final_epoch" if metric_type == 'final_epoch' else f"best_{metric_type}"
                columns.append(f"{display_type}_{metric_name}")
        
        # 添加数据行
        row = [model_name]
        for metric_type in metric_types:
            for metric_name in metric_names:
                value = structured_results[model_name][metric_type][metric_name]
                if value is not None:
                    row.append(f"{value:.6f}")
                else:
                    row.append("N/A")
        csv_data.append(row)
        
        # 保存到CSV文件
        df = pd.DataFrame(csv_data, columns=columns)
        csv_filename = os.path.join(log_dir, f"{model_name}_{model_type}_results.csv")
        df.to_csv(csv_filename, index=False)
        
        # 保存摘要文件
        summary_filename = os.path.join(log_dir, f"{model_name}_{model_type}_summary.txt")
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write(f"模型训练和测试摘要 - {model_name} ({model_type})\n")
            f.write(f"{'='*60}\n")
            f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"GPU设备: {gpu_id}\n\n")
            
            f.write("测试结果:\n")
            f.write("-" * 40 + "\n")
            for metric_type in metric_types:
                metrics = structured_results[model_name][metric_type]
                if metrics['precision'] is not None:
                    display_type = "FINAL" if metric_type == 'final_epoch' else metric_type.upper()
                    f.write(f"{display_type:<12} P={metrics['precision']:<8.4f} R={metrics['recall']:<8.4f} F1={metrics['f1']:<8.4f} PR-AUC={metrics['pr_auc']:<8.4f} MSE={metrics['mse']:<10.6f} MAE={metrics['mae']:<10.6f}\n")
        
        print(f"📄 结果摘要已保存: {summary_filename}")
        print(f"🎉 {model_name} ({model_type}) 训练和测试完成!")
        
        return True
        
    except Exception as e:
        print(f"💥 训练过程中出现异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    args = parse_args()
    
    print(f"🔥 单模型训练器")
    print(f"📋 模型: {args.model}")
    print(f"📋 类型: {args.type}")
    print(f"📋 GPU: {args.gpu}")
    print(f"📋 日志目录: {args.log_dir}")
    print("=" * 50)
    
    # 确保日志目录存在
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 设置环境并训练模型
    success = train_single_model_task(args.model, args.type, args.gpu, args.log_dir)
    
    if success:
        print("🎉 训练成功完成!")
        sys.exit(0)
    else:
        print("❌ 训练失败!")
        sys.exit(1)

if __name__ == "__main__":
    main() 