#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
H5数据集样本计数脚本
统计wildfire数据集中的样本数量和分布情况
"""

import os
import sys
import time
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append('/home/zhengsen/wildfire/forecasting')

from dataload import TimeSeriesDataLoader

# 简化的配置定义
DATA_CONFIG = {
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'firms_min': 0,
    'firms_max': 100,
    'raster_size': 64,
    'enable_position_features': True,
    'enable_future_weather': True,
    'weather_channels': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'positive_ratio': 1.0,
    'pos_neg_ratio': 1.0,
    'resample_each_epoch': False,
    'train_years': list(range(2000, 2021)),
    'val_years': list(range(2021, 2023)),
    'test_years': list(range(2023, 2025)),
    'sampling_ratio': 0.3,
    'enable_dynamic_sampling': False
}

def count_samples_detailed(h5_dir, show_details=True):
    """
    详细统计H5数据集中的样本数量
    
    Args:
        h5_dir (str): H5文件目录路径
        show_details (bool): 是否显示详细信息
    """
    print(f"正在统计H5数据集: {h5_dir}")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # 创建数据加载器
        data_loader = TimeSeriesDataLoader(
            h5_dir=h5_dir,
            positive_ratio=1.0,
            pos_neg_ratio=1.0,
            resample_each_epoch=False
        )
        
        # 获取总样本数
        total_samples = len(data_loader.dataset)
        
        if show_details:
            print(f"数据集配置:")
            print(f"  - 目录: {h5_dir}")
            print(f"  - 总样本数: {total_samples:,}")
            print(f"  - 正样本比例: {data_loader.positive_ratio}")
            print(f"  - 正负样本比例: {data_loader.pos_neg_ratio}")
            print(f"  - 每epoch重采样: {data_loader.resample_each_epoch}")
            print()
            
            # 获取数据集基本信息
            print("数据集基本信息:")
            print(f"  - 过去天数: {data_loader.dataset.past_days}")
            print(f"  - 未来天数: {data_loader.dataset.future_days}")
            print(f"  - 时间范围: {data_loader.dataset.start_date} - {data_loader.dataset.end_date}")
            print()
            
            # 统计年份分布
            print("年份分布统计:")
            year_counts = {}
            for i in range(min(1000, total_samples)):  # 抽样统计前1000个样本
                sample = data_loader.dataset[i]
                if len(sample) >= 3:
                    metadata = sample[2]
                    if isinstance(metadata, dict) and 'start_date' in metadata:
                        year = metadata['start_date'][:4]
                        year_counts[year] = year_counts.get(year, 0) + 1
            
            if year_counts:
                for year, count in sorted(year_counts.items()):
                    print(f"  - {year}: {count} 样本")
            else:
                print("  - 无法获取年份分布信息")
            print()
            
            # 计算数据分割信息
            print("数据分割信息:")
            train_ratio = DATA_CONFIG.get('train_ratio', 0.7)
            val_ratio = DATA_CONFIG.get('val_ratio', 0.15)
            test_ratio = DATA_CONFIG.get('test_ratio', 0.15)
            
            train_samples = int(total_samples * train_ratio)
            val_samples = int(total_samples * val_ratio)
            test_samples = total_samples - train_samples - val_samples
            
            print(f"  - 训练集: {train_samples:,} 样本 ({train_ratio*100:.1f}%)")
            print(f"  - 验证集: {val_samples:,} 样本 ({val_ratio*100:.1f}%)")
            print(f"  - 测试集: {test_samples:,} 样本 ({test_ratio*100:.1f}%)")
            print()
            
            # 计算批次信息
            print("批次信息:")
            batch_sizes = [32, 64, 128, 256]
            for batch_size in batch_sizes:
                train_batches = (train_samples + batch_size - 1) // batch_size
                val_batches = (val_samples + batch_size - 1) // batch_size
                test_batches = (test_samples + batch_size - 1) // batch_size
                print(f"  - Batch Size {batch_size}: 训练{train_batches}批, 验证{val_batches}批, 测试{test_batches}批")
            print()
            
        else:
            print(f"总样本数: {total_samples:,}")
        
        end_time = time.time()
        print(f"统计完成，用时: {end_time - start_time:.2f}秒")
        
        return total_samples
        
    except Exception as e:
        print(f"统计失败: {str(e)}")
        return 0

def compare_configurations():
    """比较不同配置下的样本数量"""
    print("比较不同配置下的样本数量")
    print("="*60)
    
    h5_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/pixel_samples_merged"
    
    configs = [
        {"positive_ratio": 1.0, "pos_neg_ratio": 4.0, "resample": False},
        # {"positive_ratio": 0.3, "pos_neg_ratio": 1.0, "resample": False},
        # {"positive_ratio": 0.1, "pos_neg_ratio": 1.0, "resample": False},
        # {"positive_ratio": 1.0, "pos_neg_ratio": 0.5, "resample": False},
        # {"positive_ratio": 0.3, "pos_neg_ratio": 0.5, "resample": False},
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n配置 {i}: positive_ratio={config['positive_ratio']}, pos_neg_ratio={config['pos_neg_ratio']}")
        print("-" * 40)
        
        try:
            data_loader = TimeSeriesDataLoader(
                h5_dir=h5_dir,
                positive_ratio=config['positive_ratio'],
                pos_neg_ratio=config['pos_neg_ratio'],
                resample_each_epoch=config['resample']
            )
            
            total_samples = len(data_loader.dataset)
            print(f"样本数量: {total_samples:,}")
            
        except Exception as e:
            print(f"配置失败: {str(e)}")

def main():
    """主函数"""
    print("H5数据集样本计数工具")
    print("="*60)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 默认H5目录
    default_h5_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/pixel_samples_merged"
    
    # 检查目录是否存在
    if not os.path.exists(default_h5_dir):
        print(f"警告: 默认目录不存在: {default_h5_dir}")
        print("请确认数据集路径是否正确")
        return
    
    # 基本统计
    print("1. 基本样本统计")
    print("-" * 40)
    total_samples = count_samples_detailed(default_h5_dir, show_details=True)
    
    if total_samples > 0:
        print("\n2. 配置比较")
        print("-" * 40)
        compare_configurations()
    
    print("\n" + "="*60)
    print("统计完成")

if __name__ == "__main__":
    main() 