#!/usr/bin/env python3
"""
将prediction.tif文件转换为概率值：
1. 读取prediction.tif文件
2. 除以255进行归一化
3. 应用sigmoid函数映射到[0,1]
4. 保存为prediction_sigmoid.tif
"""

import os
import glob
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F


def sigmoid_normalize(data, invalid_value=-9999):
    """
    对数据进行sigmoid归一化，跳过无效值
    Args:
        data: numpy数组
        invalid_value: 无效值标记（默认-9999）
    Returns:
        归一化后的数组，范围[0,1]，无效值保持为-9999
    """
    # 创建有效值掩码
    valid_mask = (data != invalid_value)
    
    # 初始化输出数组，保持原始无效值
    result = np.full_like(data, invalid_value, dtype=np.float32)
    
    # 只对有效值进行处理
    if np.any(valid_mask):
        valid_data = data[valid_mask]
        
        # 转换为tensor进行sigmoid计算
        tensor_data = torch.from_numpy(valid_data.astype(np.float32))
        # 除以255归一化到[0,1]
        normalized = tensor_data / 255.0
        # 应用sigmoid函数
        sigmoid_data = torch.sigmoid(normalized)
        
        # 将处理后的值写回结果数组
        result[valid_mask] = sigmoid_data.numpy()
    
    return result


def process_prediction_file(input_path, output_path, invalid_value=-9999):
    """
    处理单个prediction.tif文件
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        invalid_value: 无效值标记
    """
    try:
        with rasterio.open(input_path, 'r') as src:
            # 读取数据
            data = src.read()
            profile = src.profile.copy()
            
            # 处理每个波段
            processed_data = np.zeros_like(data, dtype=np.float32)
            for i in range(data.shape[0]):
                processed_data[i] = sigmoid_normalize(data[i], invalid_value)
            
            # 更新profile
            profile.update({
                'dtype': 'float32',
                'nodata': invalid_value,  # 保持无效值标记
                'compress': 'lzw'  # 添加压缩
            })
            
            # 写入新文件
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(processed_data)
                
        return True, None
        
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description='将prediction.tif文件转换为sigmoid概率值')
    parser.add_argument('--input-dir', '-i', 
                        default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/7to1_focal_withRegressionLoss_withfirms_baseline/s_mamba_org_best_f1/visualizations', 
                        help='包含prediction.tif文件的输入目录')
    parser.add_argument('--invalid-value', default=-9999, type=float,
                        help='无效值标记（默认：-9999）')
    parser.add_argument('--output-dir', '-o', 
                       help='输出目录（默认与输入目录相同）')
    parser.add_argument('--pattern', '-p', default='*prediction.tif',
                       help='文件匹配模式（默认：*prediction.tif）')
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='递归搜索子目录')
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir is None:
        output_dir = args.input_dir
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    # 构建搜索模式
    if args.recursive:
        search_pattern = os.path.join(args.input_dir, '**', args.pattern)
    else:
        search_pattern = os.path.join(args.input_dir, args.pattern)
    
    # 查找所有匹配的文件
    input_files = glob.glob(search_pattern, recursive=args.recursive)
    
    if not input_files:
        print(f"❌ 在 {args.input_dir} 中未找到匹配 {args.pattern} 的文件")
        return
    
    print(f"🔍 找到 {len(input_files)} 个prediction.tif文件")
    
    # 处理文件
    success_count = 0
    error_count = 0
    
    for input_file in tqdm(input_files, desc="处理文件"):
        # 生成输出文件名
        filename = os.path.basename(input_file)
        output_filename = filename.replace('prediction.tif', 'prediction_sigmoid.tif')
        
        # 如果输出目录与输入目录不同，需要保持相对路径结构
        if args.output_dir and args.recursive:
            rel_path = os.path.relpath(input_file, args.input_dir)
            rel_dir = os.path.dirname(rel_path)
            if rel_dir:
                output_subdir = os.path.join(output_dir, rel_dir)
                os.makedirs(output_subdir, exist_ok=True)
                output_file = os.path.join(output_subdir, output_filename)
            else:
                output_file = os.path.join(output_dir, output_filename)
        else:
            output_file = os.path.join(output_dir, output_filename)
        
        # 处理文件
        success, error_msg = process_prediction_file(input_file, output_file, args.invalid_value)
        
        if success:
            success_count += 1
            tqdm.write(f"✅ {filename} -> {output_filename}")
        else:
            error_count += 1
            tqdm.write(f"❌ {filename}: {error_msg}")
    
    # 输出统计信息
    print(f"\n📊 处理完成:")
    print(f"   ✅ 成功: {success_count} 个文件")
    print(f"   ❌ 失败: {error_count} 个文件")
    print(f"   📁 输出目录: {output_dir}")


if __name__ == '__main__':
    main()
