#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的脚本，对已经生成的预测结果应用sigmoid映射
将原始预测值通过sigmoid函数映射到0-1之间
"""

import os
import numpy as np
import rasterio
import argparse
import logging
from tqdm import tqdm
import glob

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def sigmoid(x):
    """Sigmoid函数"""
    return 1 / (1 + np.exp(-x))

def calculate_global_stats(input_dir):
    """计算所有预测文件的全局统计信息"""
    logger.info("计算全局统计信息...")
    
    prediction_files = glob.glob(os.path.join(input_dir, "*_prediction.tif"))
    all_valid_values = []
    
    for input_file in tqdm(prediction_files, desc="收集全局数据"):
        try:
            with rasterio.open(input_file, 'r') as src:
                data = src.read(1)
                valid_mask = data != -9999.0
                if np.any(valid_mask):
                    valid_values = data[valid_mask]
                    all_valid_values.extend(valid_values)
        except Exception as e:
            logger.warning(f"读取文件 {input_file} 时出错: {e}")
    
    if not all_valid_values:
        logger.error("没有找到有效数据")
        return None, None
    
    all_valid_values = np.array(all_valid_values)
    global_mean = np.mean(all_valid_values)
    global_std = np.std(all_valid_values)
    
    logger.info(f"全局均值: {global_mean:.6f}")
    logger.info(f"全局标准差: {global_std:.6f}")
    logger.info(f"有效数据点数量: {len(all_valid_values)}")
    
    return global_mean, global_std

def apply_sigmoid_to_prediction_file(input_file, output_file, global_mean, global_std, flip_y=False, debug=False):
    """对单个预测文件应用z-score标准化和sigmoid映射"""
    try:
        # 读取原始预测文件
        with rasterio.open(input_file, 'r') as src:
            data = src.read(1)  # 读取第一个波段
            profile = src.profile.copy()
            
            if debug:
                logger.info(f"文件: {input_file}")
                logger.info(f"数据形状: {data.shape}")
                logger.info(f"原始数据范围: {data.min():.4f} - {data.max():.4f}")
                logger.info(f"有效数据数量: {np.sum(data != -9999.0)}")
        
        # 如果需要翻转y轴
        if flip_y:
            data = np.flipud(data)  # 上下翻转
            if debug:
                logger.info("已应用y轴翻转")
        
        # 应用z-score标准化和sigmoid映射
        # 只对有效数据（非-9999）进行处理
        valid_mask = data != -9999.0
        if np.any(valid_mask):
            # Z-score标准化: (x - mean) / std
            normalized_data = (data[valid_mask] - global_mean) / global_std
            
            # 应用sigmoid函数
            sigmoid_data = sigmoid(normalized_data)
            
            # 更新数据
            data[valid_mask] = sigmoid_data
            
            if debug:
                logger.info(f"标准化后范围: {normalized_data.min():.4f} - {normalized_data.max():.4f}")
                logger.info(f"Sigmoid后范围: {data[valid_mask].min():.4f} - {data[valid_mask].max():.4f}")
        
        # 保存结果
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(data, 1)
        
        return True
        
    except Exception as e:
        logger.error(f"处理文件 {input_file} 时出错: {e}")
        return False

def process_directory(input_dir, output_dir, flip_y=False, debug=False):
    """处理目录中的所有预测文件"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算全局统计信息
    global_mean, global_std = calculate_global_stats(input_dir)
    if global_mean is None or global_std is None:
        logger.error("无法计算全局统计信息，退出")
        return
    
    # 查找所有预测文件
    prediction_files = glob.glob(os.path.join(input_dir, "*_prediction.tif"))
    
    if not prediction_files:
        logger.warning(f"在目录 {input_dir} 中没有找到预测文件")
        return
    
    logger.info(f"找到 {len(prediction_files)} 个预测文件")
    
    success_count = 0
    failed_count = 0
    
    for input_file in tqdm(prediction_files, desc="应用z-score标准化和sigmoid映射"):
        # 生成输出文件名
        filename = os.path.basename(input_file)
        name_without_ext = os.path.splitext(filename)[0]
        output_filename = f"{name_without_ext}_sigmoid.tif"
        output_file = os.path.join(output_dir, output_filename)
        
        # 应用z-score标准化和sigmoid映射
        if apply_sigmoid_to_prediction_file(input_file, output_file, global_mean, global_std, flip_y, debug):
            success_count += 1
        else:
            failed_count += 1
    
    logger.info(f"处理完成: 成功 {success_count} 个文件, 失败 {failed_count} 个文件")

def main():
    parser = argparse.ArgumentParser(description='对预测结果应用sigmoid映射')
    parser.add_argument('--input-dir', required=True, help='包含预测文件的输入目录')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    parser.add_argument('--flip-y', action='store_true', help='翻转y轴以修复镜像问题')
    parser.add_argument('--debug', action='store_true', help='显示调试信息')
    
    args = parser.parse_args()
    
    logger.info(f"开始处理目录: {args.input_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info("直接对原始预测值应用sigmoid映射")
    if args.flip_y:
        logger.info("启用y轴翻转以修复镜像问题")
    if args.debug:
        logger.info("启用调试模式")
    
    process_directory(args.input_dir, args.output_dir, args.flip_y, args.debug)
    
    logger.info("处理完成！")

if __name__ == "__main__":
    main() 