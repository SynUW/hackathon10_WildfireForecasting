#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的脚本，对已经生成的预测结果应用sigmoid映射
处理流程：
1. 根据模型名称自动构建路径
2. 读取预测文件
3. 应用z-score标准化
4. 应用sigmoid函数映射到0-1范围
5. 使用固定阈值进行分类
6. 在同一目录中生成sigmoid和分类文件
支持自定义阈值或默认阈值
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

def calculate_fixed_thresholds(input_dir, global_mean, global_std, thresholds=None, sample_ratio=1.0, random_seed=42, num_classes=15):
    """使用固定阈值进行分类，支持自定义阈值或自动计算"""
    if thresholds is not None:
        logger.info(f"使用用户指定的固定阈值: {thresholds}")
        return np.array(thresholds)
    
    # 如果没有提供阈值，使用默认的阈值列表
    default_thresholds = [0.7, 0.75, 0.8]
    logger.info(f"使用默认固定阈值: {default_thresholds}")
    return np.array(default_thresholds)

def classify_sigmoid_data(sigmoid_data, thresholds):
    """根据固定阈值将sigmoid数据分类"""
    classified_data = np.zeros_like(sigmoid_data, dtype=np.int32)
    
    # 类别1: < 第一个阈值
    classified_data[sigmoid_data < thresholds[0]] = 1
    
    # 中间类别
    for i in range(1, len(thresholds)):
        classified_data[(sigmoid_data >= thresholds[i-1]) & (sigmoid_data < thresholds[i])] = i + 1
    
    # 最后一个类别: >= 最后一个阈值
    classified_data[sigmoid_data >= thresholds[-1]] = len(thresholds) + 1
    
    return classified_data

def apply_sigmoid_to_prediction_file(input_file, output_file, classification_file, global_mean, global_std, thresholds, flip_y=False, debug=False):
    """对单个预测文件应用z-score标准化、sigmoid映射和固定阈值分类"""
    try:
        # 读取原始预测文件
        with rasterio.open(input_file, 'r') as src:
            data = src.read(1)  # 读取第一个波段
            
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
        else:
            if debug:
                logger.warning(f"文件 {input_file} 中没有有效数据")
        
        # 保存sigmoid结果 - 使用与test_and_visualize_optimized.py完全相同的方式
        with rasterio.open(
            output_file, 'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            nodata=-9999.0
        ) as dst:
            dst.write(data, 1)
        
        if debug:
            logger.info(f"已保存sigmoid文件: {output_file}")
        
        # 生成分类结果
        if thresholds is not None:
            classification_data = np.full_like(data, -9999, dtype=np.int32)
            if np.any(valid_mask):
                classification_data[valid_mask] = classify_sigmoid_data(data[valid_mask], thresholds)
            
            # 保存分类结果 - 使用与test_and_visualize_optimized.py完全相同的方式
            with rasterio.open(
                classification_file, 'w',
                driver='GTiff',
                height=classification_data.shape[0],
                width=classification_data.shape[1],
                count=1,
                dtype=classification_data.dtype,
                nodata=-9999
            ) as dst:
                dst.write(classification_data, 1)
            
            if debug:
                logger.info(f"已保存分类文件: {classification_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"处理文件 {input_file} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_directory(input_dir, flip_y=False, debug=False, num_classes=4, fixed_thresholds=None):
    """处理目录中的所有预测文件"""
    # 使用输入目录作为输出目录
    output_dir = input_dir
    
    logger.info(f"开始处理目录: {input_dir}")
    logger.info(f"输出目录: {output_dir}")
    
    # 计算全局统计信息
    global_mean, global_std = calculate_global_stats(input_dir)
    if global_mean is None or global_std is None:
        logger.error("无法计算全局统计信息，退出")
        return
    
    # 使用固定阈值进行分类
    thresholds = calculate_fixed_thresholds(input_dir, global_mean, global_std, fixed_thresholds, num_classes=num_classes)
    if thresholds is None:
        logger.error("无法获取固定阈值，退出")
        return
    
    # 显示分类信息
    logger.info(f"固定阈值分类信息:")
    logger.info(f"  类别1: < {thresholds[0]:.6f}")
    for i in range(1, len(thresholds)):
        logger.info(f"  类别{i+1}: {thresholds[i-1]:.6f} - {thresholds[i]:.6f}")
    logger.info(f"  类别{len(thresholds)+1}: >= {thresholds[-1]:.6f}")
    
    # 查找所有预测文件
    prediction_files = glob.glob(os.path.join(input_dir, "*_prediction.tif"))
    
    logger.info(f"找到 {len(prediction_files)} 个预测文件")
    if len(prediction_files) == 0:
        logger.warning(f"在目录 {input_dir} 中没有找到预测文件")
        logger.info(f"目录内容:")
        try:
            all_files = os.listdir(input_dir)
            for file in all_files[:10]:  # 只显示前10个文件
                logger.info(f"  - {file}")
            if len(all_files) > 10:
                logger.info(f"  ... 还有 {len(all_files) - 10} 个文件")
        except Exception as e:
            logger.error(f"无法列出目录内容: {e}")
        return
    
    # 显示前几个文件名作为示例
    logger.info("预测文件示例:")
    for i, file in enumerate(prediction_files[:5]):
        logger.info(f"  {i+1}. {os.path.basename(file)}")
    if len(prediction_files) > 5:
        logger.info(f"  ... 还有 {len(prediction_files) - 5} 个文件")
    
    success_count = 0
    failed_count = 0
    
    for input_file in tqdm(prediction_files, desc="应用sigmoid映射和固定阈值分类"):
        # 生成输出文件名
        filename = os.path.basename(input_file)
        name_without_ext = os.path.splitext(filename)[0]
        
        # Sigmoid输出文件
        sigmoid_filename = f"{name_without_ext}_sigmoid.tif"
        sigmoid_file = os.path.join(output_dir, sigmoid_filename)
        
        # 分类输出文件
        classification_filename = f"{name_without_ext}_classified.tif"
        classification_file = os.path.join(output_dir, classification_filename)
        
        # 应用sigmoid映射和分类
        if apply_sigmoid_to_prediction_file(input_file, sigmoid_file, classification_file, 
                                          global_mean, global_std, thresholds, flip_y, debug):
            success_count += 1
        else:
            failed_count += 1
    
    logger.info(f"处理完成: 成功 {success_count} 个文件, 失败 {failed_count} 个文件")
    logger.info(f"每个文件生成两个输出:")
    logger.info(f"  1. *_sigmoid.tif: sigmoid映射后的概率值 (0-1)")
    logger.info(f"  2. *_classified.tif: 固定阈值 {len(thresholds)+1}分类结果 (1-{len(thresholds)+1}, -9999为无效值)")
    
    # 验证输出文件
    if success_count > 0:
        logger.info("验证输出文件:")
        sigmoid_files = glob.glob(os.path.join(output_dir, "*_sigmoid.tif"))
        classified_files = glob.glob(os.path.join(output_dir, "*_classified.tif"))
        logger.info(f"  生成的sigmoid文件: {len(sigmoid_files)} 个")
        logger.info(f"  生成的分类文件: {len(classified_files)} 个")
        
        if len(sigmoid_files) > 0:
            logger.info("Sigmoid文件示例:")
            for i, file in enumerate(sigmoid_files[:3]):
                logger.info(f"  {i+1}. {os.path.basename(file)}")
        
        if len(classified_files) > 0:
            logger.info("分类文件示例:")
            for i, file in enumerate(classified_files[:3]):
                logger.info(f"  {i+1}. {os.path.basename(file)}")


def main():
    parser = argparse.ArgumentParser(description='对预测结果应用sigmoid映射和固定阈值分类')
    parser.add_argument('--model-name', required=True, help='模型名称 (例如: iTransformer_best_recall)')
    parser.add_argument('--base-dir', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/7to1_focal_withRegressionLoss_withfirms_baseline',
                       help='基础目录路径')
    parser.add_argument('--flip-y', action='store_true', help='翻转y轴以修复镜像问题')
    parser.add_argument('--debug', action='store_true', help='显示调试信息')
    parser.add_argument('--num-classes', type=int, default=4, help='分类数量 (默认: 4)')
    parser.add_argument('--fixed-thresholds', nargs='+', type=float, help='自定义固定阈值列表 (例如: 0.1 0.2 0.3 0.4)')
    
    args = parser.parse_args()
    
    # 构建完整的输入目录路径
    input_dir = os.path.join(args.base_dir, args.model_name, "visualizations")
    
    # 验证固定阈值
    fixed_thresholds = None
    if args.fixed_thresholds:
        fixed_thresholds = args.fixed_thresholds
        if len(fixed_thresholds) != args.num_classes - 1:
            logger.warning(f"固定阈值数量({len(fixed_thresholds)})与分类数({args.num_classes})不匹配")
            logger.warning(f"需要 {args.num_classes - 1} 个阈值来定义 {args.num_classes} 个类别")
    
    logger.info(f"模型名称: {args.model_name}")
    logger.info(f"基础目录: {args.base_dir}")
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"使用固定阈值进行{args.num_classes}分类")
    
    if fixed_thresholds:
        logger.info(f"使用自定义固定阈值: {fixed_thresholds}")
    else:
        logger.info("使用默认固定阈值")
        
    if args.flip_y:
        logger.info("启用y轴翻转以修复镜像问题")
    if args.debug:
        logger.info("启用调试模式")
    
    # 检查目录是否存在
    if not os.path.exists(input_dir):
        logger.error(f"目录不存在: {input_dir}")
        logger.error("请检查模型名称是否正确")
        return
    
    process_directory(input_dir, args.flip_y, args.debug, 
                     args.num_classes, fixed_thresholds)
    
    logger.info("处理完成！")

if __name__ == "__main__":
    main() 