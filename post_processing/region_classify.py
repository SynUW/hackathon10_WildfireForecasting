#!/usr/bin/env python3
"""
区域分类脚本
功能：
1. 将指定文件夹中的所有TIFF图像相加得到sum.tif
2. 对sum.tif进行4类分类（自然断点法或K-means聚类）
3. 保存分类结果，保持无效值区域为255
"""

import os
import argparse
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from sklearn.cluster import KMeans
from jenkspy import jenks_breaks
import glob
from tqdm import tqdm
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
import threading
import re
warnings.filterwarnings('ignore')


def extract_year_from_filename(filename):
    """
    从文件名中提取年份
    支持格式：FIRMS_yyyy_mm_dd.tif
    Args:
        filename: 文件名
    Returns:
        int: 年份，如果无法提取则返回None
    """
    # 匹配 FIRMS_yyyy_mm_dd.tif 格式
    pattern = r'FIRMS_(\d{4})_\d{2}_\d{2}\.tif'
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    return None


def filter_files_by_year(tiff_files, year_range):
    """
    根据年份范围过滤文件
    Args:
        tiff_files: TIFF文件列表
        year_range: 年份范围 [start_year, end_year]
    Returns:
        list: 过滤后的文件列表
    """
    if year_range is None:
        return tiff_files
    
    start_year, end_year = year_range
    filtered_files = []
    
    for tiff_file in tiff_files:
        filename = os.path.basename(tiff_file)
        year = extract_year_from_filename(filename)
        
        if year is not None and start_year <= year <= end_year:
            filtered_files.append(tiff_file)
    
    return filtered_files


def read_tiff_file(args):
    """读取单个TIFF文件的函数，用于并行处理"""
    tiff_file, invalid_value, binary_mode, reference_shape = args
    try:
        with rasterio.open(tiff_file) as src:
            # 检查尺寸是否匹配
            if src.height != reference_shape[0] or src.width != reference_shape[1]:
                return None
            
            data = src.read(1)  # 读取第一个波段
            
            # 创建有效值掩码
            valid_mask = (data != invalid_value)
            
            if binary_mode:
                # 二值化模式：向量化操作
                binary_data = np.zeros_like(data, dtype=np.float64)
                binary_data[valid_mask & (data > 0)] = 1.0
                return binary_data, valid_mask
            else:
                # 直接累加模式：向量化操作
                float_data = data.astype(np.float64)
                return float_data, valid_mask
                
    except Exception as e:
        return None


def sum_tiff_files_parallel(input_dir, output_path, pattern="*.tif", invalid_value=255, binary_mode=False, n_workers=None, year_range=None):
    """
    并行化版本：将指定目录中的所有TIFF文件相加
    Args:
        input_dir: 输入目录
        output_path: 输出sum.tif路径
        pattern: 文件匹配模式
        invalid_value: 无效值标记
        binary_mode: 是否使用二值化模式
        n_workers: 并行工作进程数
        year_range: 年份范围 [start_year, end_year]
    Returns:
        bool: 是否成功
    """
    try:
        # 查找所有TIFF文件
        search_pattern = os.path.join(input_dir, pattern)
        tiff_files = glob.glob(search_pattern)
        
        if not tiff_files:
            print(f"❌ 在 {input_dir} 中未找到匹配 {pattern} 的文件")
            return False
        
        print(f"📁 找到 {len(tiff_files)} 个TIFF文件")
        
        # 根据年份范围过滤文件
        if year_range is not None:
            tiff_files = filter_files_by_year(tiff_files, year_range)
            print(f"📅 年份范围 {year_range[0]}-{year_range[1]} 过滤后剩余 {len(tiff_files)} 个文件")
            
            if not tiff_files:
                print(f"❌ 在指定年份范围内未找到匹配的文件")
                return False
        
        # 读取第一个文件获取参考信息
        with rasterio.open(tiff_files[0]) as src:
            reference_profile = src.profile.copy()
            height, width = src.height, src.width
        
        # 设置并行工作进程数
        if n_workers is None:
            n_workers = min(cpu_count(), len(tiff_files), 16)  # 限制最大8个进程
        
        print(f"🚀 使用 {n_workers} 个并行进程处理文件")
        
        # 准备并行处理参数
        process_args = [(tiff_file, invalid_value, binary_mode, (height, width)) 
                       for tiff_file in tiff_files]
        
        # 初始化累加数组
        sum_data = np.zeros((height, width), dtype=np.float64)
        valid_count = np.zeros((height, width), dtype=np.int32)
        
        # 并行处理文件
        mode_desc = "二值化累加" if binary_mode else "直接累加"
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # 使用tqdm显示进度
            results = list(tqdm(
                executor.map(read_tiff_file, process_args),
                total=len(tiff_files),
                desc=f"{mode_desc}TIFF文件"
            ))
        
        # 收集结果并累加
        successful_count = 0
        for result in results:
            if result is None:
                continue
            
            # 成功读取的文件
            data, valid_mask = result
            successful_count += 1
            
            # 向量化累加
            sum_data[valid_mask] += data[valid_mask]
            valid_count[valid_mask] += 1
        
        print(f"📊 成功处理 {successful_count}/{len(tiff_files)} 个文件")
        
        # 向量化计算结果
        result_data = np.full((height, width), invalid_value, dtype=np.float32)
        valid_pixels = valid_count > 0
        
        if binary_mode:
            # 二值化模式：直接使用累加值（表示有多少个文件在该像素位置有值>0）
            result_data[valid_pixels] = sum_data[valid_pixels].astype(np.float32)
        else:
            # 直接累加模式：直接使用累加值（保持原始数值范围）
            result_data[valid_pixels] = sum_data[valid_pixels].astype(np.float32)
        
        # 更新profile
        reference_profile.update({
            'dtype': 'float32',
            'nodata': invalid_value,
            'compress': 'lzw'
        })
        
        # 保存sum.tif
        with rasterio.open(output_path, 'w', **reference_profile) as dst:
            dst.write(result_data, 1)
        
        mode_name = "binary" if binary_mode else "direct"
        print(f"✅ 成功生成{mode_name}_sum.tif: {output_path}")
        print(f"📊 有效像素数: {np.sum(valid_pixels)} / {height * width}")
        if binary_mode:
            print(f"📊 二值化累加范围: {result_data[valid_pixels].min():.1f} - {result_data[valid_pixels].max():.1f}")
        else:
            print(f"📊 直接累加范围: {result_data[valid_pixels].min():.3f} - {result_data[valid_pixels].max():.3f}")
        return True
        
    except Exception as e:
        print(f"❌ 生成sum.tif失败: {e}")
        return False


def sum_tiff_files(input_dir, output_path, pattern="*.tif", invalid_value=255, binary_mode=False, n_workers=None, year_range=None):
    """保持向后兼容的函数名"""
    return sum_tiff_files_parallel(input_dir, output_path, pattern, invalid_value, binary_mode, n_workers, year_range)


def classify_with_jenks(data, n_classes=4, invalid_value=255):
    """
    使用自然断点法(Jenks)进行分类 - 向量化优化版本
    Args:
        data: 输入数据
        n_classes: 分类数量
        invalid_value: 无效值标记
    Returns:
        分类结果数组
    """
    # 获取有效值
    valid_mask = (data != invalid_value)
    valid_data = data[valid_mask]
    
    if len(valid_data) == 0:
        print("⚠️  没有有效数据进行分类")
        return np.full_like(data, invalid_value, dtype=np.uint8)
    
    try:
        # 使用Jenks自然断点法
        breaks = jenks_breaks(valid_data, n_classes=n_classes)
        print(f"📊 Jenks断点: {breaks}")
        
        # 创建分类结果
        classified = np.full_like(data, invalid_value, dtype=np.uint8)
        
        # 向量化分类：使用np.digitize进行快速分类
        # 为digitize准备边界数组（去掉第一个和最后一个边界）
        bins = breaks[1:-1] if len(breaks) > 2 else []
        
        if len(bins) > 0:
            # 使用digitize进行向量化分类
            digitized = np.digitize(data, bins)
            # 只对有效值进行分类
            classified[valid_mask] = digitized[valid_mask] + 1  # +1因为digitize从0开始
        else:
            # 特殊情况：只有两个断点，直接分类
            classified[valid_mask] = 1
        
        return classified
        
    except Exception as e:
        print(f"❌ Jenks分类失败: {e}")
        return np.full_like(data, invalid_value, dtype=np.uint8)


def classify_with_kmeans(data, n_classes=4, invalid_value=255, random_state=42):
    """
    使用K-means聚类进行分类 - 向量化优化版本
    Args:
        data: 输入数据
        n_classes: 分类数量
        invalid_value: 无效值标记
        random_state: 随机种子
    Returns:
        分类结果数组
    """
    # 获取有效值
    valid_mask = (data != invalid_value)
    valid_data = data[valid_mask]
    
    if len(valid_data) == 0:
        print("⚠️  没有有效数据进行分类")
        return np.full_like(data, invalid_value, dtype=np.uint8)
    
    try:
        # 重塑数据用于K-means
        valid_data_reshaped = valid_data.reshape(-1, 1)
        
        # 执行K-means聚类 - 使用更快的参数
        kmeans = KMeans(
            n_clusters=n_classes, 
            random_state=random_state, 
            n_init='auto',  # 自动选择初始化次数
            max_iter=100,   # 限制迭代次数
            algorithm='lloyd'  # 使用更快的算法
        )
        labels = kmeans.fit_predict(valid_data_reshaped)
        
        # 获取聚类中心并排序
        centers = kmeans.cluster_centers_.flatten()
        sorted_indices = np.argsort(centers)
        
        print(f"📊 K-means聚类中心: {centers[sorted_indices]}")
        
        # 创建分类结果
        classified = np.full_like(data, invalid_value, dtype=np.uint8)
        
        # 向量化重新分配标签（按聚类中心大小排序）
        label_mapping = np.zeros(n_classes, dtype=np.uint8)
        for i, original_label in enumerate(sorted_indices):
            label_mapping[original_label] = i + 1  # 分类标签从1开始
        
        # 向量化应用标签映射
        classified[valid_mask] = label_mapping[labels]
        
        return classified
        
    except Exception as e:
        print(f"❌ K-means分类失败: {e}")
        return np.full_like(data, invalid_value, dtype=np.uint8)


def save_classification_result(classified_data, output_path, reference_profile, method_name):
    """
    保存分类结果和查找表
    Args:
        classified_data: 分类结果数据
        output_path: 输出路径
        reference_profile: 参考profile
        method_name: 方法名称
    """
    try:
        # 更新profile
        profile = reference_profile.copy()
        profile.update({
            'dtype': 'uint8',
            'nodata': 255,
            'compress': 'lzw'
        })
        
        # 保存TIFF文件
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(classified_data, 1)
        
        print(f"✅ Successfully saved {method_name} classification result: {output_path}")
        
        # 生成查找表
        lookup_table_path = output_path.replace('.tif', '_lookup_table.txt')
        save_lookup_table(classified_data, lookup_table_path, method_name)
        
        # 统计各类别像素数量
        unique, counts = np.unique(classified_data, return_counts=True)
        print(f"📊 {method_name} classification statistics:")
        for val, count in zip(unique, counts):
            if val == 255:
                print(f"   Invalid values: {count} pixels")
            else:
                print(f"   Class {val}: {count} pixels")
        
    except Exception as e:
        print(f"❌ Failed to save classification result: {e}")


def save_lookup_table(classified_data, lookup_table_path, method_name):
    """
    保存查找表
    Args:
        classified_data: 分类结果数据
        lookup_table_path: 查找表输出路径
        method_name: 方法名称
    """
    try:
        height, width = classified_data.shape
        
        # 创建查找表
        lookup_entries = []
        
        # 遍历所有像素
        for row in range(height):
            for col in range(width):
                class_value = classified_data[row, col]
                if class_value != 255:  # 跳过无效值
                    lookup_entries.append(f"{row}_{col} {class_value}")
        
        # 保存到文件
        with open(lookup_table_path, 'w') as f:
            f.write(f"# Lookup table for {method_name} classification\n")
            f.write(f"# Format: row_col class_value\n")
            f.write(f"# Total valid pixels: {len(lookup_entries)}\n")
            f.write(f"# Image dimensions: {width} x {height}\n")
            f.write("#\n")
            
            for entry in lookup_entries:
                f.write(f"{entry}\n")
        
        print(f"✅ Successfully saved lookup table: {lookup_table_path}")
        print(f"📊 Lookup table contains {len(lookup_entries)} valid pixel entries")
        
    except Exception as e:
        print(f"❌ Failed to save lookup table: {e}")


def main():
    parser = argparse.ArgumentParser(description='TIFF图像区域分类工具')
    parser.add_argument('--input-dir', '-i', 
                        default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked_10x_without_qa_masked_clip_min_max_normalized/Firms_Detection_resampled_10x',
                        help='包含TIFF文件的输入目录')
    parser.add_argument('--output-dir', '-o', 
                        default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/7to1_focal_withRegressionLoss_withfirms_baseline',
                        help='输出目录')
    parser.add_argument('--pattern', '-p', default='*.tif',
                       help='文件匹配模式（默认：*.tif）')
    parser.add_argument('--n-classes', '-n', type=int, default=4,
                       help='分类数量（默认：4）')
    parser.add_argument('--method', '-m', choices=['jenks', 'kmeans', 'both'], 
                       default='both',
                       help='分类方法：jenks(自然断点), kmeans(K-means聚类), both(两种方法)')
    parser.add_argument('--invalid-value', type=int, default=255,
                       help='无效值标记（默认：255）')
    parser.add_argument('--sum-only', action='store_true',
                       help='只生成sum.tif，不进行分类')
    parser.add_argument('--binary-only', action='store_true',
                       help='只进行二值化累加和分类')
    parser.add_argument('--direct-only', action='store_true',
                       help='只进行直接累加和分类')
    parser.add_argument('--n-workers', type=int, default=None,
                       help='并行工作进程数（默认：自动检测）')
    parser.add_argument('--year-range', nargs=2, type=int, metavar=('START_YEAR', 'END_YEAR'),
                       help='年份范围，例如 --year-range 2000 2022（包含边界年份）')
    
    args = parser.parse_args()
    
    # 处理年份范围参数
    year_range = None
    if args.year_range:
        start_year, end_year = args.year_range
        if start_year > end_year:
            print(f"❌ 起始年份 {start_year} 不能大于结束年份 {end_year}")
            return
        year_range = [start_year, end_year]
        print(f"📅 指定年份范围: {start_year}-{end_year}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 确定处理模式
    if args.binary_only:
        modes = [True]  # 只处理二值化模式
    elif args.direct_only:
        modes = [False]  # 只处理直接累加模式
    else:
        modes = [False, True]  # 两种模式都处理
    
    for binary_mode in modes:
        mode_name = "binary" if binary_mode else "direct"
        mode_desc = "二值化累加" if binary_mode else "直接累加"
        
        print(f"\n🔄 开始{mode_desc}...")
        
        # 生成sum.tif
        sum_path = os.path.join(args.output_dir, f'{mode_name}_sum.tif')
        
        if not sum_tiff_files(args.input_dir, sum_path, args.pattern, args.invalid_value, binary_mode, args.n_workers, year_range):
            continue
        
        if args.sum_only:
            print(f"✅ 仅生成{mode_name}_sum.tif，完成！")
            continue
        
        # 读取sum.tif进行分类
        print(f"🔄 开始对{mode_desc}结果进行分类...")
        
        try:
            with rasterio.open(sum_path) as src:
                data = src.read(1)
                profile = src.profile.copy()
        except Exception as e:
            print(f"❌ 读取{mode_name}_sum.tif失败: {e}")
            continue
        
        # 执行分类
        if args.method in ['jenks', 'both']:
            print(f"📊 使用Jenks自然断点法对{mode_desc}结果分类...")
            jenks_result = classify_with_jenks(data, args.n_classes, args.invalid_value)
            jenks_path = os.path.join(args.output_dir, f'{mode_name}_classification_jenks_{args.n_classes}classes.tif')
            save_classification_result(jenks_result, jenks_path, profile, f"{mode_desc}-Jenks自然断点")
        
        if args.method in ['kmeans', 'both']:
            print(f"📊 使用K-means聚类对{mode_desc}结果分类...")
            kmeans_result = classify_with_kmeans(data, args.n_classes, args.invalid_value)
            kmeans_path = os.path.join(args.output_dir, f'{mode_name}_classification_kmeans_{args.n_classes}classes.tif')
            save_classification_result(kmeans_result, kmeans_path, profile, f"{mode_desc}-K-means聚类")
    
    print("\n🎉 所有分类完成！")


if __name__ == '__main__':
    main()
