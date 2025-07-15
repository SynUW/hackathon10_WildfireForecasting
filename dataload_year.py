#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于年份数据的时间序列数据加载器 - 无缝替换版本
适配generate_whole_dataset.py生成的年份数据，与原dataload.py接口完全一致

数据格式:
- 文件名: {year}_year_dataset.h5
- 数据集名: {row}_{col} (像素坐标)
- 数据形状: (channels, time_steps) 其中 channels=39, time_steps=365/366
- 通道0: FIRMS数据，用于正负样本判断

特性:
- 正样本: FIRMS >= min_fire_threshold 的某一天
- 负样本: 全局无火日期的随机像素位置
- 支持跨年份数据加载（历史/未来数据）
- 智能采样缓存机制，避免重复计算
- 与原dataload.py完全兼容的接口
"""

import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
import hashlib
import pickle
import random
import time
from datetime import datetime, timedelta
from collections import defaultdict
import glob
import json
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YearTimeSeriesPixelDataset(Dataset):
    """
    基于年份数据的时间序列像素数据集
    
    与原TimeSeriesPixelDataset完全兼容的接口，但使用新的年份数据源
    """
    
    def __init__(self, h5_dir, years=None, return_metadata=True, 
                 positive_ratio=1.0, pos_neg_ratio=1.0, 
                 resample_each_epoch=False, epoch_seed=None, verbose_sampling=True,
                 lookback_seq=365, forecast_hor=7, min_fire_threshold=0.001,
                 cache_dir=None, force_resample=False):
        """
        初始化年份时间序列像素数据集
        
        Args:
            h5_dir: 年份数据H5文件目录
            years: 要加载的年份列表，None表示加载所有年份
            return_metadata: 是否返回元数据（日期、坐标等）
            positive_ratio: 正样本使用比例 (0.0-1.0)
            pos_neg_ratio: 正负样本比例，即负样本数 = 正样本数 × pos_neg_ratio
            resample_each_epoch: 是否在每个epoch重新进行样本抽样
            epoch_seed: 当前epoch的随机种子
            verbose_sampling: 是否显示详细采样信息
            lookback_seq: 历史时间长度（天）
            forecast_hor: 未来时间长度（天）
            min_fire_threshold: FIRMS阈值，>=该值认为是正样本
            cache_dir: 采样缓存目录，None表示使用h5_dir/cache
            force_resample: 是否强制重新采样，忽略缓存
        """
        self.h5_dir = h5_dir
        self.years = years
        self.return_metadata = return_metadata
        self.positive_ratio = positive_ratio
        self.pos_neg_ratio = pos_neg_ratio
        self.resample_each_epoch = resample_each_epoch
        self.epoch_seed = epoch_seed
        self.verbose_sampling = verbose_sampling
        self.lookback_seq = lookback_seq
        self.forecast_hor = forecast_hor
        self.min_fire_threshold = min_fire_threshold
        self.force_resample = force_resample
        
        # 缓存配置
        self.cache_dir = cache_dir if cache_dir else os.path.join(h5_dir, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 获取年份文件列表
        self.year_files = self._get_year_files()
        
        # 构建或加载采样结果
        self.full_sample_index = []  # 存储所有样本的完整索引
        self.sample_index = []  # 当前使用的样本索引
        self.dataset_info = {}  # 存储数据集信息
        
        self._build_or_load_samples()
        
        # 初始的样本比例筛选
        if positive_ratio < 1.0 or pos_neg_ratio != 1.0:
            if resample_each_epoch:
                if self.verbose_sampling:
                    logger.info("启用每epoch重新抽样模式")
                # 保存完整索引并进行初始抽样
                self.full_sample_index = self.sample_index.copy()
                self._apply_sample_ratio_filtering()
            else:
                # 传统模式：一次性抽样
                self._apply_sample_ratio_filtering()
        
        logger.info(f"年份数据集初始化完成，共 {len(self.sample_index)} 个样本")
        logger.info(f"正样本使用比例: {positive_ratio:.2f}, 正负样本比例: 1:{pos_neg_ratio:.2f}")
        if resample_each_epoch:
            logger.info(f"启用每epoch重新抽样，总样本池: {len(self.full_sample_index)} 个样本")
    
    def _get_year_files(self):
        """获取年份H5文件列表"""
        year_files = {}
        
        # 查找所有年份文件
        for filename in os.listdir(self.h5_dir):
            if filename.endswith('_year_dataset.h5'):
                try:
                    year = int(filename.split('_')[0])
                    if self.years is None or year in self.years:
                        year_files[year] = os.path.join(self.h5_dir, filename)
                except ValueError:
                    continue
        
        if not year_files:
            raise ValueError(f"未找到年份数据文件，目录: {self.h5_dir}")
        
        logger.info(f"找到 {len(year_files)} 个年份数据文件: {sorted(year_files.keys())}")
        return year_files
    
    def _get_cache_filename(self):
        """生成缓存文件名"""
        # 根据关键参数生成唯一标识
        cache_params = {
            'h5_dir': self.h5_dir,
            'years': sorted(self.year_files.keys()),
            'lookback_seq': self.lookback_seq,
            'forecast_hor': self.forecast_hor,
            'min_fire_threshold': self.min_fire_threshold,
            'version': '1.0'  # 版本号，用于缓存格式变更
        }
        
        # 生成参数哈希
        param_str = json.dumps(cache_params, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:12]
        
        return os.path.join(self.cache_dir, f"samples_{param_hash}.h5")
    
    def _check_cache_validity(self, cache_file):
        """检查缓存文件是否有效"""
        if self.force_resample:
            return False
        
        if not os.path.exists(cache_file):
            return False
        
        try:
            with h5py.File(cache_file, 'r') as f:
                # 检查缓存参数
                cached_params = json.loads(f.attrs.get('cache_params', '{}'))
                current_params = {
                    'h5_dir': self.h5_dir,
                    'years': sorted(self.year_files.keys()),
                    'lookback_seq': self.lookback_seq,
                    'forecast_hor': self.forecast_hor,
                    'min_fire_threshold': self.min_fire_threshold,
                    'version': '1.0'
                }
                
                if cached_params != current_params:
                    return False
                
                # 检查源数据文件是否有变化
                cached_file_mtimes = json.loads(f.attrs.get('source_file_mtimes', '{}'))
                current_file_mtimes = {}
                
                for year, file_path in self.year_files.items():
                    if os.path.exists(file_path):
                        current_file_mtimes[str(year)] = os.path.getmtime(file_path)
                
                if cached_file_mtimes != current_file_mtimes:
                    return False
                
                return True
                
        except Exception as e:
            logger.warning(f"检查缓存文件失败: {e}")
            return False
    
    def _process_year_for_no_fire_days(self, year_file_info):
        """处理单个年份的无火日期查找"""
        year, file_path = year_file_info
        year_no_fire_days = []
        
        try:
            with h5py.File(file_path, 'r') as f:
                year_days = int(f.attrs.get('total_time_steps', 365))
                
                # 一次性加载所有像素的FIRMS数据
                pixel_names = [name for name in f.keys() if '_' in name]
                
                # 🔍 添加调试信息
                logger.info(f"年份 {year}: 找到 {len(pixel_names)} 个像素数据集（查找无火日期）")
                if len(pixel_names) == 0:
                    logger.warning(f"年份 {year} 文件 {file_path} 没有找到像素数据集")
                    all_keys = list(f.keys())
                    logger.warning(f"文件中的所有键: {all_keys[:10]}...")  # 显示前10个
                    return year_no_fire_days
                
                if not pixel_names:
                    return year_no_fire_days
                
                # 批量加载FIRMS数据
                firms_data_list = []
                for pixel_name in pixel_names:
                    try:
                        pixel_data = f[pixel_name]
                        firms_channel = pixel_data[0, :]  # FIRMS通道数据
                        firms_data_list.append(firms_channel)
                    except Exception as e:
                        logger.warning(f"加载像素 {pixel_name} 失败: {e}")
                        continue
                
                if not firms_data_list:
                    return year_no_fire_days
                
                # 转换为numpy数组进行向量化计算
                # shape: (num_pixels, time_steps)
                firms_array = np.array(firms_data_list)
                
                # 向量化计算每天的最大FIRMS值
                daily_max_firms = np.nanmax(firms_array, axis=0)
                
                # 🔍 添加调试信息
                overall_max_firms = np.nanmax(daily_max_firms)
                overall_min_firms = np.nanmin(daily_max_firms)
                logger.info(f"年份 {year}: 每日最大FIRMS值范围: {overall_min_firms:.2f}-{overall_max_firms:.2f}, 阈值: {self.min_fire_threshold}")
                
                # 找出无火日期（最大FIRMS值小于阈值的天）
                no_fire_days = np.where(daily_max_firms < self.min_fire_threshold)[0]
                
                # 🔍 添加调试信息
                logger.info(f"年份 {year}: 找到 {len(no_fire_days)} 个无火日期（共 {len(daily_max_firms)} 天）")
                
                # 构建结果
                start_date = datetime(year, 1, 1)
                for day_idx in no_fire_days:
                    actual_date = start_date + timedelta(days=int(day_idx))
                    year_no_fire_days.append({
                        'year': year,
                        'day_of_year': int(day_idx),
                        'date': actual_date,
                        'file_path': file_path
                    })
                    
        except Exception as e:
            logger.error(f"处理年份 {year} 时出错: {e}")
        
        # 🔍 添加汇总信息
        logger.info(f"年份 {year}: 总共找到 {len(year_no_fire_days)} 个无火日期")
        
        return year_no_fire_days

    def _find_global_no_fire_days(self):
        """查找全局无火日期 - 优化版本"""
        logger.info("正在查找全局无火日期...")
        
        global_no_fire_days = []
        
        # 使用线程池而不是进程池，避免pickle问题
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # 并行处理所有年份 (限制并行数避免内存过载)
        max_workers = min(4, len(self.year_files), os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._process_year_for_no_fire_days, (year, file_path))
                for year, file_path in self.year_files.items()
            ]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="扫描年份"):
                try:
                    year_results = future.result()
                    global_no_fire_days.extend(year_results)
                except Exception as e:
                    logger.error(f"处理年份时出错: {e}")
        
        logger.info(f"找到 {len(global_no_fire_days)} 个全局无火日期")
        return global_no_fire_days
    
    def _process_year_for_positive_samples(self, year_file_info):
        """处理单个年份的正样本查找"""
        year, file_path = year_file_info
        year_positive_samples = []
        
        try:
            with h5py.File(file_path, 'r') as f:
                year_days = int(f.attrs.get('total_time_steps', 365))
                start_date = datetime(year, 1, 1)
                
                # 获取所有像素数据集名称
                pixel_names = [name for name in f.keys() if '_' in name]
                
                # 🔍 添加调试信息
                logger.info(f"年份 {year}: 找到 {len(pixel_names)} 个像素数据集")
                if len(pixel_names) == 0:
                    logger.warning(f"年份 {year} 文件 {file_path} 没有找到像素数据集")
                    all_keys = list(f.keys())
                    logger.warning(f"文件中的所有键: {all_keys[:10]}...")  # 显示前10个
                
                for pixel_name in pixel_names:
                    try:
                        row, col = map(int, pixel_name.split('_'))
                        pixel_data = f[pixel_name]
                        firms_channel = pixel_data[0, :]  # FIRMS通道数据
                        
                        # 🔍 添加调试信息
                        max_firms = np.nanmax(firms_channel)
                        min_firms = np.nanmin(firms_channel)
                        
                        # 向量化查找超过阈值的天数
                        fire_days = np.where(firms_channel >= self.min_fire_threshold)[0]
                        
                        # 🔍 如果找到火灾天，添加调试信息
                        if len(fire_days) > 0:
                            logger.info(f"像素 {pixel_name}: 找到 {len(fire_days)} 个火灾天，FIRMS范围: {min_firms:.2f}-{max_firms:.2f}")
                        
                        # 为每个火灾天创建正样本
                        for day_idx in fire_days:
                            actual_date = start_date + timedelta(days=int(day_idx))
                            year_positive_samples.append({
                                'year': year,
                                'day_of_year': int(day_idx),
                                'date': actual_date,
                                'pixel_row': row,
                                'pixel_col': col,
                                'firms_value': float(firms_channel[day_idx]),
                                'file_path': file_path
                            })
                            
                    except ValueError:
                        # 跳过非像素数据集
                        continue
                    except Exception as e:
                        logger.warning(f"处理像素 {pixel_name} 时出错: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"处理年份 {year} 时出错: {e}")
        
        # 🔍 添加汇总信息
        logger.info(f"年份 {year}: 总共找到 {len(year_positive_samples)} 个正样本")
        
        return year_positive_samples

    def _find_positive_samples(self):
        """查找正样本 - 优化版本"""
        logger.info("正在查找正样本...")
        
        positive_samples = []
        
        # 使用线程池而不是进程池，避免pickle问题
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # 并行处理所有年份 (限制并行数避免内存过载)
        max_workers = min(4, len(self.year_files), os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._process_year_for_positive_samples, (year, file_path))
                for year, file_path in self.year_files.items()
            ]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="扫描正样本"):
                try:
                    year_results = future.result()
                    positive_samples.extend(year_results)
                except Exception as e:
                    logger.error(f"处理年份时出错: {e}")
        
        logger.info(f"找到 {len(positive_samples)} 个正样本")
        return positive_samples
    
    def _generate_negative_samples(self, global_no_fire_days, num_negative_samples):
        """生成负样本 - 优化版本"""
        logger.info(f"正在生成 {num_negative_samples} 个负样本...")
        
        if not global_no_fire_days:
            logger.warning("没有全局无火日期，无法生成负样本")
            return []
        
        negative_samples = []
        
        # 预先缓存每个年份的像素列表，避免重复读取
        year_pixels_cache = {}
        
        logger.info("预缓存年份像素信息...")
        for year, file_path in tqdm(self.year_files.items(), desc="缓存像素信息"):
            try:
                with h5py.File(file_path, 'r') as f:
                    pixels = []
                    for dataset_name in f.keys():
                        if '_' in dataset_name:
                            try:
                                row, col = map(int, dataset_name.split('_'))
                                pixels.append((row, col))
                            except ValueError:
                                continue
                    
                    if pixels:
                        year_pixels_cache[year] = pixels
                        
            except Exception as e:
                logger.warning(f"缓存年份 {year} 像素信息时出错: {e}")
                continue
        
        if not year_pixels_cache:
            logger.warning("没有可用的像素位置，无法生成负样本")
            return []
        
        # 构建可用的无火日期列表（只包含有像素数据的年份）
        valid_no_fire_days = []
        for day_info in global_no_fire_days:
            if day_info['year'] in year_pixels_cache:
                valid_no_fire_days.append(day_info)
        
        if not valid_no_fire_days:
            logger.warning("没有有效的无火日期，无法生成负样本")
            return []
        
        # 随机采样负样本
        random.seed(42)  # 确保可重复性
        
        logger.info(f"从 {len(valid_no_fire_days)} 个有效无火日期中随机采样...")
        for _ in tqdm(range(num_negative_samples), desc="生成负样本"):
            # 随机选择一个无火日期
            day_info = random.choice(valid_no_fire_days)
            
            # 随机选择一个像素位置
            available_pixels = year_pixels_cache[day_info['year']]
            pixel_row, pixel_col = random.choice(available_pixels)
            
            negative_samples.append({
                'year': day_info['year'],
                'day_of_year': day_info['day_of_year'],
                'date': day_info['date'],
                'pixel_row': pixel_row,
                'pixel_col': pixel_col,
                'firms_value': 0.0,  # 负样本FIRMS值为0
                'file_path': day_info['file_path']
            })
        
        logger.info(f"生成 {len(negative_samples)} 个负样本")
        return negative_samples
    
    def _sample_data_with_cache(self):
        """采样数据，使用缓存机制"""
        cache_file = self._get_cache_filename()
        
        # 检查缓存是否有效
        if self._check_cache_validity(cache_file):
            logger.info(f"🚀 使用缓存文件: {cache_file}")
            return self._load_samples_from_cache(cache_file)
        
        # 缓存无效，重新采样
        logger.info("💾 缓存无效，开始重新采样...")
        
        # 记录开始时间
        start_time = time.time()
        
        # 查找全局无火日期
        logger.info("🔍 步骤1: 查找全局无火日期...")
        step_start = time.time()
        global_no_fire_days = self._find_global_no_fire_days()
        step_duration = time.time() - step_start
        logger.info(f"✅ 完成，耗时: {step_duration:.2f}秒")
        
        # 查找正样本
        logger.info("🔍 步骤2: 查找正样本...")
        step_start = time.time()
        positive_samples = self._find_positive_samples()
        step_duration = time.time() - step_start
        logger.info(f"✅ 完成，耗时: {step_duration:.2f}秒")
        
        if not positive_samples:
            raise ValueError("未找到任何正样本")
        
        # 生成负样本
        logger.info("🔍 步骤3: 生成负样本...")
        step_start = time.time()
        num_negative_samples = len(positive_samples) * 2  # 默认2倍负样本
        negative_samples = self._generate_negative_samples(global_no_fire_days, num_negative_samples)
        step_duration = time.time() - step_start
        logger.info(f"✅ 完成，耗时: {step_duration:.2f}秒")
        
        # 保存到缓存
        logger.info("💾 步骤4: 保存到缓存...")
        step_start = time.time()
        self._save_samples_to_cache(cache_file, positive_samples, negative_samples, global_no_fire_days)
        step_duration = time.time() - step_start
        logger.info(f"✅ 完成，耗时: {step_duration:.2f}秒")
        
        total_duration = time.time() - start_time
        logger.info(f"🎉 采样完成！总耗时: {total_duration:.2f}秒")
        logger.info(f"📊 采样结果: {len(positive_samples)} 正样本, {len(negative_samples)} 负样本")
        
        return positive_samples, negative_samples
    
    def _load_samples_from_cache(self, cache_file):
        """从缓存加载样本 - 优化版本"""
        import time
        start_time = time.time()
        
        try:
            with h5py.File(cache_file, 'r') as f:
                positive_samples = []
                negative_samples = []
                
                # 批量加载正样本数据
                if 'positive_samples' in f:
                    pos_group = f['positive_samples']
                    # 一次性读取所有数组
                    pos_years = pos_group['year'][:]
                    pos_days = pos_group['day_of_year'][:]
                    pos_dates = pos_group['date'][:]
                    pos_rows = pos_group['pixel_row'][:]
                    pos_cols = pos_group['pixel_col'][:]
                    pos_firms = pos_group['firms_value'][:]
                    
                    # 向量化处理 - 预解码字符串
                    decoded_dates = [pos_dates[i].decode() for i in range(len(pos_dates))]
                    for i in range(len(pos_years)):
                        year = int(pos_years[i])
                        positive_samples.append({
                            'year': year,
                            'day_of_year': int(pos_days[i]),
                            'date': datetime.strptime(decoded_dates[i], '%Y-%m-%d'),
                            'pixel_row': int(pos_rows[i]),
                            'pixel_col': int(pos_cols[i]),
                            'firms_value': float(pos_firms[i]),
                            'file_path': self.year_files[year]
                        })
                
                # 批量加载负样本数据
                if 'negative_samples' in f:
                    neg_group = f['negative_samples']
                    # 一次性读取所有数组
                    neg_years = neg_group['year'][:]
                    neg_days = neg_group['day_of_year'][:]
                    neg_dates = neg_group['date'][:]
                    neg_rows = neg_group['pixel_row'][:]
                    neg_cols = neg_group['pixel_col'][:]
                    neg_firms = neg_group['firms_value'][:]
                    
                    # 向量化处理 - 预解码字符串
                    decoded_dates = [neg_dates[i].decode() for i in range(len(neg_dates))]
                    for i in range(len(neg_years)):
                        year = int(neg_years[i])
                        negative_samples.append({
                            'year': year,
                            'day_of_year': int(neg_days[i]),
                            'date': datetime.strptime(decoded_dates[i], '%Y-%m-%d'),
                            'pixel_row': int(neg_rows[i]),
                            'pixel_col': int(neg_cols[i]),
                            'firms_value': float(neg_firms[i]),
                            'file_path': self.year_files[year]
                        })
                
                load_time = time.time() - start_time
                logger.info(f"从缓存加载样本: {len(positive_samples)} 正样本, {len(negative_samples)} 负样本 (耗时: {load_time:.2f}秒)")
                return positive_samples, negative_samples
                
        except Exception as e:
            logger.error(f"从缓存加载样本失败: {e}")
            # 删除无效缓存
            try:
                os.remove(cache_file)
            except:
                pass
            
            # 重新采样
            return self._sample_data_with_cache()
    
    def _save_samples_to_cache(self, cache_file, positive_samples, negative_samples, global_no_fire_days):
        """保存样本到缓存"""
        try:
            with h5py.File(cache_file, 'w') as f:
                # 保存缓存参数
                cache_params = {
                    'h5_dir': self.h5_dir,
                    'years': sorted(self.year_files.keys()),
                    'lookback_seq': self.lookback_seq,
                    'forecast_hor': self.forecast_hor,
                    'min_fire_threshold': self.min_fire_threshold,
                    'version': '1.0'
                }
                f.attrs['cache_params'] = json.dumps(cache_params)
                
                # 保存源文件修改时间
                source_file_mtimes = {}
                for year, file_path in self.year_files.items():
                    if os.path.exists(file_path):
                        source_file_mtimes[str(year)] = os.path.getmtime(file_path)
                f.attrs['source_file_mtimes'] = json.dumps(source_file_mtimes)
                
                # 保存元数据
                f.attrs['total_positive'] = len(positive_samples)
                f.attrs['total_negative'] = len(negative_samples)
                f.attrs['global_no_fire_days'] = len(global_no_fire_days)
                f.attrs['creation_time'] = datetime.now().isoformat()
                
                # 保存正样本
                if positive_samples:
                    pos_group = f.create_group('positive_samples')
                    pos_group.create_dataset('year', data=[s['year'] for s in positive_samples])
                    pos_group.create_dataset('day_of_year', data=[s['day_of_year'] for s in positive_samples])
                    pos_group.create_dataset('date', data=[s['date'].strftime('%Y-%m-%d').encode() for s in positive_samples])
                    pos_group.create_dataset('pixel_row', data=[s['pixel_row'] for s in positive_samples])
                    pos_group.create_dataset('pixel_col', data=[s['pixel_col'] for s in positive_samples])
                    pos_group.create_dataset('firms_value', data=[s['firms_value'] for s in positive_samples])
                
                # 保存负样本
                if negative_samples:
                    neg_group = f.create_group('negative_samples')
                    neg_group.create_dataset('year', data=[s['year'] for s in negative_samples])
                    neg_group.create_dataset('day_of_year', data=[s['day_of_year'] for s in negative_samples])
                    neg_group.create_dataset('date', data=[s['date'].strftime('%Y-%m-%d').encode() for s in negative_samples])
                    neg_group.create_dataset('pixel_row', data=[s['pixel_row'] for s in negative_samples])
                    neg_group.create_dataset('pixel_col', data=[s['pixel_col'] for s in negative_samples])
                    neg_group.create_dataset('firms_value', data=[s['firms_value'] for s in negative_samples])
                
            logger.info(f"样本已保存到缓存: {cache_file}")
            
        except Exception as e:
            logger.error(f"保存样本到缓存失败: {e}")
            # 删除损坏的缓存文件
            try:
                os.remove(cache_file)
            except:
                pass
    
    def _build_or_load_samples(self):
        """构建或加载样本索引"""
        positive_samples, negative_samples = self._sample_data_with_cache()
        
        # 合并样本并构建索引
        all_samples = positive_samples + negative_samples
        
        self.sample_index = []
        self.dataset_info = {}
        
        # 批量处理样本，减少重复计算
        valid_samples = []
        for sample in all_samples:
            # 检查样本是否有效（历史和未来数据充足）
            if self._is_sample_valid(sample):
                valid_samples.append(sample)
        
        # 批量构建样本索引
        for sample in valid_samples:
            pixel_row = sample['pixel_row']
            pixel_col = sample['pixel_col']
            date_obj = sample['date']
            
            sample_metadata = {
                'year': sample['year'],
                'day_of_year': sample['day_of_year'],
                'date': date_obj,
                'date_int': int(date_obj.strftime('%Y%m%d')),
                'pixel_row': pixel_row,
                'pixel_col': pixel_col,
                'firms_value': sample['firms_value'],
                'row': pixel_row,  # 兼容原接口
                'col': pixel_col,  # 兼容原接口
                'sample_type': 'positive' if sample['firms_value'] >= self.min_fire_threshold else 'negative'
            }
            
            self.sample_index.append((sample['file_path'], f"{pixel_row}_{pixel_col}", sample_metadata))
        
        # 构建数据集信息
        for year, file_path in self.year_files.items():
            try:
                with h5py.File(file_path, 'r') as f:
                    self.dataset_info[file_path] = {
                        'year': year,
                        'total_time_steps': int(f.attrs.get('total_time_steps', 365)),
                        'total_channels': int(f.attrs.get('total_channels', 39)),
                        'past_days': self.lookback_seq,
                        'future_days': self.forecast_hor
                    }
            except Exception as e:
                logger.warning(f"读取数据集信息失败: {file_path}, {e}")
    
    def _is_sample_valid(self, sample):
        """检查样本是否有效（历史和未来数据充足）"""
        target_date = sample['date']
        
        # 检查历史数据
        history_start = target_date - timedelta(days=self.lookback_seq - 1)
        if history_start.year < min(self.year_files.keys()):
            return False
        
        # 检查未来数据
        future_end = target_date + timedelta(days=self.forecast_hor - 1)
        if future_end.year > max(self.year_files.keys()):
            return False
        
        return True
    
    def _load_pixel_data_for_date_range(self, pixel_row, pixel_col, start_date, end_date):
        """加载指定像素在指定日期范围内的数据"""
        data_segments = []
        
        current_date = start_date
        while current_date <= end_date:
            year = current_date.year
            
            if year not in self.year_files:
                # 用NaN填充缺失年份
                data_segments.append(np.full((39, 1), np.nan))
                current_date += timedelta(days=1)
                continue
            
            try:
                with h5py.File(self.year_files[year], 'r') as f:
                    dataset_name = f"{pixel_row}_{pixel_col}"
                    
                    if dataset_name not in f:
                        # 像素不存在，用NaN填充
                        data_segments.append(np.full((39, 1), np.nan))
                        current_date += timedelta(days=1)
                        continue
                    
                    pixel_data = f[dataset_name][:]  # shape: (channels, time_steps)
                    
                    # 计算在年份内的索引
                    year_start = datetime(year, 1, 1)
                    day_of_year = (current_date - year_start).days
                    
                    if day_of_year >= pixel_data.shape[1]:
                        # 超出年份范围，用NaN填充
                        data_segments.append(np.full((39, 1), np.nan))
                    else:
                        # 提取当天数据
                        day_data = pixel_data[:, day_of_year:day_of_year+1]
                        data_segments.append(day_data)
                        
            except Exception as e:
                logger.warning(f"加载像素数据失败: {pixel_row}_{pixel_col}, {current_date}, {e}")
                data_segments.append(np.full((39, 1), np.nan))
            
            current_date += timedelta(days=1)
        
        # 合并所有数据段
        if data_segments:
            return np.concatenate(data_segments, axis=1)
        else:
            return np.full((39, 0), np.nan)
    
    def __len__(self):
        """返回样本数量"""
        return len(self.sample_index)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Returns:
            - past_data: (channels, past_time_steps) 
            - future_data: (channels, future_time_steps)
            - metadata (可选): [date_int, row, col]
        """
        file_path, dataset_key, metadata = self.sample_index[idx]
        
        try:
            target_date = metadata['date']
            pixel_row = metadata['pixel_row']
            pixel_col = metadata['pixel_col']
            
            # 计算历史数据范围
            history_start = target_date - timedelta(days=self.lookback_seq - 1)
            history_end = target_date - timedelta(days=1)
            
            # 计算未来数据范围（包含当天）
            future_start = target_date
            future_end = target_date + timedelta(days=self.forecast_hor - 1)
            
            # 加载历史数据
            past_data = self._load_pixel_data_for_date_range(pixel_row, pixel_col, history_start, history_end)
            
            # 加载未来数据
            future_data = self._load_pixel_data_for_date_range(pixel_row, pixel_col, future_start, future_end)
            
            # 检查数据维度并处理不匹配情况
            if past_data.shape[1] != self.lookback_seq:
                # 处理历史数据长度不匹配（可能是闰年/平年导致）
                if past_data.shape[1] < self.lookback_seq:
                    # 数据不足，在开头填充
                    padding_days = self.lookback_seq - past_data.shape[1]
                    padding = np.zeros((39, padding_days))
                    past_data = np.concatenate([padding, past_data], axis=1)
                    logger.debug(f"历史数据长度不足: 期望{self.lookback_seq}, 实际{past_data.shape[1]-padding_days}, 已填充{padding_days}天")
                else:
                    # 数据过多，截取最近的部分
                    past_data = past_data[:, -self.lookback_seq:]
                    logger.debug(f"历史数据过长: 期望{self.lookback_seq}, 实际{past_data.shape[1]}, 已截取最近{self.lookback_seq}天")
            
            if future_data.shape[1] != self.forecast_hor:
                # 处理未来数据长度不匹配
                if future_data.shape[1] < self.forecast_hor:
                    # 数据不足，在末尾填充
                    padding_days = self.forecast_hor - future_data.shape[1]
                    padding = np.zeros((39, padding_days))
                    future_data = np.concatenate([future_data, padding], axis=1)
                    logger.debug(f"未来数据长度不足: 期望{self.forecast_hor}, 实际{future_data.shape[1]-padding_days}, 已填充{padding_days}天")
                else:
                    # 数据过多，截取前面部分
                    future_data = future_data[:, :self.forecast_hor]
                    logger.debug(f"未来数据过长: 期望{self.forecast_hor}, 实际{future_data.shape[1]}, 已截取前{self.forecast_hor}天")
            
            # 处理NaN值
            past_data = np.nan_to_num(past_data, nan=0.0, posinf=0.0, neginf=0.0)
            future_data = np.nan_to_num(future_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 转换为torch tensor
            past_data = torch.from_numpy(past_data).float()
            future_data = torch.from_numpy(future_data).float()
            
            # 最终检查
            if torch.isnan(past_data).any() or torch.isinf(past_data).any():
                past_data = torch.nan_to_num(past_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            if torch.isnan(future_data).any() or torch.isinf(future_data).any():
                future_data = torch.nan_to_num(future_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            if self.return_metadata:
                # 返回简化的metadata格式: [日期, x坐标, y坐标]
                simplified_metadata = [metadata['date_int'], metadata['row'], metadata['col']]
                return past_data, future_data, simplified_metadata
            else:
                return past_data, future_data
                
        except Exception as e:
            logger.debug(f"获取样本失败: {dataset_key}, {e}")
            raise e
    
    def custom_collate_fn(self, batch):
        """
        自定义的collate函数，与原dataload.py完全一致
        """
        if self.return_metadata:
            past_data_list, future_data_list, metadata_list = zip(*batch)
        else:
            past_data_list, future_data_list = zip(*batch)
        
        # 检查所有tensor的形状
        past_shapes = [data.shape for data in past_data_list]
        future_shapes = [data.shape for data in future_data_list]
        
        # 检查past数据形状一致性
        if len(set(past_shapes)) > 1:
            raise ValueError(f"Past数据形状不一致: {set(past_shapes)}")
        
        # 检查future数据形状一致性
        if len(set(future_shapes)) > 1:
            raise ValueError(f"Future数据形状不一致: {set(future_shapes)}")
        
        # 直接堆叠，形状必须一致
        past_batch = torch.stack(past_data_list, dim=0)
        future_batch = torch.stack(future_data_list, dim=0)
        
        if self.return_metadata:
            return past_batch, future_batch, metadata_list
        else:
            return past_batch, future_batch
    
    def resample_for_epoch(self, epoch_seed):
        """
        为新的epoch重新进行样本抽样
        """
        if not self.resample_each_epoch:
            return
        
        if not hasattr(self, 'full_sample_index') or not self.full_sample_index:
            logger.warning("无法重新抽样：没有完整样本索引")
            return
        
        self.epoch_seed = epoch_seed
        
        # 使用完整样本索引重新抽样
        self.sample_index = self.full_sample_index.copy()
        
        # 使用新的随机种子重新抽样
        self._apply_sample_ratio_filtering(seed=epoch_seed)
    
    def _apply_sample_ratio_filtering(self, seed=None):
        """应用正负样本比例筛选"""
        if seed is None:
            seed = 42
        
        if self.verbose_sampling:
            logger.info(f"开始应用正负样本比例筛选... (随机种子: {seed})")
        
        # 分离正样本和负样本
        positive_samples = []
        negative_samples = []
        
        for idx, (file_path, dataset_key, metadata) in enumerate(self.sample_index):
            if metadata['firms_value'] >= self.min_fire_threshold:
                positive_samples.append(idx)
            else:
                negative_samples.append(idx)
        
        positive_count = len(positive_samples)
        negative_count = len(negative_samples)
        
        if self.verbose_sampling:
            logger.info(f"原始样本统计: 正样本 {positive_count} 个, 负样本 {negative_count} 个")
        
        # 计算需要保留的样本数
        retained_positive_count = int(positive_count * self.positive_ratio)
        retained_negative_count = int(retained_positive_count * self.pos_neg_ratio)
        
        # 确保不超过可用数量
        retained_negative_count = min(retained_negative_count, negative_count)
        
        if self.verbose_sampling:
            logger.info(f"计划保留: 正样本 {retained_positive_count} 个, 负样本 {retained_negative_count} 个")
        
        # 随机抽样
        random.seed(seed)
        
        selected_positive_indices = random.sample(positive_samples, retained_positive_count) if retained_positive_count < len(positive_samples) else positive_samples
        selected_negative_indices = random.sample(negative_samples, retained_negative_count) if retained_negative_count < len(negative_samples) else negative_samples
        
        # 合并并重建索引
        selected_indices = selected_positive_indices + selected_negative_indices
        
        new_sample_index = []
        for idx in selected_indices:
            new_sample_index.append(self.sample_index[idx])
        
        self.sample_index = new_sample_index
        
        if self.verbose_sampling:
            logger.info(f"样本筛选完成: 正样本 {len(selected_positive_indices)} 个, 负样本 {len(selected_negative_indices)} 个")
    
    def get_current_sample_stats(self):
        """获取当前样本的统计信息"""
        positive_count = 0
        negative_count = 0
        
        for _, _, metadata in self.sample_index:
            if metadata['firms_value'] >= self.min_fire_threshold:
                positive_count += 1
            else:
                negative_count += 1
        
        return {
            'total_samples': len(self.sample_index),
            'positive_samples': positive_count,
            'negative_samples': negative_count,
            'positive_ratio': positive_count / len(self.sample_index) if len(self.sample_index) > 0 else 0,
            'pos_neg_ratio': negative_count / positive_count if positive_count > 0 else 0
        }
    
    def get_statistics(self):
        """获取数据集统计信息"""
        stats = {
            'total_samples': len(self.sample_index),
            'years': set(),
            'firms_values': set(),
            'sample_types': set(),
            'files': len(self.year_files)
        }
        
        for file_path, dataset_key, metadata in self.sample_index:
            stats['years'].add(metadata['year'])
            stats['firms_values'].add(metadata['firms_value'])
            stats['sample_types'].add(metadata['sample_type'])
        
        # 转换为列表并排序
        stats['years'] = sorted(list(stats['years']))
        stats['firms_values'] = sorted(list(stats['firms_values']))
        stats['sample_types'] = sorted(list(stats['sample_types']))
        
        return stats
    
    def get_dataset_info(self):
        """获取数据集信息"""
        return self.dataset_info


class YearTimeSeriesDataLoader:
    """
    年份时间序列数据加载器 - 与原TimeSeriesDataLoader完全兼容
    """
    
    def __init__(self, h5_dir, positive_ratio=1.0, pos_neg_ratio=1.0, 
                 resample_each_epoch=False, verbose_sampling=True,
                 lookback_seq=365, forecast_hor=7, min_fire_threshold=0.001,
                 cache_dir=None, force_resample=False, **kwargs):
        """
        初始化年份时间序列数据加载器
        
        Args:
            h5_dir: 年份数据H5文件目录
            positive_ratio: 正样本使用比例
            pos_neg_ratio: 正负样本比例
            resample_each_epoch: 是否在每个epoch重新抽样
            verbose_sampling: 是否显示详细采样信息
            lookback_seq: 历史时间长度
            forecast_hor: 未来时间长度
            min_fire_threshold: FIRMS阈值
            cache_dir: 缓存目录
            force_resample: 是否强制重新采样
            **kwargs: 其他参数（为了兼容性）
        """
        self.h5_dir = h5_dir
        self.positive_ratio = positive_ratio
        self.pos_neg_ratio = pos_neg_ratio
        self.resample_each_epoch = resample_each_epoch
        self.verbose_sampling = verbose_sampling
        self.lookback_seq = lookback_seq
        self.forecast_hor = forecast_hor
        self.min_fire_threshold = min_fire_threshold
        self.cache_dir = cache_dir
        self.force_resample = force_resample
        
        # 创建数据集
        self.dataset = YearTimeSeriesPixelDataset(
            h5_dir=h5_dir,
            positive_ratio=positive_ratio,
            pos_neg_ratio=pos_neg_ratio,
            resample_each_epoch=resample_each_epoch,
            verbose_sampling=verbose_sampling,
            lookback_seq=lookback_seq,
            forecast_hor=forecast_hor,
            min_fire_threshold=min_fire_threshold,
            cache_dir=cache_dir,
            force_resample=force_resample
        )
        
        logger.info(f"年份时间序列数据加载器初始化完成")
    
    def get_year_based_split(self, train_years, val_years, test_years, test_full_years=None):
        """
        基于年份进行数据划分 - 与原接口完全一致
        """
        train_indices = []
        val_indices = []
        test_indices = []
        
        for idx, (file_path, dataset_key, metadata) in enumerate(self.dataset.sample_index):
            year = metadata['year']
            
            if year in train_years:
                train_indices.append(idx)
            elif year in val_years:
                val_indices.append(idx)
            elif year in test_years:
                test_indices.append(idx)
        
        if self.verbose_sampling:
            logger.info(f"年份划分结果:")
            logger.info(f"  训练集: {len(train_indices)} 样本 (年份: {train_years})")
            logger.info(f"  验证集: {len(val_indices)} 样本 (年份: {val_years})")
            logger.info(f"  测试集: {len(test_indices)} 样本 (年份: {test_years})")
        
        # 如果需要完整数据测试集
        if test_full_years is not None:
            full_dataset = YearTimeSeriesPixelDataset(
                h5_dir=self.h5_dir,
                years=test_full_years,
                positive_ratio=1.0,
                pos_neg_ratio=999999,  # 使用所有负样本
                resample_each_epoch=False,
                verbose_sampling=self.verbose_sampling,
                lookback_seq=self.lookback_seq,
                forecast_hor=self.forecast_hor,
                min_fire_threshold=self.min_fire_threshold,
                cache_dir=self.cache_dir,
                force_resample=self.force_resample
            )
            
            test_full_indices = list(range(len(full_dataset)))
            logger.info(f"完整数据测试集: {len(test_full_indices)} 样本 (年份: {test_full_years})")
            
            return train_indices, val_indices, test_indices, test_full_indices, full_dataset
        
        return train_indices, val_indices, test_indices


class YearFullDatasetLoader:
    """
    年份完整数据集加载器 - 与原FullDatasetLoader完全兼容
    """
    
    def __init__(self, h5_dir, years=None, return_metadata=True, 
                 lookback_seq=365, forecast_hor=7, min_fire_threshold=0.001,
                 cache_dir=None):
        """
        初始化年份完整数据集加载器
        """
        self.h5_dir = h5_dir
        self.years = years
        self.return_metadata = return_metadata
        self.lookback_seq = lookback_seq
        self.forecast_hor = forecast_hor
        self.min_fire_threshold = min_fire_threshold
        self.cache_dir = cache_dir
        
        # 创建基础数据集，不进行任何采样
        self.dataset = YearTimeSeriesPixelDataset(
            h5_dir=h5_dir,
            years=years,
            return_metadata=return_metadata,
            positive_ratio=1.0,
            pos_neg_ratio=999999,  # 使用所有负样本
            resample_each_epoch=False,
            verbose_sampling=True,
            lookback_seq=lookback_seq,
            forecast_hor=forecast_hor,
            min_fire_threshold=min_fire_threshold,
            cache_dir=cache_dir,
            force_resample=False
        )
        
        logger.info(f"年份完整数据集加载器初始化完成，共 {len(self.dataset)} 个样本")
        
        # 获取统计信息
        stats = self.dataset.get_statistics()
        logger.info(f"数据集统计: {stats}")
    
    def create_dataloader(self, batch_size=32, shuffle=False, num_workers=4, worker_init_fn=None, **dataloader_kwargs):
        """
        创建PyTorch DataLoader
        """
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.dataset.custom_collate_fn,
            worker_init_fn=worker_init_fn,
            **dataloader_kwargs
        )


# 为了完全兼容原接口，创建别名
TimeSeriesDataLoader = YearTimeSeriesDataLoader
TimeSeriesPixelDataset = YearTimeSeriesPixelDataset
FullDatasetLoader = YearFullDatasetLoader


# 测试函数
def test_year_dataloader():
    """测试年份数据加载器 - 优化版本"""
    
    # 测试参数
    h5_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/year_datasets_h5"
    
    try:
        logger.info("🚀 开始测试年份数据加载器...")
        
        # 测试缓存和采样优化
        logger.info("📊 测试1: 缓存和采样优化...")
        start_time = time.time()
        
        # 创建数据加载器
        data_loader = YearTimeSeriesDataLoader(
            h5_dir=h5_dir,
            positive_ratio=0.1,
            pos_neg_ratio=2.0,
            lookback_seq=365,
            forecast_hor=7,
            min_fire_threshold=1.0,
            verbose_sampling=True,
            force_resample=False  # 使用缓存
        )
        
        init_time = time.time() - start_time
        logger.info(f"⏰ 数据加载器初始化耗时: {init_time:.2f}秒")
        
        # 测试年份划分
        logger.info("📊 测试2: 年份划分...")
        train_indices, val_indices, test_indices = data_loader.get_year_based_split(
            train_years=[2021, 2022],
            val_years=[2023],
            test_years=[2024]
        )
        
        logger.info(f"✅ 训练集: {len(train_indices)} 样本")
        logger.info(f"✅ 验证集: {len(val_indices)} 样本")
        logger.info(f"✅ 测试集: {len(test_indices)} 样本")
        
        # 测试数据加载性能
        logger.info("📊 测试3: 数据加载性能...")
        if train_indices:
            sample_idx = train_indices[0]
            
            # 测试单个样本加载时间
            start_time = time.time()
            past_data, future_data, metadata = data_loader.dataset[sample_idx]
            sample_time = time.time() - start_time
            
            logger.info(f"⏰ 单样本加载耗时: {sample_time:.4f}秒")
            logger.info(f"✅ 样本形状: past={past_data.shape}, future={future_data.shape}")
            logger.info(f"✅ 元数据: {metadata}")
            
            # 检查数据质量
            logger.info("📊 测试4: 数据质量检查...")
            logger.info(f"✅ Past数据范围: [{past_data.min():.4f}, {past_data.max():.4f}]")
            logger.info(f"✅ Future数据范围: [{future_data.min():.4f}, {future_data.max():.4f}]")
            logger.info(f"✅ NaN检查: Past={torch.isnan(past_data).sum()}, Future={torch.isnan(future_data).sum()}")
        
        # 测试DataLoader批处理性能
        logger.info("📊 测试5: 批处理性能...")
        from torch.utils.data import Subset
        
        test_sample_size = min(100, len(train_indices))
        train_dataset = Subset(data_loader.dataset, train_indices[:test_sample_size])
        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=data_loader.dataset.custom_collate_fn,
            num_workers=2
        )
        
        start_time = time.time()
        for i, batch in enumerate(train_loader):
            past_batch, future_batch, metadata_batch = batch
            if i == 0:
                logger.info(f"✅ 批次形状: past={past_batch.shape}, future={future_batch.shape}, metadata={len(metadata_batch)}")
            if i >= 4:  # 只测试前5个批次
                break
        
        batch_time = time.time() - start_time
        logger.info(f"⏰ 批处理耗时: {batch_time:.2f}秒 (5个批次)")
        
        # 测试样本统计
        logger.info("📊 测试6: 样本统计...")
        stats = data_loader.dataset.get_current_sample_stats()
        logger.info(f"✅ 样本统计: {stats}")
        
        # 测试第二次初始化（缓存效果）
        logger.info("📊 测试7: 缓存效果...")
        start_time = time.time()
        
        data_loader2 = YearTimeSeriesDataLoader(
            h5_dir=h5_dir,
            positive_ratio=0.1,
            pos_neg_ratio=2.0,
            lookback_seq=365,
            forecast_hor=7,
            min_fire_threshold=1.0,
            verbose_sampling=True,
            force_resample=False
        )
        
        cache_time = time.time() - start_time
        logger.info(f"⏰ 缓存加载耗时: {cache_time:.2f}秒")
        logger.info(f"🚀 加速比: {init_time/cache_time:.1f}x")
        
        logger.info("🎉 所有测试完成！")
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # 运行测试
    test_year_dataloader() 