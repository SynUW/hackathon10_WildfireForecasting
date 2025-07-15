import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
import re
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence
import glob
import random
import math

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeSeriesPixelDataset(Dataset):
    """
    适配merge_pixel_samples.py生成的H5文件的数据加载器
    
    数据格式:
    - 文件名: {year}_samples.h5 (抽样数据) 或 {year}_full.h5 (完整数据)
    - 数据集名: YYYYMMDD_{past/future}_{firms_value}_row_col (合并后的数据)
    - 数据形状: (total_bands, total_time_steps) 其中 total_time_steps = past_days + future_days
    - 数据类型: float32 (已标准化)
    - 默认配置: past_days=365, future_days=30, total_bands=39
    
    特性: 
    - 默认分离过去和未来数据，返回 (past_data, future_data)
    - 数据已经标准化，无需额外处理
    - 支持抽样数据和完整数据的区分
    - 支持按年份进行训练/验证/测试划分
    - 要求所有数据尺寸一致，不一致则报错
    - 支持每epoch重新抽样的动态样本选择
    """
    
    def __init__(self, h5_dir, years=None, firms_values=None, return_metadata=True, 
                 use_full_data=False, positive_ratio=1.0, pos_neg_ratio=1.0, 
                 resample_each_epoch=False, epoch_seed=None, verbose_sampling=True):
        """
        初始化时间序列像素数据集
        
        Args:
            h5_dir: H5文件目录
            years: 要加载的年份列表，None表示加载所有年份
            firms_values: 要加载的FIRMS值列表，None表示加载所有值
            return_metadata: 是否返回元数据（日期、坐标、FIRMS值等）
            use_full_data: 是否使用完整数据（True: {year}_full.h5, False: {year}_samples.h5）
            positive_ratio: 正样本使用比例，控制选取的正样本数占总正样本数的比例 (0.0-1.0)
            pos_neg_ratio: 正负样本比例，即负样本数 = 正样本数 × pos_neg_ratio
            resample_each_epoch: 是否在每个epoch重新进行样本抽样
            epoch_seed: 当前epoch的随机种子，用于控制重采样
        """
        self.h5_dir = h5_dir
        self.years = years
        self.firms_values = firms_values
        self.return_metadata = return_metadata
        self.use_full_data = use_full_data
        self.positive_ratio = positive_ratio
        self.pos_neg_ratio = pos_neg_ratio
        self.resample_each_epoch = resample_each_epoch
        self.epoch_seed = epoch_seed
        self.verbose_sampling = verbose_sampling
        
        # 获取H5文件列表
        self.h5_files = self._get_h5_files()
        
        # 构建样本索引
        self.full_sample_index = []  # 存储所有样本的完整索引
        self.sample_index = []  # 当前使用的样本索引
        self.dataset_info = {}  # 存储数据集信息
        
        self._build_index()
        
        # 初始的样本比例筛选
        if positive_ratio < 1.0 or pos_neg_ratio != 1.0:
            if resample_each_epoch:
                if self.verbose_sampling:
                    logger.info("启用每epoch重新抽样模式")
                # 如果启用每epoch重新抽样，保存完整索引并进行初始抽样
                self.full_sample_index = self.sample_index.copy()
                self._apply_sample_ratio_filtering()
            else:
                # 传统模式：一次性抽样
                self._apply_sample_ratio_filtering()
        
        logger.info(f"数据集初始化完成，共 {len(self.sample_index)} 个样本")
        logger.info(f"正样本使用比例: {positive_ratio:.2f}, 正负样本比例: 1:{pos_neg_ratio:.2f}")
        if resample_each_epoch:
            logger.info(f"启用每epoch重新抽样，总样本池: {len(self.full_sample_index)} 个样本")
    
    def resample_for_epoch(self, epoch_seed):
        """
        为新的epoch重新进行样本抽样
        
        Args:
            epoch_seed: 当前epoch的随机种子
        """
        if not self.resample_each_epoch:
            return
        
        if not hasattr(self, 'full_sample_index') or not self.full_sample_index:
            logger.warning("无法重新抽样：没有完整样本索引")
            return
        
        self.epoch_seed = epoch_seed
        
        # 临时保存当前样本索引，用完整样本索引替换
        temp_sample_index = self.sample_index
        self.sample_index = self.full_sample_index.copy()
        
        # 使用新的随机种子重新抽样
        self._apply_sample_ratio_filtering(seed=epoch_seed)
        
        # 静默模式，不输出抽样完成信息
        # if self.verbose_sampling:
        #     logger.info(f"Epoch {epoch_seed}: 重新抽样完成，当前样本数: {len(self.sample_index)}")
    
    def get_current_sample_stats(self):
        """获取当前样本的统计信息"""
        positive_count = 0
        negative_count = 0
        
        for _, _, metadata in self.sample_index:
            if metadata['firms_value'] > 0:
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
    
    def custom_collate_fn(self, batch):
        """
        自定义的collate函数，要求所有数据尺寸一致，不一致则报错
        默认分离过去和未来数据
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
    
    def _get_h5_files(self):
        """获取符合条件的H5文件列表"""
        h5_files = []
        
        # 根据use_full_data参数选择文件类型
        file_suffix = '_full.h5' if self.use_full_data else '_samples.h5'
        
        # 支持两种文件名格式:
        # 1. 标准格式: YYYY_samples.h5 或 YYYY_full.h5
        # 2. 扩展格式: YYYY_MMDD_MMDD_full.h5 (用于完整数据)
        if self.use_full_data:
            patterns = [
                r'(\d{4})_full\.h5',           # 标准格式: 2024_full.h5
                r'(\d{4})_\d{4}_\d{4}_full\.h5'  # 扩展格式: 2024_0719_0725_full.h5
            ]
        else:
            patterns = [r'(\d{4})_samples\.h5']  # 抽样数据只支持标准格式
        
        for filename in os.listdir(self.h5_dir):
            if not filename.endswith(file_suffix):
                continue
                
            # 尝试匹配所有模式
            year = None
            for pattern in patterns:
                year_match = re.match(pattern, filename)
                if year_match:
                    year = int(year_match.group(1))
                    break
            
            if year is None:
                continue
            
            # 检查年份过滤条件
            if self.years is not None and year not in self.years:
                continue
                
            h5_path = os.path.join(self.h5_dir, filename)
            h5_files.append((h5_path, year))
        
        data_type = "完整数据" if self.use_full_data else "抽样数据"
        logger.info(f"找到 {len(h5_files)} 个{data_type}文件")
        return h5_files
    
    def _build_index(self):
        """构建样本索引"""
        logger.info("构建样本索引...")
        
        for h5_path, year in tqdm(self.h5_files, desc="扫描H5文件"):
            try:
                with h5py.File(h5_path, 'r') as f:
                    # 获取数据集信息
                    if h5_path not in self.dataset_info:
                        self.dataset_info[h5_path] = {
                            'year': year,
                            'total_bands': f.attrs.get('total_bands', 0),
                            'driver_names': f.attrs.get('driver_names', []),
                            'past_days': f.attrs.get('past_days', 0),
                            'future_days': f.attrs.get('future_days', 0),
                            'data_format': f.attrs.get('data_format', 'unknown')
                        }
                    
                    # 扫描所有数据集
                    for dataset_key in f.keys():
                        # 解析数据集名称: YYYYMMDD_{past/future}_{firms_value}_row_col
                        metadata = self._parse_dataset_key(dataset_key)
                        if metadata is None:
                            continue
                        
                        # 检查FIRMS值过滤条件
                        if (self.firms_values is not None and 
                            metadata['firms_value'] not in self.firms_values):
                            continue
                        
                        # 添加到索引
                        self.sample_index.append((h5_path, dataset_key, metadata))
                        
            except Exception as e:
                logger.error(f"处理文件 {h5_path} 时出错: {str(e)}")
                continue
    
    def _parse_dataset_key(self, dataset_key):
        """
        解析数据集键名
        支持两种格式:
        1. YYYYMMDD_{past/future}_{firms_value}_row_col
        2. YYYY_MM_DD_{past/future}_{firms_value}_row_col (带下划线的日期格式)
        """
        try:
            # 尝试第一种格式: YYYYMMDD_{past/future}_{firms_value}_row_col
            pattern1 = r'(\d{8})_(past|future)_(\d+(?:\.\d+)?)_(\d+)_(\d+)'
            match = re.match(pattern1, dataset_key)
            
            if match:
                date_str, time_type, firms_value_str, row_str, col_str = match.groups()
                return {
                    'date': datetime.strptime(date_str, '%Y%m%d'),
                    'date_int': int(date_str),
                    'time_type': time_type,
                    'firms_value': float(firms_value_str),
                    'row': int(row_str),
                    'col': int(col_str),
                    'pixel_coord': (int(row_str), int(col_str))
                }
            
            # 尝试第二种格式: YYYY_MM_DD_{past/future}_{firms_value}_row_col
            pattern2 = r'(\d{4})_(\d{2})_(\d{2})_(past|future)_(\d+(?:\.\d+)?)_(\d+)_(\d+)'
            match = re.match(pattern2, dataset_key)
            
            if match:
                year_str, month_str, day_str, time_type, firms_value_str, row_str, col_str = match.groups()
                date_str = f"{year_str}{month_str}{day_str}"
            return {
                'date': datetime.strptime(date_str, '%Y%m%d'),
                    'date_int': int(date_str),
                'time_type': time_type,
                'firms_value': float(firms_value_str),
                'row': int(row_str),
                'col': int(col_str),
                'pixel_coord': (int(row_str), int(col_str))
            }
                
            return None
            
        except Exception as e:
            logger.debug(f"解析数据集键名失败: {dataset_key}, 错误: {str(e)}")
            return None
    
    def _is_valid_sample(self, sample_group):
        """验证样本数据是否有效"""
        try:
            # 检查必要的属性
            if not all(attr in sample_group.attrs for attr in ['year', 'driver']):
                return False
            
            # 检查年份是否符合要求
            year = int(sample_group.attrs['year'])
            if self.years and year not in self.years:
                return False
            
            # 检查数据维度
            if 'data' not in sample_group:
                return False
                
            data = sample_group['data'][:]
            if len(data.shape) != 2:  # 应该是 (bands, time_steps)
                return False
                
            # 根据样本类型检查时间步数
            time_steps = data.shape[1]
            sample_id = sample_group.name.split('/')[-1]
            
            # 解析数据集名称格式: YYYYMMDD_{past/future}_{firms_value}_row_col
            parts = sample_id.split('_')
            if len(parts) < 4:
                return False
            
            data_type = parts[1]  # past 或 future
            
            # 检查时间步数
            if data_type == 'past' and time_steps != 365:
                return False
            elif data_type == 'future' and time_steps != 30:
                return False
            elif data_type not in ['past', 'future']:
                return False
            
            # 如果指定了FIRMS值过滤，检查FIRMS值
            if self.firms_values:
                try:
                    firms_value = int(parts[2])
                    if firms_value not in self.firms_values:
                        return False
                except (ValueError, IndexError):
                    return False
                
            return True
            
        except Exception as e:
            logger.error(f"验证样本时出错: {str(e)}")
            return False
    
    def __len__(self):
        return len(self.sample_index)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Returns:
            - past_data: (bands, past_time_steps) 
            - future_data: (bands, future_time_steps)
            - metadata (可选): 包含日期、坐标等信息
        """
        h5_path, dataset_key, metadata = self.sample_index[idx]
        
        try:
            with h5py.File(h5_path, 'r') as f:
                data = f[dataset_key][:]  # shape: (total_bands, time_steps)
                
                # 确保数据是2D格式 (bands, time_steps)
                if data.ndim == 1:
                    data = data[np.newaxis, :]  # 添加波段维度
                elif data.ndim > 2:
                    logger.warning(f"数据维度异常: {data.shape}, 数据集: {dataset_key}")
                    data = data.reshape(data.shape[0], -1)  # 展平为2D
                
                # 检查并处理NaN/Inf值
                if np.isnan(data).any() or np.isinf(data).any():
                    nan_count = np.isnan(data).sum()
                    inf_count = np.isinf(data).sum()
                    # logger.debug(f"数据集 {dataset_key} 包含 {nan_count} 个NaN值和 {inf_count} 个Inf值，已替换为0")
                    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                
                # 转换为torch tensor
                data = torch.from_numpy(data).float()
                
                # 根据数据集类型返回对应的数据
                time_type = metadata['time_type']  # 'past' 或 'future'
                
                if time_type == 'past':
                    # 对于past数据，必须找到对应的future数据
                    future_dataset_key = dataset_key.replace('_past_', '_future_')
                    if future_dataset_key not in f:
                        raise ValueError(f"Past数据 {dataset_key} 缺少对应的Future数据 {future_dataset_key}")
                    
                    future_data = f[future_dataset_key][:]
                    if future_data.ndim == 1:
                        future_data = future_data[np.newaxis, :]
                    elif future_data.ndim > 2:
                        future_data = future_data.reshape(future_data.shape[0], -1)
                    
                    # 检查并处理Future数据的NaN/Inf值
                    if np.isnan(future_data).any() or np.isinf(future_data).any():
                        nan_count = np.isnan(future_data).sum()
                        inf_count = np.isinf(future_data).sum()
                        logger.debug(f"Future数据集 {future_dataset_key} 包含 {nan_count} 个NaN值和 {inf_count} 个Inf值，已替换为0")
                        future_data = np.nan_to_num(future_data, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    future_data = torch.from_numpy(future_data).float()
                    past_data = data
                    
                elif time_type == 'future':
                    # 对于future数据，必须找到对应的past数据
                    past_dataset_key = dataset_key.replace('_future_', '_past_')
                    if past_dataset_key not in f:
                        raise ValueError(f"Future数据 {dataset_key} 缺少对应的Past数据 {past_dataset_key}")
                    
                    past_data = f[past_dataset_key][:]
                    if past_data.ndim == 1:
                        past_data = past_data[np.newaxis, :]
                    elif past_data.ndim > 2:
                        past_data = past_data.reshape(past_data.shape[0], -1)
                    
                    # 检查并处理Past数据的NaN/Inf值
                    if np.isnan(past_data).any() or np.isinf(past_data).any():
                        nan_count = np.isnan(past_data).sum()
                        inf_count = np.isinf(past_data).sum()
                        logger.debug(f"Past数据集 {past_dataset_key} 包含 {nan_count} 个NaN值和 {inf_count} 个Inf值，已替换为0")
                        past_data = np.nan_to_num(past_data, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    past_data = torch.from_numpy(past_data).float()
                    future_data = data
                
                else:
                    raise ValueError(f"未知的时间类型: {time_type}")
                
                # 验证数据形状的合理性 - 从H5文件属性获取期望值
                dataset_info = self.dataset_info.get(h5_path, {})
                expected_past_steps = dataset_info.get('past_days', 365)
                expected_future_steps = dataset_info.get('future_days', 30)
                
                if past_data.shape[1] != expected_past_steps:
                    raise ValueError(f"Past数据时间步数错误: 期望{expected_past_steps}, 实际{past_data.shape[1]}, 数据集: {dataset_key}")
                
                if future_data.shape[1] != expected_future_steps:
                    raise ValueError(f"Future数据时间步数错误: 期望{expected_future_steps}, 实际{future_data.shape[1]}, 数据集: {dataset_key}")
                
                if past_data.shape[0] != future_data.shape[0]:
                    raise ValueError(f"Past和Future数据波段数不匹配: Past={past_data.shape[0]}, Future={future_data.shape[0]}, 数据集: {dataset_key}")
                
                # 最终检查：确保返回的tensor不包含NaN/Inf值
                if torch.isnan(past_data).any() or torch.isinf(past_data).any():
                    # logger.warning(f"Past数据仍包含NaN/Inf值，强制替换为0: {dataset_key}")
                    past_data = torch.nan_to_num(past_data, nan=0.0, posinf=0.0, neginf=0.0)
                
                if torch.isnan(future_data).any() or torch.isinf(future_data).any():
                    # logger.warning(f"Future数据仍包含NaN/Inf值，强制替换为0: {dataset_key}")
                    future_data = torch.nan_to_num(future_data, nan=0.0, posinf=0.0, neginf=0.0)
                
                if self.return_metadata:
                    # 返回简化的metadata格式: [日期, x坐标, y坐标]
                    simplified_metadata = [metadata['date_int'], metadata['row'], metadata['col']]
                    return past_data, future_data, simplified_metadata
                else:
                    return past_data, future_data
                        
        except Exception as e:
            logger.error(f"读取样本失败: {dataset_key}, 错误: {str(e)}")
            raise e  # 重新抛出异常，不要掩盖问题
    
    def get_dataset_info(self):
        """获取数据集信息"""
        return self.dataset_info
    
    def get_sample_by_criteria(self, year=None, firms_value=None, date_range=None):
        """
        根据条件筛选样本
        
        Args:
            year: 年份
            firms_value: FIRMS值
            date_range: 日期范围 (start_date, end_date)
        
        Returns:
            符合条件的样本索引列表
        """
        matching_indices = []
        
        for idx, (h5_path, dataset_key, metadata) in enumerate(self.sample_index):
            # 检查年份
            if year is not None and self.dataset_info[h5_path]['year'] != year:
                continue
            
            # 检查FIRMS值
            if firms_value is not None and metadata['firms_value'] != firms_value:
                continue
            
            # 检查日期范围
            if date_range is not None:
                start_date, end_date = date_range
                if not (start_date <= metadata['date'] <= end_date):
                    continue
            
            matching_indices.append(idx)
        
        return matching_indices
    
    def get_statistics(self):
        """获取数据集统计信息"""
        stats = {
            'total_samples': len(self.sample_index),
            'years': set(),
            'firms_values': set(),
            'time_types': set(),
            'files': len(self.h5_files)
        }
        
        for h5_path, dataset_key, metadata in self.sample_index:
            stats['years'].add(self.dataset_info[h5_path]['year'])
            stats['firms_values'].add(metadata['firms_value'])
            stats['time_types'].add(metadata['time_type'])
        
        # 转换为列表并排序
        stats['years'] = sorted(list(stats['years']))
        stats['firms_values'] = sorted(list(stats['firms_values']))
        stats['time_types'] = sorted(list(stats['time_types']))
        
        return stats

    def _apply_sample_ratio_filtering(self, seed=None, epoch_rotation_strategy=True):
        """
        应用正负样本比例筛选
        
        Args:
            seed: 随机种子，如果为None则使用固定种子42
            epoch_rotation_strategy: 是否使用轮换策略确保数据覆盖
        """
        if seed is None:
            seed = 42
        
        if self.verbose_sampling:
            logger.info(f"开始应用正负样本比例筛选... (随机种子: {seed})")
        
        # 分离正样本和负样本的索引
        positive_samples = []
        negative_samples = []
        
        for idx, (h5_path, dataset_key, metadata) in enumerate(self.sample_index):
            if metadata['firms_value'] > 0:
                positive_samples.append(idx)
            else:
                negative_samples.append(idx)
        
        positive_count = len(positive_samples)
        negative_count = len(negative_samples)
        
        if self.verbose_sampling:
            logger.info(f"原始样本统计: 正样本 {positive_count} 个, 负样本 {negative_count} 个")
        
        # 计算需要保留的正样本数
        retained_positive_count = int(positive_count * self.positive_ratio)
        
        # 计算需要保留的负样本数
        retained_negative_count = int(retained_positive_count * self.pos_neg_ratio)
        
        # 确保不超过可用的负样本数
        retained_negative_count = min(retained_negative_count, negative_count)
        
        if self.verbose_sampling:
            logger.info(f"计划保留: 正样本 {retained_positive_count} 个, 负样本 {retained_negative_count} 个")
        
        # 使用指定的随机种子进行抽样
        random.seed(seed)
        
        if epoch_rotation_strategy and hasattr(self, 'epoch_seed') and self.resample_each_epoch and self.epoch_seed is not None:
            # 轮换策略：确保经过足够epoch后能见到所有数据
            selected_positive_indices = self._get_rotated_samples(positive_samples, retained_positive_count, self.epoch_seed)
            selected_negative_indices = self._get_rotated_samples(negative_samples, retained_negative_count, self.epoch_seed)
        else:
            # 传统随机抽样
            selected_positive_indices = random.sample(positive_samples, retained_positive_count) if retained_positive_count < len(positive_samples) else positive_samples
            selected_negative_indices = random.sample(negative_samples, retained_negative_count) if retained_negative_count < len(negative_samples) else negative_samples
        
        # 合并选中的样本索引
        selected_indices = selected_positive_indices + selected_negative_indices
        
        # 重新构建样本索引
        new_sample_index = []
        for idx in selected_indices:
            new_sample_index.append(self.sample_index[idx])
        
        self.sample_index = new_sample_index
        
        if self.verbose_sampling:
            logger.info(f"样本筛选完成:")
            logger.info(f"  实际保留正样本: {len(selected_positive_indices)} 个")
            logger.info(f"  实际保留负样本: {len(selected_negative_indices)} 个")
            logger.info(f"  总样本数: {len(self.sample_index)} 个")
        
        # 避免除零错误
        if len(selected_positive_indices) > 0:
            ratio = len(selected_negative_indices) / len(selected_positive_indices)
            logger.info(f"  正负样本比例: 1:{ratio:.2f}")
        else:
            logger.info(f"  正负样本比例: 无正样本")
    
    def _get_rotated_samples(self, sample_pool, target_count, epoch_seed):
        """
        轮换抽样策略：确保经过足够epoch后能见到所有数据
        
        Args:
            sample_pool: 样本池（正样本或负样本的索引列表）
            target_count: 目标抽样数量
            epoch_seed: 当前epoch的种子
        
        Returns:
            选中的样本索引列表
        """
        total_samples = len(sample_pool)
        
        if target_count >= total_samples:
            # 如果目标数量大于等于总样本数，返回所有样本
            return sample_pool
        
        if self.positive_ratio >= 1.0:
            # 如果使用全部数据，直接返回所有样本
            return sample_pool
        
        # 计算需要多少个epoch才能覆盖所有数据
        epochs_for_full_coverage = math.ceil(total_samples / target_count)
        
        # 根据当前epoch确定起始位置
        # 使用epoch_seed而不是简单的epoch编号，保持一定的随机性
        random.seed(epoch_seed)
        base_offset = random.randint(0, epochs_for_full_coverage - 1)
        
        # 计算当前epoch应该使用的样本范围
        current_epoch_mod = (epoch_seed - 42) % epochs_for_full_coverage  # 减去基础种子42
        start_idx = (current_epoch_mod * target_count + base_offset * target_count // epochs_for_full_coverage) % total_samples
        
        # 选择样本，使用循环方式确保覆盖
        selected_indices = []
        for i in range(target_count):
            idx = (start_idx + i) % total_samples
            selected_indices.append(sample_pool[idx])
        
        # 为了保持一定的随机性，对选中的样本进行轻微shuffle
        random.seed(epoch_seed)
        random.shuffle(selected_indices)
        
        return selected_indices

    def get_data_coverage_info(self):
        """
        获取数据覆盖信息
        
        Returns:
            dict: 包含数据覆盖统计的字典
        """
        if not self.resample_each_epoch or self.positive_ratio >= 1.0:
            return {
                'strategy': 'full_data' if self.positive_ratio >= 1.0 else 'fixed_subset',
                'coverage_epochs': 1,
                'coverage_ratio': self.positive_ratio
            }
        
        # 分离正样本和负样本
        positive_count = sum(1 for _, _, metadata in self.full_sample_index if metadata['firms_value'] > 0)
        negative_count = len(self.full_sample_index) - positive_count
        
        # 计算覆盖所需的epoch数
        retained_positive_count = int(positive_count * self.positive_ratio)
        retained_negative_count = int(retained_positive_count * self.pos_neg_ratio)
        retained_negative_count = min(retained_negative_count, negative_count)
        
        positive_coverage_epochs = math.ceil(positive_count / retained_positive_count) if retained_positive_count > 0 else 1
        negative_coverage_epochs = math.ceil(negative_count / retained_negative_count) if retained_negative_count > 0 else 1
        
        max_coverage_epochs = max(positive_coverage_epochs, negative_coverage_epochs)
        
        return {
            'strategy': 'rotated_sampling',
            'positive_coverage_epochs': positive_coverage_epochs,
            'negative_coverage_epochs': negative_coverage_epochs,
            'max_coverage_epochs': max_coverage_epochs,
            'total_positive_samples': positive_count,
            'total_negative_samples': negative_count,
            'samples_per_epoch_positive': retained_positive_count,
            'samples_per_epoch_negative': retained_negative_count,
            'coverage_description': f"经过 {max_coverage_epochs} 个epoch后，模型将见到所有训练数据"
        }


class TimeSeriesDataLoader:
    """时间序列数据加载器的便捷包装类"""
    
    def __init__(self, h5_dir, positive_ratio=1.0, pos_neg_ratio=1.0, resample_each_epoch=False, verbose_sampling=True, **dataset_kwargs):
        """
        初始化数据加载器
        
        Args:
            h5_dir: H5文件目录
            positive_ratio: 正样本使用比例，控制选取的正样本数占总正样本数的比例 (0.0-1.0)
            pos_neg_ratio: 正负样本比例，即负样本数 = 正样本数 × pos_neg_ratio
            resample_each_epoch: 是否在每个epoch重新进行样本抽样
            verbose_sampling: 是否打印详细的样本统计信息
            **dataset_kwargs: 传递给TimeSeriesPixelDataset的其他参数
        """
        self.resample_each_epoch = resample_each_epoch
        self.verbose_sampling = verbose_sampling
        self.dataset = TimeSeriesPixelDataset(
            h5_dir, 
            positive_ratio=positive_ratio, 
            pos_neg_ratio=pos_neg_ratio, 
            resample_each_epoch=resample_each_epoch,
            verbose_sampling=verbose_sampling,
            **dataset_kwargs
        )
    
    def resample_for_epoch(self, epoch):
        """
        为新的epoch重新进行样本抽样
        
        Args:
            epoch: 当前epoch编号（从0开始）
        """
        if self.resample_each_epoch:
            # 使用预定义的种子序列，提高可重复性
            predefined_seeds = [42, 123, 456, 789, 321, 654, 987, 147, 258, 369, 
                              741, 852, 963, 159, 267, 378, 489, 591, 612, 723]
            seed = predefined_seeds[epoch % len(predefined_seeds)]
            self.dataset.resample_for_epoch(seed)
            
            # 只在verbose模式下打印样本统计信息
            if self.verbose_sampling:
                stats = self.dataset.get_current_sample_stats()
                logger.info(f"Epoch {epoch} 样本统计: 总计={stats['total_samples']}, "
                           f"正样本={stats['positive_samples']}, 负样本={stats['negative_samples']}, "
                           f"正负比例=1:{stats['pos_neg_ratio']:.2f}")
    
    def get_sample_stats(self):
        """获取当前样本统计信息"""
        return self.dataset.get_current_sample_stats()
    
    def get_data_coverage_info(self):
        """获取数据覆盖信息"""
        return self.dataset.get_data_coverage_info()
    
    def print_data_coverage_info(self):
        """打印数据覆盖信息"""
        coverage_info = self.get_data_coverage_info()
        
        print(f"\n📊 数据覆盖策略分析:")
        print(f"   策略类型: {coverage_info['strategy']}")
        
        if coverage_info['strategy'] == 'rotated_sampling':
            print(f"   正样本总数: {coverage_info['total_positive_samples']:,}")
            print(f"   负样本总数: {coverage_info['total_negative_samples']:,}")
            print(f"   每epoch正样本: {coverage_info['samples_per_epoch_positive']:,}")
            print(f"   每epoch负样本: {coverage_info['samples_per_epoch_negative']:,}")
            print(f"   正样本完全覆盖需要: {coverage_info['positive_coverage_epochs']} epochs")
            print(f"   负样本完全覆盖需要: {coverage_info['negative_coverage_epochs']} epochs")
            print(f"   🎯 {coverage_info['coverage_description']}")
        elif coverage_info['strategy'] == 'full_data':
            print(f"   ✅ 使用全部训练数据，无需轮换")
        else:
            print(f"   ⚠️  使用固定子集 ({coverage_info['coverage_ratio']:.1%})，建议启用动态抽样")
    
    def create_dataloader(self, batch_size=32, shuffle=True, num_workers=4, worker_init_fn=None, **dataloader_kwargs):
        """
        创建PyTorch DataLoader
        
        Args:
            batch_size: 批次大小
            shuffle: 是否打乱数据
            num_workers: 工作进程数
            worker_init_fn: worker初始化函数，确保多进程可重复性
            **dataloader_kwargs: 传递给DataLoader的其他参数
        
        Returns:
            torch.utils.data.DataLoader
        """
        # 使用自定义的collate函数来处理尺寸不一致的问题
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.dataset.custom_collate_fn,
            worker_init_fn=worker_init_fn,
            **dataloader_kwargs
        )
    
    def get_year_based_split(self, train_years, val_years, test_years, test_full_years=None):
        """
        基于年份创建训练、验证和测试数据集分割
        
        Args:
            train_years: 训练年份列表（使用抽样数据）
            val_years: 验证年份列表（使用抽样数据）
            test_years: 测试年份列表（使用抽样数据）
            test_full_years: 完整数据测试年份列表（使用完整数据，可选）
        
        Returns:
            如果test_full_years为None: (train_indices, val_indices, test_indices)
            如果test_full_years不为None: (train_indices, val_indices, test_indices, test_full_indices)
        """
        train_indices = []
        val_indices = []
        test_indices = []
        
        for idx, (h5_path, dataset_key, metadata) in enumerate(self.dataset.sample_index):
            year = self.dataset.dataset_info[h5_path]['year']
            
            if year in train_years:
                train_indices.append(idx)
            elif year in val_years:
                val_indices.append(idx)
            elif year in test_years:
                test_indices.append(idx)
        
        if self.verbose_sampling:
            data_type = "完整数据" if self.dataset.use_full_data else "抽样数据"
            logger.info(f"年份划分结果 ({data_type}):")
            logger.info(f"  训练集: {len(train_indices)} 样本 (年份: {train_years})")
            logger.info(f"  验证集: {len(val_indices)} 样本 (年份: {val_years})")
            logger.info(f"  测试集: {len(test_indices)} 样本 (年份: {test_years})")
        
        # 如果指定了完整数据测试年份，创建完整数据测试集
        if test_full_years is not None:
            # 创建完整数据加载器
            full_dataset = TimeSeriesPixelDataset(
                h5_dir=self.dataset.h5_dir,
                years=test_full_years,
                firms_values=self.dataset.firms_values,
                return_metadata=self.dataset.return_metadata,
                use_full_data=True
            )
            
            # 获取完整数据的所有索引
            test_full_indices = list(range(len(full_dataset)))
            
            logger.info(f"完整数据测试集: {len(test_full_indices)} 样本 (年份: {test_full_years})")
            
            return train_indices, val_indices, test_indices, test_full_indices, full_dataset
        
        return train_indices, val_indices, test_indices


class FullDatasetLoader:
    """
    专门用于完整数据集测试的数据加载器
    不进行任何采样，加载所有可用数据
    """
    
    def __init__(self, h5_dir, years=None, return_metadata=True):
        """
        初始化完整数据集加载器
        
        Args:
            h5_dir: H5文件目录
            years: 要加载的年份列表，None表示加载所有年份
            return_metadata: 是否返回元数据（日期、坐标、FIRMS值等）
        """
        self.h5_dir = h5_dir
        self.years = years
        self.return_metadata = return_metadata
        
        # 创建基础数据集，不进行任何采样
        self.dataset = TimeSeriesPixelDataset(
            h5_dir=h5_dir,
            years=years,
            firms_values=None,  # 加载所有FIRMS值
            return_metadata=return_metadata,
            use_full_data=True,  # 使用完整数据文件
            positive_ratio=1.0,  # 使用所有正样本
            pos_neg_ratio=999999  # 使用所有负样本（实际上不限制）
        )
        
        logger.info(f"完整数据集加载器初始化完成，共 {len(self.dataset)} 个样本")
        
        # 获取数据集统计信息
        stats = self.dataset.get_statistics()
        logger.info(f"数据集统计: {stats}")
    
    def create_dataloader(self, batch_size=32, shuffle=False, num_workers=4, worker_init_fn=None, **dataloader_kwargs):
        """
        创建PyTorch DataLoader
        
        Args:
            batch_size: 批次大小
            shuffle: 是否打乱数据（完整数据集测试通常不打乱）
            num_workers: 工作进程数
            worker_init_fn: worker初始化函数，确保多进程可重复性
            **dataloader_kwargs: 传递给DataLoader的其他参数
        
        Returns:
            torch.utils.data.DataLoader
        """
        from torch.utils.data import DataLoader
        
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.dataset.custom_collate_fn,
            worker_init_fn=worker_init_fn,
            **dataloader_kwargs
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def get_statistics(self):
        """获取数据集统计信息"""
        return self.dataset.get_statistics()


if __name__ == '__main__':
    
    h5_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/pixel_samples_merged'
    
    # 1. 创建抽样数据加载器
    data_loader = TimeSeriesDataLoader(h5_dir=h5_dir)
    
    # 2. 基于年份划分数据集
    train_years = [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 
                   2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
    val_years = [2021, 2022]
    test_years = [2023, 2024]
    test_full_years = []  # 使用完整数据进行最终测试
    
    result = data_loader.get_year_based_split(
        train_years, val_years, test_years, test_full_years
    )
    
    if len(result) == 5:
        train_indices, val_indices, test_indices, test_full_indices, full_dataset = result
    else:
        train_indices, val_indices, test_indices = result
        test_full_indices = []
        full_dataset = None
    
    # 4. 创建PyTorch DataLoader
    from torch.utils.data import Subset
    
    # 训练集
    train_dataset = Subset(data_loader.dataset, train_indices)
    train_dataloader = data_loader.create_dataloader(
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    # 验证集
    val_dataset = Subset(data_loader.dataset, val_indices)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=data_loader.dataset.custom_collate_fn
    )
    
    # 测试集
    test_dataset = Subset(data_loader.dataset, test_indices)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=data_loader.dataset.custom_collate_fn
    )
    
    if full_dataset is not None:
        test_full_dataset = Subset(full_dataset, test_full_indices)
        test_full_dataloader = DataLoader(
            test_full_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            collate_fn=full_dataset.custom_collate_fn
        )
        
        
