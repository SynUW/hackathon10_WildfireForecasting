import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Union

class ModelAdapter:
    """
    模型适配器：将当前的数据格式适配到标准时间序列预测模型
    
    支持的模型类型：
    标准Transformer类模型（需要x_enc, x_mark_enc, x_dec, x_mark_dec）
    """
    
    def __init__(self, config_or_seq_len=15, pred_len=7, d_model=39, label_len=0):
        """
        初始化适配器
        
        Args:
            config_or_seq_len: Config对象或输入序列长度
            pred_len: 预测序列长度  
            d_model: 特征维度
            label_len: 标签长度（用于某些模型如Autoformer）
        """
        # 如果传入的是Config对象，从中提取参数
        if hasattr(config_or_seq_len, 'seq_len'):
            config = config_or_seq_len
            self.seq_len = int(config.seq_len) if hasattr(config, 'seq_len') else 15
            self.pred_len = int(config.pred_len) if hasattr(config, 'pred_len') else 7
            self.d_model = int(config.d_model) if hasattr(config, 'd_model') else 39
            self.label_len = int(config.label_len) if hasattr(config, 'label_len') else 0
        else:
            # 传统方式，直接使用参数
            self.seq_len = int(config_or_seq_len)
            self.pred_len = int(pred_len)
            self.d_model = int(d_model)
            self.label_len = int(label_len)
        
    def create_time_marks(self, date_strings: List[str], label_len: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        创建时间标记特征
        
        Args:
            date_strings: 日期字符串列表，格式为YYYYMMDD
            label_len: 标签长度，用于某些模型（如Autoformer）
            
        Returns:
            x_mark_enc: 编码器时间标记 (B, seq_len, time_features)
            x_mark_dec: 解码器时间标记 (B, label_len + pred_len, time_features)
        """
        batch_size = len(date_strings)
        time_features = 3  # 根据freq='d'，使用3个特征：[月份, 星期, 一年中的天数]
        
        # 解码器时间标记的长度需要考虑label_len
        dec_time_len = label_len + self.pred_len
        
        # 解析基准日期
        base_dates = []
        for date_str in date_strings:
            try:
                date_str = str(date_str)
                if len(date_str) == 8:
                    year = int(date_str[:4])
                    month = int(date_str[4:6])
                    day = int(date_str[6:8])
                    base_date = datetime(year, month, day)
                else:
                    base_date = datetime(2010, 5, 6)  # 默认日期
            except:
                base_date = datetime(2010, 5, 6)
            base_dates.append(base_date)
        
        # 创建编码器时间标记（过去seq_len天）
        x_mark_enc = torch.zeros(batch_size, self.seq_len, time_features)
        for b in range(batch_size):
            base_date = base_dates[b]
            for t in range(self.seq_len):
                # 计算过去第t天的日期（从-seq_len+1到0）
                days_offset = t - self.seq_len + 1
                current_date = base_date + timedelta(days=days_offset)
                
                # 提取时间特征
                month = current_date.month - 1  # 0-11
                weekday = current_date.weekday()  # 0-6
                day_of_year = current_date.timetuple().tm_yday - 1  # 0-365
                
                # 标准化为[0,1]范围，匹配TimeFeatureEmbedding的期望输入
                month_norm = month / 11.0  # 0-1
                weekday_norm = weekday / 6.0  # 0-1  
                day_norm = day_of_year / 365.0  # 0-1
                
                x_mark_enc[b, t, :] = torch.tensor([
                    month_norm, weekday_norm, day_norm
                ])
        
        # 创建解码器时间标记（label_len + pred_len天）
        x_mark_dec = torch.zeros(batch_size, dec_time_len, time_features)
        for b in range(batch_size):
            base_date = base_dates[b]
            for t in range(dec_time_len):
                # 对于Autoformer：前label_len是历史的，后pred_len是未来的
                # 计算日期偏移：从(-label_len+1)到pred_len
                days_offset = t - label_len + 1
                current_date = base_date + timedelta(days=days_offset)
                
                # 提取时间特征
                month = current_date.month - 1  # 0-11
                weekday = current_date.weekday()  # 0-6
                day_of_year = current_date.timetuple().tm_yday - 1  # 0-365
                
                # 标准化为[0,1]范围，匹配TimeFeatureEmbedding的期望输入
                month_norm = month / 11.0  # 0-1
                weekday_norm = weekday / 6.0  # 0-1  
                day_norm = day_of_year / 365.0  # 0-1
                
                x_mark_dec[b, t, :] = torch.tensor([
                    month_norm, weekday_norm, day_norm
                ])
        
        return x_mark_enc, x_mark_dec
    
    def prepare_standard_inputs(self, past_data: torch.Tensor, future_data: torch.Tensor, 
                              date_strings: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        为标准Transformer类模型准备输入
        
        Args:
            past_data: (B, C, T_past) 过去数据
            future_data: (B, C, T_future) 未来数据
            date_strings: 日期字符串列表
            
        Returns:
            x_enc: 编码器输入 (B, seq_len, C)
            x_mark_enc: 编码器时间标记 (B, seq_len, time_features)
            x_dec: 解码器输入 (B, pred_len, C) 
            x_mark_dec: 解码器时间标记 (B, pred_len, time_features)
        """
        batch_size = past_data.shape[0]
        
        # 1. 准备编码器输入
        past_truncated = past_data[:, :, -self.seq_len:]  # 取最后seq_len天
        x_enc = past_truncated.transpose(1, 2)  # (B, seq_len, C)
        
        # 2. 准备解码器输入
        future_truncated = future_data[:, :, :self.pred_len]  # 取前pred_len天
        
        # 对于像Autoformer这样的模型，解码器输入应该与实际的seasonal_init和trend_init对应
        # 但Autoformer会在内部处理这些，所以我们只需要提供一个占位符
        # x_dec实际上不会被直接使用，因为Autoformer内部会创建seasonal_init和trend_init
        x_dec = torch.zeros(batch_size, self.pred_len, past_data.shape[1])
        
        # 3. 创建时间标记 - 考虑label_len
        x_mark_enc, x_mark_dec = self.create_time_marks(date_strings, label_len=self.label_len)
        
        return x_enc, x_mark_enc, x_dec, x_mark_dec
    
    def adapt_inputs(self, past_data: torch.Tensor, future_data: torch.Tensor, 
                    date_strings: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        为标准Transformer类模型适配输入
        
        Args:
            past_data: (B, C, T_past) 过去数据
            future_data: (B, C, T_future) 未来数据
            date_strings: 日期字符串列表
            
        Returns:
            x_enc: 编码器输入 (B, seq_len, C)
            x_mark_enc: 编码器时间标记 (B, seq_len, time_features)
            x_dec: 解码器输入 (B, pred_len, C) 
            x_mark_dec: 解码器时间标记 (B, label_len+pred_len, time_features)
        """
        return self.prepare_standard_inputs(past_data, future_data, date_strings)

def get_model_configs(model_name=None):
    """
    获取不同模型的标准配置
    """
    base_config = {
        'seq_len': 15,
        'pred_len': 7,
        'label_len': 0,  # 默认为0，特定模型会覆盖
        'd_model': 1024,  # 256
        'n_heads': 16,  # 8
        'd_ff': 1024,  # 512
        'e_layers': 2,
        'd_layers': 2,  # 1
        'dropout': 0.1,
        'activation': 'gelu',
        'output_attention': False,
        'enc_in': 39,  # 输入特征维度
        'dec_in': 39,  # 解码器输入特征维度  
        'c_out': 39,   # 输出特征维度
        'embed': 'timeF',
        'freq': 'd',   # 日频率
        'factor': 1,
        'moving_avg': 25,  # 用于Autoformer
        'channel_independence': False,
        'use_norm': True,
        'd_state': 16,  # 用于Mamba相关模型
        'd_conv': 4,    # 用于Mamba相关模型
        'expand': 2,    # 用于Mamba相关模型
        'distil': True, # 用于Informer
    }
    
    # 特定模型的配置
    model_specific_configs = {
        'Autoformer': {'label_len': 3},
        'Autoformer_M': {'label_len': 3},
        'iTransformer': {'class_strategy': 'projection'},
        'iInformer': {'class_strategy': 'projection'},
        'iReformer': {'class_strategy': 'projection'},
        'iFlowformer': {'class_strategy': 'projection'},
        'iFlashformer': {'class_strategy': 'projection'},
        # s_mamba的特殊配置：d_model应该是特征维度39
        's_mamba': {
            'd_model': 39,  # 对于s_mamba，d_model应该是特征维度
            'd_ff': 1024,  # 256
            'e_layers': 2,
            'activation': 'gelu', # relu
            'use_norm': True,
            'embed': 'timeF',
            'freq': 'd'
        },
        # 其他模型使用默认的label_len=0
    }
    
    # 应用特定模型配置
    if model_name and model_name in model_specific_configs:
        base_config.update(model_specific_configs[model_name])
    
    return base_config

# 使用示例
if __name__ == "__main__":
    # 测试适配器
    batch_size = 4
    seq_len = 15
    pred_len = 7
    d_model = 39
    
    # 模拟数据
    past_data = torch.randn(batch_size, d_model, 365)
    future_data = torch.randn(batch_size, d_model, 30)
    date_strings = ['20240101', '20240102', '20240103', '20240104']
    
    # 测试标准模型适配
    adapter_std = ModelAdapter(seq_len=seq_len, pred_len=pred_len, d_model=d_model, label_len=3)
    x_enc, x_mark_enc, x_dec, x_mark_dec = adapter_std.adapt_inputs(past_data, future_data, date_strings)
    
    print("标准模型输入:")
    print(f"  x_enc: {x_enc.shape}")
    print(f"  x_mark_enc: {x_mark_enc.shape}")
    print(f"  x_dec: {x_dec.shape}")
    print(f"  x_mark_dec: {x_mark_dec.shape}")
    print(f"  时间特征示例: {x_mark_enc[0, 0, :]} (月份, 星期, 年内天数)")
    
    print("\n✅ 适配器测试完成！") 