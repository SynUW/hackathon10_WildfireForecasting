import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Union

class UnifiedModelAdapter:
    """
    统一模型适配器：支持标准模型和10x模型的动态配置
    
    支持的模型类型：
    - 标准Transformer类模型（需要x_enc, x_mark_enc, x_dec, x_mark_dec）
    - 支持通过model_type参数动态切换标准/10x参数配置
    """
    
    def __init__(self, config_or_seq_len=15, pred_len=7, d_model=39, label_len=0, model_type='standard', **kwargs):
        """
        初始化统一适配器
        
        Args:
            config_or_seq_len: Config对象或输入序列长度
            pred_len: 预测序列长度  
            d_model: 特征维度
            label_len: 标签长度（用于某些模型如Autoformer）
            model_type: 模型类型 ('standard' 或 '10x')
        """
        # 如果传入的是Config对象，从中提取参数
        if hasattr(config_or_seq_len, 'seq_len'):
            config = config_or_seq_len
            self.seq_len = int(config.seq_len) if hasattr(config, 'seq_len') else 15
            self.pred_len = int(config.pred_len) if hasattr(config, 'pred_len') else 7
            self.d_model = int(config.d_model) if hasattr(config, 'd_model') else 39
            self.label_len = int(config.label_len) if hasattr(config, 'label_len') else 0
            self.model_type = getattr(config, 'model_type', 'standard')
            # 尝试获取模型名称用于时间特征判断
            self.model_name = getattr(config, 'model_name', None)
        else:
            # 传统方式，直接使用参数
            self.seq_len = int(kwargs.get('seq_len', config_or_seq_len))
            self.pred_len = int(pred_len)
            self.d_model = int(d_model)
            self.label_len = int(label_len)
            self.model_type = model_type
            self.model_name = kwargs.get('model_name', None)
        
    def create_time_marks(self, date_strings: List[str], label_len: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        创建时间标记特征
        
        Args:
            date_strings: 日期字符串列表，格式为YYYYMMDD
            label_len: 标签长度，用于某些模型（如Autoformer）
            
        Returns:
            x_mark_enc: 编码器时间标记 (B, seq_len, time_features)
            x_mark_dec: 解码器时间标记 (B, dec_time_len, time_features)
        """
        batch_size = len(date_strings)
        # 根据模型需求决定时间特征数量
        # 基础4个特征：year, month, day(月中天数), weekday
        # 简化版3个特征：month, day(月中天数), weekday（去掉year以兼容更多模型）
        if hasattr(self, 'model_name') and self.model_name in ['TimeMixer', 'Pyraformer']:
            time_features = 4  # 使用完整的4个特征：year, month, day, weekday
        else:
            time_features = 3  # 使用简化的3个特征：month, day, weekday
        
        # 解码器时间标记的长度：标准的label_len + pred_len
        dec_time_len = label_len + self.pred_len
        
        # 解析基准日期（设定固定小时为12点中午）
        base_dates = []
        for date_str in date_strings:
            try:
                date_str = str(date_str)
                if len(date_str) == 8:
                    year = int(date_str[:4])
                    month = int(date_str[4:6])
                    day = int(date_str[6:8])
                    base_date = datetime(year, month, day, 12, 0, 0)  # 固定设定为12点中午
                else:
                    base_date = datetime(2010, 5, 6, 12, 0, 0)  # 默认日期，12点中午
            except:
                base_date = datetime(2010, 5, 6, 12, 0, 0)  # 12点中午
            base_dates.append(base_date)
        
        # 创建编码器时间标记（过去seq_len天）
        x_mark_enc = torch.zeros(batch_size, self.seq_len, time_features)
        for b in range(batch_size):
            base_date = base_dates[b]
            for t in range(self.seq_len):
                # 计算过去第t天的日期（从-seq_len+1到0）
                days_offset = t - self.seq_len + 1
                current_date = base_date + timedelta(days=days_offset)
                
                # 提取时间特征 - 根据时间特征数量决定内容
                if time_features == 3:
                    # 3个特征：year, month, day(月中天数)
                    year = current_date.year - 2000   # 相对年份（以2000年为基准）
                    month = current_date.month - 1    # 0-11
                    day = current_date.day - 1        # 0-30 (月中天数)
                    
                    x_mark_enc[b, t, :] = torch.tensor([
                        year, month, day
                    ], dtype=torch.long)
                    
                elif time_features == 4:
                    # 4个特征：year, month, day(月中天数), weekday
                    year = current_date.year - 2000   # 相对年份（以2000年为基准）
                    month = current_date.month - 1    # 0-11
                    day = current_date.day - 1        # 0-30 (月中天数)
                    weekday = current_date.weekday()  # 0-6
                    
                    x_mark_enc[b, t, :] = torch.tensor([
                        year, month, day, weekday
                    ], dtype=torch.long)
                    
                else:
                    # 保留原有的复杂特征逻辑（如果有其他模型需要）
                    month = current_date.month - 1
                    weekday = current_date.weekday()
                    day_of_year = current_date.timetuple().tm_yday - 1
                    
                    x_mark_enc[b, t, :3] = torch.tensor([
                        month, weekday, day_of_year
                    ], dtype=torch.long)
        
        # 创建解码器时间标记（label_len + pred_len天）
        x_mark_dec = torch.zeros(batch_size, dec_time_len, time_features)
        for b in range(batch_size):
            base_date = base_dates[b]
            for t in range(dec_time_len):
                # 对于Autoformer：前label_len是历史的，后pred_len是未来的
                # 计算日期偏移：从(-label_len+1)到pred_len
                days_offset = t - label_len + 1
                current_date = base_date + timedelta(days=days_offset)
                
                # 提取时间特征 - 根据时间特征数量决定内容
                if time_features == 3:
                    # 3个特征：year, month, day(月中天数)
                    year = current_date.year - 2000   # 相对年份（以2000年为基准）
                    month = current_date.month - 1    # 0-11
                    day = current_date.day - 1        # 0-30 (月中天数)
                    
                    x_mark_dec[b, t, :] = torch.tensor([
                        year, month, day
                    ], dtype=torch.long)
                    
                elif time_features == 4:
                    # 4个特征：year, month, day(月中天数), weekday
                    year = current_date.year - 2000   # 相对年份（以2000年为基准）
                    month = current_date.month - 1    # 0-11
                    day = current_date.day - 1        # 0-30 (月中天数)
                    weekday = current_date.weekday()  # 0-6
                    
                    x_mark_dec[b, t, :] = torch.tensor([
                        year, month, day, weekday
                    ], dtype=torch.long)
                    
                else:
                    # 保留原有的复杂特征逻辑（如果有其他模型需要）
                    month = current_date.month - 1
                    weekday = current_date.weekday()
                    day_of_year = current_date.timetuple().tm_yday - 1
                    
                    x_mark_dec[b, t, :3] = torch.tensor([
                        month, weekday, day_of_year
                    ], dtype=torch.long)
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

def get_unified_model_configs(model_name=None, model_type='standard'):
    """
    获取统一的模型配置，支持标准和10x模型
    
    Args:
        model_name: 模型名称
        model_type: 模型类型 ('standard' 或 '10x')
    
    Returns:
        dict: 模型配置字典
    """
    
    # 根据模型类型选择基础配置
    if model_type == '10x':
        base_config = {
            'seq_len': 15,
            'pred_len': 7,
            'label_len': 0,  # 默认为0，特定模型会覆盖
            'd_model': 2048,     # 10x: 使用合理的参数设置
            'n_heads': 32,       # 10x: 使用合理的参数设置
            'd_ff': 2048,        # 10x: 使用合理的参数设置
            'e_layers': 4,       # 10x: 使用合理的参数设置
            'd_layers': 4,       # 10x: 使用合理的参数设置
            'dropout': 0.1,
            'activation': 'gelu',
            'output_attention': False,
            'enc_in': 39,  # 输入特征维度，改为39保持一致
            'dec_in': 39,  # 解码器输入特征维度，改为39保持一致  
            'c_out': 39,   # 输出特征维度
            'embed': 'timeF',
            'freq': 'd',   # 日频率
            'factor': 1,
            'moving_avg': 25,  # 用于Autoformer
            'channel_independence': False,
            'use_norm': True,
            'd_state': 32,       # 10x: 16 -> 32 (用于Mamba相关模型，合理设置)
            'd_conv': 4,         # 用于Mamba相关模型
            'expand': 2,         # 用于Mamba相关模型
            'distil': True,      # 用于Informer
        }
    else:  # standard
        base_config = {
            'seq_len': 15,
            'pred_len': 7,
            'label_len': 0,  # 默认为0，特定模型会覆盖
            'd_model': 512,  # 256
            'n_heads': 8,  # 8
            'd_ff': 2048,  # 512
            'e_layers': 2,
            'd_layers': 2,  # 1
            'dropout': 0.1, 
            'activation': 'gelu',
            'output_attention': False,
            'enc_in': 39,  # 输入特征维度，改为39保持一致
            'dec_in': 39,  # 解码器输入特征维度，改为39保持一致  
            'c_out': 39,   # 输出特征维度
            'embed': 'timeF',
            'freq': 'd',   # 日频率
            'factor': 1,
            'moving_avg': 25,  # 用于Autoformer
            'channel_independence': False,
            'use_norm': True,
            'd_state': 16,       # 多于多变量是16，少变量是2
            'd_conv': 4,         # 默认是2
            'expand': 2,         # 默认是1
            'distil': True,      # 用于Informer
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
        # s_mamba的特殊配置
        's_mamba': {
            'd_model': 2048 if model_type == '10x' else 1024,  # 修正：10x使用更大值
            'd_ff': 2048 if model_type == '10x' else 1024,     # 前馈网络维度
            'e_layers': 4 if model_type == '10x' else 2,       # 编码器层数
            'activation': 'gelu',
            'use_norm': True,
            'embed': 'timeF',
            'freq': 'd'
        },
        # 添加缺失配置的模型
        'Nonstationary_Transformer': {
            'p_hidden_dims': [128, 128],
            'p_hidden_layers': 2,
            'label_len': 3  # 设置合适的label_len
        },
        'FEDformer': {
            'label_len': 3,  # 设置合适的label_len
            'moving_avg': 25,  # 分解窗口大小
            'version': 'fourier',  # 默认使用fourier版本
            'mode_select': 'random',  # 模式选择方法
            'modes': 32  # 选择的模式数
        },
        'TemporalFusionTransformer': {
            'data': 'custom',  # 添加数据配置
            'hidden_size': 128,
            'lstm_layers': 1,
            'dropout': 0.1,
            'attn_heads': 4,
            'quantiles': [0.1, 0.5, 0.9]
        },
        'TimeMixer': {
            'seq_len': 30,  # TimeMixer需要更长的序列长度以配合下采样
            'down_sampling_window': 2,
            'down_sampling_layers': 3,
            'down_sampling_method': 'avg',
            'use_future_temporal_feature': True,
            'decomp_method': 'moving_avg',
            'moving_avg_window': 25,
            'channel_independence': False,
            'decomp_kernel': [32],
            'conv_kernel': [24],
            'freq': 'd'  # 使用日频率，配合4个时间特征：year, month, day, weekday
        },
        'SCINet': {
            'hidden_size': 1,
            'num_stacks': 1,
            'num_levels': 3,
            'concat_len': 0,
            'groups': 1,
            'kernel': 5,
            'dropout': 0.5,
            'single_step_output_One': 0,
            'input_len_seg': 0,
            'positionalE': False,
            'modified': True,
            'RIN': False
        },
        'Pyraformer': {
            'embed': 'fixed',  # Pyraformer使用fixed embedding
            'freq': 'd',  # 使用日频率，配合4个时间特征：year, month, day, weekday
            'window_size': [2, 2],  # 减小窗口大小以适应更短的序列
            'inner_size': 3,        # 减小inner_size以适应更短的序列
            'CSCM': 'Bottleneck_Construct',
            'truncate': True,
            'use_tvm': False,
            'decoder': 'FC'
        },
        'ETSformer': {
            'top_k': 5,  # ETSformer需要的top_k参数
            'e_layers': 2,  # 确保编码器层数
            'd_layers': 2   # 确保解码器层数相等
        },
        'TimeXer': {
            'features': 'M',  # TimeXer需要的features参数
            'patch_len': 16,  # patch相关参数
            'stride': 8,      # stride参数
            'enc_in': 38,     # 确保输入维度
            'c_out': 39       # 确保输出维度
        },
        'CrossLinear': {
            'features': 'M',  # CrossLinear需要的features参数
            'patch_len': 16,  # patch相关参数
            'alpha': 0.5,     # CrossLinear的alpha参数
            'beta': 0.5       # CrossLinear的beta参数
        },
        'TimesNet': {
            'top_k': 5,       # TimesNet需要的top_k参数
            'num_kernels': 6  # TimesNet的num_kernels参数
        },
        # 其他模型使用默认的label_len=0
    }
    
    # 应用特定模型配置
    if model_name and model_name in model_specific_configs:
        base_config.update(model_specific_configs[model_name])
    
    return base_config

# 向后兼容的别名
ModelAdapter = UnifiedModelAdapter
get_model_configs = get_unified_model_configs

# 使用示例
if __name__ == "__main__":
    # 测试统一适配器
    batch_size = 4
    seq_len = 30
    pred_len = 7
    d_model = 38
    
    # 模拟数据
    past_data = torch.randn(batch_size, d_model, 365)
    future_data = torch.randn(batch_size, d_model, 30)
    date_strings = ['20240101', '20240102', '20240103', '20240104']
    
    print("🧪 统一模型适配器测试")
    print("=" * 50)
    
    # 测试标准模型适配
    print("\n📋 标准模型配置:")
    standard_config = get_unified_model_configs('Autoformer', 'standard')
    print(f"  d_model: {standard_config['d_model']}")
    print(f"  n_heads: {standard_config['n_heads']}")
    print(f"  d_ff: {standard_config['d_ff']}")
    print(f"  e_layers: {standard_config['e_layers']}")
    
    adapter_std = UnifiedModelAdapter(seq_len=seq_len, pred_len=pred_len, d_model=d_model, label_len=3, model_type='standard')
    x_enc, x_mark_enc, x_dec, x_mark_dec = adapter_std.adapt_inputs(past_data, future_data, date_strings)
    
    print(f"\n📊 标准模型输入:")
    print(f"  x_enc: {x_enc.shape}")
    print(f"  x_mark_enc: {x_mark_enc.shape}")
    print(f"  x_dec: {x_dec.shape}")
    print(f"  x_mark_dec: {x_mark_dec.shape}")
    
    # 测试10x模型适配
    print("\n📋 10x模型配置:")
    config_10x = get_unified_model_configs('Autoformer', '10x')
    print(f"  d_model: {config_10x['d_model']}")
    print(f"  n_heads: {config_10x['n_heads']}")
    print(f"  d_ff: {config_10x['d_ff']}")
    print(f"  e_layers: {config_10x['e_layers']}")
    
    adapter_10x = UnifiedModelAdapter(seq_len=seq_len, pred_len=pred_len, d_model=d_model, label_len=3, model_type='10x')
    x_enc_10x, x_mark_enc_10x, x_dec_10x, x_mark_dec_10x = adapter_10x.adapt_inputs(past_data, future_data, date_strings)
    
    print(f"\n📊 10x模型输入:")
    print(f"  x_enc: {x_enc_10x.shape}")
    print(f"  x_mark_enc: {x_mark_enc_10x.shape}")
    print(f"  x_dec: {x_dec_10x.shape}")
    print(f"  x_mark_dec: {x_mark_dec_10x.shape}")
    
    print(f"\n🔍 时间特征示例: {x_mark_enc[0, 0, :]} (月份, 星期, 年内天数)")
    
    print("\n✅ 统一适配器测试完成！") 