import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Union

class UnifiedModelAdapter:
    """
    Unified model adapter: supports configuration for standard models
    
    Supported model types:
    - Standard Transformer-like models (require x_enc, x_mark_enc, x_dec, x_mark_dec)
    """
    
    def __init__(self, config_or_seq_len=15, pred_len=7, d_model=39, label_len=0, model_type='standard', **kwargs):
        """
        Initialize the unified adapter
        
        Args:
            config_or_seq_len: Config object or input sequence length
            pred_len: Prediction sequence length  
            d_model: Feature dimension
            label_len: Label length (for some models like Autoformer)
            model_type: Model type (default: 'standard')
        """
        # If a Config object is passed, extract parameters from it
        if hasattr(config_or_seq_len, 'seq_len'):
            config = config_or_seq_len
            self.seq_len = int(config.seq_len) if hasattr(config, 'seq_len') else 15
            self.pred_len = int(config.pred_len) if hasattr(config, 'pred_len') else 7
            self.d_model = int(config.d_model) if hasattr(config, 'd_model') else 39
            self.label_len = int(config.label_len) if hasattr(config, 'label_len') else 0
            self.model_type = getattr(config, 'model_type', 'standard')
            # Try to get model name for time feature judgment
            self.model_name = getattr(config, 'model_name', None)
        else:
            # Traditional way, use parameters directly
            self.seq_len = int(kwargs.get('seq_len', config_or_seq_len))
            self.pred_len = int(pred_len)
            self.d_model = int(d_model)
            self.label_len = int(label_len)
            self.model_type = model_type
            self.model_name = kwargs.get('model_name', None)
        
    def create_time_marks(self, date_strings: List[str], label_len: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create time feature marks (归一化+周期化编码+weekday sin/cos) - Optimized version
        Args:
            date_strings: List of date strings, format YYYYMMDD
            label_len: Label length, for some models (e.g., Autoformer)
        Returns:
            x_mark_enc: Encoder time marks (B, seq_len, time_features)
            x_mark_dec: Decoder time marks (B, dec_time_len, time_features)
        """
        import numpy as np
        from datetime import datetime, timedelta
        
        batch_size = len(date_strings)
        time_features = 7
        dec_time_len = label_len + self.pred_len
        
        # 批量解析日期并转换为基准日期
        base_dates = []
        for date_str in date_strings:
            try:
                date_str = str(date_str)
                if len(date_str) == 8:
                    year = int(date_str[:4])
                    month = int(date_str[4:6])
                    day = int(date_str[6:8])
                    base_date = datetime(year, month, day, 12, 0, 0)
                else:
                    base_date = datetime(2010, 5, 6, 12, 0, 0)
            except:
                base_date = datetime(2010, 5, 6, 12, 0, 0)
            base_dates.append(base_date)
        
        def compute_time_features_vectorized(base_dates, time_offsets):
            """向量化计算时间特征"""
            batch_results = []
            
            for base_date in base_dates:
                # 批量生成所有时间点
                dates = [base_date + timedelta(days=int(offset)) for offset in time_offsets]
                
                # 提取所有时间组件
                years = np.array([d.year for d in dates])
                months = np.array([d.month for d in dates])
                days = np.array([d.day for d in dates])
                weekdays = np.array([d.weekday() for d in dates])
                
                # 向量化计算所有特征
                year_norm = (years - 2000) / 24.0
                
                # 周期化编码 - 向量化
                month_angles = 2 * np.pi * (months - 1) / 12
                month_sin = np.sin(month_angles)
                month_cos = np.cos(month_angles)
                
                day_angles = 2 * np.pi * (days - 1) / 31
                day_sin = np.sin(day_angles)
                day_cos = np.cos(day_angles)
                
                weekday_angles = 2 * np.pi * weekdays / 7
                weekday_sin = np.sin(weekday_angles)
                weekday_cos = np.cos(weekday_angles)
                
                # 堆叠所有特征
                features = np.stack([
                    year_norm, month_sin, month_cos, 
                    day_sin, day_cos, weekday_sin, weekday_cos
                ], axis=1)
                
                batch_results.append(features)
            
            return np.stack(batch_results)
        
        # 生成编码器时间偏移
        enc_time_offsets = np.arange(self.seq_len) - self.seq_len + 1
        
        # 生成解码器时间偏移  
        dec_time_offsets = np.arange(dec_time_len) - label_len + 1
        
        # 向量化计算编码器和解码器的时间特征
        x_mark_enc_np = compute_time_features_vectorized(base_dates, enc_time_offsets)
        x_mark_dec_np = compute_time_features_vectorized(base_dates, dec_time_offsets)
        
        # 转换为torch张量
        x_mark_enc = torch.from_numpy(x_mark_enc_np).float()
        x_mark_dec = torch.from_numpy(x_mark_dec_np).float()
        
        return x_mark_enc, x_mark_dec
    
    def prepare_standard_inputs(self, past_data: torch.Tensor, future_data: torch.Tensor, 
                              date_strings: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare inputs for standard Transformer-like models
        
        Args:
            past_data: (B, C, T_past) Past data
            future_data: (B, C, T_future) Future data
            date_strings: List of date strings
            
        Returns:
            x_enc: Encoder input (B, seq_len, C)
            x_mark_enc: Encoder time marks (B, seq_len, time_features)
            x_dec: Decoder input (B, pred_len, C) 
            x_mark_dec: Decoder time marks (B, pred_len, time_features)
        """
        batch_size = past_data.shape[0]
        
        # 1. Prepare encoder input
        past_truncated = past_data[:, :, -self.seq_len:]  # Take last seq_len days
        x_enc = past_truncated.transpose(1, 2)  # (B, seq_len, C)
        
        # 2. Prepare decoder input
        future_truncated = future_data[:, :, :self.pred_len]  # Take first pred_len days
        
        # For models like Autoformer, the decoder input should correspond to seasonal_init and trend_init
        # but Autoformer handles these internally, so we just provide a placeholder
        # x_dec is actually not directly used, as Autoformer creates seasonal_init and trend_init internally
        x_dec = torch.zeros(batch_size, self.pred_len, past_data.shape[1])
        
        # 3. Create time marks - consider label_len
        x_mark_enc, x_mark_dec = self.create_time_marks(date_strings, label_len=self.label_len)
        return x_enc, x_mark_enc, x_dec, x_mark_dec
    
    def adapt_inputs(self, past_data: torch.Tensor, future_data: torch.Tensor, 
                    date_strings: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Adapt inputs for standard Transformer-like models
        
        Args:
            past_data: (B, C, T_past) Past data
            future_data: (B, C, T_future) Future data
            date_strings: List of date strings
            
        Returns:
            x_enc: Encoder input (B, seq_len, C)
            x_mark_enc: Encoder time marks (B, seq_len, time_features)
            x_dec: Decoder input (B, pred_len, C) 
            x_mark_dec: Decoder time marks (B, label_len+pred_len, time_features)
        """
        return self.prepare_standard_inputs(past_data, future_data, date_strings)

def get_unified_model_configs(model_name=None, model_type='standard'):
    """
    Get unified model configurations
    
    Args:
        model_name: Model name
        model_type: Model type (default: 'standard')
    
    Returns:
        dict: Model configuration dictionary
    """
    
    # Standard model configurations
    base_config = {
        'seq_len': 15,
        'pred_len': 7,
        'label_len': 0,  # Default to 0, specific models will override
        'd_model': 1024,
        'n_heads': 16,
        'd_ff': 1024,
        'e_layers': 2,
        'd_layers': 2,
        'dropout': 0.1,
        'activation': 'gelu',
        'output_attention': False,
        'enc_in': 39,  # Input feature dimension, changed to 39 for consistency
        'dec_in': 39,  # Decoder input feature dimension, changed to 39 for consistency  
        'c_out': 39,   # Output feature dimension
        'embed': 'timeF',
        'freq': 'd',   # Daily frequency
        'factor': 1,
        'moving_avg': 25,  # For Autoformer
        'channel_independence': False,
        'use_norm': True,
        'd_state': 16,       # For Mamba-related models
        'd_conv': 4,         # For Mamba-related models
        'expand': 2,         # For Mamba-related models
        'distil': True,      # For Informer
    }
    
    # Specific model configurations
    model_specific_configs = {
        'Autoformer': {'label_len': 3},
        'Autoformer_M': {'label_len': 3},
        'iTransformer': {'class_strategy': 'projection'},
        'iInformer': {'class_strategy': 'projection'},
        'iReformer': {'class_strategy': 'projection'},
        'iFlowformer': {'class_strategy': 'projection'},
        'iFlashformer': {'class_strategy': 'projection'},
        # s_mamba's special configuration
        's_mamba': {
            'd_model': 1024,
            'd_ff': 1024,
            'e_layers': 2,
            'activation': 'gelu',
            'use_norm': True,
            'embed': 'timeF',
            'freq': 'd'
        },
        # Add models missing configurations
        'Nonstationary_Transformer': {
            'p_hidden_dims': [128, 128],
            'p_hidden_layers': 2,
            'label_len': 3  # Set appropriate label_len
        },
        'FEDformer': {
            'label_len': 3,  # Set appropriate label_len
            'moving_avg': 25,  # Decomposition window size
            'version': 'fourier',  # Default to fourier version
            'mode_select': 'random',  # Mode selection method
            'modes': 32  # Number of selected modes
        },
        'TemporalFusionTransformer': {
            'data': 'custom',  # Add data configuration
            'hidden_size': 128,
            'lstm_layers': 1,
            'dropout': 0.1,
            'attn_heads': 4,
            'quantiles': [0.1, 0.5, 0.9]
        },
        'TimeMixer': {
            'seq_len': 30,  # TimeMixer needs a longer sequence length to match downsampling
            'down_sampling_window': 2,
            'down_sampling_layers': 3,
            'down_sampling_method': 'avg',
            'use_future_temporal_feature': True,
            'decomp_method': 'moving_avg',
            'moving_avg_window': 25,
            'channel_independence': False,
            'decomp_kernel': [32],
            'conv_kernel': [24],
            'freq': 'd'  # Use daily frequency, match 4 time features: year, month, day, weekday
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
            'embed': 'fixed',  # Pyraformer uses fixed embedding
            'freq': 'd',  # Use daily frequency, match 4 time features: year, month, day, weekday
            'window_size': [2, 2],  # Reduce window size to adapt to shorter sequence
            'inner_size': 3,        # Reduce inner_size to adapt to shorter sequence
            'CSCM': 'Bottleneck_Construct',
            'truncate': True,
            'use_tvm': False,
            'decoder': 'FC'
        },
        'ETSformer': {
            'top_k': 5,  # ETSformer requires top_k parameter
            'e_layers': 2,  # Ensure encoder layers
            'd_layers': 2   # Ensure decoder layers are equal
        },
        'TimeXer': {
            'features': 'M',  # TimeXer requires features parameter
            'patch_len': 16,  # patch related parameters
            'stride': 8,      # stride parameter
            'enc_in': 38,     # Ensure input dimension
            'c_out': 39       # Ensure output dimension
        },
        'CrossLinear': {
            'features': 'M',  # CrossLinear requires features parameter
            'patch_len': 16,  # patch related parameters
            'alpha': 0.5,     # CrossLinear's alpha parameter
            'beta': 0.5       # CrossLinear's beta parameter
        },
        'TimesNet': {
            'top_k': 5,       # TimesNet requires top_k parameter
            'num_kernels': 6  # TimesNet's num_kernels parameter
        },
        # Other models use default label_len=0
    }
    
    # Apply specific model configurations
    if model_name and model_name in model_specific_configs:
        base_config.update(model_specific_configs[model_name])
    
    return base_config

# Backward compatible aliases
ModelAdapter = UnifiedModelAdapter
get_model_configs = get_unified_model_configs

# Example usage
if __name__ == "__main__":
    # Test unified adapter
    batch_size = 4
    seq_len = 30
    pred_len = 7
    d_model = 38
    
    # Simulate data
    past_data = torch.randn(batch_size, d_model, 365)
    future_data = torch.randn(batch_size, d_model, 30)
    date_strings = ['20240101', '20240102', '20240103', '20240104']
    
    print("🧪 Unified model adapter test")
    print("=" * 50)
    
    # Test standard model adapter
    print("\n📋 Standard model configuration:")
    standard_config = get_unified_model_configs('Autoformer', 'standard')
    print(f"  d_model: {standard_config['d_model']}")
    print(f"  n_heads: {standard_config['n_heads']}")
    print(f"  d_ff: {standard_config['d_ff']}")
    print(f"  e_layers: {standard_config['e_layers']}")
    
    adapter_std = UnifiedModelAdapter(seq_len=seq_len, pred_len=pred_len, d_model=d_model, label_len=3, model_type='standard')
    x_enc, x_mark_enc, x_dec, x_mark_dec = adapter_std.adapt_inputs(past_data, future_data, date_strings)
    
    print(f"\n📊 Standard model inputs:")
    print(f"  x_enc: {x_enc.shape}")
    print(f"  x_mark_enc: {x_mark_enc.shape}")
    print(f"  x_dec: {x_dec.shape}")
    print(f"  x_mark_dec: {x_mark_dec.shape}")
    

    
    print(f"\n🔍 Time feature example: {x_mark_enc[0, 0, :]} (month, weekday, day of year)")
    
    print("\n✅ Unified adapter test completed!") 