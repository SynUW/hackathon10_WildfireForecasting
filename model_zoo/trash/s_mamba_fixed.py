"""
修复版本的 S-Mamba 模型
主要修复：
1. 输入序列长度处理
2. 数据维度匹配
3. 归一化问题
4. 输出维度对齐
"""
import torch
import torch.nn as nn
import os
import sys
import datetime
import numpy as np

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from model_zoo.layers.Mamba_EncDec import Encoder, EncoderLayer
from model_zoo.layers.Embed import DataEmbedding_inverted
from mamba_ssm import Mamba

class Configs:
    def __init__(self, seq_len=372, pred_len=7, d_model=39, d_state=16, d_ff=256, 
                 e_layers=2, dropout=0.1, activation='relu', output_attention=False,
                 use_norm=True, embed='timeF', freq='d'):
        # 模型基本参数
        self.seq_len = seq_len  # 输入序列长度
        self.pred_len = pred_len  # 预测长度
        self.d_model = d_model  # 模型维度
        self.d_state = d_state  # SSM状态扩展因子
        self.d_ff = d_ff   # 前馈网络维度
        
        # 模型结构参数
        self.e_layers = e_layers  # 编码器层数
        self.dropout = dropout  # dropout率
        self.activation = activation  # 激活函数
        
        # 其他参数
        self.output_attention = output_attention  # 是否输出注意力权重
        self.use_norm = use_norm  # 是否使用归一化
        self.embed = embed  # 嵌入类型
        self.freq = freq  # 频率

class Model(nn.Module):
    """
    修复版本的S-Mamba模型
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        
        print(f"初始化S-Mamba模型:")
        print(f"  输入序列长度: {self.seq_len}")
        print(f"  预测长度: {self.pred_len}")
        print(f"  特征维度: {self.d_model}")
        
        # 时间嵌入层 - 将时间特征映射到模型维度
        self.time_embed = nn.Linear(6, configs.d_model // 4)  # 6个时间特征 -> d_model/4
        
        # 输入投影层 - 将输入特征映射到模型维度
        self.input_projection = nn.Linear(configs.d_model, configs.d_model)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, configs.seq_len, configs.d_model) * 0.02)
        
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                        Mamba(
                            d_model=configs.d_model,  # Model dimension d_model
                            d_state=configs.d_state,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=2,  # Block expansion factor
                        ),
                        Mamba(
                            d_model=configs.d_model,  # Model dimension d_model
                            d_state=configs.d_state,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=2,  # Block expansion factor
                        ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # 输出投影层
        self.output_projection = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, configs.pred_len * configs.d_model),
        )
        
        # 最终输出层
        self.final_projection = nn.Linear(configs.d_model, configs.d_model)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def create_time_features(self, target_date):
        """创建时间特征"""
        batch_size = len(target_date)
        device = next(self.parameters()).device
        
        time_encodings = []
        for date_str in target_date:
            # 解析日期
            date_str = str(date_str)
            if len(date_str) == 8:
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
            else:
                # 默认日期
                year, month, day = 2010, 5, 6
            
            try:
                # 计算星期几 (0-6, 0表示星期一)
                weekday = datetime.datetime(year, month, day).weekday()
                # 计算一年中的第几天 (1-366)
                day_of_year = datetime.datetime(year, month, day).timetuple().tm_yday
            except:
                weekday = 0
                day_of_year = 126
            
            # 创建周期性时间编码
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            weekday_sin = np.sin(2 * np.pi * weekday / 7)
            weekday_cos = np.cos(2 * np.pi * weekday / 7)
            day_sin = np.sin(2 * np.pi * day_of_year / 366)
            day_cos = np.cos(2 * np.pi * day_of_year / 366)
            
            time_encoding = torch.tensor([month_sin, month_cos, weekday_sin, 
                                        weekday_cos, day_sin, day_cos], 
                                       dtype=torch.float32, device=device)
            time_encodings.append(time_encoding)
        
        return torch.stack(time_encodings)  # [B, 6]

    def forward(self, x_enc, target_date, mask=None):
        """
        前向传播
        
        Args:
            x_enc: [B, L, D] 输入序列，L=seq_len, D=d_model
            target_date: [B] 目标日期字符串列表
            mask: 掩码（可选）
        
        Returns:
            output: [B, pred_len, d_model] 预测输出
        """
        batch_size, seq_len, d_model = x_enc.shape
        
        # 检查输入维度
        if seq_len != self.seq_len:
            print(f"警告: 输入序列长度 {seq_len} 与配置的长度 {self.seq_len} 不匹配")
        if d_model != self.d_model:
            print(f"警告: 输入特征维度 {d_model} 与配置的维度 {self.d_model} 不匹配")
        
        # 1. 输入投影
        x = self.input_projection(x_enc)  # [B, L, D]
        
        # 2. 添加位置编码
        if seq_len <= self.pos_encoding.size(1):
            x = x + self.pos_encoding[:, :seq_len, :]
        else:
            # 如果输入序列更长，截取位置编码或者插值
            pos_enc = self.pos_encoding[:, :seq_len, :] if seq_len <= self.pos_encoding.size(1) else self.pos_encoding
            x = x + pos_enc
        
        # 3. 数据归一化（可选）
        if self.use_norm:
            # 计算均值和标准差用于后续反归一化
            means = x.mean(dim=1, keepdim=True)  # [B, 1, D]
            x_centered = x - means
            stdev = torch.sqrt(torch.var(x_centered, dim=1, keepdim=True, unbiased=False) + 1e-8)
            x = x_centered / stdev
        
        # 4. 创建时间特征
        time_features = self.create_time_features(target_date)  # [B, 6]
        time_embedded = self.time_embed(time_features)  # [B, D//4]
        
        # 将时间特征扩展到序列长度并连接
        time_expanded = time_embedded.unsqueeze(1).repeat(1, seq_len, 1)  # [B, L, D//4]
        
        # 将时间特征与输入特征连接（如果维度允许）
        if time_expanded.size(-1) + x.size(-1) <= self.d_model:
            # 填充0到相同维度
            padding_size = self.d_model - x.size(-1) - time_expanded.size(-1)
            if padding_size > 0:
                padding = torch.zeros(batch_size, seq_len, padding_size, device=x.device)
                x = torch.cat([x, time_expanded, padding], dim=-1)
            else:
                x = torch.cat([x, time_expanded], dim=-1)
        
        # 5. 通过编码器
        enc_out, _ = self.encoder(x, attn_mask=None)  # [B, L, D]
        
        # 6. 全局池化和输出投影
        # 使用多种池化策略的组合
        global_features = torch.cat([
            enc_out.mean(dim=1),  # 平均池化
            enc_out.max(dim=1)[0],  # 最大池化
            enc_out[:, -1, :],  # 最后一个时间步
        ], dim=-1)  # [B, 3*D]
        
        # 投影到预测维度
        if global_features.size(-1) != self.d_model:
            global_features = nn.functional.adaptive_avg_pool1d(
                global_features.unsqueeze(1), self.d_model
            ).squeeze(1)
        
        # 输出投影
        output_flat = self.output_projection(global_features)  # [B, pred_len * D]
        output = output_flat.view(batch_size, self.pred_len, self.d_model)  # [B, pred_len, D]
        
        # 最终投影
        output = self.final_projection(output)  # [B, pred_len, D]
        
        # 7. 反归一化（如果使用了归一化）
        if self.use_norm:
            output = output * stdev[:, :1, :] + means[:, :1, :]
        
        return output


if __name__ == '__main__':
    # 测试修复版本的模型
    configs = Configs(
        seq_len=372,  # 365 + 7
        pred_len=7,
        d_model=39,
        d_state=16,
        d_ff=256,
        e_layers=2,
        dropout=0.1,
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(configs).to(device)
    
    # 测试数据
    batch_size = 4
    x_enc = torch.randn(batch_size, configs.seq_len, configs.d_model).to(device)
    target_date = ['20010829', '20020315', '20031224', '20040601']
    
    print(f"测试输入形状: {x_enc.shape}")
    
    # 前向传播测试
    with torch.no_grad():
        output = model(x_enc, target_date)
        print(f"测试输出形状: {output.shape}")
        print(f"输出值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数数量: {total_params:,}") 