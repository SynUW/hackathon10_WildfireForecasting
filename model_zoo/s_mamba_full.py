"""
Is Mamba Effective for Time Series Forecasting?
Standalone version - All dependencies included

ToDo:
- [ ] more effective time embedding
- [ ] MOE
- [ ] Bayesian Mamba
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import datetime

from mamba_ssm import Mamba # type: ignore


class DataEmbedding_inverted(nn.Module):
# modifications:
# original: cat(x, x_mark) and then embed
# modified: embed x and x_mark separately and then cat

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, time_feat_dim=6):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.time_embedding = nn.Linear(time_feat_dim, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # x: [B, L, N]  (batch, seq_len, num_vars)
        # x_mark: [B, L, T] (batch, seq_len, time_feat_dim)
        x = x.permute(0, 2, 1)  # [B, N, L]
        # each variable is embedded to a token
        value_emb = self.value_embedding(x)  # [B, N, d_model]
        if x_mark is not None:
            # time_feat_dim is 6, so each variable has a time embedding
            time_emb = self.time_embedding(x_mark).permute(0, 2, 1)  # [B, d_model, L]
            time_emb = time_emb.unsqueeze(1).expand(-1, value_emb.shape[1], -1, -1)  # [B, N, d_model, L]
            time_emb = time_emb.mean(-1)  # [B, N, d_model]
            x = value_emb + time_emb
        else:
            x = value_emb
        return self.dropout(x)

# =============================================================================
# Included from model_zoo/layers/Mamba_EncDec.py
# =============================================================================

class EncoderLayer(nn.Module):
    def __init__(self, attention, attention_r, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.attention_r = attention_r
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # bidirectional mamba blocks
        new_x = self.attention(x) + self.attention_r(x.flip(dims=[1])).flip(dims=[1])
            
        attn = 1

        x = x + new_x
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

# =============================================================================
# Main Model Implementation
# =============================================================================

class Configs:
    def __init__(self, seq_len=10, pred_len=7, d_model=256, d_state=256, d_ff=2048, 
                 e_layers=5, dropout=0.1, activation='relu', output_attention=False,
                 use_norm=False, embed='timeF', freq='d'):
        # Model basic parameters
        self.seq_len = seq_len  # Input sequence length
        self.pred_len = pred_len  # Prediction length
        self.d_model = d_model  # Model dimension
        self.d_state = d_state  # SSM state expansion factor
        self.d_ff = d_ff   # Feed-forward network dimension
        
        # Model structure parameters
        self.e_layers = e_layers  # Number of encoder layers
        self.dropout = dropout  # Dropout rate
        self.activation = activation  # Activation function
        
        # Other parameters
        self.output_attention = output_attention  # Whether to output attention weights
        self.use_norm = use_norm  # Whether to use normalization
        self.embed = embed  # Embedding type
        self.freq = freq  # Frequency

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    Standalone version with all dependencies included
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        
        # Embedding - first parameter should be sequence length
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        # Encoder-only architecture
        # Use Mamba blocks
        encoder_layers = [
            EncoderLayer(  # each layer is a bidirectional mamba block
                Mamba(
                    d_model=configs.d_model,  # Model dimension d_model
                    d_state=configs.d_state,  # SSM state expansion factor
                    d_conv=2,  # Local convolution width
                    expand=1,  # Block expansion factor)
                ),
                Mamba(
                    d_model=configs.d_model,  # Model dimension d_model
                    d_state=configs.d_state,  # SSM state expansion factor
                    d_conv=2,  # Local convolution width
                    expand=1,  # Block expansion factor)
                ),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation
            ) for _ in range(configs.e_layers)
        ]
            
        self.encoder = Encoder(
            encoder_layers,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    # core function
    def forecast(self, x_enc, x_mark_enc):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            valid_mask = torch.ones_like(x_enc)
            valid_mask[:, :, 1:] = (x_enc[:, :, 1:] != 0).float()
            
            eps = 1e-8
            valid_counts = valid_mask.sum(dim=1, keepdim=True) + eps
            means = (x_enc * valid_mask).sum(dim=1, keepdim=True) / valid_counts
            variances = ((x_enc - means)**2 * valid_mask).sum(dim=1, keepdim=True) / valid_counts
            stdev = torch.sqrt(variances + eps)
            x_enc = ((x_enc - means) / stdev) * valid_mask

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E using linear layer to embed each token (each token is a variable))
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E
        # mamba encoder
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # B N E -> B N S -> B S N
        # linear layer to project the output to the same dimension as the input
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, target_date, mask=None):
        # x_enc: [B, L, D] where L represents data from the previous L days
        # target_date: [B] list of date strings in yyyymmdd format
        
        batch_size, seq_len, _ = x_enc.shape
        device = x_enc.device  # Get the device of input tensor
        
        # 生成每个样本输入序列的真实日期列表
        all_time_encodings = []
        for i, date_str in enumerate(target_date):
            # 目标日
            year = int(str(date_str)[:4])
            month = int(str(date_str)[4:6])
            day = int(str(date_str)[6:8])
            target_dt = datetime.datetime(year, month, day)
            # 输入序列每一天的日期（假设数据是连续的，且target_date为预测日，输入为前seq_len天）
            input_dates = [target_dt - datetime.timedelta(days=seq_len - j) for j in range(1, seq_len+1)]
            # 依次编码每一天
            time_encodings = []
            for dt in input_dates:
                # 1. 月份编码
                month_sin = torch.sin(torch.tensor(2 * np.pi * dt.month / 12, device=device))
                month_cos = torch.cos(torch.tensor(2 * np.pi * dt.month / 12, device=device))
                # 2. 星期编码
                weekday = dt.weekday()
                weekday_sin = torch.sin(torch.tensor(2 * np.pi * weekday / 7, device=device))
                weekday_cos = torch.cos(torch.tensor(2 * np.pi * weekday / 7, device=device))
                # 3. 年内天数编码
                day_of_year = dt.timetuple().tm_yday
                day_sin = torch.sin(torch.tensor(2 * np.pi * day_of_year / 366, device=device))
                day_cos = torch.cos(torch.tensor(2 * np.pi * day_of_year / 366, device=device))
                time_encoding = torch.tensor([month_sin, month_cos, weekday_sin, weekday_cos, day_sin, day_cos], device=device)
                time_encodings.append(time_encoding)
            # [seq_len, 6]
            all_time_encodings.append(torch.stack(time_encodings))
        # [B, seq_len, 6]
        x_mark_enc = torch.stack(all_time_encodings)
        
        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
    
    def get_parameter_number(self):
        """
        Number of model parameters
        """
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        trainable_ratio = trainable_num / total_num

        print('total_num:', total_num)
        print('trainable_num:', trainable_num)
        print('trainable_ratio:', trainable_ratio)
        
        return total_num, trainable_num, trainable_ratio

if __name__ == '__main__':
    configs = Configs(
        seq_len=10,
        pred_len=7,
        d_model=39,
        d_state=16,
        d_ff=256,
        e_layers=2,
        dropout=0.1,
    )
    
    # Create model and move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(configs).to(device)
    
    # Test data
    batch_size = 32
    x_enc = torch.randn(batch_size, configs.seq_len, configs.d_model).to(device)  # [32, 10, 39]
    target_date = ['20010829'] * batch_size  # Example date
    
    # Forward propagation test
    output = model(x_enc, target_date)
    print(f"Input shape: {x_enc.shape}")
    print(f"Output shape: {output.shape}")
    
    # Print model parameters
    model.get_parameter_number()
    
    # Test forecast method directly
    x_mark_enc = torch.randn(batch_size, configs.seq_len, 6).to(device)
    forecast_output = model.forecast(x_enc, x_mark_enc)
    print(f"Forecast output shape: {forecast_output.shape}") 