"""
Is Mamba Effective for Time Series Forecasting?
"""
import torch
import torch.nn as nn
import os
import sys

# Add project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from model_zoo.layers.Mamba_EncDec import Encoder, EncoderLayer
from model_zoo.layers.Embed import DataEmbedding_inverted

from mamba_ssm import Mamba

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
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding - first parameter should be sequence length
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout, time_feat_dim=7)
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
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
                ) for layer_idx in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
    # a = self.get_parameter_number()
    #
    # def get_parameter_number(self):
    #     """
    #     Number of model parameters (without stable diffusion)
    #     """
    #     total_num = sum(p.numel() for p in self.parameters())
    #     trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
    #     trainable_ratio = trainable_num / total_num
    #
    #     print('total_num:', total_num)
    #     print('trainable_num:', total_num)
    #     print('trainable_ratio:', trainable_ratio)

    def forecast(self, x_enc, x_mark_enc):
        # Per-feature time-series normalization（无 clamp）:
        # - Band 0 (FIRMS): y = log1p(x) / log1p(100) ∈ [0,1]
        # - 其它通道: 鲁棒标准化（median/MAD），y=(x-median)/(1.4826·MAD + eps)，无效(0) -> 0

        B, L, N = x_enc.shape
        eps = 1e-6
        device = x_enc.device

        # FIRMS（band 0）
        x0 = x_enc[:, :, 0]
        valid0 = (x0 != 255) & (x0 != -9999) & (x0 >= 0)
        logC = torch.log1p(torch.tensor(100.0, device=device, dtype=x_enc.dtype))
        y0 = torch.log1p(torch.where(valid0, x0, torch.zeros_like(x0))) / logC

        # 其它通道（band 1..N-1）
        if N > 1:
            rest = x_enc[:, :, 1:]
            # 无效值 255/-9999/NaN
            invalid_rest = (rest == -9999) | torch.isnan(rest) | (rest == 255)
            mask_bool = ~invalid_rest  # True=有效
            mask = mask_bool.float()

            # 逐样本逐通道统计（屏蔽无效）
            x_masked = rest.masked_fill(~mask_bool, float('nan'))
            med = torch.nanmedian(x_masked, dim=1).values                             # [B, N-1]
            mad = torch.nanmedian((x_masked - med.unsqueeze(1)).abs(), dim=1).values  # [B, N-1]

            # 安全scale：MAD过小/NaN -> 掩码std，仍不行 -> 1.0
            scale_base = mad * 1.4826
            count = mask.sum(dim=1).clamp_min(1.0)                                     # [B, N-1]
            mean_rest = (rest * mask).sum(dim=1) / count                               # [B, N-1]
            var_rest = (((rest - mean_rest.unsqueeze(1)) ** 2) * mask).sum(dim=1) / count
            std_rest = torch.sqrt(torch.relu(var_rest) + eps)                          # [B, N-1]
            scale_base = torch.nan_to_num(scale_base, nan=0.0)
            std_rest = torch.nan_to_num(std_rest, nan=1.0)
            use_std = (scale_base <= 1e-3) | torch.isnan(scale_base)
            scale = torch.where(use_std, std_rest, scale_base) + eps                   # [B, N-1]

            # 若有效点过少（<3），该通道的归一化直接置为0，并用中性统计量（med=0, scale=1）
            too_few = (count < 3.0)                                                   # [B, N-1]
            med = torch.where(too_few, torch.zeros_like(med), med)
            scale = torch.where(too_few, torch.ones_like(scale), scale)

            # 鲁棒Z-Score（无 clamp）并做平滑有界化，避免极端值引发数值不稳定
            y_rest = (rest - med.unsqueeze(1)) / scale.unsqueeze(1)
            y_rest = torch.tanh(y_rest / 3.0)
            y_rest = torch.where(mask_bool, y_rest, torch.zeros_like(y_rest))
            # 有效点过少整体置零
            y_rest = y_rest * (~too_few).unsqueeze(1).float()

            # 组合并存统计量（用于反归一化）
            x_norm = torch.cat([y0.unsqueeze(-1), y_rest], dim=2)
            self._firms_logC = logC
            self._rest_med = med.unsqueeze(1).detach()            # [B,1,N-1]
            self._rest_scale = scale.unsqueeze(1).detach()        # [B,1,N-1]
        else:
            x_norm = y0.unsqueeze(-1)
            self._firms_logC = logC

        # 防御：避免 NaN/Inf 进入编码器
        # x_norm = torch.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 调试打印如需开启请手动取消注释
        # print(torch.max(x_norm), torch.min(x_norm))

        _B, _L, _N = x_norm.shape # B L N
        
        
        # Max-Min 归一化：基于 N 维度（每个特征通道独立归一化）
        # 计算每个特征通道的 min 和 max
        # x_min = x_norm.min(dim=1, keepdim=True)[0]  # [B, 1, N] - 每个样本每个特征的最小值
        # x_max = x_norm.max(dim=1, keepdim=True)[0]  # [B, 1, N] - 每个样本每个特征的最大值
        
        # # 避免除零错误
        # x_range = x_max - x_min
        # x_range = torch.where(x_range == 0, torch.ones_like(x_range), x_range)
        
        # # Max-Min 归一化到 [0, 1] 范围
        # x_norm = (x_norm - x_min) / x_range
        
        # # 保存归一化参数用于反归一化
        # self._norm_min = x_min.detach()  # [B, 1, N]
        # self._norm_range = x_range.detach()  # [B, 1, N]
        
        # print(f"归一化后 - max: {torch.max(x_norm):.4f}, min: {torch.min(x_norm):.4f}")
        
        
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)

        enc_out = self.enc_embedding(x_norm, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :_N] # filter the covariates

        # Inverse normalization per band
        # FIRMS：输出logits，外部再做sigmoid
        y0_pred = dec_out[:, :, 0]
        x0 = y0_pred
        # 其它通道（鲁棒Z-Score反归一化）：x = y*scale + med
        if _N > 1:
            y_rest_pred = dec_out[:, :, 1:]
            x_rest = y_rest_pred * self._rest_scale + self._rest_med
            dec_out = torch.cat([x0.unsqueeze(-1), x_rest], dim=2)
        else:
            dec_out = x0.unsqueeze(-1)

        # 防御：输出去 NaN/Inf
        # dec_out = torch.nan_to_num(dec_out, nan=0.0, posinf=0.0, neginf=0.0)
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, L, D] where L=10 represents data from the previous 10 days
        # target_date: [B] list of date strings in yyyymmdd format
        
        # batch_size, seq_len, _ = x_enc.shape
        # device = x_enc.device  # Get the device of input tensor
        
        # # Create time encodings
        # time_encodings = []
        # for date_str in target_date:
            # Parse date
        #     year = int(str(date_str)[:4])
        #     month = int(str(date_str)[4:6])
        # #     day = int(str(date_str)[6:8])
            
        #     # Calculate weekday (0-6, 0 represents Monday)
        #     weekday = datetime.datetime(year, month, day).weekday()
            
            # Calculate day of year (1-366)
        #     day_of_year = datetime.datetime(year, month, day).timetuple().tm_yday
            
            # Create time encodings
            # 1. Month encoding (1-12)
            # month_sin = torch.sin(torch.tensor(2 * np.pi * month / 12, device=device))
            # month_cos = torch.cos(torch.tensor(2 * np.pi * month / 12, device=device))
            
            # 2. Weekday encoding (0-6)
            # weekday_sin = torch.sin(torch.tensor(2 * np.pi * weekday / 7, device=device))
            # weekday_cos = torch.cos(torch.tensor(2 * np.pi * weekday / 7, device=device))
            
            # 3. Day of year encoding (1-366)
            # day_sin = torch.sin(torch.tensor(2 * np.pi * day_of_year / 366, device=device))
            # day_cos = torch.cos(torch.tensor(2 * np.pi * day_of_year / 366, device=device))
            
            # Combine time features
            # time_encoding = torch.tensor([month_sin, month_cos, 
            #                             weekday_sin, weekday_cos,
            #                             day_sin, day_cos], device=device)
            # time_encodings.append(time_encoding)
        
        # Convert time encodings to tensor [B, 6]
        # x_mark_enc = torch.stack(time_encodings)
        # # Expand dimensions to match sequence length [B, L, 6]
        # x_mark_enc = x_mark_enc.unsqueeze(1).repeat(1, seq_len, 1)
        
        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, N]
    
    
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
    # print(f"Model structure:\n{model}")