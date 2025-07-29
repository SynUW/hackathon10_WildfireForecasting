import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 添加正确的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
layers_path = os.path.join(current_dir, 'layers')
sys.path.insert(0, layers_path)

from Transformer_EncDec import Encoder, EncoderLayer
from SelfAttention_Family import FlowAttention, AttentionLayer
from Embed import DataEmbedding_inverted, PositionalEmbedding
import numpy as np


class EnEmbedding(nn.Module):
    def __init__(self, d_model, dropout):
        super(EnEmbedding, self).__init__()
        # Patching
        # self.patch_len = patch_len
        self.value_embedding = nn.Linear(365, d_model, bias=False)
            
        self.glb_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1))
        # [B, N, L] -> [B, N, L/patch_len, patch_len]
        # x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)

        # [B, N, L/patch_len, patch_len] -> [B*N, L/patch_len, patch_len]
        # x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        
        x = self.value_embedding(x) + self.position_embedding(x)
        
        # Input encoding
        # [B*N, L/patch_len, patch_len] -> [B*N, L/patch_len, d_model]
        # x = self.value_embedding(x) + self.position_embedding(x)
        # [B*N, L/patch_len, d_model] -> [B, N, L/patch_len, d_model]
        # x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))

        # [B, N, L/patch_len, d_model] -> [B, N, L/patch_len+1, d_model]
        # serise-level global token is added to the last position
        x = torch.cat([x, glb], dim=1)
        
        # x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        
        # Embedding
        self.en_embedding = EnEmbedding(
            configs.d_model, 
            configs.dropout,
        )
        
        # 为exogenous_x添加RBF处理
        self.exo_rbf_centers = 50 # 10 centers by default
        self.exo_rbf_gamma = 10  # 1.0 by default
        learnable_centers=False
        if learnable_centers:
            self.exo_rbf_centers_param = nn.Parameter(torch.randn(self.exo_rbf_centers) * 2 - 1)
        else:
            self.register_buffer('exo_rbf_centers_buffer', torch.linspace(-1, 1, self.exo_rbf_centers))
        
        # 添加可学习的频率域变换矩阵
        self.freq_transform_weather_mag = nn.Parameter(torch.randn(12))  # ERA5幅度变换参数
        self.freq_transform_weather_phase = nn.Parameter(torch.randn(12))  # ERA5相位变换参数
        
        # 使用最大维度初始化MODIS变换参数，运行时动态调整
        self.max_modis_dim = 1500  # 设置最大维度
        self.freq_transform_modis_mag = nn.Parameter(torch.randn(self.max_modis_dim))  # MODIS幅度变换参数
        self.freq_transform_modis_phase = nn.Parameter(torch.randn(self.max_modis_dim))  # MODIS相位变换参数
        
        # FFT-ILT变换参数
        self.fft_filter_W = nn.Parameter(torch.randn(2 * configs.d_model, configs.d_model))  # 频域滤波矩阵 (2*D for real+imag)
        self.ilt_linear = nn.Linear(configs.d_model, configs.d_model)  # ILT线性映射
        
        # 可学习的ILT重建参数 (A_n, σ_n, w_n, φ_n)
        self.ilt_A = nn.Parameter(torch.randn(configs.d_model))  # 幅度参数
        self.ilt_sigma = nn.Parameter(torch.randn(configs.d_model))  # 衰减参数
        self.ilt_w = nn.Parameter(torch.randn(configs.d_model))  # 频率参数
        self.ilt_phi = nn.Parameter(torch.randn(configs.d_model))  # 相位参数
        
        # 控制变量
        self.fourier_as_features = False  # 是否将傅里叶特征作为额外特征
        self.fft_ifft = True  # 是否使用FFT-IFFT变换
        self.fft_ilt = True  # 是否使用FFT-ILT变换
        
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout, time_feat_dim=7)
        
        self.en_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FlowAttention(attention_dropout=configs.dropout), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    multi_variate=False,
                    moe_active=False
                ) for l in range(1)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FlowAttention(attention_dropout=configs.dropout), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    multi_variate=True,
                    moe_active=False
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def fourier_transform_features(self, x):
        """对时间序列应用傅里叶变换并提取特征"""
        # x: [B, L, N] -> [B, L, N]
        B, L, N = x.shape
        
        # 应用FFT
        fft_result = torch.fft.fft(x, dim=1)  # [B, L, N]
        
        # 提取幅度谱（前L//2个频率分量）
        magnitude = torch.abs(fft_result[:, :L//2, :])  # [B, L//2, N]
        
        # 提取相位谱
        phase = torch.angle(fft_result[:, :L//2, :])  # [B, L//2, N]
        
        # 计算功率谱密度
        power_spectrum = magnitude ** 2  # [B, L//2, N]
        
        # 提取主要频率特征（只取前2个频率分量）
        dominant_freqs = magnitude[:, :2, :]  # [B, 2, N]
        
        # 计算频谱统计特征
        mean_freq = magnitude.mean(dim=1, keepdim=True)  # [B, 1, N]
        max_freq = magnitude.max(dim=1, keepdim=True)[0]  # [B, 1, N]
        
        # 拼接特征（减少特征数量）
        fourier_features = torch.cat([
            dominant_freqs,  # [B, 2, N]
            mean_freq,       # [B, 1, N]
            max_freq         # [B, 1, N]
        ], dim=1)  # [B, 4, N]
        
        return fourier_features

    def wavelet_transform_features(self, x):
        """对时间序列应用小波变换并提取特征"""
        # x: [B, L, N] -> [B, L, N]
        B, L, N = x.shape
        
        # 使用简单的Haar小波变换（可以通过卷积实现）
        # 这里使用简化的方法：计算不同尺度的差分特征
        
        # 1尺度差分
        diff1 = x[:, 1:, :] - x[:, :-1, :]  # [B, L-1, N]
        
        # 2尺度差分
        diff2 = x[:, 2:, :] - x[:, :-2, :]  # [B, L-2, N]
        
        # 计算统计特征（只保留均值）
        mean_diff1 = diff1.mean(dim=1, keepdim=True)  # [B, 1, N]
        mean_diff2 = diff2.mean(dim=1, keepdim=True)  # [B, 1, N]
        
        # 拼接小波特征（减少特征数量）
        wavelet_features = torch.cat([
            mean_diff1,  # [B, 1, N]
            mean_diff2   # [B, 1, N]
        ], dim=1)  # [B, 2, N]
        
        return wavelet_features

    def fft_ilt_transform(self, x):
        """对特征图进行FFT-ILT变换"""
        # x: [B, L, D] -> [B, L, D]
        B, L, D = x.shape
        
        # 1. FFT变换到频域
        fft_x = torch.fft.fft(x, dim=1)  # [B, L, D]
        
        # 2. 将复数转换为实数进行处理
        fft_x_real = torch.cat([fft_x.real, fft_x.imag], dim=-1)  # [B, L, 2*D]
        
        # 3. 与可学习的滤波矩阵W相乘
        # 重塑为 [B*L, 2*D] 进行矩阵乘法
        fft_x_reshaped = fft_x_real.reshape(B * L, 2 * D)  # [B*L, 2*D]
        fft_filtered = torch.mm(fft_x_reshaped, self.fft_filter_W)  # [B*L, D]
        fft_filtered = fft_filtered.reshape(B, L, D)  # [B, L, D]
        
        # 4. 线性映射
        fft_mapped = self.ilt_linear(fft_filtered)  # [B, L, D]
        
        # 5. 可学习的逆拉普拉斯变换 (ILT)
        # 使用学习到的参数重建时域信号
        # 公式: f(t) = Σ A_n * exp(-σ_n * t) * cos(w_n * t + φ_n)
        
        # 创建时间向量
        t = torch.arange(L, dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(-1)  # [1, L, 1]
        
        # 应用可学习的ILT参数
        # 确保参数为正数（使用softplus激活）
        A = F.softplus(self.ilt_A.unsqueeze(0).unsqueeze(0))  # [1, 1, D]
        sigma = F.softplus(self.ilt_sigma.unsqueeze(0).unsqueeze(0))  # [1, 1, D]
        w = F.softplus(self.ilt_w.unsqueeze(0).unsqueeze(0))  # [1, 1, D]
        phi = self.ilt_phi.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
        
        # 计算ILT重建
        # exp(-σ_n * t) * cos(w_n * t + φ_n)
        decay = torch.exp(-sigma * t)  # [1, L, D]
        oscillation = torch.cos(w * t + phi)  # [1, L, D]
        
        # 最终重建
        ilt_reconstructed = A * decay * oscillation  # [1, L, D]
        
        # 将ILT重建与频域映射结果结合
        # 使用门控机制控制ILT的影响
        gate = torch.sigmoid(fft_mapped.mean(dim=-1, keepdim=True))  # [B, L, 1]
        output = gate * ilt_reconstructed + (1 - gate) * fft_mapped  # [B, L, D]
        
        return output

    def exo_rbf_transform(self, x):
        """为exogenous_x应用RBF变换，先插值再提取RBF特征（高效向量化版本）"""
        # x: [B, L, N] -> [B, L, N]
        B, L, N = x.shape
        
        # 获取RBF中心点
        if hasattr(self, 'exo_rbf_centers_param'):
            rbf_centers = self.exo_rbf_centers_param
        else:
            rbf_centers = self.exo_rbf_centers_buffer
        
        # 创建原始缺失值掩码（0值表示缺失）
        original_mask = (x == 0).float()  # [B, L, N]
        
        # 快速插值：使用前向填充和后向填充的组合
        x_interpolated = x.clone()
        
        # 重塑为 [B*N, L] 进行批量处理
        x_reshaped = x.reshape(B * N, L)  # [B*N, L]
        
        # 创建缺失值掩码
        missing_mask = (x_reshaped == 0)  # [B*N, L]
        
        # 前向填充
        x_forward = x_reshaped.clone()
        for i in range(1, L):
            x_forward[:, i] = torch.where(
                missing_mask[:, i],
                x_forward[:, i-1],
                x_forward[:, i]
            )
        
        # 后向填充
        x_backward = x_reshaped.clone()
        for i in range(L-2, -1, -1):
            x_backward[:, i] = torch.where(
                missing_mask[:, i],
                x_backward[:, i+1],
                x_backward[:, i]
            )
        
        # 取前向和后向填充的平均值
        x_interpolated_reshaped = (x_forward + x_backward) / 2
        
        # 恢复原始形状
        x_interpolated = x_interpolated_reshaped.view(B, L, N)
        
        # 向量化RBF变换
        # [B, L, N, 1] - [rbf_centers] -> [B, L, N, rbf_centers]
        distances = x_interpolated.unsqueeze(-1) - rbf_centers.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        # 应用RBF核函数
        rbf_output = torch.exp(-self.exo_rbf_gamma * distances ** 2)  # [B, L, N, rbf_centers]
        
        # 对RBF中心维度求平均
        rbf_features = rbf_output.mean(dim=-1)  # [B, L, N]
        
        return rbf_features, original_mask


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev

        # valid_mask = torch.ones_like(x_enc)
        # valid_mask[:, :, 1:] = (x_enc[:, :, 1:] != 0).float()
        
        # eps = 1e-8
        # valid_counts = valid_mask.sum(dim=1, keepdim=True) + eps
        # means = (x_enc * valid_mask).sum(dim=1, keepdim=True) / valid_counts
        # variances = ((x_enc - means)**2 * valid_mask).sum(dim=1, keepdim=True) / valid_counts
        # stdev = torch.sqrt(variances + eps)
        # x_enc = ((x_enc - means) / stdev) * valid_mask

        _, _, N = x_enc.shape
        
        # Embedding.
        # B, L, 1 -> B, 1, L
        endogenous_x = x_enc[:, :, 0].unsqueeze(-1).permute(0, 2, 1)
        exogenous_x = x_enc[:, :, 1:]
        
        endo_embed, n_vars = self.en_embedding(endogenous_x)
        endo_embed, attns = self.en_encoder(endo_embed, attn_mask=None)
        endo_embed = endo_embed[:, 0, :].unsqueeze(1)  # B, 1, d_model, global token
    
        # 对MODIS数据应用RBF变换
        exo_modis, original_mask_tensor = self.exo_rbf_transform(exogenous_x[:, :, 20:])  # [B, L, 18]
        exogenous_x = torch.cat([exogenous_x[:, :, :20], exo_modis], dim=2)
        
        # 对ERA5和RBF后的MODIS数据应用傅里叶变换和小波变换，并提取特征拼接输入模型
        enc_out = self.enc_embedding(exogenous_x, x_mark_enc)
        exo_weather = enc_out[:, :, 1:13]  # ERA5数据 (前12个变量)
        exo_modis_rbf = enc_out[:, :, 20:]  # RBF处理后的MODIS数据
        
        if self.fourier_as_features:
            # 应用傅里叶变换
            fourier_weather = self.fourier_transform_features(exo_weather)  # [B, 8, 12]
            fourier_modis = self.fourier_transform_features(exo_modis_rbf)  # [B, 8, 18]
            
            # 应用小波变换
            wavelet_weather = self.wavelet_transform_features(exo_weather)  # [B, 6, 12]
            wavelet_modis = self.wavelet_transform_features(exo_modis_rbf)  # [B, 6, 18]
            
            # 将变换特征与原始数据拼接
            # 原始数据: [B, L, 38] (20个原始变量 + 18个RBF特征)
            # 傅里叶特征: [B, 4, 30] (4个特征 × 30个变量)
            # 小波特征: [B, 2, 30] (2个特征 × 30个变量)
            
            # 拼接傅里叶特征
            fourier_combined = torch.cat([fourier_weather, fourier_modis], dim=2)  # [B, 4, 30]
            # 将傅里叶特征扩展到时间维度
            fourier_expanded = fourier_combined.unsqueeze(1).expand(-1, exogenous_x.shape[1], -1, -1)  # [B, L, 4, 30]
            fourier_expanded = fourier_expanded.reshape(exogenous_x.shape[0], exogenous_x.shape[1], -1)  # [B, L, 120]
            
            # 拼接小波特征
            wavelet_combined = torch.cat([wavelet_weather, wavelet_modis], dim=2)  # [B, 2, 30]
            # 将小波特征扩展到时间维度
            wavelet_expanded = wavelet_combined.unsqueeze(1).expand(-1, exogenous_x.shape[1], -1, -1)  # [B, L, 2, 30]
            wavelet_expanded = wavelet_expanded.reshape(exogenous_x.shape[0], exogenous_x.shape[1], -1)  # [B, L, 60]
            
            # 最终拼接：原始数据 + 傅里叶特征 + 小波特征。不使用小波和傅里叶，注释掉这行就可以了
            enc_out = torch.cat([exogenous_x, wavelet_expanded, fourier_expanded], dim=2)  
        elif self.fft_ifft:
            # 对ERA5和MODIS数据分别进行傅里叶变换
            # exo_weather: [B, L, 12], exo_modis_rbf: [B, L, N] (N可能变化)
            
            # 动态检测MODIS数据维度
            modis_dim = exo_modis_rbf.shape[-1]
            
            # 傅里叶变换到频率域
            fft_weather = torch.fft.fft(exo_weather, dim=1)  # [B, L, 12]
            fft_modis = torch.fft.fft(exo_modis_rbf, dim=1)  # [B, L, N]
            
            # 应用可学习的频率域变换（使用幅度和相位）
            # 对ERA5数据
            magnitude_weather = torch.abs(fft_weather)  # [B, L, 12]
            phase_weather = torch.angle(fft_weather)    # [B, L, 12]
            
            # 应用可学习的幅度和相位变换
            magnitude_weather_transformed = magnitude_weather * torch.sigmoid(self.freq_transform_weather_mag.unsqueeze(0).unsqueeze(0))
            phase_weather_transformed = phase_weather + self.freq_transform_weather_phase.unsqueeze(0).unsqueeze(0)
            
            # 重建复数
            fft_weather_transformed = magnitude_weather_transformed * torch.exp(1j * phase_weather_transformed)
            
            # 对MODIS数据 - 使用动态调整的参数
            magnitude_modis = torch.abs(fft_modis)  # [B, L, N]
            phase_modis = torch.angle(fft_modis)    # [B, L, N]
            
            # 动态调整变换参数维度
            if modis_dim <= self.max_modis_dim:
                # 使用前N个参数
                mag_params = self.freq_transform_modis_mag[:modis_dim]
                phase_params = self.freq_transform_modis_phase[:modis_dim]
            else:
                # 如果超出最大维度，使用插值
                indices = torch.linspace(0, self.max_modis_dim-1, modis_dim, device=self.freq_transform_modis_mag.device)
                mag_params = torch.interp(indices, torch.arange(self.max_modis_dim, device=self.freq_transform_modis_mag.device), self.freq_transform_modis_mag)
                phase_params = torch.interp(indices, torch.arange(self.max_modis_dim, device=self.freq_transform_modis_phase.device), self.freq_transform_modis_phase)
            
            # 应用可学习的幅度和相位变换
            magnitude_modis_transformed = magnitude_modis * torch.sigmoid(mag_params.unsqueeze(0).unsqueeze(0))
            phase_modis_transformed = phase_modis + phase_params.unsqueeze(0).unsqueeze(0)
            
            # 重建复数
            fft_modis_transformed = magnitude_modis_transformed * torch.exp(1j * phase_modis_transformed)
            
            # 逆傅里叶变换回时间域
            exo_weather_transformed = torch.fft.ifft(fft_weather_transformed, dim=1).real  # [B, L, 12]
            exo_modis_transformed = torch.fft.ifft(fft_modis_transformed, dim=1).real  # [B, L, N]
            
            # 将变换后的数据替换回原始位置（避免原地操作）
            enc_out_transformed = enc_out.clone()
            enc_out_transformed[:, :, 1:13] = exo_weather_transformed  # 替换ERA5数据
            enc_out_transformed[:, :, 20:] = exo_modis_transformed     # 替换MODIS数据
            
            enc = torch.cat([endo_embed, enc_out_transformed], dim=1)
        
        enc_out, attns = self.encoder(enc, attn_mask=None)
        
        # 应用FFT-ILT变换到encoder输出
        if self.fft_ilt:
            enc_out = self.fft_ilt_transform(enc_out)  # [B, L, D]
        
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        # dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        # dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # print(x_mark_enc.shape)  B T 3 (year, month, day)
        # print(x_mark_enc[0, :, :])
        
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
