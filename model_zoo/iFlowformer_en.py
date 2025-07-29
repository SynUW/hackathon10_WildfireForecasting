import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FlowAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
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
        exo_weather = exogenous_x[:, :, 1:13]  # ERA5数据 (前12个变量)
        exo_modis_rbf = exogenous_x[:, :, 20:]  # RBF处理后的MODIS数据
        
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
        exogenous_x = torch.cat([exogenous_x, wavelet_expanded], dim=2)  
        
        enc_out = self.enc_embedding(exogenous_x, x_mark_enc)
        enc = torch.cat([endo_embed, enc_out], dim=1)
        
        enc_out, attns = self.encoder(enc, attn_mask=None)

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
