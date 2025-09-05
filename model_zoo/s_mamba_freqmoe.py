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
from model_zoo.layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import datetime
import numpy as np
import torch.nn.functional as F

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
        
class EnEmbedding(nn.Module):
    def __init__(self, d_model, dropout, seq_len=365):
        super(EnEmbedding, self).__init__()
        # Patching
        # self.patch_len = patch_len
        self.value_embedding = nn.Linear(seq_len, d_model, bias=False)
            
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

class FrequencyMoE(nn.Module):
    """
    输入:  x ∈ R^{B×L×N_sel}  (时间维在中间)
    输出:  y ∈ R^{B×L×N_sel}
    作用:  学习 E-1 个频带边界，将频谱切为 E 段；用门控权重融合后 iFFT 回时域。
    备注:  与数据无关的可学习边界(全局共享)；门控权重按样本自适应。
    """
    def __init__(self, seq_len: int, n_experts: int = 3, residual_add: bool = True, eps: float = 1e-6):
        super().__init__()
        assert n_experts >= 1, "n_experts must be >= 1"
        self.seq_len = int(seq_len)
        self.n_experts = int(n_experts)
        self.residual_add = bool(residual_add)
        self.eps = float(eps)

        # (E-1) learnable boundaries in (0,1) after sigmoid
        if self.n_experts >= 2:
            self.band_boundaries = nn.Parameter(torch.randn(self.n_experts - 1))
        else:
            self.register_parameter('band_boundaries', None)

        # Gate MLP: 输入是幅值谱在通道(N_sel)维的均值 [B, F]，输出 [B, E]
        FreqBins = self.seq_len // 2 + 1
        self.gate = nn.Sequential(
            nn.Linear(FreqBins, FreqBins),
            nn.ReLU(inplace=True),
            nn.Linear(FreqBins, self.n_experts)
        )

        # 可选调试缓存
        self.last_indices = None          # torch.LongTensor[E+1]
        self.last_boundaries01 = None     # torch.FloatTensor[E+1]
        self.last_weights = None          # torch.FloatTensor[B, E]

    def _make_indices(self, FreqBins: int, device: torch.device) -> torch.LongTensor:
        """从可学习边界产生离散频点下标 indices（长度 E+1，含 0 和 F）"""
        if self.n_experts == 1:
            indices = torch.tensor([0, FreqBins], dtype=torch.long, device=device)
            self.last_indices = indices
            self.last_boundaries01 = torch.tensor([0.0, 1.0], device=device)
            return indices

        bb = torch.sort(self.band_boundaries.sigmoid()).values              # (E-1) ∈ (0,1)
        boundaries01 = torch.cat([
            torch.zeros(1, device=device),
            bb,
            torch.ones(1, device=device)
        ], dim=0)                                                           # (E+1)
        indices = torch.clamp((boundaries01 * FreqBins).long(), 0, FreqBins)
        indices[-1] = FreqBins                                              # 保证覆盖到最后
        self.last_indices = indices.detach().clone()
        self.last_boundaries01 = boundaries01.detach().clone()
        return indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, N_sel]
        """
        assert x.dim() == 3, "Input must be [B, L, N_sel]"
        B, L, N_sel = x.shape
        assert L == self.seq_len, f"FreqMoE: seq_len mismatch: got {L}, expected {self.seq_len}"

        # [B, L, N] -> [B, N, L]
        xN = x.permute(0, 2, 1).contiguous()

        # 逐通道（每变量）按时间维标准化
        mean = xN.mean(dim=-1, keepdim=True)                                # [B, N, 1]
        var = xN.var(dim=-1, unbiased=False, keepdim=True) + self.eps       # [B, N, 1]
        x_norm = (xN - mean) / var.sqrt()
        x_norm = xN

        # rFFT 沿时间维
        freq = torch.fft.rfft(x_norm, n=self.seq_len, dim=-1)               # [B, N, F]
        FreqBins = freq.size(-1)

        # 频带索引
        indices = self._make_indices(FreqBins, device=x.device)             # [E+1]

        # 子带掩膜 & 组件
        comps = []
        for i in range(self.n_experts):
            s = int(indices[i].item()); e = int(indices[i + 1].item())
            if e <= s:
                comp = torch.zeros_like(freq).unsqueeze(-1)                 # [B, N, F, 1]
            else:
                mask = torch.zeros_like(freq)
                mask[..., s:e] = 1                                          # 1 -> 1+0j
                comp = (freq * mask).unsqueeze(-1)
            comps.append(comp)
        comps = torch.cat(comps, dim=-1)                                    # [B, N, F, E]

        # 门控权重：幅值谱在 N_sel 维均值 -> [B, F] -> [B, E]
        mag = freq.abs().mean(dim=1)                                        # [B, F]
        weights = torch.softmax(self.gate(mag), dim=-1)                     # [B, E]
        self.last_weights = weights.detach().clone()
        w = weights.unsqueeze(1).unsqueeze(1)                               # [B,1,1,E]

        # 频域加权求和 & iFFT
        freq_out = (comps * w).sum(dim=-1)                                  # [B, N, F]
        x_rec = torch.fft.irfft(freq_out, n=self.seq_len, dim=-1)           # [B, N, L]
        x_rec = x_rec * var.sqrt() + mean                                   # 反标准化

        outN = xN + x_rec if self.residual_add else x_rec                   # [B, N, L]
        out = outN.permute(0, 2, 1).contiguous()                            # [B, L, N]
        return out


# ---------------------------------------------------------
# 只对“第 2–13 个特征（1-based）”应用 Frequency-MoE
# （即 0-based 下标 1..12）
# ---------------------------------------------------------
class SelectiveFrequencyMoE(nn.Module):
    """
    Applies FrequencyMoE ONLY on a 1-based closed interval [start_1b, end_1b] of features.
    Default: [2, 13] → 0-based indices [1..12].
    """
    def __init__(self, seq_len: int, n_experts: int = 3, residual_add: bool = True,
                 selected_idx_1based=(2, 13)):
        super().__init__()
        self.seq_len = int(seq_len)
        self.start_1b, self.end_1b = int(selected_idx_1based[0]), int(selected_idx_1based[1])
        assert self.start_1b >= 1 and self.end_1b >= self.start_1b
        self.fmoe = FrequencyMoE(seq_len=seq_len, n_experts=n_experts, residual_add=residual_add)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, N]
        return: [B, L, N] （只对选定特征做了频域 MoE，其余透传）
        """
        assert x.dim() == 3
        B, L, N = x.shape
        assert L == self.seq_len

        s0 = self.start_1b - 1
        e0 = min(self.end_1b - 1, N - 1)
        if s0 >= N or e0 < s0:
            return x  # 区间越界 → 直接返回

        x_sel = x[:, :, s0:e0 + 1]            # [B, L, N_sel]
        x_sel_out = self.fmoe(x_sel)          # [B, L, N_sel]
        out = x.clone()
        out[:, :, s0:e0 + 1] = x_sel_out
        return out


class EndoExoAttention(nn.Module):
    """
    Cross-attention where endogenous (Q) attends to exogenous (K,V).
    Works on [B, N, D] embeddings (variables × embedding dimension).
    """
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super(EndoExoAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj   = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, endo, exo):
        """
        endo: [B, N_endo, D]
        exo:  [B, N_exo, D]
        """
        B, Nq, _ = endo.shape
        B, Nk, _ = exo.shape

        # Project
        Q = self.query_proj(endo)  # [B, Nq, D]
        K = self.key_proj(exo)     # [B, Nk, D]
        V = self.value_proj(exo)   # [B, Nk, D]

        # Split into heads
        Q = Q.view(B, Nq, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, Nq, Dh]
        K = K.view(B, Nk, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, Nk, Dh]
        V = V.view(B, Nk, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, Nk, Dh]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)  # [B, H, Nq, Nk]
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)  # [B, H, Nq, Dh]

        # Merge heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, Nq, self.d_model)  # [B, Nq, D]
        return self.out_proj(attn_out)

'''
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
        
        self.output_attention = getattr(configs, "output_attention", False)

        # 1) 选择性频域 MoE：只处理第 2–13 个特征（1-based）
        moe_range = getattr(configs, "moe_range_1b", (2, 13))
        n_experts = getattr(configs, "n_experts", 3)
        moe_residual = getattr(configs, "moe_residual", True)
        self.sel_freq_moe = SelectiveFrequencyMoE(
            seq_len=configs.seq_len,
            n_experts=n_experts,
            residual_add=moe_residual,
            selected_idx_1based=moe_range
        )
        
        # firms (endogenous) embedding
        self.endogenous_embedding = EnEmbedding(configs.d_model, configs.dropout, configs.seq_len)
        
        # Embedding - first parameter should be number of features (c_in)
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout, time_feat_dim=7)

        self.cross_attn = EndoExoAttention(d_model=configs.d_model, n_heads=4, dropout=configs.dropout)
        
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
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc):
        
        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates
        # print(x_enc.shape)
        # 2. 将 -9999 替换为 0
        x_enc = torch.where(x_enc == -9999, torch.zeros_like(x_enc), x_enc)

        # use freqmoe or not for era5 data
        if 1 == 0:
            x_enc = self.sel_freq_moe(x_enc)
            
        # Embedding
        if 1 == 0:
            # separate endogenous and exogenous embedding
            # B L N -> B N E
            
            # 1. firms (endogenous) embedding
            endo_out, n_vars = self.endogenous_embedding(x_enc[:, :, 0].unsqueeze(-1).transpose(1, 2))
            endo_out = endo_out[:, 1, :].unsqueeze(1)  # B, D, 1， global token
            # 2. exogenous embedding
            enc_out = self.enc_embedding(x_enc[:, :, 1:], x_mark_enc)
            # 3. concatenate endogenous and exogenous embedding
            enc_out = torch.cat([endo_out, enc_out], dim=1)
        else:
            # B L N -> B N E
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # B N E -> B N E
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates


        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, L, D] where L=10 represents data from the previous 10 days
        # target_date: [B] list of date strings in yyyymmdd format
        
        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, N]
'''
    
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
        self.output_attention = getattr(configs, "output_attention", False)

        # Frequency MoE
        moe_range = getattr(configs, "moe_range_1b", (2, 13))
        n_experts = getattr(configs, "n_experts", 3)
        moe_residual = getattr(configs, "moe_residual", True)
        self.sel_freq_moe = SelectiveFrequencyMoE(
            seq_len=configs.seq_len,
            n_experts=n_experts,
            residual_add=moe_residual,
            selected_idx_1based=moe_range
        )
        
        # Endogenous + exogenous embeddings
        self.endogenous_embedding = EnEmbedding(configs.d_model, configs.dropout, configs.seq_len)
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq,
            configs.dropout, time_feat_dim=7
        )

        # Cross-attention
        self.cross_attn = EndoExoAttention(d_model=configs.d_model, n_heads=4, dropout=configs.dropout)
        
        # Encoder (Mamba)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    Mamba(
                        d_model=configs.d_model,
                        d_state=configs.d_state,
                        d_conv=2,
                        expand=1,
                    ),
                    Mamba(
                        d_model=configs.d_model,
                        d_state=configs.d_state,
                        d_conv=2,
                        expand=1,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc):
        """
        x_enc: [B, L, N]  -> input sequence
        x_mark_enc: [B, L, time_features]
        """
        B, L, N = x_enc.shape

        # Replace -9999 with 0
        x_enc = torch.where(x_enc == -9999, torch.zeros_like(x_enc), x_enc)

        # use freqmoe or not for era5 data
        if 1 == 1:
            x_enc = self.sel_freq_moe(x_enc)

        # --- Split endogenous (Q) vs exogenous (K,V) ---
        endo = x_enc[:, :, 0:1]      # FIRMS or first channel
        exo  = x_enc[:, :, 1:] if N > 1 else None

        # Embeddings
        enc_endo = self.enc_embedding(endo, x_mark_enc)   # [B, N_endo, D]
        enc_exo  = self.enc_embedding(exo, x_mark_enc) if exo is not None else None

        # --- Cross Attention ---
        if enc_exo is not None:
            enc_endo = self.cross_attn(enc_endo, enc_exo)  # [B, N_endo, D]

        # --- Pass through Mamba encoder ---
        enc_out, attns = self.encoder(enc_endo, attn_mask=None)

        # --- Project back to predictions ---
        dec_out = self.projector(enc_out).permute(0, 2, 1)  # [B, pred_len, N_endo]

        # Expand predictions to match number of target channels
        dec_out = dec_out.repeat(1, 1, N)  # broadcast to N variables if needed

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, N]
    
    
if __name__ == '__main__':
    configs = Configs(
    seq_len=365,
    pred_len=1,
    d_model=1024,
    d_state=256,
    d_ff=2048,
    e_layers=5,
    dropout=0.1,
)
    # Create model and move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(configs).to(device)
    
    # Test data
    batch_size = 32
    x_enc = torch.randn(batch_size, configs.seq_len, 40).to(device)  # [32, 365, 40]
    x_mark_enc = torch.randn(batch_size, configs.seq_len, 7).to(device)  # Time features
    x_dec = torch.randn(batch_size, configs.pred_len, configs.d_model).to(device)  # Decoder input
    x_mark_dec = torch.randn(batch_size, configs.pred_len, 7).to(device)  # Decoder time features
    
    # Forward propagation test
    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    print(f"Input shape: {x_enc.shape}")
    print(f"Output shape: {output.shape}")
    # print(f"Model structure:\n{model}")