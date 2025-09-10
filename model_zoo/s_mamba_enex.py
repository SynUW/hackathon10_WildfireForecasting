"""
Is Mamba Effective for Time Series Forecasting?
"""
from re import X
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
import torch.nn.functional as F

from mamba_ssm import Mamba
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import datetime
import numpy as np

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
            s = int(indices[i].item())
            e = int(indices[i + 1].item())
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
        
        if 1 == 1:            
            x_before_fill = x_enc.clone()
            x_enc = self._fill_missing_nearest_mean(x_enc, invalid_value=-9999, zero_invalid_from_feature_idx=21)
            # 2) 平滑：Hampel → Savitzky–Golay，既去尖刺又保留拐点
            x_before_smooth = x_enc.clone()
            x_enc = self._smooth_hampel_sg(
                x_enc,
                window_length=21,  # 21  # 日尺度：建议21–31
                polyorder=3,
                hampel_window=7,
                hampel_nsig=3.0, # 3.0 
            )
            x_enc[:, :, 0] = x_before_smooth[:, :, 0]
            
            # 4) 对第一个变量（此处按 x_enc[:, :, 0]）进行高斯扩散，将稀疏事件转为高斯分布
            x_before_gaussian = x_enc.clone()
            x_firms = x_enc[:, :, 0]  # [B, L]
            x_firms_diffused = self._gaussian_diffuse_1d(x_firms, sigma=7.0)
            x_enc[:, :, 0] = x_firms_diffused
            
            # --- Split endogenous (Q) vs exogenous (K,V) ---
            endo = x_enc[:, :, 0:1]      # FIRMS or first channel
            exo  = x_enc[:, :, 1:] if N > 1 else None

            # Embeddings
            enc_endo = self.enc_embedding(endo, x_mark_enc)   # [B, N_endo, D]
            enc_exo  = self.enc_embedding(exo, x_mark_enc) if exo is not None else None

            # --- Cross Attention ---
            enc_endo = self.cross_attn(enc_endo, enc_exo)  # [B, N_endo, D]
        else:
            x_enc = torch.where(x_enc == -9999, torch.full_like(x_enc, 0.5), x_enc)
            # --- Pass through Mamba encoder ---
            enc_endo = self.enc_embedding(x_enc, x_mark_enc)   # [B, N_endo, D]
        enc_out, attns = self.encoder(enc_endo, attn_mask=None)

        # --- Project back to predictions ---
        dec_out = self.projector(enc_out).permute(0, 2, 1)  # [B, pred_len, N_endo]

        # Expand predictions to match number of target channels
        dec_out = dec_out.repeat(1, 1, N)  # broadcast to N variables if needed

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, N]
    
    @staticmethod
    def _fill_missing_nearest_mean(x: torch.Tensor, invalid_value: float = -9999,
                                   zero_invalid_from_feature_idx: int = 21,
                                   zero_tol: float = 1e-8) -> torch.Tensor:
        """
        最近邻插值：对每个像素时间序列（按最后一维为变量，中间为时间）用最近的左右有效值的均值填补缺失。
        规则：
          - 同时存在前后最近有效值 -> 取均值
          - 仅一侧存在 -> 取该侧值
          - 两侧都不存在（全缺失）-> 置0
        输入: x [B, L, N]
        输出: 同形状张量
        """
        B, L, N = x.shape
        device = x.device
        dtype = x.dtype

        # [B,N,L]
        x_bnl = x.permute(0, 2, 1)
        # 对于特征索引 >= zero_invalid_from_feature_idx，将 |x|<zero_tol 视为无效
        var_idx = torch.arange(N, device=device).view(1, N, 1)  # [1,N,1]
        zero_invalid_mask = (var_idx >= zero_invalid_from_feature_idx)
        is_zero_invalid = (x_bnl.abs() < zero_tol) & zero_invalid_mask
        valid = (x_bnl != invalid_value) & (~is_zero_invalid)

        # 前向最近有效索引（<= t）
        t_idx = torch.arange(L, device=device).view(1, 1, L).expand(B, N, L)
        prev_idx = torch.where(valid, t_idx, torch.full_like(t_idx, -1))
        prev_idx = torch.cummax(prev_idx, dim=2).values  # [B,N,L] 每时刻最近的左侧有效位置（含自身）
        prev_exists = prev_idx >= 0
        prev_idx_safe = torch.clamp(prev_idx, min=0)
        prev_vals = torch.gather(x_bnl, 2, prev_idx_safe)

        # 后向最近有效索引（>= t）：在反向序列上做cummax再映回
        x_rev_valid = valid.flip(dims=[2])
        t_rev = torch.arange(L, device=device).view(1, 1, L).expand(B, N, L)
        next_rev_idx = torch.where(x_rev_valid, t_rev, torch.full_like(t_rev, -1))
        next_rev_idx = torch.cummax(next_rev_idx, dim=2).values  # 反向的“最近左侧”
        next_exists = next_rev_idx >= 0
        # 映射回正向索引：idx_fwd = L-1 - idx_rev
        next_idx = (L - 1) - next_rev_idx
        next_idx_safe = torch.where(next_exists, next_idx, torch.zeros_like(next_idx))
        next_vals = torch.gather(x_bnl, 2, next_idx_safe)

        # 计算每个样本-特征的全局中位数（忽略无效）作为兜底
        nan = torch.tensor(float('nan'), device=device, dtype=dtype)
        vals_masked = torch.where(valid, x_bnl, nan)
        # torch.nanmedian 返回 (values, indices)
        feat_median = torch.nanmedian(vals_masked, dim=2).values  # [B,N]
        # 若全为 NaN，则用0回退
        feat_median = torch.where(torch.isnan(feat_median), torch.zeros_like(feat_median), feat_median)
        feat_median_exp = feat_median.unsqueeze(-1).expand(B, N, L)

        # 组合填充值：
        # - 两侧都有 -> 均值
        # - 仅一侧 -> 该侧
        # - 两侧都无 -> 全局中位数
        both = prev_exists & next_exists
        neither = (~prev_exists) & (~next_exists)
        fill_vals = torch.where(both, 0.5 * (prev_vals + next_vals), prev_vals)
        fill_vals = torch.where((~both) & next_exists, next_vals, fill_vals)
        fill_vals = torch.where(neither, feat_median_exp, fill_vals)

        # 写回缺失位置
        x_filled = torch.where(valid, x_bnl, fill_vals)
        return x_filled.permute(0, 2, 1)

    @staticmethod
    def _smooth_hampel_sg(x: torch.Tensor,
                      window_length: int = 21,
                      polyorder: int = 3,
                      hampel_window: int = 7,
                      hampel_nsig: float = 3.0) -> torch.Tensor:
        """
        先做 Hampel 去尖刺，再做 Savitzky–Golay 平滑（GPU 友好，torch 实现）
        x: [B, L, N]  ->  返回同形状 [B, L, N]（保持 device/dtype）
        注意：SG 这里用 reflect 边界填充，避免零填充带来的首尾偏移。
        """
        if x.ndim != 3:
            raise ValueError("x must be [B, L, N]")
        B, L, N = x.shape
        if L <= 2:
            return x

        device, dtype = x.device, x.dtype

        # ---- helpers ----
        def _odd_within(v: int, max_len: int) -> int:
            """确保奇数，且不超过序列长度（至少为3）"""
            v = int(v)
            if v % 2 == 0:
                v += 1
            v = min(v, max_len if (max_len % 2 == 1) else max_len - 1)
            return max(3, v)

        # ----------------- 参数整理 -----------------
        wl   = _odd_within(window_length, L)
        poly = max(1, min(int(polyorder), wl - 1))
        hw   = max(1, int(hampel_window))
        nsig = float(hampel_nsig)

        # ----------------- 1) Hampel 去尖刺 -----------------
        # 到 [B, N, L]
        x_bnl = x.permute(0, 2, 1).contiguous()  # [B,N,L]

        win = hw * 2 + 1
        if win > L:
            win = _odd_within(win, L)

        # 边缘复制填充，再滑窗
        x_pad = F.pad(x_bnl, (hw, hw), mode='replicate')        # [B,N,L+2*hw]
        x_win = x_pad.unfold(dimension=2, size=win, step=1)     # [B,N,L,win]

        med = x_win.median(dim=-1).values                       # [B,N,L]
        mad = (x_win - med.unsqueeze(-1)).abs().median(dim=-1).values  # [B,N,L]
        thr = 1.4826 * mad * nsig
        outlier = (x_bnl - med).abs() > thr
        x_hampel = torch.where(outlier, med, x_bnl)             # [B,N,L]

        # ----------------- 2) SG 平滑（反射填充 + depthwise conv） -----------------
        half = wl // 2

        # 设计矩阵 A 与中心权重 h（在 CPU 上一次性求）
        # A: [wl, poly+1]，行是 -half..half 的幂
        A = np.vander(np.arange(-half, half + 1, dtype=np.float64),
                    N=poly + 1, increasing=True)              # [wl, poly+1]
        H = A @ np.linalg.pinv(A)                                # [wl, wl]
        h = H[half, :].astype(np.float32)                        # 中心行 -> 卷积核
        h_t = torch.from_numpy(h).to(device=device, dtype=dtype) # [wl]

        # depthwise kernel: [N,1,wl]
        weight = h_t.view(1, 1, wl).repeat(N, 1, 1)

        # 用 reflect 显式填充，再 conv1d（padding=0，避免 zero-pad 端点偏移）
        x_reflect = F.pad(x_hampel, (half, half), mode='reflect')  # [B,N,L+2*half]
        x_sg = F.conv1d(x_reflect.contiguous(), weight, bias=None,
                        stride=1, padding=0, groups=N)             # [B,N,L]

        # 回到 [B, L, N]
        return x_sg.permute(0, 2, 1).contiguous()

    @staticmethod
    def _gaussian_diffuse_1d(
        x_bL: torch.Tensor,
        sigma: float = 3.0,
        mode: str = "sum",          # "sum": 面积=1；"max": 峰值=1
        padding: str = "reflect",   # "reflect" | "constant" | "replicate"
        normalize_series: bool = False
    ) -> torch.Tensor:
        """
        稀疏事件的一维高斯扩散（卷积）
        x_bL: [B, L]，sigma 用“时间步”为单位
        """
        if sigma <= 0:
            return x_bL

        B, L = x_bL.shape
        device, dtype = x_bL.device, x_bL.dtype

        # ---- 核参数：半径≤L//2，长度为奇数 ----
        radius = int(max(1, round(3.0 * float(sigma))))
        radius = min(radius, max(1, L // 2))
        k = 2 * radius + 1

        t = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel = torch.exp(-0.5 * (t / sigma) ** 2)

        if mode == "sum":
            # 面积守恒（单脉冲扩散后峰值 < 1，但总能量不变）
            kernel = kernel / kernel.sum().clamp_min(1e-8)
        elif mode == "max":
            # 峰值守恒（单脉冲扩散后峰值 = 1，面积会随 sigma 变大）
            kernel = kernel / kernel.max().clamp_min(1e-8)
        else:
            raise ValueError("mode must be 'sum' or 'max'")

        x = x_bL.unsqueeze(1)                 # [B,1,L]
        weight = kernel.view(1, 1, k)         # [1,1,K]

        # 边界填充
        if padding == "constant":
            x_pad = F.pad(x, (radius, radius), mode="constant", value=0.0)
        else:
            x_pad = F.pad(x, (radius, radius), mode=padding)

        y = F.conv1d(x_pad, weight)           # [B,1,L]
        y = y.squeeze(1)                      # [B,L]

        if normalize_series:
            # 可选：对每条时间序列做 0-1 归一（只用于可视化）
            y_min = y.amin(dim=1, keepdim=True)
            y_max = y.amax(dim=1, keepdim=True)
            y = (y - y_min) / (y_max - y_min).clamp_min(1e-8)

        return y
    
    
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