"""
Is Mamba Effective for Time Series Forecasting?
"""
import torch
import torch.nn as nn
import os
import sys

from typing import Optional, Dict
from typing import Tuple
import torch.nn.functional as F


# Add project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from model_zoo.layers.Mamba_EncDec import Encoder, EncoderLayer
from model_zoo.layers.Embed import DataEmbedding_inverted
import datetime
from model_zoo.utils.plot_utils import save_fill_missing_plots
import numpy as np

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


class CrossFuseEndoExo(nn.Module):
    """
    endo: [B, T, D]               # 内生变量时间序列
    exo:  [B, N_exo, T, D]        # 外生变量时间序列
    endo_mask: [B, T] (可选)      # True=有效；用于屏蔽无效时间步
    exo_mask:  [B, N_exo, T] (可选)

    返回:
      fused:  [B, N_exo+1, T, D]  # 1=更新后的内生 + N_exo=外生（可选也更新）
      attn_endo2exo: [B, H, T, N_exo*T]   # 解释性用
      attn_exo2endo: [B, N_exo, H, T, T]  # 仅在 mode='exo_cross' 时返回，否则为 None
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1, mode: str = "exo_static"):
        super().__init__()
        assert mode in ("exo_static", "exo_cross")
        self.mode = mode
        self.mha_endo = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.norm1_endo = nn.LayerNorm(d_model)
        self.ffn_endo = nn.Sequential(
            nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model), nn.Dropout(dropout)
        )
        self.norm2_endo = nn.LayerNorm(d_model)

        if mode == "exo_cross":
            self.mha_exo  = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
            self.norm1_exo = nn.LayerNorm(d_model)
            self.ffn_exo = nn.Sequential(
                nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(4*d_model, d_model), nn.Dropout(dropout)
            )
            self.norm2_exo = nn.LayerNorm(d_model)

    def forward(self, endo, exo, endo_mask=None, exo_mask=None):
        B, T, D = endo.shape
        B2, N_exo, T2, D2 = exo.shape
        assert (B, T, D) == (B2, T2, D2), f"shape mismatch: endo {endo.shape}, exo {exo.shape}"

        # ---------- 内生 <- 外生：endo作为Q；外生(展平)作为K,V ----------
        exo_seq = exo.reshape(B, N_exo*T, D)               # [B, N_exo*T, D]
        key_padding_mask = None
        if exo_mask is not None:
            # key_padding_mask: True=屏蔽
            key_padding_mask = (~exo_mask).reshape(B, N_exo*T)

        attn_out, attn_endo2exo = self.mha_endo(
            query=endo, key=exo_seq, value=exo_seq,
            key_padding_mask=key_padding_mask, need_weights=True
        )  # attn_endo2exo: [B, H, T, N_exo*T]

        endo_upd = self.norm1_endo(endo + attn_out)
        endo_upd = self.norm2_endo(endo_upd + self.ffn_endo(endo_upd))   # [B, T, D]

        # ---------- 外生（可选）<- 内生：每个外生流分别以endo作K,V ----------
        attn_exo2endo = None
        if self.mode == "exo_cross":
            # 组合 batch 维处理 N_exo 条外生序列
            exo_flat = exo.reshape(B*N_exo, T, D)                # [B*N_exo, T, D]
            endo_rep = endo.unsqueeze(1).expand(B, N_exo, T, D).reshape(B*N_exo, T, D)

            # masks
            kpm = None
            if endo_mask is not None:
                kpm = (~endo_mask).repeat_interleave(N_exo, dim=0)  # [B*N_exo, T]

            attn_out_exo, attn_exo2endo = self.mha_exo(
                query=exo_flat, key=endo_rep, value=endo_rep,
                key_padding_mask=kpm, need_weights=True
            )  # attn_exo2endo: [B*N_exo, H, T, T]

            exo_upd = self.norm1_exo(exo_flat + attn_out_exo)
            exo_upd = self.norm2_exo(exo_upd + self.ffn_exo(exo_upd))
            exo_upd = exo_upd.reshape(B, N_exo, T, D)
        else:
            exo_upd = exo

        # ---------- 拼回 [B, N_exo+1, T, D] ----------
        fused = torch.cat([endo_upd.unsqueeze(1), exo_upd], dim=1)  # [B, N_exo+1, T, D]
        # 统一 attn_exo2endo 形状以便可视化
        if attn_exo2endo is not None:
            H = attn_exo2endo.shape[1]
            attn_exo2endo = attn_exo2endo.reshape(B, N_exo, H, T, T)

        return fused, attn_endo2exo, attn_exo2endo


class EndoWeightsExoGating(nn.Module):
    r"""
    用内生序列 endo 对外生 exo 进行“时间×变量”双重加权，并安全注入外生表示。
    - 支持 Tq != Tk（endo 的时间长度与 exo 的时间长度不同）
    - 从 endo->exo 的 cross-attn 得到注意力 A ∈ R^{B,H,Tq,Sk}（Sk 为实际 key 长度）
      将其复原为 A ∈ R^{B,H,Tq,N_exo,Tk_attn} 后：
        * β(b,t,·) = 时间权重 ∈ R^{Tk_attn}      （对变量维求和，再对多头均值）  => [B, Tq, Tk_attn]
        * γ(b,t,·) = 变量权重 ∈ R^{N_exo}        （对时间维求和，再对多头均值）  => [B, Tq, N_exo]
      然后：
        * 时间重混合：exo_mixed[n,t,:] = Σ_s β(t,s) * exo[n,s,:]  -> [B,N_exo,Tq,D]
        * 变量门控：  exo_weighted[n,t,:] = γ(t,n) * exo_mixed[n,t,:]
        * 残差门：    exo_upd = exo_mixed + α * (exo_weighted - exo_mixed)，α 可学习、初值 0
      （可选）也更新 endo：endo_upd = endo + α_e * attn_out

    Args:
        d_model:      特征维 D
        n_heads:      多头数
        dropout:      FFN / Attention dropout
        update_endo:  是否同时更新 endo（默认 False）
        causal:       是否使用因果遮罩（endo 查询不看未来的外生时间；默认 False）

    Inputs:
        endo:      [B, Tq, D]               # 内生变量（单变量）的时间序列
        exo:       [B, N_exo, Tk, D]        # 外生变量时间序列
        endo_mask: [B, Tq]   (bool, 可选)   # True=有效（当前未用于 padding，可按需扩展）
        exo_mask:  [B, N_exo, Tk] (bool, 可选)  # True=有效 → 会在 MHA 中取反作为 key_padding_mask

    Returns:
        fused:     [B, N_exo+1, Tq, D]      # 第 0 维是 endo（更新或原样），后面是 exo_upd
        stats:     dict:
                   - 'beta_TT':  [B, Tq, Tk_exo]    # 时间→时间权重（已对齐到 exo 的时间长度，每行归一化）
                   - 'gamma_TN': [B, Tq, N_exo]     # 每个时间对各变量的权重（每行归一化）
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        update_endo: bool = False,
        causal: bool = False,
    ):
        super().__init__()
        self.update_endo = update_endo
        self.causal = causal

        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
            dropout=dropout,
        )

        # 外生分支：门控残差 + LN + FFN
        self.alpha = nn.Parameter(torch.tensor(0.0))     # 稳定注入门（初值 0）
        self.norm_exo1 = nn.LayerNorm(d_model)
        self.ffn_exo = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm_exo2 = nn.LayerNorm(d_model)

        # （可选）内生分支更新
        if update_endo:
            self.alpha_endo = nn.Parameter(torch.tensor(0.0))
            self.norm_endo1 = nn.LayerNorm(d_model)
            self.ffn_endo = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_model, d_model),
                nn.Dropout(dropout),
            )
            self.norm_endo2 = nn.LayerNorm(d_model)

    @staticmethod
    def _safe_row_normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        """对最后一维做归一化，避免除零/NaN。"""
        denom = x.sum(dim=-1, keepdim=True).clamp_min(eps)
        x = x / denom
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _build_causal_mask_Tq_Tk(Tq: int, Tk: int, N_exo: int, device, dtype) -> torch.Tensor:
        """
        生成因果遮罩: [Tq, N_exo*Tk]，对每个 query 时刻 t，屏蔽所有 key 时刻 s>t。
        注意：若实际 Sk != N_exo*Tk，建议 causal=False（以免形状不匹配）。
        """
        q_idx = torch.arange(Tq, device=device).unsqueeze(1)  # [Tq,1]
        k_idx = torch.arange(Tk, device=device).unsqueeze(0)  # [1,Tk]
        base = (k_idx > q_idx)                                # [Tq,Tk] True=屏蔽未来
        mask = base.repeat(1, N_exo)                          # [Tq, N_exo*Tk]
        return mask

    def forward(
        self,
        endo: torch.Tensor,                     # [B, Tq, D]
        exo: torch.Tensor,                      # [B, N_exo, Tk, D]
        endo_mask: Optional[torch.Tensor] = None,  # [B, Tq] (bool)  当前未使用
        exo_mask: Optional[torch.Tensor] = None,   # [B, N_exo, Tk] (bool) True=有效
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        B, Tq, D = endo.shape
        B2, N_exo, Tk_exo, D2 = exo.shape
        assert (B, D) == (B2, D2), f"Batch/D mismatch: endo={endo.shape}, exo={exo.shape}"

        # --- key padding mask（True=屏蔽） ---
        key_padding_mask = None
        if exo_mask is not None:
            if exo_mask.dtype != torch.bool:
                exo_mask = exo_mask.bool()
            key_padding_mask = (~exo_mask).reshape(B, N_exo * Tk_exo)  # [B, N_exo*Tk_exo]

        # --- 因果遮罩（谨慎使用；若实际 Sk != N_exo*Tk_exo 可能不匹配） ---
        attn_mask = None
        if self.causal:
            attn_mask = self._build_causal_mask_Tq_Tk(
                Tq, Tk_exo, N_exo, endo.device, endo.dtype
            )  # [Tq, N_exo*Tk_exo]

        # --- cross-attn: Q=endo (Tq), K/V=exo_tok (N_exo*Tk_exo) ---
        exo_tok = exo.reshape(B, N_exo * Tk_exo, D)  # [B, N_exo*Tk_exo, D]
        attn_out, attn_w = self.mha(
            query=endo,
            key=exo_tok,
            value=exo_tok,
            key_padding_mask=key_padding_mask,   # True=屏蔽
            need_weights=True,
            average_attn_weights=False,          # 取每个头：[B,H,Tq,Sk]
            attn_mask=attn_mask,                 # 可能为 None；若提供需与 Sk 匹配
        )

        # 统一为 [B,H,Tq,Sk]
        if attn_w.dim() == 3:                    # 兜底：若被平均过头
            attn_w = attn_w.unsqueeze(1)         # [B,1,Tq,Sk]
        B_, H, Tq2, Sk = attn_w.shape
        assert B_ == B and Tq2 == Tq, f"attn_w shape={attn_w.shape}, expect B={B} Tq={Tq}"

        # === 关键：不假设 Sk == N_exo*Tk_exo，动态反推 Tk_attn ===
        assert Sk % N_exo == 0, f"attn_w last dim {Sk} not divisible by N_exo={N_exo}"
        Tk_attn = Sk // N_exo                      # 注意力里“每个变量”的时间长度
        # 复原为 [B,H,Tq,N_exo,Tk_attn]
        attn_w = attn_w.view(B, H, Tq, N_exo, Tk_attn)

        # --- 从注意力得到 β (时间×时间) 与 γ (时间×变量) ---
        beta  = attn_w.mean(dim=1).sum(dim=3)      # [B, Tq, Tk_attn]
        beta  = self._safe_row_normalize(beta)

        gamma = attn_w.mean(dim=1).sum(dim=-1)     # [B, Tq, N_exo]
        gamma = self._safe_row_normalize(gamma)

        # === 对齐：把 beta 的 Tk_attn → Tk_exo（若不同） ===
        if Tk_attn != Tk_exo:
            beta = F.interpolate(
                beta.transpose(1, 2),              # [B, Tk_attn, Tq]
                size=Tk_exo, mode='linear', align_corners=False
            ).transpose(1, 2)                       # [B, Tq, Tk_exo]

        # --- 时间重混合: exo_mixed = Σ_s β(t,s) * exo[:, :, s, :] ---
        # 兜底校准：若仍存在 s 维不一致（数值问题或上游形状变动），做截断/填充以严格对齐
        S_exo = exo.size(2)
        S_beta = beta.size(-1)
        if S_beta != S_exo:
            if S_beta > S_exo:
                beta = beta[..., :S_exo]
            else:
                beta = F.pad(beta, (0, S_exo - S_beta))  # 在最后一维右侧补零
        # exo: [B, N_exo, Tk_exo, D], beta: [B, Tq, Tk_exo]  -> [B, N_exo, Tq, D]
        exo_mixed = torch.einsum('b n s d, b t s -> b n t d', exo, beta)

        # --- 变量门控: 按 γ(t,n) 缩放各变量 ---
        gamma_bnt = gamma.permute(0, 2, 1).unsqueeze(-1)   # [B, N_exo, Tq, 1]
        exo_weighted = exo_mixed * gamma_bnt               # [B, N_exo, Tq, D]

        # --- 稳定残差注入（以 exo_mixed 为基线，避免时间维不一致） ---
        exo_upd = exo_mixed + self.alpha * (exo_weighted - exo_mixed)  # [B,N_exo,Tq,D]
        exo_upd = self.norm_exo1(exo_upd)
        exo_upd = self.norm_exo2(exo_upd + self.ffn_exo(exo_upd))

        # --- （可选）更新内生分支 ---
        if self.update_endo:
            endo_upd = endo + self.alpha_endo * attn_out               # [B,Tq,D]
            endo_upd = self.norm_endo1(endo_upd)
            endo_upd = self.norm_endo2(endo_upd + self.ffn_endo(endo_upd))
        else:
            endo_upd = endo

        fused = torch.cat([endo_upd.unsqueeze(1), exo_upd], dim=1)     # [B, N_exo+1, Tq, D]
        stats = {"beta_TT": beta, "gamma_TN": gamma}
        return fused, stats



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
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            time_feat_dim=7,
            dropout=configs.dropout,
        )
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
        
        # self.cross_attn = EndoWeightsExoGating(d_model=128, n_heads=4, dropout=configs.dropout, update_endo=False)
        self.cross_attn = EndoExoAttention(d_model=configs.d_model, n_heads=4, dropout=configs.dropout)
    def forecast(self, x_enc, x_mark_enc):


        _, _, N = x_enc.shape # B L N
        
        # x_enc: (B, L, N)
        # 1) 最近邻插值填补缺失值（以-9999为缺失标记）：
        #    - 若左右最近有效点都存在，取二者均值
        #    - 若只存在一侧，取该侧值
        #    - 若整条序列无有效点，则回退为0
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
        # smooth modis only
        # x_enc[:, :, 0:22] = x_before_smooth[:, :, 0:22]
        
        # without smooth firms
        # x_enc[:, :, 0] = x_before_smooth[:, :, 0]
        
        # w/o modis data but with lai
        lai = x_before_smooth[:, :, -1]
        x_enc = x_enc[:, :, 0:21]
        x_enc = torch.cat([x_enc, lai.unsqueeze(-1)], dim=-1)
        x_enc[:, :, 0] = x_before_smooth[:, :, 0]
        x_enc[:, :, 19:21] = x_before_smooth[:, :, 19:21]
        
        # 3) 按时间步应用线性递减权重（从1到0，长度为L）
        L = x_enc.size(1)
        # 从1到0，长度为L
        weights = torch.linspace(1.0, 0.0, steps=L, device=x_enc.device, dtype=x_enc.dtype).view(1, L, 1)
        # x_enc = x_enc * weights.flip(dims=[1])

        # 4) 对第一个变量（此处按 x_enc[:, :, 0]）进行高斯扩散，将稀疏事件转为高斯分布
        x_before_gaussian = x_enc.clone()
        x_firms = x_enc[:, :, 0]  # [B, L]
        x_firms_diffused = self.gaussian_diffuse_1d(x_firms, sigma=7.0)
        x_enc[:, :, 0] = x_firms_diffused
        
        # save_fill_missing_plots(x_before_gaussian, x_enc, save_path=os.path.join(parent_dir, "outputs", "gaussian_compare_s_mamba_4d.png"))
        
        # 可视化并保存每个特征（40个特征）的平滑前后曲线
        # save_fill_missing_plots(x_before_smooth, x_enc, save_path=os.path.join(parent_dir, "outputs", "smooth_compare_s_mamba_4d.png"))
        # 可视化并保存每个特征（40个特征）的填补前后曲线
        # save_fill_missing_plots(x_enc, x_enc, save_path=os.path.join(parent_dir, "outputs", "fill_missing_s_mamba_4d.png"))
        
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> (x_tok [B,N,T,D], m_tok [B,N,T])
        # weights = weights.flip(dims=[1]).repeat(x_enc.shape[0], 1, 1)
        # x_mark_enc = torch.cat([x_mark_enc, weights], dim=-1)         
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # B N L//7 28 -> 交叉注意力融合
        # cross_attn 返回 fused: [B, N_exo+1, T, D], stats: 字典
        # fused, attn = self.cross_attn(enc_out[:, 0, :, :], enc_out[:, 1:, :, :], None, None)
        # enc_out = fused   # 已是 [B, N_exo+1, T, D]
        
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates


        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, L, D] where L=10 represents data from the previous 10 days
        # target_date: [B] list of date strings in yyyymmdd format
        
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
    def gaussian_diffuse_1d(
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