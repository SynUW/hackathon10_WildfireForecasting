"""
TimeXer-style Global Token + Patchify (Mamba-keeping, OOM-safe)
- Endo = first feature of N
- Exo  = other N-1 features
- Patchify endo -> tokens + learnable [GLOBAL] token
- Cross-attn: [GLOBAL] (query)  x  Exo tokens (key/value)
- Token-axis encoder keeps Mamba
- Exo path: chunked + optional lite Conv1d instead of Mamba
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from contextlib import nullcontext


# -----------------------------
# Configs
# -----------------------------
class Configs:
    def __init__(
        self,
        seq_len=10,
        pred_len=7,
        d_model=256,
        d_state=256,
        d_ff=1024,
        e_layers=3,
        dropout=0.1,
        activation='relu',
        output_attention=False,
        use_norm=False,
        embed='timeF',
        freq='d',
        # TimeXer-style
        patch_len=7,
        patch_stride=7,         # =patch_len -> non-overlap; <patch_len -> overlap
        n_heads=4,
        # OOM-safe knobs
        exo_use_mamba=True,    # True to use Mamba on exo; False uses lite Conv1d
        exo_chunk=16,           # process exo variates in chunks to limit B*V
        d_model_exo=None,       # inner feature for exo path; default=min(64, d_model//4)
        amp_dtype='bf16',       # 'bf16' | 'fp16' | None
        checkpoint_mamba=False  # gradient checkpointing for token-encoder Mamba
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.d_state = d_state
        self.d_ff = d_ff
        self.e_layers = e_layers
        self.dropout = dropout
        self.activation = activation
        self.output_attention = output_attention
        self.use_norm = use_norm
        self.embed = embed
        self.freq = freq

        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.n_heads = n_heads

        self.exo_use_mamba = exo_use_mamba
        self.exo_chunk = exo_chunk
        self.d_model_exo = d_model_exo if d_model_exo is not None else max(16, min(64, d_model // 4))
        self.amp_dtype = 'bf16'
        self.checkpoint_mamba = checkpoint_mamba


# -----------------------------
# Utilities
# -----------------------------
def overlap_patchify_1d(x_endo, patch_len, stride):
    """
    x_endo: [B, L]  (single endogenous series)
    return: patches [B, P, patch_len]
    """
    B, L = x_endo.shape
    if L < patch_len:
        pad = patch_len - L
        x_endo = F.pad(x_endo, (0, pad), mode='constant', value=0.)
        L = patch_len
    x = x_endo.unsqueeze(1)  # [B, 1, L]
    patches = x.unfold(dimension=2, size=patch_len, step=stride)  # [B, 1, P, patch_len]
    patches = patches.squeeze(1)  # [B, P, patch_len]
    return patches


# -----------------------------
# Token Builders
# -----------------------------
class EndoPatchEmbedding(nn.Module):
    """
    Turn endogenous series into patch tokens + a learnable [GLOBAL] token.
    """
    def __init__(self, patch_len, d_model, add_pos=True):
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model
        self.proj = nn.Linear(patch_len, d_model)
        self.add_pos = add_pos
        self.register_parameter("pos", None)  # lazy init
        self.global_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.global_token, std=0.02)

    def forward(self, x_endo, patch_stride):
        """
        x_endo: [B, L]  (single endo variable)
        return tokens: [B, 1+P, d_model]   (prepend [GLOBAL])
        """
        patches = overlap_patchify_1d(x_endo, self.patch_len, patch_stride)  # [B, P, patch_len]
        tok = self.proj(patches)  # [B, P, d]
        if self.add_pos:
            B, P, D = tok.shape
            need_len = max(P, 1)
            # 若未初始化或长度/设备/精度不匹配，则重建
            if (self.pos is None) or (self.pos.shape[1] < need_len) or (self.pos.shape[2] != D) or (self.pos.device != tok.device) or (self.pos.dtype != tok.dtype):
                pos = torch.zeros((1, need_len, D), device=tok.device, dtype=tok.dtype)
                nn.init.trunc_normal_(pos, std=0.02)
                # 用 Parameter 保存，便于被保存/加载；不会参与优化器更新（不在optimizer里）
                self.pos = nn.Parameter(pos, requires_grad=True)
            tok = tok + self.pos[:, :P, :]

        B = x_endo.size(0)
        g = self.global_token.expand(B, -1, -1).to(tok.device, tok.dtype)  # [B, 1, d]
        tokens = torch.cat([g, tok], dim=1)  # [B, 1+P, d]
        return tokens


class ExoVariateEmbeddingLite(nn.Module):
    """
    Lite exo encoder: Conv1d + GELU + pool  -> per-variates token.
    Greatly reduces memory vs Mamba on exo.
    """
    def __init__(self, seq_len, d_model_out, d_model_exo, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(1, d_model_exo)
        self.conv = nn.Conv1d(d_model_exo, d_model_exo, kernel_size=3, padding=1, groups=d_model_exo)  # depthwise
        self.pw = nn.Conv1d(d_model_exo, d_model_exo, kernel_size=1)  # pointwise
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model_exo, d_model_out)

    def forward(self, x_exo):
        """
        x_exo: [B, L, V]
        return: [B, V, d_model_out]
        """
        B, L, V = x_exo.shape
        x = x_exo.permute(0, 2, 1).reshape(B * V, L, 1)   # [B*V, L, 1]
        x = self.in_proj(x)                               # [B*V, L, C]
        x = x.transpose(1, 2)                             # [B*V, C, L]
        x = self.pw(F.gelu(self.conv(x)))                 # [B*V, C, L]
        x = x.mean(dim=2)                                 # [B*V, C]
        x = self.dropout(x)
        x = self.out_proj(x)                              # [B*V, d_out]
        x = x.reshape(B, V, -1)
        return x


class ExoVariateEmbeddingMamba(nn.Module):
    """
    Original exo encoder with Mamba (heavier). Now chunked to avoid OOM.
    """
    def __init__(self, seq_len, d_model_out, d_model_exo, d_state=64, d_conv=2, expand=1, dropout=0.1, chunk=16):
        super().__init__()
        self.in_proj = nn.Linear(1, d_model_exo)
        self.temporal_mamba = Mamba(d_model=d_model_exo, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model_exo, d_model_out)
        self.chunk = max(1, chunk)

    def forward(self, x_exo):
        """
        x_exo: [B, L, V]
        return: [B, V, d_model_out]
        """
        B, L, V = x_exo.shape
        outs = []
        for s in range(0, V, self.chunk):
            e = min(V, s + self.chunk)
            xs = x_exo[:, :, s:e]                        # [B, L, v]
            v = e - s
            x = xs.permute(0, 2, 1).reshape(B * v, L, 1) # [B*v, L, 1]
            x = self.in_proj(x)                          # [B*v, L, c]
            x = self.temporal_mamba(x)                   # [B*v, L, c]
            x = self.dropout(x)
            x = x.mean(dim=1)                            # [B*v, c]
            x = self.out_proj(x)                         # [B*v, d_out]
            x = x.reshape(B, v, -1)                      # [B, v, d_out]
            outs.append(x)
        return torch.cat(outs, dim=1) if outs else x_exo.new_zeros(B, 0, self.out_proj.out_features)


# -----------------------------
# Fusion: Global ↔ Exo cross-attention
# -----------------------------
class GlobalExoFusion(nn.Module):
    """
    Cross-attention: Query = [GLOBAL], Key/Value = Exo tokens.
    """
    def __init__(self, d_model, n_heads=4, dropout=0.1, return_weights=False):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ln = nn.LayerNorm(d_model)
        self.return_weights = return_weights

    def forward(self, global_token, exo_tokens):
        """
        global_token: [B, 1, d]
        exo_tokens:   [B, V, d]
        """
        g0 = global_token
        g, w = self.mha(query=global_token, key=exo_tokens, value=exo_tokens, need_weights=self.return_weights)
        g = self.ln(g + g0)
        return (g, w) if self.return_weights else g


# -----------------------------
# Encoder Block (token-wise) with Mamba + FFN
# -----------------------------
class TokenEncoderLayer(nn.Module):
    def __init__(self, d_model, d_state=256, d_conv=2, expand=1, d_ff=1024, dropout=0.1, activation='relu'):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation != 'relu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x_tokens):
        y = self.mamba(x_tokens)          # [B, T, d]
        x = self.ln1(x_tokens + y)
        y = self.ffn(x)
        x = self.ln2(x + y)
        return x


class TokenEncoder(nn.Module):
    def __init__(self, d_model, e_layers=3, d_state=256, d_conv=2, expand=1, d_ff=1024, dropout=0.1,
                 activation='relu', checkpoint_mamba=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TokenEncoderLayer(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                d_ff=d_ff, dropout=dropout, activation=activation
            ) for _ in range(e_layers)
        ])
        self.checkpoint = checkpoint_mamba

    def forward(self, x_tokens):
        if not self.checkpoint:
            for blk in self.layers:
                x_tokens = blk(x_tokens)
            return x_tokens
        # gradient checkpointing (saves memory at cost of compute)
        def _wrap(blk):
            def fn(inp):
                return blk(inp)
            return fn
        for blk in self.layers:
            x_tokens = torch.utils.checkpoint.checkpoint(_wrap(blk), x_tokens, use_reentrant=False)
        return x_tokens


# -----------------------------
# Full Model
# -----------------------------
class Model(nn.Module):
    """
    TimeXer-style global token + patchify for endo; exo fusion; keep Mamba backbone.
    Output [B, pred_len, N]
    """
    def __init__(self, cfg: Configs):
        super().__init__()
        self.cfg = cfg
        self.seq_len = cfg.seq_len
        self.pred_len = cfg.pred_len
        self.d_model = cfg.d_model

        # AMP autocast dtype
        # if cfg.amp_dtype == 'bf16' and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        self.amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        # elif cfg.amp_dtype == 'fp16' and torch.cuda.is_available():
        #     self.amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
        # else:
        #     self.amp_ctx = nullcontext()

        # Endo tokens
        self.endo_tok = EndoPatchEmbedding(patch_len=7, d_model=cfg.d_model)
        self.exo_use_mamba = True
        # Exo tokens (chunked)
        if self.exo_use_mamba == True:
            self.exo_tok = ExoVariateEmbeddingMamba(
                seq_len=cfg.seq_len, d_model_out=cfg.d_model, d_model_exo=None,
                d_state=min(128, cfg.d_state), d_conv=2, expand=1, dropout=cfg.dropout, chunk=16
            )
        else:
            self.exo_tok = ExoVariateEmbeddingLite(
                seq_len=cfg.seq_len, d_model_out=cfg.d_model, d_model_exo=None, dropout=cfg.dropout
            )

        # Global <-> Exo fusion
        self.global_exo = GlobalExoFusion(d_model=cfg.d_model, n_heads=cfg.n_heads, dropout=cfg.dropout,
                                          return_weights=False)

        # Token encoder over [GLOBAL] + endo patches
        self.encoder = TokenEncoder(
            d_model=cfg.d_model,
            e_layers=cfg.e_layers,
            d_state=cfg.d_state,
            d_conv=2,
            expand=1,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            activation=cfg.activation,
            checkpoint_mamba=cfg.checkpoint_mamba
        )

        # Heads
        self.endo_pool = nn.AdaptiveAvgPool1d(1)
        self.endo_head = nn.Linear(cfg.d_model, cfg.pred_len)
        self.exo_head = nn.Linear(cfg.d_model, cfg.pred_len)

    @staticmethod
    def _mask_and_fill(x):
        # replace -9999 -> 0
        return torch.where(x == -9999, torch.zeros_like(x), x)

    def forecast(self, x_enc, x_mark_enc=None):
        """
        x_enc: [B, L, N]  (first variable is endogenous)
        return: [B, pred_len, N]
        """
        with self.amp_ctx:
            B, L, N = x_enc.shape
            assert N >= 1, "Need at least 1 variable; first is endogenous."
            x_enc = self._mask_and_fill(x_enc)

            # split
            endo = x_enc[:, :, 0]               # [B, L]
            exo  = x_enc[:, :, 1:] if N > 1 else None  # [B, L, V]

            # endo -> [GLOBAL]+patches
            endo_tokens = self.endo_tok(endo, patch_stride=self.cfg.patch_stride)  # [B, 1+P, d]
            global_tok  = endo_tokens[:, :1, :]
            endo_patches = endo_tokens[:, 1:, :]

            # exo -> tokens
            if exo is not None and exo.size(-1) > 0:
                exo_tokens = self.exo_tok(exo)              # [B, V, d]
                # fusion
                global_tok = self.global_exo(global_tok, exo_tokens)  # [B, 1, d]
            else:
                exo_tokens = None

            # encode [GLOBAL] + patches
            tokens = torch.cat([global_tok, endo_patches], dim=1)     # [B, 1+P, d]
            tokens = self.encoder(tokens)                              # [B, 1+P, d]

            # heads
            pooled = self.endo_pool(tokens.transpose(1, 2)).squeeze(-1)  # [B, d]
            endo_pred = self.endo_head(pooled)                            # [B, S]

            if exo_tokens is not None:
                exo_pred = self.exo_head(exo_tokens).permute(0, 2, 1)     # [B, S, V]
                out = torch.cat([endo_pred.unsqueeze(-1), exo_pred], dim=-1)  # [B, S, N]
            else:
                out = endo_pred.unsqueeze(-1)                             # [B, S, 1]
        return out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        out = self.forecast(x_enc, x_mark_enc)
        return out[:, -self.cfg.pred_len:, :]
