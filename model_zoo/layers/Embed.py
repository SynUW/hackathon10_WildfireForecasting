import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
from typing import Literal, Optional


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 7, 'b': 3}  # 'd' for daily should be 7 (year, month, day, weekday, etc.)
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark == None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)

# 最经典的embedding
class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, time_feat_dim=7):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # [B, L, N] -> [B, N, L]
        x = x.permute(0, 2, 1)

        # x: [Batch Variate Time]
        if x_mark is None:

            x = self.value_embedding(x)
        else:
            # [B, N, L] -> [B, N+7, D] time_feat_dim=7 and each variable is embedded to a token
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
            # x = torch.cat([x, x_mark.permute(0, 2, 1)], dim=1)
        return x # self.dropout(x)
  
# 这个embedding似乎有逻辑错误。对于x_mark，不应该将时间特征数量映射到d_model，而是应该将时间序列映射到d_model
# 也就是用时间特征生成一个全局的（对所有变量共享的）时间摘要，在“变量作为 token”的空间里去调制每个变量的 token
# class DataEmbedding_inverted(nn.Module):
#     # modifications:
#     # original: cat(x, x_mark) and then embed
#     # modified: embed x and x_mark separately and then cat

#     def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, time_feat_dim=6):
#         super(DataEmbedding_inverted, self).__init__()
#         self.value_embedding = nn.Linear(c_in, d_model)
#         self.value_embedding_1 = nn.Linear(1, d_model//(c_in//1), bias=False)
#         self.value_embedding_2 = nn.Linear(990, d_model, bias=False)
#         self.time_embedding = nn.Linear(time_feat_dim, d_model)
#         self.temporal_score = nn.Linear(d_model, 1, bias=False)  # 对每个时间步的 d_model 向量打分
#         self.final_proj = nn.Linear(d_model, d_model)
#         self.dropout = nn.Dropout(p=dropout)
        
#         self.token_size = None

#     def forward(self, x, x_mark):
#         # x: [B, L, N]  (batch, seq_len, num_vars)
#         # x_mark: [B, L, T] (batch, seq_len, time_feat_dim)
#         x = x.permute(0, 2, 1)  # [B, N, L]
#         B, N, L = x.shape
#         # each variable is embedded to a token
#         if self.token_size is not None:
#             # tokenlize time series to tokens
#             x = x.unfold(dimension=-1, size=8, step=8)
#             x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
#             value_emb = self.value_embedding_1(x)  # [B*N, L/16, d_model//16]
        
#             value_emb = value_emb.reshape(B, N, -1)  # [B, N, d_model]
#             value_emb = self.dropout(value_emb)
#             value_emb = self.value_embedding_2(value_emb)
#         else:
#             value_emb = self.value_embedding(x)
        
#         if x_mark is not None:
#             # time_feat_dim is 6, so each variable has a time embedding
#             time_emb = self.time_embedding(x_mark).permute(0, 2, 1)  # [B, d_model, L]
#             time_emb = time_emb.unsqueeze(1).expand(-1, value_emb.shape[1], -1, -1)  # [B, N, d_model, L]
            
#             # original time embedding is the mean of all time steps
#             # time_emb = time_emb.mean(-1)  # [B, N, d_model]
            
#             # learnable time embedding
#             scores = self.temporal_score(time_emb.transpose(-2, -1))   # [B, N, L, 1]
#             alpha  = torch.softmax(scores.squeeze(-1), dim=-1)         # [B, N, L]
#             time_emb = (time_emb * alpha.unsqueeze(2)).sum(dim=-1)     # [B, N, d_model]
            
#             x = value_emb + time_emb
#             # x = torch.cat([value_emb, time_emb], dim=1)
#             # x = self.final_proj(x)
#         else:
#             x = value_emb
#         return self.dropout(x)


# class DataEmbedding_inverted(nn.Module):
#     """
#     Variable-as-token embedding with per-variable time context (输出: [B, N, d_model])
#     --------------------------------------------------------------------------------
#     输入:
#         x:      [B, L, N]  每个变量在 L 个时间步的数值
#         x_mark: [B, L, T]  每个时间步的时间特征 (如: 年/月/日/小时/工作日等); 可为 None
#     输出:
#         out:    [B, N, d_model]

#     设计要点:
#     - value 分支: 将每个变量的整段时间序列压缩为一个 token: [B,N,L] -> Linear(L->d) -> [B,N,d]
#       (与您现有做法一致, 需要固定 seq_len)
#     - time 分支: 先用时间特征做键/值, 再用变量的时间轨迹做查询, 生成"每变量"的时间注意力,
#       聚合得到 [B,N,d], 与 value 融合 (add/concat/film)
#     """

#     def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, time_feat_dim=6,
#                  use_layernorm=False, fuse='add'):
#         super().__init__()
#         self.seq_len = int(c_in)
#         self.d_model = int(d_model)
#         self.fuse = fuse
#         self.use_layernorm = use_layernorm

#         # -------- value 分支: 每变量把时间序列 L 压到 d_model --------
#         # 输入将会是 [B, N, L] -> [B, N, d_model]
#         self.value_proj = nn.Linear(self.seq_len, d_model, bias=True)

#         # -------- time 分支: 变量条件的时间注意力 --------
#         # 时间键/值（来自 x_mark）
#         self.time_key   = nn.Linear(time_feat_dim, 64, bias=True)   # 可调隐层维
#         self.time_value = nn.Linear(time_feat_dim, d_model, bias=True)

#         # 变量查询（来自 x 的数值轨迹; 让权重因变量不同而不同）
#         # x_per_var: [B,N,L,1] -> proj -> [B,N,L,64]
#         self.var_query  = nn.Linear(1, 64, bias=True)

#         # 融合头
#         if fuse == 'add':
#             self.final_proj = nn.Identity()
#         elif fuse == 'concat':
#             self.final_proj = nn.Linear(2 * d_model, d_model, bias=True)
#         elif fuse == 'film': 
#             self.gamma_proj = nn.Linear(d_model, d_model, bias=True)
#             self.beta_proj  = nn.Linear(d_model, d_model, bias=True)
#             self.final_proj = nn.Identity()

#         self.ln = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x: torch.Tensor, x_mark: Optional[torch.Tensor] = None) -> torch.Tensor:
#         """
#         x:      [B, L, N]
#         x_mark: [B, L, T] 或 None
#         return: [B, N, d_model]
#         """
#         B, L, N = x.shape
#         assert L == self.seq_len, f"seq_len mismatch: got L={L}, expected {self.seq_len}"

#         # ===== value 分支: [B, L, N] -> [B, N, L] -> Linear(L->d) -> [B, N, d]
#         xN = x.permute(0, 2, 1).contiguous()          # [B, N, L]
#         V  = self.value_proj(xN)                      # [B, N, d_model]

#         if x_mark is None:
#             out = V
#         else:
#             # ===== time 分支: 构造时间键/值 =====
#             # time_key/value: [B, L, *]
#             tk = self.time_key(x_mark)               # [B, L, 64]
#             tv = self.time_value(x_mark)             # [B, L, d_model]

#             # 变量条件查询（由 x 的时间轨迹提供）
#             # 先把 xN: [B,N,L] -> [B,N,L,1] 再投影
#             q  = self.var_query(xN.unsqueeze(-1))    # [B, N, L, 64]

#             # 计算 attention scores（逐时间步），形成每变量的权重
#             # 对齐 tk: [B,  1, L, 64] 与 q: [B, N, L, 64]
#             tk_b = tk.unsqueeze(1)                   # [B, 1, L, 64]
#             scores = (q * tk_b).sum(dim=-1)          # [B, N, L]  (点积)
#             alpha  = torch.softmax(scores, dim=-1)   # [B, N, L]

#             # 用权重聚合时间值 tv -> [B, N, d_model]
#             tv_b = tv.unsqueeze(1)                   # [B, 1, L, d]
#             time_ctx = torch.einsum('bnl, bnld -> bnd', alpha, tv_b)  # [B,N,d]

#             # ===== 融合 =====
#             if self.fuse == 'add':
#                 out = V + time_ctx                   # [B, N, d]
#             elif self.fuse == 'concat':
#                 out = torch.cat([V, time_ctx], dim=-1)   # [B, N, 2d]
#                 out = self.final_proj(out)               # [B, N, d]
#             else:  # 'film'
#                 gamma = self.gamma_proj(time_ctx)    # [B, N, d]
#                 beta  = self.beta_proj(time_ctx)     # [B, N, d]
#                 out = V * (1.0 + gamma) + beta       # [B, N, d]

#         out = self.ln(out)
#         out = self.dropout(out)                      # [B, N, d_model]time_feat_dim
#         return out


class DataEmbedding_inverted_new(nn.Module):
    # 不改变时间维度的长度，但是生成一个新的维度
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, time_feat_dim=6):
        super().__init__()
        d_model = 64
        # 数值映射：每个变量各自线性到 d
        self.value_proj = nn.Linear(1, d_model)   # 对每个变量的标量做投影
        # 时间 FiLM：生成 γ/β（逐时刻，不跨时相平均）
        self.time_mlp = nn.Sequential(
            nn.Linear(time_feat_dim, 64), nn.GELU(),
            nn.Linear(64, 2*d_model)  # -> [gamma | beta]
        )
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, x_mark):
        # x: [B, L, N], x_mark: [B, L, T_feat]
        B, L, N = x.shape
        # 按变量拆开做投影，保持时间维
        xv = x.reshape(B*L*N, 1)
        ve = self.value_proj(xv).reshape(B, L, N, -1)  # [B,L,N,d]

        # 时间 FiLM：逐时刻产生 γ/β
        gb = self.time_mlp(x_mark)                  # [B,L,2d]
        gamma, beta = gb.chunk(2, dim=-1)           # [B,L,d], [B,L,d]
        gamma = 1.0 + 0.1*torch.tanh(gamma)         # 稳定起步≈1

        # 广播到变量轴，逐时刻调制（不平均时间）
        h = gamma.unsqueeze(2) * ve + beta.unsqueeze(2)  # [B,L,N,d]
        h = self.ln(h)
        
        return self.dropout(h)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars

