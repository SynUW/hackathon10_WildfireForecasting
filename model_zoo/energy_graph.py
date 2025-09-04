# scalestf_minimal.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# 工具：按变量做 z-score，并处理 -9999/NaN
# ------------------------------
class PerVarStandardScaler:
    def fit(self, x):  # x: [B, L, N]
        x_ = x.detach()
        x_ = x_.clone()
        x_[torch.isinf(x_)] = float('nan')
        mask = torch.isnan(x_) | (x_ <= -9999 + 1e-6)
        x_[mask] = float('nan')
        # 按 N 维求均值/方差（跨 B、L）
        mean = torch.nanmean(x_, dim=(0,1), keepdim=True)  # [1,1,N]
        # 手动计算标准差，避免使用 torch.nanstd（老版本不支持）
        x_centered = x_ - mean
        x_centered = torch.where(torch.isnan(x_centered), torch.zeros_like(x_centered), x_centered)
        std = torch.sqrt(torch.nanmean(x_centered**2, dim=(0,1), keepdim=True))  # [1,1,N]
        # 缺失的变量均值/方差兜底
        std = torch.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
        mean = torch.nan_to_num(mean, nan=0.0)
        # 保存
        self.mean = mean
        self.std = std
        # 用均值填充缺失
        x_filled = torch.where(torch.isnan(x_), mean.expand_as(x_), x_)
        return x_filled

    def transform(self, x):  # x 已填充
        return (x - self.mean) / (self.std + 1e-6)

    def inverse_transform(self, y):  # y: 任意含 N 的最后一维
        return y * (self.std + 1e-6) + self.mean

# ------------------------------
# 模型组件：MLP
# ------------------------------
class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ------------------------------
# 低秩结点嵌入（LRAE，Eq. 18）
# EN ≈ E_r @ P，rank = r << N, DN
# ------------------------------
class LowRankNodeEmbed(nn.Module):
    def __init__(self, N, DN, rank):
        super().__init__()
        self.Er = nn.Parameter(torch.randn(N, rank) * 0.02)     # E_r ∈ R^{N×r}
        self.P  = nn.Parameter(torch.randn(rank, DN) * 0.02)    # P   ∈ R^{r×DN}

    @property
    def EN(self):
        return self.Er @ self.P    # [N, DN]

# ------------------------------
# 调制式结点注意力（Eq. 21–22 的高效实现）
# 输入 H: [B, N, D]
# 输出: [B, N, D]
# 复杂度 ~ O(N·D·DN) 近线性
# ------------------------------
class ModulatedNodeAttention(nn.Module):
    def __init__(self, D, DN, Dm):
        super().__init__()
        self.WQ = nn.Linear(D, Dm, bias=False)
        self.WV = nn.Linear(D, D,  bias=False)
        # 层内调制器 M^(ℓ) ∈ R^{DN×Dm}
        self.M  = nn.Parameter(torch.randn(DN, Dm) * 0.02)
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)
        self.ffn = MLP(D, 4*D, D)

    def forward(self, H, EN):  # EN: [N, DN]
        # Self-Attn（分解近似）：Softmax(H WQ M^T EN^T / sqrt(Dm)) · Softmax(EN^T) · (H WV)
        B, N, D = H.shape
        Dm = self.M.shape[1]
        
        # 检查 EN 的维度
        if EN.shape[0] != N:
            print(f"警告: EN.shape[0]={EN.shape[0]} != N={N}")
            # 如果维度不匹配，重新计算 EN
            if hasattr(self, 'node_lrae'):
                EN = self.node_lrae.EN
        
        # 右侧：Softmax(EN^T) @ (H WV)
        # EN^T: [DN, N] -> softmax along rows -> [DN, N]
        EN_t = EN.t()  # [DN, N]
        S_r = F.softmax(EN_t, dim=-1)                 # 右侧 softmax(EN^T)
        HV = self.WV(H)                                # [B, N, D]
        
        # 正确的实现：S_r: [DN, N], HV: [B, N, D] -> [B, DN, D]
        # 使用更明确的下标来避免混淆
        right = torch.einsum('mn,bnd->bmd', S_r, HV)  # [B, DN, D]

        # 左侧：Softmax(H WQ M^T EN^T / sqrt(Dm))
        Q = self.WQ(H)                                  # [B, N, Dm]
        QM_t = torch.matmul(Q, self.M.t())              # [B, N, DN]
        logits = torch.matmul(QM_t, EN.t()) / math.sqrt(Dm)   # [B, N, N]
        A_left = F.softmax(logits, dim=-1)              # [B, N, N]

        # 聚合：A_left @ ( Softmax(EN^T) @ (H WV) )
        # 我们先把 right 变回 [B, DN, D] 再用 EN 映射到 N
        right = right.permute(0, 2, 1).contiguous()     # [B, DN, D]
        # 用 EN （[N, DN]）把 DN 维汇聚到 N 维： softmax(EN) 做行归一
        S_l = F.softmax(EN, dim=-1)                     # [N, DN]
        rightN = torch.einsum('nd,bdm->bnm', S_l, right)  # [B, N, D]

        out = torch.matmul(A_left, rightN)              # [B, N, D]

        # 残差 & FFN
        H2 = self.ln1(H + out)
        H3 = self.ln2(H2 + self.ffn(H2))
        return H3

# ------------------------------
# 输入编码（Eq. 15–16）：把 [B,L,N] 展平到每结点的特征，再与结点/时间嵌入拼接
# ------------------------------
class InputEmbedding(nn.Module):
    def __init__(self, L, d_in, D_feature, DN, use_time_embed=True, time_dim=0):
        super().__init__()
        self.use_time_embed = use_time_embed
        self.L = L
        self.d_in = d_in
        self.flatten = nn.Linear(L * d_in, D_feature)
        self.node_embed = None  # 由外部 LRAE 提供 EN: [N, DN]
        # 简化：不加 time-in-day/day-in-week，保留接口
        D_total = D_feature + DN
        self.proj = MLP(D_total, D_total, D_total)

    def forward(self, x_enc, EN):  # x_enc: [B, L, N], EN: [N, DN]
        B, L, N = x_enc.shape
        assert L == self.L
        x = x_enc.permute(0, 2, 1).contiguous()  # [B,N,L]
        x = x.reshape(B, N, L * self.d_in)       # [B,N,L*d_in]
        z = self.flatten(x)                      # [B,N,D_feature]
        # 拼结点嵌入
        EN_b = EN.unsqueeze(0).expand(B, -1, -1)     # [B,N,DN]
        h0 = torch.cat([z, EN_b], dim=-1)            # [B,N,D_feature+DN]
        h0 = self.proj(h0) + h0                      # Eq.16
        return h0                                    # [B,N,D_total]

# ------------------------------
# 多层神经扩散块（由调制式结点注意力堆叠）
# ------------------------------
class NeuralDiffusion(nn.Module):
    def __init__(self, layers, D, DN, Dm):
        super().__init__()
        self.blocks = nn.ModuleList([ModulatedNodeAttention(D, DN, Dm) for _ in range(layers)])

    def forward(self, H, EN):
        for blk in self.blocks:
            H = blk(H, EN)   # 逐层“扩散+降能量”
        return H

# ------------------------------
# ScaleSTF（最小版）：InputEmbedding → NeuralDiff → Readout
# 输出：预测 H 步、每个结点 1 维（可改）
# ------------------------------
class ScaleSTF(nn.Module):
    def __init__(self, N, L, d_in=1, H_out=12, d_out=1,
                 D_feature=64, DN=32, rank=16, layers=4, Dm=64):
        super().__init__()
        self.N = N
        self.H_out = H_out
        self.node_lrae = LowRankNodeEmbed(N, DN, rank)
        self.input_embed = InputEmbedding(L, d_in, D_feature, DN)
        self.D = D_feature + DN
        self.diffusion = NeuralDiffusion(layers, self.D, DN, Dm)
        self.readout = nn.Linear(self.D, H_out * d_out)

    def forward(self, x_enc):  # x_enc: [B,L,N]
        EN = self.node_lrae.EN                  # [N,DN]
        H0 = self.input_embed(x_enc, EN)       # [B,N,D]
        H  = self.diffusion(H0, EN)            # [B,N,D]
        Y  = self.readout(H)                   # [B,N,H_out*d_out]
        B, N, _ = Y.shape
        Y = Y.view(B, N, self.H_out, -1).permute(0, 2, 1, 3).contiguous()  # [B,H_out,N,d_out]
        return Y.squeeze(-1)  # [B,H_out,N]

# ------------------------------
# 快速示例
# ------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, N = 8, 96, 500   # 批大小、历史窗口、结点数
    H_out   = 12

    x = torch.randn(B, L, N)
    # 注入缺失值与无效值
    x[0, :10, :50] = float('nan')
    x[1, :5,  :20] = -9999

    # 预处理
    scaler = PerVarStandardScaler()
    x_filled = scaler.fit(x)
    x_norm   = scaler.transform(x_filled)

    # 模型
    model = ScaleSTF(N=N, L=L, d_in=1, H_out=H_out,
                     D_feature=64, DN=32, rank=16, layers=4, Dm=64)

    y_hat = model(x_norm)      # [B, H_out, N]
    print("y_hat:", y_hat.shape)
