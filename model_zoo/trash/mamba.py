import torch.nn.functional as F
from osgeo import gdal
import h5py
import seaborn as sns
import torch
import torch.nn as nn
from einops import rearrange, reduce


class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias


def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


# ==================== CORE COMPONENTS ====================
class GroupedConvStem(nn.Module):
    """
    Stem layer using a grouped Conv2D to process time steps independently spatially.
    Requires hidden_dim to be divisible by the number of time steps (T).
    """

    def __init__(self, num_bands: int, bands_per_step: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()

        if num_bands % bands_per_step != 0:
            raise ValueError(f"num_bands ({num_bands}) must be divisible by bands_per_step ({bands_per_step})")

        self.num_timesteps = num_bands // bands_per_step
        self.features_per_step = bands_per_step  # This is num_bands in the user snippet's context
        self.total_input_channels = num_bands
        self.final_hidden_dim = hidden_dim
        T = self.num_timesteps

        # --- Crucial Constraint Check ---
        if hidden_dim % T != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by the number of time steps T ({T}) for GroupedConvStem.")
        # ---------------------------------

        padding = kernel_size // 2

        self.grouped_conv = nn.Conv2d(
            in_channels=self.total_input_channels,  # = T * F
            out_channels=self.final_hidden_dim,  # = hidden_dim
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=T  # Group by time step
        )
        # Apply normalization and activation after the grouped convolution
        self.norm = nn.BatchNorm2d(self.final_hidden_dim)
        self.act = nn.GELU()

        # print(f"GroupedConvStem: Input {self.total_input_channels} channels, Output {self.final_hidden_dim} channels, Groups={T}")

    def forward(self, x):
        # Input x: (B, C=T*F, H, W)
        x = self.grouped_conv(x)  # (B, hidden_dim, H, W)
        x = self.norm(x)
        output = self.act(x)
        return output


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # Assuming last dimension is feature dimension
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Using the standard RMSNorm definition
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def batched_gather(tensor, indices):
    """
    Gathers elements from a tensor based on indices across batches.
    Args:
        tensor (torch.Tensor): Input tensor (B, L, E).
        indices (torch.Tensor): Indices to gather (B, k).
    Returns:
        torch.Tensor: Gathered tensor (B, k, E).
    """
    B, L, E = tensor.shape
    _, k = indices.shape
    # Expand indices to match the embedding dimension
    indices_expanded = indices.unsqueeze(-1).expand(B, k, E)
    # Gather elements
    gathered_tensor = torch.gather(tensor, dim=1, index=indices_expanded)
    return gathered_tensor


class ChannelWiseAttention(nn.Module):
    def __init__(self, d_model, k_dim=1, sparsity_ratio=0.3):
        super().__init__()
        self.k_dim = k_dim
        self.sparsity_ratio = sparsity_ratio

        # Channel-wise projections for QKV
        self.to_q = nn.Linear(d_model, k_dim)  # Query projection
        self.to_k = nn.Linear(d_model, k_dim)  # Key projection
        self.to_v = nn.Linear(d_model, d_model)  # Value projection

        # Learnable scaling factor
        self.scale = (k_dim) ** -0.5

    def forward(self, x):
        B, C, X = x.shape  # Input shape: [Batch, Channels, Features]

        # 1. Calculate Q, K, V projections
        # Transpose for linear projection then transpose back
        x_reshaped = x.reshape(B * C, X)
        Q = self.to_q(x_reshaped).reshape(B, C, self.k_dim)  # [B, C, k_dim]
        K = self.to_k(x_reshaped).reshape(B, C, self.k_dim)  # [B, C, k_dim]
        V = x  # self.to_v(x_reshaped).reshape(B, C, X)  # [B, C, X]

        # 2. Calculate attention scores across channels
        # Einstein notation: batch, channel1, key_dim x batch, channel2, key_dim -> batch, channel1, channel2
        attn_scores = torch.einsum('bck,bmk->bcm', Q, K) * self.scale  # [B, C, C]
        attn_weights = F.softmax(attn_scores, dim=-1)  # Normalize along last dimension (C)

        # 3. Calculate global channel importance (average across the channel dimension)
        channel_importance = attn_weights.mean(dim=1)  # [B, C]

        # 4. Select top-k most important channels
        k = max(1, int(C * self.sparsity_ratio))
        topk_values, topk_indices = torch.topk(channel_importance, k=k, dim=1)  # [B, k]

        # 5. Collect selected features by channel
        sparse_feat = torch.stack([
            V[i, topk_indices[i]] for i in range(B)
        ], dim=0)  # [B, k, X]

        return sparse_feat, topk_indices, k


class GaussianKernel(nn.Module):
    """Gaussian kernel transformation as used in FMamba's fast-attention"""

    def forward(self, x):
        return torch.exp(-x ** 2 / 2)

class SimplifiedMambaBlock(nn.Module):
    # Standard Mamba block (more aligned with typical implementations)
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, drop_rate=0.5):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.expanded_dim = dim * expand

        self.norm = RMSNorm(dim)
        self.in_proj = nn.Linear(dim, 2 * self.expanded_dim, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.expanded_dim, out_channels=self.expanded_dim, bias=True,
            kernel_size=d_conv, groups=self.expanded_dim, padding=d_conv - 1,
        )
        self.act = nn.SiLU()

        # SSM parameters projection from input x
        # Project to dt, B, C => d_state, d_state, expanded_dim
        self.x_proj = nn.Linear(self.expanded_dim, self.d_state + self.d_state + self.expanded_dim, bias=False)

        # State dynamics matrix A (learnable log)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.expanded_dim, 1)))
        # Learnable Delta parameter bias
        self.D = nn.Parameter(torch.ones(self.expanded_dim))
        # Output projection
        self.out_proj = nn.Linear(self.expanded_dim, dim, bias=False)
        self.drop_out = nn.Dropout(drop_rate)

    def forward(self, x):
        B, L, C = x.shape
        residual = x
        x_norm = self.norm(x)  # Apply norm first
        xz = self.in_proj(x_norm)  # Project normed input
        x, z = xz.chunk(2, dim=-1)  # x:(B, L, E), z:(B, L, E)

        # Conv branch - Apply to x before activation
        x_conv = x.transpose(1, 2)  # (B, E, L)
        x_conv = self.conv1d(x_conv)[:, :, :L]  # (B, E, L)
        x_conv = x_conv.transpose(1, 2)  # (B, L, E) - Output of conv branch

        # Activation for SSM input x and gating z
        x = self.act(x)  # (B, L, E)
        z = self.act(z)  # (B, L, E) # Apply activation to z for gating later
       
        x_ssm_params = self.x_proj(x)  # (B, L, dt + B + C = S + S + E)
        dt, B_ssm, C_ssm = torch.split(x_ssm_params, [self.d_state, self.d_state, self.expanded_dim], dim=-1)

        # Apply activations/transforms needed for the loop logic
        dt = F.softplus(dt)  # (B, L, S)
        # B_ssm (B, L, S) - Ready for loop
        # C_ssm (B, L, E) - Ready for loop? Check loop logic dimensions

        A = -torch.exp(self.A_log.float())  # (E, S)
        h = torch.zeros(B, self.expanded_dim, self.d_state, device=x.device)  # (B, E, S)
        outputs = []

        # Loop remains conceptually flawed and slow, but avoids AttributeError
        for t in range(L):
            # Discretize A, B for step t (using dt directly for simplicity)
            dt_t = dt[:, t].unsqueeze(1)  # (B, 1, S)
            A_t = torch.exp(dt_t * A.unsqueeze(0))  # (B, E, S)
            Bx_t = B_ssm[:, t].unsqueeze(1) * x[:, t].unsqueeze(-1)  
            h = A_t * h  # (B, E, S) - Simplified update

            y_t = h.sum(-1) * C_ssm[:, t, :]  # (B, E) * (B, E) -> (B, E) - Just an example

            outputs.append(y_t)

        ssm_out = torch.stack(outputs, dim=1) if outputs else torch.zeros_like(x)  # (B, L, E)
            # --- End Inefficient Loop ---

        # Gating mechanism (combine SSM output with activated z)
        # Ensure ssm_out and z have the same shape (B, L, E)
        y = ssm_out * z  # Element-wise product

        # Output projection
        output = self.out_proj(y)  # (B, L, dim)
        output = self.drop_out(output)
        return output + residual


# ==================== MAMBA BLOCKS ====================
class SparseDeformableMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, sparsity_ratio=0.3, drop_rate=0.3):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.expanded_dim = dim * expand
        self.sparsity_ratio = sparsity_ratio

        self.norm = DyT(dim)
        self.proj_in = nn.Linear(dim, self.expanded_dim)
        self.proj_out = nn.Linear(self.expanded_dim, dim)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
        self.A = nn.Parameter(torch.zeros(d_state, d_state))

        self.B = nn.Parameter(torch.zeros(1, 1, d_state))
        self.C = nn.Parameter(torch.zeros(self.expanded_dim, d_state))

        self.conv = nn.Conv1d(
            in_channels=self.expanded_dim,
            out_channels=self.expanded_dim,
            kernel_size=d_conv,
            padding=d_conv-1,
            groups=self.expanded_dim,
            bias=False
        )

    def _build_controllable_matrix(self, n):
        A = torch.zeros(n, n)
        for i in range(n-1):
            A[i, i+1] = 1.0
        A[-1, :] = torch.randn(n) * 0.02
        return A

    def forward(self, x):
        B, L, C = x.shape
        #L = H * W
        residual = x

        # Flatten spatial dimensions
        x_flat = x #x.reshape(B, L, C)

        # Normalize and project
        x_norm = self.norm(x_flat)
        x_proj = self.proj_in(x_norm)  # [B, L, expanded_dim]

        # Token selection
        center_idx = L // 2
        center = x_proj[:, center_idx:center_idx+1, :]


        x_proj_norm = F.normalize(x_proj, p=2, dim=-1)        # [B, L, D]
        center_norm = F.normalize(center, p=2, dim=-1)        # [B, 1, D]

        sim = torch.matmul(x_proj_norm, center_norm.transpose(-1, -2)).squeeze(-1)

        #im = torch.matmul(x_proj, center.transpose(-1, -2)).squeeze(-1)  # [B, L]
        sim = torch.softmax(sim, dim=-1)  # Normalized probabilities

        k = max(1, int(L * self.sparsity_ratio))

        _, topk_idx = torch.topk(sim, k=k, dim=-1)

        x_sparse = batched_index_select(x_proj, 1, topk_idx)  # [B, k, expanded_dim]

        # Conv processing
        x_conv = x_sparse.transpose(1, 2)
        x_conv = self.conv(x_conv)[..., :L]
        x_conv = x_conv.transpose(1, 2)

        # SSM processing
        h = torch.zeros(B, self.expanded_dim, self.d_state, device=x.device)
        outputs = []

        for t in range(k):
            x_t = x_conv[:, t].unsqueeze(-1)
            Bx = torch.sigmoid(self.B.to(x.device)) * x_t
            h = torch.matmul(h, self.A.to(x.device).T) + Bx
            out_t = (h * torch.sigmoid(self.C.to(x.device).unsqueeze(0))).sum(-1)
            outputs.append(out_t)

        x_processed = torch.stack(outputs, dim=1)
        x_processed = self.proj_out(x_processed)

        # Combine with residual
        #x_processed = x_processed + batched_index_select(residual.reshape(B, L, C), 1, topk_idx)

        # Scatter back to original positions
        output = torch.zeros(B, L, C, device=x.device)
        output.scatter_(1, topk_idx.unsqueeze(-1).expand(-1, -1, C), x_processed)

        #return output.reshape(B, H, W, C) + x
        return output + residual


class SparseDeformableChannelMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, sparsity_ratio=0.3, drop_rate=0.3):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.expanded_dim = dim * expand
        self.sparsity_ratio = sparsity_ratio

        self.norm = DyT(dim)
        self.proj_in = nn.Linear(dim, self.expanded_dim)
        self.proj_out = nn.Linear(self.expanded_dim, dim)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
        self.A = nn.Parameter(torch.zeros(d_state, d_state))

        self.B = nn.Parameter(torch.zeros(1, 1, d_state))
        self.C = nn.Parameter(torch.zeros(self.expanded_dim, d_state))

        self.conv = nn.Conv1d(
            in_channels=self.expanded_dim,
            out_channels=self.expanded_dim,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.expanded_dim,
            bias=False
        )

        self.channel_attention = ChannelWiseAttention(self.expanded_dim, sparsity_ratio=sparsity_ratio)

    def _build_controllable_matrix(self, n):
        A = torch.zeros(n, n)
        for i in range(n - 1):
            A[i, i + 1] = 1.0
        A[-1, :] = torch.randn(n) * 0.02
        return A

    def forward(self, x):
        B, L, C = x.shape
        # L = H * W
        residual = x

        # Flatten spatial dimensions
        x_flat = x  # x.reshape(B, L, C)

        # Normalize and project
        x_norm = self.norm(x_flat)
        x_proj = self.proj_in(x_norm)  # [B, L, expanded_dim]

        x_proj_norm = F.normalize(x_proj, p=2, dim=-1)  # [B, L, D]

        x_sparse, topk_idx, k = self.channel_attention(x_proj_norm)

        # Conv processing
        x_conv = x_sparse.transpose(1, 2)
        x_conv = self.conv(x_conv)[..., :L]
        x_conv = x_conv.transpose(1, 2)

        # SSM processing
        h = torch.zeros(B, self.expanded_dim, self.d_state, device=x.device)
        outputs = []

        for t in range(k):
            x_t = x_conv[:, t].unsqueeze(-1)
            Bx = torch.sigmoid(self.B.to(x.device)) * x_t
            h = torch.matmul(h, self.A.to(x.device).T) + Bx
            out_t = (h * torch.sigmoid(self.C.to(x.device).unsqueeze(0))).sum(-1)
            outputs.append(out_t)

        x_processed = torch.stack(outputs, dim=1)
        x_processed = self.proj_out(x_processed)

        # Combine with residual
        # x_processed = x_processed + batched_index_select(residual.reshape(B, L, C), 1, topk_idx)

        # Scatter back to original positions
        output = torch.zeros(B, L, C, device=x.device)
        output.scatter_(1, topk_idx.unsqueeze(-1).expand(-1, -1, C), x_processed)

        # return output.reshape(B, H, W, C) + x
        return output + residual
# ==================== TEMPORAL-SPATIAL MODULE ====================

class TemporalSpatialMambaBlock(nn.Module):
    """Combines Spatial Mamba and Pixelwise Temporal Attention processing with Attention fusion."""

    # <<< MODIFIED __init__ >>>
    def __init__(self, dim: int, num_timesteps: int, num_heads=8, drop_rate=0.5, **mamba_kwargs):  # Add num_timesteps
        super().__init__()
        head_dim = dim // num_heads if num_heads > 0 else dim
        self.hidden_dim = dim
        self.num_timesteps = num_timesteps  # Store T

        # Spatial Mamba path (unchanged)
        self.temporal_mamba = SimplifiedMambaBlock(dim=10*10*39)
        self.spectral_mamba = SimplifiedMambaBlock(dim=int(10 * 10*10))
        self.spatial_mamba = SimplifiedMambaBlock(dim=390, **mamba_kwargs)
        
        self.norm = nn.GroupNorm(num_groups=10, num_channels=dim)


        # Post-fusion Conv - unchanged
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Dropout2d(drop_rate)
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    # <<< MODIFIED forward >>>
    def forward(self, x):
        B, C, H, W = x.shape

        residual = x
        x_norm = self.norm(x)  # B, C, H, W
        x_norm = x_norm.reshape(B, 10, C // 10, H, W)

        # Spatial processing path (unchanged)

        # === Temporal processing path (uses Attention now) ===
        x_temporal_in = x_norm.reshape(B, 10, C//10*H*W)

        x_temporal_processed = self.temporal_mamba(x_temporal_in) 

        x_spectral_in = x_temporal_processed.reshape(B, 10, C//10, H * W).permute(0, 2, 1, 3).reshape(B, C // 10,
                                                                                                        10*H * W)
        x_spectral_processed = self.spectral_mamba(x_spectral_in)
        
        x_spatial_in = x_spectral_processed.reshape(B, 10, C//10, H * W).reshape(B, 10, C//10, H*W).reshape(B, C, H*W).permute(0, 2, 1)
        x_spatial_out = self.spatial_mamba(x_spatial_in)  # (B, HW, C)

        x_spatial_out = x_spatial_out.permute(0, 2, 1)

        # =====================================================

        # Final conv + residual - unchanged
        # pdb.set_trace()
        output = self.conv_fuse(x_spatial_out.reshape(B, C, H, W))
        return residual + output

# ==================== FULL MODEL ====================

class TemporalSpatialHybridMamba(nn.Module):
    def __init__(self, num_classes, num_bands, bands_per_step=6, hidden_dim=64,
                 num_temporal_spatial_blocks=1, drop_rate=0.1,
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super().__init__()

        # Stem layer (MUST allow separation of T steps, e.g., TimeDistributedStem)
        self.stem = GroupedConvStem(  # Or GroupedConvStem if constraints met
            num_bands=num_bands, bands_per_step=bands_per_step, hidden_dim=hidden_dim
        )
        # <<< Calculate T >>>
        if num_bands % bands_per_step != 0:
            raise ValueError("num_bands must be divisible by bands_per_step")
        num_timesteps = num_bands // bands_per_step

        # Mamba kwargs still needed for spatial Mamba and standard blocks
        mamba_kwargs = {
            'd_state': mamba_d_state, 'd_conv': mamba_d_conv, 'expand': mamba_expand
        }
        num_attn_heads = 8  # Example, make configurable if needed

        # Temporal-Spatial blocks (Instantiate modified block)
        ts_blocks = []
        for _ in range(num_temporal_spatial_blocks):
            # === Pass num_timesteps to the block ===
            ts_blocks.append(TemporalSpatialMambaBlock(
                dim=hidden_dim,
                num_timesteps=num_timesteps,  # Pass T here
                num_heads=num_attn_heads,
                drop_rate=drop_rate,
                **mamba_kwargs
            ))
            # =======================================
        self.temporal_spatial_blocks = nn.Sequential(*ts_blocks)

        # Fusion weights and Head (remain the same)
        self.fusion_weights = nn.Parameter(torch.ones(2))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # Forward pass remains structurally the same
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)
        x = self.stem(x)  # (B, C, H, W)
        x_ts = self.temporal_spatial_blocks(x)  # Uses Attention internally now

        logits = self.head(x_ts)
        return logits


if __name__ == '__main__':
    dummy_tensor = torch.randn(1, 10, 39, 10, 10).cuda()

    model = TemporalSpatialHybridMamba(num_classes=7, num_bands=390, bands_per_step=390//10, hidden_dim=390).cuda()
    output = model(dummy_tensor)
    print(output.shape)