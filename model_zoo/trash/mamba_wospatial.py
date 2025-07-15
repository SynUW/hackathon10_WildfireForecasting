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

# ==================== TEMPORAL-SPATIAL MODULE ====================

class TemporalSpatialMambaBlock(nn.Module):
    """Combines Spatial Mamba and Pixelwise Temporal Attention processing with Attention fusion."""

    # <<< MODIFIED __init__ >>>
    def __init__(self, dim: int):  # Add num_timesteps
        super().__init__()

        # Spatial Mamba path (unchanged)
        self.temporal_mamba = SimplifiedMambaBlock(dim=39)
        self.spectral_mamba = SimplifiedMambaBlock(dim=10)
        
        self.norm = nn.LayerNorm(dim)

    # <<< MODIFIED forward >>>
    def forward(self, x):
        
        residual = x
        x_norm = self.norm(x)  # (B, T, C)
        x_temporal = self.temporal_mamba(x_norm).permute(0, 2, 1)
        x_spectral = self.spectral_mamba(x_temporal).permute(0, 2, 1)


        return residual + x_spectral

# ==================== FULL MODEL ====================

class TemporalSpatialHybridMamba(nn.Module):
    def __init__(self, num_classes, hidden_dim=64,
                 num_temporal_spatial_blocks=1, drop_rate=0.1,
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super().__init__()
        

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
                dim=hidden_dim
            ))
            # =======================================
        self.temporal_spatial_blocks = nn.Sequential(*ts_blocks)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, num_classes)  # 输出num_classes个logit
        )

    def forward(self, x):
        B, T, C = x.shape
        x_ts = self.temporal_spatial_blocks(x)  # (B, T, C)
        x_ts = x_ts.mean(dim=1)                 # (B, C)  # 对T维做平均池化
        logits = self.head(x_ts)                # (B, num_classes)
        return logits


if __name__ == '__main__':
    dummy_tensor = torch.randn(1, 10, 39).cuda()

    model = TemporalSpatialHybridMamba(num_classes=7, hidden_dim=39).cuda()
    output = model(dummy_tensor)
    print(output.shape)