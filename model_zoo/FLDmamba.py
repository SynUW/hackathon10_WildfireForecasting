"""
FLDmamba for Wildfire Time Series Forecasting
Enhanced with Fourier and Laplace Transform Decomposition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np

# Add project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


from model_zoo.layers.Mamba_EncDec import Encoder, EncoderLayer
from model_zoo.layers.Embed import DataEmbedding_inverted, PositionalEmbedding
from mamba_ssm import Mamba

class Configs:
    def __init__(self, seq_len=365, pred_len=7, d_model=256, d_state=16, d_ff=2048, 
                 e_layers=5, dropout=0.1, activation='relu', output_attention=False,
                 use_norm=True, embed='timeF', freq='d', base=0, i_or_cos=0):
        # Model basic parameters
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.d_state = d_state
        self.d_ff = d_ff
        
        # Model structure parameters
        self.e_layers = e_layers
        self.dropout = dropout
        self.activation = activation
        
        # FLDmamba specific parameters
        self.base = base  # Base for RBF functions
        self.i_or_cos = i_or_cos  # 0: cosine, 1: complex exponential, 2: simple projection
        
        # Other parameters
        self.output_attention = output_attention
        self.use_norm = use_norm
        self.embed = embed
        self.freq = freq

class RBF(nn.Module):
    """Radial Basis Function layer for feature transformation"""
    def __init__(self, configs):
        super(RBF, self).__init__()
        self.d_model = configs.d_model
        self.rbf_centers = nn.Parameter(torch.randn(configs.d_model, configs.d_model))
        self.rbf_widths = nn.Parameter(torch.ones(configs.d_model))
        
    def forward(self, x):
        # x: [B, L, d_model]
        B, L, D = x.shape
        x_expanded = x.unsqueeze(-1)  # [B, L, D, 1]
        centers = self.rbf_centers.unsqueeze(0).unsqueeze(0)  # [1, 1, D, D]
        
        # Compute RBF transformation
        diff = x_expanded - centers  # [B, L, D, D]
        rbf_output = torch.exp(-torch.sum(diff**2, dim=2) / (2 * self.rbf_widths**2))
        return rbf_output

class Model(nn.Module):
    """
    FLDmamba: Fourier and Laplace Transform Decomposition Mamba
    for wildfire time series forecasting
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.d_model = configs.d_model
        self.configs = configs
        
        # Embedding layer for input sequences
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, 
            configs.d_model, 
            configs.embed, 
            configs.freq,
            configs.dropout
        )
        
        # RBF transformation layer
        self.rbf = RBF(configs)
        
        # Dual Mamba blocks: FFT-enhanced and standard
        self.fft_blocks = nn.ModuleList()
        self.standard_blocks = nn.ModuleList()
        
        for _ in range(configs.e_layers):
            # FFT-enhanced Mamba block
            self.fft_blocks.append(
                Mamba(
                    d_model=self.d_model,
                    d_state=configs.d_state,
                    d_conv=4,
                    expand=2
                )
            )
            # Standard Mamba block
            self.standard_blocks.append(
                Mamba(
                    d_model=self.d_model,
                    d_state=configs.d_state,
                    d_conv=4,
                    expand=2
                )
            )
        
        # Determine base size for Fourier/Laplace decomposition
        if configs.base == 0:
            N_base = configs.pred_len ** 2
        elif configs.base > 0:
            N_base = configs.pred_len * configs.base
        else:
            N_base = configs.pred_len
            
        # Fourier/Laplace transform projectors
        projector_layers = [
            nn.Linear(configs.d_model, configs.d_model * 2, bias=True),
            nn.SiLU(),
            nn.Linear(configs.d_model * 2, configs.d_model * 2, bias=True),
            nn.SiLU(),
            nn.Linear(configs.d_model * 2, N_base, bias=True)
        ]
        
        # Amplitude projector (for Laplace transform)
        self.projector_alpha = nn.Sequential(*projector_layers)
        # Magnitude projector
        self.projector_A = nn.Sequential(*projector_layers)
        # Frequency projector (for Fourier transform)
        self.projector_omega = nn.Sequential(*projector_layers)
        # Phase projector
        self.projector_fai = nn.Sequential(*projector_layers)
        
        # Time transformation projector
        self.projector_t = nn.Sequential(
            nn.Linear(1, 10),
            nn.SiLU(),
            nn.Linear(10, 1)
        )
        
        # Fallback linear projector
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forward pass for wildfire forecasting
        
        Args:
            x_enc: Input sequences [B, L, N] - historical wildfire data
            x_mark_enc: Time features for encoder [B, L, time_features]
            x_dec: Decoder input (not used in this architecture)
            x_mark_dec: Decoder time features (not used)
        
        Returns:
            dec_out: Forecasted wildfire risk [B, pred_len, N]
        """
        
        # Step 1: Normalization (crucial for wildfire data stability)
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        
        # Handle missing values (-9999 -> 0) for wildfire data
        x_enc = torch.where(x_enc == -9999, torch.zeros_like(x_enc), x_enc)
        
        _, _, N = x_enc.shape
        
        # Step 2: Input embedding and RBF transformation
        enc_out = self.enc_embedding(x_enc, None)  # [B, N, d_model]
        enc_out = self.rbf(enc_out)  # RBF feature transformation
        
        # Step 3: Multi-layer Mamba processing with FFT enhancement
        for i in range(len(self.fft_blocks)):
            # Combine FFT-enhanced and standard Mamba outputs
            fft_out = self.fft_blocks[i](enc_out)
            std_out = self.standard_blocks[i](enc_out)
            # Weighted combination with residual connection
            enc_out = (fft_out + std_out + enc_out) / 3
        
        # Step 4: Fourier/Laplace Transform Decomposition
        # Apply FFT to capture frequency domain information
        hidden = torch.fft.fft(enc_out)
        enc_out_real, enc_out_imag = hidden.real, hidden.imag
        
        B, L, N_model = enc_out.shape
        pred_len = self.configs.pred_len
        
        # Project to Fourier/Laplace coefficients
        A = self.projector_A(enc_out).reshape(B, L, pred_len, -1)  # Amplitude
        alpha = self.projector_alpha(enc_out_real).reshape(B, L, pred_len, -1)  # Decay rate
        omega = self.projector_omega(enc_out_imag).reshape(B, L, pred_len, -1)  # Frequency
        fai = self.projector_fai(enc_out_imag).reshape(B, L, pred_len, -1)  # Phase
        
        # Step 5: Generate time grid for prediction
        if self.configs.base == 0:
            t = torch.linspace(0.0001, 1, pred_len, device=A.device)
        else:
            t = torch.linspace(0.0001, 1, self.configs.base, device=A.device)
        
        t = t.unsqueeze(-1)
        t = self.projector_t(t).squeeze(-1)
        
        # Ensure alpha is negative (for stability in exponential decay)
        alpha = -F.elu(-alpha)
        
        # Step 6: Reconstruct time series using Fourier/Laplace decomposition
        if self.configs.i_or_cos == 0:
            # Cosine-based reconstruction (good for periodic wildfire patterns)
            dec_out = (A * torch.exp(alpha * t) * torch.cos(omega * t + fai)).sum(-1)
        elif self.configs.i_or_cos == 1:
            # Complex exponential reconstruction (captures transient dynamics)
            omega_complex = torch.complex(real=torch.zeros_like(omega), imag=omega)
            dec_out = ((A * torch.exp(alpha * t) * torch.exp(omega_complex * t)).real).sum(-1)
        else:
            # Fallback to simple linear projection
            dec_out = self.projector(enc_out)
        
        # Transpose to match expected output format
        dec_out = dec_out.permute(0, 2, 1)  # [B, pred_len, N]
        
        # Step 7: De-normalization
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out

# Enhanced configuration for wildfire detection
class WildfireConfigs(Configs):
    def __init__(self):
        super().__init__(
            seq_len=365,  # One year of historical data
            pred_len=1,   # Predict 1 days ahead
            d_model=512,  # Increased model dimension for complexity
            d_state=32,   # Increased state dimension
            d_ff=2048,
            e_layers=6,   # More layers for better feature extraction
            dropout=0.1,
            use_norm=True,
            base=2,       # Base for RBF functions
            i_or_cos=0    # Use cosine reconstruction for seasonal patterns
        )

if __name__ == '__main__':
    # Test the FLDmamba model for wildfire forecasting
    configs = WildfireConfigs()
    
    # Create model and move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(configs).to(device)
    
    # Test data simulation
    batch_size = 16
    num_features = 40  # Meteorological + vegetation + geographical features
    
    # Simulate wildfire-related time series data
    x_enc = torch.randn(batch_size, configs.seq_len, num_features).to(device)
    x_mark_enc = torch.randn(batch_size, configs.seq_len, 7).to(device)  # Time features
    x_dec = torch.randn(batch_size, configs.pred_len, configs.d_model).to(device)
    x_mark_dec = torch.randn(batch_size, configs.pred_len, 7).to(device)
    
    # Add some wildfire-like patterns (seasonal trends, extreme values)
    seasonal_pattern = torch.sin(torch.linspace(0, 4*np.pi, configs.seq_len)).unsqueeze(0).unsqueeze(-1)
    x_enc += seasonal_pattern.to(device) * 0.5
    
    # Forward pass
    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    print("="*50)
    print("FLDmamba Wildfire Forecasting Model Test")
    print("="*50)
    print(f"Input shape (Historical data): {x_enc.shape}")
    print(f"Output shape (Forecasted risk): {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Sequence length: {configs.seq_len} days")
    print(f"Prediction horizon: {configs.pred_len} days")
    print(f"Number of features: {num_features}")
    print("="*50)