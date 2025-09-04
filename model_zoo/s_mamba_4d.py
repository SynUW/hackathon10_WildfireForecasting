"""
S-Mamba model for 4D embedding output [B, L, N, d] where:
- B: batch size
- L: sequence length  
- N: number of variables
- d: embedding dimension (token channels)

This model handles:
- Input: [B, L, N] 
- Embedding output: [B, L, N, d]
- Final output: [B, pred_len, N]
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
from model_zoo.layers.Embed import DataEmbedding_inverted
import datetime
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

class Model(nn.Module):
    """
    S-Mamba model for 4D embedding output [B, L, N, d]
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        
        # Embedding layer that outputs [B, L, N, d_model]
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq,
            configs.dropout, time_feat_dim=7
        )
        
        # Channel attention to effectively use the d dimension
        # We'll determine the actual embedding dimension dynamically
        self.channel_attention = None  # Will be created after first forward pass
        
        # Dimension reduction layer to reduce d_model to a manageable size
        self.dim_reducer = None  # Will be created after first forward pass
        
        # Encoder layers that process [B, L*N, 64]
        self.encoder = Encoder(
            [
                EncoderLayer(
                        Mamba(
                            d_model=64,  # Model dimension d_model
                            d_state=configs.d_state,  # SSM state expansion factor
                            d_conv=2,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                        Mamba(
                            d_model=64,  # Model dimension d_model
                            d_state=configs.d_state,  # SSM state expansion factor
                            d_conv=2,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                    64,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(64)
        )
        
        # Projector: [B, L*N, 64] -> [B, L*N, pred_len]
        self.projector = nn.Linear(64, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc):
        # x_enc: [B, L, N] where L=seq_len, N=num_vars
        B, L, N = x_enc.shape
        
        # Step 1: Embedding [B, L, N] -> [B, L, N, d_model]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, L, N, d_model]
        
        # Step 2: Apply channel attention to effectively use d dimension
        # [B, L, N, d_model] -> [B*L*N, d_model] for attention
        enc_out_flat = enc_out.view(B * L * N, -1)  # [B*L*N, d_model]
        
        # Create channel attention layer if not exists
        if self.channel_attention is None:
            actual_dim = enc_out_flat.shape[-1]
            attention_dim = min(actual_dim, 64)
            self.channel_attention = nn.Sequential(
                nn.Linear(actual_dim, attention_dim),
                nn.ReLU(),
                nn.Linear(attention_dim, actual_dim),
                nn.Sigmoid()
            ).to(enc_out_flat.device)
        
        attention_weights = self.channel_attention(enc_out_flat)  # [B*L*N, d_model]
        enc_out_attended = enc_out_flat * attention_weights  # [B*L*N, d_model]
        enc_out = enc_out_attended.view(B, L, N, -1)  # [B, L, N, d_model]
        
        # Step 3: Reduce dimension and reshape for encoder
        # [B, L, N, d_model] -> [B, L*N, 64]
        
        # Create dimension reducer if not exists
        if self.dim_reducer is None:
            actual_dim = enc_out.shape[-1]
            self.dim_reducer = nn.Linear(actual_dim, 64).to(enc_out.device)
        
        enc_out_reduced = self.dim_reducer(enc_out)  # [B, L, N, 64]
        enc_out_3d = enc_out_reduced.view(B, L * N, 64)  # [B, L*N, 64]
        
        # Step 4: Process through encoder: [B, L*N, 64] -> [B, L*N, 64]
        enc_out_3d, attns = self.encoder(enc_out_3d, attn_mask=None)
        
        # Step 5: Project to prediction length: [B, L*N, 64] -> [B, L*N, pred_len]
        projected = self.projector(enc_out_3d)  # [B, L*N, pred_len]
        
        # Step 6: Reshape back to 4D: [B, L*N, pred_len] -> [B, L, N, pred_len]
        projected_4d = projected.view(B, L, N, self.pred_len)  # [B, L, N, pred_len]
        
        # Step 7: Aggregate across time dimension: [B, L, N, pred_len] -> [B, pred_len, N]
        # Take the last time step as output
        dec_out = projected_4d[:, -1, :, :]  # [B, N, pred_len]
        dec_out = dec_out.transpose(-1, -2)  # [B, pred_len, N]
        
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, L, N] where L=seq_len, N=num_vars
        # x_mark_enc: [B, L, time_feat_dim]
        
        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out  # [B, pred_len, N]
    
    
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
    
    # Test data with 3D input (as expected)
    batch_size = 32
    seq_len = 10
    num_vars = 39
    
    x_enc = torch.randn(batch_size, seq_len, num_vars).to(device)  # [32, 10, 39]
    x_mark_enc = torch.randn(batch_size, seq_len, 7).to(device)  # [32, 10, 7]
    
    # Forward propagation test
    output = model(x_enc, x_mark_enc, None, None)
    print(f"Input shape: {x_enc.shape}")
    print(f"Output shape: {output.shape}")
    # Expected: Input [32, 10, 39], Output [32, 7, 39] 