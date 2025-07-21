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
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding - first parameter should be sequence length
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout, time_feat_dim=6)
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
    # a = self.get_parameter_number()
    #
    # def get_parameter_number(self):
    #     """
    #     Number of model parameters (without stable diffusion)
    #     """
    #     total_num = sum(p.numel() for p in self.parameters())
    #     trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
    #     trainable_ratio = trainable_num / total_num
    #
    #     print('total_num:', total_num)
    #     print('trainable_num:', total_num)
    #     print('trainable_ratio:', trainable_ratio)

    def forecast(self, x_enc, x_mark_enc):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            """
            mask = (x_enc != 0).float() # mask out the zero (ignore) values
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            """
            valid_mask = torch.ones_like(x_enc)
            valid_mask[:, :, 1:] = (x_enc[:, :, 1:] != 0).float()
            
            eps = 1e-8
            valid_counts = valid_mask.sum(dim=1, keepdim=True) + eps
            means = (x_enc * valid_mask).sum(dim=1, keepdim=True) / valid_counts
            variances = ((x_enc - means)**2 * valid_mask).sum(dim=1, keepdim=True) / valid_counts
            stdev = torch.sqrt(variances + eps)
            x_enc = ((x_enc - means) / stdev) * valid_mask


        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forward(self, x_enc, target_date, mask=None):
        # x_enc: [B, L, D] where L=10 represents data from the previous 10 days
        # target_date: [B] list of date strings in yyyymmdd format
        
        batch_size, seq_len, _ = x_enc.shape
        device = x_enc.device  # Get the device of input tensor
        
        # Create time encodings
        time_encodings = []
        for date_str in target_date:
            # Parse date
            year = int(str(date_str)[:4])
            month = int(str(date_str)[4:6])
            day = int(str(date_str)[6:8])
            
            # Calculate weekday (0-6, 0 represents Monday)
            weekday = datetime.datetime(year, month, day).weekday()
            
            # Calculate day of year (1-366)
            day_of_year = datetime.datetime(year, month, day).timetuple().tm_yday
            
            # Create time encodings
            # 1. Month encoding (1-12)
            month_sin = torch.sin(torch.tensor(2 * np.pi * month / 12, device=device))
            month_cos = torch.cos(torch.tensor(2 * np.pi * month / 12, device=device))
            
            # 2. Weekday encoding (0-6)
            weekday_sin = torch.sin(torch.tensor(2 * np.pi * weekday / 7, device=device))
            weekday_cos = torch.cos(torch.tensor(2 * np.pi * weekday / 7, device=device))
            
            # 3. Day of year encoding (1-366)
            day_sin = torch.sin(torch.tensor(2 * np.pi * day_of_year / 366, device=device))
            day_cos = torch.cos(torch.tensor(2 * np.pi * day_of_year / 366, device=device))
            
            # Combine time features
            time_encoding = torch.tensor([month_sin, month_cos, 
                                        weekday_sin, weekday_cos,
                                        day_sin, day_cos], device=device)
            time_encodings.append(time_encoding)
        
        # Convert time encodings to tensor [B, 6]
        x_mark_enc = torch.stack(time_encodings)
        # Expand dimensions to match sequence length [B, L, 6]
        x_mark_enc = x_mark_enc.unsqueeze(1).repeat(1, seq_len, 1)
        
        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
    
    
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