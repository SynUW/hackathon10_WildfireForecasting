"""
Is Mamba Effective for Time Series Forecasting?
"""
import torch
import torch.nn as nn
import os
import sys
from model_zoo.layers.SelfAttention_Family import FullAttention, AttentionLayer
import torch.nn.functional as F

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
        return self.dropout(x)

class EncoderLayer(nn.Module):
    def __init__(self, attention, attention_r, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.attention_r = attention_r
                
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        
        self.man = Mamba(
            d_model=47,  # Model dimension should match number of features
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor)
        )
        self.man2 = Mamba(
            d_model=47,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor)
        )
        
        self.a = AttentionLayer(
                        FullAttention(False, 2, attention_dropout=0.1,
                                      output_attention=True), d_model, 1)
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x: [B, N, D]
        new_x = self.attention(x) + self.attention_r(x.flip(dims=[1])).flip(dims=[1])
        # Temporal Mamba which is not used in the original paper
        # new_x: [B, N, D] -> [B, D, N] -> Mamba -> [B, D, N] -> [B, N, D]
        # new_x_transpose = new_x.transpose(-1, 1)
        # new_x_transpose = self.man2(new_x_transpose.flip(dims=[1])).flip(dims=[1])
        # new_x = new_x_transpose.transpose(-1, 1)
        
        attn = 1

        x = x + new_x
        y = x = self.norm1(x)
        
        # Temporal Mamba which is not used in the original paper
        # y: [B, N, D] -> [B, D, N] -> Mamba -> [B, D, N] -> [B, N, D]
        # y = y.transpose(-1, 1)
        # y = self.man(y) + self.man2(y.flip(dims=[1])).flip(dims=[1])
        # y = y.transpose(-1, 1)
        
        # why transpose? B N L -> B L N -> B N L
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

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
                                                    configs.dropout, time_feat_dim=7)
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

    def forecast(self, x_enc, x_mark_enc):


        _, _, N = x_enc.shape # B L N
        
        # x_enc: (B, L, N)
        # 1. 生成 mask (有效=1, 无效=0)
        mask = (x_enc != -9999).float()

        # 2. 将 -9999 替换为 0
        # x_enc = torch.where(x_enc == -9999, torch.zeros_like(x_enc), x_enc)
        # 将 -9999 替换为 -1
        x_enc = torch.where(x_enc == -9999, torch.full_like(x_enc, -1), x_enc)
        # x_enc = torch.cat([x_enc, mask], dim=-1)  # (B, L, 2N)

        
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


        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, L, D] where L=10 represents data from the previous 10 days
        # target_date: [B] list of date strings in yyyymmdd format
        
        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, N]
    
    
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