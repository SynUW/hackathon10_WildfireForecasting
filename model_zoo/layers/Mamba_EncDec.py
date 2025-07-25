import concurrent.futures
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model_zoo.layers.SelfAttention_Family import FullAttention, AttentionLayer
from mamba_ssm import Mamba


# class GatedMoE(nn.Module):
#     def __init__(self, d_model, d_ff, num_experts=4, top_k=2, dropout=0.1):
#         super(GatedMoE, self).__init__()
#         self.num_experts = num_experts
#         self.top_k = top_k
#         self.experts = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(d_model, int(d_model*4)),
#                 nn.ReLU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(int(d_model*4), d_model)
#             ) for _ in range(num_experts)
#         ])
#         self.gate = nn.Linear(d_model, num_experts)

#     def forward(self, x):
#         # x: [B, N, D]
#         gate_scores = torch.softmax(self.gate(x), dim=-1)  # [B, N, E]
#         topk_scores, topk_idx = torch.topk(gate_scores, self.top_k, dim=-1)  # [B, N, K]

#         B, N, D = x.shape
#         output = torch.zeros_like(x)

#         # Compute outputs per top-k expert
#         for k in range(self.top_k):
#             idx = topk_idx[:, :, k]  # [B, N]
#             score = topk_scores[:, :, k].unsqueeze(-1)  # [B, N, 1]

#             # Apply expert independently
#             expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=0)  # [E, B, N, D]

#             # Gather outputs for each position's top-k expert
#             gathered = torch.gather(
#                 expert_outputs.permute(1, 2, 3, 0),  # [B, N, D, E]
#                 dim=3,
#                 # index=idx.unsqueeze(-2).expand(-1, -1, D).unsqueeze(-1)  # [B, N, D, 1]
#                 index = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[-1], 1)

#             ).squeeze(-1)  # [B, N, D]

#             output += score * gathered

#         return output


# class EncoderLayer(nn.Module):
#     def __init__(self, attention, attention_r, d_model, d_ff=None, dropout=0.1, activation="relu"):
#         super(EncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.attention = attention
#         self.attention_r = attention_r
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)

#         self.moe = GatedMoE(d_model=d_model, d_ff=d_ff, num_experts=4, top_k=2, dropout=dropout)

#     def forward(self, x, attn_mask=None, tau=None, delta=None):
#         new_x = self.attention(x) + self.attention_r(x.flip(dims=[1])).flip(dims=[1])
#         attn = 1
#         x = x + new_x
#         x = self.norm1(x)

#         # MoE forward
#         y = self.moe(x)
#         y = self.dropout(y)

#         return self.norm2(x + y), attn
    

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
            d_model=11,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=2,  # Local convolution width
            expand=1,  # Block expansion factor)
        )
        self.man2 = Mamba(
            d_model=11,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=2,  # Local convolution width
            expand=1,  # Block expansion factor)
        )
        self.a = AttentionLayer(
                        FullAttention(False, 2, attention_dropout=0.1,
                                      output_attention=True), d_model, 1)
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x = self.attention(x) + self.attention_r(x.flip(dims=[1])).flip(dims=[1])
        attn = 1

        x = x + new_x
        y = x = self.norm1(x)
        # why transpose? B N E -> B E N -> B N E
        # 在N上滑动，捕捉特征间相关性, idea应该来自于timexer--由相同的conv1d捕捉特征间相关性
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

