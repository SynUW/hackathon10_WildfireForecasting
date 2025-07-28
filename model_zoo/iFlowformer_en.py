import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FlowAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import numpy as np


class EnEmbedding(nn.Module):
    def __init__(self, d_model, dropout, rbf_centers=10, rbf_gamma=1.0, learnable_centers=True):
        super(EnEmbedding, self).__init__()
        # Patching
        # self.patch_len = patch_len

        self.use_rbf = False
        if self.use_rbf:
            # RBF层参数
            self.rbf_centers = rbf_centers
            self.rbf_gamma = rbf_gamma
            self.learnable_centers = learnable_centers
            
            if learnable_centers:
                # 可学习的RBF中心点
                self.rbf_centers_param = nn.Parameter(torch.randn(rbf_centers) * 2 - 1)  # 初始化为[-1, 1]范围
            else:
                # 固定的RBF中心点（从输入数据范围中采样）
                self.register_buffer('rbf_centers_buffer', torch.linspace(-5, 5, rbf_centers))
            
            # RBF输出维度
            self.rbf_output_dim = rbf_centers
            
            # 从RBF输出到d_model的映射
            self.value_embedding = nn.Linear(self.rbf_output_dim, d_model, bias=False)
        else:
            self.value_embedding = nn.Linear(365, d_model, bias=False)
            
        self.glb_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def rbf_transform(self, x):
        """应用RBF变换"""
        # x: [B, N, L] -> [B*N*L, 1]
        B, N, L = x.shape
        x_flat = x.reshape(-1, 1)  # [B*N*L, 1]
        
        # 获取RBF中心点
        if self.learnable_centers:
            rbf_centers = self.rbf_centers_param
        else:
            rbf_centers = self.rbf_centers_buffer
        
        # 计算与每个RBF中心的距离
        # [B*N*L, 1] - [rbf_centers] -> [B*N*L, rbf_centers]
        distances = x_flat - rbf_centers.unsqueeze(0)
        
        # 应用RBF核函数: exp(-gamma * distance^2)
        rbf_output = torch.exp(-self.rbf_gamma * distances ** 2)
        
        # 重塑回原始形状
        rbf_output = rbf_output.reshape(B, N, L, self.rbf_output_dim)
        
        # 对时间维度求平均，得到每个变量的RBF特征
        rbf_features = rbf_output.mean(dim=2)  # [B, N, rbf_output_dim]
        
        return rbf_features

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1))
        # [B, N, L] -> [B, N, L/patch_len, patch_len]
        # x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)

        # [B, N, L/patch_len, patch_len] -> [B*N, L/patch_len, patch_len]
        # x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        
                
        # 应用RBF变换（如果启用）
        if self.use_rbf:
            # 应用RBF变换
            rbf_features = self.rbf_transform(x)  # [B, N, rbf_output_dim]
            
            # 通过value_embedding映射到d_model
            x = self.value_embedding(rbf_features)  # [B, N, d_model]
            
            # 添加位置编码（对每个变量添加相同的位置编码）
            pos_encoding = self.position_embedding(x)  # [B, N, d_model]
            x = x + pos_encoding
        else:
            # 原始处理方式
            x = self.value_embedding(x) + self.position_embedding(x)
        
        # Input encoding
        # [B*N, L/patch_len, patch_len] -> [B*N, L/patch_len, d_model]
        # x = self.value_embedding(x) + self.position_embedding(x)
        # [B*N, L/patch_len, d_model] -> [B, N, L/patch_len, d_model]
        # x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))

        # [B, N, L/patch_len, d_model] -> [B, N, L/patch_len+1, d_model]
        # serise-level global token is added to the last position
        x = torch.cat([x, glb], dim=1)
        
        # x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        
        # RBF参数（从configs中获取，如果没有则使用默认值）
        rbf_centers = 10  # 10个center
        rbf_gamma = 1.0
        learnable_centers = True  # otherwise, use fixed centers
        
        # Embedding
        self.en_embedding = EnEmbedding(
            configs.d_model, 
            configs.dropout,
            rbf_centers=rbf_centers,
            rbf_gamma=rbf_gamma,
            learnable_centers=learnable_centers
        )
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout, time_feat_dim=7)
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FlowAttention(attention_dropout=configs.dropout), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    multi_variate=True,
                    moe_active=False
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        self.en_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FlowAttention(attention_dropout=configs.dropout), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    multi_variate=False,
                    moe_active=False
                ) for l in range(1)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)



    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev

        # valid_mask = torch.ones_like(x_enc)
        # valid_mask[:, :, 1:] = (x_enc[:, :, 1:] != 0).float()
        
        # eps = 1e-8
        # valid_counts = valid_mask.sum(dim=1, keepdim=True) + eps
        # means = (x_enc * valid_mask).sum(dim=1, keepdim=True) / valid_counts
        # variances = ((x_enc - means)**2 * valid_mask).sum(dim=1, keepdim=True) / valid_counts
        # stdev = torch.sqrt(variances + eps)
        # x_enc = ((x_enc - means) / stdev) * valid_mask

        _, _, N = x_enc.shape
        
        # Embedding.
        # B, L, 1 -> B, 1, L
        endogenous_x = x_enc[:, :, 0].unsqueeze(-1).permute(0, 2, 1)
        exogenous_x = x_enc[:, :, 1:]
        
        endo_embed, n_vars = self.en_embedding(endogenous_x)
        endo_embed, attns = self.en_encoder(endo_embed, attn_mask=None)
        endo_embed = endo_embed[:, 0, :].unsqueeze(1)  # B, 1, d_model, global token
    
        enc_out = self.enc_embedding(exogenous_x, x_mark_enc)
        enc = torch.cat([endo_embed, enc_out], dim=1)
        enc_out, attns = self.encoder(enc, attn_mask=None)

        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        # dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        # dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # print(x_mark_enc.shape)  B T 3 (year, month, day)
        # print(x_mark_enc[0, :, :])
        
        
        
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
