import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 添加正确的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
layers_path = os.path.join(current_dir, 'layers')
sys.path.insert(0, layers_path)

from SelfAttention_Family import FullAttention, AttentionLayer
from Embed import DataEmbedding_inverted, PositionalEmbedding
import numpy as np

# 添加utils路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
utils_path = os.path.join(parent_dir, 'model_zoo', 'utils')
sys.path.insert(0, utils_path)

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))
        # [B, N, L] -> [B, N, L/patch_len, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)

        # [B, N, L/patch_len, patch_len] -> [B*N, L/patch_len, patch_len]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        
        # Input encoding
        # [B*N, L/patch_len, patch_len] -> [B*N, L/patch_len, d_model]
        x = self.value_embedding(x) + self.position_embedding(x)
        # [B*N, L/patch_len, d_model] -> [B, N, L/patch_len, d_model]
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))

        # [B, N, L/patch_len, d_model] -> [B, N, L/patch_len+1, d_model]
        # serise-level global token is added to the last position
        x = torch.cat([x, glb], dim=2)
        
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            # endogenous: x, exogenous: cross
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        # self-attention for endogenous
        # x: [B*N, L/patch_len+1, d_model]
        # cross: [B, N, d_model]
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        # global attention as Q, cross as K, V
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])
        x_glb_attn = torch.reshape(x_glb_attn,
                                   (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        # s_mamba used the same conv1d for variables interaction modeling
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        self.patch_num = int(configs.seq_len // configs.patch_len)
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in
        # Embedding
        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)

        self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                   configs.dropout, time_feat_dim=7)

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.head_nf = configs.d_model * (self.patch_num + 1)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # if self.use_norm:
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

        # [B, L, N] -> [B, N, L] -> [B*N, L/patch_len, d_model]
        en_embed, n_vars = self.en_embedding(x_enc[:, :, 0].unsqueeze(-1).permute(0, 2, 1))
        # [B, L, N-1] -> [B, N-1, L] -> [B, N-1, d_model]
        ex_embed = self.ex_embedding(x_enc[:, :, 1:], x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # if self.use_norm:
        #     # De-Normalization from Non-stationary Transformer
        #     dec_out = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
        #     dec_out = dec_out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # if self.use_norm:
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
            
        _, _, N = x_enc.shape  # 所以TimeXer的输入是BLN！！！

        # [B, L, N] -> [B, N, L]
        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # if self.use_norm:
        #     # De-Normalization from Non-stationary Transformer
        #     dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        #     dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.features == 'M':
            dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]  BTC



if __name__ == "__main__":

    
    # 修复bug：未定义Configs，改为简单的命名空间或使用types.SimpleNamespace
    class Configs:
        def __init__(self, task_name, features, seq_len, pred_len, use_norm):
            self.task_name = task_name
            self.features = features
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.use_norm = use_norm
            # 添加所有缺失的属性
            self.patch_len = 16  # 默认patch长度
            self.d_model = 512   # 默认模型维度
            self.n_heads = 8     # 默认注意力头数
            self.e_layers = 2    # 默认编码器层数
            self.d_ff = 2048     # 默认前馈网络维度
            self.dropout = 0.1   # 默认dropout率
            self.activation = 'gelu'  # 默认激活函数
            self.enc_in = 39     # 默认输入通道数
            self.c_out = 39      # 默认输出通道数
            self.embed = 'timeF' # 默认嵌入类型
            self.freq = 'd'      # 默认频率
            self.output_attention = False  # 默认不输出注意力权重
            self.factor = 5      # 默认factor参数

    configs = Configs(
        task_name='forecasting',
        features='M',
        seq_len=365,
        pred_len=1,
        use_norm=True,
    )
    model = Model(configs)
    # 修复测试数据：使序列长度大于patch_len，并调整维度顺序
    x_enc = torch.randn(1, 39, 365)  # [batch, seq_len, features] - 正确的格式
    x_mark_enc = torch.randn(1, 7, 365)  # [batch, seq_len, time_features]
    x_dec = torch.randn(1, 1, 39)  # [batch, pred_len, features]
    x_mark_dec = torch.randn(1, 1, 7)  # [batch, pred_len, time_features]
    
    # 测试模型
    try:
        dec_out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        print(f"✅ TimeXer model test successful!")
        print(f"Output shape: {dec_out.shape}")
    except Exception as e:
        print(f"❌ TimeXer model test failed: {e}")
        import traceback
        traceback.print_exc()