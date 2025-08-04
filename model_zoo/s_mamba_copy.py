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

class I2MoE(nn.Module):
    """
    I²MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts
    重新设计：每个专家内部有encoder，模态专家处理组内信息，交互专家处理所有特征
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(I2MoE, self).__init__()
        self.d_model = d_model
        
        # 模态分割器
        self.modality_splitter = ModalitySplitter(
            fire_features=(0, 1),
            weather_features=(1, 13),
            terrain_features=(13, 20),
            modis_features=(20, 39)
        )
        
        # 模态范围定义（用于掩码训练）
        self.modality_ranges = {
            'fire': (0, 1),
            'weather': (1, 13),
            'terrain': (13, 20),
            'modis': (20, 39)
        }
        
        # 为不同模态设计专门的专家架构
        def create_fire_expert():
            # 火灾检测专家：需要快速响应，使用较小的状态空间
            return Encoder([
                EncoderLayer(
                    Mamba(d_model=d_model, d_state=16, d_conv=2, expand=1),
                    Mamba(d_model=d_model, d_state=16, d_conv=2, expand=1),
                    d_model, max(d_ff // 2, 64), dropout, activation="gelu"
                )
            ])
        
        def create_weather_expert():
            # 天气专家：需要处理复杂的天气模式，使用较大的状态空间
            return Encoder([
                EncoderLayer(
                    Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2),
                    Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2),
                    d_model, d_ff * 2, dropout, activation="gelu"
                )
            ])
        
        def create_terrain_expert():
            # 地形专家：相对稳定，使用中等复杂度
            return Encoder([
                EncoderLayer(
                    Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2),
                    Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2),
                    d_model, d_ff, dropout, activation="gelu"
                )
            ])
        
        def create_modis_expert():
            # MODIS专家：处理遥感数据，需要高精度
            return Encoder([
                EncoderLayer(
                    Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2),
                    Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2),
                    d_model, int(d_ff * 1.5), dropout, activation="gelu"
                )
            ])
        
        def create_interaction_expert():
            # 交互专家：需要处理复杂的多模态交互
            return Encoder([
                EncoderLayer(
                    Mamba(d_model=d_model, d_state=16, d_conv=4, expand=3),
                    Mamba(d_model=d_model, d_state=16, d_conv=4, expand=3),
                    d_model, int(d_ff * 3), dropout, activation="gelu"
                )
            ])
        
        # 模态专家：每个模态使用专门的架构
        self.fire_expert = create_fire_expert()
        self.weather_expert = create_weather_expert()
        self.terrain_expert = create_terrain_expert()
        self.modis_expert = create_modis_expert()
        
        # 交互专家：使用更复杂的架构处理多模态交互
        self.synergy_expert = create_interaction_expert()
        self.redundancy_expert = create_interaction_expert()
        
        # 改进的交互权重门控 - 使用注意力机制
        self.modality_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=8, 
            dropout=dropout,
            batch_first=True
        )
        
        # 模态重要性评估网络
        self.modality_importance = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # 交互权重门控 - 基于模态间的差异计算交互重要性
        self.interaction_gate = nn.Sequential(
            nn.Linear(d_model * 4, d_model),  # 4个模态的特征
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)  # 2个交互专家
        )
        
    def apply_modality_mask(self, x, mask_ratio=0.3, training=True):
        """
        应用模态掩码训练
        
        Args:
            x: 输入数据 [B, L, N]
            mask_ratio: 掩码比例，默认0.3
            training: 是否在训练模式
            
        Returns:
            masked_x: 掩码后的数据 [B, L, N]
            mask_info: 掩码信息，用于记录哪些模态被掩码
        """
        if not training or mask_ratio == 0:
            return x, None
            
        B, L, N = x.shape
        device = x.device
        
        # 为每个样本随机选择要掩码的模态
        mask = torch.rand(B, 4, device=device) < mask_ratio  # [B, 4] 4个模态
        
        # 创建掩码后的数据
        masked_x = x.clone()
        mask_info = {
            'fire': mask[:, 0],
            'weather': mask[:, 1], 
            'terrain': mask[:, 2],
            'modis': mask[:, 3]
        }
        
        # 应用智能掩码 - 使用高斯噪声而不是直接置零
        for i, (name, (start, end)) in enumerate(self.modality_ranges.items()):
            if mask[:, i].any():
                # 使用高斯噪声替代直接置零，保持数据分布
                noise = torch.randn_like(masked_x[:, :, start:end]) * 0.1
                masked_x[:, :, start:end] = noise
                
        return masked_x, mask_info
        
    def forward(self, x, mask_ratio=0.0, training=True):
        """
        前向传播
        
        Args:
            x: 输入数据 [B, N, D] where N is number of features (39), D is d_model (1024)
            mask_ratio: 掩码比例，0表示不使用掩码
            training: 是否在训练模式
            
        Returns:
            output: 输出 [B, N, D]
            combined_weights: 专家权重 [B, 6]
            mask_info: 掩码信息（如果使用掩码）
        """
        # x: [B, N, D] where N is number of features (39), D is d_model (1024)
        B, N, D = x.shape
        
        # 1. 应用模态掩码（如果启用）
        masked_x, mask_info = self.apply_modality_mask(x, mask_ratio, training)
        
        # 2. 模态专家处理 - 每个专家处理对应的特征子集
        # 输入是 [B, N, D] 其中 N=39个特征，D=d_model=1024
        # 按模态范围分割特征
        fire_features = masked_x[:, 0:1, :]  # [B, 1, D] - fire特征
        weather_features = masked_x[:, 1:13, :]  # [B, 12, D] - weather特征
        terrain_features = masked_x[:, 13:20, :]  # [B, 7, D] - terrain特征
        modis_features = masked_x[:, 20:39, :]  # [B, 19, D] - modis特征
        
        # 每个专家处理对应模态的特征
        fire_output, _ = self.fire_expert(fire_features)  # [B, 1, D]
        weather_output, _ = self.weather_expert(weather_features)  # [B, 12, D]
        terrain_output, _ = self.terrain_expert(terrain_features)  # [B, 7, D]
        modis_output, _ = self.modis_expert(modis_features)  # [B, 19, D]
        
        # 4. 计算模态特征表示（用于权重计算）
        fire_feat = fire_output.mean(dim=1)  # [B, D]
        weather_feat = weather_output.mean(dim=1)  # [B, D]
        terrain_feat = terrain_output.mean(dim=1)  # [B, D]
        modis_feat = modis_output.mean(dim=1)  # [B, D]
        
        # 5. 交互专家处理 - 真正建模模态间的交互
        # 协同性专家：建模模态间的协同效应
        # 将所有模态的输出拼接在一起
        all_modalities = torch.cat([fire_output, weather_output, terrain_output, modis_output], dim=1)  # [B, 39, D]
        synergy_output, _ = self.synergy_expert(all_modalities)
        
        # 冗余性专家：建模模态间的冗余信息
        # 计算模态间的相似性矩阵
        modality_features = torch.stack([fire_output.mean(dim=1), weather_output.mean(dim=1), 
                                       terrain_output.mean(dim=1), modis_output.mean(dim=1)], dim=1)  # [B, 4, D]
        
        # 计算相似性矩阵
        similarity_matrix = torch.bmm(modality_features, modality_features.transpose(1, 2))  # [B, 4, 4]
        
        # 基于相似性重新加权模态特征
        redundancy_weights = torch.softmax(similarity_matrix, dim=-1)  # [B, 4, 4]
        redundancy_features = torch.bmm(redundancy_weights, modality_features)  # [B, 4, D]
        
        # 扩展回特征维度 - 修复维度问题
        # redundancy_features: [B, 4, D] -> 需要扩展为 [B, N, D]
        # 对4个模态的特征求平均，然后扩展到所有39个特征
        redundancy_output = redundancy_features.mean(dim=1, keepdim=True).expand(-1, N, -1)  # [B, N, D]
        redundancy_output, _ = self.redundancy_expert(redundancy_output)
        
        # 6. 计算权重 - 使用改进的注意力机制
        # 模态特征矩阵 [B, 4, D]
        modality_features = torch.stack([fire_feat, weather_feat, terrain_feat, modis_feat], dim=1)
        
        # 使用多头注意力计算模态间的交互
        attended_features, attention_weights = self.modality_attention(
            modality_features, modality_features, modality_features
        )
        
        # 计算每个模态的重要性分数
        importance_scores = []
        for i in range(4):
            score = self.modality_importance(attended_features[:, i, :])  # [B, 1]
            importance_scores.append(score)
        
        # 模态权重 - 基于注意力机制的重要性评估
        modality_weights = torch.softmax(torch.cat(importance_scores, dim=1), dim=-1)  # [B, 4]
        
        # 交互权重 - 基于模态间的差异
        flattened_features = torch.cat([fire_feat, weather_feat, terrain_feat, modis_feat], dim=1)  # [B, 4*D]
        interaction_weights = torch.softmax(self.interaction_gate(flattened_features), dim=-1)  # [B, 2]
        
        # 7. 加权组合
        # 将所有专家输出重新组合为原始维度 [B, N, D]
        # 首先重新组合各个模态的输出
        output = torch.zeros(B, N, D, device=x.device, dtype=x.dtype)
        
        # 填充各个模态的输出
        output[:, 0:1, :] = fire_output  # [B, 1, D]
        output[:, 1:13, :] = weather_output  # [B, 12, D]
        output[:, 13:20, :] = terrain_output  # [B, 7, D]
        output[:, 20:39, :] = modis_output  # [B, 19, D]
        
        # 添加交互专家贡献
        synergy_contribution = synergy_output[:, :, :D]  # 取前D维度
        redundancy_contribution = redundancy_output[:, :, :D]  # 取前D维度
        
        # 加权组合
        output = (1 - interaction_weights[:, 0:1].view(B, 1, 1) - interaction_weights[:, 1:2].view(B, 1, 1)) * output + \
                 interaction_weights[:, 0:1].view(B, 1, 1) * synergy_contribution + \
                 interaction_weights[:, 1:2].view(B, 1, 1) * redundancy_contribution
        
        # 组合权重用于解释
        combined_weights = torch.cat([modality_weights, interaction_weights], dim=1)  # [B, 6]
        
        return output, combined_weights, mask_info

class ModalitySplitter(nn.Module):
    """
    Split input features into different modalities
    """
    def __init__(self, weather_features, modis_features, terrain_features, fire_features):
        super(ModalitySplitter, self).__init__()
        self.weather_features = weather_features
        self.modis_features = modis_features
        self.terrain_features = terrain_features
        self.fire_features = fire_features
        
    def forward(self, x):
        # x: [B, N, D] -> split into modalities (N=39 features, D=d_model)
        fire = x[:, self.fire_features[0]:self.fire_features[1], :]
        weather = x[:, self.weather_features[0]:self.weather_features[1], :]
        terrain = x[:, self.terrain_features[0]:self.terrain_features[1], :]
        modis = x[:, self.modis_features[0]:self.modis_features[1], :]
        
        return {
            'fire': fire,
            'weather': weather,
            'terrain': terrain,
            'modis': modis
        }

class Configs:
    def __init__(self, seq_len=7, pred_len=7, d_model=1024, d_state=256, d_ff=2048, 
                 e_layers=2, dropout=0.1, activation='gelu', output_attention=False,
                 use_norm=True, embed='timeF', freq='d',
                 # I²MoE parameters
                 use_i2moe=True, num_experts=6, expert_dropout=0.1,
                 # Modality configuration
                 fire_features=(0, 1),      # Fire detection (0, 1 feature)
                 weather_features=(1, 13),  # ERA5-Land features (1-12, 12 features)
                 terrain_features=(13, 20), # Terrain features (13-19, 7 features)
                 modis_features=(20, 39)):  # MODIS features (20-38, 19 features)
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
        
        # I²MoE parameters
        self.use_i2moe = use_i2moe  # Whether to use I²MoE
        self.num_experts = num_experts  # Number of experts (4 uniqueness + 1 synergy + 1 redundancy)
        self.expert_dropout = expert_dropout  # Dropout for experts
        
        # Modality feature ranges (start, end)
        self.weather_features = weather_features
        self.modis_features = modis_features
        self.terrain_features = terrain_features
        self.fire_features = fire_features

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
        self.use_i2moe = configs.use_i2moe
        
        # Embedding - first parameter should be sequence length
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout, time_feat_dim=7)
        
        # I²MoE components (only initialized if use_i2moe=True)
        if self.use_i2moe:
            self.modality_splitter = ModalitySplitter(
                configs.weather_features,
                configs.modis_features, 
                configs.terrain_features,
                configs.fire_features
            )
            self.i2moe = I2MoE(
                d_model=configs.d_model,
                d_ff=configs.d_ff,
                dropout=configs.expert_dropout
            )
        
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
        # 正确的投影层设计
        if self.use_i2moe:
            # I²MoE需要特殊的投影层：从多模态特征到预测序列
            # 输出应该是 [B, pred_len, num_features] 其中 num_features=39
            self.projector = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model * 2),
                nn.GELU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_model * 2, configs.pred_len * 39)  # 输出 pred_len * 39
            )
        else:
            # 标准Mamba的投影层
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

    def forecast(self, x_enc, x_mark_enc, mask_ratio=0.0, training=True):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,L,N] -> [B,N,D], N=39, D=1024
        
        # I²MoE processing
        if self.use_i2moe:
            enc_out, expert_weights, mask_info = self.i2moe(enc_out, mask_ratio, training)
            self.last_expert_weights = expert_weights
            self.last_mask_info = mask_info
        else:
            # Standard encoder
            enc_out, _ = self.encoder(enc_out)
            self.last_mask_info = None
        
        # Projection - 正确的I²MoE投影设计
        if self.use_i2moe:
            # I²MoE输出: [B, N, D] -> 需要转换为 [B, pred_len, num_features]
            # 保持专家系统的优势：对多模态特征进行聚合
            B, N, D = enc_out.shape
            
            # 加权聚合多模态特征（保持专家权重信息）
            if hasattr(self, 'last_expert_weights') and self.last_expert_weights is not None:
                modality_weights = self.last_expert_weights[:, :4]  # [B, 4] 前4个是模态权重
                # 对每个模态的特征进行加权
                weighted_features = torch.zeros(B, 4, D, device=enc_out.device, dtype=enc_out.dtype)
                
                # 按模态范围聚合特征
                feature_ranges = [(0, 1), (1, 13), (13, 20), (20, 39)]  # fire, weather, terrain, modis
                for i, (start, end) in enumerate(feature_ranges):
                    # 对特征维度求平均，得到每个模态的特征表示
                    modality_feat = enc_out[:, start:end, :].mean(dim=1)  # [B, D]
                    weighted_features[:, i, :] = modality_weights[:, i:i+1] * modality_feat
                
                # 聚合所有模态特征
                aggregated_features = weighted_features.sum(dim=1)  # [B, D]
            else:
                # 如果没有专家权重，使用简单的平均
                aggregated_features = enc_out.mean(dim=1)  # [B, D]
            
            # 投影到预测序列
            dec_out = self.projector(aggregated_features)  # [B, pred_len * num_features]
            
            # 重塑为 [B, pred_len, num_features] 以匹配损失函数期望
            dec_out = dec_out.view(B, self.pred_len, 39)  # [B, pred_len, 39]
        else:
            # 标准Mamba: [B, N, D] -> [B, pred_len, N]
            # 参考s_mamba.py的投影逻辑
            dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]  # [B, pred_len, N]
        
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask_ratio=0.0, training=True):
        # x_enc: [B, L, N] where L=10 represents data from the previous 10 days, N=39 features
        
        dec_out = self.forecast(x_enc, x_mark_enc, mask_ratio, training)
        return dec_out[:, -self.pred_len:, :]  # [B, pred_len, N]
    
    
if __name__ == '__main__':
    # Test without I²MoE
    print("=== Testing without I²MoE ===")
    configs = Configs(
        seq_len=10,
        pred_len=7,
        d_model=39,
        d_state=16,
        d_ff=256,
        e_layers=2,
        dropout=0.1,
        use_i2moe=False  # Disable I²MoE
    )
    
    # Create model and move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(configs).to(device)
    
    # Test data
    batch_size = 32
    x_enc = torch.randn(batch_size, configs.seq_len, configs.d_model).to(device)  # [32, 10, 39]
    x_mark_enc = torch.randn(batch_size, configs.seq_len, 7).to(device)  # [32, 10, 7]
    x_dec = torch.randn(batch_size, configs.pred_len, configs.d_model).to(device)  # [32, 7, 39]
    x_mark_dec = torch.randn(batch_size, configs.pred_len, 7).to(device)  # [32, 7, 7]
    
    # Forward propagation test
    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    print(f"Input shape: {x_enc.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with I²MoE
    print("\n=== Testing with I²MoE ===")
    configs_i2moe = Configs(
        seq_len=10,
        pred_len=7,
        d_model=39,
        d_state=16,
        d_ff=256,
        e_layers=2,
        dropout=0.1,
        use_i2moe=True,  # Enable I²MoE
        num_experts=6,
        expert_dropout=0.1
    )
    
    model_i2moe = Model(configs_i2moe).to(device)
    
    # Forward propagation test with I²MoE
    output_i2moe = model_i2moe(x_enc, x_mark_enc, x_dec, x_mark_dec)
    print(f"Input shape: {x_enc.shape}")
    print(f"Output shape: {output_i2moe.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model_i2moe.parameters()):,}")
    
    # Check expert weights if available
    if hasattr(model_i2moe, 'last_expert_weights'):
        print(f"Expert weights shape: {model_i2moe.last_expert_weights.shape}")
        print(f"Expert weights: {model_i2moe.last_expert_weights[0]}")  # First batch
    
    # Test modality splitter
    print("\n=== Testing Modality Splitter ===")
    splitter = model_i2moe.modality_splitter
    modalities = splitter(x_enc)
    for name, data in modalities.items():
        print(f"{name}: {data.shape}")
    
    # Test masked modality training
    print("\n=== Testing Masked Modality Training ===")
    
    # Test without masking
    output_normal = model_i2moe(x_enc, x_mark_enc, x_dec, x_mark_dec, mask_ratio=0.0, training=True)
    print(f"Normal output shape: {output_normal.shape}")
    
    # Test with masking
    output_masked = model_i2moe(x_enc, x_mark_enc, x_dec, x_mark_dec, mask_ratio=0.3, training=True)
    print(f"Masked output shape: {output_masked.shape}")
    
    # Check mask info
    if hasattr(model_i2moe, 'last_mask_info') and model_i2moe.last_mask_info is not None:
        print("Mask info:")
        for modality, mask in model_i2moe.last_mask_info.items():
            masked_count = mask.sum().item()
            total_count = mask.shape[0]
            print(f"  {modality}: {masked_count}/{total_count} samples masked ({masked_count/total_count*100:.1f}%)")
    
    # Test different mask ratios
    print("\n=== Testing Different Mask Ratios ===")
    for mask_ratio in [0.1, 0.2, 0.3, 0.5]:
        output = model_i2moe(x_enc, x_mark_enc, x_dec, x_mark_dec, mask_ratio=mask_ratio, training=True)
        if hasattr(model_i2moe, 'last_mask_info') and model_i2moe.last_mask_info is not None:
            total_masked = sum(mask.sum().item() for mask in model_i2moe.last_mask_info.values())
            print(f"Mask ratio {mask_ratio}: {total_masked}/{batch_size*4} total features masked")
    
    print("\n=== Test completed successfully! ===")
    
    # Usage example for training
    print("\n=== Usage Example ===")
    print("To use I²MoE with masked modality training:")
    print("1. Set use_i2moe=True in Configs")
    print("2. During training, use mask_ratio > 0:")
    print("   output = model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask_ratio=0.3, training=True)")
    print("3. During inference, use mask_ratio=0:")
    print("   output = model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask_ratio=0.0, training=False)")
    print("4. The model will automatically:")
    print("   - Split 39 features into 4 modalities")
    print("   - Apply random masking during training")
    print("   - Use 6 interaction experts")
    print("   - Store expert weights and mask info for analysis")