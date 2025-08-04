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
        
        # 创建共享的EncoderLayer结构
        def create_expert():
            return Encoder([
                EncoderLayer(
                    Mamba(
                        d_model=d_model,
                        d_state=256,
                        d_conv=4,
                        expand=2
                    ),
                    Mamba(
                        d_model=d_model,
                        d_state=256,
                        d_conv=4,
                        expand=2
                    ),
                    d_model, d_ff, dropout, activation="gelu"
                )
            ])
        
        # 模态专家：每个模态一个专门专家，内部有Mamba encoder处理组内时序信息
        self.fire_expert = create_expert()
        self.weather_expert = create_expert()
        self.terrain_expert = create_expert()
        self.modis_expert = create_expert()
        
        # 交互专家：处理所有特征的全局时序交互，内部有Mamba encoder
        # 协同性专家：建模模态间的协同效应
        self.synergy_expert = create_expert()
        
        # 冗余性专家：建模模态间的冗余信息
        self.redundancy_expert = create_expert()
        
        # 交互权重门控 - 基于模态间的差异计算交互重要性
        self.interaction_gate = nn.Sequential(
            nn.Linear(d_model * 4, d_model),  # 4个模态的特征
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)  # 2个交互专家
        )
        
    def forward(self, x):
        # x: [B, N, D] where N is number of features, D is d_model
        B, N, D = x.shape
        
        # 1. 模态分割
        modalities = self.modality_splitter(x.permute(0, 2, 1))  # 需要调整维度
        
        # 2. 模态专家处理 - 每个专家处理对应的模态，内部有Mamba encoder处理组内时序信息
        fire_output, _ = self.fire_expert(modalities['fire'].permute(0, 2, 1))  # [B, 1, D]
        weather_output, _ = self.weather_expert(modalities['weather'].permute(0, 2, 1))  # [B, 12, D]
        terrain_output, _ = self.terrain_expert(modalities['terrain'].permute(0, 2, 1))  # [B, 7, D]
        modis_output, _ = self.modis_expert(modalities['modis'].permute(0, 2, 1))  # [B, 19, D]
        
        # 3. 计算模态特征表示（用于权重计算）
        fire_feat = fire_output.mean(dim=1)  # [B, D]
        weather_feat = weather_output.mean(dim=1)  # [B, D]
        terrain_feat = terrain_output.mean(dim=1)  # [B, D]
        modis_feat = modis_output.mean(dim=1)  # [B, D]
        
        # 4. 交互专家处理 - 真正建模模态间的交互
        # 创建组合输入
        combined_input = torch.cat([fire_output, weather_output, terrain_output, modis_output], dim=1)  # [B, 39, D]
        
        # 协同性专家：建模模态间的协同效应
        synergy_output, _ = self.synergy_expert(combined_input)
        
        # 冗余性专家：建模模态间的冗余信息
        redundancy_output, _ = self.redundancy_expert(combined_input)
        
        # 5. 计算权重
        # 交互权重 - 基于模态间的差异
        modality_features = torch.cat([fire_feat, weather_feat, terrain_feat, modis_feat], dim=1)  # [B, 4*D]
        # 计算两个交互专家各自的权重
        interaction_weights = torch.softmax(self.interaction_gate(modality_features), dim=-1)  # [B, 2]
        
        # 模态权重 - 基于各模态的重要性
        modality_weights = torch.softmax(torch.stack([
            fire_feat.mean(dim=1),
            weather_feat.mean(dim=1),
            terrain_feat.mean(dim=1),
            modis_feat.mean(dim=1)
        ], dim=1), dim=-1)  # [B, 4]
        
        # 6. 加权组合
        output = torch.zeros_like(x)
        
        # 模态专家贡献
        modality_contributions = [
            modality_weights[:, 0:1].view(B, 1, 1) * fire_output,
            modality_weights[:, 1:2].view(B, 1, 1) * weather_output,
            modality_weights[:, 2:3].view(B, 1, 1) * terrain_output,
            modality_weights[:, 3:4].view(B, 1, 1) * modis_output
        ]
        
        # 重新组合模态贡献
        modality_combined = torch.cat(modality_contributions, dim=1)
        output += modality_combined
        
        # 交互专家贡献 - 基于交互权重
        output += interaction_weights[:, 0:1].view(B, 1, 1) * synergy_output
        output += interaction_weights[:, 1:2].view(B, 1, 1) * redundancy_output
        
        # 组合权重用于解释
        combined_weights = torch.cat([modality_weights, interaction_weights], dim=1)  # [B, 6]
        
        return output, combined_weights

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
        # x: [B, L, N] -> split into modalities
        fire = x[:, :, self.fire_features[0]:self.fire_features[1]]
        weather = x[:, :, self.weather_features[0]:self.weather_features[1]]
        terrain = x[:, :, self.terrain_features[0]:self.terrain_features[1]]
        modis = x[:, :, self.modis_features[0]:self.modis_features[1]]
        
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
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        
        # I²MoE processing
        if self.use_i2moe:
            enc_out, expert_weights = self.i2moe(enc_out)
            self.last_expert_weights = expert_weights
        else:
            # Standard encoder
            enc_out, _ = self.encoder(enc_out)
        
        # Projection
        dec_out = self.projector(enc_out.transpose(2, 1)).transpose(1, 2)  # [B, L, D]
        
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, L, D] where L=10 represents data from the previous 10 days
        
        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
    
    
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
    
    print("\n=== Test completed successfully! ===")
    
    # Usage example for training
    print("\n=== Usage Example ===")
    print("To use I²MoE in training:")
    print("1. Set use_i2moe=True in Configs")
    print("2. The model will automatically:")
    print("   - Split 39 features into 4 modalities:")
    print("     * Fire detection (1 feature)")
    print("     * Weather factors (12 features)") 
    print("     * Terrain features (7 features)")
    print("     * MODIS products (19 features)")
    print("   - Apply 6 interaction experts")
    print("   - Use reweighting model for expert combination")
    print("3. Expert weights are stored in model.last_expert_weights")
    print("4. Modality splitter available at model.modality_splitter")
    
    # Example configuration for training
    print("\nExample training config:")
    print("configs = Configs(")
    print("    seq_len=10,")
    print("    pred_len=7,")
    print("    d_model=39,")
    print("    use_i2moe=True,  # Enable I²MoE")
    print("    num_experts=6,   # 4 uniqueness + 1 synergy + 1 redundancy")
    print("    expert_dropout=0.1")
    print(")")