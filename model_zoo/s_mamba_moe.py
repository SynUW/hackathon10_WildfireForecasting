"""
Is Mamba Effective for Time Series Forecasting?
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys

# Add project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from model_zoo.layers.Mamba_EncDec import Encoder, EncoderLayer
from model_zoo.layers.Embed import DataEmbedding_inverted

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
            # 火灾检测专家：使用较小的状态空间
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
        
        def create_synergy_expert():
            # 协同专家：更强容量，偏重跨模态协同
            return Encoder([
                EncoderLayer(
                    Mamba(d_model=d_model, d_state=16, d_conv=4, expand=3),
                    Mamba(d_model=d_model, d_state=16, d_conv=4, expand=3),
                    d_model, int(d_ff * 3), dropout, activation="gelu"
                )
            ])

        def create_redundancy_expert():
            # 冗余专家：轻量瓶颈结构，偏重去冗与对齐
            return Encoder([
                EncoderLayer(
                    Mamba(d_model=d_model, d_state=8, d_conv=2, expand=2),
                    Mamba(d_model=d_model, d_state=8, d_conv=2, expand=2),
                    d_model, int(d_ff * 1.5), dropout, activation="gelu"
                )
            ])
        
        # 模态专家：每个模态使用专门的架构
        self.fire_expert = create_fire_expert()
        self.weather_expert = create_weather_expert()
        self.terrain_expert = create_terrain_expert()
        self.modis_expert = create_modis_expert()
        
        # 交互专家：结构差异化以提升多样性
        self.synergy_expert = create_synergy_expert()
        self.redundancy_expert = create_redundancy_expert()
        
        # 改进的交互权重门控 - 使用注意力机制
        self.modality_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=8, 
            dropout=dropout,
            batch_first=True
        )
        # 注意力温度参数（可学习），用于缩放Query，缓解大维度导致的softmax塌缩
        self.modality_tau = nn.Parameter(torch.tensor(1.0))

        # 冗余映射的跨注意力：token级Query，模态级Key/Value
        self.redundancy_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        self.redundancy_tau = nn.Parameter(torch.tensor(1.0))

        # 时间上下文映射（将简单时间统计映射到d_model，用于调制模态权重）
        self.use_temporal_context = True
        self.temporal_proj = nn.ModuleDict({
            'fire': nn.Sequential(nn.Linear(4, d_model), nn.GELU()),
            'weather': nn.Sequential(nn.Linear(4, d_model), nn.GELU()),
            'terrain': nn.Sequential(nn.Linear(4, d_model), nn.GELU()),
            'modis': nn.Sequential(nn.Linear(4, d_model), nn.GELU()),
        })
        
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
            # 三权重：基础/协同/冗余，softmax后和为1
            nn.Linear(d_model, 3)
        )
        # 稀疏门控与负载均衡配置
        # gating_tau_init/final: 门控温度退火起止值（值越大越平滑、越不稀疏；值越小越尖锐、越稀疏）
        # sparse_gating_warmup_epochs: 预热轮数；预热内关闭稀疏，仅退火温度，稳定训练
        # enable_sparse_gating: 是否开启稀疏路由（预热结束后自动置True）
        # top_k_interactions: 在交互两支{协同,冗余}中选择的top-k（默认1），基础支权重不稀疏
        # lb_coeff: 负载均衡损失系数（仅作用于{协同,冗余}两支，鼓励两支使用均衡）
        # last_aux_losses/current_lb_loss: 便于日志与在训练循环中加到总loss
        self.gating_tau_init: float = 2.0
        self.gating_tau_final: float = 1.0
        self.sparse_gating_warmup_epochs: int = 5
        self.gating_tau = nn.Parameter(torch.tensor(self.gating_tau_init))
        self.enable_sparse_gating: bool = False
        self.top_k_interactions: int = 1  # 在{协同,冗余}中做top-1
        self.lb_coeff: float = 0.01  # 负载均衡损失系数（推荐0.005–0.02）
        self.last_aux_losses = {}
        self.current_lb_loss = torch.tensor(0.0)

    # ==== 门控/稀疏路由配置与调度 ====
    def set_sparse_gating(self, enabled: bool) -> None:
        self.enable_sparse_gating = bool(enabled)

    def set_gating_tau(self, value: float) -> None:
        value = float(value)
        with torch.no_grad():
            self.gating_tau.data = torch.tensor(max(value, 1e-6), device=self.gating_tau.data.device)

    def configure_gating_schedule(self, init: float = 2.0, final: float = 1.0, warmup_epochs: int = 5) -> None:
        """配置门控温度退火计划。
        init/final: 起止温度；warmup_epochs: 预热轮数（内置关闭稀疏以稳定训练）。
        调用后将温度重置为init并关闭稀疏，需在每个epoch调用step_gating_schedule推进。"""
        self.gating_tau_init = float(init)
        self.gating_tau_final = float(final)
        self.sparse_gating_warmup_epochs = int(max(warmup_epochs, 0))
        self.set_gating_tau(self.gating_tau_init)
        self.enable_sparse_gating = False

    def step_gating_schedule(self, current_epoch: int) -> None:
        # 线性退火 gating_tau，并在warmup结束后开启稀疏路由
        # 用法：在训练循环每个epoch开始处调用，以实现温度从init→final线性退火
        e = max(int(current_epoch), 0)
        if self.sparse_gating_warmup_epochs > 0:
            t = min(e / float(self.sparse_gating_warmup_epochs), 1.0)
            tau = (1 - t) * self.gating_tau_init + t * self.gating_tau_final
            self.set_gating_tau(tau)
            if e >= self.sparse_gating_warmup_epochs:
                self.enable_sparse_gating = True
        else:
            # 无预热，直接使用final值并打开稀疏路由
            self.set_gating_tau(self.gating_tau_final)
            self.enable_sparse_gating = True
        
    def apply_modality_mask(self, x, mask_ratio=0.1, training=True, noise_type: str = 'gaussian_match'):
        """
        应用模态掩码训练
        
        Args:
            x: 输入数据 [B, N, D]，N为特征数（39），D为特征嵌入维度（d_model）
            mask_ratio: 掩码比例，默认0.1（建议范围0.05–0.2）
            training: 是否在训练模式
            noise_type: 掩码替换策略，'gaussian_match'基于模态分布匹配的高斯噪声，'zero'为Dropout风格置零
            
        Returns:
            masked_x: 掩码后的数据 [B, N, D]
            mask_info: 掩码信息，用于记录哪些模态被掩码
        """
        if not training or mask_ratio == 0:
            return x, None
            
        B, N, D = x.shape
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
        
        # 应用智能掩码 - 分布匹配高斯或Dropout风格置零
        for i, (name, (start, end)) in enumerate(self.modality_ranges.items()):
            sample_mask = mask[:, i]  # [B]
            num_masked = int(sample_mask.sum().item())
            if num_masked == 0:
                continue
            # 仅对被选中的样本、对应模态的特征维度进行替换（沿N维切片）
            if noise_type == 'zero':
                masked_x[sample_mask, start:end, :] = 0.0
            else:
                # 基于当前batch该模态的统计量生成高斯噪声，匹配分布，避免固定尺度失真
                modality_slice = x[:, start:end, :]  # [B, N_i, D]
                mean = modality_slice.mean(dim=(0, 1), keepdim=True)  # [1,1,D]
                std = modality_slice.std(dim=(0, 1), unbiased=False, keepdim=True)
                std = torch.clamp(std, min=1e-5)  # 数值稳定
                noise = torch.randn(
                    (num_masked, end - start, D), device=device, dtype=x.dtype
                ) * std + mean
                masked_x[sample_mask, start:end, :] = noise
                
        return masked_x, mask_info
        
    def forward(self, x, mask_ratio=0.0, training=True, temporal_stats=None):
        """
        前向传播
        
        Args:
            x: 输入数据 [B, N, D] where N is number of features (39), D is d_model (1024)
            mask_ratio: 掩码比例，0表示不使用掩码
            training: 是否在训练模式
            temporal_stats: 可选的时间统计，用于时间敏感的模态权重计算。
                期望为dict：{modality: Tensor[B, 4]}，4维为(time-mean, time-max, time-std, trend)
            
        Returns:
            output: 输出 [B, N, D]
            combined_weights: 专家权重 [B, 7]（4个模态权重 + 3个融合权重：基础/协同/冗余）
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
        modis_features = masked_x[:, 20:, :]  # [B, 19, D] - modis特征
        
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
        
        # 冗余性专家：为每个token从4个模态做跨注意力映射（避免平均后平铺）
        modality_features = torch.stack([
            fire_output.mean(dim=1),
            weather_output.mean(dim=1),
            terrain_output.mean(dim=1),
            modis_output.mean(dim=1)
        ], dim=1)  # [B, 4, D]

        # token级查询（使用各模态专家的token输出拼接）
        token_queries = torch.cat([fire_output, weather_output, terrain_output, modis_output], dim=1)  # [B, N, D]

        # 跨注意力：Q=token_queries, K=V=modality_features -> [B, N, D]（归一化 + 温度缩放）
        tau_r = F.softplus(self.redundancy_tau) + 1e-6
        q_tok = F.normalize(token_queries, dim=-1) / tau_r
        k_mod = F.normalize(modality_features, dim=-1)
        v_mod = modality_features
        redundancy_output, _ = self.redundancy_attention(q_tok, k_mod, v_mod)
        redundancy_output, _ = self.redundancy_expert(redundancy_output)
        
        # 6. 计算权重 - 使用改进的注意力机制
        # 模态特征矩阵 [B, 4, D]
        # 可选：注入时间上下文，增强模态重要性评估的时间敏感性
        if self.use_temporal_context and (temporal_stats is not None):
            fire_feat_enh = fire_feat + self.temporal_proj['fire'](temporal_stats['fire'])
            weather_feat_enh = weather_feat + self.temporal_proj['weather'](temporal_stats['weather'])
            terrain_feat_enh = terrain_feat + self.temporal_proj['terrain'](temporal_stats['terrain'])
            modis_feat_enh = modis_feat + self.temporal_proj['modis'](temporal_stats['modis'])
        else:
            fire_feat_enh, weather_feat_enh, terrain_feat_enh, modis_feat_enh = fire_feat, weather_feat, terrain_feat, modis_feat

        modality_features = torch.stack([fire_feat_enh, weather_feat_enh, terrain_feat_enh, modis_feat_enh], dim=1)
        
        # 使用多头注意力计算模态间的交互（归一化 + 温度缩放）
        tau_m = F.softplus(self.modality_tau) + 1e-6
        q_modal = F.normalize(modality_features, dim=-1) / tau_m
        k_modal = F.normalize(modality_features, dim=-1)
        v_modal = modality_features
        attended_features, attention_weights = self.modality_attention(
            q_modal, k_modal, v_modal
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
        # 门控温度缩放 + softmax
        tau_g = F.softplus(self.gating_tau) + 1e-6
        gate_logits = self.interaction_gate(flattened_features) / tau_g
        interaction_weights = torch.softmax(gate_logits, dim=-1)  # [B, 3]

        # 稀疏门控：仅在{协同,冗余}中执行top-k（默认top-1），基础权重保留
        if self.enable_sparse_gating and self.top_k_interactions == 1 and self.training:
            w = interaction_weights.clone()
            # 仅对列1、2执行top-1：保留较大者，另一个归零
            keep_is_synergy = (w[:, 1:2] >= w[:, 2:3]).float()
            keep_is_redund = 1.0 - keep_is_synergy
            w[:, 1:2] = w[:, 1:2] * keep_is_synergy
            w[:, 2:3] = w[:, 2:3] * keep_is_redund
            # 归一化到和为1，避免基础权重被动抵消
            w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
            interaction_weights = w

        # 负载均衡损失（仅统计交互专家的使用分布，鼓励接近均匀[0.5,0.5]）
        p_interact = interaction_weights[:, 1:3].mean(dim=0)  # [2]
        uniform = torch.full_like(p_interact, 0.5)
        kl = (p_interact * (torch.log(p_interact + 1e-8) - torch.log(uniform))).sum()
        self.current_lb_loss = self.lb_coeff * kl
        self.last_aux_losses = {"load_balance": float(self.current_lb_loss.detach().item())}
        
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
        base_w = interaction_weights[:, 0:1].view(B, 1, 1)
        synergy_w = interaction_weights[:, 1:2].view(B, 1, 1)
        redundancy_w = interaction_weights[:, 2:3].view(B, 1, 1)
        output = base_w * output + synergy_w * synergy_contribution + redundancy_w * redundancy_contribution
        
        # 组合权重用于解释
        combined_weights = torch.cat([modality_weights, interaction_weights], dim=1)  # [B, 7]
        
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
                ) for layer_idx in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # 逐 token 的投影层（保持每个特征token独立预测）
        # 与标准Mamba一致：对最后一维 D 做线性映射到 pred_len，随后在 forecast 中按 [B,N,D]->[B,N,pred_len]->[B,pred_len,N]
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # 统一权重初始化
        self._init_weights()

    def _init_weights(self) -> None:
        gelu_gain = math.sqrt(2.0)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=gelu_gain)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # in_proj_weight / in_proj_bias 以及 out_proj
                if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight, gain=gelu_gain)
                if hasattr(module, 'in_proj_bias') and module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                if hasattr(module, 'out_proj') and isinstance(module.out_proj, nn.Linear):
                    nn.init.xavier_uniform_(module.out_proj.weight, gain=gelu_gain)
                    if module.out_proj.bias is not None:
                        nn.init.zeros_(module.out_proj.bias)
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
            # 推理阶段强制关闭掩码
            effective_mask_ratio = mask_ratio if (training and self.training) else 0.0
            enc_out, expert_weights, mask_info = self.i2moe(enc_out, effective_mask_ratio, training)
            self.last_expert_weights = expert_weights
            self.last_mask_info = mask_info
        else:
            # Standard encoder
            enc_out, _ = self.encoder(enc_out)
            self.last_mask_info = None
        
        # Projection - 正确的I²MoE投影设计
        if self.use_i2moe:
            # I²MoE 输出保持为 [B, N, D]，在投影前做 token 级模态权重调制，避免丢失特征粒度
            B, N, D = enc_out.shape
            if hasattr(self, 'last_expert_weights') and self.last_expert_weights is not None:
                modality_weights = self.last_expert_weights[:, :4]  # [B, 4]
                token_weights = torch.ones(B, N, 1, device=enc_out.device, dtype=enc_out.dtype)
                feature_ranges = [(0, 1), (1, 13), (13, 20), (20, 39)]
                for i, (start, end) in enumerate(feature_ranges):
                    token_weights[:, start:end, :] = modality_weights[:, i:i+1].unsqueeze(-1)
                enc_out = enc_out * token_weights
            # 逐 token 线性映射到 pred_len
            dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]  # [B, pred_len, N]
        else:
            # 标准Mamba: [B, N, D] -> [B, pred_len, N]
            dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]
        
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask_ratio=0.1, training=True):
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