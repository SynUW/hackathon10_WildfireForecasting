import torch.nn as nn
import torch.nn.functional as F
import torch


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x

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


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
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
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class GatedMoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=4, top_k=2, dropout=0.1):
        super(GatedMoE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # x: [B, N, D]
        gate_scores = torch.softmax(self.gate(x), dim=-1)  # [B, N, E]
        topk_scores, topk_idx = torch.topk(gate_scores, self.top_k, dim=-1)  # [B, N, K]

        B, N, D = x.shape
        output = torch.zeros_like(x)

        # Compute outputs per top-k expert
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=0)  # [E, B, N, D]
        for k in range(self.top_k):
            idx = topk_idx[:, :, k]  # [B, N]
            score = topk_scores[:, :, k].unsqueeze(-1)  # [B, N, 1]
            # Gather outputs for each position's top-k expert
            gathered = torch.gather(
                expert_outputs.permute(1, 2, 3, 0),  # [B, N, D, E]
                dim=3,
                index=idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[-1], 1)
            ).squeeze(-1)  # [B, N, D]
            output += score * gathered
        return output


class TopKFeatureMoE(nn.Module):
    """
    Top-K Feature Selection MoE: Select top-k most important features and process them with fewer experts
    """
    def __init__(self, d_model, d_ff, num_features, num_experts=None, top_k=None, dropout=0.1):
        super(TopKFeatureMoE, self).__init__()
        self.num_features = num_features
        self.num_experts = num_experts or (num_features // 2)  # Default: half of features
        self.top_k = top_k or (num_features // 2)  # Default: use half of features
        
        # Feature importance scoring network
        self.feature_importance = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Experts (fewer than original features)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model)
            ) for _ in range(self.num_experts)
        ])
        
        # Expert selection gate (for selected features)
        self.expert_gate = nn.Linear(d_model, self.num_experts)
        
        # Feature reconstruction network (to handle unselected features)
        self.feature_reconstructor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model)
        )
        
    def forward(self, x):
        # x: [B, N, D] - batch, features, d_model
        B, N, D = x.shape
        
        # 1. Calculate feature importance scores
        feature_importance = self.feature_importance(x)  # [B, N, 1]
        feature_scores = feature_importance.squeeze(-1)  # [B, N]
        
        # 2. Select top-k most important features
        topk_scores, topk_idx = torch.topk(feature_scores, self.top_k, dim=-1)  # [B, top_k]
        
        # 3. Create feature selection mask
        feature_mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)
        feature_mask.scatter_(1, topk_idx, True)  # [B, N]
        
        # 4. Process selected features with experts
        selected_features = x * feature_mask.unsqueeze(-1).float()  # [B, N, D]
        
        # 5. Apply expert selection for selected features
        expert_scores = torch.softmax(self.expert_gate(selected_features), dim=-1)  # [B, N, E]
        
        # 6. Compute expert outputs
        expert_outputs = torch.stack([expert(selected_features) for expert in self.experts], dim=0)  # [E, B, N, D]
        
        # 7. Weighted combination of expert outputs
        expert_outputs_weighted = torch.zeros_like(selected_features)  # [B, N, D]
        for i, expert_output in enumerate(expert_outputs):
            expert_outputs_weighted += expert_output * expert_scores[:, :, i:i+1]  # [B, N, D]
        
        # 8. Handle unselected features with reconstruction network
        unselected_features = x * (~feature_mask).unsqueeze(-1).float()  # [B, N, D]
        reconstructed_features = self.feature_reconstructor(unselected_features)  # [B, N, D]
        
        # 9. Combine selected and reconstructed features
        output = expert_outputs_weighted + reconstructed_features
        
        return output
    
    def get_feature_importance(self, x):
        """Get feature importance scores for analysis"""
        feature_importance = self.feature_importance(x)  # [B, N, 1]
        return feature_importance.squeeze(-1)  # [B, N]
    
    def get_selected_features(self, x):
        """Get which features are selected for analysis"""
        feature_importance = self.feature_importance(x)  # [B, N, 1]
        feature_scores = feature_importance.squeeze(-1)  # [B, N]
        _, topk_idx = torch.topk(feature_scores, self.top_k, dim=-1)  # [B, top_k]
        return topk_idx


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", num_experts=4, top_k=2, moe_active=False, multi_variate=False):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        
        self.multi_variate = multi_variate
        
        self.moe_active = moe_active
        self.feature_moe_active = False
        
        self.moe = GatedMoE(d_model=d_model, d_ff=d_ff, num_experts=num_experts, top_k=top_k, dropout=dropout)
        
                # Top-K Feature MoE
        self.feature_moe = TopKFeatureMoE(
            d_model=d_model,
            d_ff=d_ff,
            num_features=39,  # 39 features in total
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout
        )
        
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x: [B, N, D]
        if self.multi_variate:
            new_x, attn = self.attention(
                x[:, 0, :].unsqueeze(1), x[:, 1:, :], x[:, 1:, :],
                attn_mask=attn_mask,
                tau=tau, delta=delta
            )
        
        else:
            new_x, attn = self.attention(
                x, x, x,
                attn_mask=attn_mask,
                tau=tau, delta=delta
            )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        
        if self.moe_active:
            y = self.moe(x)
            # y = self.dropout(y)
            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))
        elif self.feature_moe_active:
            # Top-K Feature MoE
            y = self.feature_moe(x)
            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        else:
            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm2(x + y), attn
