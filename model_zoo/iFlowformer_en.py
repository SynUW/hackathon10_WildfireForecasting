import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add correct path
current_dir = os.path.dirname(os.path.abspath(__file__))
layers_path = os.path.join(current_dir, 'layers')
sys.path.insert(0, layers_path)

from Transformer_EncDec import Encoder, EncoderLayer
from SelfAttention_Family import FlowAttention, AttentionLayer
from Embed import DataEmbedding_inverted, PositionalEmbedding
import numpy as np

from timm.models.layers import trunc_normal_

class EnEmbedding(nn.Module):
    def __init__(self, d_model, dropout):
        super(EnEmbedding, self).__init__()
        # Patching
        # self.patch_len = patch_len
        self.value_embedding = nn.Linear(365, d_model, bias=False)
            
        self.glb_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1))
        # [B, N, L] -> [B, N, L/patch_len, patch_len]
        # x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)

        # [B, N, L/patch_len, patch_len] -> [B*N, L/patch_len, patch_len]
        # x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        
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
    

class LearnableFilterLayer(nn.Module):
    def __init__(self, dim):
        super(LearnableFilterLayer, self).__init__()
        # Dynamically adjust weight dimensions to match input
        self.complex_weight_1 = nn.Parameter(torch.randn(1, 1, dim, dtype=torch.float32) * 0.02)
        self.complex_weight_2 = nn.Parameter(torch.randn(1, 1, dim, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight_1, std=.02)
        trunc_normal_(self.complex_weight_2, std=.02)

    def forward(self, x_fft):
        # x_fft shape: [B, N, C] -> [B, N, C//2+1] (rfft)
        B, N, C = x_fft.shape
        
        # Ensure weight dimensions match
        if self.complex_weight_1.shape[-1] != C:
            # Dynamically adjust weight dimensions - use correct dimensions
            weight_1 = F.interpolate(self.complex_weight_1, size=C, mode='linear')
            weight_2 = F.interpolate(self.complex_weight_2, size=C, mode='linear')
        else:
            weight_1 = self.complex_weight_1
            weight_2 = self.complex_weight_2
            
        # Expand weights to match batch and sequence dimensions
        weight_1 = weight_1.expand(B, N, -1)
        weight_2 = weight_2.expand(B, N, -1)
        
        x_weighted = x_fft * weight_1
        x_weighted = complex_relu(x_weighted)
        x_weighted = x_weighted * weight_2
        return x_weighted

def complex_relu(x):
    real = F.relu(x.real)
    imag = F.relu(x.imag)
    return torch.complex(real, imag)

class Adaptive_Fourier_Filter_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.learnable_filter_layer_1 = LearnableFilterLayer(dim)
        self.learnable_filter_layer_2 = LearnableFilterLayer(dim)
        self.learnable_filter_layer_3 = LearnableFilterLayer(dim)
        
        self.threshold_param = nn.Parameter(torch.rand(1) * 0.5)
        self.low_pass_cut_freq_param = nn.Parameter(dim // 2 - torch.rand(1) * 0.5)  # Used to determine low-pass filter cutoff frequency
        self.high_pass_cut_freq_param = nn.Parameter(dim // 4 - torch.rand(1) * 0.5)  # High-pass filter cutoff frequency
        self.adaptive_filter = True
    
    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)
        flat_energy = energy.view(B, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]
        median_energy = median_energy.view(B, 1)
        normalized_energy = energy / (median_energy + 1e-6)
        threshold = torch.quantile(normalized_energy, self.threshold_param_high)
        dominant_frequencies = normalized_energy > threshold
        adaptive_mask = torch.zeros_like(x_fft, device=x_fft.device)
        adaptive_mask[dominant_frequencies] = 1
        return adaptive_mask
        
    def adaptive_freq_pass(self, x_fft, flag="high"):
        B, H, W_half = x_fft.shape
        W = (W_half - 1) * 2
        freq = torch.fft.rfftfreq(W, d=1/W).to(x_fft.device)
        if flag == "high":
            freq_mask = torch.abs(freq) >= self.high_pass_cut_freq_param.to(x_fft.device)
        else:
            freq_mask = torch.abs(freq) <= self.low_pass_cut_freq_param.to(x_fft.device)
        return x_fft * freq_mask
        
    def forward(self, x_in):
        B, N, C = x_in.shape
        dtype = x_in.dtype
        x = x_in.to(torch.float32)
         
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')

        if self.adaptive_filter:
            x_low_pass = self.adaptive_freq_pass(x_fft, flag="low")
            x_high_pass = self.adaptive_freq_pass(x_fft, flag="high")

        x_weighted = self.learnable_filter_layer_1(x_fft) + self.learnable_filter_layer_2(x_low_pass) + self.learnable_filter_layer_3(x_high_pass)
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')
        x = x.to(dtype)
        x = x.view(B, N, C)
        return x


class Adaptive_Wavelet_Filter_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.learnable_filter_layer_1 = LearnableFilterLayer(dim)
        self.learnable_filter_layer_2 = LearnableFilterLayer(dim)
        self.learnable_filter_layer_3 = LearnableFilterLayer(dim)
        
        # Wavelet-specific parameters
        self.wavelet_scales = 4  # Number of wavelet decomposition scales
        self.threshold_param = nn.Parameter(torch.rand(1) * 0.5)  # Adaptive threshold for wavelet coefficients
        self.scale_weights = nn.Parameter(torch.randn(self.wavelet_scales))  # Learnable weights for different scales
        self.adaptive_filter = True
        
        # Wavelet reconstruction parameters
        self.reconstruction_weight = nn.Parameter(torch.randn(1))
        
    def wavelet_decomposition(self, x):
        """Multi-scale wavelet decomposition using Haar wavelets"""
        B, N, C = x.shape
        wavelet_coeffs = []
        current_signal = x
        
        for scale in range(self.wavelet_scales):
            if current_signal.shape[1] >= 4:  # Ensure sufficient data points
                # Ensure sequence length is even
                if current_signal.shape[1] % 2 == 1:
                    current_signal = torch.cat([current_signal, current_signal[:, -1:, :]], dim=1)
                
                # Haar wavelet decomposition
                approx = (current_signal[:, ::2, :] + current_signal[:, 1::2, :]) / 2.0  # Approximation coefficients
                detail = (current_signal[:, ::2, :] - current_signal[:, 1::2, :]) / 2.0  # Detail coefficients
                
                wavelet_coeffs.append((approx, detail))
                current_signal = approx  # Continue decomposition with approximation
            else:
                break
                
        return wavelet_coeffs
    
    def adaptive_wavelet_filtering(self, wavelet_coeffs):
        """Apply adaptive filtering to wavelet coefficients"""
        filtered_coeffs = []
        
        for scale_idx, (approx, detail) in enumerate(wavelet_coeffs):
            # Apply learnable scale weights
            scale_weight = torch.sigmoid(self.scale_weights[scale_idx % len(self.scale_weights)])
            
            # Adaptive thresholding for detail coefficients (edge detection)
            detail_energy = torch.abs(detail).pow(2).mean(dim=-1, keepdim=True)  # [B, L_scale, 1]
            threshold = torch.quantile(detail_energy, self.threshold_param)
            detail_mask = (detail_energy > threshold).float()
            
            # Filter detail coefficients based on energy
            filtered_detail = detail * detail_mask * scale_weight
            
            # Keep approximation coefficients mostly unchanged (low-frequency components)
            filtered_approx = approx * scale_weight
            
            filtered_coeffs.append((filtered_approx, filtered_detail))
            
        return filtered_coeffs
    
    def wavelet_reconstruction(self, filtered_coeffs, original_shape):
        """Reconstruct signal from filtered wavelet coefficients"""
        if not filtered_coeffs:
            return torch.zeros(original_shape, device=original_shape[0].device if hasattr(original_shape[0], 'device') else 'cpu')
        
        # Start reconstruction from the coarsest scale
        reconstructed = filtered_coeffs[-1][0]  # Coarsest approximation coefficients
        
        # Reconstruct from coarse to fine scales
        for scale_idx in range(len(filtered_coeffs) - 2, -1, -1):
            approx, detail = filtered_coeffs[scale_idx]
            
            # Ensure dimensions match before upsampling
            target_length = approx.shape[1]
            
            # Use interpolation to match dimensions
            if reconstructed.shape[1] != target_length:
                reconstructed = F.interpolate(
                    reconstructed.permute(0, 2, 1),  # [B, C, L]
                    size=target_length,
                    mode='linear',
                    align_corners=False
                ).permute(0, 2, 1)  # [B, L, C]
            
            # Simple addition for reconstruction (avoid complex upsampling)
            reconstructed = reconstructed + detail
        
        # Ensure final output has correct length
        if reconstructed.shape[1] != original_shape[1]:
            reconstructed = F.interpolate(
                reconstructed.permute(0, 2, 1),  # [B, C, L]
                size=original_shape[1],
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)  # [B, L, C]
            
        return reconstructed
    
    def forward(self, x_in):
        B, N, C = x_in.shape
        dtype = x_in.dtype
        x = x_in.to(torch.float32)
        
        # 1. Multi-scale wavelet decomposition
        wavelet_coeffs = self.wavelet_decomposition(x)
        
        # 2. Apply adaptive filtering to wavelet coefficients
        filtered_coeffs = self.adaptive_wavelet_filtering(wavelet_coeffs)
        
        # 3. Reconstruct signal from filtered coefficients
        x_reconstructed = self.wavelet_reconstruction(filtered_coeffs, x.shape)
        
        # 4. Apply learnable transformations (simplified for real-valued signals)
        if self.adaptive_filter:
            # Simple element-wise transformations for real-valued signals
            # Use the real part of complex weights as simple scaling factors
            weight_1 = self.learnable_filter_layer_1.complex_weight_1.squeeze().real.mean()
            weight_2 = self.learnable_filter_layer_2.complex_weight_1.squeeze().real.mean()
            weight_3 = self.learnable_filter_layer_3.complex_weight_1.squeeze().real.mean()
            
            # Apply different weights to different components
            x_approx = x_reconstructed * weight_1  # Main signal
            x_detail = x_reconstructed * weight_2  # Detail components
            x_edge = x_reconstructed * weight_3    # Edge components
            
            # Combine filtered components
            x_weighted = x_approx + 0.5 * x_detail + 0.2 * x_edge
        else:
            x_weighted = x_reconstructed
        
        # 5. Apply reconstruction weight
        x_weighted = x_weighted * torch.sigmoid(self.reconstruction_weight)
        
        x = x_weighted.to(dtype)
        return x


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
        
        # Embedding
        self.en_embedding = EnEmbedding(
            configs.d_model, 
            configs.dropout,
        )
        
        # Add RBF processing for exogenous_x
        self.exo_rbf_centers = 50 # 10 centers by default
        self.exo_rbf_gamma = 10  # 1.0 by default
        learnable_centers=False
        if learnable_centers:
            self.exo_rbf_centers_param = nn.Parameter(torch.randn(self.exo_rbf_centers) * 2 - 1)
        else:
            self.register_buffer('exo_rbf_centers_buffer', torch.linspace(-1, 1, self.exo_rbf_centers))
        
        # Add learnable frequency domain transformation matrices
        self.freq_transform_weather_mag = nn.Parameter(torch.randn(12))  # ERA5 magnitude transformation parameters
        self.freq_transform_weather_phase = nn.Parameter(torch.randn(12))  # ERA5 phase transformation parameters
        
        # Initialize MODIS transformation parameters with maximum dimensions, dynamically adjusted at runtime
        self.max_modis_dim = 1500  # Set maximum dimension
        self.freq_transform_modis_mag = nn.Parameter(torch.randn(self.max_modis_dim))  # MODIS magnitude transformation parameters
        self.freq_transform_modis_phase = nn.Parameter(torch.randn(self.max_modis_dim))  # MODIS phase transformation parameters
        
        # FFT-ILT transformation parameters
        self.fft_filter_W = nn.Parameter(torch.randn(configs.d_model, configs.d_model))  # Frequency domain filter matrix
        self.ilt_linear = nn.Linear(configs.d_model, configs.d_model)  # ILT linear mapping
        
        # Learnable ILT reconstruction parameters (A_n, σ_n, w_n, φ_n)
        self.ilt_A = nn.Parameter(torch.randn(configs.d_model))  # Amplitude parameters
        self.ilt_sigma = nn.Parameter(torch.randn(configs.d_model))  # Decay parameters
        self.ilt_w = nn.Parameter(torch.randn(configs.d_model))  # Frequency parameters
        self.ilt_phi = nn.Parameter(torch.randn(configs.d_model))  # Phase parameters
        
        # Wavelet transformation output parameters
        self.wavelet_scales = 4  # Number of wavelet decomposition scales
        self.wavelet_filter_W = nn.Parameter(torch.randn(configs.d_model, configs.d_model))  # Wavelet domain filter matrix
        self.wavelet_linear = nn.Linear(configs.d_model, configs.d_model)  # Wavelet linear mapping
        
        # Learnable wavelet reconstruction parameters
        self.wavelet_A = nn.Parameter(torch.randn(configs.d_model))  # Wavelet amplitude parameters
        self.wavelet_scale = nn.Parameter(torch.randn(configs.d_model))  # Wavelet scale parameters
        self.wavelet_shift = nn.Parameter(torch.randn(configs.d_model))  # Wavelet shift parameters
        
        # Control variables
        self.fourier_as_features = False  # Whether to use Fourier features as additional features
        self.fft_ifft = False  # Whether to use FFT-IFFT transformation for ERA5 and MODIS
        self.affirm_transform = False  # Whether to use affirm transformation for ERA5 and MODIS
        self.wavelet_transform = False  # Whether to use adaptive wavelet transformation for ERA5 and MODIS
        
        self.fft_ilt = False  # Whether to use FFT-ILT transformation as output
        self.wavelet_output = False  # Whether to use wavelet transformation as output
        
        self.affirm_adaptive_filter = Adaptive_Fourier_Filter_Block(configs.d_model)
        self.adaptive_wavelet_filter = Adaptive_Wavelet_Filter_Block(configs.d_model)
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout, time_feat_dim=7)
        
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
                    multi_variate=False,
                    moe_active=False
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def fourier_transform_features(self, x):
        """Apply Fourier transform to time series and extract features"""
        # x: [B, L, N] -> [B, L, N]
        B, L, N = x.shape
        
        # Apply FFT
        fft_result = torch.fft.fft(x, dim=1)  # [B, L, N]
        
        # Extract magnitude spectrum (first L//2 frequency components)
        magnitude = torch.abs(fft_result[:, :L//2, :])  # [B, L//2, N]
        
        # Extract phase spectrum
        phase = torch.angle(fft_result[:, :L//2, :])  # [B, L//2, N]
        
        # Calculate power spectral density
        power_spectrum = magnitude ** 2  # [B, L//2, N]
        
        # Extract dominant frequency features (only take first 2 frequency components)
        dominant_freqs = magnitude[:, :2, :]  # [B, 2, N]
        
        # Calculate spectral statistical features
        mean_freq = magnitude.mean(dim=1, keepdim=True)  # [B, 1, N]
        max_freq = magnitude.max(dim=1, keepdim=True)[0]  # [B, 1, N]
        
        # Concatenate features (reduce feature count)
        fourier_features = torch.cat([
            dominant_freqs,  # [B, 2, N]
            mean_freq,       # [B, 1, N]
            max_freq         # [B, 1, N]
        ], dim=1)  # [B, 4, N]
        
        return fourier_features

    def wavelet_transform_features(self, x):
        """Apply wavelet transform to time series and extract features"""
        # x: [B, L, N] -> [B, L, N]
        B, L, N = x.shape
        
        # Use simple Haar wavelet transform (can be implemented via convolution)
        # Here use simplified method: calculate difference features at different scales
        
        # 1-scale difference
        diff1 = x[:, 1:, :] - x[:, :-1, :]  # [B, L-1, N]
        
        # 2-scale difference
        diff2 = x[:, 2:, :] - x[:, :-2, :]  # [B, L-2, N]
        
        # Calculate statistical features (only keep mean)
        mean_diff1 = diff1.mean(dim=1, keepdim=True)  # [B, 1, N]
        mean_diff2 = diff2.mean(dim=1, keepdim=True)  # [B, 1, N]
        
        # Concatenate wavelet features (reduce feature count)
        wavelet_features = torch.cat([
            mean_diff1,  # [B, 1, N]
            mean_diff2   # [B, 1, N]
        ], dim=1)  # [B, 2, N]
        
        return wavelet_features

    def fft_ilt_transform(self, x):
        """Simplified FFT-ILT transformation with learnable parameter matrices and inverse Laplace transform"""
        # x: [B, L, D] -> [B, L, D]
        B, L, D = x.shape
        
        # 1. Simple FFT transformation
        fft_x = torch.fft.fft(x, dim=1)  # [B, L, D]
        
        # 2. Extract real and imaginary parts
        fft_real = fft_x.real  # [B, L, D]
        fft_imag = fft_x.imag  # [B, L, D]
        
        # 3. Multiply with learnable filter matrix W
        fft_combined = fft_real + fft_imag  # [B, L, D]
        fft_reshaped = fft_combined.reshape(B * L, D)  # [B*L, D]
        fft_filtered = torch.mm(fft_reshaped, self.fft_filter_W)  # [B*L, D]
        fft_filtered = fft_filtered.reshape(B, L, D)  # [B, L, D]
        
        # 4. Linear mapping
        fft_mapped = self.ilt_linear(fft_filtered)  # [B, L, D]
        
        # 5. Inverse Laplace Transform (ILT)
        # Use learnable parameters to reconstruct time domain signal
        # Formula: f(t) = Σ A_n * exp(-σ_n * t) * cos(w_n * t + φ_n)
        
        # Create time vector
        t = torch.arange(L, dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(-1)  # [1, L, 1]
        
        # Apply learnable ILT parameters
        A = F.softplus(self.ilt_A.unsqueeze(0).unsqueeze(0))  # [1, 1, D]
        sigma = F.softplus(self.ilt_sigma.unsqueeze(0).unsqueeze(0))  # [1, 1, D]
        w = F.softplus(self.ilt_w.unsqueeze(0).unsqueeze(0))  # [1, 1, D]
        phi = self.ilt_phi.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
        
        # Calculate ILT reconstruction
        decay = torch.exp(-sigma * t)  # [1, L, D]
        oscillation = torch.cos(w * t + phi)  # [1, L, D]
        ilt_reconstructed = A * decay * oscillation  # [1, L, D]
        
        # 6. Process classification and regression features separately
        # Classification features (1st) - combine with ILT enhancement
        class_feature = x[:, :, 0:1]  # [B, L, 1]
        class_fft = fft_mapped[:, :, 0:1]  # [B, L, 1]
        class_ilt = ilt_reconstructed[:, :, 0:1]  # [B, L, 1]
        
        # Classification feature enhancement (combine FFT and ILT)
        class_enhanced = class_feature + torch.tanh(class_fft) + 0.2 * class_ilt
        
        # Regression features (2nd-39th) - combine with ILT
        reg_features = x[:, :, 1:]  # [B, L, D-1]
        reg_fft = fft_mapped[:, :, 1:]  # [B, L, D-1]
        reg_ilt = ilt_reconstructed[:, :, 1:]  # [B, L, D-1]
        
        # Regression feature processing (combine FFT and ILT)
        reg_enhanced = reg_features + 0.1 * torch.tanh(reg_fft) + 0.05 * reg_ilt
        
        # 7. Combine output
        output = torch.cat([class_enhanced, reg_enhanced], dim=-1)  # [B, L, D]
        
        return output

    def wavelet_output_transform(self, x):
        """Simplified multi-scale wavelet transformation output, referencing fft_ilt_transform structure"""
        # x: [B, L, D] -> [B, L, D]
        B, L, D = x.shape
        
        # 1. Multi-scale wavelet decomposition (simplified version)
        wavelet_features = []
        current_signal = x
        
        for scale in range(min(self.wavelet_scales, 3)):  # Limit maximum number of scales
            if current_signal.shape[1] >= 4:  # Ensure sufficient data points
                # Ensure sequence length is even
                if current_signal.shape[1] % 2 == 1:
                    current_signal = torch.cat([current_signal, current_signal[:, -1:, :]], dim=1)
                
                # Haar wavelet decomposition
                approx = (current_signal[:, ::2, :] + current_signal[:, 1::2, :]) / 2.0
                detail = (current_signal[:, ::2, :] - current_signal[:, 1::2, :]) / 2.0
                
                # Collect features
                wavelet_features.append(approx)
                wavelet_features.append(detail)
                
                current_signal = approx
            else:
                break
        
        # 2. Feature fusion
        if wavelet_features:
            # Interpolate all scale features to original length
            interpolated_features = []
            for feat in wavelet_features:
                if feat.shape[1] != L:
                    feat_interp = F.interpolate(
                        feat.permute(0, 2, 1),  # [B, D, L_scale]
                        size=L,
                        mode='linear',
                        align_corners=False
                    ).permute(0, 2, 1)  # [B, L, D]
                else:
                    feat_interp = feat
                interpolated_features.append(feat_interp)
            
            # Average all scale features
            wavelet_combined = torch.stack(interpolated_features, dim=0).mean(dim=0)  # [B, L, D]
        else:
            wavelet_combined = x
        
        # 3. Apply learnable transformation
        # Reshape and apply filter matrix
        wavelet_reshaped = wavelet_combined.reshape(B * L, D)  # [B*L, D]
        wavelet_filtered = torch.mm(wavelet_reshaped, self.wavelet_filter_W)  # [B*L, D]
        wavelet_filtered = wavelet_filtered.reshape(B, L, D)  # [B, L, D]
        
        # Linear mapping
        wavelet_mapped = self.wavelet_linear(wavelet_filtered)  # [B, L, D]
        
        # 4. Apply learnable wavelet reconstruction parameters
        A = F.softplus(self.wavelet_A.unsqueeze(0).unsqueeze(0))  # [1, 1, D]
        scale = F.softplus(self.wavelet_scale.unsqueeze(0).unsqueeze(0))  # [1, 1, D]
        shift = torch.tanh(self.wavelet_shift.unsqueeze(0).unsqueeze(0))  # [1, 1, D]
        
        # Apply transformation
        wavelet_reconstructed = A * torch.tanh(scale * wavelet_mapped + shift)  # [B, L, D]
        
        # 5. Process classification and regression features separately
        # Classification features (1st) - combine with wavelet enhancement
        class_feature = x[:, :, 0:1]  # [B, L, 1]
        class_wavelet = wavelet_reconstructed[:, :, 0:1]  # [B, L, 1]
        
        # Classification feature enhancement (combine original features and wavelet features)
        class_enhanced = class_feature + torch.tanh(class_wavelet)
        
        # Regression features (2nd-39th) - combine with wavelet
        reg_features = x[:, :, 1:]  # [B, L, D-1]
        reg_wavelet = wavelet_reconstructed[:, :, 1:]  # [B, L, D-1]
        
        # Regression feature processing (combine original features and wavelet features)
        reg_enhanced = reg_features + 0.1 * torch.tanh(reg_wavelet)
        
        # 6. Combine output
        output = torch.cat([class_enhanced, reg_enhanced], dim=-1)  # [B, L, D]
        
        return output

    def exo_rbf_transform(self, x):
        """Apply RBF transformation to exogenous_x, interpolate first then extract RBF features (efficient vectorized version)"""
        # x: [B, L, N] -> [B, L, N]
        B, L, N = x.shape
        
        # Get RBF centers
        if hasattr(self, 'exo_rbf_centers_param'):
            rbf_centers = self.exo_rbf_centers_param
        else:
            rbf_centers = self.exo_rbf_centers_buffer
        
        # Create original missing value mask (0 values indicate missing)
        original_mask = (x == 0).float()  # [B, L, N]
        
        # Fast interpolation: use combination of forward fill and backward fill
        x_interpolated = x.clone()
        
        # Reshape to [B*N, L] for batch processing
        x_reshaped = x.reshape(B * N, L)  # [B*N, L]
        
        # Create missing value mask
        missing_mask = (x_reshaped == 0)  # [B*N, L]
        
        # Forward fill
        x_forward = x_reshaped.clone()
        for i in range(1, L):
            x_forward[:, i] = torch.where(
                missing_mask[:, i],
                x_forward[:, i-1],
                x_forward[:, i]
            )
        
        # Backward fill
        x_backward = x_reshaped.clone()
        for i in range(L-2, -1, -1):
            x_backward[:, i] = torch.where(
                missing_mask[:, i],
                x_backward[:, i+1],
                x_backward[:, i]
            )
        
        # Take average of forward and backward fill
        x_interpolated_reshaped = (x_forward + x_backward) / 2
        
        # Restore original shape
        x_interpolated = x_interpolated_reshaped.view(B, L, N)
        
        # Vectorized RBF transformation
        # [B, L, N, 1] - [rbf_centers] -> [B, L, N, rbf_centers]
        distances = x_interpolated.unsqueeze(-1) - rbf_centers.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        # Apply RBF kernel function
        rbf_output = torch.exp(-self.exo_rbf_gamma * distances ** 2)  # [B, L, N, rbf_centers]
        
        # Average over RBF center dimensions
        rbf_features = rbf_output.mean(dim=-1)  # [B, L, N]
        
        return rbf_features, original_mask


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
        
        # vendo_embed, n_vars = self.en_embedding(endogenous_x)
        # endo_embed, attns = self.en_encoder(endo_embed, attn_mask=None)
        # endo_embed = endo_embed[:, 0, :].unsqueeze(1)  # B, 1, d_model, global token
                    
        # Apply RBF transformation to MODIS data
        exo_modis, original_mask_tensor = self.exo_rbf_transform(exogenous_x[:, :, 19:])  # [B, L, 18]
        exogenous_x = torch.cat([exogenous_x[:, :, :19], exo_modis], dim=2)
        
        # Apply Fourier transform and wavelet transform to ERA5 and RBF-processed MODIS data, and extract features to concatenate as model input
        # enc_out = self.enc_embedding(torch.cat([endogenous_x.permute(0, 2, 1), exogenous_x], dim=2), x_mark_enc)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # 当enc_embedding内是x_enc时，ex0_weather是前12个变量；但当x_enc是endogenous_x时，exo_weather是1-13个变量
        # exo_weather = enc_out[:, :, 0:12]  # ERA5 data (first 12 variables)
        # exo_modis_rbf = enc_out[:, :, 19:]  # RBF-processed MODIS data
        exo_weather = enc_out[:, :, 1:13]
        exo_modis_rbf = enc_out[:, :, 20:]
        
        if self.fourier_as_features:
            # Apply Fourier transform to ERA5 and MODIS data separately, and extract features to concatenate as model input
            # Apply Fourier transform
            fourier_weather = self.fourier_transform_features(exo_weather)  # [B, 8, 12]
            fourier_modis = self.fourier_transform_features(exo_modis_rbf)  # [B, 8, 18]
            
            # Apply wavelet transform
            wavelet_weather = self.wavelet_transform_features(exo_weather)  # [B, 6, 12]
            wavelet_modis = self.wavelet_transform_features(exo_modis_rbf)  # [B, 6, 18]
            
            # Concatenate transformed features with original data
            # Original data: [B, L, 38] (20 original variables + 18 RBF features)
            # Fourier features: [B, 4, 30] (4 features × 30 variables)
            # Wavelet features: [B, 2, 30] (2 features × 30 variables)
            
            # Concatenate Fourier features
            fourier_combined = torch.cat([fourier_weather, fourier_modis], dim=2)  # [B, 4, 30]
            # Expand Fourier features to time dimension
            fourier_expanded = fourier_combined.unsqueeze(1).expand(-1, exogenous_x.shape[1], -1, -1)  # [B, L, 4, 30]
            fourier_expanded = fourier_expanded.reshape(exogenous_x.shape[0], exogenous_x.shape[1], -1)  # [B, L, 120]
            
            # Concatenate wavelet features
            wavelet_combined = torch.cat([wavelet_weather, wavelet_modis], dim=2)  # [B, 2, 30]
            # Expand wavelet features to time dimension
            wavelet_expanded = wavelet_combined.unsqueeze(1).expand(-1, exogenous_x.shape[1], -1, -1)  # [B, L, 2, 30]
            wavelet_expanded = wavelet_expanded.reshape(exogenous_x.shape[0], exogenous_x.shape[1], -1)  # [B, L, 60]
            
            # Final concatenation: original data + Fourier features + wavelet features. To disable wavelet and Fourier, comment out this line
            # 如果enc_embedding内是x_enc时，则不需要catendo_embed
            # enc_out = torch.cat([endo_embed, enc_out, wavelet_expanded, fourier_expanded], dim=2)  
            enc_out = torch.cat([enc_out, wavelet_expanded, fourier_expanded], dim=2)  
        elif self.fft_ifft:
            # Apply Fourier transform to ERA5 and MODIS data separately, apply learnable frequency domain transformation matrices, inverse Fourier transform back to time domain, replace back to original positions
            # exo_weather: [B, L, 12], exo_modis_rbf: [B, L, N] (N may vary)
            
            # Dynamically detect MODIS data dimensions
            modis_dim = exo_modis_rbf.shape[-1]
            
            # Fourier transform to frequency domain
            fft_weather = torch.fft.fft(exo_weather, dim=1)  # [B, L, 12]
            fft_modis = torch.fft.fft(exo_modis_rbf, dim=1)  # [B, L, N]
            
            # Apply learnable frequency domain transformation (using magnitude and phase)
            # For ERA5 data
            magnitude_weather = torch.abs(fft_weather)  # [B, L, 12]
            phase_weather = torch.angle(fft_weather)    # [B, L, 12]
            
            # Apply learnable magnitude and phase transformation
            magnitude_weather_transformed = magnitude_weather * torch.sigmoid(self.freq_transform_weather_mag.unsqueeze(0).unsqueeze(0))
            phase_weather_transformed = phase_weather + self.freq_transform_weather_phase.unsqueeze(0).unsqueeze(0)
            
            # Reconstruct complex numbers
            fft_weather_transformed = magnitude_weather_transformed * torch.exp(1j * phase_weather_transformed)
            
            # For MODIS data - use dynamically adjusted parameters
            magnitude_modis = torch.abs(fft_modis)  # [B, L, N]
            phase_modis = torch.angle(fft_modis)    # [B, L, N]
            
            # Dynamically adjust transformation parameter dimensions
            if modis_dim <= self.max_modis_dim:
                # Use first N parameters
                mag_params = self.freq_transform_modis_mag[:modis_dim]
                phase_params = self.freq_transform_modis_phase[:modis_dim]
            else:
                # If exceeding maximum dimension, use interpolation
                indices = torch.linspace(0, self.max_modis_dim-1, modis_dim, device=self.freq_transform_modis_mag.device)
                mag_params = torch.interp(indices, torch.arange(self.max_modis_dim, device=self.freq_transform_modis_mag.device), self.freq_transform_modis_mag)
                phase_params = torch.interp(indices, torch.arange(self.max_modis_dim, device=self.freq_transform_modis_phase.device), self.freq_transform_modis_phase)
            
            # Apply learnable magnitude and phase transformation
            magnitude_modis_transformed = magnitude_modis * torch.sigmoid(mag_params.unsqueeze(0).unsqueeze(0))
            phase_modis_transformed = phase_modis + phase_params.unsqueeze(0).unsqueeze(0)
            
            # Reconstruct complex numbers
            fft_modis_transformed = magnitude_modis_transformed * torch.exp(1j * phase_modis_transformed)
            
            # Inverse Fourier transform back to time domain
            exo_weather_transformed = torch.fft.ifft(fft_weather_transformed, dim=1).real  # [B, L, 12]
            exo_modis_transformed = torch.fft.ifft(fft_modis_transformed, dim=1).real  # [B, L, N]
            
            # Replace transformed data back to original positions (avoid in-place operations)
            enc_out_transformed = enc_out.clone()
            # enc_out_transformed[:, :, :12] = exo_weather_transformed  # Replace ERA5 data
            # enc_out_transformed[:, :, 19:] = exo_modis_transformed     # Replace MODIS data
            enc_out_transformed[:, :, 1:13] = exo_weather_transformed  # Replace ERA5 data
            enc_out_transformed[:, :, 20:] = exo_modis_transformed     # Replace MODIS data
            
            enc = torch.cat([endo_embed, enc_out_transformed], dim=1)
        elif self.affirm_transform:
            enc_out_weather = self.affirm_adaptive_filter(exo_weather)
            enc_out_modis = self.affirm_adaptive_filter(exo_modis_rbf)
            # Avoid in-place operations, use clone()
            enc_out_transformed = enc_out.clone()
            # enc_out_transformed[:, :, 0:12] = enc_out_weather
            # enc_out_transformed[:, :, 19:] = enc_out_modis
            enc_out_transformed[:, :, 1:13] = enc_out_weather
            enc_out_transformed[:, :, 20:] = enc_out_modis
            enc = torch.cat([endo_embed, enc_out_transformed], dim=1)
        elif self.wavelet_transform:
            # Use adaptive wavelet filtering
            enc_out_weather = self.adaptive_wavelet_filter(exo_weather)
            enc_out_modis = self.adaptive_wavelet_filter(exo_modis_rbf)
            # Avoid in-place operations, use clone()
            enc_out_transformed = enc_out.clone()
            # enc_out_transformed[:, :, 0:12] = enc_out_weather
            # enc_out_transformed[:, :, 19:] = enc_out_modis
            enc_out_transformed[:, :, 1:13] = enc_out_weather
            enc_out_transformed[:, :, 20:] = enc_out_modis
            enc = torch.cat([endo_embed, enc_out_transformed], dim=1)
        else:
            # Default case: direct concatenation
            # enc = torch.cat([endo_embed, enc_out], dim=1)
            enc = enc_out

        
        enc_out, attns = self.encoder(enc, attn_mask=None)
        
        # Apply FFT-ILT transformation to encoder output
        if self.fft_ilt:
            enc_out = self.fft_ilt_transform(enc_out)  # [B, L, D]
        elif self.wavelet_output:
            enc_out = self.wavelet_output_transform(enc_out)
        
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
