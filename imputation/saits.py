"""
SAITS model for time-series imputation.

If you use code in this repository, please cite our paper as below. Many thanks.

@article{DU2023SAITS,
title = {{SAITS: Self-Attention-based Imputation for Time Series}},
journal = {Expert Systems with Applications},
volume = {219},
pages = {119619},
year = {2023},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2023.119619},
url = {https://www.sciencedirect.com/science/article/pii/S0957417423001203},
author = {Wenjie Du and David Cote and Yan Liu},
}

or

Wenjie Du, David Cote, and Yan Liu. SAITS: Self-Attention-based Imputation for Time Series. Expert Systems with Applications, 219:119619, 2023. https://doi.org/10.1016/j.eswa.2023.119619

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: MIT


from layers import *
from utils import masked_mae_cal


class SAITS(nn.Module):
    def __init__(
        self,
        n_groups,
        n_group_inner_layers,
        d_time,
        d_feature,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout,
        **kwargs
    ):
        super().__init__()
        self.n_groups = n_groups
        self.n_group_inner_layers = n_group_inner_layers
        self.input_with_mask = kwargs["input_with_mask"]
        actual_d_feature = d_feature * 2 if self.input_with_mask else d_feature
        self.param_sharing_strategy = kwargs["param_sharing_strategy"]
        self.MIT = kwargs["MIT"]
        self.device = kwargs["device"]

        if kwargs["param_sharing_strategy"] == "between_group":
            # For between_group, only need to create 1 group and repeat n_groups times while forwarding
            self.layer_stack_for_first_block = nn.ModuleList(
                [
                    EncoderLayer(
                        d_time,
                        actual_d_feature,
                        d_model,
                        d_inner,
                        n_head,
                        d_k,
                        d_v,
                        dropout,
                        0,
                        **kwargs
                    )
                    for _ in range(n_group_inner_layers)
                ]
            )
            self.layer_stack_for_second_block = nn.ModuleList(
                [
                    EncoderLayer(
                        d_time,
                        actual_d_feature,
                        d_model,
                        d_inner,
                        n_head,
                        d_k,
                        d_v,
                        dropout,
                        0,
                        **kwargs
                    )
                    for _ in range(n_group_inner_layers)
                ]
            )
        else:  # then inner_group，inner_group is the way used in ALBERT
            # For inner_group, only need to create n_groups layers
            # and repeat n_group_inner_layers times in each group while forwarding
            self.layer_stack_for_first_block = nn.ModuleList(
                [
                    EncoderLayer(
                        d_time,
                        actual_d_feature,
                        d_model,
                        d_inner,
                        n_head,
                        d_k,
                        d_v,
                        dropout,
                        0,
                        **kwargs
                    )
                    for _ in range(n_groups)
                ]
            )
            self.layer_stack_for_second_block = nn.ModuleList(
                [
                    EncoderLayer(
                        d_time,
                        actual_d_feature,
                        d_model,
                        d_inner,
                        n_head,
                        d_k,
                        d_v,
                        dropout,
                        0,
                        **kwargs
                    )
                    for _ in range(n_groups)
                ]
            )

        self.dropout = nn.Dropout(p=dropout)
        self.position_enc = PositionalEncoding(d_model, n_position=d_time)
        # for the 1st block
        self.embedding_1 = nn.Linear(actual_d_feature, d_model)
        self.reduce_dim_z = nn.Linear(d_model, d_feature)
        # for the 2nd block
        self.embedding_2 = nn.Linear(actual_d_feature, d_model)
        self.reduce_dim_beta = nn.Linear(d_model, d_feature)
        self.reduce_dim_gamma = nn.Linear(d_feature, d_feature)
        # for the 3rd block
        self.weight_combine = nn.Linear(d_feature + d_time, d_feature)

    def _normalize_batch(self, X: torch.Tensor, observed_mask: torch.Tensor):
        """Per-sample per-channel normalization using mask to define valid values.
        - Channel 0 (FIRMS): y = log1p(x)/log1p(100) for valid (mask=1 and x>=0), else 0
        - Others (1..D-1): robust z-score with median/MAD (fallback to masked std when MAD too small/NaN),
          then apply tanh(y/3.0) to bound heavy tails. Invalid positions set to 0.
        Returns X_norm and stats dict for inverse.
        """
        eps = 1e-6
        B, L, D = X.shape
        device = X.device

        # Channel 0
        x0 = X[:, :, 0]
        m0 = observed_mask[:, :, 0].bool()
        valid0 = m0 & (x0 >= 0)
        logC = torch.log1p(torch.tensor(100.0, device=device, dtype=X.dtype))
        y0 = torch.log1p(torch.where(valid0, x0, torch.zeros_like(x0))) / logC

        if D > 1:
            rest = X[:, :, 1:]
            m_rest = observed_mask[:, :, 1:].bool()
            mask = m_rest.float()
            # masked stats over time
            x_masked = rest.masked_fill(~m_rest, float('nan'))
            med = torch.nanmedian(x_masked, dim=1).values                     # [B, D-1]
            mad = torch.nanmedian((x_masked - med.unsqueeze(1)).abs(), dim=1).values
            scale_base = mad * 1.4826
            count = mask.sum(dim=1).clamp_min(1.0)
            mean_rest = (rest * mask).sum(dim=1) / count
            var_rest = (((rest - mean_rest.unsqueeze(1)) ** 2) * mask).sum(dim=1) / count
            std_rest = torch.sqrt(torch.relu(var_rest) + eps)
            scale_base = torch.nan_to_num(scale_base, nan=0.0)
            std_rest = torch.nan_to_num(std_rest, nan=1.0)
            use_std = (scale_base <= 1e-3) | torch.isnan(scale_base)
            scale = torch.where(use_std, std_rest, scale_base) + eps           # [B, D-1]
            # few valid points -> neutral stats
            too_few = (count < 3.0)
            med = torch.where(too_few, torch.zeros_like(med), med)
            scale = torch.where(too_few, torch.ones_like(scale), scale)
            # normalize and bound
            y_rest = (rest - med.unsqueeze(1)) / scale.unsqueeze(1)
            y_rest = torch.tanh(y_rest / 3.0)
            y_rest = torch.where(m_rest, y_rest, torch.zeros_like(y_rest))
            y_rest = y_rest * (~too_few).unsqueeze(1).float()
            X_norm = torch.cat([y0.unsqueeze(-1), y_rest], dim=2)
            stats = {
                "firms_logC": logC,
                "rest_med": med.unsqueeze(1).detach(),     # [B,1,D-1]
                "rest_scale": scale.unsqueeze(1).detach(), # [B,1,D-1]
            }
        else:
            X_norm = y0.unsqueeze(-1)
            stats = {"firms_logC": logC}

        return X_norm, stats

    def _apply_norm_with_stats(self, X: torch.Tensor, stats: dict, observed_mask: torch.Tensor):
        eps = 1e-6
        B, L, D = X.shape
        # Channel 0
        x0 = X[:, :, 0]
        m0 = observed_mask[:, :, 0].bool()
        valid0 = m0 & (x0 >= 0)
        logC = stats.get("firms_logC", torch.log1p(torch.tensor(100.0, device=X.device, dtype=X.dtype)))
        y0 = torch.log1p(torch.where(valid0, x0, torch.zeros_like(x0))) / logC
        if D > 1 and ("rest_med" in stats and "rest_scale" in stats):
            rest = X[:, :, 1:]
            m_rest = observed_mask[:, :, 1:].bool()
            y_rest = (rest - stats["rest_med"]) / stats["rest_scale"]
            y_rest = torch.tanh(y_rest / 3.0)
            y_rest = torch.where(m_rest, y_rest, torch.zeros_like(y_rest))
            return torch.cat([y0.unsqueeze(-1), y_rest], dim=2)
        return y0.unsqueeze(-1)

    def _denormalize_batch(self, X_norm: torch.Tensor, stats: dict):
        """Invert normalization back to raw domain.
        - Channel 0: expm1(y*logC)
        - Others: atanh(y)*3*scale + med
        """
        eps = 1e-6
        B, L, D = X_norm.shape
        # ch 0
        y0 = X_norm[:, :, 0]
        x0 = torch.expm1(y0 * stats["firms_logC"]) if "firms_logC" in stats else y0
        if D > 1 and ("rest_med" in stats and "rest_scale" in stats):
            y_rest = X_norm[:, :, 1:]
            y_adj = y_rest * (1.0 - 2.0 * eps) + eps
            atanh = 0.5 * (torch.log1p(y_adj) - torch.log1p(-y_adj))
            z = atanh * 3.0
            x_rest = z * stats["rest_scale"] + stats["rest_med"]
            return torch.cat([x0.unsqueeze(-1), x_rest], dim=2)
        return x0.unsqueeze(-1)

    def impute(self, inputs):
        X, masks = inputs["X"], inputs["missing_mask"]
        # the first DMSA block
        input_X_for_first = torch.cat([X, masks], dim=2) if self.input_with_mask else X
        input_X_for_first = self.embedding_1(input_X_for_first)
        enc_output = self.dropout(
            self.position_enc(input_X_for_first)
        )  # namely term e in math algo
        if self.param_sharing_strategy == "between_group":
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_first_block:
                    enc_output, _ = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack_for_first_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, _ = encoder_layer(enc_output)

        X_tilde_1 = self.reduce_dim_z(enc_output)
        X_prime = masks * X + (1 - masks) * X_tilde_1

        # the second DMSA block
        input_X_for_second = (
            torch.cat([X_prime, masks], dim=2) if self.input_with_mask else X_prime
        )
        input_X_for_second = self.embedding_2(input_X_for_second)
        enc_output = self.position_enc(
            input_X_for_second
        )  # namely term alpha in math algo
        if self.param_sharing_strategy == "between_group":
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_second_block:
                    enc_output, attn_weights = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack_for_second_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, attn_weights = encoder_layer(enc_output)

        X_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(enc_output)))

        # the attention-weighted combination block
        attn_weights = attn_weights.squeeze(dim=1)  # namely term A_hat in math algo
        if len(attn_weights.shape) == 4:
            # if having more than 1 head, then average attention weights from all heads
            attn_weights = torch.transpose(attn_weights, 1, 3)
            attn_weights = attn_weights.mean(dim=3)
            attn_weights = torch.transpose(attn_weights, 1, 2)

        combining_weights = F.sigmoid(
            self.weight_combine(torch.cat([masks, attn_weights], dim=2))
        )  # namely term eta
        # combine X_tilde_1 and X_tilde_2
        X_tilde_3 = (1 - combining_weights) * X_tilde_2 + combining_weights * X_tilde_1
        # replace non-missing part with original data
        X_c = masks * X + (1 - masks) * X_tilde_3
        return X_c, [X_tilde_1, X_tilde_2, X_tilde_3]

    def forward(self, inputs, stage):
        # move inputs to the same device as model
        model_device = self.embedding_1.weight.device
        X = inputs["X"].to(model_device)
        masks = inputs["missing_mask"].to(model_device)
        if "X_holdout" in inputs:
            inputs["X_holdout"] = inputs["X_holdout"].to(model_device)
        if "indicating_mask" in inputs:
            inputs["indicating_mask"] = inputs["indicating_mask"].to(model_device)
        # Optional reconstruction mask: if provided, use it for reconstruction loss; otherwise fall back to masks
        recon_mask = inputs.get("reconstruction_mask", None)
        if recon_mask is not None:
            recon_mask = recon_mask.to(model_device)
        else:
            recon_mask = masks
        # Reconstruct original observed mask (1 for originally observed): masks_after OR indicating_mask
        indicating = inputs.get("indicating_mask", torch.zeros_like(masks))
        original_observed = torch.clamp(masks + indicating, max=1.0)

        # Normalize X with stats derived from original observed positions
        X_norm, stats = self._normalize_batch(X, original_observed)
        # Also normalize X_holdout with the SAME stats for aligned normalized-domain loss if needed
        if "X_holdout" in inputs:
            X_holdout_norm = self._apply_norm_with_stats(inputs["X_holdout"], stats, original_observed)
        else:
            X_holdout_norm = None

        reconstruction_loss = 0
        imputed_norm, [X_tilde_1, X_tilde_2, X_tilde_3] = self.impute({
            "X": X_norm,
            "missing_mask": masks,
            **{k: v for k, v in inputs.items() if k not in ["X", "missing_mask"]}
        })
        # Compute losses in normalized domain (numerically safer)
        reconstruction_loss += masked_mae_cal(X_tilde_1, X_norm, recon_mask)
        reconstruction_loss += masked_mae_cal(X_tilde_2, X_norm, recon_mask)
        final_reconstruction_MAE = masked_mae_cal(X_tilde_3, X_norm, recon_mask)
        reconstruction_loss += final_reconstruction_MAE
        reconstruction_loss /= 3        

        if (self.MIT or stage == "val") and stage != "test":
            # have to cal imputation loss in the val stage; no need to cal imputation loss here in the test stage
            target_norm = X_holdout_norm if X_holdout_norm is not None else X_norm
            imputation_MAE = masked_mae_cal(
                # compare in normalized domain on artificial-masked positions only
                X_tilde_3, target_norm, inputs["indicating_mask"]
            )
        else:
            imputation_MAE = torch.tensor(0.0)
        # Denormalize imputed output back to raw domain for users
        imputed_data = self._denormalize_batch(imputed_norm, stats)

        return {
            "imputed_data": imputed_data,
            "reconstruction_loss": reconstruction_loss,
            "imputation_loss": imputation_MAE,
            "reconstruction_MAE": final_reconstruction_MAE,
            "imputation_MAE": imputation_MAE,
        }
        
if __name__ == "__main__":
    run_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = SAITS(
        n_groups=1,
        n_group_inner_layers=1,
        d_time=10,
        d_feature=10,
        d_model=10,
        d_inner=10,
        n_head=1,
        d_k=10,
        d_v=10,
        dropout=0.1,
        input_with_mask=True,
        param_sharing_strategy="between_group",
        MIT=True,
        device=str(run_device),
        diagonal_attention_mask=True,
    )
    model = model.to(run_device)
    dummy_input = torch.randn(1, 10, 10, device=run_device)  # 
    dummy_mask = torch.randint(low=0, high=2, size=(1, 10, 10), device=run_device)
    dummy_holdout = torch.randn(1, 10, 10, device=run_device)
    dummy_indicating_mask = torch.randint(low=0, high=2, size=(1, 10, 10), device=run_device)
    # X is the masked data, masks contains both the missing data and the artificial mask
    # X_holdout is the original data without mask, indicating_mask is the artificial mask of the original data
    # X is used for reconstruction loss, X_holdout is used as the ground truth for imputation loss
    inputs = {"X": dummy_input, "missing_mask": dummy_mask, "X_holdout": dummy_holdout, "indicating_mask": dummy_indicating_mask}
    outputs = model(inputs, stage="train")
    print(outputs["imputed_data"].shape)