import torch
import torch.nn as nn


class FrameQualityEstimator(nn.Module):
    def __init__(self, dim, reduction=4, dropout=0.0):
        super(FrameQualityEstimator, self).__init__()
        hidden_dim = max(dim // reduction, 1)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class QualityWeightedPooling(nn.Module):
    def __init__(
        self,
        dim,
        reduction=4,
        dropout=0.0,
        temperature=1.0,
        return_weights=False,
        mode="plain",
        alpha=1.0,
    ):
        super(QualityWeightedPooling, self).__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if mode not in ("plain", "residual"):
            raise ValueError("mode must be 'plain' or 'residual'")
        self.estimator = FrameQualityEstimator(dim, reduction=reduction, dropout=dropout)
        self.temperature = temperature
        self.return_weights = return_weights
        self.mode = mode
        self.alpha = alpha

    def forward(self, x, mask=None, return_weights=None):
        if x.dim() != 3:
            raise ValueError("QualityWeightedPooling expects x with shape [B, T, C]")

        mean_pool = x.mean(dim=1)
        score = self.estimator(x) / self.temperature
        if mask is not None:
            mask = mask.to(device=x.device, dtype=torch.bool)
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            score = score.masked_fill(~mask, torch.finfo(score.dtype).min)

        weights = torch.softmax(score, dim=1)
        qpool = torch.sum(weights * x, dim=1)
        if self.mode == "plain":
            pooled = qpool
        else:
            pooled = mean_pool + self.alpha * (qpool - mean_pool)
        weights = weights.squeeze(-1)

        should_return_weights = self.return_weights if return_weights is None else return_weights
        if should_return_weights:
            return pooled, weights
        return pooled
