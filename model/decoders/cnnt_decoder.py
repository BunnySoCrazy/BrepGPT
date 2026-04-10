import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block with dropout: Linear -> BN -> ReLU -> Dropout -> Linear -> BN."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class CnntDecoder(nn.Module):
    """
    Predicts edge existence between node pairs from quantized node features.

    Takes per-node features and candidate edge pairs; returns a logit per edge.
    """

    def __init__(self, node_feature_dim=16, hidden_dim=64):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        self.output_blocks = nn.Sequential(
            ResBlock(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edges):
        u = x[edges[:, 0]]
        v = x[edges[:, 1]]

        u = self.input_proj(u)
        v = self.input_proj(v)

        # Asymmetric difference captures directional edge signal
        feature_len = u.shape[1] // 2
        diff_u = u[:, :feature_len] - v[:, feature_len:]
        diff_v = v[:, :feature_len] - u[:, feature_len:]

        combined = torch.cat([diff_u, diff_v], dim=1)
        return self.output_blocks(combined)
