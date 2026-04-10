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


class VHPDecoder(nn.Module):
    """
    Decodes quantized node features into B-rep half-patch geometry.

    Given per-node features, edge pairs, and node positions, produces:
      - curve line samples  [num_edges, 6, 1, 3]
      - surface patch samples [num_edges, 6, 3, 3]
      - next half-edge samples [num_edges, 4, 3]
      - edge existence logit [num_edges, 1]
    All output heads are zero-initialized.
    """

    def __init__(self, node_feature_dim, hidden_dim=256):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.curve_samples = 6
        self.normal_samples = 4

        input_dim = 2 * node_feature_dim + 3  # start + end node features + relative position

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        self.backbone = nn.Sequential(
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim)
        )

        # Curve samples: [num_edges, 6, 1, 3]
        self.curve_line_head = nn.Sequential(
            ResBlock(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, self.curve_samples * 3)
        )

        # Surface patch samples: [num_edges, 6, 3, 3]
        self.surface_patch_head = nn.Sequential(
            ResBlock(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, self.curve_samples * 3 * 3)
        )

        # Next half-edge samples: [num_edges, 4, 3]
        self.next_edge_head = nn.Sequential(
            ResBlock(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, self.normal_samples * 3)
        )

        # Edge existence logit: [num_edges, 1]
        self.edge_head = nn.Sequential(
            ResBlock(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Zero-init output layers for stable early training
        for head in [self.curve_line_head, self.surface_patch_head,
                     self.next_edge_head, self.edge_head]:
            last_linear = head[-1]
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

    def forward(self, node_features, edges, pos):
        start_nodes = edges[:, 0]
        end_nodes = edges[:, 1]

        start_features = node_features[start_nodes]
        end_features = node_features[end_nodes]

        start_pos = pos[start_nodes]
        end_pos = pos[end_nodes]

        edge_features = torch.cat([start_features, end_features, end_pos - start_pos], dim=-1)

        x = self.input_proj(edge_features)
        hidden = self.backbone(x)

        # Curve line: [num_edges, 6, 1, 3]
        curve_line_output = self.curve_line_head(hidden)
        curve_line_output = curve_line_output.view(-1, self.curve_samples, 1, 3)

        # Surface patch: [num_edges, 6, 3, 3]
        surface_patch_output = self.surface_patch_head(hidden)
        surface_patch_output = surface_patch_output.view(-1, self.curve_samples, 3, 3)

        # Next half-edge: [num_edges, 4, 3]
        next_edge_output = self.next_edge_head(hidden)
        next_edge_output = next_edge_output.view(-1, self.normal_samples, 3)

        # Combine curve and surface into half-patch: [num_edges, 6, 4, 3]
        half_patch_output = torch.cat([curve_line_output, surface_patch_output], dim=2)

        # Full geometry output: [num_edges, 7, 4, 3]
        geom_output = torch.cat([
            half_patch_output,
            next_edge_output.unsqueeze(1)
        ], dim=1)

        edge_output = self.edge_head(hidden)

        return geom_output, edge_output
