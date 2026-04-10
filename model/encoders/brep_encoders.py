import torch
from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import NNConv
from dgl.nn.pytorch import EdgeGATConv
from dgl.nn.pytorch import SAGEConv


def _conv1d(in_channels, out_channels, kernel_size=3, padding=0, bias=False):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(),
    )


def _conv2d(in_channels, out_channels, kernel_size, padding=0, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
    )


def _fc(in_features, out_features, bias=False):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=bias),
        nn.BatchNorm1d(out_features),
        nn.LeakyReLU(),
    )


class _MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(_MLP, self).__init__()
        self.linear_or_not = True
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("Number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class CurveEncoder(nn.Module):
    def __init__(self, in_channels=6, output_dims=64):
        """1D convolutional encoder for B-rep edge geometry (curve samples)."""
        super(CurveEncoder, self).__init__()
        self.in_channels = in_channels
        self.conv1 = _conv1d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.conv2 = _conv1d(64, 128, kernel_size=3, padding=1, bias=False)
        self.conv3 = _conv1d(128, 256, kernel_size=3, padding=1, bias=False)
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = _fc(256, output_dims, bias=False)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        assert x.size(1) == self.in_channels
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_pool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


class SurfaceEncoder(nn.Module):
    def __init__(self, in_channels=7, output_dims=64):
        """2D convolutional encoder for B-rep face geometry (surface patches)."""
        super(SurfaceEncoder, self).__init__()
        self.in_channels = in_channels
        self.conv1 = _conv2d(in_channels, 64, 3, padding=1, bias=False)
        self.conv2 = _conv2d(64, 128, 3, padding=1, bias=False)
        self.conv3 = _conv2d(128, 256, 3, padding=1, bias=False)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = _fc(256, output_dims, bias=False)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        assert x.size(1) == self.in_channels
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_pool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


class _EdgeConv(nn.Module):
    def __init__(self, edge_feats, out_feats, node_feats, num_mlp_layers=2, hidden_mlp_dim=64):
        """Updates edge features using node features at endpoints."""
        super(_EdgeConv, self).__init__()
        self.proj = _MLP(1, hidden_mlp_dim, hidden_mlp_dim, edge_feats)
        self.mlp = _MLP(num_mlp_layers, edge_feats, hidden_mlp_dim, out_feats)
        self.batchnorm = nn.BatchNorm1d(out_feats)
        self.eps = torch.nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, graph, nfeat, efeat):
        src, dst = graph.edges()
        proj1, proj2 = self.proj(nfeat[src]), self.proj(nfeat[dst])
        agg = proj1 + proj2
        h = self.mlp((1 + self.eps) * efeat + agg)
        h = F.leaky_relu(self.batchnorm(h))
        return h


class _NodeConv(nn.Module):
    def __init__(self, node_feats, out_feats, edge_feats, num_mlp_layers=2, hidden_mlp_dim=64):
        """Updates node features using neighboring node and edge features."""
        super(_NodeConv, self).__init__()
        self.gconv = NNConv(
            in_feats=node_feats,
            out_feats=node_feats,
            edge_func=nn.Linear(edge_feats, node_feats * node_feats),
            aggregator_type="sum",
            bias=False,
        )
        self.batchnorm = nn.BatchNorm1d(out_feats)
        self.mlp = _MLP(num_mlp_layers, node_feats, hidden_mlp_dim, out_feats)
        self.eps = torch.nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, graph, nfeat, efeat):
        h = (1 + self.eps) * nfeat
        h = self.gconv(graph, h, efeat)
        h = self.mlp(h)
        h = F.leaky_relu(self.batchnorm(h))
        return h


class BrepGraphEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        input_edge_dim,
        output_dim,
        hidden_dim=64,
        learn_eps=True,
        num_layers=3,
        num_mlp_layers=2,
    ):
        super(BrepGraphEncoder, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        self.node_conv_layers = torch.nn.ModuleList()
        self.edge_conv_layers = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            node_feats = input_dim if layer == 0 else hidden_dim
            edge_feats = input_edge_dim if layer == 0 else hidden_dim
            self.node_conv_layers.append(
                _NodeConv(
                    node_feats=node_feats,
                    out_feats=hidden_dim,
                    edge_feats=edge_feats,
                    num_mlp_layers=num_mlp_layers,
                    hidden_mlp_dim=hidden_dim,
                ),
            )
            if layer < self.num_layers - 2:
                self.edge_conv_layers.append(
                    _EdgeConv(
                        edge_feats=edge_feats,
                        out_feats=hidden_dim,
                        node_feats=node_feats,
                        num_mlp_layers=num_mlp_layers,
                        hidden_mlp_dim=hidden_dim,
                    )
                )

        self.final_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, g, h, efeat):
        he = efeat
        for i in range(self.num_layers - 1):
            h = self.node_conv_layers[i](g, h, he)
            if i < self.num_layers - 2:
                he = self.edge_conv_layers[i](g, h, he)
        out = self.final_linear(h)
        return out, None


class SAGEEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=3, dropout=0.1, aggr='mean'):
        super(SAGEEncoder, self).__init__()
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.skip_layer = nn.Linear(hidden_dim, hidden_dim)
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.num_layers = num_layers

        for _ in range(num_layers):
            self.layers.append(SAGEConv(in_feats=hidden_dim, out_feats=hidden_dim, aggregator_type=aggr))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, h, efeat=None):
        h = self.node_encoder(h)
        middle_layer = self.num_layers // 2

        for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
            if i == 0:
                first_h = h
            h_res = h
            h = layer(g, h)
            h = F.relu(h)
            h = self.dropout(h)
            h = norm(h + h_res)
            if i == middle_layer:
                h = h + self.skip_layer(first_h)

        out = self.output_layer(h)
        return out, None


class GATEncoder(nn.Module):
    def __init__(self, input_dim, input_edge_dim, output_dim, hidden_dim=64, num_layers=3, num_heads=4, dropout=0.1):
        super(GATEncoder, self).__init__()
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.edge_encoder = nn.Linear(input_edge_dim, hidden_dim)
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(
                EdgeGATConv(
                    in_feats=hidden_dim,
                    edge_feats=hidden_dim,
                    out_feats=hidden_dim // num_heads,
                    num_heads=num_heads,
                    feat_drop=dropout,
                    attn_drop=dropout,
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, h, efeat):
        h = self.node_encoder(h)
        efeat = self.edge_encoder(efeat)
        for layer, norm in zip(self.layers, self.layer_norms):
            h_res = h
            h = layer(g, h, edge_feat=efeat)
            h = h.flatten(1)
            h = F.relu(h)
            h = self.dropout(h)
            h = norm(h + h_res)
        out = self.output_layer(h)
        return out, None


def get_encoder(encoder_type):
    encoder_dict = {
        'gat': GATEncoder,
        'sage': SAGEEncoder,
    }
    return encoder_dict[encoder_type]


class ConnectEncoder(nn.Module):
    def __init__(
        self,
        num_classes=1,
        crv_in_channels=6,
        crv_emb_dim=8,
        srf_emb_dim=8,
        graph_emb_dim=128,
        hidden_dim=64,
        dropout=0.1,
        num_layers=3,
        encoder_type='sage'
    ):
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type == 'gat':
            self.curv_encoder = CurveEncoder(in_channels=6, output_dims=crv_emb_dim)

        self.vertex_encoder = CurveEncoder(in_channels=3, output_dims=srf_emb_dim)

        Encoder = get_encoder(encoder_type)
        if encoder_type == 'gat':
            self.graph_encoder = Encoder(srf_emb_dim, crv_emb_dim, graph_emb_dim, hidden_dim, num_layers=num_layers)
        else:  # sage
            self.graph_encoder = Encoder(srf_emb_dim, graph_emb_dim, hidden_dim, num_layers=num_layers)

    def forward(self, batched_graph):
        if self.encoder_type == 'gat':
            src, dst = batched_graph.edges()
            src_coords = batched_graph.ndata["x"][src]
            dst_coords = batched_graph.ndata["x"][dst]
            edge_coords = torch.cat([src_coords, dst_coords], dim=-1)
            input_crv_feat = edge_coords.unsqueeze(-1)

        input_vertex_feat = batched_graph.ndata["x"].unsqueeze(-1)

        if self.encoder_type == 'gat':
            hidden_crv_feat = self.curv_encoder(input_crv_feat)
        else:
            hidden_crv_feat = None

        hidden_vertex_feat = self.vertex_encoder(input_vertex_feat)

        node_emb, graph_emb = self.graph_encoder(batched_graph, hidden_vertex_feat, hidden_crv_feat)
        return node_emb


class VHPEncoder(nn.Module):
    def __init__(
        self,
        crv_in_channels=6,
        crv_emb_dim=8,
        vertex_emb_dim=8,
        graph_emb_dim=128,
        hidden_dim=64,
        num_layers=3,
        encoder_type='gat'
    ):
        super().__init__()
        self.edge_type_embedding = nn.Embedding(2, crv_emb_dim)

        self.curve_line_encoder = CurveEncoder(in_channels=3, output_dims=crv_emb_dim)
        self.surface_patch_encoder = SurfaceEncoder(in_channels=3, output_dims=crv_emb_dim)
        self.next_edge_encoder = CurveEncoder(in_channels=3, output_dims=crv_emb_dim)
        self.vertex_encoder = CurveEncoder(in_channels=3, output_dims=vertex_emb_dim)

        self.edge_fusion = nn.Sequential(
            nn.Linear(crv_emb_dim * 4, crv_emb_dim),
            nn.LayerNorm(crv_emb_dim),
            nn.GELU()
        )

        Encoder = get_encoder(encoder_type)
        if encoder_type == 'gat':
            self.graph_encoder = Encoder(vertex_emb_dim, crv_emb_dim, graph_emb_dim, hidden_dim, num_layers=num_layers)

    def forward(self, batched_graph):
        """
        batched_graph.ndata["x"]              [B, 3]
        batched_graph.edata["x"]              [B, 6, 4, 3]
        batched_graph.edata["next_half_edge"] [B, 4, 3]
        batched_graph.edata["edge_inner_outer"] [B, 1]
        """
        edge_data = batched_graph.edata["x"]          # [B, 6, 4, 3]
        curve_line_data = edge_data[:, :, 0, :]       # [B, 6, 3]
        surface_patch_data = edge_data[:, :, 1:, :]   # [B, 6, 3, 3]
        next_half_edge_data = batched_graph.edata["next_half_edge"]  # [B, 4, 3]

        edge_types = batched_graph.edata["edge_inner_outer"].long()
        edge_type_emb = self.edge_type_embedding(edge_types.squeeze(-1))

        curve_line_feat = self.curve_line_encoder(curve_line_data.permute(0, 2, 1))
        surface_patch_feat = self.surface_patch_encoder(surface_patch_data.permute(0, 3, 1, 2))
        next_edge_feat = self.next_edge_encoder(next_half_edge_data.permute(0, 2, 1))

        combined_edge_feat = torch.cat([curve_line_feat, surface_patch_feat, next_edge_feat, edge_type_emb], dim=-1)
        hidden_crv_feat = self.edge_fusion(combined_edge_feat)

        input_vertex_feat = batched_graph.ndata["x"].unsqueeze(-1)
        hidden_vertex_feat = self.vertex_encoder(input_vertex_feat)

        node_emb, _ = self.graph_encoder(batched_graph, hidden_vertex_feat, hidden_crv_feat)
        return node_emb
