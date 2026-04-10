import torch
import pytorch_lightning as pl
import os
import random
import torch.nn as nn
import dgl
import pickle
import trimesh

from model.encoders.brep_encoders import VHPEncoder as Encoder
from model.decoders.vhp_decoder import VHPDecoder


def inverse_process_graph(graph, num_curve_samples=6, num_normal_samples=4):
    """De-normalize edge geometry: recover absolute coordinates from relative offsets."""
    g = graph
    src, dst = g.edges()
    g.edata['ori_edge_data'] = g.edata["x"].clone()

    for eid in range(g.number_of_edges()):
        edge_data = g.edata["x"][eid].clone()
        next_edge_data = g.edata["next_half_edge"][eid].clone()
        start_points = g.ndata["x"][src[eid]]
        end_points = g.ndata["x"][dst[eid]]

        t = torch.linspace(0, 1, num_curve_samples + 2, device=edge_data.device)[1:-1]
        interpolated_points = (
            start_points.unsqueeze(0).unsqueeze(0) * (1 - t).unsqueeze(-1).unsqueeze(0)
            + end_points.unsqueeze(0).unsqueeze(0) * t.unsqueeze(-1).unsqueeze(0)
        )

        edge_data[:, 0, :] = edge_data[:, 0, :] + interpolated_points.squeeze(0)
        for i in range(1, num_normal_samples):
            edge_data[:, i, :] = edge_data[:, i, :] + edge_data[:, i - 1, :]

        next_edge_data[0, :] = next_edge_data[0, :] + end_points
        for i in range(1, num_normal_samples):
            next_edge_data[i, :] = next_edge_data[i, :] + next_edge_data[i - 1, :]

        g.edata["x"][eid] = edge_data
        g.edata["next_half_edge"][eid] = next_edge_data

    return g


class Trainer_vhp_vq(pl.LightningModule):
    """Lightning module for Stage 2: half-patch geometry VQ-VAE."""

    def __init__(self, specs):
        super().__init__()
        self.save_hyperparameters()
        self.specs = specs
        self.experiment_directory = specs['experiment_directory']
        self.log_dir = os.path.join(specs['experiment_directory'], 'log/')

        self.build_net()

        self.lr = specs["learning_rate"]
        self.betas = specs["betas"]

        from vector_quantize_pytorch import ResidualVQ
        self.vector_quantizer = ResidualVQ(
            dim=self.specs['node_feature_dim'],
            num_quantizers=self.specs['num_codebooks'],
            codebook_size=self.specs['codebook_size'],
            threshold_ema_dead_code=0.1
        )
        self.test_output_list = []

    def build_net(self):
        self.encoder = Encoder(
            crv_emb_dim=self.specs["crv_emb_dim"],
            vertex_emb_dim=self.specs["vertex_emb_dim"],
            graph_emb_dim=self.specs['node_feature_dim'],
            hidden_dim=self.specs['hidden_dim_encode'],
            num_layers=self.specs['n_layers_encode']
        )
        self.decoder = VHPDecoder(
            node_feature_dim=self.specs['node_feature_dim'],
            hidden_dim=self.specs['hidden_dim_decode']
        )

    def configure_optimizers(self):
        param_groups = [
            {"params": self.encoder.parameters(), "lr": self.lr, "betas": tuple(self.betas)},
            {"params": self.decoder.parameters(), "lr": self.lr, "betas": tuple(self.betas)},
        ]
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }

    def training_step(self, batch, batch_idx):
        graphs = batch["graph"].to(self.device)
        pos = graphs.ndata["x"]
        pos_edges = torch.stack(graphs.edges()).t()

        continuous_features = self.encoder(graphs)
        quantized_features, indices, vq_loss = self.vector_quantizer(continuous_features.unsqueeze(1))
        quantized_features = quantized_features.squeeze(1)

        geom_output, cls_output = self.decoder(quantized_features, pos_edges, pos)

        gt_half_patch = graphs.edata["x"]
        gt_next_curve = graphs.edata["next_half_edge"]
        gt_cls = graphs.edata["edge_inner_outer"]

        pred_half_patch = geom_output[:, :-1, :, :]
        pred_curve = pred_half_patch[:, :, 0, :]
        pred_surface = pred_half_patch[:, :, 1:, :]
        gt_curve = gt_half_patch[:, :, 0, :]
        gt_surface = gt_half_patch[:, :, 1:, :]
        pred_next_curve = geom_output[:, -1, :, :]

        mse_loss = nn.MSELoss(reduction='mean')

        zero_mask = (gt_curve.abs().sum(dim=(1, 2)) == 0)
        num_zero = zero_mask.sum().item()
        num_nonzero = (~zero_mask).sum().item()
        total = gt_curve.shape[0]
        alpha = 0.7
        if num_nonzero > 0 and num_zero > 0:
            w_zero = total * (1 - alpha) / num_zero
            w_nonzero = total * alpha / num_nonzero
            weights = torch.where(zero_mask, w_zero, w_nonzero)
        else:
            weights = torch.ones(total, device=gt_curve.device)

        per_edge_loss = ((pred_curve - gt_curve) ** 2).mean(dim=(1, 2))
        x_recon_curve_loss = (per_edge_loss * weights).mean()
        x_recon_surface_loss = mse_loss(pred_surface, gt_surface)
        next_edge_reconstruction_loss = mse_loss(pred_next_curve, gt_next_curve)

        pos_weight = (1 - gt_cls.float()).sum() / (gt_cls.float().sum() + 1e-6)
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=self.device))
        classification_loss = bce_loss(cls_output[:, 0], gt_cls.float())
        pred_labels = (cls_output[:, 0] > 0).float()

        total_loss = (
            (x_recon_curve_loss + x_recon_surface_loss + next_edge_reconstruction_loss) * 10
            + 0.1 * classification_loss
            + 0.01 * vq_loss.mean()
        )

        bs = self.specs['batch_size']
        self.log("train_cls_loss", classification_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log("x_recon_curve_loss", x_recon_curve_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log("x_recon_surface_loss", x_recon_surface_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log("next_edge_recon_loss", next_edge_reconstruction_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log("cls_acc", (pred_labels == gt_cls.float()).float().mean(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log("train_vq_loss", vq_loss.mean(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log("active_codes", indices.unique().numel())

        del continuous_features, graphs, pos, pos_edges
        torch.cuda.empty_cache()

        return total_loss

    def validation_step(self, batch, batch_idx):
        return None

    def on_after_backward(self):
        torch.cuda.empty_cache()

    def test_step(self, batch, batch_idx):
        graphs = batch["graph"].to(self.device)
        pos = graphs.ndata["x"]

        continuous_features = self.encoder(graphs)
        quantized_features, indices, vq_loss = self.vector_quantizer(continuous_features.unsqueeze(1))
        indices = indices.squeeze(1)

        node_offset = 0
        for i, size in enumerate(batch['sizes']):
            node_coordinates = pos[node_offset:node_offset + size]
            current_indices = indices[node_offset:node_offset + size]
            results = {
                "filename": batch['filename'][i],
                "node_coordinates": node_coordinates.detach().cpu().numpy(),
                "indices": current_indices.detach().cpu().numpy(),
            }
            self.test_output_list.append(results)
            node_offset += size

        return results

    def on_test_epoch_end(self):
        outputs = self.test_output_list
        all_indices = [x['indices'] for x in outputs]
        all_file_name = [x['filename'] for x in outputs]
        all_node_coordinates = [x['node_coordinates'] for x in outputs]

        for i in range(len(all_indices)):
            if isinstance(all_indices[i], torch.Tensor):
                all_indices[i] = all_indices[i].detach().cpu().numpy()
            if isinstance(all_node_coordinates[i], torch.Tensor):
                all_node_coordinates[i] = all_node_coordinates[i].detach().cpu().numpy()

        save_dict = {
            'indices': all_indices,
            'filename': all_file_name,
            'node_coordinates': all_node_coordinates,
        }
        with open(f'data/{self.specs["dataset"]}_vhp_encoding.pkl', 'wb') as f:
            pickle.dump(save_dict, f)

    def decode(self, graphs, output_dir='output/decoded_graphs'):
        """
        Decode graph dicts (with VQ indices) into geometry .bin files and .ply point clouds.

        graphs: list of dicts with keys:
            node_positions, node_geometric_features, edges, filename
        """
        os.makedirs(output_dir, exist_ok=True)
        id_offset = 0
        self.decoder.eval()
        self.vector_quantizer.eval()
        total_success = 0

        for idx, graph in enumerate(graphs):
            try:
                node_geometric_features = graph['node_geometric_features']
                edges = graph['edges']
                filename = graph['filename']
                node_positions = graph['node_positions']

                device = next(self.parameters()).device
                if isinstance(node_geometric_features, torch.Tensor):
                    node_geometric_features = node_geometric_features.to(device)
                if isinstance(edges, torch.Tensor):
                    edges = edges.to(device)
                if isinstance(node_positions, torch.Tensor):
                    node_positions = node_positions.to(device)

                normalized_positions = (node_positions / 128) * 2 - 1
                node_features = self.vector_quantizer.get_output_from_indices(node_geometric_features)
                geom_output, cls_output = self.decoder(node_features, edges, normalized_positions)
                num_nodes = node_positions.shape[0]

                edges_cpu = edges.cpu()
                result_graph = dgl.graph((edges_cpu[:, 0], edges_cpu[:, 1]), num_nodes=num_nodes)
                result_graph.ndata['x'] = normalized_positions.detach().cpu()
                result_graph.edata['x'] = geom_output[:, :-1, :, :].detach().cpu()
                result_graph.edata['next_half_edge'] = geom_output[:, -1, :, :].detach().cpu()
                result_graph.edata['edge_inner_outer'] = (cls_output[:, 0] > 0).detach().cpu()

                result_graph = inverse_process_graph(result_graph)
                save_path = os.path.join(output_dir, f'g_result_{filename}_{idx + id_offset}.bin')
                dgl.save_graphs(save_path, result_graph)
                total_success += 1

                all_coords = torch.cat([
                    result_graph.ndata['x'],
                    result_graph.edata['x'][:, :-1].reshape(-1, 3)
                ], dim=0)
                cloud = trimesh.PointCloud(vertices=all_coords.cpu().numpy())
                output_ply_path = f'output_gen/test_result_{filename}_{idx + id_offset}_{random.randint(0, 1000)}.ply'
                os.makedirs(os.path.dirname(output_ply_path), exist_ok=True)
                cloud.export(output_ply_path)

            except Exception as e:
                print(f"Error decoding graph {idx}: {e}")

        print(f"Total success: {total_success}")
        return None
