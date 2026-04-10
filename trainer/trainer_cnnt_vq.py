import torch
import pytorch_lightning as pl
import os
import torch.nn as nn
import pickle
from tqdm import tqdm

from model.encoders.brep_encoders import ConnectEncoder as Encoder
from model.decoders.cnnt_decoder import CnntDecoder


class Trainer_cnnt_vq(pl.LightningModule):
    """Lightning module for Stage 1: connectivity VQ-VAE."""

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
            threshold_ema_dead_code=0.1,
        )

        self.perfect_f1_count = 0
        self.total_samples = 0
        self.test_output_list = []

        max_token = 4096
        self.START_TOKEN = max_token + 1
        self.END_TOKEN = max_token + 2
        self.SEP_TOKEN = max_token + 3

    def build_net(self):
        self.encoder = Encoder(
            crv_emb_dim=self.specs["crv_emb_dim"],
            srf_emb_dim=self.specs["vertex_emb_dim"],
            graph_emb_dim=self.specs["node_feature_dim"],
            hidden_dim=self.specs["hidden_dim_encode"],
            num_layers=self.specs["n_layers_encode"],
            encoder_type=self.specs["graph_encoder_type"]
        )
        self.decoder = CnntDecoder(
            node_feature_dim=self.specs["node_feature_dim"],
            hidden_dim=1024
        )

    def configure_optimizers(self):
        param_groups = [
            {"params": self.encoder.parameters(), "lr": self.lr, "betas": tuple(self.betas)},
            {"params": self.decoder.parameters(), "lr": self.lr, "betas": tuple(self.betas)},
            {"params": self.vector_quantizer.parameters(), "lr": self.lr, "betas": tuple(self.betas)},
        ]
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }

    def training_step(self, batch, batch_idx):
        graphs = batch["graph"].to(self.device)
        pos_edges = batch["pos_edges"]
        neg_edges = batch["neg_edges"]

        if pos_edges:
            pos_edges = torch.cat(pos_edges, dim=0)
        if neg_edges:
            neg_edges = torch.cat(neg_edges, dim=0)

        node_features = self.encoder(graphs)
        quantized_features, indices, vq_loss = self.vector_quantizer(node_features.unsqueeze(1))
        quantized_features = quantized_features.squeeze(1)

        all_edges = torch.cat([pos_edges, neg_edges], dim=0)
        predictions = self.decoder(quantized_features, all_edges)

        labels = torch.cat([
            torch.ones(pos_edges.size(0)),
            torch.zeros(neg_edges.size(0))
        ]).to(predictions.device)

        pos_weight = neg_edges.size(0) / pos_edges.size(0)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(predictions.device))
        loss = loss_fn(predictions.squeeze(), labels)
        total_loss = loss + self.specs['vq_loss_weight'] * vq_loss.mean()

        predictions = torch.sigmoid(predictions.squeeze())
        pred_labels = (predictions > 0.5).float()
        accuracy = (pred_labels == labels).float().mean()

        true_positives = ((pred_labels == 1) & (labels == 1)).sum()
        false_positives = ((pred_labels == 1) & (labels == 0)).sum()
        false_negatives = ((pred_labels == 0) & (labels == 1)).sum()
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        bs = self.specs['batch_size']
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log("train_vq_loss", vq_loss.mean(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log("train_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log("train_f1", f1, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log("active_codes", indices.unique().numel())

        return total_loss

    def validation_step(self, batch, batch_idx):
        return None

    def split_sequences(self, node_sequences):
        """Split padded token sequences into per-node subsequences using delimiter tokens."""
        START_TOKEN = 4097
        END_TOKEN = 4098
        SEP_TOKEN = 4099

        all_sequences = []
        for b in range(node_sequences.shape[0]):
            seq = node_sequences[b]
            seq = seq[seq != -2]

            start_indices = (seq == START_TOKEN).nonzero(as_tuple=True)[0]
            end_indices = (seq == END_TOKEN).nonzero(as_tuple=True)[0]

            if len(start_indices) > 0:
                for start_idx, end_idx in zip(start_indices, end_indices):
                    if end_idx > start_idx:
                        sub_seq = seq[start_idx + 1:end_idx]
                        if len(sub_seq) > 0:
                            sep_indices = (sub_seq == SEP_TOKEN).nonzero(as_tuple=True)[0]
                            if len(sep_indices) > 0:
                                sequence_parts = []
                                prev_idx = 0
                                for sep_idx in sep_indices:
                                    if sep_idx > prev_idx:
                                        part = sub_seq[prev_idx:sep_idx]
                                        if len(part) > 0:
                                            sequence_parts.append(part)
                                    prev_idx = sep_idx + 1
                                if prev_idx < len(sub_seq):
                                    last = sub_seq[prev_idx:]
                                    if len(last) > 0:
                                        sequence_parts.append(last)
                                if sequence_parts:
                                    all_sequences.append(sequence_parts)
                            else:
                                all_sequences.append([sub_seq])
        return all_sequences

    def decode(self, dataloader):
        """Decode token sequences into graph structures (node positions + edges)."""
        all_graphs = []
        self.decoder.eval()
        self.vector_quantizer.eval()

        for idx, batch in enumerate(tqdm(dataloader)):
            node_sequences = batch["node_sequences"].to(self.device)
            sequences = self.split_sequences(node_sequences)

            with torch.no_grad():
                for idx, sequence_parts in enumerate(sequences):
                    all_positions = []
                    all_geometric_features = []
                    all_node_features = []
                    all_edges = []
                    offset = 0
                    all_good = True

                    for sequence in sequence_parts:
                        seq_length = len(sequence)
                        if seq_length % 11 != 0:
                            print(f"seq_length % 11 != 0: {seq_length}")
                            all_good = False
                            break

                        num_nodes = seq_length // 11
                        features_per_node = 11

                        positions = torch.cat([
                            sequence[0::features_per_node].reshape(num_nodes, 1),
                            sequence[1::features_per_node].reshape(num_nodes, 1),
                            sequence[2::features_per_node].reshape(num_nodes, 1),
                        ], dim=1)
                        if positions.max() > 128:
                            print(f"positions.max() > 128: {positions.max()}")
                            all_good = False
                            break

                        connection_features = torch.stack([
                            sequence[3::features_per_node],
                            sequence[4::features_per_node],
                            sequence[5::features_per_node],
                            sequence[6::features_per_node],
                        ], dim=1)
                        geometric_features = torch.stack([
                            sequence[7::features_per_node],
                            sequence[8::features_per_node],
                            sequence[9::features_per_node],
                            sequence[10::features_per_node],
                        ], dim=1)

                        node_features = self.vector_quantizer.get_output_from_indices(connection_features)

                        possible_edges = []
                        for i in range(num_nodes):
                            for j in range(i + 1, num_nodes):
                                possible_edges.append([i, j])
                                possible_edges.append([j, i])

                        if possible_edges:
                            possible_edges = torch.tensor(possible_edges).to(self.device)
                            edge_predictions = self.decoder(node_features, possible_edges)
                            edge_predictions = torch.sigmoid(edge_predictions.squeeze())
                            pred_edges = possible_edges[edge_predictions > 0.5]

                            if pred_edges.numel() > 0:
                                adjusted_edges = pred_edges + offset
                                reverse_edges = torch.stack(
                                    [adjusted_edges[:, 1], adjusted_edges[:, 0]], dim=1
                                )
                                all_edges_combined = torch.unique(
                                    torch.cat([adjusted_edges, reverse_edges], dim=0), dim=0
                                )

                                if all_edges_combined.numel() > 0:
                                    src_nodes, counts = torch.unique(
                                        all_edges_combined[:, 0], return_counts=True
                                    )
                                    valid_shape = True
                                    for node_idx in range(offset, offset + num_nodes):
                                        mask = (src_nodes == node_idx)
                                        if not mask.any() or counts[mask][0] < 2:
                                            valid_shape = False
                                            break
                                    if valid_shape:
                                        all_edges.append(all_edges_combined)
                                    else:
                                        print("Invalid shape, skipping this subgraph")
                                        all_good = False
                                        break

                        if not all_good:
                            continue

                        all_positions.append(positions)
                        all_geometric_features.append(geometric_features)
                        all_node_features.append(node_features)
                        offset += num_nodes

                    try:
                        combined_positions = torch.cat(all_positions, dim=0)
                        combined_geometric_features = torch.cat(all_geometric_features, dim=0)
                        combined_edges = (
                            torch.cat(all_edges, dim=0) if all_edges
                            else torch.tensor([], dtype=torch.long, device=self.device).reshape(0, 2)
                        )
                        graph = {
                            'node_positions': combined_positions,
                            'node_geometric_features': combined_geometric_features,
                            'edges': combined_edges,
                            'filename': batch["filenames"][idx],
                        }
                        all_graphs.append(graph)
                    except Exception as e:
                        print(f"Error decoding graph {idx}: {e}")

        return all_graphs

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
        save_dict = {
            'indices': [x['indices'] for x in outputs],
            'filename': [x['filename'] for x in outputs],
            'node_coordinates': [x['node_coordinates'] for x in outputs],
        }
        with open('data/DeepCAD_cnnt_encoding.pkl', 'wb') as f:
            pickle.dump(save_dict, f)
