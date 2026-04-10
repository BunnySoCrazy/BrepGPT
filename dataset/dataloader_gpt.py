import numpy as np
from dataset.base import BaseDataset
import pathlib
import torch
import json
import os
from tqdm import tqdm
from dgl.data.utils import load_graphs
import dgl
import pickle
import os.path as osp
import networkx as nx

class gptDataset(BaseDataset):
    
    def load_graphs(self, file_paths):
        self.data = []
        for fn in tqdm(file_paths):
            fn = pathlib.Path(fn)

            if not fn.exists():
                continue
            sample = self.load_one_graph(fn)
            if sample is None:
                continue

            self.data.append(sample)

    def _get_bin_files(self, split='train'):
        json_path = os.path.join(f'data/{self.specs["dataset"]}_dataset_splits', f'{split}.json')
        with open(json_path, 'r') as f:
            ids = json.load(f)
        print(f"all bin files: {len(ids)}")
        data_root = self.specs['data_root']
        return [f"{data_root}/{id}.bin" for id in ids]

    def __init__(self, specs, split="train"):
        self.specs = specs
        cache_dir = specs['experiment_directory']
        self.split = split

        os.makedirs(cache_dir, exist_ok=True)
        cache_dir = 'data/GPT_data_cache/'
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = osp.join(cache_dir, f'gpt_dataset_{split}_gpt_{self.specs["dataset"]}.pkl')

        if osp.exists(cache_file):
            print(f"Load cached data: {cache_file}")
            with open(cache_file, 'rb') as f:
                all_data = pickle.load(f)
                min_seq_length = 11 * 0
                original_count = len(all_data)
                self.data = [item for item in all_data if len(item.get('node_sequence', [])) >= min_seq_length]

                seq_lengths = [len(item.get('node_sequence', [])) for item in self.data]
                max_seq_len = max(seq_lengths)
                min_seq_len = min(seq_lengths)
                print(f"Sequence length stats: max={max_seq_len}, min={min_seq_len}")

                filtered_count = original_count - len(self.data)
                print(f"Loaded {len(self.data)} files from cache successfully")
                print(f"Filter info: original {original_count}, filtered {filtered_count} samples due to insufficient sequence length")
        else:
            vhp_dir = 'data/'
            vhp_file = osp.join(vhp_dir, f'{self.specs["dataset"]}_vhp_encoding.pkl')

            with open(vhp_file, 'rb') as f:
                vhp_data = pickle.load(f)
                filename_list = vhp_data['filename']
                node_coordinates_list = vhp_data['node_coordinates']
                indices_list = vhp_data['indices']

                self.vhp_dict = {}
                for filename, indices, node_coordinates in zip(filename_list, indices_list, node_coordinates_list):
                    self.vhp_dict[filename] = (indices, node_coordinates)

            cnt_dir = 'data/'
            cnt_file = osp.join(cnt_dir, f'{self.specs["dataset"]}_cnnt_encoding.pkl')

            with open(cnt_file, 'rb') as f:
                cnt_data = pickle.load(f)
                filename_list = cnt_data['filename']
                indices_list = cnt_data['indices']
                node_coordinates_list = cnt_data['node_coordinates']
                f1 = cnt_data['f1'] if 'f1' in cnt_data else [1] * len(node_coordinates_list)
                self.cnt_dict = {}
                for filename, indices, node_coordinates, f1 in zip(filename_list, indices_list, node_coordinates_list, f1):
                    self.cnt_dict[filename] = (indices, node_coordinates, f1)

            print(f"Cache not found, loading data from raw files...")
            bin_paths = self._get_bin_files(self.split)
            selected_paths = bin_paths[:]
            self.load_graphs(selected_paths)

            print(f"Saving data to cache: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(self.data, f)

            print(f"Finished loading {len(self.data)} files")

    def _collate(self, samples):
        samples = [s for s in samples if s is not None]
        if len(samples) == 0:
            return None

        max_seq_len = max(len(s['node_sequence']) for s in samples)
        batch_sequences = []
        batch_filenames = []
        for sample in samples:
            seq = torch.tensor(sample['node_sequence'], dtype=torch.long)
            padding_length = max_seq_len - len(seq)
            padded_seq = torch.cat([
                seq,
                torch.full((padding_length,), -1, dtype=torch.long)
            ])
            batch_sequences.append(padded_seq)
            batch_filenames.append(sample['filename'])

        return {
            'node_sequences': torch.stack(batch_sequences),
            'filenames': batch_filenames,
        }

    def load_one_graph(self, file_path):
        try:
            graph = load_graphs(str(file_path))[0][0]
            filename = file_path.stem
            if self.specs["dataset"] == "DeepCAD" and (graph.number_of_nodes() > 128 or graph.number_of_nodes() < 24):
                return None
            if self.specs["dataset"] == "ABC" and (graph.number_of_nodes() > 128):
                return None

            new_graph = dgl.graph((graph.edges()[0], graph.edges()[1]), num_nodes=graph.number_of_nodes())
            new_graph.ndata["x"] = graph.ndata["x"]

            node_features = new_graph.ndata["x"].numpy()
            node_coords = node_features[:, :3]
            if filename not in self.cnt_dict or filename not in self.vhp_dict:
                return None

            cnn_feature, cnn_node_coords, f1 = self.cnt_dict[filename]
            vhp_feature, vhp_node_coords = self.vhp_dict[filename]
            if f1 != 1:
                return None
            assert len(cnn_feature) == graph.number_of_nodes(), f"cnn_feature: {len(cnn_feature)}, graph.number_of_nodes(): {graph.number_of_nodes()}"
            assert len(cnn_feature) == len(vhp_feature), f"cnn_feature: {len(cnn_feature)}, vhp_feature: {len(vhp_feature)}"
            cnn_node_coords = np.array(cnn_node_coords)
            vhp_node_coords = np.array(vhp_node_coords)
            assert np.all(cnn_node_coords == vhp_node_coords), f"cnn_node_coords: {cnn_node_coords}, vhp_node_coords: {vhp_node_coords}"
            assert np.all(cnn_node_coords == node_features), f"cnn_node_coords: {cnn_node_coords}, node_features: {node_features}"

            cnn_feature = np.array(cnn_feature)
            vhp_feature = np.array(vhp_feature)
            other_features = np.concatenate([cnn_feature, vhp_feature], axis=1)

            nx_graph = new_graph.to_networkx().to_undirected()
            connected_components = list(nx.connected_components(nx_graph))

            # Sort connected components by their minimum-coordinate node (z, y, x)
            component_min_coords = []
            for component in connected_components:
                component_nodes = list(component)
                component_coords = node_coords[component_nodes]
                min_idx = np.lexsort((component_coords[:, 0], component_coords[:, 1], component_coords[:, 2]))[0]
                component_min_coords.append(component_coords[min_idx])

            component_min_coords = np.array(component_min_coords)
            sorted_component_indices = np.lexsort((component_min_coords[:, 0],
                                                component_min_coords[:, 1],
                                                component_min_coords[:, 2]))
            sorted_components = [connected_components[i] for i in sorted_component_indices]

            sorted_nodes = []
            for component in sorted_components:
                component_nodes = list(component)
                component_coords = node_coords[component_nodes]
                sort_indices = np.lexsort((component_coords[:, 0],
                                        component_coords[:, 1],
                                        component_coords[:, 2]))
                sorted_component_nodes = [component_nodes[i] for i in sort_indices]
                sorted_nodes.append(sorted_component_nodes)

            normalized_coords = (node_coords + 1) / 2
            max_token = 4096
            START_TOKEN = max_token + 1
            END_TOKEN = max_token + 2
            SEP_TOKEN = max_token + 3

            resolution = 128
            discretized_coords = np.floor(normalized_coords * resolution).astype(int)
            node_sequence = [START_TOKEN]

            for i, component_nodes in enumerate(sorted_nodes):
                for node_idx in component_nodes:
                    node_sequence.extend(discretized_coords[node_idx])
                    node_sequence.extend(other_features[node_idx])
                if i < len(sorted_nodes) - 1:
                    node_sequence.append(SEP_TOKEN)

            node_sequence.append(END_TOKEN)

            sample = {
                "filename": file_path.stem,
                "graph": new_graph,
                "node_sequence": node_sequence,
            }

        except RuntimeError as e:
            print(f"RuntimeError: {e}")
            return None
        return sample
