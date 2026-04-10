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
from torch import FloatTensor
import networkx as nx


class CnntDataset(BaseDataset):

    def load_graphs(self, file_paths):
        self.data = []
        for fn in tqdm(file_paths):
            fn = pathlib.Path(fn)

            if not fn.exists():
                continue
            sample = self.load_one_graph(fn)
            if sample is None:
                continue

            if "x" not in sample["graph"].ndata:
                continue

            if sample["graph"].ndata["x"].size(0) == 0:
                continue

            self.data.append(sample)

    def _get_bin_files(self, split='train'):
        json_path = os.path.join(f'data/{self.dataset}_dataset_splits', f'{split}.json')
        with open(json_path, 'r') as f:
            ids = json.load(f)
        print(f"all bin files: {len(ids)}")
        data_root = self.specs['data_root']
        return [f"{data_root}/{id}.bin" for id in ids]

    def __init__(self, specs, split="train"):
        self.specs = specs
        cache_dir = specs['experiment_directory']
        self.split = split
        self.dataset = specs['dataset']

        os.makedirs(cache_dir, exist_ok=True)
        cache_dir = 'data/connect_cache'
        cache_file = osp.join(cache_dir, f'connect_dataset_{split}_{self.dataset}.pkl')
        os.makedirs(cache_dir, exist_ok=True)

        if osp.exists(cache_file):
            print(f"Load cached data: {cache_file}")
            with open(cache_file, 'rb') as f:
                self.data = pickle.load(f)
                self.data = self.data[:]
            print("load cache data done")
            print(f"Loaded {len(self.data)} files from cache successfully")
        else:
            print(f"Cache not found, loading data from raw files...")
            bin_paths = self._get_bin_files(self.split)
            if split == "train":
                selected_paths = bin_paths[:]
            else:
                selected_paths = bin_paths[:]
            self.load_graphs(selected_paths)

            print(f"Saving data to cache: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(self.data, f)

            print(f"Finished loading {len(self.data)} files")

    def convert_to_float32(self):
        for i in range(len(self.data)):
            self.data[i]["graph"].ndata["x"] = self.data[i]["graph"].ndata["x"].type(FloatTensor)

    def _collate(self, batch):
        graph_sizes = [sample["graph"].number_of_nodes() for sample in batch]
        batched_graph = dgl.batch([sample["graph"] for sample in batch])
        batched_filenames = [sample["filename"] for sample in batch]

        cumsum_nodes = 0
        adjusted_pos_edges = []
        adjusted_neg_edges = []
        for sample, size in zip(batch, graph_sizes):
            if sample["pos_edges"].numel() > 0:
                adjusted_pos_edges.append(sample["pos_edges"].long() + cumsum_nodes)
            if sample["neg_edges"].numel() > 0:
                adjusted_neg_edges.append(sample["neg_edges"].long() + cumsum_nodes)
            cumsum_nodes += size

        return {
            "graph": batched_graph,
            "filename": batched_filenames,
            "sizes": graph_sizes,
            "pos_edges": adjusted_pos_edges,
            "neg_edges": adjusted_neg_edges,
        }

    def load_one_graph(self, file_path):
        try:
            graph = load_graphs(str(file_path))[0][0]
            if self.specs["dataset"] == "ABC" and (graph.number_of_nodes() > 128):
                return None
            if self.specs["dataset"] == "DeepCAD" and (graph.number_of_nodes() > 128):
                return None

            new_graph = dgl.graph((graph.edges()[0], graph.edges()[1]), num_nodes=graph.number_of_nodes())
            new_graph.ndata["x"] = graph.ndata["x"]

            nx_graph = new_graph.to_networkx().to_undirected()
            connected_components = list(nx.connected_components(nx_graph))

            pos_edges = torch.stack(new_graph.edges()).t()
            pos_edges_set = {(int(i), int(j)) for i, j in pos_edges}

            # Negative edges: non-adjacent pairs within each connected component
            neg_edges = []
            for component in connected_components:
                component_pairs = torch.combinations(torch.tensor(list(component), dtype=torch.long), r=2)
                for i, j in component_pairs:
                    if (int(i), int(j)) not in pos_edges_set and (int(j), int(i)) not in pos_edges_set:
                        neg_edges.append([i, j])
                        neg_edges.append([j, i])

            neg_edges = torch.tensor(neg_edges) if neg_edges else torch.empty((0, 2), dtype=torch.long)
            sample = {
                "graph": new_graph,
                "filename": file_path.stem,
                "pos_edges": pos_edges,
                "neg_edges": neg_edges,
            }

        except RuntimeError as e:
            print(f"RuntimeError: {e}")
            return None
        return sample
