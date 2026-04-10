import numpy as np
from dataset.base import BaseDataset
import pathlib
import json
import os
from tqdm import tqdm
from torch import FloatTensor
import pickle
import os.path as osp
from dgl.data.utils import load_graphs


class VhpDataset(BaseDataset):

    def load_graphs(self, file_paths):
        self.data = []
        for fn in tqdm(file_paths):
            fnp = pathlib.Path(fn)

            if not fnp.exists():
                continue
            sample = self.load_one_graph(fnp)

            if sample is None:
                continue
            if sample["graph"].edata["x"].max() > 2 or sample["graph"].edata["x"].min() < -2:
                continue
            if sample["graph"].edata["next_half_edge"].max() > 2 or sample["graph"].edata["next_half_edge"].min() < -2:
                print(f"next_half_edge: {sample['graph'].edata['next_half_edge'].max()}")
                continue
            if sample["graph"].edata["x"].size(0) == 0:
                continue
            self.data.append(sample)
        self.convert_to_float32()

    def _get_bin_files(self, split='train'):
        json_path = os.path.join(f'data/{self.dataset}_dataset_splits', f'{split}.json')
        with open(json_path, 'r') as f:
            ids = json.load(f)
        print(f"all bin files: {len(ids)}")
        data_root = self.specs['data_root']
        return [f"{data_root}/{id}.bin" for id in ids]

    def __init__(self, specs, split="train"):
        self.specs = specs
        self.split = split
        self.dataset = specs["dataset"]

        cache_dir = 'data/vhp_cache'
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = osp.join(cache_dir, f'vhp_{split}_{self.dataset}.pkl')

        if osp.exists(cache_file):
            print(f"Load cached data: {cache_file}")
            with open(cache_file, 'rb') as f:
                self.data = pickle.load(f)
            print(f"Loaded {len(self.data)} files from cache successfully")
        else:
            print(f"Cache {cache_file} not found, loading from raw files...")
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
            self.data[i]["graph"].edata["x"] = self.data[i]["graph"].edata["x"].type(FloatTensor)

    def load_one_graph(self, file_path):
        try:
            graph = load_graphs(str(file_path))[0][0]
            file_name = file_path.stem
            if self.specs["dataset"] == "DeepCAD" and (graph.number_of_nodes() > 128):
                return None
            if self.specs["dataset"] == "ABC" and (graph.number_of_nodes() > 128):
                return None

            sample = {"graph": graph, "filename": file_name}
        except RuntimeError:
            return None
        return sample
