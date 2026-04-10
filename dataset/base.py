from torch.utils.data import Dataset, DataLoader
from torch import FloatTensor
import dgl
from dgl.data.utils import load_graphs
from tqdm import tqdm
import pathlib
from multiprocessing import Pool
import torch
import gc

def _process_single_file(args):
    self_obj, fn = args
    try:
        fn = pathlib.Path(fn)
        if not fn.exists():
            return None
            
        sample = self_obj.load_one_graph(fn)
        if sample is None:
            return None
            
        if "x" not in sample["graph"].edata:
            return None
            
        if sample["graph"].edata["x"].size(0) == 0:
            return None
            
        sample["graph"] = sample["graph"].to('cpu')
        torch.cuda.empty_cache()
        
        return sample
    except :
        return None

class BaseDataset(Dataset):

    def load_graphs(self, file_paths, center_and_scale=True):
        self.data = []
        for fn in tqdm(file_paths):
            fn = pathlib.Path(fn)

            if not fn.exists():
                continue
            sample = self.load_one_graph(fn)
            if sample is None:
                continue

            if "x" not in sample["graph"].edata:
                continue
                
            if sample["graph"].edata["x"].size(0) == 0:
                continue

            self.data.append(sample)
        self.convert_to_float32()
    
    def load_one_graph(self, file_path):
        try:
            graph = load_graphs(str(file_path))[0][0]
            if self.specs["dataset"] == "DeepCAD" and (graph.number_of_nodes() > 128 or graph.number_of_nodes() <= 48):
                return None
            if self.specs["dataset"] == "ABC" and (graph.number_of_nodes() > 128 or graph.number_of_nodes() <= 24):
                return None
            sample = {"graph": graph, "filename": file_path.stem}
        except RuntimeError:
            return None
        return sample

    def load_graphs_parallel(self, file_paths, center_and_scale=True, num_workers=4, batch_size=5000):
        results = []
        for i in range(0, len(file_paths), batch_size):
            batch_files = file_paths[i:i+batch_size]
            args_list = [(self, fn) for fn in batch_files]
            
            with Pool(num_workers) as pool:
                batch_results = list(tqdm(
                    pool.imap(_process_single_file, args_list),
                    total=len(batch_files),
                    desc=f"batch {i//batch_size + 1}/{len(file_paths)//batch_size + 1}"
                ))
                
            results.extend([r for r in batch_results if r is not None])
            gc.collect()
            torch.cuda.empty_cache()
        
        self.data = results
        self.convert_to_float32()

    def convert_to_float32(self):
        for i in range(len(self.data)):
            self.data[i]["graph"].ndata["x"] = self.data[i]["graph"].ndata["x"].type(FloatTensor)
            self.data[i]["graph"].edata["x"] = self.data[i]["graph"].edata["x"].type(FloatTensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

    def _collate(self, batch):
        graph_sizes = [sample["graph"].number_of_nodes() for sample in batch]
        batched_graph = dgl.batch([sample["graph"] for sample in batch])
        batched_filenames = [sample["filename"] for sample in batch]
        return {"graph": batched_graph, "filename": batched_filenames, "sizes": graph_sizes}

    def get_dataloader(self, batch_size=128, shuffle=True, num_workers=0):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
            num_workers=num_workers,
        )
