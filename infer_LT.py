import os
import argparse
import pickle
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from utils import init_seeds, load_experiment_specifications
from trainer.trainer_cnnt_vq import Trainer_cnnt_vq
from trainer.trainer_vhp_vq import Trainer_vhp_vq
from trainer.trainer_GPT import Trainer_gpt

START_TOKEN = 4097  # max_token(4096) + 1
END_TOKEN   = 4098  # max_token(4096) + 2
PAD_TOKEN   = -1


class GeneratedSequenceDataset(Dataset):
    """Wraps a sequences pkl file for cnnt_vq.decode()."""

    def __init__(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)  # list of {'node_sequence': list, 'filename': str}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, samples):
        max_len = max(len(s['node_sequence']) for s in samples)
        padded, filenames = [], []
        for s in samples:
            seq = torch.tensor(s['node_sequence'], dtype=torch.long)
            pad_len = max_len - len(seq)
            seq = torch.cat([seq, torch.full((pad_len,), PAD_TOKEN, dtype=torch.long)])
            padded.append(seq)
            filenames.append(s['filename'])
        return {
            'node_sequences': torch.stack(padded),
            'filenames': filenames,
        }


def load_model(trainer_cls, specs, exp_dir):
    model = trainer_cls(specs)
    ckpt_path = os.path.join(exp_dir, 'last.ckpt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model = model.cuda()
    model.eval()
    return model


def generate_sequences(gpt_model, num_samples, batch_size, save_dir):
    """Run GPT autoregressive generation and save results as pkl."""
    os.makedirs(save_dir, exist_ok=True)
    all_sequences = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for b in tqdm(range(num_batches), desc="GPT generation"):
            current_bs = min(batch_size, num_samples - b * batch_size)
            start_tokens = torch.full((current_bs, 1), START_TOKEN + 1, device='cuda')
            generated = gpt_model.model.generate(
                start_tokens,
                max_new_tokens=2048,
                end_token=END_TOKEN + 1,
                temperature=1,
                top_k=20,
            )
            generated = generated - 1  # shift tokens back to original range
            for i in range(current_bs):
                sample_id = b * batch_size + i
                all_sequences.append({
                    'node_sequence': generated[i].cpu().numpy().tolist(),
                    'filename': f'generated_{sample_id}',
                })

    pkl_path = os.path.join(save_dir, 'generated_sequences.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(all_sequences, f)
    print(f"Saved {len(all_sequences)} sequences → {pkl_path}")
    return pkl_path


def get_gt_pkl_path(gpt_specs, split, num_samples):
    """
    Return a pkl path of ground-truth token sequences from the GPT data cache,
    trimmed to num_samples if needed.  The format is identical to the GPT
    generation output, so Steps 2 and 3 can consume it unchanged.
    """
    dataset_name = gpt_specs["dataset"]
    cache_file = f'data/GPT_data_cache/gpt_dataset_{split}_gpt_{dataset_name}.pkl'
    if not os.path.exists(cache_file):
        raise FileNotFoundError(
            f"GPT cache not found: {cache_file}\n"
            "Run the GPT dataset loader first to build the cache."
        )

    with open(cache_file, 'rb') as f:
        all_data = pickle.load(f)

    if num_samples >= len(all_data):
        print(f"Using full GT cache ({len(all_data)} samples): {cache_file}")
        return cache_file

    subset = all_data[:num_samples]
    tmp_path = cache_file.replace('.pkl', f'_gt_subset_{num_samples}.pkl')
    with open(tmp_path, 'wb') as f:
        pickle.dump(subset, f)
    print(f"Wrote GT subset ({len(subset)} samples) → {tmp_path}")
    return tmp_path


def main(args):
    init_seeds(42)
    torch.set_float32_matmul_precision('medium')

    gpt_exp_dir  = os.path.join('./exp_log', args.gpt_experiment)
    cnnt_exp_dir = os.path.join('./exp_log', args.cnnt_experiment)
    vhp_exp_dir  = os.path.join('./exp_log', args.vhp_experiment)

    gpt_specs  = load_experiment_specifications(gpt_exp_dir)
    gpt_specs["experiment_directory"]  = gpt_exp_dir
    cnnt_specs = load_experiment_specifications(cnnt_exp_dir)
    cnnt_specs["experiment_directory"] = cnnt_exp_dir
    vhp_specs  = load_experiment_specifications(vhp_exp_dir)
    vhp_specs["experiment_directory"]  = vhp_exp_dir

    # ------------------------------------------------------------------
    # Step 1: get token sequences
    #   Normal mode : GPT autoregressive generation
    #   GT-decode   : load ground-truth sequences from the data cache
    # ------------------------------------------------------------------
    print("=" * 45)
    if args.gt_decode:
        print(f"Step 1 / 3  —  Load GT sequences ({args.gt_split} split)")
        print("=" * 45)
        pkl_path = get_gt_pkl_path(gpt_specs, split=args.gt_split, num_samples=args.num_samples)
    else:
        print("Step 1 / 3  —  GPT generation")
        print("=" * 45)
        gpt_model = load_model(Trainer_gpt, gpt_specs, gpt_exp_dir)
        seq_save_dir = os.path.join(gpt_exp_dir, 'generated_sequences')
        pkl_path = generate_sequences(gpt_model, args.num_samples, args.batch_size, seq_save_dir)

    # ------------------------------------------------------------------
    # Step 2: Decode connectivity  (cnnt_vq)
    #   Input : token sequences
    #   Output: list of graph dicts with node_positions / node_geometric_features / edges
    # ------------------------------------------------------------------
    print("=" * 45)
    print("Step 2 / 3  —  Decode connectivity (cnnt_vq)")
    print("=" * 45)
    cnnt_model = load_model(Trainer_cnnt_vq, cnnt_specs, cnnt_exp_dir)
    seq_dataset = GeneratedSequenceDataset(pkl_path)
    seq_loader  = DataLoader(
        seq_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4,
        collate_fn=seq_dataset.collate_fn,
    )
    graphs = cnnt_model.decode(seq_loader)
    print(f"Decoded {len(graphs)} graph structures.")

    # ------------------------------------------------------------------
    # Step 3: Decode geometry  (vhp_vq)
    #   Input : graph dicts (uses node_geometric_features = vhp VQ indices)
    #   Output: .bin files saved to output_dir
    # ------------------------------------------------------------------
    print("=" * 45)
    print("Step 3 / 3  —  Decode geometry (vhp_vq)")
    print("=" * 45)
    vhp_model = load_model(Trainer_vhp_vq, vhp_specs, vhp_exp_dir)
    vhp_model.decode(graphs, output_dir=args.output_dir)
    print(f"Results saved to: {args.output_dir}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full BrepGPT inference: GPT generation → connectivity decode → geometry decode"
    )
    parser.add_argument("--gpt_experiment",  "-g", dest="gpt_experiment",  required=True,
                        help="exp_log sub-path of the trained GPT model")
    parser.add_argument("--cnnt_experiment", "-c", dest="cnnt_experiment", required=True,
                        help="exp_log sub-path of the trained cnnt_vq model")
    parser.add_argument("--vhp_experiment",  "-v", dest="vhp_experiment",  required=True,
                        help="exp_log sub-path of the trained vhp_vq model")
    parser.add_argument("--num_samples", "-n", dest="num_samples", type=int, default=8,
                        help="Number of shapes to generate (or to take from GT cache)")
    parser.add_argument("--batch_size",  "-b", dest="batch_size",  type=int, default=1,
                        help="Batch size for generation and decoding")
    parser.add_argument("--output_dir",  "-o", dest="output_dir",  default="output/VHP",
                        help="Directory to save decoded .bin graph files")
    parser.add_argument("--gt_decode", action="store_true",
                        help="Skip GPT generation; use ground-truth token sequences from "
                             "the data cache as input to Steps 2 and 3 (upper-bound check)")
    parser.add_argument("--gt_split", dest="gt_split", default="train",
                        choices=["train", "val", "test"],
                        help="Which data split to use for GT-decode mode (default: train)")
    args = parser.parse_args()


    main(args)
