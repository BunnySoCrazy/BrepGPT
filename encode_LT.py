import os
import argparse
import pytorch_lightning as pl
from utils import init_seeds, load_experiment_specifications
from dataset import dataloader_cnnt, dataloader_vhp
from trainer.trainer_cnnt_vq import Trainer_cnnt_vq
from trainer.trainer_vhp_vq import Trainer_vhp_vq
import torch


def main(args):
    init_seeds(0)
    torch.set_float32_matmul_precision('medium')

    experiment_directory = os.path.join('./exp_log', args.experiment_directory)
    specs = load_experiment_specifications(experiment_directory)
    specs["experiment_directory"] = experiment_directory

    if args.mode == 'cnnt_vq':
        model = Trainer_cnnt_vq(specs)
        dataset = dataloader_cnnt.CnntDataset(specs, split='train')
    elif args.mode == 'vhp_vq':
        model = Trainer_vhp_vq(specs)
        dataset = dataloader_vhp.VhpDataset(specs, split='train')

    loader = dataset.get_dataloader(batch_size=64, shuffle=False, num_workers=4)

    checkpoint_path = os.path.join(experiment_directory, 'last.ckpt')
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'], strict=True)

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
    )

    print("=================================")
    print("====== Start VQ Encoding  =======")
    print("=================================")
    trainer.test(model, loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-e", dest="experiment_directory", required=True)
    parser.add_argument("--mode", "-m", dest="mode", required=True, choices=["cnnt_vq", "vhp_vq"])
    args = parser.parse_args()

    main(args)
