import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import init_seeds, load_experiment_specifications
from dataset import (dataloader_cnnt,dataloader_vhp, dataloader_gpt)
from trainer.trainer_cnnt_vq import (Trainer_cnnt_vq)
from trainer.trainer_vhp_vq import (Trainer_vhp_vq)
from trainer.trainer_GPT import (Trainer_gpt)
import torch


def main(args):
	# Set random seed
	init_seeds()
	print(f"PyTorch version: {torch.__version__}")
	print(f"CUDA version: {torch.version.cuda}")
	print(f"CUDA available: {torch.cuda.is_available()}")
	torch.set_float32_matmul_precision('medium')
 
	# Create experiment directory
	experiment_directory = os.path.join('./exp_log', args.experiment_directory)

	# Load experiment config
	specs = load_experiment_specifications(experiment_directory)
	specs["experiment_directory"] = experiment_directory
 
	# Create dataset and data loader
	if args.mode == 'cnnt_vq':
		model = Trainer_cnnt_vq(specs)
		train_dataset = dataloader_cnnt.CnntDataset(specs,split='train')
		val_dataset = dataloader_cnnt.CnntDataset(specs,split='val')
	if args.mode == 'vhp_vq':
		model = Trainer_vhp_vq(specs)
		train_dataset = dataloader_vhp.VhpDataset(specs, split='train')
		val_dataset = dataloader_vhp.VhpDataset(specs, split='val')
	if args.mode == 'gpt':
		model = Trainer_gpt(specs)
		train_dataset = dataloader_gpt.gptDataset(specs,split='train')
		val_dataset = dataloader_gpt.gptDataset(specs,split='val')

	train_loader = train_dataset.get_dataloader(batch_size=specs["batch_size"],shuffle=True,num_workers=4)
	val_loader = val_dataset.get_dataloader(batch_size=specs["batch_size"],shuffle=False,num_workers=4)
	
	# Setup checkpoint callback
	checkpoint_callback = ModelCheckpoint(
		dirpath=experiment_directory,
		filename='{epoch}',
		save_top_k=1,
		verbose=True,
		monitor='train_total_loss',
		mode='min',
		save_last=True
	)

	# Create Lightning trainer
	trainer = pl.Trainer(
		default_root_dir=experiment_directory,
		max_epochs=specs["num_epochs"] if int(args.epoch) <= 0 else int(args.epoch),
		accelerator='gpu' if torch.cuda.is_available() else 'cpu',
		devices='auto',
		# strategy='ddp_find_unused_parameters_true',
		callbacks=[checkpoint_callback],
		check_val_every_n_epoch=5,
		log_every_n_steps=5,
		gradient_clip_val=1,
		gradient_clip_algorithm="norm"
	)

	# Resolve checkpoint path for resuming
	ckpt_path = None
	if args.cont:
		if args.ckpt == 'last':
			ckpt_path = os.path.join(experiment_directory, 'last.ckpt')
		else:
			ckpt_path = args.ckpt  # treat as explicit file path
		assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"

		ckpt = torch.load(ckpt_path, map_location='cpu')
		is_full_ckpt = 'optimizer_states' in ckpt  # weights-only ckpt lacks this
		if is_full_ckpt:
			print(f"Resuming full training state from: {ckpt_path}")
		else:
			print(f"Loading weights-only checkpoint from: {ckpt_path} (optimizer/epoch state not restored)")
			model.load_state_dict(ckpt['state_dict'])
			ckpt_path = None  # let trainer.fit start fresh loop

	print("===========================")
	print("====== Start Training =====")
	print("===========================")

	trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

if __name__ == "__main__":
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument(
		"--experiment",
		"-e",
		dest="experiment_directory",
		required=True
	)
	arg_parser.add_argument(
		"--epoch",
		dest="epoch",
		default=-1
	)
	arg_parser.add_argument(
		"--mode",
		"-m",
		dest="mode",
		default='vq'
	)
	arg_parser.add_argument('--test_data', dest='test_data', default=False, action='store_true', help="train on test dataset")
	arg_parser.add_argument('--continue', dest='cont', default=False, action='store_true', help="continue training from checkpoint")
	arg_parser.add_argument('--ckpt', type=str, default='last', required=False, help="checkpoint to resume from: 'last' or an explicit .ckpt path")

	args = arg_parser.parse_args()
	main(args)
