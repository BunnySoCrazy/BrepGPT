import torch
import pytorch_lightning as pl
import os
from model.decoders.GPT import GPT, GPTConfig
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer


class Trainer_gpt(pl.LightningModule):
    def __init__(self, specs):
        super().__init__()
        self.save_hyperparameters()
        self.specs = specs
        self.experiment_directory = specs['experiment_directory']
        self.log_dir = os.path.join(specs['experiment_directory'], 'log/')

        self.build_net()

        self.lr = specs["learning_rate"]
        self.betas = specs["betas"]

        self.perfect_f1_count = 0
        self.total_samples = 0

    def build_net(self):
        gptconf = GPTConfig(
            block_size=self.specs["block_size"],
            vocab_size=self.specs["vocab_size"],
            n_layer=self.specs["n_layer"],
            n_head=self.specs["n_head"],
            n_embd=self.specs["n_embd"],
            dropout=self.specs["dropout"],
            bias=self.specs["bias"],
        )
        self.model = GPT(gptconf)

    def configure_optimizers(self):
        params_to_optimize = [
            {"params": self.model.parameters(), "lr": self.lr, "betas": (self.betas[0], self.betas[1])}
        ]
        optimizer = torch.optim.AdamW(params_to_optimize)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }

    def training_step(self, batch, batch_idx):
        node_sequences = batch["node_sequences"].to(self.device)
        x = node_sequences[:, :-1]
        targets = node_sequences[:, 1:]
        logits, loss = self.model(x + 1, targets + 1)
        acc = self.accuracy(logits, targets + 1, ignore_label=0)
        self.log("train_total_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.specs['batch_size'])
        self.log("train_accuracy", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.specs['batch_size'])
        return loss

    def accuracy(self, logits, targets, ignore_label=None):
        pred = logits.argmax(dim=-1)
        if ignore_label is not None:
            normalizer = torch.sum(targets != ignore_label)
            ignore_mask = torch.where(
                targets == ignore_label,
                torch.zeros_like(targets, device=self.device),
                torch.ones_like(targets, device=self.device)
            ).float()
        else:
            normalizer = targets.numel()
            ignore_mask = torch.ones_like(targets, device=self.device).float()
        acc = torch.sum((pred == targets).float() * ignore_mask) / normalizer
        return acc

    def validation_step(self, batch, batch_idx):
        return None

    def test_step(self, batch, batch_idx):
        batch_size = 1
        start_token = torch.tensor([[4097 + 1]], device=self.device)
        self.model.eval()
        generated_seq = self.model.generate(
            start_token, max_new_tokens=1024, end_token=4098 + 1, temperature=1.0, top_k=None
        )
        generated_seq = generated_seq - 1

        if not hasattr(self, 'generated_sequences'):
            self.generated_sequences = []
        self.generated_sequences.append({
            'batch_idx': batch_idx,
            'sequence': generated_seq.cpu().numpy()
        })
        return None

    def on_test_epoch_end(self):
        import pickle
        save_dir = "generation_results"
        os.makedirs(save_dir, exist_ok=True)
        formatted_data = []
        for item in self.generated_sequences:
            formatted_data.append({
                'node_sequence': item['sequence'][0].tolist(),
                'filename': f"generated_{item['batch_idx']}"
            })
        output_path = os.path.join(save_dir, 'generated_sequences.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(formatted_data, f)
        print(f"Generated {len(formatted_data)} sequences saved to {output_path}")
        self.generated_sequences = []

    def generate_samples(self, num_samples=1, batch_size=4, save_path="output/generation_results"):
        self.model.eval()
        num_batches = (num_samples + batch_size - 1) // batch_size
        generated_sequences = []
        from tqdm import tqdm

        for batch_idx in tqdm(range(num_batches)):
            current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
            start_tokens = torch.full((current_batch_size, 1), 4098, device=self.device)
            with torch.no_grad():
                generated_seqs = self.model.generate(
                    start_tokens, max_new_tokens=2048, end_token=4099, temperature=0.8, top_k=None
                )
            generated_seqs = generated_seqs - 1
            for idx in range(current_batch_size):
                sample_id = batch_idx * batch_size + idx
                generated_sequences.append({
                    'sample_id': sample_id,
                    'sequence': generated_seqs[idx].cpu().numpy()
                })

        self._save_sequences(generated_sequences, save_path)
        return generated_sequences

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        current_batch_size = self.predict_batch_size
        start_tokens = torch.full((current_batch_size, 1), 4098, device=self.device)
        generated_seqs = self.model.generate(
            start_tokens, max_new_tokens=2048, end_token=4099, temperature=1, top_k=None, top_p=None
        )
        generated_seqs = generated_seqs - 1
        batch_results = []
        for idx in range(current_batch_size):
            sample_id = batch_idx * current_batch_size + idx
            batch_results.append({
                'sample_id': sample_id,
                'sequence': generated_seqs[idx].cpu().numpy()
            })
        return batch_results

    def generate_samples_lightning(self, num_samples=1, batch_size=4, save_path="output/generation_results"):
        self.predict_batch_size = batch_size
        num_gpus = torch.cuda.device_count()
        samples_per_gpu = num_samples // num_gpus
        num_batches = (num_samples + batch_size - 1) // batch_size
        dummy_dataloader = DataLoader(torch.zeros(num_batches), batch_size=1, num_workers=8)

        trainer = Trainer(accelerator="gpu", devices=-1, strategy="ddp")
        predictions = trainer.predict(self, dummy_dataloader)

        all_sequences = []
        for batch_predictions in predictions:
            all_sequences.extend(batch_predictions)
        all_sequences = all_sequences[:num_samples]

        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        start_idx = local_rank * samples_per_gpu
        for i, seq in enumerate(all_sequences):
            seq['sample_id'] = start_idx + i

        self._save_sequences(all_sequences, save_path, local_rank)
        return all_sequences

    def _save_sequences(self, generated_sequences, save_path, local_rank=-1):
        import pickle
        import time
        os.makedirs(save_path, exist_ok=True)
        formatted_data = []
        print(f"generated_sequences: {len(generated_sequences)}")
        for item in generated_sequences:
            formatted_data.append({
                'node_sequence': item['sequence'],
                'filename': f"generated_{item['sample_id']}"
            })
        if local_rank == -1:
            output_path = os.path.join(save_path, 'generated_sequences.pkl')
        else:
            current_time = int(time.time() * 1000)
            output_path = os.path.join(save_path, f'generated_sequences_gpu_{local_rank}_{current_time}.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(formatted_data, f)
        print(f"Generated {len(formatted_data)} sequences saved to {output_path}")
