"""Training utilities for Zero-Inflated Beta VAE with Weights & Biases logging.

Usage (CLI):
  python train_models.py \
	--expr-path data_processed/aggregated_expression.pt \
	--signature-path data_processed/gene_signature.pt \
	--gene-names-path data_processed/gene_names.txt \
	--project spatial_zib_vae --entity your_wandb_entity

Within a notebook:
  from train_models import run_multiseed_experiments
  results = run_multiseed_experiments(expression_tensor, signature_tensor, gene_names=gene_names)

Assumptions:
  - expression_tensor shape: [n_samples, n_genes], already normalized to [0,1]
  - signature_tensor shape:  [n_genes, n_signatures]
  - Optional gene_names list length == n_genes

If wandb is not installed, logging gracefully degrades to stdout only.
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from utils.processing_utils import pytorch_pearson_corr
from sklearn.model_selection import train_test_split

try:  # Optional wandb
	import wandb  # type: ignore
	WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover
	WANDB_AVAILABLE = False
	class _RunStub:
		def __init__(self):
			self.config = {}
			self.name = "stub"
			self.summary = {}
		def log(self, *a, **k): pass
		def finish(self): pass
	class _ArtifactStub:
		def add_file(self, *a, **k): pass
	class _WBStub:  # minimal stub with expected API surface
		def init(self, *a, **k): return _RunStub()
		def Image(self, x, caption=None): return None
		def save(self, *a, **k): pass
		def Artifact(self, *a, **k): return _ArtifactStub()
		def log(self, *a, **k): pass
		def log_artifact(self, *a, **k): pass
	wandb = _WBStub()  # type: ignore

from models.Sig_ZIB_VAE import Sig_ZIB_VAE, train_zib_model  # reuse training loop

# ----------------------------- Utility Dataclasses ----------------------------- #

@dataclass
class TrainConfig:
	encoder_dim_list: List[int] = None  # type: ignore
	decoder_dim_list: List[int] = None  # type: ignore
	latent_dim: int = 20
	activation: str = "leaky_relu"
	batch_size: int = 256
	val_split: float = 0.2
	num_epochs: int = 800
	kl_weight: float = 0.0025
	class_weight: float = 5.0
	zero_weight: float = 2.0
	kl_warmup_epochs: int = 100
	learning_rate: float = 5e-4
	weight_decay: float = 1e-5
	early_stopping_patience: int = 60
	checkpoint_root: str = "ZIB_VAE_runs"
	project: str = "zib_vae"
	entity: Optional[str] = None
	group: Optional[str] = None
	seeds: List[int] = None  # type: ignore
	use_umap: bool = True

	def finalize(self):
		if self.encoder_dim_list is None:
			self.encoder_dim_list = [100, 80, 40, 20]
		if self.decoder_dim_list is None:
			self.decoder_dim_list = [20, 40, 80, 100]
		if self.seeds is None:
			self.seeds = [137, 42, 123, 456, 789, 999, 111, 222, 333, 444]
		return self


# ----------------------------- Core Helper Functions -------------------------- #

def set_seed(seed: int):
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def build_dataloaders(expression_tensor: torch.Tensor,
					signature_tensor: torch.Tensor,
					batch_size: int,
					val_split: float,
					data_seed: int = 73,  # fixed for reproducible results
					training_seed: Optional[int] = None):
	
    correlation = pytorch_pearson_corr(expression_tensor, signature_tensor)
    correlation_probs = F.softmax(correlation, dim = 1)
    cell_labels = torch.argmax(correlation_probs, dim = 1)

    # Use fixed seed for consistent train/val split across all runs
    X_train, X_val, y_train, y_val = train_test_split(
        expression_tensor,
        cell_labels,
        test_size = val_split,
        random_state = data_seed,  # Keep this fixed (73) for all seeds
        stratify = cell_labels
    )

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    full_dataset = TensorDataset(expression_tensor, cell_labels)

    # Create generator for reproducible shuffling
    generator = None
    if training_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(training_seed)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=generator)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=generator)

    return train_dataloader, val_dataloader, (X_train, y_train, X_val, y_val)


def extract_latent(model: Sig_ZIB_VAE, X: torch.Tensor) -> torch.Tensor:
	model.eval()
	zs = []
	with torch.no_grad():
		for i in range(0, X.shape[0], 1024):
			batch = X[i:i+1024]
			mu, logvar = model.encode(batch)
			zs.append(mu.cpu())  # use mean for deterministic embedding
	return torch.cat(zs, dim=0)


def make_projections(z: np.ndarray, seed: int, use_umap: bool = True) -> Dict[str, np.ndarray]:
	from sklearn.manifold import TSNE
	proj = {}
	tsne_2d = TSNE(n_components=2, random_state=seed, init="pca", perplexity=min(30, max(5, z.shape[0]//50)))
	proj["tsne2d"] = tsne_2d.fit_transform(z)
	try:
		tsne_3d = TSNE(n_components=3, random_state=seed, init="pca", perplexity=min(30, max(5, z.shape[0]//50)))
		proj["tsne3d"] = tsne_3d.fit_transform(z)
	except Exception:
		pass
	if use_umap:
		try:
			import umap  # type: ignore
			reducer2d = umap.UMAP(n_components=2, random_state=seed)
			proj["umap2d"] = reducer2d.fit_transform(z)
			reducer3d = umap.UMAP(n_components=3, random_state=seed)
			proj["umap3d"] = reducer3d.fit_transform(z)
		except Exception:
			pass
	return proj


def plot_projection(arr: np.ndarray, labels: np.ndarray, title: str, palette: Optional[Dict[Union[int,str], Union[str,tuple]]] = None):
	import matplotlib.pyplot as plt
	import seaborn as sns
	plt.figure(figsize=(6,5))
	if palette is None:
		uniq = np.unique(labels)
		palette = {u: sns.color_palette("tab10")[i % 10] for i,u in enumerate(uniq)}
	colors = [palette[l] for l in labels]
	if arr.shape[1] == 2:
		plt.scatter(arr[:,0], arr[:,1], c=colors, s=8, alpha=0.7)
	else:
		from mpl_toolkits.mplot3d import Axes3D  # noqa
		ax = plt.figure(figsize=(6,5)).add_subplot(111, projection='3d')
		ax.scatter(arr[:,0], arr[:,1], arr[:,2], c=colors, s=8, alpha=0.7)
		ax.set_title(title)
		return plt.gcf()
	plt.title(title)
	plt.tight_layout()
	return plt.gcf()


def summarize_reconstruction(model: Sig_ZIB_VAE, X: torch.Tensor) -> Dict[str, float]:
	model.eval()
	with torch.no_grad():
		x_recon, *_ = model(X)
	mse = F.mse_loss(x_recon, X, reduction='mean').item()
	var = torch.var(X, unbiased=False).item()
	nmse = mse / (var + 1e-8)
	zero_rate_true = (X < 1e-8).float().mean().item()
	zero_rate_pred = (x_recon < 1e-4).float().mean().item()
	return {"mse": mse, "nmse": nmse, "zero_rate_true": zero_rate_true, "zero_rate_pred": zero_rate_pred}


# ----------------------------- Multi-seed Runner ------------------------------ #

def run_single_seed(seed: int, config: TrainConfig, expression_tensor: torch.Tensor, signature_tensor: torch.Tensor, gene_names: Optional[List[str]] = None):
	set_seed(seed)
	run_name = f"zib_seed_{seed}"
	arch_str = "_".join(map(str, config.encoder_dim_list))
	ckpt_dir = Path(config.checkpoint_root) / f"{arch_str}_latent_{config.latent_dim}" / f"seed_{seed}"
	ckpt_dir.mkdir(parents=True, exist_ok=True)

	wb_run = None
	if WANDB_AVAILABLE:
		wb_run = wandb.init(
			project=config.project,
			entity=config.entity,
			group=config.group or f"arch_{arch_str}_lat{config.latent_dim}",
			name=run_name,
			config={**asdict(config), "seed": seed, "arch": arch_str},
			reinit=True,
		)

	train_loader, val_loader, (X_train, y_train, X_val, y_val) = build_dataloaders(expression_tensor, 
																					signature_tensor,
																					config.batch_size,
																					config.val_split,
																					data_seed=73,
																					training_seed=seed)


	model = Sig_ZIB_VAE(
		signature_tensor=signature_tensor,
		encoder_dim_list=config.encoder_dim_list,
		decoder_dim_list=config.decoder_dim_list,
		latent_dim=config.latent_dim,
		activation=config.activation,
        min_beta_param=0.01,  # Prevent numerical instability
        max_beta_param=50.0   # Prevent extreme distributions
	)

	optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, betas = (0.9, 0.999))

	# For simplicity, we wrap train_zib_model without modifying its internals;
    # wandb captures metrics after training via history.
	train_zib_model(
		model,
		train_loader,
		optimizer,
		num_epochs=config.num_epochs,
		val_dataloader=val_loader,
		kl_weight=config.kl_weight,
		class_weight=config.class_weight,
		zero_weight=config.zero_weight,
		kl_warmup_epochs=config.kl_warmup_epochs,
		checkpoint_dir=str(ckpt_dir),
		checkpoint_freq=50,
		early_stopping_patience=config.early_stopping_patience,
		resume_from=None,
	)

	# Log history to wandb
	if wb_run and hasattr(model, 'history'):
		hist = model.history
		for i in range(len(hist["epoch"])):
			row = {k: hist[k][i] for k in hist if k != "epoch"}
			row["epoch"] = hist["epoch"][i]
			wandb.log(row)

	# Metrics & projections
	recon_stats = summarize_reconstruction(model, X_val)
	z_val = extract_latent(model, X_val).numpy()
	projections = make_projections(z_val, seed=seed, use_umap=config.use_umap)

	if wb_run:
		for k, v in recon_stats.items():
			wandb.log({f"val/{k}": v})
		# Plot and log projections
		label_arr = y_val.cpu().numpy()
		for name, arr in projections.items():
			fig = plot_projection(arr, label_arr, title=f"{run_name}_{name}")
			wandb.log({f"proj/{name}": wandb.Image(fig)})
		# Save best model artifact
		best_path = ckpt_dir / 'best_model.pt'
		if best_path.exists():
			art = wandb.Artifact(run_name + "_model", type="model")
			art.add_file(str(best_path))
			wandb.log_artifact(art)
		wb_run.summary.update(recon_stats)
		wb_run.finish()

	return {
		"seed": seed,
		"checkpoint_dir": str(ckpt_dir),
		"recon_stats": recon_stats,
		"val_loss": model.best_val_loss if hasattr(model, 'best_val_loss') else None,
		"model": model,
		"projections": projections,
	}


def run_multiseed_experiments(expression_tensor: torch.Tensor, signature_tensor: torch.Tensor, gene_names: Optional[List[str]] = None, config: Optional[TrainConfig] = None) -> Dict[str, Any]:
	config = (config or TrainConfig()).finalize()
	results = []
	for seed in config.seeds:
		res = run_single_seed(seed, config, expression_tensor, signature_tensor, gene_names)
		results.append(res)
	# Select best
	best = min(results, key=lambda r: r["val_loss"] if r["val_loss"] is not None else float('inf'))
	print(f"Best seed: {best['seed']} (val_loss={best['val_loss']:.4f})")
	return {"all_results": results, "best": best, "config": asdict(config)}


# ----------------------------- CLI Interface ---------------------------------- #

def load_tensor(path: str) -> torch.Tensor:
	p = Path(path)
	if p.suffix == '.pt':
		return torch.load(p)
	elif p.suffix in ('.npy', '.npz'):
		return torch.tensor(np.load(p), dtype=torch.float32)
	else:
		raise ValueError(f"Unsupported tensor file format: {p.suffix}")


def main():  # pragma: no cover
	parser = argparse.ArgumentParser(description="Train Signature-Guided Zero-Inflated Beta VAE with multi-seed wandb logging")
	parser.add_argument('--expr-path', type=str, required=True, help='Path to expression tensor (.pt or .npy)')
	parser.add_argument('--signature-path', type=str, required=True, help='Path to signature tensor (.pt or .npy)')
	parser.add_argument('--gene-names-path', type=str, required=False, help='Optional gene names text file')
	parser.add_argument('--project', type=str, default='zib_vae')
	parser.add_argument('--entity', type=str, default=None)
	parser.add_argument('--group', type=str, default=None)
	parser.add_argument('--latent-dim', type=int, default=20)
	parser.add_argument('--epochs', type=int, default=500)
	parser.add_argument('--batch-size', type=int, default=256)
	parser.add_argument('--seeds', type=str, default='137,42,123,456,789,999,111,222,333,444')
	parser.add_argument('--no-umap', action='store_true', help='Disable UMAP projections')
	args = parser.parse_args()

	if args.expr_path is None or args.signature_path is None:
		raise SystemExit('--expr-path and --signature-path are required for CLI usage.')

	expression_tensor = load_tensor(args.expr_path)
	signature_tensor = load_tensor(args.signature_path)

	print("Required tensors loaded successfully!")

	gene_names = None
	if args.gene_names_path and Path(args.gene_names_path).exists():
		with open(args.gene_names_path) as f:
			gene_names = [ln.strip() for ln in f if ln.strip()]

	config = TrainConfig(
		latent_dim=args.latent_dim,
		project=args.project,
		entity=args.entity,
		group=args.group,
		num_epochs=args.epochs,
		batch_size=args.batch_size,
		use_umap=not args.no_umap,
	).finalize()
	# Override seeds if provided
	if args.seeds:
		config.seeds = [int(s) for s in args.seeds.split(',')]

	print("Model successfully initialized!")
	print("Beginning training with multiple seeds...")

	run_multiseed_experiments(expression_tensor, signature_tensor, gene_names, config)

if __name__ == '__main__':
	main()

