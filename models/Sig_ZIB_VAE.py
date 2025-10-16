import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from typing import List, Tuple, Optional
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


class Sig_ZIB_VAE(nn.Module):
    """
    Signature-Guided Zero-Inflated Beta VAE for normalized gene expression data.
    
    This model assumes data is normalized to [0,1] and follows a zero-inflated beta distribution:
    - Zero component: Models dropout/zero expression events
    - Beta component: Models continuous expression values in (0,1)
    """
    def __init__(
            self,
            signature_tensor: torch.Tensor,
            encoder_dim_list: List[int] = [20, 10, 5],
            decoder_dim_list: List[int] = [5, 10, 20],
            latent_dim: int = 5,
            activation: str = "relu",
            min_beta_param: float = 1e-3,  # Minimum value for beta distribution parameters
            max_beta_param: float = 100.0,  # Maximum value for beta distribution parameters
            ):
        """
        Initialize the Zero-Inflated Beta VAE model.

        Args:
            signature_tensor: A 2D tensor containing the molecular signatures [n_genes, n_signatures]
            encoder_dim_list: List of dimensions for the encoder layers
            decoder_dim_list: List of dimensions for the decoder layers
            latent_dim: Dimension of the latent space
            activation: Activation function to use in the model
            min_beta_param: Minimum value for beta distribution parameters (for numerical stability)
            max_beta_param: Maximum value for beta distribution parameters (for numerical stability)
        """

        super(Sig_ZIB_VAE, self).__init__()

        # Register the signature tensor as a buffer
        self.register_buffer("signature_tensor", signature_tensor)

        # Ensure signature tensor has shape [n_genes, n_signatures]
        if len(self.signature_tensor.shape) != 2:
            raise ValueError(f"Signature tensor must be 2D, got shape {self.signature_tensor.shape}")
        
        # Check if dimensions appear to be transposed
        if self.signature_tensor.shape[0] < self.signature_tensor.shape[1]:
            warnings.warn(
                f"WARNING: Your signature tensor shape {self.signature_tensor.shape} has more columns than rows. "
                f"The expected format is [n_genes, n_signatures] where typically n_genes > n_signatures. "
                f"Please verify your input tensor orientation.")

        self.n_genes = self.signature_tensor.shape[0]
        self.n_signatures = self.signature_tensor.shape[1]
        self.latent_dim = latent_dim
        self.activation = activation
        self.min_beta_param = min_beta_param
        self.max_beta_param = max_beta_param

        # Define network structures
        self.encoder_layers = [self.n_genes] + encoder_dim_list + [self.latent_dim]
        self.decoder_layers = [self.latent_dim] + decoder_dim_list
        
        # Build the encoder
        self.encoder = self._build_network(self.encoder_layers, activation, network_type="encoder")
        
        # Build shared decoder trunk
        self.decoder_shared = self._build_network(self.decoder_layers, activation, network_type="decoder_shared")
        
        # Build decoder heads
        last_decoder_dim = decoder_dim_list[-1] if decoder_dim_list else self.latent_dim
        
        # Zero probability decoder (dropout events)
        self.zero_decoder = nn.Sequential(
            nn.Linear(last_decoder_dim, self.n_genes),
            nn.Sigmoid()  # Outputs probability of being zero
        )
        
        # Beta distribution parameter decoders
        # Alpha and beta parameters for Beta distribution
        self.alpha_decoder = nn.Sequential(
            nn.Linear(last_decoder_dim, self.n_genes),
            nn.Softplus(),  # Ensures positive values
        )
        
        self.beta_decoder = nn.Sequential(
            nn.Linear(last_decoder_dim, self.n_genes),
            nn.Softplus(),  # Ensures positive values
        )

        # Latent space parameters
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

        # Initialize latent space params
        nn.init.xavier_normal_(self.fc_mu.weight, gain=1.0)
        nn.init.xavier_normal_(self.fc_logvar.weight, gain=1.0)

        # Cell type classifier
        if self.latent_dim <= self.n_signatures:
            self.classifier = nn.Sequential(
                nn.Linear(latent_dim, self.n_signatures),
                nn.LayerNorm(self.n_signatures),
                nn.ReLU(),
                nn.Linear(self.n_signatures, self.n_signatures)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(latent_dim, latent_dim // 2),
                nn.LayerNorm(latent_dim // 2),
                nn.ReLU(),
                nn.Linear(latent_dim // 2, self.n_signatures)
            )

    def _build_network(self, layers: List[int], activation: str, network_type: str) -> nn.Sequential:
        """
        Build a MLP neural network with the specified layer sizes and activation function.
        """
        net = []
        for i in range(1, len(layers)):
            net.append(nn.Linear(layers[i-1], layers[i]))
            net.append(nn.LayerNorm(layers[i]))

            if activation == "relu":
                net.append(nn.ReLU())
            elif activation == "sigmoid":
                net.append(nn.Sigmoid())
            elif activation == "leaky_relu":
                net.append(nn.LeakyReLU(0.1))
            elif activation == "elu":
                net.append(nn.ELU())
            else:
                raise ValueError(f"Unsupported activation function: {activation}")

        # Remove last activation for shared decoder
        if network_type == "decoder_shared" and len(net) > 0:
            if isinstance(net[-1], (nn.ReLU, nn.Sigmoid, nn.LeakyReLU, nn.ELU)):
                net.pop()

        return nn.Sequential(*net)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode gene expression to latent parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling from a Gaussian."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def classify(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict cell type probabilities from latent representation."""
        logits = self.classifier(z)
        probs = F.softmax(logits, dim=1)
        return logits, probs
    
    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode latent representation to distribution parameters.
        
        Returns:
            - zero_probs: Probability of zero expression for each gene
            - alpha: Alpha parameter of Beta distribution
            - beta: Beta parameter of Beta distribution
        """
        # Shared decoder layers
        h = self.decoder_shared(z)
        
        # Zero probability (dropout)
        zero_probs = self.zero_decoder(h)
        
        # Beta distribution parameters (clipped for numerical stability)
        alpha = torch.clamp(self.alpha_decoder(h), min=self.min_beta_param, max=self.max_beta_param)
        beta = torch.clamp(self.beta_decoder(h), min=self.min_beta_param, max=self.max_beta_param)
        
        return zero_probs, alpha, beta
    
    def get_expression_mean(self, zero_probs: torch.Tensor, alpha: torch.Tensor, 
                           beta: torch.Tensor) -> torch.Tensor:
        """
        Compute the expected expression value.
        E[X] = (1 - p_zero) * E[Beta(alpha, beta)]
        where E[Beta(alpha, beta)] = alpha / (alpha + beta)
        """
        beta_mean = alpha / (alpha + beta + 1e-8)
        expected_expr = (1 - zero_probs) * beta_mean
        return expected_expr
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through the model.
        
        Returns:
            x_recon: Reconstructed gene expression (expected value)
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
            z: Sampled latent vector
            cell_logits: Unnormalized cell type predictions
            cell_probs: Cell type probabilities
            zero_probs: Probability of zero expression
            alpha: Alpha parameter of Beta distribution
            beta: Beta parameter of Beta distribution
        """
        # Encode gene expression to latent parameters
        mu, logvar = self.encode(x)

        # Sample from latent vector
        z = self.reparameterize(mu, logvar)

        # Predict cell types
        cell_logits, cell_probs = self.classify(z)

        # Decode to distribution parameters
        zero_probs, alpha, beta_param = self.decode(z)
        
        # Compute expected value for reconstruction
        x_recon = self.get_expression_mean(zero_probs, alpha, beta_param)
        
        return x_recon, mu, logvar, z, cell_logits, cell_probs, zero_probs, alpha, beta_param


def compute_zib_loss(
    x: torch.Tensor, 
    x_recon: torch.Tensor,
    zero_probs: torch.Tensor,
    alpha: torch.Tensor,
    beta_param: torch.Tensor,
    mu: torch.Tensor, 
    logvar: torch.Tensor, 
    cell_logits: torch.Tensor, 
    cell_probs: torch.Tensor, 
    labels: Optional[torch.Tensor] = None,
    kl_weight: float = 1.0, 
    class_weight: float = 1.0,
    zero_weight: float = 1.0,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Zero-Inflated Beta VAE loss.

    Args:
        x: Original gene expression data (normalized to [0,1])
        x_recon: Reconstructed expected expression
        zero_probs: Probability of zero expression
        alpha: Alpha parameter of Beta distribution
        beta_param: Beta parameter of Beta distribution
        mu: Latent mean
        logvar: Latent log variance
        cell_logits: Logits from cell type classifier
        cell_probs: Cell type probabilities
        labels: True cell type labels (if available)
        kl_weight: Weight for KL divergence
        class_weight: Weight for classification loss
        zero_weight: Weight for zero-inflation loss
        eps: Small constant for numerical stability
        
    Returns:
        Tuple of (total_loss, recon_loss, kl_loss, class_loss, zero_loss)
    """
    batch_size = x.shape[0]

    # 1. Create masks for zero and non-zero values
    zero_mask = (x < eps).float()  # 1 where x is zero, 0 otherwise
    nonzero_mask = 1 - zero_mask

    # 2. Zero-inflation loss (BCE for predicting zeros)
    zero_bce = F.binary_cross_entropy(zero_probs, zero_mask, reduction='none')
    zero_loss = zero_bce.sum() / batch_size

    # 3. Beta distribution loss for non-zero values
    # Clip x to avoid numerical issues at boundaries
    x_clipped = torch.clamp(x, min=eps, max=1-eps)
    
    # Compute log probability of Beta distribution
    # log p(x | alpha, beta) = log[Gamma(alpha + beta)] - log[Gamma(alpha)] - log[Gamma(beta)]
    #                          + (alpha - 1) * log(x) + (beta - 1) * log(1 - x)
    
    # Using the Beta distribution from PyTorch
    try:
        beta_dist = Beta(alpha, beta_param)
        log_prob = beta_dist.log_prob(x_clipped)
        
        # Only compute loss for non-zero values
        beta_nll = -nonzero_mask * log_prob
        
        # Average over non-zero values
        n_nonzero = nonzero_mask.sum() + eps
        beta_loss = beta_nll.sum() / n_nonzero
        
    except ValueError:
        # Fallback to manual computation if Beta distribution fails
        log_x = torch.log(x_clipped + eps)
        log_1mx = torch.log(1 - x_clipped + eps)
        
        # Simplified beta log probability (ignoring normalizing constant for stability)
        log_prob_unnorm = (alpha - 1) * log_x + (beta_param - 1) * log_1mx
        
        beta_nll = -nonzero_mask * log_prob_unnorm
        n_nonzero = nonzero_mask.sum() + eps
        beta_loss = beta_nll.sum() / n_nonzero

    # Combined reconstruction loss
    recon_loss = beta_loss + zero_weight * zero_loss

    # 4. KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

    # 5. Classification loss if labels are provided
    if labels is not None:
        class_loss = F.cross_entropy(cell_logits, labels, reduction='sum') / batch_size
    else:
        class_loss = torch.tensor(0.0, device=x.device)

    # Total loss
    total_loss = recon_loss + kl_weight * kl_loss + class_weight * class_loss
    
    return total_loss, recon_loss, kl_weight * kl_loss, class_weight * class_loss, zero_loss


def train_zib_model(
    model: Sig_ZIB_VAE,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 100,
    val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    kl_weight: float = 0.01,
    class_weight: float = 1.0,
    zero_weight: float = 1.5,
    kl_warmup_epochs: int = 50,  # Gradual KL warmup
    checkpoint_dir: str = "./checkpoints",
    checkpoint_freq: int = 10,
    early_stopping_patience: int = 20,
    resume_from: Optional[str] = None
):
    """
    Train the Zero-Inflated Beta VAE model with checkpointing and KL warmup.
    """
    # Ensure checkpoint directory exists
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize or get training history
    if hasattr(model, 'history'):
        history = model.history
    else:
        history = {
            "epoch": [],
            "train_total_loss": [],
            "train_recon_loss": [],
            "train_kl_loss": [],
            "train_class_loss": [],
            "train_zero_loss": [],
            "val_total_loss": [],
            "val_recon_loss": [],
            "val_kl_loss": [],
            "val_class_loss": [],
            "val_zero_loss": [],
        }
        model.history = history
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = getattr(model, 'best_val_loss', float('inf'))
    if resume_from:
        try:
            checkpoint = torch.load(resume_from)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            model.best_val_loss = best_val_loss
            
            if 'history' in checkpoint:
                model.history = checkpoint['history']
                history = model.history
            
            print(f"Resuming training from epoch {start_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch")
    
    # Early stopping variables
    early_stopping_counter = 0
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # KL warmup schedule
        if kl_warmup_epochs > 0:
            current_kl_weight = kl_weight * min(1.0, (epoch + 1) / kl_warmup_epochs)
        else:
            current_kl_weight = kl_weight
        
        # Training phase
        model.train()

        train_losses = {
            "total": 0.0,
            "recon": 0.0,
            "kl": 0.0,
            "class": 0.0,
            "zero": 0.0,
        }
        
        for batch_idx, (data, labels) in enumerate(train_dataloader):
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            x_recon, mu, logvar, z, logits, cell_probs, zero_probs, alpha, beta_param = model(data)
            
            # Compute loss
            total_loss, recon_loss, kl_loss, class_loss, zero_loss = compute_zib_loss(
                data, x_recon, zero_probs, alpha, beta_param,
                mu, logvar, logits, cell_probs, 
                labels=labels,
                kl_weight=current_kl_weight,
                class_weight=class_weight,
                zero_weight=zero_weight
            )
            
            # Backward pass and optimize
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            # Track losses
            train_losses["total"] += total_loss.item()
            train_losses["recon"] += recon_loss.item()
            train_losses["kl"] += kl_loss.item()
            train_losses["class"] += class_loss.item()
            train_losses["zero"] += zero_loss.item()
        
        # Compute average training losses
        for key in train_losses:
            train_losses[key] /= len(train_dataloader)
        
        # Validation phase
        val_losses = {
            "total": 0.0,
            "recon": 0.0,
            "kl": 0.0,
            "class": 0.0,
            "zero": 0.0,
        }
        
        if val_dataloader:
            model.eval()
            with torch.no_grad():
                for batch_idx, (data, labels) in enumerate(val_dataloader):                    
                    # Forward pass
                    x_recon, mu, logvar, z, logits, cell_probs, zero_probs, alpha, beta_param = model(data)
                    
                    # Compute loss (use full KL weight for validation)
                    total_loss, recon_loss, kl_loss, class_loss, zero_loss = compute_zib_loss(
                        data, x_recon, zero_probs, alpha, beta_param,
                        mu, logvar, logits, cell_probs,
                        labels=labels,
                        kl_weight=kl_weight,  # Use full weight for validation
                        class_weight=class_weight,
                        zero_weight=zero_weight
                    )
                    
                    # Track losses
                    val_losses["total"] += total_loss.item()
                    val_losses["recon"] += recon_loss.item()
                    val_losses["kl"] += kl_loss.item()
                    val_losses["class"] += class_loss.item()
                    val_losses["zero"] += zero_loss.item()
            
            # Compute average validation losses
            for key in val_losses:
                val_losses[key] /= len(val_dataloader)
        
        # Update history
        model.history["epoch"].append(epoch + 1)
        model.history["train_total_loss"].append(train_losses["total"])
        model.history["train_recon_loss"].append(train_losses["recon"])
        model.history["train_kl_loss"].append(train_losses["kl"])
        model.history["train_class_loss"].append(train_losses["class"])
        model.history["train_zero_loss"].append(train_losses["zero"])
        
        if val_dataloader:
            model.history["val_total_loss"].append(val_losses["total"])
            model.history["val_recon_loss"].append(val_losses["recon"])
            model.history["val_kl_loss"].append(val_losses["kl"])
            model.history["val_class_loss"].append(val_losses["class"])
            model.history["val_zero_loss"].append(val_losses["zero"])
            
            # Check for best model
            current_val_loss = val_losses["total"]
            is_best = current_val_loss < best_val_loss
            
            if is_best:
                best_val_loss = current_val_loss
                model.best_val_loss = best_val_loss
                early_stopping_counter = 0
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'history': model.history,
                }, os.path.join(checkpoint_path, 'best_model.pt'))
            else:
                early_stopping_counter += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            log_msg = f"Epoch: {epoch + 1}/{num_epochs}, "
            log_msg += f"Train Loss: {train_losses['total']:.4f} "
            log_msg += f"(R:{train_losses['recon']:.3f}, "
            log_msg += f"KL:{train_losses['kl']:.3f}, "
            log_msg += f"C:{train_losses['class']:.3f}, "
            log_msg += f"Z:{train_losses['zero']:.3f})"
            
            if val_dataloader:
                log_msg += f", Val: {val_losses['total']:.4f}"
                if is_best:
                    log_msg += " *"
            
            print(log_msg)
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_freq == 0 or epoch == num_epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss if val_dataloader else None,
                'history': model.history,
            }, os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch+1}.pt'))
        
        # Early stopping
        if val_dataloader and early_stopping_patience > 0 and early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Load best model if validation was used
    if val_dataloader:
        try:
            best_model_path = os.path.join(checkpoint_path, 'best_model.pt')
            if os.path.exists(best_model_path):
                checkpoint = torch.load(best_model_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'history' in checkpoint:
                    model.history = checkpoint['history']
                print(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")
        except Exception as e:
            print(f"Error loading best model: {e}")



def generate_samples_from_signatures(model,
                                     n_samples_per_type = 100,
                                     signature_names = None,
                                     sample_from_distribution = True):
    """
    Generate new samples for each cell type signature.
    Returns:
        Dictionary with generated samples and metadata for each cell type.
    """
    if signature_names is None:
        signature_names = ["C3", "C4", "EMT", "C10", "ciliated"]
    
    model.eval()
    n_signatures = len(signature_names)
    latent_dim = model.latent_dim

    generated_data = {}

    with torch.no_grad():
        for sig_idx, sig_name in enumerate(signature_names):
            print(f"Generating samples for signature {sig_name}...")

            # Sample from learnt latent distribution of this cell type.
            # First, we need to find the latent representation of this signature.
            # We can do this by encoding the signature from the signature matrix.

            sig_vec = model.signature_tensor[:, sig_idx].unsqueeze(0)  # Shape (1, n_genes)

            # Encode to get the latent distribution parameters
            mu_sig, logvar_sig = model.encode(sig_vec)


            # Sample multiple points from this distribution
            z_samples = []
            for _ in range(n_samples_per_type):
                z = model.reparameterize(mu_sig, logvar_sig)
                z_samples.append(z)
            
            z_samples = torch.cat(z_samples, dim=0)  # [n_samples, latent_dim]
            
            # Decode to get distribution parameters
            zero_probs, alpha, beta_param = model.decode(z_samples)
            
            # Generate samples based on the distribution parameters
            if sample_from_distribution:
                # Sample from the Zero-Inflated Beta distribution
                samples = sample_from_zi_beta(zero_probs, alpha, beta_param)
            else:
                # Use expected value
                samples = model.get_expression_mean(zero_probs, alpha, beta_param)
            
            # Store results
            generated_data[sig_name] = {
                'samples': samples.cpu().numpy(),
                'zero_probs': zero_probs.cpu().numpy(),
                'alpha': alpha.cpu().numpy(),
                'beta': beta_param.cpu().numpy(),
                'latent_codes': z_samples.cpu().numpy(),
                'signature_index': sig_idx
            }

    return generated_data



def sample_from_zi_beta(zero_probs, alpha, beta_param):
    """
    Sample from Zero-Inflated Beta distribution.
    
    Args:
        zero_probs: Probability of zero expression [batch_size, n_genes]
        alpha: Alpha parameter of Beta distribution [batch_size, n_genes]
        beta_param: Beta parameter of Beta distribution [batch_size, n_genes]
    
    Returns:
        Sampled expression values [batch_size, n_genes]
    """
    # Sample binary mask for zero/non-zero
    zero_mask = torch.bernoulli(zero_probs)
    
    # Sample from Beta distribution
    beta_dist = Beta(alpha, beta_param)
    beta_samples = beta_dist.sample()
    
    # Apply zero mask
    samples = beta_samples * (1 - zero_mask)
    
    return samples
