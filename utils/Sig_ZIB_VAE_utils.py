import os, re
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from tabulate import tabulate

from utils.processing_utils import pytorch_pearson_corr
from models.Sig_ZIB_VAE import Sig_ZIB_VAE

def compute_basic_metrics(model, X_tr, y_tr, X_va, y_va):
    model.eval()
    with torch.no_grad():
        # Train
        x_recon_tr, _, _, _, _, cell_probs_tr, _, _, _ = model(X_tr)
        y_pred_tr = torch.argmax(cell_probs_tr, dim=1)
        mse_tr = nn.MSELoss()(x_recon_tr, X_tr).item()
        mae_tr = torch.mean(torch.abs(x_recon_tr - X_tr)).item()
        acc_tr = accuracy_score(y_tr.numpy(), y_pred_tr.numpy())
        prec_tr = precision_score(y_tr.numpy(), y_pred_tr.numpy(), average='weighted', zero_division=0)
        rec_tr = recall_score(y_tr.numpy(), y_pred_tr.numpy(), average='weighted', zero_division=0)
        f1_tr = f1_score(y_tr.numpy(), y_pred_tr.numpy(), average='weighted', zero_division=0)

        # Val
        x_recon_va, _, _, _, _, cell_probs_va, _, _, _ = model(X_va)
        y_pred_va = torch.argmax(cell_probs_va, dim=1)
        mse_va = nn.MSELoss()(x_recon_va, X_va).item()
        mae_va = torch.mean(torch.abs(x_recon_va - X_va)).item()
        acc_va = accuracy_score(y_va.numpy(), y_pred_va.numpy())
        prec_va = precision_score(y_va.numpy(), y_pred_va.numpy(), average='weighted', zero_division=0)
        rec_va = recall_score(y_va.numpy(), y_pred_va.numpy(), average='weighted', zero_division=0)
        f1_va = f1_score(y_va.numpy(), y_pred_va.numpy(), average='weighted', zero_division=0)

    return dict(
        mse_tr=mse_tr, mae_tr=mae_tr, acc_tr=acc_tr, prec_tr=prec_tr, rec_tr=rec_tr, f1_tr=f1_tr,
        mse_va=mse_va, mae_va=mae_va, acc_va=acc_va, prec_va=prec_va, rec_va=rec_va, f1_va=f1_va
    )

def load_all_best_models(root_dir="ZIB_VAE_runs"):
    """
    Traverses ZIB_VAE_runs/*_latent_*/seed_*/best_model.pt
    Assumes consistent architecture (uses global encoder_dim_list, decoder_dim_list, latent_dim, activation).
    """
    pattern_arch = re.compile(r".+_latent_(\d+)$")
    model_entries = []
    if not os.path.isdir(root_dir):
        print(f"Directory {root_dir} not found.")
        return []

    for arch_dir in sorted(os.listdir(root_dir)):
        full_arch_path = os.path.join(root_dir, arch_dir)
        if not os.path.isdir(full_arch_path):
            continue
        if not pattern_arch.match(full_arch_path):
            # Still accept anything with 'latent_' substring
            if "latent_" not in arch_dir:
                continue
        # Seed subfolders
        for seed_dir in sorted(os.listdir(full_arch_path)):
            if not seed_dir.startswith("seed_"):
                continue
            seed_path = os.path.join(full_arch_path, seed_dir)
            model_path = os.path.join(seed_path, "best_model.pt")
            if os.path.isfile(model_path):
                # Extract seed number
                try:
                    seed_num = int(seed_dir.split("_")[1])
                except:
                    seed_num = seed_dir
                model_entries.append((arch_dir, seed_num, model_path))
    return model_entries



def plot_model_training_curves_zib(model):
    """
    Method for visualising training curves, for both training and validation datasets.
    """
    
    # 1. Training curves
    if hasattr(model, 'history') and model.history:
        epoch = model.history["epoch"]
        total_loss = model.history["train_total_loss"]
        recon_loss = model.history["train_recon_loss"]
        kl_loss = model.history["train_kl_loss"]
        class_loss = model.history["train_class_loss"]
        zero_loss = model.history.get("train_zero_loss", [0]*len(epoch))
        
        plt.figure(figsize=(12, 3.5))
        
        # Training loss components
        plt.subplot(1, 3, 1)
        plt.plot(epoch, total_loss, label="Total loss", linewidth=2)
        plt.plot(epoch, recon_loss, label="Reconstruction loss")
        plt.plot(epoch, kl_loss, label="KL loss")
        plt.plot(epoch, class_loss, label="Classification loss")
        plt.plot(epoch, zero_loss, label="Zero-inflation loss", linestyle='--')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Components")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Train vs Validation
        if "val_total_loss" in model.history and model.history["val_total_loss"]:
            plt.subplot(1, 3, 2)
            plt.plot(epoch, model.history["train_total_loss"], label="Train")
            plt.plot(epoch, model.history["val_total_loss"], label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Total Loss")
            plt.title("Train vs Validation Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Zero loss evolution
        plt.subplot(1, 3, 3)
        plt.plot(epoch, zero_loss, 'g-', linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Zero-inflation Loss")
        plt.title("Zero Prediction Loss Evolution")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def visualise_beta_parameters(model, X_data, gene_names = None, n_genes_to_show = None):
    """
    Visualize the learned Beta distribution parameters.
    """
    model.eval()
    with torch.no_grad():
        _, _, _, _, _, _, zero_probs, alpha, beta_param = model(X_data)
    
    # Convert to numpy
    alpha_np = alpha.numpy()
    beta_np = beta_param.numpy()
    zero_probs_np = zero_probs.numpy()
    
    # Select genes to visualize
    if n_genes_to_show is None:
        gene_indices = np.arange(X_data.shape[1])
        n_genes_to_show = X_data.shape[1]
    else:
        gene_indices = np.random.choice(len(gene_names), n_genes_to_show, replace=False)
    
    fig, axes = plt.subplots(4, n_genes_to_show // 4, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, gene_idx in enumerate(gene_indices):
        ax = axes[idx]
        
        # Get parameters for this gene across all samples
        alphas = alpha_np[:, gene_idx]
        betas = beta_np[:, gene_idx]
        zeros = zero_probs_np[:, gene_idx]
        
        # Plot histogram of actual data
        actual_data = X_data[:, gene_idx].numpy()
        ax.hist(actual_data, bins=30, alpha=0.7, density=True, label='Actual', color='blue')
        
        # Plot average Beta distribution
        mean_alpha = np.mean(alphas)
        mean_beta = np.mean(betas)
        mean_zero = np.mean(zeros)
        
        # Create Beta distribution for visualization
        from scipy.stats import beta as beta_dist
        x_range = np.linspace(0.001, 0.999, 100)
        beta_pdf = beta_dist.pdf(x_range, mean_alpha, mean_beta)
        
        # Scale by non-zero probability
        beta_pdf_scaled = beta_pdf * (1 - mean_zero)
        
        ax.plot(x_range, beta_pdf_scaled, 'r-', lw=2, 
                label=rf'Beta($\alpha$={mean_alpha:.2f}, $\beta$={mean_beta:.2f})')
        
        # Add zero component
        ax.axvline(x=0, color='green', linestyle='--', 
                  label=f'P(zero)={mean_zero:.2f}')
        
        ax.set_title(f'{gene_names[gene_idx]}')
        ax.set_xlabel('Expression')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Learned Beta Distributions for Selected Genes', fontsize=20)
    plt.tight_layout()

def analyze_zib_reconstruction(model, X_data, gene_names):
    """
    Comprehensive analysis of Zero-Inflated Beta VAE reconstruction.
    """

    np.random.seed(137)
    torch.manual_seed(137)

    model.eval()
    with torch.no_grad():
        x_recon, _, _, _, _, _, zero_probs, alpha, beta_param = model(X_data)
    
    # Convert to numpy
    X_data_np = X_data.numpy()
    x_recon_np = x_recon.numpy()
    zero_probs_np = zero_probs.numpy()
    alpha_np = alpha.numpy()
    beta_np = beta_param.numpy()
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Original vs Reconstructed Heatmaps
    ax1 = plt.subplot(3, 4, 1)
    sns.heatmap(X_data_np[:20].T, cmap='viridis', ax=ax1, cbar_kws={'label': ' Normalized Expression'})
    ax1.set_title('Original Expression')
    ax1.set_ylabel('Genes')
    # Fix: Set the y-tick positions first, then labels
    ax1.set_yticks(range(len(gene_names)))
    ax1.set_yticklabels(gene_names, rotation=0, fontsize=8)
    
    ax2 = plt.subplot(3, 4, 2)
    sns.heatmap(x_recon_np[:20].T, cmap='viridis', ax=ax2, cbar_kws={'label': 'Normalized Expression'})
    ax2.set_title('Reconstructed Expression')
    ax2.set_yticklabels([])
    
    # 3. Reconstruction Error
    ax3 = plt.subplot(3, 4, 3)
    error = np.abs(X_data_np[:20] - x_recon_np[:20])
    sns.heatmap(error.T, cmap='Reds', ax=ax3, cbar_kws={'label': 'Error'})
    ax3.set_title('Absolute Error')
    ax3.set_yticklabels([])
    
    # 4. Zero Probability Map
    ax4 = plt.subplot(3, 4, 4)
    sns.heatmap(zero_probs_np[:20].T, cmap='RdYlBu_r', vmin=0, vmax=1, ax=ax4,
                cbar_kws={'label': 'P(zero)'})
    ax4.set_title('Zero Probability')
    ax4.set_yticklabels([])
    
    # 5. Beta Parameters (Alpha)
    ax5 = plt.subplot(3, 4, 5)
    sns.heatmap(np.log1p(alpha_np[:20].T), cmap='plasma', ax=ax5,
                cbar_kws={'label': rf'log($\alpha +1$)'})
    ax5.set_title(rf'Beta $\alpha$ Parameter (log scale)')
    ax5.set_ylabel('Genes')
    # Fix: Set the y-tick positions first, then labels
    ax5.set_yticks(range(len(gene_names)))
    ax5.set_yticklabels(gene_names, rotation=0, fontsize=8)
    
    # 6. Beta Parameters (Beta)
    ax6 = plt.subplot(3, 4, 6)
    sns.heatmap(np.log1p(beta_np[:20].T), cmap='plasma', ax=ax6,
                cbar_kws={'label': rf'log($\beta +1$)'})
    ax6.set_title(rf'Beta $\beta$ Parameter (log scale)')
    ax6.set_yticklabels([])
    
    # 7. Scatter plot: Original vs Reconstructed
    ax7 = plt.subplot(3, 4, 7)
    # Separate zero and non-zero values
    zero_mask = X_data_np.flatten() < 1e-8
    nonzero_mask = ~zero_mask
    
    ax7.scatter(X_data_np.flatten()[nonzero_mask], 
               x_recon_np.flatten()[nonzero_mask], 
               alpha=0.2, s=1, label='Non-zero', c = "blue")
    ax7.scatter(X_data_np.flatten()[zero_mask], 
               x_recon_np.flatten()[zero_mask], 
               alpha=0.5, s=1, c='red', label='Zero')
    ax7.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
    ax7.set_xlabel('Original')
    ax7.set_ylabel('Reconstructed')
    ax7.set_title('Reconstruction Scatter')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Distribution of Alpha parameters
    ax8 = plt.subplot(3, 4, 8)
    ax8.hist(alpha_np.flatten(), bins=50, alpha=0.7, edgecolor='black', color = "blue")
    ax8.set_xlabel('Alpha')
    ax8.set_ylabel('Frequency')
    ax8.set_title(rf'Distribution of $\alpha$ Parameters')
    ax8.axvline(x=1, color='red', linestyle='--', label='α=1 (uniform)')
    ax8.legend()
    
    # 9. Distribution of Beta parameters
    ax9 = plt.subplot(3, 4, 9)
    ax9.hist(beta_np.flatten(), bins=50, alpha=0.7, edgecolor='black', color = "blue")
    ax9.set_xlabel('Beta')
    ax9.set_ylabel('Frequency')
    ax9.set_title(rf'Distribution of $\beta$ Parameters')
    ax9.axvline(x=1, color='red', linestyle='--', label='β=1 (uniform)')
    ax9.legend()
    
    # 10. Zero prediction accuracy per gene
    ax10 = plt.subplot(3, 4, 10)
    zero_mask_true = (X_data_np < 1e-8).astype(float)
    zero_mask_pred = (zero_probs_np > 0.5).astype(float)
    gene_zero_acc = []
    for g in range(len(gene_names)):
        acc = ((zero_mask_true[:, g] == zero_mask_pred[:, g]).mean())
        gene_zero_acc.append(acc)
    
    bars = ax10.barh(range(len(gene_names)), gene_zero_acc, color = "blue", alpha = 0.7)
    ax10.set_yticks(range(len(gene_names)))
    ax10.set_yticklabels(gene_names, fontsize=8)
    ax10.set_xlabel('Zero Prediction Accuracy')
    ax10.set_title('Per-Gene Zero Accuracy')
    ax10.axvline(x=0.8, color='red', linestyle='--', alpha=0.5)
    
    # 11. Reconstruction MSE per gene
    ax11 = plt.subplot(3, 4, 11)
    gene_mse = []
    for g in range(len(gene_names)):
        mse = ((X_data_np[:, g] - x_recon_np[:, g])**2).mean()
        gene_mse.append(mse)
    
    bars = ax11.barh(range(len(gene_names)), gene_mse, color = "blue", alpha = 0.7)
    ax11.set_yticks(range(len(gene_names)))
    ax11.set_yticklabels(gene_names, fontsize=8)
    ax11.set_xlabel('MSE')
    ax11.set_title('Per-Gene Reconstruction MSE')
    
    # 12. Summary Statistics
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    # Calculate statistics
    overall_mse = ((X_data_np - x_recon_np)**2).mean()
    overall_mae = np.abs(X_data_np - x_recon_np).mean()
    zero_acc = ((zero_mask_true == zero_mask_pred).mean())
    zero_precision = (((zero_mask_true == 1) & (zero_mask_pred == 1)).sum() / 
                     (zero_mask_pred == 1).sum()) if (zero_mask_pred == 1).sum() > 0 else 0
    zero_recall = (((zero_mask_true == 1) & (zero_mask_pred == 1)).sum() / 
                  (zero_mask_true == 1).sum()) if (zero_mask_true == 1).sum() > 0 else 0
    
    stats_text = f"""
    Reconstruction Statistics:
    
    MSE: {overall_mse:.5f}
    MAE: {overall_mae:.5f}
    
    Zero Prediction:
    Accuracy: {zero_acc:.3f}
    Precision: {zero_precision:.3f}
    Recall: {zero_recall:.3f}
    
    Data Statistics:
    True zeros: {zero_mask_true.mean():.3f}
    Pred zeros: {zero_mask_pred.mean():.3f}
    
    Beta Parameters:
    Mean α: {alpha_np.mean():.2f}
    Mean β: {beta_np.mean():.2f}
    """
    
    ax12.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
             family='monospace')
    
    plt.suptitle('Zero-Inflated Beta VAE Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    
    return fig, {
        'mse': overall_mse,
        'mae': overall_mae,
        'zero_accuracy': zero_acc,
        'zero_precision': zero_precision,
        'zero_recall': zero_recall
    }


def visualise_latent_space_and_reconstruction(model,
                                                expression_tensor,
                                                signature_tensor,
                                                gene_names = None,
                                                projection = "umap",
                                                latent_plot_kws = dict(),
                                                return_projection = True):
    """
    Visualise the latent space and reconstruction results of the model.

    Requires:
        projection = "umap" or "tsne"
    """

    np.random.seed(137)
    torch.manual_seed(137)

    # NOTE: if title_fontsize is provided in latent_plot_kws, this automatically sets the same title font for all plots
    title_fontsize = latent_plot_kws.get('title_fontsize', 12)

    fig = plt.figure(figsize=(22, 5))

    # 1 & 2. Plot reconstruction comparison
    model.eval()
    with torch.no_grad():
        x_recon, _, _, _, logits, cell_probs, _, _, _ = model(expression_tensor)
    
    # Heatmap comparison (Original vs Reconstructed)
    ax1 = plt.subplot(1, 4, 1)
    sns.heatmap(expression_tensor.T, cmap="viridis", yticklabels=gene_names,
                cbar_kws={'label': 'Expression level'})
    cbar = ax1.collections[0].colorbar
    cbar.set_label('Expression Level', size=12, labelpad=10)
    cbar.ax.tick_params(labelsize=10)
    ax1.set_xticklabels([])
    ax1.set_title("Original Gene Expression", fontsize = title_fontsize)
    ax1.set_xlabel("Samples")
    
    ax2 = plt.subplot(1, 4, 2)
    sns.heatmap(x_recon.T, cmap="viridis", yticklabels=gene_names,
                cbar_kws={'label': 'Expression level'})
    cbar = ax2.collections[0].colorbar
    cbar.set_label('Expression Level', size=12, labelpad=10)
    cbar.ax.tick_params(labelsize=10)
    ax2.set_xticklabels([])
    ax2.set_title("Reconstructed Gene Expression (Zero-Inflated Beta VAE)", fontsize=title_fontsize)
    ax2.set_xlabel("Samples")

    # Plot 3 - Confusion Matrix
    correlation = pytorch_pearson_corr(expression_tensor, signature_tensor)
    correlation_probs = F.softmax(correlation, dim = 1)
    cell_labels = torch.argmax(correlation_probs, dim = 1)
    cell_labels_pred = torch.argmax(cell_probs, dim=1)
    cm = confusion_matrix(cell_labels.numpy(), cell_labels_pred.numpy())
    
    ax3 = plt.subplot(1, 4, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["C3", "C4", "EMT", "C10", "ciliated"],
                yticklabels=["C3", "C4", "EMT", "C10", "ciliated"])
    ax3.set_xlabel('Predicted Cell Type')
    ax3.set_ylabel('True Cell Type')
    ax3.set_title('Cell Type Classification - Confusion Matrix', fontsize = title_fontsize)

    # Plot 4 - Latent Space - with Signature Projection
    model.eval()
    with torch.no_grad():
        # Concatenate expression data with transposed signature matrix
        combined_data = torch.cat([expression_tensor, signature_tensor.T])
        
        # Forward pass through ZIB model
        _, _, _, z_S, _, cell_probs_S, _, _, _ = model(combined_data)

    z_S = z_S.numpy()
    if projection == "umap":
        umap = UMAP(n_components=2, random_state=42, min_dist = 0.05)
        latent_2d = umap.fit_transform(z_S)
    elif projection == "tsne":
        tsne = TSNE(n_components=2, random_state=73, perplexity=40)
        latent_2d = tsne.fit_transform(z_S)

    # Get cluster assignments
    assignments = torch.argmax(cell_probs_S, dim=1).numpy()

    # Map assignments to cell types
    mapping = {
        0: 'C3',
        1: 'C4',
        2: 'EMT',
        3: 'C10',
        4: 'ciliated'
    }

    # Convert numerical assignments to cell type names
    cell_types = np.array([mapping[a] for a in assignments])

    # Define colors for each cell type
    colors = {
        'EMT': '#FFD700',      # yellow
        'ciliated': '#9370DB', # medium purple
        'C4': '#FFA500',       # orange
        'C10': '#00CED1',      # dark turquoise
        'C3': '#8B2E8B'        # dark magenta
    }

    # Create a color array based on cell types
    point_colors = [colors[cell_type] for cell_type in cell_types]

    # Create figure and axes objects explicitly
    ax4 = plt.subplot(1, 4, 4)
    # Plot everything except the last 5 points (signature centroids) with assigned colors
    scatter1 = ax4.scatter(latent_2d[:-5, 0], latent_2d[:-5, 1], c=point_colors[:-5],
                            alpha=0.7, edgecolors='none')

    # Plot the last 5 points (signature centroids) in black with labels
    scatter2 = ax4.scatter(latent_2d[-5:, 0], latent_2d[-5:, 1], c='black',
                            alpha=0.8, edgecolors='white', s=100)

    # Add labels to the last 5 points
    signature_cols = ["C3", "C4", "EMT", "C10", "ciliated"]
    x_shift = latent_plot_kws.get('x_shift', 0.00)  # Default shift if not provided
    y_shift = latent_plot_kws.get('y_shift', -1.1)  # Default shift if not provided
    for i, label in enumerate(signature_cols):
        ax4.annotate(label, (latent_2d[-5+i, 0] + x_shift,
                             latent_2d[-5+i, 1] + y_shift),
                     fontsize=12)

    # Create legend
    legend_loc = latent_plot_kws.get('legend_loc', 'upper right')
    legend_fontsize = latent_plot_kws.get('legend_fontsize', 10)

    patches = [mpatches.Patch(color=colors[cell_type], label=cell_type) for cell_type in colors]
    ax4.legend(handles=patches, loc=legend_loc, fontsize=legend_fontsize)

    label_fontsize = latent_plot_kws.get('label_fontsize', 12)

    ax4.grid(alpha=0.3)
    ax4.set_title(f'Latent Space Visualization ({projection.upper()} Projection)', fontsize=title_fontsize)
    ax4.set_xlabel('Latent Dimension 1', fontsize=label_fontsize)
    ax4.set_ylabel('Latent Dimension 2', fontsize=label_fontsize    )
    plt.tight_layout()

    if return_projection:
        return latent_2d[:-5, :]


def compute_cell_proportions(model,
                             expression_tensor,
                             aggregated_malignant_df,
                             cell_type_mapping=None,
                             cell_types=("C3","C4","EMT","C10","ciliated"),
                             plotting=True,
                             export=False,
                             return_proportions=True,
                             per_fov=True,
                             figsize_patient=(6,8),
                             figsize_fov=(6,12),
                             copy_df=True,
                             use_model_predictions=True,
                             weight_by_cells=True):
    """
    Compute malignant subtype fractions per patient and per (patient,FOV).

    Parameters additions:
      use_model_predictions: if False, keep existing df['Assignment'].
      weight_by_cells: if True, sum No_cells instead of counting bins.
    """
    cell_types = list(cell_types)
    if cell_type_mapping is None:
        cell_type_mapping = {0:'C3',1:'C4',2:'EMT',3:'C10',4:'ciliated'}

    df = aggregated_malignant_df.copy() if copy_df else aggregated_malignant_df

    if use_model_predictions:
        model.eval()
        with torch.no_grad():
            _, _, _, _, _, cell_probs, _, _, _ = model(expression_tensor)
        assignments = torch.argmax(cell_probs, dim=1).cpu().numpy()
        df['Assignment'] = pd.Series(assignments, index=df.index).map(cell_type_mapping)

    df_use = df[df['Assignment'].isin(cell_types)]

    value_col = 'No_cells' if (weight_by_cells and 'No_cells' in df_use.columns) else None
    agg_func = (lambda g: g[value_col].sum()) if value_col else (lambda g: len(g))

    def _pivot(group_cols):
        counts = (df_use
                  .groupby(group_cols + ['Assignment'])
                  .apply(agg_func)
                  .unstack('Assignment', fill_value=0)
                  .reindex(columns=cell_types, fill_value=0))
        props = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100
        return counts, props

    counts_patient, props_patient = _pivot(['Patient_ID'])

    counts_patient_fov = props_patient_fov = None
    if per_fov:
        counts_patient_fov, props_patient_fov = _pivot(['Patient_ID','Frame'])

    # plotting
    if plotting and not props_patient.empty:
        plot_cols = props_patient.columns.tolist()
        plt.figure(figsize=figsize_patient)
        sns.heatmap(props_patient.loc[:, plot_cols], cmap="coolwarm",
                    yticklabels=props_patient.index,
                    cbar_kws={'format':'%.0f%%'})
        plt.xlabel("Cell Type"); plt.ylabel("Patient ID")
        plt.title("Malignant subtype proportions (per patient)")
        plt.tight_layout(); plt.show()

    if plotting and per_fov and props_patient_fov is not None and not props_patient_fov.empty:
        display_df = props_patient_fov.copy()
        display_df.index = [f"{p}|{f}" for p,f in display_df.index]
        plot_cols = props_patient_fov.columns.tolist()
        plt.figure(figsize=figsize_fov)
        sns.heatmap(display_df.loc[:, plot_cols], cmap="coolwarm",
                    yticklabels=display_df.index,
                    cbar_kws={'format':'%.0f%%'})
        plt.xlabel("Cell Type"); plt.ylabel("Patient|FOV")
        plt.title("Malignant subtype proportions (per patient-FOV)")
        plt.tight_layout(); plt.show()

    if export:
        os.makedirs("data_processed", exist_ok=True)
        with open("data_processed/aggregated_malignant_df_with_assignments.pkl","wb") as f:
            pkl.dump(df, f)

    if return_proportions:
        out = {
            'per_patient': props_patient,
            'counts_per_patient': counts_patient
        }
        if per_fov:
            out['per_patient_fov'] = props_patient_fov
            out['counts_per_patient_fov'] = counts_patient_fov
        return out
    

