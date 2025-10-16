import anndata as ad
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import warnings
import copy

import matplotlib.pyplot as plt
import pickle as pkl
import matplotlib.cm as cm
from matplotlib.lines import Line2D


def create_anndata_object(coor, cell_types, patients, frames, tma, cells, count_data, tpm_data, genes, frames_metadata):
    # Create coordinate dataframe
    coor_df = pd.DataFrame({
        'Local_x': [item["x"] for item in coor],
        'Local_y': [item["y"] for item in coor],
        'Local_row': [item["_row"] for item in coor],
        'Cell_type': cell_types,
        'Patient': patients,
        'Frame': frames,
        'TMA': tma,
        'Cell_ID': cells
    })

    # Create frame metadata dataframe
    frames_metadata_df = pd.DataFrame({
        'Frame': [item["Frame"] for item in frames_metadata],
        'Frame_no_cells': [item["No_cells"] for item in frames_metadata],
        'Frame_x': [item["x"] for item in frames_metadata],
        'Frame_y': [item["y"] for item in frames_metadata],
        'Frame_size': [item["size"] for item in frames_metadata],
        'Frame_cell_dens': [item.get("cell.dens") for item in frames_metadata],
        'Frame_QC': [item["QC"] for item in frames_metadata]
    })

    # Convert expression data to numpy arrays
    count_matrix = np.array(count_data)
    tpm_matrix = np.array(tpm_data)

    # Check dimensions and adjust genes if necessary
    print(f"Expression matrix shape: {tpm_matrix.shape}")
    print(f"Total gene IDs available: {len(genes)}")

    # If there's a mismatch, take only the genes that match the expression data
    if tpm_matrix.shape[1] != len(genes):
        print(f"Dimension mismatch detected. Using first {tpm_matrix.shape[1]} genes from genes")
        genes = genes[:tpm_matrix.shape[1]]

    print(f"Data loaded successfully.")

    # Create AnnData object
    adata = ad.AnnData(
        X=tpm_matrix.astype(np.float32),
        obs=coor_df,
        var=pd.DataFrame(index=genes)
    )
    adata.layers['counts'] = count_matrix.astype(np.float32)
    adata.obs = adata.obs.merge(frames_metadata_df, on='Frame', how='left')

    # drop all cells for which the frame QC is False
    print("Dropping cells with Frame_QC == False")
    adata = adata[adata.obs["Frame_QC"] == True].copy()
    # Drop the Frame QC column
    adata.obs.drop(columns=["Frame_QC"], inplace=True)
    # Drop cells for that are low confidence
    print("Dropping low confidence cells.")
    adata = adata[~adata.obs["Cell_type"].str.endswith("_LC")].copy()
    # reset the index
    adata.obs = adata.obs.reset_index(drop=True)

    print(f"Cells after QC: {adata.shape[0]}")
    print(f"AnnData object created: {adata}")

    return adata


def remove_non_expressive_cells(adata):
    data = adata.X
    zero_count_indices = np.where(np.sum(data, axis=1) == 0)[0]
    adata = adata[~np.isin(np.arange(adata.shape[0]), zero_count_indices)].copy()
    return adata


def normalize_for_deconvolution(adata):
    adata_copy = copy.deepcopy(adata)
    expression_matrix = adata_copy.layers["counts"]
    total_counts = adata_copy.layers["counts"].sum(axis = 0)

    # normalize by diving by the total count to account for different expression leveles across all genes
    scaling_factors = 1 / (total_counts + 1e-10)
    normalized_counts = expression_matrix * scaling_factors

    # scale each cell's gene profile to [0, 1] range, like the signature matrix
    gene_max = np.max(normalized_counts, axis=1, keepdims = True)
    normalized_expr = normalized_counts / (gene_max + 1e-10)

    adata_copy.layers["normalized_counts"] = normalized_expr

    return adata_copy


def extract_xy(frame):
    """Extract X and Y coordinates from frame identifier."""
    parts = frame.split('_')
    x = int(parts[3][1:]) # Extract the X value
    y = int(parts[4][1:]) # Extract the Y value
    return x, y

def process_spatial_data(adata_malignant, grid_size=250, min_cells=3,
                              plotting = False, save_dir = None):

    aggregated_data_list = []

    patients_to_process = adata_malignant.obs["Patient"].unique()

    # process each patient
    for current_patient in tqdm(patients_to_process, desc = "Processing patients"):

        patient_data = adata_malignant.obs[adata_malignant.obs["Patient"] == current_patient]

        # Get unique FOVs for the current patient
        fov_names = patient_data["Frame"].str.split("_").str[2].unique()

        for fov in fov_names:

            fov_data = patient_data[patient_data["Frame"].str.split("_").str[2] == fov]

            x_values = fov_data["Local_x"]
            y_values = fov_data["Local_y"]

            # Calculate grid dimensions
            x_min, x_max = np.min(x_values), np.max(x_values)
            y_min, y_max = np.min(y_values), np.max(y_values)
            x_length = x_max - x_min
            y_length = y_max - y_min

            no_x_grids = int(x_length // grid_size)
            no_y_grids = int(y_length // grid_size)
            
            # Center the grid
            x_start = x_min + (x_length - no_x_grids * grid_size) / 2
            y_start = y_min + (y_length - no_y_grids * grid_size) / 2

            if plotting:
                plt.figure(figsize=(9, 6))

            # Process each grid cell
            for x_bin_no in range(no_x_grids):
                for y_bin_no in range(no_y_grids):

                    # plot the cells, with grid overlaid
                    current_x = x_start + x_bin_no * grid_size
                    current_y = y_start + y_bin_no * grid_size

                    # Only include malignant cells from this grid cell
                    x_mask = (fov_data["Local_x"] >= current_x) & (fov_data["Local_x"] < current_x + grid_size)
                    y_mask = (fov_data["Local_y"] >= current_y) & (fov_data["Local_y"] < current_y + grid_size)
                    grid_mask = x_mask & y_mask

                    # plot the grids
                    if plotting: 
                        plt.gca().add_patch(plt.Rectangle((current_x, current_y), grid_size, grid_size, fill=False, color='blue'))
                        plt.text(current_x + grid_size / 3, current_y + grid_size / 2, f"{x_bin_no}_{y_bin_no}", fontsize=8)

                    # Check if there are enough cells in this grid
                    if grid_mask.sum() >= min_cells:
                        
                        # plot the aggregated cells
                        if plotting:
                            plt.plot(fov_data["Local_x"][grid_mask], fov_data["Local_y"][grid_mask], 'o', color='red')

                        # Process the grid cell
                        grid_data = fov_data[grid_mask]
                        
                        indices = grid_data.index
                        
                        # Aggregate cell expression data for the cells belonging to this grid cell
                        gene_counts_aggregated_normalized = adata_malignant[indices].layers["normalized_counts"].sum(axis = 0)
                        gene_counts_aggregated_raw = adata_malignant[indices].layers["counts"].sum(axis = 0)

                        # Create row dictionary for this grid cell
                        aggregated_data = {
                            "Patient_ID": current_patient,
                            "Frame": fov,
                            "Grid_X_Y": f"{x_bin_no}_{y_bin_no}",
                            "No_cells": len(grid_data),
                            "Cell_names": ','.join(grid_data["Cell_ID"].tolist())
                        }

                        # Add normalized gene expressions and put the column titles adata_malignant.var_names
                        for gene, expression in zip(adata_malignant.var_names, gene_counts_aggregated_normalized):
                            aggregated_data[gene] = expression
                        # Add raw gene expressions
                        for gene, expression in zip(adata_malignant.var_names, gene_counts_aggregated_raw):
                            aggregated_data[f"{gene}_raw"] = expression

                        # Append to results list
                        aggregated_data_list.append(aggregated_data)

                    else:
                        
                        # plot the unused cells
                        if plotting:
                            plt.plot(fov_data["Local_x"][grid_mask], fov_data["Local_y"][grid_mask], 'o', color='grey')
            if plotting:
                plt.title(f"Patient: {current_patient}, FOV: {fov}")
                if save_dir is not None:
                    plt.savefig(f"{save_dir}/{current_patient}_{fov}_{x_bin_no}_{y_bin_no}.png")
                else:
                    raise ValueError("Save directory is not specified.")
                plt.close()

    aggregated_malignant_df = pd.DataFrame(aggregated_data_list)
    return aggregated_malignant_df.reset_index(drop=True)
    

def pytorch_pearson_corr(data: torch.Tensor, signature_matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute the Pearson correlation between the data and the signature matrix.
    
    Args:
        data: Input data tensor [batch_size, n_features]
        signature_matrix: Signature matrix [n_features, n_signatures]
        
    Returns:
        Correlation matrix [batch_size, n_signatures]
    """
    # Center the data
    data_centered = data - data.mean(dim=1, keepdim=True)
    signature_centered = signature_matrix - signature_matrix.mean(dim=0, keepdim=True)
    
    # Compute dot products
    numerator = torch.matmul(data_centered, signature_centered)
    
    # Compute norms
    data_norm = torch.norm(data_centered, dim=1, keepdim=True)
    signature_norm = torch.norm(signature_centered, dim=0, keepdim=True)
    
    # Avoid division by zero
    denominator = torch.clamp(data_norm * signature_norm, min=1e-8)
    
    # Compute correlation
    correlation = numerator / denominator
    
    return correlation
    

