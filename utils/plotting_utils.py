import anndata as ad
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import copy

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle



def plot_spatial_grid(adata_full,
                      adata_malignant,
                      grid_size = 250,
                      min_cells = 3,
                      fig_size = (9, 6),
                      save_path = None,
                      legend = False,
                      markersize = 10,
                      close = True
                      ):
    """
    Utility function to plot spatial grids for cell aggregation.
    """
    
    cell_colors = {
                    'TNK.cell':   '#4B0082',
                    'B.cell':     '#FF1493',
                    'Fibroblast': '#228B22',
                    'Endothelial':'#B8860B',
                    'Monocyte':   '#A52A2A',
                    'Mast.cell':  '#2B4162',
                    'Malignant':  '#808080'     
                }

    patients_to_plot = adata_full.obs["Patient"].unique()

    # plot each the spatial arrangement for each patient
    for current_patient in tqdm(patients_to_plot, desc = "Plotting patients"):

        patient_data_full = adata_full.obs[adata_full.obs["Patient"] == current_patient]
        patient_data_malignant = adata_malignant.obs[adata_malignant.obs["Patient"] == current_patient]


        # Get unique FOVs for the current patient
        fovs = patient_data_malignant["Frame"].str.split("_").str[2].unique()

        for fov in fovs:

            fov_data_all = patient_data_full[patient_data_full["Frame"].str.split("_").str[2] == fov]

            fov_data_malignant = patient_data_malignant[patient_data_malignant["Frame"].str.split("_").str[2] == fov]

            x_values = fov_data_malignant["Local_x"]
            y_values = fov_data_malignant["Local_y"]

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


            fig, ax = plt.subplots(figsize = fig_size)

            # Process each grid cell
            for x_bin_no in range(no_x_grids):
                for y_bin_no in range(no_y_grids):
                    
                    # plot the cells, with grid overlaid
                    current_x = x_start + x_bin_no * grid_size
                    current_y = y_start + y_bin_no * grid_size

                    # Only include malignant cells from this grid cell
                    x_mask = (fov_data_malignant["Local_x"] >= current_x) & (fov_data_malignant["Local_x"] < current_x + grid_size)
                    y_mask = (fov_data_malignant["Local_y"] >= current_y) & (fov_data_malignant["Local_y"] < current_y + grid_size)
                    grid_mask = x_mask & y_mask

                    # Check if enough cells in the grid and plot the grid
                    if grid_mask.sum() >= min_cells:
                        # ax.gca().add_patch(plt.Rectangle((current_x, current_y), grid_size, grid_size, fill=False, color='blue'))
                        ax.add_patch(
                                    Rectangle(
                                        (current_x, current_y),
                                        grid_size,
                                        grid_size,
                                        fill=False,
                                        edgecolor='blue',
                                        linewidth=1.5,
                                        linestyle='--')
                                    )
                        
                    ax.scatter(fov_data_malignant["Local_x"][grid_mask], fov_data_malignant["Local_y"][grid_mask], c='grey', s=markersize)

            # plot the other cell types
            fov_data_rest = fov_data_all[~fov_data_all["Cell_type"].isin(["Malignant"])]

            x_coords = fov_data_rest["Local_x"]
            y_coords = fov_data_rest["Local_y"]
            ax.scatter(x_coords, y_coords, c=fov_data_rest["Cell_type"].map(cell_colors), s=markersize, label=fov_data_rest["Cell_type"].unique())

            # Create legend
            if legend:
                legend_elements = [Line2D([0], [0], marker='o', color='w', label=cell_type, markerfacecolor=color, markersize=5) for cell_type, color in cell_colors.items()]
                
                ax.legend(handles=legend_elements, title="Cell Types",
                            loc="upper right", bbox_to_anchor=(1.2, 0.75), fontsize=10)


            plt.title(f"Patient: {current_patient}, FOV: {fov}")
            if save_path is not None:
                plt.savefig(f"{save_path}/patient_{current_patient}_fov_{fov}.png")

            if close:
                plt.close()


def plot_spatial_arrangement_with_assignment(adata_spatial, patient_fovs, legend = False, save_path = None):
    """
    Utility function to plot spatial arrangement of cells with their assigned types.
    """
    for (patient, fov), indices in tqdm(patient_fovs.indices.items(), desc="Processing patients"):

        # print(f"Analyzing {patient} - FOV {fov}")
        
        # Subset the data
        adata_subset = adata_spatial[indices].copy()

        color_mapping = {
        'Malignant.C3':       '#8B2E8B',
        'Malignant.C4':       '#FFA500',
        'Malignant.EMT':      '#FFD700',
        'Malignant.C10':      '#00CED1',
        'Malignant.ciliated': '#9370DB',
        'Malignant.Other':    '#808080', # These malignant cells are not further classified,
        'TNK.cell':           '#4B0082',
        'B.cell':             '#FF1493',
        'Fibroblast':         '#228B22',
        'Endothelial':        '#B8860B',
        'Monocyte':           '#A52A2A',
        'Mast.cell':          '#2B4162',
        }

        plt.figure(figsize = (12, 7))
        plt.scatter(adata_subset.obs["Local_x"], adata_subset.obs["Local_y"],
                c=adata_subset.obs["Cell_type"].map(color_mapping),
                s=10)

        plt.title(f"Patient {adata_subset.obs['Patient'].values[0]} - {adata_subset.obs['Frame'].values[0]}")

        if legend:
            plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=cell_type,
                                              markerfacecolor=color) for cell_type, color in color_mapping.items()],
                       title="Cell Types", bbox_to_anchor=(1.05, 1), loc='upper left')
            
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}/patient_{patient}_fov_{fov}.png")

        plt.close()

# TODO: Also add an utility for overlaying the spatial grids used for aggregation.
