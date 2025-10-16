import anndata as ad
import pandas as pd
import numpy as np
import squidpy as sq
from tqdm import tqdm

def analyse_spatial_metrics_multiscale(adata_subset, patient, fov):
    """
    Computes spatial metrics for a given patient and FOV at multiple radii.
    This returns:
        1. Centrality Scores
        2. Neighbourhood Enrichment Scores
        3. Co-localization Quotients
        4. Ripley's L statistics.
    """
    results = {
        'patient': patient,
        'FOV': fov,
        'n_cells': adata_subset.shape[0],
        'metrics_by_radius': {}
    }

    try:
        # Convert categorical columns to categorical dtype
        if 'Cell_type' in adata_subset.obs.columns:
            adata_subset.obs["Cell_type"] = adata_subset.obs["Cell_type"].astype("category")
        else:
            print(f"Warning: Cell_type column not found for {patient}-{fov}")
            return results

        cell_types = adata_subset.obs["Cell_type"].cat.categories.tolist()

        # First, we need to compute a connectivity matrix from spatial coordinates, i.e. compute the spatial neighbors
        # We can do this using a fixed radius or using a Delaunay triangulation - the second option being less subjective, but not as infrmative for cell-cell interactions.
        # radius = 175 here is equivalent to a radius of 30μm

        # Define radii in pixels (~30, ~100, ~300 μm)
        radii_px = {
            "30um": 175,
            "100um": 600,
            "300um": 1800
        }

        # Loop over scales
        for label, rad in radii_px.items():
            metrics = {}

            print(f"Building spatial graph for radii = {label}")

            # Build the spatial graph
            sq.gr.spatial_neighbors(
                    adata_subset,
                    radius = rad,
                    coord_type = "generic",
                    delaunay = False,
                    key_added = f"spatial_neighbors_{label}"
                )
            
            # Compute centrality scores
            print("   Computing centrality scores")
            sq.gr.centrality_scores(adata_subset,
                    cluster_key="Cell_type",
                    connectivity_key=f"spatial_neighbors_{label}",
                    score="degree_centrality",
                    show_progress_bar=True
                    )

            metrics["centrality"] = adata_subset.uns["Cell_type_centrality_scores"].to_dict()

            # Compute neighbourhood enrichment
            print("   Computing neighbourhood enrichment")
            sq.gr.nhood_enrichment(adata_subset,
                       cluster_key = "Cell_type",
                       connectivity_key=f"spatial_neighbors_{label}")
            
            enrichment = adata_subset.uns['Cell_type_nhood_enrichment']
            metrics['enrichment'] = {
                'zscore': pd.DataFrame(enrichment['zscore'], index=cell_types, columns=cell_types).to_dict(),
                'count': pd.DataFrame(enrichment['count'], index=cell_types, columns=cell_types).to_dict()
            }

            # Compute co-localization Quotient (CLQ) for all pairs
            print("   Computing co-localization Quotient (CLQ)")
            clq_features = {}
            for a in cell_types:
                for b in cell_types:
                    if a != b:

                        clq = compute_co_localization_quotient(
                                adata_subset,
                                a, b, 
                                cluster_key="Cell_type",
                                neighbors_key = f"spatial_neighbors_{label}"
                        )

                        clq_features[f"{a}_{b}"] = clq
            metrics['co_localization'] = clq_features
            results['metrics_by_radius'][label] = metrics

        # Ripley's statistics (only needs to run once, independent of graph)
        sq.gr.ripley(adata_subset,
                    cluster_key="Cell_type",
                    mode="L",
                    max_dist = 3000)

        if "Cell_type_ripley_L" in adata_subset.uns:
            ripley_data = adata_subset.uns["Cell_type_ripley_L"]
            l_stats = ripley_data["L_stat"]
            bins = ripley_data["bins"]

            ripley_features = {}
            for cell_type in cell_types:
                if cell_type in l_stats["Cell_type"].values:
                    values = l_stats[l_stats["Cell_type"] == cell_type]["stats"].to_numpy()
                    max_l = float(np.max(values))
                    auc = float(np.trapz(values, bins))
                    early_idx = len(bins) // 4
                    mid_idx = len(bins) // 2
                    if early_idx > 0 and mid_idx > early_idx:
                        early_slope = float((values[early_idx] - values[0]) / (bins[early_idx] - bins[0]))
                        mid_slope = float((values[mid_idx] - values[early_idx]) / (bins[mid_idx] - bins[early_idx]))
                        late_slope = float((values[-1] - values[mid_idx]) / (bins[-1] - bins[mid_idx]))
                    else:
                        early_slope = mid_slope = late_slope = 0.0
                    ripley_features[cell_type] = {
                        'max_clustering': max_l,
                        'auc_clustering': auc,
                        'early_slope': early_slope,
                        'mid_slope': mid_slope,
                        'late_slope': late_slope
                    }
            results['ripley'] = ripley_features

    except Exception as e:
        print(f"Error analyzing {patient}-{fov}: {str(e)}")
        results['error'] = str(e)

    return results


def compute_co_localization_quotient(adata, cell_type_a, cell_type_b,
                                                cluster_key="Cell_type",
                                                neighbors_key=None):
    """
    Vectorized computation of the co-localization quotient (CLQ) for two cell types.
    """
    # Get adjacency
    if neighbors_key:
        adj_matrix = adata.obsp[f'{neighbors_key}_connectivities']
    else:
        adj_matrix = adata.obsp['spatial_connectivities']

    mask_a = (adata.obs[cluster_key] == cell_type_a).values
    mask_b = (adata.obs[cluster_key] == cell_type_b).values

    N_a = mask_a.sum()
    N_b = mask_b.sum()
    N = adata.n_obs

    # Skip if either type is rare
    if N_a < 20 or N_b < 20:
        return np.nan

    # Total edges from A to B
    C_b_a = adj_matrix[mask_a][:, mask_b].sum()

    observed_ratio = C_b_a / N_a
    expected_ratio = N_b / (N - 1)

    return observed_ratio / expected_ratio if expected_ratio > 0 else np.nan