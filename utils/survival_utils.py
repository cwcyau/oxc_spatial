import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def compute_oob_cindex(imputed_dataset, alphas, covariates, n_bootstrap=1000, l1_ratio=0.5, max_iter=1000, rng = np.random.RandomState(73)):
    # storage
    c_index_mat = np.full((n_bootstrap, len(alphas)), np.nan) # n_bootstrap, n_alphas

    n = imputed_dataset.shape[0] # 54 samples

    for b in range(n_bootstrap):
        
        print(f"Bootstrap {b+1}/{n_bootstrap}...", end="\r")

        bidx = rng.randint(0, n, size = n)                 # bootstrap indices
        oob = np.setdiff1d(np.arange(n), np.unique(bidx))  # out-of-bag indices

        if len(oob) == 0:
            continue  # no OOB data

        # fit scaler on bootstrap sample only
        Xb_raw = imputed_dataset.iloc[bidx][covariates].astype(float).to_numpy()
        Xoob_raw = imputed_dataset.iloc[oob][covariates].astype(float).to_numpy()
        scaler_b = StandardScaler().fit(Xb_raw)
        Xb = scaler_b.transform(Xb_raw)
        Xoob = scaler_b.transform(Xoob_raw)

        yb = Surv.from_arrays(event=imputed_dataset.iloc[bidx]['event'].astype(bool).to_numpy(),
                                time=imputed_dataset.iloc[bidx]['duration'].astype(float).to_numpy())
        yoob = Surv.from_arrays(event=imputed_dataset.iloc[oob]['event'].astype(bool).to_numpy(),
                                time=imputed_dataset.iloc[oob]['duration'].astype(float).to_numpy())


        # Check if there are events in the bootstrap sample and in the oob samples
        if yb['event'].sum() == 0 or yoob['event'].sum() == 0:
            # print("Warning: no events in bootstrap or OOB sample, skipping this bootstrap.")
            continue

        model = CoxnetSurvivalAnalysis(
            l1_ratio=l1_ratio,
            alphas=alphas,   # use the same grid
            max_iter=max_iter,
            tol=1e-7,
        )
        model.fit(Xb, yb)

        # print("model.coef_.shape:", model.coef_.shape)

        if model.coef_.shape[1] != len(alphas):
            # print("Warning: model did not converge, skipping this bootstrap.")
            continue

        # coef_ is shape (n_features, n_alphas)
        try:
            for j, coef_j in enumerate(model.coef_.T):
                risk = Xoob @ coef_j
                c_index = concordance_index_censored(
                    event_indicator=yoob["event"],
                    event_time=yoob["time"],
                    estimate=risk
                )[0]
                c_index_mat[b, j] = c_index
        except Exception as e:
            print("Error during C-index calculation:", e)
            print("Skipping this bootstrap.")
            continue
    
    return c_index_mat



def compute_best_alpha(imputed_datasets, covariates, l1_ratio=0.05, alpha_min_ratio=0.01, max_iter=2000, n_bootstrap=200, n_alpha=100, random_state=42, plotting=True):


    c_index_all = pd.DataFrame()
    rng = np.random.RandomState(random_state)

    for i, imputed_dataset in enumerate(imputed_datasets):

        print(f"Bootstrap evaluation on imputation {i+1}/{len(imputed_datasets)}...")

        X_raw = imputed_dataset[covariates].astype(float).to_numpy()
        scaler_full = StandardScaler().fit(X_raw)
        X_full = scaler_full.transform(X_raw)

        y_full = Surv.from_arrays(
            event=imputed_dataset['event'].astype(bool),
            time=imputed_dataset['duration'].astype(float)
        )

        base_model = CoxnetSurvivalAnalysis(
            l1_ratio = l1_ratio,
            alpha_min_ratio = alpha_min_ratio,
            max_iter = max_iter,
            n_alphas = n_alpha,
            tol = 1e-7,
        )

        base_model.fit(X_full, y_full)
        alphas = base_model.alphas_

        c_index_mat = compute_oob_cindex(imputed_dataset, alphas, covariates, n_bootstrap=n_bootstrap, l1_ratio=l1_ratio, max_iter=max_iter, rng=rng)

        results_oob = pd.DataFrame(c_index_mat, columns=alphas)

        # place the mean value across all bootstraps into a new dataframe, so we can report the robust C-index
        mean_cindex = results_oob.mean(axis=0)  # mean across bootstr
        # this is for the current imputed dataset
        c_index_all = pd.concat([c_index_all, mean_cindex.to_frame().T], ignore_index=True)

        # fix the c_index_all to 100 columns
        c_index_all = c_index_all.iloc[:, :100]

    # We can pick the alpha that gives the best (highest) mean C-index across all imputations
    alphas = c_index_all.columns
    mean_cindex_all = c_index_all.mean(axis=0)
    std_cindex_all = c_index_all.std(axis=0)
    
    # pick the best
    best_c_idx = mean_cindex_all.max()
    best_alpha = mean_cindex_all[mean_cindex_all == best_c_idx].index[0]

    print(f"\nBest alpha: {best_alpha:.4g} with mean C-index: {best_c_idx:.3f}")


    if plotting:
        plt.figure(figsize=(7.5,4.5))
        plt.plot(alphas, mean_cindex_all, color = "blue", label="Mean C-index")
        plt.fill_between(alphas, mean_cindex_all - std_cindex_all, mean_cindex_all + std_cindex_all, color='#6666FF', alpha=0.3, label="± 1 std")
        plt.axhline(0.5, color='black', linestyle='--', label="Random guess)")
        plt.axvline(best_alpha, color='red', linestyle='--', label=f"Best alpha: {best_alpha:.4g}")
        plt.xscale('log')
        plt.xlabel(r'$\alpha$') 
        plt.ylabel('Bootstrap C-index (OOB)')
        plt.title(r'Bootstrap C-index vs $\alpha$')
        plt.legend(loc = "upper left")
        plt.tight_layout()
        plt.show()

    # NOTE: This is done on the out of bag samples, so it is a more honest estimate of performance.

    return best_alpha, best_c_idx