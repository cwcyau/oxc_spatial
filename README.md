# Application of the Oxford Classic to Spatial Transcriptomics

This repository contains the code and analysis workflows used in: **“Understanding the spatial determinants of the Oxford Classic prognostic signature for high-grade serous ovarian cancer.”**

## Background

The [Oxford Classic](https://pubmed.ncbi.nlm.nih.gov/33446563/) (OxC) is a 52-gene classification system developed using [single-cell expression profiling of fallopian tube epithelial (FTE) cells](https://pubmed.ncbi.nlm.nih.gov/32049047/). The OxC classification system uses this gene set to identify five cellular programs (named C3, C4, EMT, C10 and ciliated):

- **C3** (differentiated)  
- **C4** (KRT17 subtype)  
- **C10** (cell cycle)  
- **Ciliated**  
- **EMT** (epithelial-to-mesenchymal transition)

Among these, the **EMT program** has been consistently associated with poor prognosis and immune modulation.

Traditionally, OxC is applied to bulk RNA expression data using deconvolution methods (e.g., CIBERSORT/CIBERSORTx) to estimate the contribution of each transcriptional program within a tumor sample.

Our main interest is the EMT program. 

## Extending OxC to Spatial Transcriptomics

Typically we use the OxC system by first performing non-single cell or “bulk” expression profiling of an ovarian cancer. This is expression profiling by averaging over many tumour cells. Given the expression data, we then deconvolve the bulk expression profile to identify how much of the tumour profile is contributed to by each of the five OxC cellular programs. This is typically done using a piece of software called [CIBERSORT](https://cibersortx.stanford.edu/) which is produced by Stanford.

However, the recent [spatial transcriptomics study of ovarian cancer](https://www.nature.com/articles/s41590-024-01943-5) allows us to examine the potential spatial distribution of EMT cells within ovarian cancers. The spatial transcriptomics study uses the 960 gene-panel [CosMx](https://nanostring.com/products/atomx-spatial-informatics-platform/cosmx-and-atomx-the-first-fully-integrated-single-cell-spatial-solution/?utm_source=google&utm_medium=cpc&utm_campaign=atomx&utm_source=google&utm_medium=atomx&utm_id=SpatialLeadership_Search_CombinedTier&utm_source=google&utm_medium=cpc&utm_campaign=17395569806&utm_agid=136545311039&utm_term=spatial%20multiomics&creative=601515004923&device=c&placement=&network=g&gad_source=1&gclid=Cj0KCQjwmOm3BhC8ARIsAOSbapUelLuq9PqYqQrR0FjddCoMafQ8RChMa5we-DMqWbRmNFVUcoi4ZT4aAv74EALw_wcB) technology. However, only 11 genes overlap between the 960 genes surveyed by CosMx and the OxC.

## This study

In this study, we:

1. Construct a reduced OxC signature based on gene overlap with the CosMx panel.  
2. Develop a signature-guided deep generative model (**Sig-ZIB-VAE**) tailored to spatial transcriptomics data.  
3. Infer spatial distributions of OxC programs at single-cell resolution.  
4. Quantify spatial organization using:
   - Neighborhood enrichment  
   - Co-localization quotient (CLQ)  
   - Ripley’s L clustering statistics  
   - Graph-based connectivity metrics  
5. Evaluate the prognostic relevance of spatial tumor microenvironment architecture using penalized Cox regression.

---

## Repository Contents

This repository includes:

- Implementation of the **Sig-ZIB-VAE** model  
- Preprocessing and spatial aggregation workflows  
- Construction of reduced OxC signature matrices  
- Spatial graph construction and metric computation  
- Survival modeling pipelines (elastic net Cox regression with multiple imputation)  
- Scripts to reproduce figures and analyses from the manuscript  

---

## Methodological Focus

This project addresses the broader challenge of:

> Transferring a molecular classification system developed on one gene panel to a new platform with partial feature overlap.

Specifically, we investigate:

- Biological signal retention under feature reduction  
- Stability of latent representations under weak supervision  
- Prognostic value of spatial organization beyond cellular composition  

---

## Data Availability

Spatial transcriptomics data were obtained from the CosMx discovery cohort described in:

Yeh et al., *Nature Immunology* (2024)

Access details are provided in the manuscript.

---

## Citation

If you use this code or build upon this framework, please cite:

> Stihi A, Yau C. *Understanding the spatial determinants of the Oxford Classic prognostic signature for high-grade serous ovarian cancer.* (Year)
