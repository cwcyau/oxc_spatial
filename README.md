# Application of the Oxford Classic to Spatial Transcriptomics

The [Oxford Classic](https://pubmed.ncbi.nlm.nih.gov/33446563/) (OxC) is a 52-gene classification system developed using [single-cell expression profiling of fallopian tube epithelial (FTE) cells](https://pubmed.ncbi.nlm.nih.gov/32049047/). The OxC classification system uses this gene set to identify five cellular programs (named C3, C4, EMT, C10 and ciliated). Our main interest is the EMT program. 

Typically we use the OxC system by first performing non-single cell or “bulk” expression profiling of an ovarian cancer. This is expression profiling by averaging over many tumour cells. Given the expression data, we then deconvolve the bulk expression profile to identify how much of the tumour profile is contributed to by each of the five OxC cellular programs. This is typically done using a piece of software called [CIBERSORT](https://cibersortx.stanford.edu/) which is produced by Stanford.

However, the recent [spatial transcriptomics study of ovarian cancer](https://www.nature.com/articles/s41590-024-01943-5) allows us to examine the potential spatial distribution of EMT cells within ovarian cancers. The spatial transcriptomics study uses the 960 gene-panel [CosMx](https://nanostring.com/products/atomx-spatial-informatics-platform/cosmx-and-atomx-the-first-fully-integrated-single-cell-spatial-solution/?utm_source=google&utm_medium=cpc&utm_campaign=atomx&utm_source=google&utm_medium=atomx&utm_id=SpatialLeadership_Search_CombinedTier&utm_source=google&utm_medium=cpc&utm_campaign=17395569806&utm_agid=136545311039&utm_term=spatial%20multiomics&creative=601515004923&device=c&placement=&network=g&gad_source=1&gclid=Cj0KCQjwmOm3BhC8ARIsAOSbapUelLuq9PqYqQrR0FjddCoMafQ8RChMa5we-DMqWbRmNFVUcoi4ZT4aAv74EALw_wcB) technology. However, only 11 genes overlap between the 960 genes surveyed by CosMx and the OxC.

**Question 1: Is it possible to apply the OxC classification system to the CosMx data using a reduced gene set of 11 genes?**

**Question 2: Would we benefit from returning to the FTE data to retrieve a new gene set and updated OxC system which has greater overlap with the CosMx gene panel?**

This applied problem leads to some generic machine learning/statistical research questions:

**Question 3: What is the loss of efficiency in transferring a classification system built on P features to a new data set where only Q (< P) features are present?** 

**Question 4: Are there existing or novel metrics that allow us to characterise the loss of efficiency? Or even indicate that it won’t be possible to transfer at all.**

A literature search for similar related challenges, e.g. feature subset selection, missing features, variable imputation.

The latter questions could be addressed first in simulation and using toy examples, e.g. build a digit classifier using MNIST, reduce feature set, apply reduced feature set to classify masked MNIST images. 
