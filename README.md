# XRD Phase Identification via Representation Learning and Similarity Retrieval

This repository provides a research implementation for automated analysis of **X-ray diffraction (XRD) patterns** using representation learning and similarity-based retrieval.  
The project focuses on building latent feature spaces that capture structural relationships between diffraction patterns and support phase identification in high-throughput materials experiments.

---

## Overview

X-ray diffraction is a fundamental characterization technique for determining crystalline structure and identifying material phases.  
However, interpretation of diffraction patterns becomes challenging in:

- combinatorial materials libraries  
- multi-component systems  
- noisy experimental measurements  
- peak shifts caused by strain or composition variation  
- large-scale high-throughput screening pipelines  

This repository implements a **machine learning workflow** to:

- learn compact latent representations of diffraction patterns  
- compare structural similarity between samples  
- perform retrieval-based phase analysis  
- analyze embedding geometry and clustering behavior  
- support exploratory materials informatics research  

---

## Key Features

- Deep representation learning for XRD signals  
- Latent embedding extraction and comparison  
- Similarity-based retrieval of structurally related samples  
- Internal geometry analysis of learned feature spaces  
- t-SNE visualization of embedding distributions  
- Composition-aware query experiments  
- Experimental evaluation notebooks for multiple datasets  

---

## Repository Structure
```bash
xrd_classification
│
├── README.md
├── requirements.txt
├── config.py
│
├── notebooks/
│ ├── train_cs.ipynb
│ ├── train_sg.ipynb
│ ├── Full_inference_SG_CS.ipynb
│ ├── Save_latent_cs.ipynb
│ ├── Save_latent_SG.ipynb
│ ├── IG_Data_CS_analysis.ipynb
│ ├── IG_Data_SG_analysis.ipynb
│ ├── tsne_testsets_cs_ruff.ipynb
│ ├── XRD_sum_cosin.ipynb
│ ├── query_composition.ipynb
│ └── clean_comp_compare.ipynb
│
├── models/
├── utils/
├── data_cleaning/
├── pipeline/
└── data/
```

---

## Methodology

### Representation Learning

Deep neural models are trained to encode diffraction patterns into low-dimensional latent vectors.  
These embeddings aim to preserve structural similarity while being robust to noise and peak variation.

### Similarity Retrieval

Distances in the latent space (e.g., cosine similarity) are used to:

- retrieve nearest structural neighbors  
- support phase candidate ranking  
- analyze relationships between materials  

### Internal Geometry Analysis

The repository includes experiments that study:

- clustering behavior of embeddings  
- separation between structural classes  
- influence of dataset composition  

### Visualization

Dimensionality-reduction techniques such as t-SNE are used to visualize:

- embedding distributions  
- class separability  
- retrieval neighborhoods  

---

## Installation

Clone the repository:

```bash
git clone https://github.com/lab-mids/xrd_classification.git
cd xrd_classification
```
pip install -r requirements.txt

Typical Workflow
1. Train Representation Models

Run one of the training notebooks:

train_cs.ipynb

train_sg.ipynb

These notebooks train models for different structural labeling schemes.

2. Extract Latent Embeddings

Generate feature representations:

Save_latent_cs.ipynb

Save_latent_SG.ipynb

Embeddings are stored for later retrieval and visualization.

3. Perform Retrieval and Inference

Run:

Full_inference_SG_CS.ipynb

query_composition.ipynb

These notebooks demonstrate similarity search and composition-aware querying.

4. Analyze Embedding Geometry

Use analysis notebooks:

IG_Data_CS_analysis.ipynb

IG_Data_SG_analysis.ipynb

tsne_testsets_cs_ruff.ipynb

These explore clustering behavior and latent-space structure.

5. Evaluate Similarity Metrics

Run:

XRD_sum_cosin.ipynb

clean_comp_compare.ipynb

These notebooks compare similarity strategies and retrieval performance.

Applications

automated XRD phase identification

similarity-driven materials discovery

analysis of combinatorial synthesis libraries

exploratory research in materials informatics

decision support for experimental characterization

Data

Large datasets and intermediate experiment outputs are not included in this repository.
Users should provide their own diffraction datasets.

Reproducibility

To reproduce experiments:

install dependencies

prepare dataset in expected format

run training notebooks

extract embeddings

execute retrieval and analysis notebooks

Citation

If you use this repository in your research, please cite the related publication:
Knowledge-Driven XRD Phase Identification via Representation Learning  
(ECML-PKDD submission)
