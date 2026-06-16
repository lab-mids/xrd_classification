# XRD Phase Identification via Representation Learning and Similarity Retrieval


This repository provides a research implementation for automated analysis of X-ray diffraction (XRD) patterns using representation learning and similarity-based retrieval.
The project focuses on constructing latent feature spaces that capture structural relationships between diffraction patterns and support phase identification and structural reasoning in high-throughput materials experiments.
---

## Overview

X-ray diffraction is a fundamental characterization technique for determining crystalline structure and identifying material phases.
However, interpretation of diffraction patterns becomes challenging in:

- combinatorial materials libraries

- multi-component systems

- noisy experimental measurements

- peak shifts caused by strain, defects, or composition variation

- large-scale high-throughput screening pipelines

- This repository implements a machine learning workflow that enables:

- learning compact latent representations of diffraction patterns

- measuring structural similarity between samples

- retrieval-based phase analysis

- explainability analysis using attribution methods

- exploratory materials informatics studies

---

## Key Features

- Deep representation learning models for XRD signals

- Latent embedding extraction for structural comparison

- Similarity-based retrieval of structurally related diffraction patterns

- Integrated Gradients analysis for interpretability of model predictions

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


### Internal Geometry Analysis

The repository includes experiments that study:

- clustering behavior of embeddings  
- separation between structural classes  
- influence of dataset composition  
---

## Installation

Clone the repository:

```bash
git clone https://github.com/lab-mids/xrd_classification.git
cd xrd_classification
pip install -r requirements.txt
```


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

4. Analyze Model Explanations and Embeddings

Use analysis notebooks:

IG_Data_CS_analysis.ipynb

IG_Data_SG_analysis.ipynb


These explore attribution patterns and latent-space structure.

5. Evaluate Similarity Metrics of Qwen and Magpie

Run:
clean_comp_compare.ipynb

These notebooks compare similarity strategies and retrieval performance.

6. Identify the most distant test samples in the embedding space

Run:
tsne_testsets_cs_ruff.ipynb


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
Dataset Description

This dataset provides the experimental data and model outputs used in the study:

“Knowledge-Driven XRD Phase Identification via Representation Learning”

The archive contains two main folders:

1. clean/

This folder contains the cleaned X-ray diffraction (XRD) datasets used as input for representation learning experiments.

The preprocessing pipeline includes:

noise reduction and signal normalization

peak alignment and interpolation

removal of corrupted or incomplete measurements

formatting into machine-learning-ready arrays

These datasets were used for:

training representation learning models

latent embedding extraction

similarity retrieval experiments

structural clustering and visualization

2. results/

This folder contains the experimental outputs generated by the trained models, including:

latent embedding representations

similarity retrieval results

model prediction outputs

evaluation metrics and analysis artifacts

intermediate experiment logs and processed data

These results support:

structural similarity analysis

phase candidate ranking

embedding space visualization

explainability studies using attribution methods

Usage

To reproduce the experiments:

Download and extract the dataset archive.

Place the folders inside the repository data directory:

xrd_classification/data/

Run the notebooks described in the repository README.

Anonymous Submission Note

This dataset is available in the following link.

[[[[https://zenodo.org/records/18968260?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijc2YTg5NDMyLWQ1OTAtNGI5ZC1iOGRhLTM0OWMxOGNlNWM1MSIsImRhdGEiOnt9LCJyYW5kb20iOiI0MWZlYTUyOWEwYWYyYWFiMzMzMGQxZTNhYzRlY2Y2ZCJ9.zQK7EwhX42h7EaQvLCmI6snz8oMpxBDheobhAn9_XR9e1j9foMAo6kE-RJM4r_XsvPnjMWWy3rLthZPn5YAHFw](https://zenodo.org/records/18968260?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjEyZWQ3ZjZkLWFmOTctNDUyZi05OWY3LTNmY2MxODNjODA0MCIsImRhdGEiOnt9LCJyYW5kb20iOiI1NjY0ODViNDMwMGVkMTNjZmEwNDdkZDJhZGIxNGQyMSJ9.MwqXxhwZh2a4Ck4B-px2j6ytXRC0R6xZWJzyK51FRj3AkfThZ3HzlFfgwncOqhOq1-5FD4L1CniNc41RAHEj-A)](https://zenodo.org/records/18968260?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImU1NWI3MDQ4LTA0ODAtNDRkNC1iMzg5LWIxMjNiZThkYTZhNCIsImRhdGEiOnt9LCJyYW5kb20iOiIzYzU2YzBkNDkzNWE0OGRiNTQyYWY4ZWRmZDQ0ZjA3NyJ9.uK93n_i46lw7b608WWCAV58s21lxNu4-tNc8q_c5oIw8Z0EIrlBIzRUB1kUQJYOaotGKgL5W1etadjr7aIx4ig)](https://zenodo.org/records/18968260?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImMxMDIzNTQ4LTEwODgtNDEzMS1iZDE4LTlhM2FkYmE4ZWFmYyIsImRhdGEiOnt9LCJyYW5kb20iOiIxOGZhZjE4YTQ3ZDVhMTVhNGE0YTg5MGM5ZWRkNTljYyJ9.o-30VzLath5XEnaphlBdFsDqjLahsf1v0oviHRuPGzhBbPJlkg2tZUtACal09Lc2B3vqHIYUTNoIZYotCoGV6Q)](https://zenodo.org/records/18968260?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjdiZjgyOGI0LWZkOWYtNGVhMC05OWQ4LTNmY2QxYWZkYTgxZCIsImRhdGEiOnt9LCJyYW5kb20iOiI5YTJiMDQwN2RiMTY1MTRmYTc1Y2I0ZmRjMGJjZTg3OCJ9.pnKSoLeG0VXcv6-Sv8WYouVAUvEA17QCa_Bmx2Lc3D6TmibHdYE-csnuVXBs6i1wVnx2mD5SVnK_7FCf8rSx1w)


# Run the App
Run from the project root by the following command
```bash
streamlit run app.py
```



## Citation

If you use this repository in your research, please cite the related publication:
Knowledge-Driven XRD Phase Identification via Representation Learning  
(ECML-PKDD submission)
## License

This project is licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).

You may copy, modify, and distribute this software in accordance with the terms of the license. A copy of the license is included in the LICENSE file in the root directory of this repository.

For more information, see the GNU Lesser General Public License v3.0:
https://www.gnu.org/licenses/lgpl-3.0.en.html



