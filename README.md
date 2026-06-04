# SMR-Net

## Learning Multi-Resolution Neuroanatomical Representations for Alzheimer's Disease Analysis through Cross-Resolution Fusion, Slice-Level Attention Aggregation, and Graph-Based Neuroanatomical Reasoning

---

## Overview

SMR-Net is a neuroanatomically guided deep learning framework designed for structural magnetic resonance imaging (sMRI)-based Alzheimer's disease (AD) analysis. The proposed framework addresses three key challenges in existing neuroimaging studies:

1. Poor utilization of complementary global and local neurodegenerative information.
2. Limited attention to diagnostically important MRI slices.
3. Lack of explicit modeling of neuroanatomical relationships among disease-sensitive brain regions.

To overcome these limitations, SMR-Net integrates:

* **Multi-Resolution Learning (MRL)** for capturing global and local structural patterns.
* **Cross-Resolution Attention Fusion (CRAF)** for adaptive multi-scale feature integration.
* **Slice-Level Attention Aggregation (SLAA)** for emphasizing diagnostically informative slices.
* **Graph-Based Neuroanatomical Reasoning (GNR)** for modeling interactions among disease-relevant brain regions.
* **Group Normalization (GN)** for stable optimization under limited batch-size settings.

The framework is evaluated using the ADNI dataset and externally validated on OASIS and AIBL cohorts.

---

## Architecture

The overall architecture of SMR-Net consists of the following stages:

```text
Structural MRI
      │
      ▼
Multi-Resolution Learning
(Low-Resolution Branch + High-Resolution Branch)
      │
      ▼
Cross-Resolution Attention Fusion
      │
      ▼
Slice-Level Attention Aggregation
      │
      ▼
Graph-Based Neuroanatomical Reasoning
      │
      ▼
Classification Layer
      │
      ▼
CN / MCI / AD
```

---

## Datasets

### ADNI

The Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset is used for model development and training.

Website:

https://adni.loni.usc.edu

### OASIS

The Open Access Series of Imaging Studies (OASIS) dataset is used for external validation.

Website:

https://www.oasis-brains.org

### AIBL

The Australian Imaging, Biomarkers and Lifestyle (AIBL) dataset is used for independent cross-cohort evaluation.

Website:

https://aibl.csiro.au

---

## Data Preparation

### Structural MRI Preprocessing

The following preprocessing pipeline is applied:

1. Skull stripping
2. Intensity normalization
3. Spatial registration to MNI space
4. Slice extraction
5. Image resizing
6. Z-score normalization

### Label Encoding

| Class | Label |
| ----- | ----- |
| CN    | 0     |
| MCI   | 1     |
| AD    | 2     |

---

## Repository Structure

```text
SMR-Net/
│
├── README.md
├── requirements.txt
├── train.py
├── test.py
├── model.py
├── dataset.py
├── utils.py
├── config.yaml
│
├── checkpoints/
│
├── data/
│   └── README.md
│
├── splits/
│   ├── adni_train.csv
│   ├── adni_val.csv
│   ├── oasis_test.csv
│   └── aibl_test.csv
│

│
└── results/


## Installation

Clone the repository:

```bash
git clone https://github.com/USERNAME/SMR-Net.git
cd SMR-Net
```

Create a virtual environment:

```bash
python -m venv smrnet_env
source smrnet_env/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

```text
torch
torchvision
numpy
pandas
scikit-learn
opencv-python
nibabel
matplotlib
seaborn
pyyaml
tqdm
```

---

## Dataset Format

CSV files should follow:

```csv
path,label
/path/to/subject1.nii.gz,0
/path/to/subject2.nii.gz,1
/path/to/subject3.nii.gz,2
```

---

## Training

Train SMR-Net using:

```bash
python train.py
```

Configuration settings can be modified in:

```text
config.yaml
```

Example:

```yaml
epochs: 100
batch_size: 4
learning_rate: 0.0001
weight_decay: 0.00001
num_classes: 3
image_size: 128
num_slices: 32
```

---

## Testing

Evaluate on OASIS:

```bash
python test.py --dataset oasis
```

Evaluate on AIBL:

```bash
python test.py --dataset aibl
```

---

## External Validation

The proposed framework is externally validated using:

* OASIS dataset
* AIBL dataset

without additional fine-tuning to evaluate cross-cohort generalization capability.



## Ablation Studies

The contribution of each module is evaluated using:

| Variant  | Components                                                     |
|----------|----------------------------------------------------------------|
| BaseNet  | Single-resolution baseline                                     |
| SMR-M    | Multi-Resolution Learning (MRL)                                |
| SMR-MF   | MRL + Cross-Resolution Attention Fusion (CRAF)                 |
| SMR-MFA  | MRL + CRAF + Slice-Level Attention Aggregation (SLAA)          |
| SMR-Net  | MRL + CRAF + SLAA + Graph-Based Neuroanatomical Reasoning (GNR)|


## Explainability Analysis

The repository includes:

* Slice-level attention visualization
* Cross-resolution fusion analysis
* Grad-CAM visualizations
* Neuroanatomical graph visualization
* Top brain-region importance ranking

Key disease-sensitive regions include:

* Hippocampus
* Entorhinal Cortex
* Precuneus
* Posterior Cingulate Cortex
* Parahippocampal Gyrus
* Temporal Lobe

---

## Statistical Analysis

The repository provides scripts for:

* Wilcoxon signed-rank test
* Paired t-test
* Confidence interval estimation
* Calibration analysis
* Expected Calibration Error (ECE)

---

## Reproducibility

To ensure full reproducibility, the repository includes:

* Training scripts
* Testing scripts
* Cross-validation splits
* Hyperparameter configurations
* Random seed settings
* Model checkpoints
* External validation protocols

---

## Results

Representative performance is reported on:

* ADNI
* OASIS
* AIBL

including classification performance, robustness analysis, calibration analysis, and neuroanatomical interpretability evaluation.

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{smrnet2026,
  title={Learning Multi-Resolution Neuroanatomical Representations for Alzheimer's Disease Analysis through Cross-Resolution Fusion, Slice-Level Attention Aggregation, and Graph-Based Neuroanatomical Reasoning},
  author={Kaur, Arshpreet and Kaur, Jagdeep},
  journal={Submitted},
  year={2026}
}


