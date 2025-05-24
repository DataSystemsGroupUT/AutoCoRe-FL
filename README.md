
# AutoCoRe-FL: Automatic Concept-based Rule Reasoning in Federated Learning

This repository contains the official PyTorch implementation and experimental setup for the paper: **"AutoCoRe-FL: Automatic Concept-based Rule Reasoning in Federated Learning"**.


**Authors:** Ahmed Wael, Radwa El Shawi

## Abstract
Federated learning (FL) is a decentralized paradigm for collaboratively training machine learning models while maintaining data privacy across clients. However, the inherent distribution of data and privacy constraints in FL pose significant challenges to achieving global interpretability and model transparency. To overcome these limitations, we propose AutoCoRe-FL, a framework for symbolic reasoning in FL that enables interpretable model explanations without requiring predefined or manually labeled concepts. In AutoCoRe-FL, each client automatically discovers high-level, semantically meaningful concepts from their local data. These concepts represent the abstract, human-understandable explanation units that capture the underlying structure of the data. Clients then represent their data samples as binary vectors of these concepts and generate symbolic rules based on them, which serve as interpretable explanations for model predictions. These rules are sent to the server, where an iterative symbolic aggregation process refines and aligns the rules into a coherent global model. Experimental results on benchmark datasets show that AutoCoRe-FL achieves competitive predictive performance while producing compact, accurate, and transparent symbolic explanations, significantly outperforming LR-XFL—the current state-of-the-art interpretable FL baseline that relies on predefined concept supervision.
![image](https://github.com/user-attachments/assets/077d22c5-ffeb-4330-bd2c-232e490a3e61)

## Features

*   **Automated Concept Discovery:** Self-supervised extraction of visual concepts from local client data using image segmentation (SAM2), representation learning (DINOv2), and federated clustering (Federated K-Means).
*   **Symbolic Reasoning:** Clients learn local symbolic rule-based models (FIGS - Fast Interpretable Greedy-Tree Sums) over the discovered concepts.
*   **Federated Rule Aggregation:** A central server iteratively aggregates and refines client rules into a global, interpretable model using a boosting-inspired approach, without accessing raw data.
*   **Privacy-Preserving Interpretability:** Achieves global model explanations in FL settings while respecting data privacy.
*   **No Manual Concept Labels Required:** Eliminates the dependency on predefined or manually annotated concepts, enhancing scalability.
*   **State-of-the-art Performance:** Out-perform state-of-the-art algorithms under different scenarios.

## Repository Structure

```
AutoCoRe-FL/
├── configs/                    # YAML configuration files for experiments
├── data_preparation/           # Scripts to download and pre-process datasets (ADE20K, SUNRGBD)
│   ├── prepare_ade20k_gt_concepts.py
│   ├── prepare_sunrgbd_subset_scenes.py
│   └── ...
├── features/                   # Directory for storing intermediate predicted concept features (e.g., from ResNet18)
├── trained_models/             # Directory for storing trained concept predictors or final models
├── autocore_fl/                # Core AutoCoRe-FL library code
│   ├── concepts/               # Concept discovery, detector training, vectorization
│   ├── embedding/              # Embedding model loaders (DINOv2)
│   ├── federated/              # Client, Server, Aggregation logic for FL
│   ├── segmentation/           # SAM model loader, segment processing
│   └── utils/                  # Utility functions (logging, config, etc.)
├── lens_framework_stubs/       # Stubs or interface code for LENS/LR-XFL components if adapted
├── scripts/                    # Main runnable experiment scripts
│   ├── run_autocore_cent_auto_ade20k.py    # Centralized AutoCoRe with automatic concept extraction (ADE20k)
│   ├── run_autocore_cent_auto_sun.py       # Centralized AutoCoRe with automatic concept extraction ( SUN )
│   ├── run_autocore_cent_resnet_ds.py # Centralized AutoCoRe on ResNet concepts
│   ├── run_lens_cent_auto_ds.py        # Centralized LENS with automatic concept extraction
│   ├── run_lens_cent_resnet_ds.py        # Centralized LENS with GT/predicted concepts
│   ├── run_autocore_fl_auto_ds.py    # Federated AutoCoRe with automatic concept extraction (Our proposed approach)
│   ├── run_autocore_fl_resnet_ds.py # Federated AutoCoRe on ResNet concepts
│   ├── run_lr_fl_auto_ds.py        # Federated LR-XFL with automatic concept extraction
│   ├── run_lr_fl_resnet_ds.py        # Federated LR-XFL with GT/predicted concepts
│   └── ... (other experiment scripts)
├── sulrm_jobs/                 # slurm jobs to run on HPC environment
├── results/                    # Output directory for experiment results, logs, plots
├── requirements.txt            # Pip requirements file

└── README.md                   # This file
```
Note that we have 2 scripts for each experiment. ds is short for dataset, either ade20k or sun
## Setup

### 1. Prerequisites

*   Python 3.8+
*   PyTorch (refer to `requirements.txt` for version, CUDA recommended)
*   Other dependencies listed in `requirements.txt` (e.g., scikit-learn, transformers, OpenCV, imodels, etc.)
*   Access to a machine with a GPU is highly recommended for efficient training and segmentation.

### 2. Clone Repository

```bash
git clone https://github.com/[Your GitHub Username]/AutoCoRe-FL.git
cd AutoCoRe-FL
```

### 3. Environment Setup

We recommend using Conda to manage dependencies:

```bash
conda env create -f environment.yml
conda activate autocore_fl_env
```
Alternatively, using pip:
```bash
pip install -r requirements.txt
```
You might also need to install PyTorch separately according to your CUDA version from [pytorch.org](https://pytorch.org/).

### 4. Download Pretrained Models

*   **SAM (Segment Anything Model v2):** Download the SAM2 checkpoint (e.g., `sam2_hiera_tiny.pt`) and place it in an accessible directory. Update the path in the relevant configuration files (e.g., `configs/sam_config.yaml` or directly in experiment scripts).
    *   Refer to the [SAM GitHub repository](https://github.com/facebookresearch/segment-anything-2) for model checkpoints.
*   **DINOv2:** The DINOv2 model will be downloaded automatically by the Hugging Face `transformers` library upon first use.

### 5. Dataset Preparation

Raw datasets (ADE20K, SUNRGBD) need to be downloaded and placed in a designated data directory.

*   **ADE20K:**
    *   Download from the [official ADE20K website](http://groups.csail.mit.edu/vision/datasets/ADE20K/).
    *   Place it such that you have `[your_data_root]/ade20k/ADEChallengeData2016/`.
    *   Update `USER_ADE20K_DATA_ROOT` in `autocore_fl/data_preparation/utils_ade20k_data.py` (or relevant config files) to point to `[your_data_root]/ade20k/`.
    *   Run the ground truth concept preparation script for ADE20K (if running baselines that require it):
        ```bash
        python data_preparation/prepare_ade20k_gt_concepts.py 
        ```
        *(Adjust script name and path if different)*

*   **SUNRGBD:**
    *   The script `data_preparation/prepare_sunrgbd_subset_scenes.py` will attempt to download SUNRGBD.zip if not found in the specified download root.
    *   Ensure `SUNRGBD_DOWNLOAD_ROOT` in that script points to your desired data directory.
    *   Run the script to prepare the 3-class subset (bathroom, bedroom, bookstore) in an ADE20K-like format:
        ```bash
        python data_preparation/prepare_sunrgbd_subset_scenes.py
        ```
        This will create a directory (e.g., `data/sun_final/`) with `images/`, `attributes.npy`, `scene_labels.npy`, etc.

**(Optional but Recommended) Caching Pre-processed Data for AutoCoRe-FL:**
For faster subsequent runs of the full AutoCoRe-FL pipeline (which includes SAM segmentation and DINOv2 embeddings), you can pre-generate and cache these for each client's data partition.
*   Run `data_preparation/generate_deterministic_cached_data.py`. This script will:
    1.  Partition the raw dataset (e.g., ADE20K images for chosen classes, or the prepared SUNRGBD 3-class images).
    2.  For each partition (client/server holdout):
        *   Perform SAM segmentation.
        *   Extract DINOv2 embeddings for segments.
        *   Save segment information, masks, and embeddings to disk.
    *   Update the configuration files used by the scripts to point to these cached data directories.

## Running Experiments

All main experiment scripts are located in the `scripts/` directory. Before running, ensure all paths in the corresponding YAML configuration files (in `configs/`) or at the top of the Python scripts are correctly set for your environment (dataset paths, model checkpoint paths, output directories).

### Example: Running Centralized AutoCoRe on SUNRGBD (using pre-cached segment/embedding data)

1.  **Prepare Cached Data:** Ensure you have run `data_preparation/generate_deterministic_cached_data.py` for the SUNRGBD 3-class setup, which should create cached segment infos and embeddings for different partitions.
2.  **Configure:** Modify `configs/centralized_autocore_sunrgbd_cached.yaml` to point to these cached data directories and set other hyperparameters.
3.  **Run:**
    ```bash
    python scripts/run_cent_autocore.py --config_path configs/autocore_cent_auto_sunrgbd_cached.yaml 
    ```
    *(Adjust script name and arguments as per your final structure)*

### Example: Running Full AutoCoRe-FL on ADE20K (using pre-cached segment/embedding data)

1.  **Prepare Cached Data:** Run `data_preparation/generate_deterministic_cached_data.py` for ADE20K for the desired number of clients and chosen classes.
2.  **Configure:** Modify `configs/federated_autocore_ade20k_cached.yaml` (example name).
3.  **Run:**
    ```bash
    python scripts/run_autocore_fl_main.py --config_path configs/federated_autocore_ade20k_cached.yaml
    ```

### Example: Running AutoCore on Pre-defined ResNet18 Concepts (Centralized)

This pipeline involves three stages:
1.  **Stage 0 (Data Prep for Concept Predictor):**
    *   For ADE20K: `python data_preparation/prepare_ade20k_gt_concepts.py`
    *   For SUNRGBD: `python data_preparation/prepare_sunrgbd_subset_scenes.py`
2.  **Stage 1 (Train Concept Predictor):**
    *   For ADE20K: `python scripts/1_train_concept_predictor_ade20k.py`
    *   For SUNRGBD: `python scripts/1_train_concept_predictor_sunrgbd.py`
    (Ensure model architecture and output paths are set correctly in these scripts).
3.  **Stage 2 (Generate Predicted Concept Features):**
    *   For ADE20K: `python scripts/2_generate_predicted_concept_features_ade20k.py`
    *   For SUNRGBD: `python scripts/2_generate_predicted_concept_features_sunrgbd.py`
    (These load the model from Stage 1 and save `.npy` feature files).
4.  **Stage 3 (Run FIGS with these features):**
    *   For ADE20K: `python scripts/run_figs_cent_resnet_concepts_ade20k.py`
    *   For SUNRGBD: `python scripts/run_figs_cent_resnet_concepts_sunrgbd.py`
    (These load the `.npy` files from Stage 2).

Refer to the comments and configurations within each script in the `scripts/` directory for more detailed instructions on running specific experiments (e.g., LR-XFL baselines, AutoCoRe-FL variants).

## Results

Experimental results, including model accuracy, rule accuracy, fidelity, complexity, and generated explanations, will be saved in the `results/` directory, organized by experiment run ID. Logs are also stored here.


## License

This project is licensed under the [Specify License - e.g., MIT License, Apache 2.0]. See the `LICENSE` file for details.
