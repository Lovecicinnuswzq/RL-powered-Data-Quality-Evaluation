# RL-powered Data Quality Evaluation for Construction Computer Vision

This repository contains the official implementation, experimental protocols, and reproducible resources for our study on reinforcement learning (RL)-powered data quality evaluation for object detection in construction.

It provides data split lists (training, validation, and test images), core training/curation code, and environment configurations. 

> **Note on Datasets:** The original raw image files from the MOCS and SODA benchmark datasets are not distributed in this repository due to licensing. Researchers interested in accessing the raw images should obtain them directly from the original dataset creators.

---

# Experimental Protocol & Environment Setup

To ensure strict reproducibility across different computing environments, all experimental protocols, deterministic settings, and hardware/software configurations are detailed below.

### 1. Hardware & Operating System
* **CPU:** AMD EPYC 7543P 32-Core Processor
* **GPU:** NVIDIA GeForce RTX 4090 (24 GB VRAM)
* **RAM:** 64 GB
* **OS:** Ubuntu 22.04 LTS (Linux 5.15.0 x86_64)

### 2. Software & Core Dependencies
* **CUDA Version:** 12.1
* **cuDNN Version:** 8.9.2
* **Python Version:** 3.10.12
* **PyTorch Version:** 2.0.0+
* **Torchvision Version:** 0.15.0+
* **Ultralytics (YOLO11):** 8.1.0+
* **Gymnasium (RL Environment):** 0.28.1+
* **Stable-Baselines3:** 2.0.0+

### 3. Dataset Splits & Instance-Level Sizing
All dataset filtering and scaling are strictly anchored to human worker instance counts:
* **MOCS Dataset:** Extracted worker-containing images. Sampled into Small (1,000 instances), Medium (5,000 instances), and Large (10,000 instances) subsets. Each subset is split into **Training (80%)** and **Validation (20%)**. An independent **Test Set (5,000 instances)** is strictly isolated for final evaluation.
* **SODA Dataset:** Follows the exact same pipeline and proportions as MOCS (Small, Medium, Large at 8:2 Train/Val split, plus a 5,000-instance Test Set).

### 4. Image Preprocessing & Data Augmentation
* **Image Input Resolution:** Resized to 640x640 pixels maintaining aspect ratio with letterboxing.
* **Feature Extraction Preprocessing:** Normalized using standard ImageNet mean (`[0.485, 0.456, 0.406]`) and std (`[0.229, 0.224, 0.225]`) for ResNet-50 feature encoding.
* **Augmentation Protocol:** Default YOLO11 online augmentations (mosaic, HSV color space jitter, translation, scaling, and horizontal flip) were maintained consistently across all detector training and retraining runs.

### 5. Deterministic Settings & Random Seeds
To systematically eliminate variance induced by algorithmic stochasticity, deterministic modes were enforced across PyTorch, CUDA backend (`torch.backends.cudnn.deterministic = True`), and NumPy. The full experimental pipeline (RL policy update -> scoring -> curation -> detector retraining) was independently repeated across three fixed random seeds:
```
RANDOM_SEEDS = [0, 42, 3407]
```
# Installation
Install the required dependencies:
```
pip install -r requirements.txt
```
# Usage Pipeline
### 1. Feature Extraction
Before training the RL-evaluator, extract image-level features using the pre-trained ResNet-50 backbone.
```
python feature_extractor.py \
    --img_folder /path/to/your/train/images \
```
### 2. RL-evaluator Training
```
python train.py \
    --train_img_dir /path/to/your/train/images \
    --train_lbl_dir /path/to/your/train/labels \
    --val_img_dir /path/to/your/val/images \
    --val_lbl_dir /path/to/your/val/labels \
    --save_dir ./checkpoints \
    --model_path yolo11n.pt \
    --num_episodes 1000
```