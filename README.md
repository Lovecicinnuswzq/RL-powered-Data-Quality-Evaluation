# RL-powered-Data-Quality-Evaluation
This repository contains the resources used in our study on reinforcement learning-powered data quality evaluation. It provides a list of training, validation, and test images used in our experiments, along with the training code. The original image files from MOCS and SODA datasets are not included here. Researchers interested in accessing these datasets should contact the respective dataset authors or data owners.
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