# SRPU-Model: A Self-Evolving Framework for SEM Image Segmentation

## 1\. Project Overview

`SRPU-Model` is an advanced project for Scanning Electron Microscope (SEM) image segmentation. It aims to build a sustainable, self-optimizing intelligent segmentation system by combining Supervised Learning, Self-Supervised Learning, Semi-Supervised Learning, and Active Learning strategies. The core philosophy of this project is to leverage a limited amount of labeled data along with a vast pool of unlabeled data to progressively enhance the model's performance, ultimately creating a "living" system capable of continuous learning and evolution from new data.

The project adopts a modular architecture that clearly separates data processing, model definitions, task execution, and configuration management, making it easy to extend and maintain.

## 2\. Project Goals

The long-term goal of this project is to construct a closed-loop, self-evolving segmentation system, achieved through the following phases:

**Preparation Phase:** Establish a solid project foundation, including environment setup, data preparation, and a reliable supervised learning baseline model.
  **Phase 1: Learning General Visual Knowledge.** Utilize Self-Supervised Learning (SSL) to enable the model to learn underlying features from a large amount of unlabeled SEM images. This is augmented by leveraging external pre-trained models (e.g., ResNet, EfficientNet) for enhanced capabilities.
  **Phase 2: Efficient Utilization of Data and Human Resources.** Employ Semi-Supervised Learning (pseudo-labeling) to extract knowledge from unlabeled data. This is combined with Active Learning strategies to intelligently select the most valuable samples for manual annotation, thereby maximizing annotation efficiency.
  **Phase 3: Automation and Continuous Evolution.** Explore the use of generative models to create novel training data to fill knowledge gaps. The final objective is to build an automated MLOps pipeline that can handle monitoring, feedback, retraining, and deployment automatically.

## 3\. Project Structure

The project follows a clear, modular design with the following directory structure:

```
SRPU-Model/
├── configs/                # Stores all configuration files
│   ├── finetune/           # Configs for fine-tuning tasks
│   ├── selfup/             # Configs for self-supervised learning tasks
│   └── base.py             # Base shared configurations
├── data/                   # Stores datasets
│   ├── raw/                # All original SEM images
│   └── labeled/            # Manually labeled images and masks
├── datasets/               # PyTorch Dataset definition scripts
├── json/                   # JSON index files for datasets
├── logs/                   # Training logs (e.g., for TensorBoard)
├── models/                 # Model definitions
│   ├── checkpoints/        # Stores trained model weights
│   ├── decoders/           # Decoder modules (U-Net, DeepLabV3+)
│   └── encoders/           # Encoder modules (U-Net, ResNet, etc.)
├── tasks/                  # Execution scripts for various learning tasks
├── utils/                  # Utility scripts and helper functions
├── environment.yml         # Conda environment dependency file
└── README.md               # Project documentation
```

## 4\. Core Features and Implementation

### 4.1 Data Processing Workflow

1.  **Dataset Generation (`utils/json_generator.py`)**:

      * Automatically scans the `data/raw` and `data/labeled` directories.
      * Parses sequence and frame numbers from filenames (e.g., `4.2V-001.png`).
      * Constructs 3D stacked samples (`input_depth`) for images with temporal relationships.
      * Generates `master_labeled_dataset.json` and `master_unlabeled_dataset.json` to index samples with and without masks, respectively.
      * Supports automatic splitting of the labeled dataset into training, validation, and test sets.

2.  **Data Loading (`datasets/sem_datasets.py`)**:

      * The `SemSegmentationDataset` class loads images and masks based on the JSON files.
      * Supports the conversion from RGB color masks to class indices based on the `MAPPING` in `configs/base.py`.
      * Integrates the `albumentations` library for powerful data augmentation.

3.  **Data Augmentation & Normalization (`utils/augmentation.py`)**:

      * Provides configurable augmentation strategies for training and validation sets.
      * The `dataset_statistics_calculator.py` script can automatically compute the mean and standard deviation of the training set for normalization.

### 4.2 Model Architecture

  * **Encoders (`models/encoders/`)**:

      * `unet_encoder.py`: A classic U-Net encoder structure.
      * `resnet_encoder.py`: Supports ResNet34 and ResNet50 as encoders, with the ability to load ImageNet pre-trained weights.
      * `efficientnet_encoder.py`: Supports EfficientNet as an encoder.
      * `dinov2_encoder.py`: Supports DINOv2 as an encoder.

  * **Decoders (`models/decoders/`)**:

      * `unet_decoder.py`: A classic U-Net decoder, compatible with skip connections from the encoder.
      * `deeplab_decoder.py`: A DeepLabV3+ decoder, featuring an Atrous Spatial Pyramid Pooling (ASPP) module.

### 4.3 Training Tasks

  * **Supervised Learning (`tasks/train_task.py`)**:
      * Implements a complete supervised training pipeline.
      * Supports various strategies, including training from scratch, fine-tuning with a frozen encoder, and fine-tuning with differential learning rates.
      * Integrates with TensorBoard for visualizing the training process.
      * Automatically saves the best model and the latest checkpoint, enabling resuming from a checkpoint.

## 5\. Environment Setup

You can set up your Conda environment using the `environment.yml` file.

```bash
# 1. Create the Conda environment from the environment.yml file
conda env create -f environment.yml

# 2. Activate the newly created environment
conda activate SRPU-Model
```

This environment includes all necessary dependencies to run the project, such as PyTorch, TorchVision, and OpenCV.

## 6\. Usage Guide

### Step 1: Prepare Your Data

1.  Place all your original SEM images (e.g., `.png`, `.tif`) into the `data/raw/` directory.
2.  Place your labeled mask images (PNG format is recommended) into the `data/labeled/` directory. **Please ensure that the filename of each mask matches the corresponding original image's filename (excluding the extension).**

### Step 2: Generate Dataset Indexes

Run the `json_generator.py` script to create and split your datasets.

```bash
# It is recommended to use the 'generate_all' mode, which automates all steps.
# This will scan the data, create master_labeled/unlabeled.json, and then split the labeled set.
python utils/json_generator.py --mode generate_all

# You can also execute the steps individually:
# python utils/json_generator.py --mode generate_labeled_unlabeled
# python utils/json_generator.py --mode split_labeled --input_json master_labeled_dataset.json
```

After execution, files like `master_labeled_dataset_train.json` and `master_labeled_dataset_val.json` will be generated in the `json/` directory.

### Step 3: Calculate Dataset Statistics (Optional but Recommended)

For better model performance, it is recommended to calculate the mean and standard deviation of your training set for normalization.

```bash
python utils/dataset_statistics_calculator.py
```

This script will read `master_labeled_dataset_train.json`, compute the statistics, and save them to `json/dataset_stats.json`.

### Step 4: Configure and Start Training

1.  Open the `configs/finetune/train_config.py` file.
2.  Modify the configuration according to your needs, for example:
      * `TASK_NAME`: Give a unique name to your training session.
      * `TRAINING_MODE`: Choose a training mode (for a first run, `finetune_differential` with a pre-trained weights path is recommended).
      * Hyperparameters like `BATCH_SIZE`, `NUM_EPOCHS`, `BASE_LEARNING_RATE`.
3.  Run the training task script.

<!-- end list -->

```bash
python tasks/train_task.py
```

During training, logs will be saved in the `logs/` directory, and model weights will be saved in `models/checkpoints/`. You can monitor the process using TensorBoard:

```bash
tensorboard --logdir logs
```

## 7\. Next Steps

Based on your project plan, the following features can be implemented progressively:

  **Self-Supervised Pre-training (`tasks/ssl_pretrain_task.py`)**: Pre-train the encoder on a large volume of unlabeled data**Semi-Supervised & Active Learning**: Develop pseudo-labeling and intelligent sampling strategies.
  **Automation Pipeline (`pipeline.py`)**: Orchestrate the entire workflow to achieve a closed-loop, automated system.
