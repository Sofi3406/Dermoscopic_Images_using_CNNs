# Dermoscopic Image Cancer Detection

## Binary Classification with CNNs and Transfer Learning

This project explores automated classification of dermoscopic or histopathology images as **benign** or **malignant**. It compares a custom convolutional neural network with pretrained MobileNetV2 and EfficientNetB0 backbones.

> **Important:** This is an educational and portfolio project. It is not a medical diagnostic tool and must not be used to make clinical decisions.

---

## Project Overview

The notebook implements an end-to-end computer vision workflow:

```text
Dataset
  -> Image validation and duplicate removal
  -> Stratified train / validation / test split
  -> Resize, normalize, and augment training images
  -> Train a custom CNN baseline
  -> Fine-tune MobileNetV2 and EfficientNetB0
  -> Evaluate and compare model performance
  -> Predict the class of a new image
```

## Dataset

The notebook is designed for a folder-based Kaggle dataset such as [Skin Cancer: Malignant vs Benign](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign).

The expected dataset layout is:

```text
cnn_cancer_detector/
|-- benign/
|   |-- 0000.jpg
|   `-- ...
`-- malignant/
|   |-- 0000.jpg
|   `-- ...
```

The reference dataset contains approximately 3,297 images:

| Class | Images |
| --- | ---: |
| Benign | 1,800 |
| Malignant | 1,497 |

The class imbalance is handled with augmentation and balanced class weights.

## Notebook Workflow

1. Load the required Python libraries and configure reproducibility.
2. Build a dataframe from the `benign` and `malignant` image folders.
3. Check image counts, class balance, sample images, dimensions, and pixel intensity.
4. Remove unreadable files and exact duplicate images.
5. Create a stratified 70% / 15% / 15% train, validation, and test split.
6. Resize images to `224 x 224` and scale pixel values to `[0, 1]`.
7. Apply augmentation to the training set only.
8. Train and evaluate a custom CNN.
9. Train MobileNetV2 and EfficientNetB0 in two phases: frozen feature extractor, then fine-tuning.
10. Compare accuracy, precision, recall, F1-score, confusion matrices, and ROC-AUC.

## Models

| Model | Role |
| --- | --- |
| Custom CNN | Baseline trained from scratch with convolution, batch normalization, pooling, and dropout |
| MobileNetV2 | Lightweight ImageNet-pretrained backbone |
| EfficientNetB0 | Efficient ImageNet-pretrained backbone |

For this task, malignant-class **recall** and ROC-AUC deserve particular attention because false negatives are more serious than false positives.

## Getting Started

### Install dependencies

```bash
pip install numpy pandas matplotlib seaborn opencv-python tensorflow scikit-learn jupyter
```

### Prepare the data

1. Download a compatible dataset from Kaggle.
2. Extract it so the class folders are available beside the notebook, or update `DATA_DIR` in the first configuration cells.
3. Confirm that the folders are named exactly `benign` and `malignant`.

### Run the project

Open [cnn_cancer_detector.ipynb](cnn_cancer_detector.ipynb) in Jupyter Notebook or VS Code and run the cells from top to bottom. Training creates model checkpoints such as `best_custom_cnn.keras`, `best_mobilenet.keras`, and `best_effnet.keras`.

## Project File

| File | Description |
| --- | --- |
| [cnn_cancer_detector.ipynb](cnn_cancer_detector.ipynb) | Complete data preparation, training, evaluation, comparison, and inference workflow |

## Limitations and Future Work

- Kaggle data may not represent the diversity of clinical images, devices, or patient populations.
- Labels may contain noise and should be verified against reliable clinical ground truth.
- The binary target does not represent the full range of lesion subtypes and diagnoses.
- Future improvements could include Grad-CAM explanations, cross-validation, stronger augmentation, additional backbones, and a carefully validated demonstration API.
