# PH2 Lesion Segmentation Using U-Net

A deep learning project for **binary skin lesion segmentation** from dermoscopic images using the **PH2 dataset** and a **U-Net** architecture.

The objective is to automatically identify and segment the lesion region from a dermoscopic skin image by assigning every pixel to either the **lesion** or **background** class.

---

## Overview

Skin lesion segmentation is an important computer vision task in medical image analysis. Accurate segmentation of the lesion region can help isolate the diseased area from surrounding healthy skin.

This project implements a U-Net based binary segmentation pipeline on the PH2 dermoscopic dataset.

The project compares two segmentation approaches:

1. **Baseline:** U-Net with Cross-Entropy Loss
2. **Proposed:** U-Net with a Compound Cross-Entropy + Soft Dice Loss

The models are trained and evaluated using the same dataset split and architecture to provide a fair comparison.

---

## Problem Statement

Given a dermoscopic skin image, the goal is to generate a binary segmentation mask identifying the lesion region.

For every pixel:

- `0` → Background
- `1` → Lesion

The predicted segmentation mask has the same spatial dimensions as the input image.

The project evaluates segmentation performance using:

- Precision
- Recall
- F1-score
- Intersection over Union (IoU)

---

## Dataset

The project uses the **PH2 Dermoscopic Image Dataset**, which contains **200 dermoscopic images** with corresponding lesion segmentation masks.

Each case is organized in the following structure:

```text
IMDxxx/
├── IMDxxx_Dermoscopic_Image/
│   └── IMDxxx.bmp
│
├── IMDxxx_lesion/
│   └── IMDxxx_lesion.bmp
│
└── IMDxxx_roi/
    └── ROI annotation files
```

The `Dermoscopic_Image` file is used as the input image, while the dedicated `lesion` file is used as the ground-truth segmentation mask.

ROI annotation files are not used as the segmentation ground truth.

### Dataset Statistics

| Property | Value |
|---|---:|
| Total cases | 200 |
| Training images | 140 |
| Validation images | 30 |
| Test images | 30 |
| Image size | 256 × 256 |
| Task | Binary segmentation |
| Random seed | 42 |

The raw PH2 dataset is not included in this repository.

---

## Data Preprocessing

The dataset preparation pipeline performs the following steps:

### 1. Image-Mask Pairing

Each dermoscopic image is automatically paired with its corresponding lesion mask using the case ID.

For example:

```text
IMD002.bmp
        ↓
IMD002_lesion.bmp
```

A dataset manifest is generated containing the image and mask paths.

### 2. Mask Verification

The lesion masks are checked to ensure that they contain valid binary segmentation labels.

The original PH2 masks contain:

```text
0   → Background
255 → Lesion
```

### 3. Train-Validation-Test Split

The 200 samples are divided into:

```text
Training      = 140 images
Validation    = 30 images
Testing       = 30 images
```

A fixed random seed of `42` is used to ensure reproducibility.

### 4. Image Resizing

All images are resized to:

```text
256 × 256
```

Bilinear interpolation is used for RGB images.

### 5. Mask Resizing

Segmentation masks are resized using **nearest-neighbor interpolation** so that the binary class labels are preserved.

After preprocessing, masks contain only:

```text
0
255
```

---

## Model Architecture

### U-Net

The project uses the **U-Net** architecture for semantic segmentation.

U-Net follows an encoder-decoder structure with skip connections.

```text
                 Input Image
                      │
                      ▼
              ┌─────────────┐
              │   Encoder   │
              └─────────────┘
                      │
                      ▼
              ┌─────────────┐
              │ Bottleneck  │
              └─────────────┘
                      │
                      ▼
              ┌─────────────┐
              │   Decoder   │
              └─────────────┘
                      │
              Skip Connections
                      │
                      ▼
             Segmentation Mask
```

### Encoder

The encoder progressively extracts high-level features while reducing the spatial resolution of the image.

### Bottleneck

The bottleneck contains the deepest feature representation of the input image.

### Decoder

The decoder progressively reconstructs the spatial resolution to produce a pixel-level segmentation map.

### Skip Connections

Skip connections transfer spatial information from the encoder to the corresponding decoder layers.

This helps the network preserve lesion boundaries and fine spatial details.

---

# Baseline Method

The baseline model uses:

```text
U-Net + Cross-Entropy Loss
```

Cross-Entropy Loss treats segmentation as a pixel-wise classification problem.

For binary segmentation, each pixel is classified as either:

```text
Background
or
Lesion
```

The baseline provides a reference against which the compound-loss model can be compared.

---

# Proposed Method

The proposed approach uses a compound loss combining:

```text
Cross-Entropy Loss
+
Soft Dice Loss
```

The total loss is:

\[
L_{total} = L_{CE} + \lambda L_{Dice}
\]

where:

- \(L_{CE}\) is the Cross-Entropy Loss
- \(L_{Dice}\) is the Soft Dice Loss
- \(\lambda\) controls the contribution of the Dice term

The Soft Dice Loss is defined as:

\[
L_{Dice}
=
1 -
\frac{
2\sum_i p_i y_i + \epsilon
}{
\sum_i p_i + \sum_i y_i + \epsilon
}
\]

where:

- \(p_i\) is the predicted probability for pixel \(i\)
- \(y_i\) is the ground-truth label
- \(\epsilon\) is a small smoothing constant

### Motivation

Cross-Entropy focuses on pixel-level classification, while Dice Loss directly encourages better overlap between the predicted lesion region and the ground-truth lesion region.

Combining the two objectives provides a loss function that considers both:

- Pixel-wise classification
- Region-level overlap

This is particularly useful for segmentation tasks where the foreground lesion occupies a different proportion of the image than the background.

---

# Evaluation Metrics

The segmentation models are evaluated on the unseen test set using four metrics.

## Precision

Precision measures how many pixels predicted as lesion are actually lesion pixels.

\[
Precision = \frac{TP}{TP + FP}
\]

where:

- TP = True Positives
- FP = False Positives

---

## Recall

Recall measures how many actual lesion pixels are correctly detected.

\[
Recall = \frac{TP}{TP + FN}
\]

where:

- TP = True Positives
- FN = False Negatives

---

## F1-Score

F1-score is the harmonic mean of precision and recall.

\[
F1 =
2 \times
\frac{Precision \times Recall}
{Precision + Recall}
\]

---

## Intersection over Union

IoU measures the overlap between the predicted lesion region and the ground-truth lesion region.

\[
IoU =
\frac{TP}
{TP + FP + FN}
\]

Higher IoU indicates better segmentation overlap.

---

# Experimental Comparison

Both models use the same:

- Dataset
- Train/validation/test split
- Input resolution
- U-Net architecture
- Training procedure
- Evaluation dataset

The main difference is the loss function.

| Model | Architecture | Loss |
|---|---|---|
| Baseline | U-Net | Cross-Entropy |
| Proposed | U-Net | Cross-Entropy + Soft Dice |

This setup allows the effect of the compound loss to be studied independently of architectural changes.

---

# Ablation Study

An ablation study is used to analyze the contribution of the Dice component.

The baseline configuration is:

\[
L = L_{CE}
\]

The proposed configuration is:

\[
L = L_{CE} + \lambda L_{Dice}
\]

Comparing these configurations helps determine the contribution of the overlap-based Dice component.

---

# Visualization

The project includes visualization of:

- Original dermoscopic images
- Ground-truth lesion masks
- Predicted segmentation masks
- Image and mask overlays
- Training and validation loss curves
- Metric comparisons between models

Example visualization workflow:

```text
Original Image
      │
      ├──────────────┐
      │              │
      ▼              ▼
Ground Truth      U-Net
   Mask           Prediction
      │              │
      └──────┬───────┘
             ▼
       Visual Comparison
```

---

# Project Structure

```text
LesionSegmentation/
│
├── dataset/
│   ├── raw/
│   │   └── PH2 Dataset images/
│   │
│   ├── manifest.csv
│   │
│   └── processed/
│       ├── train/
│       │   ├── images/
│       │   └── masks/
│       │
│       ├── val/
│       │   ├── images/
│       │   └── masks/
│       │
│       ├── test/
│       │   ├── images/
│       │   └── masks/
│       │
│       └── split.csv
│
├── models/
│
├── notebooks/
│
├── results/
│   ├── graphs/
│   ├── metrics/
│   └── predictions/
│
├── src/
│   ├── check_setup.py
│   ├── check_ph2_structure.py
│   ├── build_manifest.py
│   ├── verify_pairs.py
│   ├── prepare_dataset.py
│   └── verify_preprocessing.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# Reproducibility

The dataset split uses:

```text
Random Seed = 42
```

The same train, validation, and test partitions are used for both the baseline and proposed models.

This ensures that model comparisons are performed under the same experimental conditions.

---

# Results

The final experiments compare the baseline and proposed models using:

| Metric | U-Net + CE | U-Net + CE + Dice |
|---|---:|---:|
| Precision | — | — |
| Recall | — | — |
| F1-score | — | — |
| IoU | — | — |

The final values are obtained from evaluation on the held-out test set.

---

# Requirements

The project is implemented in Python using PyTorch and common machine learning and image-processing libraries.

Main dependencies include:

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Pillow
- OpenCV
- Matplotlib
- Scikit-learn
- tqdm

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

# Reproducing the Dataset Preparation

After placing the PH2 dataset in:

```text
dataset/raw/PH2 Dataset images/
```

the dataset preparation pipeline can be executed using:

```bash
python src/build_manifest.py
```

Verify image-mask pairs:

```bash
python src/verify_pairs.py
```

Prepare the train/validation/test datasets:

```bash
python src/prepare_dataset.py
```

Verify the processed dataset:

```bash
python src/verify_preprocessing.py
```

---

# Applications

Accurate lesion segmentation can support medical image analysis applications such as:

- Automated lesion localization
- Computer-aided skin image analysis
- Lesion boundary extraction
- Medical image preprocessing
- Computer vision research in dermatology

This project is intended for **academic and research purposes** and is not a medical diagnostic system.

---

# Author

**Aman Kumar**

B.Tech Computer Science and Engineering  
Indian Institute of Technology Patna  
2024–2028