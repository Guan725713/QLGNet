# QLGNet

**QLGNet: A Medical Image Segmentation Network with Query-Driven Local–Global Representation and Prediction-Guided Cross-Layer Recalibration**

![Task](https://img.shields.io/badge/Task-Medical%20Image%20Segmentation-4776B6)
![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C)
![Release](https://img.shields.io/badge/Full%20Code%20%26%20Weights-Coming%20After%20Acceptance-E5A34B)

> **Release notice:** This repository is the official project page for **QLGNet**. The complete source code and pretrained model weights have **not yet been publicly released**. We plan to release the full implementation, training configurations, evaluation scripts, and pretrained weights here **after the paper is accepted**.

## Overview

**QLGNet** is a medical image segmentation network designed to improve feature representation and cross-layer information fusion. It combines **query-driven local–global representation learning** with **prediction-guided recalibration** in an encoder–decoder framework.

The method focuses on three complementary components:

- **Global Query Context Unit (GQCU):** Uses image-adaptive queries to aggregate information from compressed spatial tokens and capture image-level context.
- **Two-stage Local–Global Bottleneck:** Integrates local structural aggregation and global contextual modeling in deep feature representations.
- **Prediction-Guided Skip Recalibration (PGSR):** Uses soft guidance maps derived from auxiliary predictions to recalibrate skip features during decoding.

## Experimental Evaluation

We evaluate QLGNet on four publicly available two-dimensional medical image segmentation datasets. The following results are the **five-fold cross-validation mean Dice scores** reported in the manuscript.


| Dataset            | Modality          | Mean Dice (%) |
| :----------------- | :---------------- | ------------: |
| BUSI               | Breast ultrasound |     **81.78** |
| BUS-BRA            | Breast ultrasound |     **91.04** |
| BREAST-LESIONS-USG | Breast ultrasound |     **79.16** |
| GLAS               | Histopathology    |     **93.10** |

**Model size:** approximately **6.45 million parameters**.

> **Evaluation note:** These results refer to the experimental protocol described in the manuscript. Please consult the paper for dataset preparation, evaluation settings, and comparisons.

## Repository Status and Planned Release

The repository is being prepared for reproducibility. After acceptance, we plan to provide:

- [X]  Complete model implementation and dependency information
- [X]  Training scripts and reproducible configuration files
- [X]  Dataset preparation and split instructions
- [X]  Inference and evaluation scripts
- [X]  Pretrained model weights

**Availability:** The full release is pending paper acceptance. Any files currently present in the repository should not be interpreted as the complete reproducibility package.

## Citation

If you find this work useful, please consider citing our paper once its bibliographic information is available. The official citation will be added here following publication.

## Updates

Please check this repository for code-release announcements and documentation updates.
