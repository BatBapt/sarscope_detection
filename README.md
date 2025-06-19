# 🚢 SARScope Ship Detection 🚢

## Table of Contents
- [About](#about)
- [Dataset](#dataset)
- [Setup](#setup)
- [Training Process](#training-process)
- [Performance Metrics](#performance-metrics)
- [Limitations and Future Work](#limitations-and-future-work)
- [Acknowledgments](#acknowledgments)

## About
Welcome to the SARScope Ship Detection project! 🌊 This repository is dedicated to detecting ships in Synthetic Aperture Radar (SAR) images using a FastRCNN model. Our goal is to accurately detect and classify each ship within an image. For more details on the FastRCNN model, check out the [paper](https://arxiv.org/abs/1506.01497).

## Dataset
We use an open-source dataset from Kaggle for this project:
📊 [SARScope: Unveiling the Maritime Landscape](https://www.kaggle.com/datasets/kailaspsudheer/sarscope-unveiling-the-maritime-landscape)

## Setup
To get started with this project, follow these steps:
1. Download the repository to your local machine.
2. Update the path to the dataset in the Jupyter notebook file:
```
   base_path = 'YOUR_PATH'  # The data to your data
```

3. Ensure all dependencies are installed.

## Training Process
The model was trained using the following setup:
- **GPU**: Nvidia GeForce RTX 4060 with 8GB RAM
- **Epochs**: 100

Here are some visualizations from the training process:

![mAP during the training](assets/map_score.png "mAP during the training")
![Loss during the training](assets/loss_training.png "Loss during the training")

The best model weights were achieved at epoch 64 (starting from 0).

## Performance Metrics

Below are the performance metrics on the test data:

| Metric | Value |
|--------|-------|
| **map** | 0.5436151623725891 |
| **map_50** | 0.843625545501709 |
| **map_75** | 0.6185094714164734 |
| **map_small** | 0.49138033390045166 |
| **map_medium** | 0.6922172904014587 |
| **map_large** | 0.6687196493148804 |
| **mar_1** | 0.25759682059288025 |
| **mar_10** | 0.562661349773407 |
| **mar_100** | 0.6185203790664673 |
| **mar_small** | 0.5695593953132629 |
| **mar_medium** | 0.7633684277534485 |
| **mar_large** | 0.734883725643158 |
| **map_per_class** | -1.0 |
| **mar_100_per_class** | -1.0 |
| **classes** | 1 |

## Limitations and Future Work
The current model has limitations, particularly with images containing multiple ships. Future improvements could include:
- 🔹 Adding more data
- 🔹 Enhancing data augmentation techniques
- 🔹 Experimenting with different model backbones
- 🔹 Fine-tuning hyperparameters

## Acknowledgments
This project was enhanced with the help of my Mistral AI Agent.
