# 🚢 SARScope Ship Detection 🚢

Welcome to the SARScope Ship Detection project! 🌊 This repository is dedicated to detecting ships in Synthetic Aperture Radar (SAR) images using a FastRCNN model. Our goal is to accurately detect and classify each ship within an image. For more details on the FastRCNN model, check out the [paper](https://arxiv.org/abs/1506.01497).

## Table of Contents

- [About](#about)
- [Dataset](#dataset)
- [Setup](#setup)
- [Files](#files)
- [Training Process](#training-process)
- [Performance Metrics](#performance-metrics)
- [Limitations and Future Work](#limitations-and-future-work)
- [Acknowledgments](#acknowledgments)

## About

The SARScope Ship Detection project aims to detect ships in SAR images using a FastRCNN model. This project is designed to help in maritime surveillance and monitoring by accurately identifying ships in radar images.

## Dataset

We use an open-source dataset from Kaggle for this project:
📊 [SARScope: Unveiling the Maritime Landscape](https://www.kaggle.com/datasets/kailaspsudheer/sarscope-unveiling-the-maritime-landscape)

## Setup

To get started with this project, follow these steps:

1. Download the repository to your local machine.
2. Update the path to the dataset in the Jupyter notebook file:

   base_path = 'YOUR_PATH'  # path to your data

3. Ensure all dependencies are installed.

## Files

Here are the main Python files in the project:

- `main.py`: The main script to run the ship detection model. This script loads the model, processes the input SAR images, and outputs the detection results.
- `model.py`: Contains the implementation of the FastRCNN model used for ship detection. This includes the model architecture and training loop.
- `utils.py`: Utility functions used across the project, such as helper functions for data processing, visualization, and evaluation metrics.
- `train.py`: Script to train the model. It sets up the training process, including data loading, model training, and validation.
- `predict.py`: Script to evaluate the performance of the trained model on the test dataset. It calculates and logs various performance metrics.
- `config.py`: Configuration file that contains path and options for the project
## Training Process 📈

The model was trained using the following setup:

- **GPU**: Nvidia GeForce RTX 4060 with 8GB RAM
- **Epochs**: 10

Here are some visualizations from the training process:

![mAP during the training](assets/map_score.png)
![Loss during the training](assets/loss_training.png)

The best model weights were achieved at epoch 64 (starting from 0).

## Performance Metrics

Below are the performance metrics on the test data:

| Metric                | Value               |
|-----------------------|---------------------|
| **map**               | 0.5215848088264465  |
| **map_50**            | 0.8285369873046875  |
| **map_75**            | 0.5805584788322449  |
| **map_small**         | 0.46896490454673767 |
| **map_medium**        | 0.6799818277359009  |
| **map_large**         | 0.5940195322036743  |
| **mar_1**             | 0.2477656453847885  |
| **mar_10**            | 0.5351539254188538  |
| **mar_100**           | 0.5943892598152161  |
| **mar_small**         | 0.5425901412963867  |
| **mar_medium**        | 0.7498947381973267  |
| **mar_large**         | 0.6860465407371521  |
| **map_per_class**     | -1.0                |
| **mar_100_per_class** | -1.0                |
| **classes**           | 1                   |


## Plots

In this folder, you will find plots after a prediction from the model.
They are organized by the threshold used for the prediction.

As we can see, the model performs well with a threshold of 0.5, but loses accuracy with a threshold of 0.9 (multiple ships in the image).

## Limitations and Future Work 🏁

The current model has limitations, particularly with images containing multiple ships. Future improvements could include:

- [ ] More epochs during training to enhance accuracy. ⏳📈
- [ ] Automate training with a YAML file. ✅📄
- [ ] Test multiple backbones to improve model performance. 🏋️‍♂️🔁
- [x] Add images and graphs to illustrate results. 📊📈
## Acknowledgments 📚

- Mask R-CNN: [arXiv:1703.06870](https://arxiv.org/abs/1703.06870)

This project was enhanced with the help of my Mistral AI Agent.
