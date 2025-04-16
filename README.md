# sarscope_detection


## About

This repository is about detect ship in SAR image (https://en.wikipedia.org/wiki/Synthetic-aperture_radar)
The goal is to be able to detect and predict the class for each ship within one image. To do that, I used a FastRCNN model (https://arxiv.org/abs/1506.01497)

## Datas
I used an open source dataset in Kaggle: https://www.kaggle.com/datasets/kailaspsudheer/sarscope-unveiling-the-maritime-landscape

## Run the file:

You can download the file and use it like that, you will just need to change the path to data in the jupyter file
```base_path=YOUR_PATH```

## The training process
I used my personal GPU, a Nvidia GeForce RTX 4060 8Go RAM to train the model for 100 epoch.

![mAP during the training](assets/map_score.png "mAP during the training")\
![loss during the training](assets/loss_training.png "loss during the training")\

The current best weights is at epoch 64 (starting from 0) with current metrics on the test data
| Metric  | Value     |
| :--------------- |:---------------|
| mAP  | 0.54       |
| mAP_50  | 0.84            |
| mAP_75  | 0.61         |
