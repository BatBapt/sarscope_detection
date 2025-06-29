import os
import torch


BASE_PATH = "D:/Programmation/IA/datas/sarscope/SARscope/"

TORCH_MODEL_PATH = "D:/models/torch/hub"
torch.hub.set_dir(TORCH_MODEL_PATH) # Un/comment this line to un/set the directory to download pytorch models

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

MODEL_WEIGHTS = "checkpoints/best_model.pth"  # where the model is saved during the training