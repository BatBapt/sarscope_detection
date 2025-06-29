import torchvision.models.detection as detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import configuration as cfg


def get_model(num_classes):
    """
    Returns a Faster R-CNN model pre-trained on COCO, with a modified head for the specified number of classes.
    :param num_classes: Number of classes for the model (including background).
    :return: A Faster R-CNN model with a modified head.
    """
    model = detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace pretrained head with new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def count_parameters_and_layers(model):
    """
    Count the total number of parameters and trainable parameters in the model,
    and return the names of the trainable layers.
    :param model: The PyTorch model to analyze.
    :return: A tuple containing total parameters, trainable parameters, and a list of trainable layer names.
    """
    total_params = 0
    trainable_params = 0
    trainable_layers = []

    for name, module in model.named_modules():
        for param in module.parameters(recurse=False):
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                trainable_layers.append(name)
    return total_params, trainable_params, trainable_layers


if __name__ == "__main__":
    num_classes = 1 + 1  # background + class, ie ship
    model = get_model(num_classes).to(cfg.DEVICE)

    print(model)

    total_params, trainable_params, trainable_layers = count_parameters_and_layers(model)

    # Afficher les résultats
    print(f"Nombre total de paramètres : {total_params:,}".replace(",", " "))
    print(f"Nombre de paramètres entraînables : {trainable_params:,}".replace(",", " "))
    print("\nCouches entraînables :")
    for layer in trainable_layers:
        print(f" - {layer}")