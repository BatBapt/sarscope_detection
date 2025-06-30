import os
import numpy as np
from tqdm import tqdm
import torch
from torchvision.datasets import CocoDetection
from torchvision.transforms.functional import to_pil_image
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt
import matplotlib.patches as patches


import configuration as cfg
import model as my_model
import tools as tools


def load_model(num_classes, model_weights_path, device):
    """
    Load the model with the specified number of classes and weights.
    Args:
        num_classes (int): Number of classes for the model.
        model_weights_path (str): Path to the model weights.
        device (torch.device): Device to load the model on (CPU or GPU).
    Returns:
        model (torch.nn.Module): The loaded model.
    """
    model = my_model.get_model(num_classes)
    checkpoint = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def extract_true_bboxes(json_file):
    """
    Extract true bounding boxes from the COCO annotations file.
    Args:
        json_file (str): Path to the COCO annotations JSON file.
    Returns:
        true_bboxes (list): List of true bounding boxes for each image.
    """
    annotations_test = tools.load_annotations(json_file)

    true_bboxes = []
    for img_info in annotations_test['images']:
        real_bboxes = [ann['bbox'] for ann in annotations_test['annotations'] if ann['image_id'] == img_info['id']]
        true_bboxes.append(real_bboxes)

    true_bboxes = [true_bboxes[i:i + 8] for i in range(0, len(true_bboxes), 8)]
    return true_bboxes


def plot_image_with_bboxes(image, real_bboxes, pred_bboxes, pred_labels, pred_scores, score_threshold=0.5):
    """
    Plot an image with real and predicted bounding boxes.
    Args:
        image (PIL.Image or torch.Tensor): The image to plot.
        real_bboxes (list): List of real bounding boxes in the format [x, y, w, h].
        pred_bboxes (list): List of predicted bounding boxes in the format [x_min, y_min, x_max, y_max].
        pred_labels (list): List of predicted labels for each bounding box.
        pred_scores (list): List of scores for each predicted bounding box.
        score_threshold (float): Minimum score to display a predicted bounding box.
    Returns:
        None
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    if isinstance(image, torch.Tensor):
        image = to_pil_image(image.cpu())
    axes[0].imshow(image)
    axes[0].set_title('Bounding Boxes Réelles')
    for bbox in real_bboxes:
        x, y, w, h = bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        axes[0].add_patch(rect)

    # Afficher l'image avec les bounding boxes prédites
    axes[1].imshow(image)
    axes[1].set_title('Bounding Boxes Prédites')
    for i, bbox in enumerate(pred_bboxes):
        if pred_scores is not None and pred_scores[i] < score_threshold:
            continue
        x_min, y_min, x_max, y_max = bbox
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='b', facecolor='none')
        axes[1].add_patch(rect)
        if pred_labels is not None:
            axes[1].text(x_min, y_min, f'{pred_labels[i]}', color='b')


def run_inference(model, dataloader, device):
    """
    Run inference on the dataset using the provided model and dataloader.
    Args:
        model (torch.nn.Module): The model to use for inference.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device): Device to run the inference on (CPU or GPU).
    Returns:
        metrics (dict): Computed metrics from the inference.
        real_images (list): List of real images used for inference.
        outputs (list): List of model outputs for each image.
    """
    metric = MeanAveragePrecision(iou_type="bbox")
    real_images = []
    outputs = []

    with torch.no_grad(), tqdm(total=len(dataloader), desc="Computing inference") as progress_bar:
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs_test = model(images)

            preds = [
                {"boxes": out["boxes"], "scores": out["scores"], "labels": out["labels"]}
                for out in outputs_test
            ]

            targs = [
                {"boxes": tgt["boxes"], "labels": tgt["labels"]}
                for tgt in targets
            ]

            metric.update(preds, targs)
            progress_bar.update(1)

            real_images.append(images)
            outputs.append(preds)

    metrics = metric.compute()

    return metrics, real_images, outputs


def plot_sample_images(real_images, true_bboxes, outputs, nb_sample=3, nb_per_batch=3, score_threshold=0.9):
    """
    Plot sample images with their true and predicted bounding boxes.
    Args:
        real_images (list): List of real images.
        true_bboxes (list): List of true bounding boxes for each image.
        outputs (list): List of model outputs for each image.
        nb_sample (int): Number of samples to plot.
        nb_per_batch (int): Number of images to plot per batch.
        score_threshold (float): Minimum score to display a predicted bounding box.
    Returns:
        None
    """
    indices = np.random.randint(0, len(real_images), nb_sample)
    for indice in indices:
        sample_true_bboxes = true_bboxes[indice]
        sample_pred_bboxes = outputs[indice]
        images = real_images[indice]
        for i in range(len(images)):
            curr_image = images[i]
            real_bboxes = sample_true_bboxes[i]
            pred_bboxes = sample_pred_bboxes[i]['boxes'].cpu().numpy()  # Convertir les tenseurs en tableaux numpy
            pred_labels = sample_pred_bboxes[i]['labels'].cpu().numpy() if 'labels' in sample_pred_bboxes[i] else None
            pred_scores = sample_pred_bboxes[i]['scores'].cpu().numpy() if 'scores' in sample_pred_bboxes[i] else None
            plot_image_with_bboxes(curr_image, real_bboxes, pred_bboxes, pred_labels, pred_scores, score_threshold)

            plt.savefig(f"plots/sample_{indice}_image_{i}_threshold_{score_threshold}.png")
            if i == nb_per_batch:
                break


if __name__ == "__main__":
    num_classes = 2  # Adjust based on your dataset
    model_weights_path = cfg.MODEL_WEIGHTS

    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Model weights not found at {model_weights_path}")

    device = cfg.DEVICE
    model = load_model(num_classes, model_weights_path, device)
    print("Model loaded")

    json_file = os.path.join(cfg.BASE_PATH, f"annotations_test.coco.json")
    image_dir = os.path.join(cfg.BASE_PATH, "test")

    test_dataset = CocoDetection(root=image_dir, annFile=json_file, transform=tools.get_transform(False))

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=tools.collate_fn,
        num_workers=2,
        pin_memory=True
    )

    true_bboxes = extract_true_bboxes(os.path.join(cfg.BASE_PATH, "annotations_test.coco.json"))

    metric, real_images, outputs = run_inference(model, test_data_loader, device)

    for k, v in metric.items():
        print(f"{k}: {v.item()}")

    nb_sample = 3
    indices = np.random.randint(0, len(real_images), nb_sample)
    nb_per_batch = 3
    plot_sample_images(real_images, true_bboxes, outputs, nb_sample=nb_sample, nb_per_batch=nb_per_batch, score_threshold=0.75)