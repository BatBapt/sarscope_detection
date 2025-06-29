import os
import json
from PIL import Image, ImageDraw
import random
import matplotlib.pyplot as plt
import torch
from torchvision import transforms


import configuration as cfg


def load_annotations(annotation_path):
    """
    Load annotations from a JSON file.

    :param annotation_path: Path to the JSON file containing annotations.
    :return: Parsed annotations as a dictionary.
    """
    with open(annotation_path, "r") as f:
        annotations = json.load(f)
    return annotations


def draw_bounding_boxes(image, bboxes):
    """
    Draw bounding boxes on an image.
    :param image: PIL Image object.
    :param bboxes: List of bounding boxes, where each box is a tuple (x, y, w, h, category_id).
    :return: Image with bounding boxes drawn.
    """
    draw = ImageDraw.Draw(image)
    font_size = int(min(image.size) * 0.02)  # Adjust font size based on image size

    for bbox, category_id in bboxes:
        x, y, w, h = bbox
        x1, y1, x2, y2 = x, y, x + w, y + h
        # color = category_colors.get(category_id, 'white')  # Default to white if category_id is unknown
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - font_size), str(category_id), fill="red")
    return image


def visualize_samples(image_dir, annotation_path, num_samples=5):
    """
    Visualize a few samples from the dataset with bounding boxes.
    :param image_dir: Directory containing images.
    :param annotation_path: Path to the JSON file containing annotations.
    :param num_samples: Number of samples to visualize.
    :return: None
    """
    annotations = load_annotations(annotation_path)
    images_info = annotations["images"]
    bboxes_infos = annotations["annotations"]

    image_with_bboxes = {}
    for bbox in bboxes_infos:
        image_id = bbox["image_id"]
        if image_id not in image_with_bboxes:
            image_with_bboxes[image_id] = []
        image_with_bboxes[image_id].append((bbox['bbox'], bbox['category_id']))

    random.shuffle(images_info)

    plt.figure(figsize=(12, num_samples * 2))
    sample_count = 0
    for image_info in images_info:
        if sample_count >= num_samples:
            break

        image_path = os.path.join(image_dir, image_info['file_name'])

        if not os.path.exists(image_path):
            continue

        image = Image.open(image_path)
        image_id = image_info['id']
        if image_id in image_with_bboxes:
            bboxes = image_with_bboxes[image_id]
            image = draw_bounding_boxes(image, bboxes)
        else:
          image = image

        plt.subplot(num_samples, 1, sample_count + 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Image ID: {image_id}")
        sample_count += 1

    plt.tight_layout()
    plt.show()

def collate_fn(batch):
    """
    Custom collate function to handle batches of images and targets.
    This function converts bounding boxes from (x, y, w, h) format to (x1, y1, x2, y2) format,
    which is required by Faster R-CNN.
    :param batch: List of tuples (image, target) where target is a list of annotations.
    :return: Tuple of images and filtered targets.
    """
    images, targets = zip(*batch)
    filtered_targets = []
    for target in targets:
        boxes = []
        labels = []
        for ann in target:
            x, y, w, h = ann['bbox']
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])  # <-- conversion ici
                labels.append(ann['category_id'])
        if boxes:
            filtered_targets.append({
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64)
            })
        else:
            # Pour éviter des batches vides (Faster R-CNN n'aime pas ça)
            filtered_targets.append({
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64)
            })
    return images, filtered_targets


def get_transform(train):
    """
    Returns a set of transformations for the dataset.
    If `train` is True, it includes random horizontal and vertical flips. (add more if needed)
    Else, it only includes conversion to tensor.
    """
    transform_list = []
    if train:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.3))
        transform_list.append(transforms.RandomVerticalFlip(p=0.3))
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


if __name__ == "__main__":
    subset = "train"
    subset_path = os.path.join(cfg.BASE_PATH, subset)
    visualize_samples(
        image_dir=subset_path,
        annotation_path=os.path.join(cfg.BASE_PATH, f"annotations_{subset}.coco.json"),
        num_samples=1
    )
