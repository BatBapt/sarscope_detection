import os
import torch
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
import configuration as cfg
import model as my_model
import tools as tools


def train_one_epoch(model, data_loader, device, optimizer, epoch, scaler=None, writer=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Training Epoch {epoch}:"
    model.to(device)

    with tqdm(data_loader, desc=header) as tq:
        for i, (images, targets) in enumerate(tq):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()

            losses.backward()
            optimizer.step()

            metric_logger.update(loss=losses, **loss_dict)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            # Update tqdm postfix to display loss on the progress bar
            tq.set_postfix(loss=losses.item(), lr=optimizer.param_groups[0]["lr"])

            # Log losses to TensorBoard
            if writer is not None:
                writer.add_scalar('Loss/train', losses.item(), epoch * len(data_loader) + i)
                for k, v in loss_dict.items():
                    writer.add_scalar(f'Loss/train_{k}', v.item(), epoch * len(data_loader) + i)

    print(f"Average Loss: {metric_logger.meters['loss'].global_avg:.4f}")
    if writer is not None:
        writer.add_scalar('Loss/avg_train', metric_logger.meters['loss'].global_avg, epoch)


def evaluate(model, data_loader, device, epoch, writer=None):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")
    header = "Validation:"
    total_steps = len(data_loader)

    with torch.no_grad(), tqdm(total=total_steps, desc=header) as progress_bar:
        for i, (images, targets) in enumerate(data_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)

            # Convert outputs for torchmetrics
            preds = [
                {"boxes": out["boxes"], "scores": out["scores"], "labels": out["labels"]}
                for out in outputs
            ]
            targs = [
                {"boxes": tgt["boxes"], "labels": tgt["labels"]}
                for tgt in targets
            ]

            # Update metric for mAP calculation
            metric.update(preds, targs)

            progress_bar.update(1)

    results = metric.compute()
    print("mAP results:")
    print(results)

    # Log mAP to TensorBoard

    if writer is not None:
        for k, v in results.items():
            if v.numel() == 1:  # Single element tensor
                writer.add_scalar(f'mAP/{k}', v.item(), epoch)
            else:  # Multi-element tensor, log each element separately
                for idx, value in enumerate(v):
                    writer.add_scalar(f'mAP/{k}_{idx}', value.item(), epoch)
    return results


def train(model, train_data_loader, val_data_loader, device, num_epochs, optimizer, scaler=None, writer=None):

    best_map = -float('inf')
    for epoch in range(num_epochs):
        train_one_epoch(model, train_data_loader, device, optimizer, epoch, scaler=scaler, writer=writer)
        results = evaluate(model, val_data_loader, device, epoch, writer=writer)

        # Save the model checkpoint if it's the best mAP
        current_map = results['map'].item()
        if current_map > best_map:
            best_map = current_map
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_map': best_map,
                'scaler': scaler.state_dict() if scaler is not None else None
            }, cfg.MODEL_WEIGHTS)
            best_epoch_model = epoch
            print(f"\tModel saved at {best_epoch_model}")

    print("That's it!")
    writer.close()


if __name__ == "__main__":
    subset = "train"
    json_file = os.path.join(cfg.BASE_PATH, f"annotations_{subset}.coco.json")
    image_dir = os.path.join(cfg.BASE_PATH, subset)

    train_dataset = CocoDetection(root=image_dir, annFile=json_file, transform=tools.get_transform(True))

    subset = "valid"
    json_file = os.path.join(cfg.BASE_PATH, f"annotations_{subset}.coco.json")
    image_dir = os.path.join(cfg.BASE_PATH, subset)

    val_dataset = CocoDetection(root=image_dir, annFile=json_file, transform=tools.get_transform(False))

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")


    batch_size = 8
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=tools.collate_fn,
        num_workers=2,
        pin_memory=True
    )



    val_data_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=tools.collate_fn,
        num_workers=2,
        pin_memory=True
    )

    num_classes = 1 + 1  # background + class, ie ship
    model = my_model.get_model(num_classes)

    num_epochs = 10
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0005)
    scaler = torch.cuda.amp.GradScaler()

    writer = SummaryWriter(log_dir='runs/sar_detection')

    train(
        model,
        train_data_loader,
        val_data_loader,
        cfg.DEVICE,
        num_epochs,
        optimizer,
        scaler=scaler,
        writer=writer
    )
