import os
import math
import yaml
from dataclasses import dataclass
from pathlib import Path
from safetensors.torch import save_file as safetensors_save_file


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import DeforestationDataset
from src.models.unet import UNet


@dataclass
class TrainConfig:
    pt_dir: str
    train_split: float
    epochs: int
    batch_size: int
    lr: float
    num_workers: int
    amp: bool
    in_channels: int
    num_classes: int
    class_weights: list


def load_config(path: str) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return TrainConfig(
        pt_dir=cfg["data"]["pt_dir"],
        train_split=float(cfg["data"]["train_split"]),
        epochs=int(cfg["training"]["epochs"]),
        batch_size=int(cfg["training"]["batch_size"]),
        lr=float(cfg["training"]["lr"]),
        num_workers=int(cfg["training"]["num_workers"]),
        amp=bool(cfg["training"]["amp"]),
        in_channels=int(cfg["model"]["in_channels"]),
        num_classes=int(cfg["model"]["num_classes"]),
        class_weights=list(cfg["loss"]["class_weights"]),
    )


@torch.no_grad()
def compute_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int):
    # pred/target: [B,H,W] int64
    k = (target >= 0) & (target < num_classes)
    idx = num_classes * target[k] + pred[k]
    cm = torch.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)
    return cm


@torch.no_grad()
def iou_from_cm(cm: torch.Tensor):
    # cm: [C,C], rows=gt, cols=pred
    tp = torch.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = tp + fp + fn
    iou = tp / torch.clamp(denom, min=1)
    miou = iou.mean()
    return iou, miou


def main():
    cfg = load_config("configs/train.yaml")

    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Dataset + split
    full = DeforestationDataset(cfg.pt_dir)
    n = len(full)
    n_train = int(math.floor(n * cfg.train_split))
    indices = torch.randperm(n).tolist()
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_ds = DeforestationDataset(cfg.pt_dir, indices=train_idx)
    val_ds = DeforestationDataset(cfg.pt_dir, indices=val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
    )

    # Model
    model = UNet(in_channels=cfg.in_channels, num_classes=cfg.num_classes, base=32).to(device)

    # Loss
    weights = torch.tensor(cfg.class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and cfg.amp))

    # Checkpoints
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    best_path = ckpt_dir / "best.pt"                 # for resume (contains optimizer etc.)
    best_weights_path = ckpt_dir / "best.safetensors" # for secure inference/export (tensors only)


    best_miou = -1.0

    for epoch in range(1, cfg.epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} [train]")
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)  # [B,10,512,512]
            masks = masks.to(device, non_blocking=True)    # [B,512,512]

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device == "cuda" and cfg.amp)):
                logits = model(images)                     # [B,3,512,512]
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        train_loss /= max(len(train_loader), 1)

        # ---- val ----
        model.eval()
        val_loss = 0.0
        cm = torch.zeros((cfg.num_classes, cfg.num_classes), dtype=torch.int64, device=device)

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.epochs} [val]"):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=(device == "cuda" and cfg.amp)):
                    logits = model(images)
                    loss = criterion(logits, masks)

                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1).to(torch.int64)  # [B,H,W]
                cm += compute_confusion_matrix(preds, masks.to(torch.int64), cfg.num_classes)

        val_loss /= max(len(val_loader), 1)
        iou, miou = iou_from_cm(cm)

        print(f"\nEpoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} mIoU={miou.item():.4f}")
        for c in range(cfg.num_classes):
            print(f"  IoU class {c}: {iou[c].item():.4f}")

        # Save best
        if miou.item() > best_miou:
            best_miou = miou.item()

    # 1) Save resume checkpoint (pickle-based; OK for your own local files)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "best_miou": best_miou,
            "config": cfg.__dict__,
        },
        best_path,
    )

    # 2) Save secure weights-only checkpoint (no pickle)
    # safetensors requires CPU tensors
    state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    safetensors_save_file(state_cpu, str(best_weights_path))

    print(
        f"✅ Saved new best checkpoint: {best_path.as_posix()} (resume)\n"
        f"🔒 Saved secure weights: {best_weights_path.as_posix()} (safetensors, mIoU={best_miou:.4f})\n"
    )

    print("Training done. Best mIoU:", best_miou)


if __name__ == "__main__":
    main()
