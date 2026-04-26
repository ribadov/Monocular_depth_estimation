import os
from pathlib import Path
import numpy as np
from PIL import Image
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision import transforms
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
DATA_ROOT = Path("/cluster/courses/cil/monocular-depth-estimation/train")

IMG_SIZE = 128
BATCH_SIZE = 16
NUM_EPOCHS = 10
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------


# ---------------- DATASET ----------------
class DepthDataset(Dataset):
    def __init__(self, root, img_size=128, augment=False):
        self.root = Path(root)
        self.img_size = img_size
        self.augment = augment

        self.rgb_files = sorted(self.root.glob("*_rgb.png"))
        assert len(self.rgb_files) > 0

        self.resize = transforms.Resize((img_size, img_size))

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_path = self.rgb_files[idx]
        depth_path = Path(str(rgb_path).replace("_rgb.png", "_depth.npy"))

        rgb = Image.open(rgb_path).convert("RGB")
        depth = np.load(depth_path).astype(np.float32)

        # augmentations
        if self.augment and random.random() > 0.5:
            rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
            depth = np.fliplr(depth).copy()

        rgb = self.resize(rgb)
        depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        depth = F.interpolate(depth, size=(self.img_size, self.img_size), mode="nearest")
        depth = depth.squeeze(0)

        rgb = transforms.ToTensor()(rgb)

        # normalize image
        rgb = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)(rgb)

        # normalize depth
        depth = torch.clamp(depth, 0, 80) / 80.0
        mask = (depth > 0).float()

        return rgb, depth, mask


# ---------------- MODEL ----------------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(3, 32)
        self.enc2 = DoubleConv(32, 64)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = DoubleConv(64, 32)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        b = self.bottleneck(self.pool(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.out(d1))


# ---------------- LOSS ----------------
def silog_loss(pred, target, mask):
    pred = pred[mask > 0]
    target = target[mask > 0]

    pred = torch.clamp(pred, 1e-6)
    target = torch.clamp(target, 1e-6)

    log_diff = torch.log(pred) - torch.log(target)
    return torch.mean(log_diff**2) - 0.5 * torch.mean(log_diff)**2


def combined_loss(pred, target, mask):
    return silog_loss(pred, target, mask) + 0.1 * F.l1_loss(pred[mask > 0], target[mask > 0])


# ---------------- TRAIN ----------------
def train_epoch(loader, model, optimizer, scaler):
    model.train()
    total = 0

    for rgb, depth, mask in loader:
        rgb, depth, mask = rgb.to(DEVICE), depth.to(DEVICE), mask.to(DEVICE)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            pred = model(rgb)
            loss = combined_loss(pred, depth, mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total += loss.item()

    return total / len(loader)


@torch.no_grad()
def eval_epoch(loader, model):
    model.eval()
    total = 0

    for rgb, depth, mask in loader:
        rgb, depth, mask = rgb.to(DEVICE), depth.to(DEVICE), mask.to(DEVICE)
        pred = model(rgb)
        loss = combined_loss(pred, depth, mask)
        total += loss.item()

    return total / len(loader)


# ---------------- MAIN ----------------
def main():
    dataset = DepthDataset(DATA_ROOT, IMG_SIZE, augment=True)

    n_val = int(0.1 * len(dataset))
    n_train = len(dataset) - n_val

    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = UNet().to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.cuda.amp.GradScaler()

    best_val = float("inf")

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(train_loader, model, optimizer, scaler)
        val_loss = eval_epoch(val_loader, model)

        scheduler.step()

        print(f"Epoch {epoch+1}: train={train_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_model.pt")


if __name__ == "__main__":
    main()