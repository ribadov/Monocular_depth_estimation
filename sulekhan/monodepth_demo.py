import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU device name:", torch.cuda.get_device_name(0))

import os
from pathlib import Path
import random
import numpy as np
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib.pyplot as plt


# ---- paths ----
DATA_ROOT = Path("/cluster/courses/cil/monocular-depth-estimation/train")

# ---- training config ----
IMG_SIZE = 128          # 128 is nice for a live demo
BATCH_SIZE = 8
NUM_EPOCHS = 2
LR = 1e-3
MAX_TRAIN_SAMPLES = 3000   # keep small for speed in live demo
MAX_VAL_SAMPLES = 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)


class SimpleDepthDataset(Dataset):
    def __init__(self, root: Path, img_size=128, max_samples=None):
        self.root = Path(root)
        self.img_size = img_size
        
        self.rgb_files = sorted(self.root.glob("*_rgb.png"))
        if max_samples is not None:
            self.rgb_files = self.rgb_files[:max_samples]
        
        assert len(self.rgb_files) > 0, f"No *_rgb.png files found in {self.root}"

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_path = self.rgb_files[idx]
        depth_path = Path(str(rgb_path).replace("_rgb.png", "_depth.npy"))
        
        # load rgb
        rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
        
        # load depth
        depth = np.load(depth_path).astype(np.float32)
        
        # resize rgb
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)   # [1,3,H,W]
        rgb_t = F.interpolate(rgb_t, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        rgb_t = rgb_t.squeeze(0)  # [3,H,W]
        
        # resize depth
        depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
        depth_t = F.interpolate(depth_t, size=(self.img_size, self.img_size), mode="nearest")
        depth_t = depth_t.squeeze(0)  # [1,H,W]
        
        # valid mask: depth > 0
        valid_mask = (depth_t > 0).float()
        
        # optional normalization of valid depth values
        # keeps the target range smaller and easier for the toy model
        depth_t = torch.clamp(depth_t, min=0.0, max=80.0)
        depth_t = depth_t / 80.0
        
        return {
            "image": rgb_t,
            "depth": depth_t,
            "mask": valid_mask,
            "name": rgb_path.name
        }


full_dataset = SimpleDepthDataset(DATA_ROOT, img_size=IMG_SIZE, max_samples=MAX_TRAIN_SAMPLES + MAX_VAL_SAMPLES)

n_total = len(full_dataset)
n_val = min(MAX_VAL_SAMPLES, max(1, int(0.15 * n_total)))
n_train = n_total - n_val

train_dataset, val_dataset = random_split(
    full_dataset,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples:   {len(val_dataset)}")


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.enc1 = DoubleConv(3, 16)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        
        self.bottleneck = DoubleConv(32, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64, 32)
        
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(32, 16)
        
        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)          # [B,16,H,W]
        e2 = self.enc2(self.pool1(e1))   # [B,32,H/2,W/2]
        
        b = self.bottleneck(self.pool2(e2))  # [B,64,H/4,W/4]
        
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.out_conv(d1)
        out = torch.sigmoid(out)   # output in [0,1]
        return out


model = TinyUNet().to(DEVICE)
print(model)


def silog_loss(pred, target, mask, lambda_=0.5, eps=1e-6):
    """
    Scale-Invariant Log RMSE (SILog)

    pred, target: [B,1,H,W]
    mask:         [B,1,H,W] (1 = valid, 0 = ignore)
    """
    # keep only valid pixels
    pred = pred[mask > 0]
    target = target[mask > 0]
    
    # avoid log(0)
    pred = torch.clamp(pred, min=eps)
    target = torch.clamp(target, min=eps)
    
    log_diff = torch.log(pred) - torch.log(target)
    
    mse = torch.mean(log_diff ** 2)
    mean = torch.mean(log_diff)
    
    loss = mse - lambda_ * (mean ** 2)
    return loss


optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def run_epoch(loader, model, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    total_loss = 0.0
    
    for batch in loader:
        images = batch["image"].to(DEVICE)
        depths = batch["depth"].to(DEVICE)
        masks = batch["mask"].to(DEVICE)
        
        with torch.set_grad_enabled(is_train):
            preds = model(images)
            loss = silog_loss(preds, depths, masks)
            
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

for epoch in range(NUM_EPOCHS):
    train_loss = run_epoch(train_loader, model, optimizer=optimizer)
    val_loss = run_epoch(val_loader, model, optimizer=None)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")


model.eval()

batch = next(iter(val_loader))
images = batch["image"].to(DEVICE)
depths = batch["depth"].to(DEVICE)
masks = batch["mask"].to(DEVICE)
names = batch["name"]

with torch.no_grad():
    preds = model(images)

i = 0
img = images[i].cpu().permute(1, 2, 0).numpy()
gt = depths[i, 0].cpu().numpy()
pred = preds[i, 0].cpu().numpy()
mask = masks[i, 0].cpu().numpy()

# hide invalid gt pixels
gt_vis = gt.copy()
gt_vis[mask == 0] = np.nan

plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("RGB")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(gt_vis, cmap="viridis")
plt.title("Ground Truth Depth")
plt.axis("off")
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(1, 3, 3)
plt.imshow(pred, cmap="viridis")
plt.title("Predicted Depth")
plt.axis("off")
plt.colorbar(fraction=0.046, pad=0.04)

plt.suptitle(names[i])
plt.tight_layout()
plt.show()
