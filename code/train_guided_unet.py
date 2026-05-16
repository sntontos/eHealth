"""
Phase 1D: Guided U-Net Training
Transforms Semantic Segmentation to Instance Segmentation.
Input: 4 Channels (RGB Image + Binary BBox Guidance Mask).

Experiment IDs: U042 (BUSI), U043 (BraTS)
"""

import os
import cv2
import glob
import torch
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- Configuration ---
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Change these when running in Colab for BUSI vs BraTS
DATASET_NAME = "BUSI" 
EXP_ID = "U042"
IMG_DIR = f"data/{DATASET_NAME}/images/train"
MASK_DIR = f"data/{DATASET_NAME}/masks/train"
BOX_DIR = f"data/{DATASET_NAME}/labels/train" # YOLO txt files
VAL_IMG_DIR = f"data/{DATASET_NAME}/images/val"
VAL_MASK_DIR = f"data/{DATASET_NAME}/masks/val"
VAL_BOX_DIR = f"data/{DATASET_NAME}/labels/val"

def load_first_yolo_box(filepath, img_w, img_h):
    """Reads YOLO format and returns the first box as pixel coordinates."""
    if not os.path.exists(filepath): return None
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    cx, cy, w, h = map(float, parts[1:5])
                    x_min = int((cx - w/2) * img_w)
                    y_min = int((cy - h/2) * img_h)
                    x_max = int((cx + w/2) * img_w)
                    y_max = int((cy + h/2) * img_h)
                    return [max(0, x_min), max(0, y_min), min(img_w, x_max), min(img_h, y_max)]
                except ValueError: continue
    return None

class GuidedSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, box_dir):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        self.mask_dir = mask_dir
        self.box_dir = box_dir

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        filename = os.path.basename(img_path)
        basename = os.path.splitext(filename)[0]
        
        # Load Image
        img = cv2.imread(img_path)
        orig_h, orig_w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        
        # Create Guidance Mask (Channel 4)
        guidance = np.zeros((orig_h, orig_w), dtype=np.float32)
        box_path = os.path.join(self.box_dir, basename + ".txt")
        box = load_first_yolo_box(box_path, orig_w, orig_h)
        
        if box:
            x1, y1, x2, y2 = box
            guidance[y1:y2, x1:x2] = 1.0
            
        guidance = cv2.resize(guidance, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        guidance = np.expand_dims(guidance, axis=-1)
        
        # Combine RGB and Guidance
        input_tensor = np.concatenate([img, guidance], axis=-1) # Shape: (H, W, 4)
        input_tensor = np.transpose(input_tensor, (2, 0, 1)) # HWC to CHW

        # Load GT Mask
        mask_path = os.path.join(self.mask_dir, basename + ".png")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
            
        mask = np.expand_dims(mask, axis=0) # 1HW
        
        return torch.tensor(input_tensor), torch.tensor(mask)

def calculate_metrics(preds, targets):
    preds = (torch.sigmoid(preds) > 0.5).float()
    intersection = (preds * targets).sum((1, 2, 3))
    union = preds.sum((1, 2, 3)) + targets.sum((1, 2, 3))
    
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    iou = (intersection + 1e-6) / (union - intersection + 1e-6)
    
    return dice.mean().item(), iou.mean().item()

def train():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results/phase1d_training", exist_ok=True)
    
    train_dataset = GuidedSegmentationDataset(IMG_DIR, MASK_DIR, BOX_DIR)
    val_dataset = GuidedSegmentationDataset(VAL_IMG_DIR, VAL_MASK_DIR, VAL_BOX_DIR)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Initialize model with in_channels=4
    model = smp.Unet('resnet34', encoder_weights=None, in_channels=4, classes=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    history = []
    best_iou = 0.0

    print(f"Starting {EXP_ID} ({DATASET_NAME}) Guided U-Net Training on {DEVICE}...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for inputs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            inputs, masks = inputs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss, val_dice, val_iou = 0.0, 0.0, 0.0
        with torch.no_grad():
            for inputs, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
                inputs, masks = inputs.to(DEVICE), masks.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                dice, iou = calculate_metrics(outputs, masks)
                val_dice += dice
                val_iou += iou
                
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_dice': val_dice,
            'val_iou': val_iou,
            'lr': LR
        })
        
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), f"checkpoints/{DATASET_NAME.lower()}_guided_unet.pt")
            
    df = pd.DataFrame(history)
    df.to_csv(f"results/phase1d_training/{EXP_ID}_{DATASET_NAME}_guided_unet.csv", index=False)
    print("Training Complete! Logs and checkpoints saved.")

if __name__ == "__main__":
    train()