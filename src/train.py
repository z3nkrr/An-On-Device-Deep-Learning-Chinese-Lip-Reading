import os
import sys
import json
import argparse
import time
import csv
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass

# --- 1. 自動定位當前目錄 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from dataset import DMCLRVideoDataset
from model import FullModel

# ==========================================
# 2. 輔助類別與函式
# ==========================================
class EarlyStopping:
    def __init__(self, patience=25):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class CollateFn:
    def __init__(self, label2idx, num_classes):
        self.label2idx = label2idx
        self.num_classes = num_classes

    def __call__(self, batch):
        frames = torch.stack([b['frames'] for b in batch], dim=0)
        labels = []
        for b in batch:
            label_idx = b.get('label', -1)
            if (label_idx is None or label_idx == -1) and b.get('text') in self.label2idx:
                label_idx = self.label2idx[b['text']]
            labels.append(label_idx if label_idx is not None else -1)
        return {'frames': frames, 'label': torch.tensor(labels, dtype=torch.long)}

# ==========================================
# 3. 訓練與驗證函式
# ==========================================
def train_epoch(model, loader, optimizer, criterion, device, epoch, num_epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0
    num_batches = len(loader)
    start_time = time.time()

    print(f"\n>>> Epoch [{epoch+1}/{num_epochs}] Training")
    for batch_idx, batch in enumerate(loader):
        frames, labels = batch['frames'].to(device), batch['label'].to(device)
        mask = labels >= 0
        if mask.sum() == 0: continue
        
        optimizer.zero_grad()
        logits, _ = model(frames[mask])
        loss = criterion(logits, labels[mask])
        loss.backward()
        
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == labels[mask]).sum().item()
        total += labels[mask].size(0)

        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == num_batches:
            progress = (batch_idx + 1) / num_batches * 100
            elapsed = time.time() - start_time
            print(f"\r    [Batch {batch_idx+1:03d}/{num_batches}] {progress:>3.0f}% | "
                  f"Loss: {loss.item():.4f} | Acc: {correct/total:.4f} | Time: {elapsed:.1f}s", end="")
    print() 
    return total_loss / num_batches, correct / max(1, total)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    num_batches = len(loader)
    print(f">>> Validating...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            frames, labels = batch['frames'].to(device), batch['label'].to(device)
            mask = labels >= 0
            if mask.sum() == 0: continue
            logits, _ = model(frames[mask])
            loss = criterion(logits, labels[mask])
            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == labels[mask]).sum().item()
            total += labels[mask].size(0)
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == num_batches:
                print(f"\r    [Val Batch {batch_idx+1:03d}/{num_batches}] Acc: {correct/total:.4f}", end="")
    print()
    return total_loss / num_batches, correct / max(1, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-4) 
    parser.add_argument('--save-dir', default='./checkpoints')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device('xpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    csv_path = os.path.join(args.save_dir, 'result.csv')

    # 載入標籤
    labels_json = os.path.join(SCRIPT_DIR, 'labels.json')
    
    with open(labels_json, 'r', encoding='utf-8') as f:
        label2idx = json.load(f).get('label2idx')
    
    # 自動計算 num_classes
    num_classes = max(int(v) for v in label2idx.values()) + 1

    # 建立 DataLoader
    train_ds = DMCLRVideoDataset(args.data, split='train', seq_len=40)
    val_ds = DMCLRVideoDataset(args.data, split='test', seq_len=40, shuffle_frames=False)

    collate = CollateFn(label2idx, num_classes)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=6, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                            num_workers=4, collate_fn=collate)

    # 初始化模型
    model = FullModel(num_classes=num_classes, in_channels=1).to(device)
    
    # --- 2. 強化 L2 正則化 (Weight Decay) ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # --- 3. 損失函數 (標籤平滑) ---
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    early_stopping = EarlyStopping(patience=20)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])

    print(f"Starting Balanced Training on {device}...")

    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch, args.epochs)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, f"{train_loss:.4f}", f"{train_acc:.4f}", 
                                    f"{val_loss:.4f}", f"{val_acc:.4f}", f"{current_lr:.6f}"])

        torch.save(model.state_dict(), os.path.join(args.save_dir, 'latest.pt'))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best.pt'))
            print(f"[*] Best Acc updated: {best_acc:.4f} -> Saved!")

        early_stopping(val_loss)
        if early_stopping.early_stop: break

if __name__ == '__main__':
    main()