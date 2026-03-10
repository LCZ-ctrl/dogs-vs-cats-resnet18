import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from pathlib import Path
from config import *
from model import ResNet18, ResNet18_transfer
from dataset import get_train_val_loaders
from utils import *


def train():
    set_seed(SEED)
    train_loader, val_loader = get_train_val_loaders()

    # train from scratch
    model = ResNet18(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # transfer learning
    # model = ResNet18_transfer(num_classes=NUM_CLASSES).to(DEVICE)
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # params_1x = [param for name, param in model.named_parameters()
    #              if 'fc' not in name]
    # optimizer = optim.Adam([
    #     {'params': params_1x},
    #     {'params': model.fc.parameters(), 'lr': LEARNING_RATE * 10}
    # ], lr=LEARNING_RATE, weight_decay=1e-4)

    # reduce lr on plateau
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
    )

    # device info (gpu or cpu)
    if DEVICE == 'cuda':
        device = torch.cuda.current_device()
        gpu = torch.cuda.get_device_name(device)
        tqdm.write(f"💻 Device: {gpu}")
    else:
        tqdm.write("💻 Device: CPU")

    best_val_acc = 0.0
    start_epoch = 0

    save_path = Path('models')
    save_path.mkdir(parents=True, exist_ok=True)

    history_path = save_path / 'history.pkl'
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # resume from the latest checkpoint
    latest_checkpoint = save_path / "latest_checkpoint.pth"
    if latest_checkpoint.exists():
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']

        if history_path.exists():
            with open(history_path, 'rb') as f:
                history = pickle.load(f)
            tqdm.write(f"Loaded history: {len(history['train_loss'])} epochs recorded")

        tqdm.write(f"Resumed from epoch {start_epoch + 1}")
    tqdm.write("🚀 Start training...")

    for epoch in range(start_epoch, NUM_EPOCHS):
        # Training
        model.train()
        train_loss = train_correct = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{NUM_EPOCHS}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_correct += calculate_correct(outputs, labels)

        # Validation
        model.eval()
        val_loss = val_correct = 0.0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Val Epoch {epoch + 1}/{NUM_EPOCHS}"):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                val_correct += calculate_correct(outputs, labels)

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_acc = train_correct / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_acc = val_correct / len(val_loader.dataset)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_acc)

        with open(history_path, 'wb') as f:
            pickle.dump(history, f)

        scheduler.step(avg_val_loss)

        tqdm.write(
            f"[epoch {epoch + 1}] "
            f"train loss: {avg_train_loss:.4f} | train acc: {avg_train_acc:.4f} | "
            f"val loss: {avg_val_loss:.4f} | val acc:{avg_val_acc:.4f}"
        )

        # save the best model
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc
            }
            model_path = save_path / f"best_model.pth"
            torch.save(best_checkpoint, model_path)

        # checkpoint saving (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
            }
            torch.save(checkpoint, save_path / f"checkpoint_epoch_{epoch + 1}.pth")

        # always save the latest checkpoint for resuming
        latest = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
        }
        torch.save(latest, save_path / "latest_checkpoint.pth")

    tqdm.write(f"\n🎉 Training complete!")
    tqdm.write(f"💾 Best model saved with acc: {best_val_acc:.4f}")

    plt.figure(figsize=(15, 6))
    epochs = range(1, len(history['train_loss']) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='#1f77b4')
    plt.plot(epochs, history['val_loss'], label='Val Loss', color='#ff7f0e')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc', color='#2ca02c')
    plt.plot(epochs, history['val_acc'], label='Val Acc', color='#d62728')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.title('Accuracy Curve')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path / 'training_metrics.png', dpi=150)
    plt.show()
    plt.close()


if __name__ == "__main__":
    train()
