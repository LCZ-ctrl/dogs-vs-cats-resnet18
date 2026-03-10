import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from config import *

# data argumentation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.4, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    ]),

    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}


def get_train_val_loaders():
    train_tf = data_transforms['train']
    val_tf = data_transforms['val']

    # return image & label
    train_dataset = datasets.ImageFolder(root="data/processed/train", transform=train_tf)
    val_dataset = datasets.ImageFolder(root="data/processed/val", transform=val_tf)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Training set sample: {len(train_dataset)}")
    print(f"Validation set sample: {len(val_dataset)}")
    return train_loader, val_loader


class TestDataset(Dataset):
    def __init__(self, test_dir: str = "data/processed/test", transform=None):
        self.test_path = Path(test_dir)
        self.imgs = sorted([f for f in os.listdir(self.test_path) if f.endswith('.jpg')],
                           key=lambda x: int(x.split('.')[0]))
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.test_path / self.imgs[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        img_id = int(self.imgs[idx].split('.')[0])
        return img, img_id


def get_test_loader():
    val_tf = data_transforms['val']
    test_dataset = TestDataset(transform=val_tf)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=True)
    return test_loader
