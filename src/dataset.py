#src/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

class FacesDataset(Dataset):
    def __init__(self, root_dir, img_size=320, transform=None):
        self.root = Path(root_dir)
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.paths = []
        for cls in self.classes:
            for p in (self.root / cls).glob("*"):
                self.paths.append((str(p), self.class_to_idx[cls]))
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path, label = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label

def make_dataloader(root_dir, batch_size=32, img_size=320, shuffle=True, num_workers=4):
    ds = FacesDataset(root_dir, img_size=img_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers), ds
