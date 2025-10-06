#src/train.py
import os
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from src.dataset import make_dataloader
from src.model import EmbeddingNet, ArcMarginProduct

def train(cfg_path="config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    # dataloader
    train_loader, train_ds = make_dataloader(cfg['data']['augmented_dir'], batch_size=cfg['train']['batch_size'], img_size=cfg['train']['img_size'], shuffle=True, num_workers=cfg['train']['num_workers'])
    num_classes = len(train_ds.classes)
    print("Classes:", num_classes, train_ds.classes[:10])

    # model
    emb_size = cfg['train']['embedding_size']
    emb_net = EmbeddingNet(embedding_size=emb_size, pretrained=cfg['train']['pretrained_backbone']).to(device)
    arc = ArcMarginProduct(in_features=emb_size, out_features=num_classes, s=cfg['train']['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    params = list(emb_net.parameters()) + list(arc.parameters())
    optimizer = AdamW(params, lr=float(cfg['train']['lr']), weight_decay=float(cfg['train']['weight_decay']))
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    best_loss = 1e9
    os.makedirs("models/checkpoints", exist_ok=True)

    for epoch in range(cfg['train']['epochs']):
        emb_net.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            embeddings = emb_net(imgs) # [B, emb_size]
            logits = arc(embeddings, labels) # [B, num_classes]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            pbar.set_postfix(loss=running_loss/total, acc=correct/total)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: loss={epoch_loss:.4f} acc={epoch_acc:.4f}")

        # scheduler step
        scheduler.step()

        # checkpoint
        ckpt_path = f"models/checkpoints/epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch+1,
            'model_state': emb_net.state_dict(),
            'arc_state': arc.state_dict(),
            'optimizer': optimizer.state_dict(),
            'classes': train_ds.classes
        }, ckpt_path)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch+1,
                'model_state': emb_net.state_dict(),
                'arc_state': arc.state_dict(),
                'optimizer': optimizer.state_dict(),
                'classes': train_ds.classes
            }, "models/final/best_model.pth")
            print("Saved best model.")

if __name__ == "__main__":
    train()