#src/evaluate.py
import torch
import numpy as np
import os
from pathlib import Path
from torch.nn import Embedding
import yaml
from src.dataset import make_dataloader
from src.model import EmbeddingNet

def compute_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []
    paths = []
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            embs = model(imgs)
            embeddings.append(embs.cpu().numpy())
            labels.extend(lbls.numpy().tolist())
    embeddings = np.vstack(embeddings)
    return embeddings, np.array(labels)

def save_embeddings(cfg_path="config.yaml", checkpoint="models/final/best_model.pth"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_net = EmbeddingNet(embedding_size=int(cfg['train']['embedding_size']), pretrained=False).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    emb_net.load_state_dict(ckpt['model_state'])
    loader, ds = make_dataloader(cfg['data']['augmented_dir'], batch_size=32, img_size=int(cfg['train']['img_size']), shuffle=False, num_workers=int(cfg['train']['num_workers']))
    embs, labels = compute_embeddings(emb_net, loader, device)
    os.makedirs(cfg['data']['embeddings_dir'], exist_ok=True)
    # save as numpy with class mapping
    np.save(os.path.join(cfg['data']['embeddings_dir'], "embeddings.npy"), embs)
    np.save(os.path.join(cfg['data']['embeddings_dir'], "labels.npy"), labels)
    with open(os.path.join(cfg['data']['embeddings_dir'], "classes.txt"), "w") as f:
        f.write("\n".join(ds.classes))
    print("Saved embeddings and labels")

if __name__ == "__main__":
    save_embeddings()