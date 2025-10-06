#src/recognise.py
# Removed unused imports
import torch
import numpy as np
import yaml
import cv2
from pathlib import Path
from src.model import EmbeddingNet
from src.dataset import FacesDataset
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

def load_embeddings(emb_dir):
    embs = np.load(Path(emb_dir) / "embeddings.npy")
    labels = np.load(Path(emb_dir) / "labels.npy")
    with open(Path(emb_dir) / "classes.txt") as f:
        classes = [l.strip() for l in f.readlines()]
    return embs, labels, classes

def preprocess_image(pil_img, img_size):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
    return tf(pil_img).unsqueeze(0)

def recognise_image(image_path, cfg_path="config.yaml", checkpoint="models/final/best_model.pth", topk=1):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_net = EmbeddingNet(embedding_size=int(cfg['train']['embedding_size'])).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    emb_net.load_state_dict(ckpt['model_state'])
    emb_net.eval()
    embs, labels, classes = load_embeddings(cfg['data']['embeddings_dir'])
    # compute mean embedding per class
    class_embs = {}
    for c in range(len(classes)):
        idxs = np.where(labels == c)[0]
        if len(idxs) == 0: continue
        class_embs[c] = embs[idxs].mean(axis=0)
    class_ids = list(class_embs.keys())
    class_matrix = np.vstack([class_embs[c] for c in class_ids]) #[C, D]

    # Check if image file exists
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    pil = Image.open(image_path).convert("RGB")
    x = preprocess_image(pil, int(cfg['train']['img_size'])).to(device)
    with torch.no_grad():
        qemb = emb_net(x)
        qnp = qemb.cpu().numpy()[0]
    # cosine similarity
    sims = (class_matrix @ qnp) / (np.linalg.norm(class_matrix, axis=1) * (np.linalg.norm(qnp) + 1e-8))
    best_idx = np.argmax(sims)
    best_class = class_ids[best_idx]
    best_score = sims[best_idx]
    return classes[best_class], float(best_score)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.recognise path/to/image.jpg")
        print("Example: python -m src.recognise \"C:\\Users\\Username\\Pictures\\image.jpg\"")
        sys.exit(1)
    
    img = sys.argv[1]
    print(f"Processing image: {img}")
    
    try:
        name, score = recognise_image(img)
        print(f"Identified as {name} (score: {score:.3f})")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check that the file path is correct and the file exists.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing image: {e}")
        sys.exit(1)