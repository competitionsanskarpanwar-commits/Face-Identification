#src/data_preprocessing.py

import os
from re import S, X
import cv2
from pathlib import Path
from tqdm import tqdm

HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def detect_and_crop(input_root="data/raw", output_root="data/processed"):
    detector = cv2.CascadeClassifier(HAAR_PATH)
    input_root = Path(input_root)
    output_root = Path(output_root)
    for person_dir in tqdm(list(input_root.iterdir())):
        if not person_dir.is_dir(): continue
        out_person = output_root / person_dir.name
        ensure_dir(out_person)
        for img_path in person_dir.glob("*"):
            try:
                img = cv2.imread(str(img_path))
                if img is None: 
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(faces) == 0:
                    continue
                #pick the largest face
                x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
                #square crop around the face
                cx, cy = x + w//2, y + h//2
                side = max(w, h)
                #add small margin to the side
                margin = int(side * 0.2)
                side = side + margin
                x1 = max(cx - side//2, 0)
                y1 = max(cy - side//2, 0)
                x2 = min(cx + side//2, img.shape[1])
                y2 = min(cy + side//2, img.shape[0])
                face = img[y1:y2, x1:x2]
                face = cv2.resize(face, (320, 320))
                out_path = out_person / img_path.name
                cv2.imwrite(str(out_path), face)
            except Exception as e:
                print("Skipping", img_path, "error:", e)

if __name__ == "__main__":
    detect_and_crop()