from pathlib import Path
from typing import List, Dict, Any
import sys

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import pandas as pd
import numpy as np
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).parent))

from dataset_sanskrit_images import (
    SanskritImageDataset,
    MATERIAL_CLASSES,
    SCRIPT_CLASSES,
    MANUSCRIPT_FLAGS,
)
from backend.services.vfn_model import load_vfn_model
from backend.services.textual_service import extract_text_features

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4  # Smaller batch for multimodal processing
MODEL_PATH = Path("vfn_model.pt")
OUT_CSV = Path("data/embeddings/multimodal_embeddings.csv")
TEXT_CACHE_CSV = Path("data/embeddings/text_cache.csv")


def load_text_cache():
    """Load cached text extractions"""
    if TEXT_CACHE_CSV.exists():
        df = pd.read_csv(TEXT_CACHE_CSV)
        return dict(zip(df["image_path"], df["text"]))
    return {}


def save_text_cache(cache: dict):
    """Save text cache"""
    TEXT_CACHE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(cache.items()), columns=["image_path", "text"])
    df.to_csv(TEXT_CACHE_CSV, index=False)


def make_loader():
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    ds = SanskritImageDataset(
        split="all",
        restrict_to_hf=False
    )
    ds.transform = transform

    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    return dl


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights not found: {MODEL_PATH}")

    print(f"[*] Loading model from {MODEL_PATH}")
    model, preprocess = load_vfn_model()
    model = model.to(DEVICE)
    model.eval()
    

    if not hasattr(model, 'backbone'):
        raise AttributeError("Model does not have 'backbone' attribute for feature extraction")

    text_cache = load_text_cache()
    print(f"[*] Loaded {len(text_cache)} cached text extractions")

    dl = make_loader()
    records: List[Dict[str, Any]] = []

    print(f"[*] Processing {len(dl.dataset)} images...")
    print(f"[*] Using device: {DEVICE}")

    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(tqdm(dl, desc="Extracting embeddings")):
            imgs = imgs.to(DEVICE)

            with torch.no_grad():
                img_feats = model.backbone(imgs)  
            
            img_feats = img_feats.cpu().numpy()

            mat_ids = labels["material_type"]
            scr_ids = labels["script_family"]
            man_ids = labels["is_manuscript"]
            paths = labels["image_path"]

            B = len(paths)

            for i in range(B):
                path = paths[i]
                mat_idx = int(mat_ids[i])
                scr_idx = int(scr_ids[i])
                man_idx = int(man_ids[i])

                # Extract text features
                try:
                    img_path = Path(path)
                    if img_path.exists():

                        if path in text_cache and text_cache[path]:

                            from backend.services.textual_service import get_text_embedding
                            text_emb = get_text_embedding(text_cache[path])
                        else:
                            with open(img_path, 'rb') as f:
                                img_bytes = f.read()
                            text_features = extract_text_features(img_bytes)
                            text_emb = text_features["text_embedding"]
                            text_cache[path] = text_features.get("text", "")
                    else:
                        text_emb = np.zeros(768, dtype=np.float32)
                        text_cache[path] = ""
                except Exception as e:

                    text_emb = np.zeros(768, dtype=np.float32)
                    text_cache[path] = ""
                    if batch_idx % 100 == 0:  
                        print(f"[!] Text extraction failed for some images (using zero embeddings)")

                img_feat = img_feats[i]
                combined = np.concatenate([img_feat, text_emb])
                combined = combined / (np.linalg.norm(combined) + 1e-8)

                record = {
                    "image_path": path,
                    "material_type": MATERIAL_CLASSES[mat_idx],
                    "script_family": SCRIPT_CLASSES[scr_idx],
                    "is_manuscript": MANUSCRIPT_FLAGS[man_idx],
                }

                # Add embedding dimensions f0..f2047
                for j in range(len(combined)):
                    record[f"f{j}"] = float(combined[j])

                records.append(record)

    # Save text cache
    save_text_cache(text_cache)
    print(f"[✓] Saved text cache with {len(text_cache)} entries")

    if not records:
        print("[!] No embeddings generated.")
        return

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print(f"[✓] Multimodal embeddings saved.")
    print(f"    File: {OUT_CSV}")
    print(f"    Rows: {len(df)}")
    print(f"    Embedding dimension: 2048 (1280 image + 768 text)")


if __name__ == "__main__":
    main()

