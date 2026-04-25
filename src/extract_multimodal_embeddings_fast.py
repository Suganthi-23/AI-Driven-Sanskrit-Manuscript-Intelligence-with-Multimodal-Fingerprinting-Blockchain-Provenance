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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
MODEL_PATH = Path("vfn_model.pt")
OUT_CSV = Path("data/embeddings/multimodal_embeddings.csv")


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

    dl = make_loader()
    records: List[Dict[str, Any]] = []

    print(f"[*] Processing {len(dl.dataset)} images...")
    print(f"[*] Using device: {DEVICE}")
    print(f"[!] FAST MODE: Skipping OCR, using zero text embeddings")
    print(f"[!] This creates 2048-dim embeddings (1280 image + 768 zero text)")

    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(tqdm(dl, desc="Extracting embeddings")):
            imgs = imgs.to(DEVICE)

            # Extract image features
            img_feats = model.backbone(imgs)  # (B, 1280)
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

                # Use zero text embedding (fast mode)
                text_emb = np.zeros(768, dtype=np.float32)

                # Combine image (1280) + text (768) = 2048
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

    if not records:
        print("[!] No embeddings generated.")
        return

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print(f"[✓] Multimodal embeddings saved (FAST MODE - no OCR).")
    print(f"    File: {OUT_CSV}")
    print(f"    Rows: {len(df)}")
    print(f"    Embedding dimension: 2048 (1280 image + 768 zero text)")
    print(f"[!] Note: Text embeddings are zero. Run with OCR later if needed.")


if __name__ == "__main__":
    main()



