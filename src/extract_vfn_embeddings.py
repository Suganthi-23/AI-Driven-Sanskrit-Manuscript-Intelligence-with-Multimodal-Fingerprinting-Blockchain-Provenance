from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import pandas as pd

from dataset_sanskrit_images import (
    SanskritImageDataset,
    MATERIAL_CLASSES,
    SCRIPT_CLASSES,
    MANUSCRIPT_FLAGS,
)
from model_vfn import VisualFingerprintNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
MODEL_PATH = Path("vfn_model.pt")
OUT_CSV = Path("data/embeddings/vfn_embeddings.csv")


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
    model = VisualFingerprintNet().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    backbone = model.backbone

    dl = make_loader()

    records: List[Dict[str, Any]] = []

    with torch.no_grad():
        for imgs, labels in dl:
            imgs = imgs.to(DEVICE)

            # Feature extraction
            feats = backbone(imgs)  # shape: (B, feat_dim)
            feats = feats.cpu()

            mat_ids = labels["material_type"]
            scr_ids = labels["script_family"]
            man_ids = labels["is_manuscript"]
            paths = labels["image_path"]

            B, D = feats.shape

            for i in range(B):
                mat_idx = int(mat_ids[i])
                scr_idx = int(scr_ids[i])
                man_idx = int(man_ids[i])

                record = {
                    "image_path": paths[i],
                    "material_type": MATERIAL_CLASSES[mat_idx],
                    "script_family": SCRIPT_CLASSES[scr_idx],
                    "is_manuscript": MANUSCRIPT_FLAGS[man_idx],
                }

               
                for j in range(D):
                    record[f"f{j}"] = float(feats[i, j].item())

                records.append(record)

    if not records:
        print("[!] No embeddings generated.")
        return

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print(f"[✓] Embeddings saved.")
    print(f"    File: {OUT_CSV}")
    print(f"    Rows: {len(df)}")
    print(f"    Embedding dimension: {len([c for c in df.columns if c.startswith('f')])}")


if __name__ == "__main__":
    main()
