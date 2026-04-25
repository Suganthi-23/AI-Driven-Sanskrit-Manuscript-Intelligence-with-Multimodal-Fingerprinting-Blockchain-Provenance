from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

IMG_META_CSV = Path("data/metadata/all_images_processed_manifest.csv")
LABELS_CSV = Path("data/annotations/image_labels_template.csv")

MATERIAL_CLASSES = ["printed", "manuscript", "unknown"]
SCRIPT_CLASSES = ["Devanagari", "Grantha", "Tamil-Grantha", "Sharada", "unknown"]
MANUSCRIPT_FLAGS = ["yes", "no", "unknown"]

def encode_label(value: str, vocab: list) -> int:
    if not isinstance(value, str):
        return vocab.index("unknown")
    value = value.strip()
    if value in vocab:
        return vocab.index(value)
    return vocab.index("unknown")


class SanskritImageDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        img_meta_csv: Path = IMG_META_CSV,
        labels_csv: Path = LABELS_CSV,
        restrict_to_hf: bool = False,
    ):
        if not img_meta_csv.exists():
            raise FileNotFoundError(f"Image metadata not found: {img_meta_csv}")
        if not labels_csv.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_csv}")

        df_img = pd.read_csv(img_meta_csv)
        df_lbl = pd.read_csv(labels_csv)


        if "processed_image_path" not in df_img.columns:
            raise KeyError("processed_image_path missing in image metadata CSV")
        if "processed_image_path" not in df_lbl.columns:
            raise KeyError("processed_image_path missing in labels CSV")

        df = pd.merge(
            df_img,
            df_lbl[
                [
                    "processed_image_path",
                    "material_type",
                    "script_family",
                    "is_manuscript",
                ]
            ],
            on="processed_image_path",
            how="inner",
            suffixes=("", "_lbl"),
        )

        if split in ("train", "validation"):
            df = df[df["split"] == split]
        elif split == "all":
            pass
        else:
            raise ValueError(f"Unknown split: {split}")

        if restrict_to_hf:
            df = df[df["source_root"] == "hf_sanskrit_ocr"]

        if df.empty:
            raise RuntimeError("No rows left after filtering; check your CSVs and filters.")

        self.df = df.reset_index(drop=True)

        # Image transforms
        self.transform = T.Compose(
            [
                T.ToTensor(),  
               
            ]
        )

        print(
            f"[SanskritImageDataset] Loaded {len(self.df)} samples "
            f"(split={split}, restrict_to_hf={restrict_to_hf})"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        row = self.df.iloc[idx]

        img_path = Path(row["processed_image_path"])
        if not img_path.exists():
            raise FileNotFoundError(f"Processed image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        img_t = self.transform(img)

        # Encode labels
        material = encode_label(str(row["material_type"]), MATERIAL_CLASSES)
        script = encode_label(str(row["script_family"]), SCRIPT_CLASSES)
        is_manu = encode_label(str(row["is_manuscript"]), MANUSCRIPT_FLAGS)

        labels = {
            "material_type": material,
            "script_family": script,
            "is_manuscript": is_manu,
            "image_path": str(img_path),
        }

        return img_t, labels


# Simple test routine
if __name__ == "__main__":
    ds = SanskritImageDataset(split="train", restrict_to_hf=True)
    print("Dataset size:", len(ds))
    img, lbl = ds[0]
    print("Image tensor shape:", img.shape)
    print("Labels:", lbl)
