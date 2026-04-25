#!/usr/bin/env python
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Manifest built in previous step
RAW_MANIFEST = Path("data/metadata/all_images_raw_manifest.csv")

# Output folders
OUT_IMG_DIR = Path("data/processed/images_1024")
OUT_THUMB_DIR = Path("data/processed/thumbnails")
OUT_META = Path("data/metadata/all_images_processed_manifest.csv")

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_THUMB_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_single_image(src_path: Path, out_path: Path, thumb_path: Path) -> bool:
    img = cv2.imread(str(src_path))
    if img is None:
        print(f"[WARN] Could not read image: {src_path}")
        return False

    # -----------------------------
    # 1. CLAHE for contrast
    # -----------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # -----------------------------
    # 2. Bilateral Denoise
    # -----------------------------
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # -----------------------------
    # 3. Resize + maintain aspect ratio within 1024x1024
    # -----------------------------
    h, w = img.shape[:2]
    scale = 1024.0 / max(h, w)
    resized_w, resized_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    # -----------------------------
    # 4. Pad into 1024x1024 canvas (white)
    # -----------------------------
    canvas = np.full((1024, 1024, 3), 255, dtype=np.uint8)
    y_off = (1024 - resized_h) // 2
    x_off = (1024 - resized_w) // 2
    canvas[y_off:y_off + resized_h, x_off:x_off + resized_w] = img_resized

    # -----------------------------
    # 5. Save processed + thumbnail
    # -----------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    thumb_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_path), canvas)

    thumb = cv2.resize(canvas, (256, 256), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(thumb_path), thumb)

    return True


def main():
    if not RAW_MANIFEST.exists():
        print(f"[ERROR] Manifest not found: {RAW_MANIFEST}")
        return

    df = pd.read_csv(RAW_MANIFEST)
    print(f"[*] Preprocessing {len(df)} images...")

    records = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        raw_path = Path(row["image_path"])
        rel_path = Path(row["relative_path"])

        out_img_path = OUT_IMG_DIR / rel_path
        out_thumb_path = OUT_THUMB_DIR / rel_path

        ok = preprocess_single_image(raw_path, out_img_path, out_thumb_path)
        if not ok:
            continue

        records.append({
            "raw_image_path": str(raw_path),
            "processed_image_path": str(out_img_path),
            "thumbnail_path": str(out_thumb_path),
            "source_root": row["source_root"],
            "split": row["split"],
            "filename": row["filename"]
        })

    if not records:
        print("[!] No images processed.")
        return

    df_out = pd.DataFrame(records)
    OUT_META.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_META, index=False, encoding="utf-8")

    print(f"[✓] Preprocessing complete!")
    print(f"   Processed images → {OUT_IMG_DIR}")
    print(f"   Thumbnails → {OUT_THUMB_DIR}")
    print(f"   Metadata → {OUT_META}")


if __name__ == "__main__":
    main()
