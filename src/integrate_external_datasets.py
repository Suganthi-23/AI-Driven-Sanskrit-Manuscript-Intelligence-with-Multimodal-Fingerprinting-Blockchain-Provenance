#!/usr/bin/env python
import os
import subprocess
import zipfile
from pathlib import Path

import requests
from datasets import load_dataset


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DATA_ROOT = Path("data/raw/external_datasets")
GH_REPO_URL = "https://github.com/ihdia/sanskrit-ocr.git"

HF_DATASET_ID = "Process-Venue/Sanskrit-OCR-Typed-Dataset"
HF_OUT_DIR = DATA_ROOT / "hf_sanskrit_ocr"

IIT_POSTOCR_ZIP_URL = (
    "https://cdn.iiit.ac.in/cdn/ilocr.iiit.ac.in/dataset/static/assets/"
    "img/publication/handwritten/Post-OCR_Sanskrit.zip"
)
IIT_POSTOCR_ZIP_PATH = DATA_ROOT / "Post-OCR-Sanskrit.zip"
IIT_POSTOCR_OUT = DATA_ROOT / "Post-OCR-Sanskrit"


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def clone_git_repo(url: str, dest: Path):
    if dest.exists():
        print(f"[✓] Git repo already exists: {dest}")
        return

    print(f"[+] Cloning {url}")
    subprocess.run(["git", "clone", url, str(dest)], check=True)
    print(f"[✓] Cloned into: {dest}")


def download_file(url: str, dest: Path):
    print(f"[+] Downloading: {url}")
    ensure_dir(dest.parent)

    r = requests.get(url, stream=True)
    r.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)

    print(f"[✓] Saved: {dest}")


def extract_zip(zip_path: Path, out_dir: Path):
    print(f"[+] Extracting: {zip_path}")
    ensure_dir(out_dir)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    print(f"[✓] Extracted to: {out_dir}")


# ---------------------------------------------------------------------------
# DOWNLOAD HUGGINGFACE DATASET PROPERLY
# ---------------------------------------------------------------------------

def download_hf_dataset():
    print(f"\n[+] Loading HuggingFace dataset: {HF_DATASET_ID}")
    ensure_dir(HF_OUT_DIR)

    ds_dict = load_dataset(HF_DATASET_ID)  # downloads full dataset (~3.4k images)

    meta_rows = []
    for split_name, split in ds_dict.items():
        split_dir = HF_OUT_DIR / split_name
        ensure_dir(split_dir)

        print(f"[*] Saving split '{split_name}' ({len(split)} images)")

        for i, item in enumerate(split):
            img = item["image"]
            label = item["label"]

            filename = item.get("filename", f"{split_name}_{i:05d}.png")
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                filename += ".png"

            out_path = split_dir / filename
            img.save(out_path)

            meta_rows.append(f"{split_name},{filename},{label},{out_path}")

    # Save metadata CSV
    meta_file = HF_OUT_DIR / "metadata.csv"
    with open(meta_file, "w", encoding="utf-8") as f:
        f.write("split,filename,label,path\n")
        for row in meta_rows:
            f.write(row + "\n")

    print(f"[✓] HuggingFace dataset saved → {HF_OUT_DIR}")
    print(f"[✓] Metadata: {meta_file}")


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def main():
    ensure_dir(DATA_ROOT)

    # 1. Clone GitHub repo
    clone_git_repo(GH_REPO_URL, DATA_ROOT / "sanskrit-ocr-gh")

    # 2. Download HuggingFace dataset
    download_hf_dataset()

    # 3. Download IIT Post-OCR dataset (zip)
    try:
        download_file(IIT_POSTOCR_ZIP_URL, IIT_POSTOCR_ZIP_PATH)
        extract_zip(IIT_POSTOCR_ZIP_PATH, IIT_POSTOCR_OUT)
    except Exception as e:
        print(f"[!] Failed to download/extract IIT dataset: {e}")

    print("\n[✓] All external datasets integrated successfully!")
    print(f"Location: {DATA_ROOT}")

if __name__ == "__main__":
    main()
