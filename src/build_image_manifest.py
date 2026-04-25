import os
import csv
from pathlib import Path


RAW_ROOT = Path("data/raw/manuscripts")


OUT_CSV = Path("data/metadata/all_images_raw_manifest.csv")


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def main():
    if not RAW_ROOT.exists():
        print(f"[!] RAW_ROOT does not exist: {RAW_ROOT}")
        return

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    idx = 0

    print(f"[*] Scanning for images under: {RAW_ROOT}")

    for root, dirs, files in os.walk(RAW_ROOT):
        for name in files:
            ext = Path(name).suffix.lower()
            if ext not in IMAGE_EXTS:
                continue

            full_path = Path(root) / name
            rel_path = full_path.relative_to(RAW_ROOT)


            parts = rel_path.parts
            source_root = parts[0] if len(parts) > 1 else ""


            split = ""
            if len(parts) > 2 and parts[1] in {"train", "validation", "test"}:
                split = parts[1]

            idx += 1
            rows.append({
                "image_id": idx,
                "source_root": source_root,       
                "split": split,                  
                "filename": name,
                "relative_path": str(rel_path),  
                "image_path": str(full_path),    
            })

    if not rows:
        print("[!] No images found. Put images under data/raw/manuscripts/ and run again.")
        return

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[✓] Manifest saved: {OUT_CSV}")
    print(f"    Total images listed: {len(rows)}")


if __name__ == "__main__":
    main()
