import pandas as pd
from pathlib import Path

IMG_META = Path("data/metadata/all_images_processed_manifest.csv")

OUT_CSV = Path("data/annotations/image_labels_template.csv")


def main():
    if not IMG_META.exists():
        print(f"[!] Processed image metadata not found: {IMG_META}")
        print("    Run preprocess_images.py first.")
        return

    df = pd.read_csv(IMG_META).reset_index(drop=True)

    df["image_id"] = df.index + 1

    df_out = df[[
        "image_id",
        "raw_image_path",
        "processed_image_path",
        "thumbnail_path",
        "source_root",
        "split",
        "filename",
    ]].copy()

    df_out["material_type"] = ""   
    df_out["script_family"] = ""   
    df_out["is_manuscript"] = ""  
    df_out["notes"] = ""          

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print(f"[✓] Image label template created.")
    print(f"    File: {OUT_CSV}")
    print(f"    Rows: {len(df_out)}")
    print("    You can open this CSV in Excel and fill material_type/script_family/etc.")


if __name__ == "__main__":
    main()
