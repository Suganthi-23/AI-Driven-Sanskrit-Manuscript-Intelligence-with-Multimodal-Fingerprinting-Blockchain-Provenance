import pandas as pd
from pathlib import Path

TEMPLATE_CSV = Path("data/annotations/image_labels_template.csv")


def main():
    if not TEMPLATE_CSV.exists():
        print(f"[!] Template not found: {TEMPLATE_CSV}")
        print("    Run build_image_label_template.py first.")
        return

    df = pd.read_csv(TEMPLATE_CSV)


    required_cols = [
        "image_id",
        "raw_image_path",
        "processed_image_path",
        "thumbnail_path",
        "source_root",
        "split",
        "filename",
        "material_type",
        "script_family",
        "is_manuscript",
        "notes",
    ]
    for col in required_cols:
        if col not in df.columns:
            print(f"[!] Missing column in template: {col}")
            return

    mask_hf = df["source_root"].fillna("") == "hf_sanskrit_ocr"

    df.loc[mask_hf, "material_type"] = "printed"
    df.loc[mask_hf, "script_family"] = "Devanagari"
    df.loc[mask_hf, "is_manuscript"] = "no"
    df.loc[mask_hf, "notes"] = df.loc[mask_hf, "notes"].fillna("") + " auto: HF OCR typed sample"


    mask_other = ~mask_hf

    df.loc[mask_other, "material_type"] = df.loc[mask_other, "material_type"].replace("", "unknown")
    df.loc[mask_other, "script_family"] = df.loc[mask_other, "script_family"].replace("", "unknown")
    df.loc[mask_other, "is_manuscript"] = df.loc[mask_other, "is_manuscript"].replace("", "unknown")


    df.loc[mask_other, "notes"] = df.loc[mask_other, "notes"].fillna("")
    df.loc[mask_other & (df["notes"] == ""), "notes"] = "auto: needs manual review"


    df.to_csv(TEMPLATE_CSV, index=False, encoding="utf-8")

    print("[✓] Auto-labeling complete.")
    print(f"    Updated file: {TEMPLATE_CSV}")
    print(f"    Total rows: {len(df)}")
    print("    - hf_sanskrit_ocr rows -> printed / Devanagari / no (not manuscript)")
    print("    - other rows -> unknown / unknown / unknown (flagged for manual review)")


if __name__ == "__main__":
    main()
