import pandas as pd
from pathlib import Path

IMG_META = Path("data/metadata/all_images_processed_manifest.csv")
TEXT_ROOT = Path("data/texts_clean/gretil")

MASTER_OUT = Path("data/metadata/master_metadata.csv")


def main():
    if not IMG_META.exists():
        print("[!] Image metadata missing. Run preprocess_images.py first.")
        return

    df_img = pd.read_csv(IMG_META).reset_index(drop=True)
    df_img["image_id"] = df_img.index + 1  

    df_img["data_type"] = "image"
    df_img["text_file_path"] = None
    df_img["script_family"] = None
    df_img["material_type"] = None
    df_img["date_start_ce"] = None
    df_img["date_end_ce"] = None
    df_img["work_title"] = None

    df_img["manuscript_id"] = df_img.apply(
        lambda r: f"{r['source_root'] or 'imgsrc'}_{int(r['image_id']):06d}", axis=1
    )
    df_img["folio_id"] = df_img.apply(
        lambda r: f"FOLIO_{int(r['image_id']):06d}", axis=1
    )

    rows_text = []
    for txt in TEXT_ROOT.rglob("*.txt"):
        rows_text.append({
            "data_type": "text",
            "raw_image_path": None,
            "processed_image_path": None,
            "thumbnail_path": None,
            "source_root": "gretil",
            "split": None,
            "filename": txt.name,
            "image_id": None,
            "text_file_path": str(txt),
            "script_family": None,
            "material_type": None,
            "date_start_ce": None,
            "date_end_ce": None,
            "work_title": txt.stem,
            "manuscript_id": None,
            "folio_id": None,
        })

    df_text = pd.DataFrame(rows_text)

    common_cols = sorted(set(df_img.columns).union(df_text.columns))
    df_img = df_img.reindex(columns=common_cols)
    df_text = df_text.reindex(columns=common_cols)

    df_master = pd.concat([df_img, df_text], ignore_index=True)

    MASTER_OUT.parent.mkdir(parents=True, exist_ok=True)
    df_master.to_csv(MASTER_OUT, index=False, encoding="utf-8")

    print("[✓] Master metadata created.")
    print(f"    Rows: {len(df_master)}")
    print(f"    File: {MASTER_OUT}")


if __name__ == "__main__":
    main()
