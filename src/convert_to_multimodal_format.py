import pandas as pd
import numpy as np
from pathlib import Path

INPUT_CSV = Path("data/embeddings/vfn_embeddings.csv")
OUTPUT_CSV = Path("data/embeddings/multimodal_embeddings.csv")

def main():
    if not INPUT_CSV.exists():
        print(f"[!] Input file not found: {INPUT_CSV}")
        print("[!] Run: python extract_vfn_embeddings.py first")
        return
    
    print(f"[*] Reading existing embeddings from {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    feat_cols = [c for c in df.columns if c.startswith("f")]
    if not feat_cols:
        print("[!] No feature columns found")
        return
    
    print(f"[*] Found {len(df)} embeddings of dimension {len(feat_cols)}")

    text_emb_dim = 768
    zero_text = np.zeros((len(df), text_emb_dim), dtype=np.float32)

    image_features = df[feat_cols].values.astype(np.float32)
    image_dim = len(feat_cols)
    
    print(f"[*] Image features: {image_dim}-dim")
    print(f"[*] Text features: {text_emb_dim}-dim (zero)")
    print(f"[*] Combined: {image_dim + text_emb_dim}-dim")

    combined = np.concatenate([image_features, zero_text], axis=1)

    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    combined = combined / norms

    metadata_df = df[["image_path", "material_type", "script_family", "is_manuscript"]].copy()

    feature_dict = {f"f{j}": combined[:, j] for j in range(combined.shape[1])}
    feature_df = pd.DataFrame(feature_dict)

    new_df = pd.concat([metadata_df, feature_df], axis=1)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    
    print(f"[✓] Multimodal embeddings saved!")
    print(f"    File: {OUTPUT_CSV}")
    print(f"    Rows: {len(new_df)}")
    print(f"    Embedding dimension: {combined.shape[1]} ({image_dim} image + {text_emb_dim} text)")
    print(f"[!] Note: Text embeddings are zero. System will work with image features.")


if __name__ == "__main__":
    main()

