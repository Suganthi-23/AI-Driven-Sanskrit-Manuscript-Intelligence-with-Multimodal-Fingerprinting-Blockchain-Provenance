import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

EMB_CSV = Path("data/embeddings/vfn_embeddings.csv")
OUT_CSV = Path("data/embeddings/vfn_fingerprints.csv")


def main():
    if not EMB_CSV.exists():
        print(f"[!] Embeddings file not found: {EMB_CSV}")
        return

    print(f"[*] Loading embeddings from {EMB_CSV}")
    df = pd.read_csv(EMB_CSV)

    feat_cols = [c for c in df.columns if c.startswith("f")]
    if not feat_cols:
        print("[!] No feature columns (f0, f1, ...) found in embeddings CSV.")
        return

    X = df[feat_cols].values.astype("float32") 
    N, D = X.shape
    print(f"[*] Found {N} embeddings of dimension {D}")

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms

    Xq = np.round(Xn * 1000).astype("int32")

    fingerprints = []
    for i in range(N):
        ints = Xq[i].tolist()
        s = ",".join(map(str, ints))
        h = hashlib.sha256(s.encode("utf-8")).hexdigest()
        fingerprints.append(h)

    df_out = df[["image_path", "material_type", "script_family", "is_manuscript"]].copy()
    df_out["fingerprint_sha256"] = fingerprints

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print(f"[✓] Fingerprints saved.")
    print(f"    File: {OUT_CSV}")
    print(f"    Rows: {len(df_out)}")


if __name__ == "__main__":
    main()
