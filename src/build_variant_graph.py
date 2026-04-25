from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from backend.services.variant_graph import get_variant_graph
from backend.services.similarity_service import matrix, df

SIMILARITY_THRESHOLD = 0.82
MAX_VARIANTS_PER_MANUSCRIPT = 10


def main():
    if matrix is None or df is None:
        print("[!] Embeddings not loaded. Run extract_multimodal_embeddings.py first")
        return
    
    print(f"[*] Building variant graph from {len(df)} manuscripts")
    print(f"[*] Similarity threshold: {SIMILARITY_THRESHOLD}")
    
    graph = get_variant_graph()
    
    for idx, row in df.iterrows():
        manuscript_id = row.get("image_path", f"ms_{idx}")
        graph.add_manuscript(manuscript_id, {
            "material_type": row.get("material_type", "unknown"),
            "script_family": row.get("script_family", "unknown"),
            "is_manuscript": row.get("is_manuscript", "unknown")
        })
    

    print("[*] Computing similarities and linking variants...")
    links_created = 0
    
    for i in tqdm(range(len(df)), desc="Linking variants"):
        manuscript_id = df.iloc[i].get("image_path", f"ms_{i}")
        

        similarities = matrix @ matrix[i]
        

        top_indices = np.argsort(similarities)[::-1]
        top_indices = [idx for idx in top_indices if idx != i][:MAX_VARIANTS_PER_MANUSCRIPT]
        
        for j in top_indices:
            similarity = float(similarities[j])
            if similarity >= SIMILARITY_THRESHOLD:
                variant_id = df.iloc[j].get("image_path", f"ms_{j}")
                if graph.link_variants(manuscript_id, variant_id, similarity):
                    links_created += 1
    

    graph.save()
    
    stats = graph.get_statistics()
    print(f"\n[✓] Variant graph built:")
    print(f"    Nodes: {stats['nodes']}")
    print(f"    Edges: {stats['edges']}")
    print(f"    Clusters: {stats['clusters']}")
    print(f"    Density: {stats['density']:.4f}")
    print(f"    Links created: {links_created}")

if __name__ == "__main__":
    main()



