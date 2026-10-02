"""
TRUVI-EV — Dense Retrieval with FAISS
=======================================
Builds FAISS index from evidence corpus and retrieves top-k evidence for each claim.

Uses: BAAI/bge-base-en-v1.5 embeddings (768-dim)
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_EVIDENCE_DIR = PROJECT_ROOT / "data" / "evidence"
DATA_CLAIMS_DIR = PROJECT_ROOT / "data" / "claims"
DATA_PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

TOP_K = 5
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 64


def load_jsonl(filepath):
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    print("=" * 70)
    print("TRUVI-EV — Phase 5: Dense Retrieval (FAISS)")
    print("=" * 70)
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Top-K: {TOP_K}")
    print("=" * 70)

    DATA_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Check dependencies
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Installing required packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "sentence-transformers", "faiss-cpu", "-q"])
        from sentence_transformers import SentenceTransformer
        import faiss

    # Load evidence corpus
    print("\n[1/5] Loading evidence corpus...")
    corpus = load_jsonl(DATA_EVIDENCE_DIR / "evidence_corpus.jsonl")
    corpus_texts = [c["chunk_text"] for c in corpus]
    corpus_ids = [c["chunk_id"] for c in corpus]
    print(f"  Loaded {len(corpus):,} evidence chunks")

    # Load embedding model
    print(f"\n[2/5] Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  Dimension: {model.get_sentence_embedding_dimension()}")

    # Encode evidence corpus
    print("\n[3/5] Encoding evidence corpus...")
    t0 = time.time()
    corpus_embeddings = model.encode(
        corpus_texts, batch_size=BATCH_SIZE, show_progress_bar=True,
        normalize_embeddings=True
    )
    print(f"  Encoded {len(corpus_embeddings)} chunks in {time.time()-t0:.1f}s")
    print(f"  Shape: {corpus_embeddings.shape}")

    # Build FAISS index
    print("\n[4/5] Building FAISS index...")
    dim = corpus_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine with normalized vectors)
    index.add(corpus_embeddings.astype(np.float32))
    print(f"  Index size: {index.ntotal} vectors")

    # Save index
    index_path = DATA_EVIDENCE_DIR / "faiss_index.bin"
    faiss.write_index(index, str(index_path))
    print(f"  Saved: {index_path}")

    # Save corpus ID mapping
    mapping_path = DATA_EVIDENCE_DIR / "corpus_id_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump({"ids": corpus_ids, "model": EMBEDDING_MODEL, "dim": dim}, f)

    # Retrieve evidence for each claim in each split
    print("\n[5/5] Retrieving evidence for all claims...")
    total_claims = 0
    all_stats = {}

    for split in ["train", "val", "test"]:
        claims_path = DATA_CLAIMS_DIR / f"claims_{split}.jsonl"
        if not claims_path.exists():
            continue

        claims = load_jsonl(claims_path)
        print(f"\n  Processing {split}: {len(claims):,} claims")

        claim_texts = [c["claim_text"] for c in claims]

        # Encode claims in batches
        t0 = time.time()
        claim_embeddings = model.encode(
            claim_texts, batch_size=BATCH_SIZE, show_progress_bar=True,
            normalize_embeddings=True
        )

        # Search
        distances, indices = index.search(claim_embeddings.astype(np.float32), TOP_K)

        # Build retrieval results
        results = []
        for i, claim in enumerate(claims):
            retrieved = []
            for j in range(TOP_K):
                idx = indices[i][j]
                if idx < 0:
                    continue
                retrieved.append({
                    "chunk_id": corpus_ids[idx],
                    "source_id": corpus[idx]["source_id"],
                    "chunk_text": corpus_texts[idx][:500],
                    "score": float(distances[i][j]),
                    "rank": j + 1,
                })

            results.append({
                "claim_id": claim["claim_id"],
                "response_id": claim["response_id"],
                "source_id": claim["source_id"],
                "claim_text": claim["claim_text"],
                "derived_label": claim["derived_label"],
                "split": split,
                "model": claim.get("model", ""),
                "task_type": claim.get("task_type", ""),
                "retrieved_evidence": retrieved,
                "top1_score": float(distances[i][0]) if len(retrieved) > 0 else 0.0,
                "top5_mean_score": float(np.mean(distances[i][:TOP_K])),
            })

        elapsed = time.time() - t0

        # Save
        out_path = DATA_PREDICTIONS_DIR / f"retrieval_{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        total_claims += len(results)
        scores = [r["top1_score"] for r in results]
        all_stats[split] = {
            "claims": len(results),
            "elapsed": round(elapsed, 1),
            "top1_score_mean": round(np.mean(scores), 4),
            "top1_score_median": round(float(np.median(scores)), 4),
        }
        print(f"  {split}: {len(results):,} claims, top-1 mean={np.mean(scores):.4f}, {elapsed:.1f}s")

    # Summary
    summary = {
        "phase": "Phase 5 - Dense Retrieval",
        "timestamp": datetime.now().isoformat(),
        "embedding_model": EMBEDDING_MODEL,
        "top_k": TOP_K,
        "corpus_size": len(corpus),
        "index_type": "IndexFlatIP (cosine)",
        "total_claims_processed": total_claims,
        "splits": all_stats,
    }
    with open(REPORTS_DIR / "retrieval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("DENSE RETRIEVAL COMPLETE")
    print("=" * 70)
    print(f"  Corpus: {len(corpus):,} chunks | Claims: {total_claims:,}")
    for s, st in all_stats.items():
        print(f"  {s}: {st['claims']:,} claims, top-1 cosine={st['top1_score_mean']:.4f}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
