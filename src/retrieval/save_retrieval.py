"""
TRUVI-EV — Ultra-low-memory claim retrieval.
For 8GB RAM systems. Loads model, builds lean corpus lookup, encodes+searches+writes in tiny batches.
"""
import json, numpy as np, time, sys, gc, os
from pathlib import Path

ROOT = Path('/Users/ydvji/Downloads/capstone/TRUVI-EV')
EV = ROOT / 'data' / 'evidence'
CL = ROOT / 'data' / 'claims'
PR = ROOT / 'data' / 'predictions'
PR.mkdir(parents=True, exist_ok=True)

TOP_K = 5
BATCH = 200  # Very small batches for 8GB RAM

def flush():
    sys.stdout.flush()
    sys.stderr.flush()

# Step 1: Build lean corpus lookup (only IDs and source_ids, not full text)
print("Loading corpus metadata (lean)...")
flush()
corpus_ids = []
corpus_source_ids = []
corpus_texts_file = EV / 'evidence_corpus.jsonl'  # We'll re-read when needed
with open(corpus_texts_file) as f:
    for line in f:
        if line.strip():
            rec = json.loads(line)
            corpus_ids.append(rec['chunk_id'])
            corpus_source_ids.append(rec['source_id'])
print(f"  {len(corpus_ids)} chunk IDs loaded")
flush()

# Step 2: Load FAISS index
print("Loading FAISS index...")
flush()
import faiss
index = faiss.read_index(str(EV / 'faiss_index.bin'))
print(f"  Index: {index.ntotal} vectors")
flush()

# Step 3: Load model
print("Loading SentenceTransformer...")
flush()
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-base-en-v1.5')
print("  Model loaded")
flush()

# Step 4: Build a text lookup function that reads from disk (not memory)
def get_chunk_texts(indices_list):
    """Read specific chunk texts from the corpus file by line number."""
    needed = set(int(i) for i in indices_list if i >= 0)
    if not needed:
        return {}
    texts = {}
    with open(corpus_texts_file) as f:
        for line_idx, line in enumerate(f):
            if line_idx in needed:
                rec = json.loads(line)
                texts[line_idx] = rec['chunk_text'][:500]
            if len(texts) == len(needed):
                break
    return texts

# Step 5: Process each split
for split in ['train', 'val', 'test']:
    claims_path = CL / f'claims_{split}.jsonl'
    if not claims_path.exists():
        continue

    # Count claims first
    n_claims = sum(1 for line in open(claims_path) if line.strip())
    print(f"\n{split}: {n_claims:,} claims")
    flush()

    t0 = time.time()
    out_path = PR / f'retrieval_{split}.jsonl'
    written = 0

    with open(out_path, 'w', encoding='utf-8') as fout:
        # Read claims in batches from file
        batch_claims = []
        with open(claims_path) as fin:
            for line in fin:
                if not line.strip():
                    continue
                batch_claims.append(json.loads(line))

                if len(batch_claims) >= BATCH:
                    # Process batch
                    texts = [c['claim_text'] for c in batch_claims]
                    emb = model.encode(texts, batch_size=32,
                                      show_progress_bar=False,
                                      normalize_embeddings=True)
                    D, I = index.search(emb.astype(np.float32), TOP_K)

                    # Get needed chunk texts from disk
                    all_needed = set()
                    for bi in range(len(batch_claims)):
                        for j in range(TOP_K):
                            if I[bi][j] >= 0:
                                all_needed.add(int(I[bi][j]))
                    chunk_texts = get_chunk_texts(all_needed)

                    # Write results
                    for bi in range(len(batch_claims)):
                        claim = batch_claims[bi]
                        ev = []
                        for j in range(TOP_K):
                            idx = int(I[bi][j])
                            if idx < 0:
                                continue
                            ev.append({
                                'chunk_id': corpus_ids[idx],
                                'source_id': corpus_source_ids[idx],
                                'chunk_text': chunk_texts.get(idx, ''),
                                'score': float(D[bi][j]),
                                'rank': j + 1,
                            })
                        rec = {
                            'claim_id': claim['claim_id'],
                            'response_id': claim['response_id'],
                            'source_id': claim['source_id'],
                            'claim_text': claim['claim_text'],
                            'derived_label': claim['derived_label'],
                            'split': split,
                            'model': claim.get('model', ''),
                            'task_type': claim.get('task_type', ''),
                            'retrieved_evidence': ev,
                            'top1_score': float(D[bi][0]) if ev else 0.0,
                            'top5_mean_score': float(np.mean(D[bi][:TOP_K])),
                        }
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        written += 1

                    del emb, D, I, texts, chunk_texts, batch_claims
                    gc.collect()
                    batch_claims = []

                    elapsed = time.time() - t0
                    if (written // BATCH) % 10 == 0:
                        print(f"  {written:>7,}/{n_claims:,} ({written/n_claims*100:.0f}%) | {elapsed:.0f}s")
                        flush()

        # Process remaining
        if batch_claims:
            texts = [c['claim_text'] for c in batch_claims]
            emb = model.encode(texts, batch_size=32,
                              show_progress_bar=False,
                              normalize_embeddings=True)
            D, I = index.search(emb.astype(np.float32), TOP_K)
            all_needed = set()
            for bi in range(len(batch_claims)):
                for j in range(TOP_K):
                    if I[bi][j] >= 0:
                        all_needed.add(int(I[bi][j]))
            chunk_texts = get_chunk_texts(all_needed)

            for bi in range(len(batch_claims)):
                claim = batch_claims[bi]
                ev = []
                for j in range(TOP_K):
                    idx = int(I[bi][j])
                    if idx < 0:
                        continue
                    ev.append({
                        'chunk_id': corpus_ids[idx],
                        'source_id': corpus_source_ids[idx],
                        'chunk_text': chunk_texts.get(idx, ''),
                        'score': float(D[bi][j]),
                        'rank': j + 1,
                    })
                rec = {
                    'claim_id': claim['claim_id'],
                    'response_id': claim['response_id'],
                    'source_id': claim['source_id'],
                    'claim_text': claim['claim_text'],
                    'derived_label': claim['derived_label'],
                    'split': split,
                    'model': claim.get('model', ''),
                    'task_type': claim.get('task_type', ''),
                    'retrieved_evidence': ev,
                    'top1_score': float(D[bi][0]) if ev else 0.0,
                    'top5_mean_score': float(np.mean(D[bi][:TOP_K])),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            del emb, D, I, texts, chunk_texts, batch_claims
            gc.collect()

    elapsed = time.time() - t0
    fsize = out_path.stat().st_size / 1024 / 1024
    print(f"  DONE: {out_path.name} ({fsize:.1f}MB, {written:,} records, {elapsed:.0f}s)")
    flush()
    gc.collect()

print("\nRETRIEVAL SAVE COMPLETE")
flush()
