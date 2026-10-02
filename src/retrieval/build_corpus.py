"""
TRUVI-EV — Evidence Corpus Construction
=========================================
Builds the evidence corpus from source texts for retrieval.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_EVIDENCE_DIR = PROJECT_ROOT / "data" / "evidence"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

CHUNK_SIZE = 256
CHUNK_OVERLAP = 64


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


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
    print("TRUVI-EV — Phase 4: Evidence Corpus Construction")
    print("=" * 70)
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Chunk size: {CHUNK_SIZE} words, Overlap: {CHUNK_OVERLAP} words")
    print("=" * 70)

    DATA_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Loading processed data...")
    all_records = load_jsonl(DATA_PROCESSED_DIR / "responses_processed.jsonl")
    print(f"  Loaded {len(all_records):,} records")

    print("\n[2/4] Extracting unique source texts...")
    sources = {}
    for r in all_records:
        sid = r["source_id"]
        if sid not in sources:
            sources[sid] = {
                "source_id": sid,
                "source_text": r["source_text"],
                "task_type": r["task_type"],
                "source_origin": r["source_origin"],
            }
    print(f"  Found {len(sources):,} unique sources")

    print("\n[3/4] Chunking source texts...")
    evidence_chunks = []
    chunk_counts = []
    for sid, source in sources.items():
        text = source["source_text"]
        if not text.strip():
            continue
        chunks = chunk_text(text)
        chunk_counts.append(len(chunks))
        for i, chunk in enumerate(chunks):
            evidence_chunks.append({
                "chunk_id": f"{sid}_chunk_{i}",
                "source_id": sid,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_text": chunk,
                "task_type": source["task_type"],
                "source_origin": source["source_origin"],
                "word_count": len(chunk.split()),
            })

    print(f"  Total chunks: {len(evidence_chunks):,}")
    print(f"  Chunks/source: min={min(chunk_counts)}, max={max(chunk_counts)}, "
          f"mean={sum(chunk_counts)/len(chunk_counts):.1f}")

    print("\n[4/4] Saving evidence corpus...")
    corpus_path = DATA_EVIDENCE_DIR / "evidence_corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for chunk in evidence_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"  Saved: {corpus_path} ({len(evidence_chunks):,} chunks)")

    source_map_path = DATA_EVIDENCE_DIR / "source_map.json"
    source_map = {}
    for sid, s in sources.items():
        nc = sum(1 for c in evidence_chunks if c["source_id"] == sid)
        source_map[sid] = {"task_type": s["task_type"], "source_origin": s["source_origin"], "num_chunks": nc}
    with open(source_map_path, "w", encoding="utf-8") as f:
        json.dump(source_map, f, indent=2)

    word_counts = [c["word_count"] for c in evidence_chunks]
    by_task = Counter(c["task_type"] for c in evidence_chunks)
    by_source = Counter(c["source_origin"] for c in evidence_chunks)

    summary = {
        "phase": "Phase 4 - Evidence Corpus Construction",
        "timestamp": datetime.now().isoformat(),
        "config": {"chunk_size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP},
        "unique_sources": len(sources),
        "total_chunks": len(evidence_chunks),
        "chunks_per_source": {"min": min(chunk_counts), "max": max(chunk_counts),
                              "mean": round(sum(chunk_counts)/len(chunk_counts), 2)},
        "chunk_word_count": {"min": min(word_counts), "max": max(word_counts),
                             "mean": round(sum(word_counts)/len(word_counts), 1)},
        "by_task_type": dict(by_task),
        "by_source_origin": dict(by_source),
    }
    with open(REPORTS_DIR / "evidence_corpus_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("EVIDENCE CORPUS COMPLETE")
    print("=" * 70)
    print(f"  Sources: {len(sources):,}  |  Chunks: {len(evidence_chunks):,}")
    print(f"  By task: {dict(by_task)}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
