"""
TRUVI-EV — Verification Signal Computation (Phases 6-9)
=========================================================
Orchestrates all four verification signals for each claim:

  Phase 6: NLI scores (entailment/contradiction/neutral)
  Phase 7: Semantic similarity (cosine between claim and evidence)
  Phase 8: Evidence reliability (retrieval confidence + source match + NLI conf)
  Phase 9: Evidence agreement (agreement among top-k evidence)

Input:  data/predictions/retrieval_{split}.jsonl
Output: data/predictions/signals_{split}.jsonl

Features:
  - Checkpointing: saves progress every N chunks, resumes on restart
  - Memory-efficient: streams claims in chunks of CHUNK_SIZE
  - MPS support: uses Apple Silicon GPU when available
  - Modular: delegates to src/verification/{nli,similarity,reliability,agreement}

Usage:
  python -m src.verification.compute_signals            # all splits
  python -m src.verification.compute_signals --split val # single split
  python -m src.verification.compute_signals --resume    # resume from checkpoint
"""

import sys
import os
import json
import time
import gc
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "predictions" / ".checkpoints"

# Configuration — tuned for 8GB RAM Mac
CHUNK_SIZE = 200        # Claims per processing chunk (low for 8GB)
CHECKPOINT_EVERY = 5    # Save checkpoint every N chunks (every 1000 claims)
NLI_BATCH_SIZE = 32     # NLI model batch size (low for 8GB unified memory)


def flush():
    sys.stdout.flush()


def clean_record(c):
    """Convert numpy types to JSON-serializable Python types."""
    clean = {}
    for k, v in c.items():
        if isinstance(v, (np.floating, np.float32, np.float64)):
            clean[k] = float(v)
        elif isinstance(v, (np.integer, np.int32, np.int64)):
            clean[k] = int(v)
        elif isinstance(v, np.bool_):
            clean[k] = bool(v)
        elif isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        elif isinstance(v, list):
            clean[k] = [clean_record(item) if isinstance(item, dict) else
                        (float(item) if isinstance(item, (np.floating,)) else
                         int(item) if isinstance(item, (np.integer,)) else item)
                        for item in v]
        elif isinstance(v, dict):
            clean[k] = clean_record(v)
        else:
            clean[k] = v
    return clean


def get_checkpoint_path(split: str) -> Path:
    """Get checkpoint file path for a split."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / f"checkpoint_{split}.json"


def load_checkpoint(split: str) -> dict:
    """Load checkpoint if it exists."""
    cp_path = get_checkpoint_path(split)
    if cp_path.exists():
        with open(cp_path) as f:
            return json.load(f)
    return {"lines_processed": 0, "chunks_done": 0}


def save_checkpoint(split: str, lines_processed: int, chunks_done: int):
    """Save checkpoint."""
    cp_path = get_checkpoint_path(split)
    with open(cp_path, "w") as f:
        json.dump({
            "lines_processed": lines_processed,
            "chunks_done": chunks_done,
            "timestamp": datetime.now().isoformat(),
        }, f)


def clear_checkpoint(split: str):
    """Remove checkpoint after successful completion."""
    cp_path = get_checkpoint_path(split)
    if cp_path.exists():
        cp_path.unlink()


def process_split(split: str, nli_model, resume: bool = False, max_claims: int = 0):
    """Process a single split: compute all 4 signals."""
    from src.verification.nli import compute_nli_for_claims
    from src.verification.similarity import compute_similarity_from_retrieval_scores
    from src.verification.reliability import compute_reliability_signals
    from src.verification.agreement import compute_agreement_signals

    retrieval_path = DATA_PREDICTIONS_DIR / f"retrieval_{split}.jsonl"
    if not retrieval_path.exists():
        print(f"\n[SKIP] {split} — no retrieval data found at {retrieval_path}")
        return None

    # Count total lines
    n_total = sum(1 for line in open(retrieval_path) if line.strip())
    if max_claims > 0:
        n_total = min(n_total, max_claims)
        print(f"\n[LIMITED MODE] Processing only {n_total:,} claims (--max-claims={max_claims})")
    print(f"\n{'=' * 60}")
    print(f"Processing: {split} ({n_total:,} claims)")
    print(f"{'=' * 60}")
    flush()

    # Check for checkpoint
    out_path = DATA_PREDICTIONS_DIR / f"signals_{split}.jsonl"
    skip_lines = 0

    if resume:
        cp = load_checkpoint(split)
        skip_lines = cp["lines_processed"]
        if skip_lines > 0 and skip_lines < n_total:
            print(f"  [RESUME] Continuing from line {skip_lines:,} "
                  f"(chunk {cp['chunks_done']})")
            flush()
        elif skip_lines >= n_total:
            print(f"  [SKIP] Already complete ({skip_lines:,}/{n_total:,})")
            flush()
            return _read_stats_from_file(out_path, split)
        # If resuming, open in append mode
        file_mode = "a" if skip_lines > 0 else "w"
    else:
        file_mode = "w"
        skip_lines = 0

    t0 = time.time()
    written = skip_lines  # Total lines including already-written
    chunks_done = skip_lines // CHUNK_SIZE

    # Tracking stats
    agg_ent, agg_con, agg_sim, agg_rel, agg_agr = [], [], [], [], []

    with open(out_path, file_mode, encoding="utf-8") as fout:
        chunk = []
        lines_read = 0

        with open(retrieval_path) as fin:
            for line in fin:
                if not line.strip():
                    continue
                lines_read += 1

                # Stop if max_claims limit reached
                if max_claims > 0 and lines_read > max_claims:
                    break

                # Skip already-processed lines
                if lines_read <= skip_lines:
                    continue

                chunk.append(json.loads(line))

                if len(chunk) >= CHUNK_SIZE:
                    # Process this chunk through all 4 signals
                    chunk = compute_nli_for_claims(chunk, nli_model, batch_size=NLI_BATCH_SIZE)
                    chunk = compute_similarity_from_retrieval_scores(chunk)
                    chunk = compute_reliability_signals(chunk)
                    chunk = compute_agreement_signals(chunk)

                    # Write results
                    for c in chunk:
                        fout.write(json.dumps(clean_record(c), ensure_ascii=False) + "\n")
                        written += 1
                        agg_ent.append(c.get("nli_mean_entailment", 0))
                        agg_con.append(c.get("nli_mean_contradiction", 0))
                        agg_sim.append(c.get("sim_mean", 0))
                        agg_rel.append(c.get("evidence_reliability", 0))
                        agg_agr.append(c.get("evidence_agreement", 0))

                    chunks_done += 1
                    elapsed = time.time() - t0
                    rate = (written - skip_lines) / elapsed if elapsed > 0 else 0
                    eta = (n_total - written) / rate if rate > 0 else 0

                    print(f"  {written:>7,}/{n_total:,} ({written/n_total*100:.1f}%) | "
                          f"{elapsed:.0f}s | {rate:.1f} claims/s | "
                          f"ETA: {eta/60:.0f}min | "
                          f"NLI_ent={np.mean(agg_ent[-CHUNK_SIZE:]):.4f}")
                    flush()

                    # Checkpoint
                    if chunks_done % CHECKPOINT_EVERY == 0:
                        fout.flush()
                        save_checkpoint(split, written, chunks_done)

                    del chunk
                    gc.collect()
                    chunk = []

        # Process remaining
        if chunk:
            chunk = compute_nli_for_claims(chunk, nli_model, batch_size=NLI_BATCH_SIZE)
            chunk = compute_similarity_from_retrieval_scores(chunk)
            chunk = compute_reliability_signals(chunk)
            chunk = compute_agreement_signals(chunk)

            for c in chunk:
                fout.write(json.dumps(clean_record(c), ensure_ascii=False) + "\n")
                written += 1
                agg_ent.append(c.get("nli_mean_entailment", 0))
                agg_con.append(c.get("nli_mean_contradiction", 0))
                agg_sim.append(c.get("sim_mean", 0))
                agg_rel.append(c.get("evidence_reliability", 0))
                agg_agr.append(c.get("evidence_agreement", 0))

            del chunk
            gc.collect()

    # Done — clear checkpoint
    elapsed = time.time() - t0
    clear_checkpoint(split)

    # Compute stats
    fsize = out_path.stat().st_size / 1024 / 1024

    # Count labels
    from collections import Counter
    labels = Counter()
    with open(out_path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                labels[rec.get("derived_label", "UNKNOWN")] += 1

    stats = {
        "claims": written,
        "elapsed": round(elapsed, 1),
        "file_size_mb": round(fsize, 1),
        "labels": dict(labels),
        "nli_mean_ent": round(float(np.mean(agg_ent)), 4) if agg_ent else 0.0,
        "nli_mean_con": round(float(np.mean(agg_con)), 4) if agg_con else 0.0,
        "sim_mean": round(float(np.mean(agg_sim)), 4) if agg_sim else 0.0,
        "rel_mean": round(float(np.mean(agg_rel)), 4) if agg_rel else 0.0,
        "agr_mean": round(float(np.mean(agg_agr)), 4) if agg_agr else 0.0,
    }

    print(f"\n  {split} DONE: {written:,} claims, {fsize:.1f}MB, {elapsed:.0f}s")
    print(f"    NLI  ent={stats['nli_mean_ent']:.4f} con={stats['nli_mean_con']:.4f}")
    print(f"    Sim={stats['sim_mean']:.4f} Rel={stats['rel_mean']:.4f} "
          f"Agr={stats['agr_mean']:.4f}")
    flush()

    return stats


def _read_stats_from_file(out_path, split):
    """Read stats from an already-computed signals file."""
    from collections import Counter

    if not out_path.exists():
        return None

    labels = Counter()
    ent_vals, con_vals, sim_vals, rel_vals, agr_vals = [], [], [], [], []

    with open(out_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            labels[rec.get("derived_label", "UNKNOWN")] += 1
            ent_vals.append(rec.get("nli_mean_entailment", 0))
            con_vals.append(rec.get("nli_mean_contradiction", 0))
            sim_vals.append(rec.get("sim_mean", 0))
            rel_vals.append(rec.get("evidence_reliability", 0))
            agr_vals.append(rec.get("evidence_agreement", 0))

    return {
        "claims": sum(labels.values()),
        "elapsed": 0,
        "file_size_mb": round(out_path.stat().st_size / 1024 / 1024, 1),
        "labels": dict(labels),
        "nli_mean_ent": round(float(np.mean(ent_vals)), 4) if ent_vals else 0.0,
        "nli_mean_con": round(float(np.mean(con_vals)), 4) if con_vals else 0.0,
        "sim_mean": round(float(np.mean(sim_vals)), 4) if sim_vals else 0.0,
        "rel_mean": round(float(np.mean(rel_vals)), 4) if rel_vals else 0.0,
        "agr_mean": round(float(np.mean(agr_vals)), 4) if agr_vals else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="TRUVI-EV: Compute verification signals (Phases 6-9)"
    )
    parser.add_argument(
        "--split", type=str, choices=["train", "val", "test"],
        help="Process only one split (default: all)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--max-claims", type=int, default=0,
        help="Max claims per split (0=all). Use for quick pipeline testing."
    )
    args = parser.parse_args()

    splits = [args.split] if args.split else ["train", "val", "test"]

    print("=" * 70)
    print("TRUVI-EV — Phases 6-9: Verification Signal Computation")
    print("=" * 70)
    print(f"NLI Model: cross-encoder/nli-deberta-v3-base")
    print(f"Chunk size: {CHUNK_SIZE} claims")
    print(f"NLI batch size: {NLI_BATCH_SIZE}")
    print(f"Splits: {splits}")
    print(f"Resume: {args.resume}")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 70)
    flush()

    # Detect device
    import torch
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"\n[DEVICE] Using: {device}")
    flush()

    # Load NLI model
    print(f"\n[SETUP] Loading NLI model: cross-encoder/nli-deberta-v3-base")
    flush()
    from src.verification.nli import load_nli_model
    nli_model = load_nli_model(device=None)  # Let CrossEncoder handle device
    print(f"  Model loaded successfully")
    flush()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    all_stats = {}

    for split in splits:
        stats = process_split(
            split, nli_model,
            resume=args.resume,
            max_claims=args.max_claims,
        )
        if stats:
            all_stats[split] = stats

    # Save summary report
    summary = {
        "phase": "Phases 6-9 - Verification Signals",
        "timestamp": datetime.now().isoformat(),
        "nli_model": "cross-encoder/nli-deberta-v3-base",
        "similarity_model": "BAAI/bge-base-en-v1.5 (same as retrieval)",
        "chunk_size": CHUNK_SIZE,
        "nli_batch_size": NLI_BATCH_SIZE,
        "signals": [
            "NLI (Phase 6)",
            "Semantic Similarity (Phase 7)",
            "Evidence Reliability (Phase 8)",
            "Evidence Agreement (Phase 9)",
        ],
        "reliability_formula": (
            "0.3*retrieval_confidence + 0.3*source_match_ratio "
            "+ 0.2*score_gap_normalized + 0.2*nli_confidence"
        ),
        "agreement_metric": "majority_class_proportion",
        "splits": all_stats,
    }
    with open(REPORTS_DIR / "signals_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL VERIFICATION SIGNALS COMPLETE (Phases 6-9)")
    print("=" * 70)
    for s, st in all_stats.items():
        print(f"  {s}: {st['claims']:,} claims | "
              f"NLI_ent={st['nli_mean_ent']:.4f} | "
              f"Sim={st['sim_mean']:.4f} | "
              f"Rel={st['rel_mean']:.4f} | "
              f"Agr={st['agr_mean']:.4f} | "
              f"{st['elapsed']:.0f}s")
    print("=" * 70)
    flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
