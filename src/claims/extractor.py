"""
TRUVI-EV — Claim Extractor
============================
Extracts factual claims from LLM responses.

Methods:
1. LLM-based: Uses instruction-following model (Qwen2.5-7B-Instruct or similar)
2. Sentence-based: Rule-based sentence splitting as practical fallback

The sentence-based method is used when no LLM is available locally.
It splits responses into sentences, filters non-factual ones, and treats
each sentence as an atomic claim. This is documented and can be replaced
with LLM-based extraction when compute is available.

Usage:
    python3 -m src.claims.extractor
"""

import os
import re
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter

from src.claims.validator import (
    parse_claims_json,
    validate_claims_batch,
    align_claim_with_spans,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_CLAIMS_DIR = PROJECT_ROOT / "data" / "claims"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"


# ============================================================
# Sentence-Based Claim Extractor (practical fallback)
# ============================================================

# Patterns to filter out non-factual sentences
NON_FACTUAL_PATTERNS = [
    r'^(in\s+)?(summary|conclusion|overall|to\s+summarize)',
    r'^(please|note|remember|keep\s+in\s+mind)',
    r'^(i\s+hope|i\s+think|i\s+believe|in\s+my\s+opinion)',
    r'^(however|therefore|thus|hence|consequently)\s*,?\s*$',
    r'^\s*$',
    r'^(yes|no|sure|absolutely|definitely)\s*[.!]?\s*$',
]


def is_factual_sentence(sentence):
    """Check if a sentence is likely a factual claim (not opinion/meta)."""
    s = sentence.strip().lower()
    
    if len(s) < 10:
        return False
    
    # Filter non-factual patterns
    for pattern in NON_FACTUAL_PATTERNS:
        if re.match(pattern, s, re.IGNORECASE):
            return False
    
    # Must have at least 3 words
    if len(s.split()) < 3:
        return False
    
    return True


def split_into_sentences(text):
    """
    Split text into sentences using regex-based rules.
    Handles abbreviations, decimals, and common edge cases.
    """
    if not text:
        return []
    
    # Protect common abbreviations
    text = text.replace("Mr.", "Mr\x00")
    text = text.replace("Mrs.", "Mrs\x00")
    text = text.replace("Dr.", "Dr\x00")
    text = text.replace("vs.", "vs\x00")
    text = text.replace("U.S.", "U\x00S\x00")
    text = text.replace("i.e.", "i\x00e\x00")
    text = text.replace("e.g.", "e\x00g\x00")
    text = text.replace("etc.", "etc\x00")
    text = text.replace("Inc.", "Inc\x00")
    text = text.replace("Ltd.", "Ltd\x00")
    text = text.replace("St.", "St\x00")
    text = text.replace("Ave.", "Ave\x00")
    
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text)
    
    # Restore abbreviations
    sentences = [s.replace("\x00", ".") for s in sentences]
    
    # Clean up
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def extract_claims_sentence_based(response_text):
    """
    Extract claims using sentence-based splitting.
    
    This is the practical fallback when no LLM is available.
    Each sentence that passes the factual filter becomes a claim.
    
    Returns:
        list of claim dicts: [{"claim_id": "c1", "claim": "..."}, ...]
    """
    sentences = split_into_sentences(response_text)
    
    claims = []
    claim_idx = 1
    
    for sent in sentences:
        sent = sent.strip()
        if is_factual_sentence(sent):
            claims.append({
                "claim_id": f"c{claim_idx}",
                "claim": sent,
            })
            claim_idx += 1
    
    return claims


# ============================================================
# Main Extraction Pipeline
# ============================================================

def load_jsonl(filepath):
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def process_single_response(record, method="sentence"):
    """
    Extract claims from a single response and align with hallucination spans.
    
    Returns:
        list of claim-level records
    """
    response_text = record.get("response_text", "")
    response_id = record.get("response_id", "")
    original_labels = record.get("original_labels", [])
    
    # Extract claims
    if method == "sentence":
        raw_claims = extract_claims_sentence_based(response_text)
    else:
        # LLM-based extraction would go here
        raw_claims = extract_claims_sentence_based(response_text)
    
    # Validate claims
    valid_claims, errors = validate_claims_batch(raw_claims)
    
    # Align each claim with hallucination spans and derive labels
    claim_records = []
    for claim in valid_claims:
        alignment = align_claim_with_spans(
            claim["claim"],
            response_text,
            original_labels,
        )
        
        claim_record = {
            # === Claim identity ===
            "claim_id": f"{response_id}_{claim['claim_id']}",
            "response_id": response_id,
            "source_id": record.get("source_id", ""),
            
            # === Claim text ===
            "claim_text": claim["claim"],
            "claim_index": claim["claim_id"],
            
            # === Response context ===
            "model": record.get("model", ""),
            "task_type": record.get("task_type", ""),
            "source_origin": record.get("source_origin", ""),
            "split": record.get("split", record.get("original_split", "")),
            
            # === Source text (for later retrieval) ===
            "source_text": record.get("source_text", ""),
            
            # === Alignment & Labels ===
            "alignment_method": alignment.get("alignment_method", ""),
            "claim_start": alignment.get("claim_start"),
            "claim_end": alignment.get("claim_end"),
            "num_overlapping_spans": len(alignment.get("overlapping_spans", [])),
            "overlap_label_types": alignment.get("overlap_label_types", []),
            
            # === Original label (PRESERVED) ===
            "original_response_labels": original_labels,
            
            # === Derived 3-class label ===
            "derived_label": alignment.get("derived_label", "SUPPORTED"),
        }
        
        claim_records.append(claim_record)
    
    return claim_records, errors


def main():
    """Run claim extraction on all processed data."""
    print("=" * 70)
    print("TRUVI-EV — Phase 3: Claim Extraction")
    print("=" * 70)
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Method: sentence-based (practical fallback)")
    print(f"  Note: LLM-based extraction (Qwen2.5-7B-Instruct) can be")
    print(f"  swapped in when compute is available. Recording method used.")
    print("=" * 70)
    
    DATA_CLAIMS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each split separately
    splits = ["train", "val", "test"]
    all_stats = {}
    all_errors = []
    
    for split_name in splits:
        split_path = DATA_PROCESSED_DIR / f"{split_name}.jsonl"
        if not split_path.exists():
            print(f"\n[SKIP] {split_name}.jsonl not found")
            continue
        
        print(f"\n{'='*50}")
        print(f"Processing: {split_name}")
        print(f"{'='*50}")
        
        records = load_jsonl(split_path)
        print(f"  Loaded {len(records):,} responses")
        
        all_claims = []
        split_errors = []
        
        start_time = time.time()
        
        for i, record in enumerate(records):
            claims, errors = process_single_response(record, method="sentence")
            all_claims.extend(claims)
            
            if errors:
                for e in errors:
                    split_errors.append({
                        "response_id": record.get("response_id"),
                        "split": split_name,
                        "error": e,
                    })
            
            if (i + 1) % 2000 == 0:
                elapsed = time.time() - start_time
                print(f"  Processed {i+1:,}/{len(records):,} responses "
                      f"({len(all_claims):,} claims so far, {elapsed:.1f}s)")
        
        elapsed = time.time() - start_time
        
        # Save claims for this split
        output_path = DATA_CLAIMS_DIR / f"claims_{split_name}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for claim in all_claims:
                # Don't save full source_text and original_labels in claims file
                # to keep it manageable — those are in the processed data
                save_claim = {k: v for k, v in claim.items() 
                             if k not in ("source_text", "original_response_labels")}
                f.write(json.dumps(save_claim, ensure_ascii=False) + "\n")
        
        # Statistics
        label_dist = Counter(c["derived_label"] for c in all_claims)
        claims_per_response = len(all_claims) / len(records) if records else 0
        
        stats = {
            "split": split_name,
            "input_responses": len(records),
            "output_claims": len(all_claims),
            "claims_per_response": round(claims_per_response, 2),
            "label_distribution": dict(label_dist),
            "extraction_errors": len(split_errors),
            "elapsed_seconds": round(elapsed, 1),
        }
        all_stats[split_name] = stats
        all_errors.extend(split_errors)
        
        print(f"\n  Results for {split_name}:")
        print(f"    Claims extracted: {len(all_claims):,}")
        print(f"    Claims/response:  {claims_per_response:.2f}")
        print(f"    Labels: {dict(label_dist)}")
        print(f"    Errors: {len(split_errors)}")
        print(f"    Time: {elapsed:.1f}s")
        print(f"    Saved: {output_path}")
    
    # Save errors log
    if all_errors:
        error_path = LOGS_DIR / "claim_extraction_errors.json"
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(all_errors[:500], f, indent=2)  # cap at 500
        print(f"\n[LOG] Errors saved: {error_path} ({len(all_errors)} total)")
    
    # Generate extraction summary
    summary = {
        "phase": "Phase 3 - Claim Extraction",
        "timestamp": datetime.now().isoformat(),
        "method": "sentence-based (practical fallback)",
        "method_note": "LLM-based extraction with Qwen2.5-7B-Instruct to be used when compute available",
        "splits": all_stats,
        "total_claims": sum(s["output_claims"] for s in all_stats.values()),
        "total_errors": len(all_errors),
    }
    
    summary_path = REPORTS_DIR / "claim_extraction_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    # Create manual review sample (required by supervisor)
    print("\n[REVIEW] Creating manual review sample...")
    create_review_sample(all_stats)
    
    # Final summary
    total_claims = sum(s["output_claims"] for s in all_stats.values())
    print("\n" + "=" * 70)
    print("CLAIM EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"  Total claims: {total_claims:,}")
    for split_name, stats in all_stats.items():
        print(f"  {split_name:5s}: {stats['input_responses']:,} responses -> "
              f"{stats['output_claims']:,} claims ({stats['claims_per_response']:.1f}/resp)")
    
    # Overall label distribution
    all_labels = Counter()
    for stats in all_stats.values():
        all_labels.update(stats["label_distribution"])
    print(f"\n  Overall label distribution:")
    total = sum(all_labels.values())
    for label, count in sorted(all_labels.items()):
        print(f"    {label}: {count:,} ({count/total*100:.1f}%)")
    print("=" * 70)
    
    return 0


def create_review_sample(all_stats):
    """
    Create a manual review CSV for claim extraction quality check.
    Samples a few responses and their extracted claims for human review.
    """
    import csv
    
    review_path = REPORTS_DIR / "claim_extraction_review.csv"
    
    # Load a sample from train claims
    train_claims_path = DATA_CLAIMS_DIR / "claims_train.jsonl"
    if not train_claims_path.exists():
        print("  [SKIP] No train claims to sample")
        return
    
    # Load claims
    claims = []
    with open(train_claims_path, "r") as f:
        for line in f:
            if line.strip():
                claims.append(json.loads(line))
    
    # Group by response_id, take first 10 unique responses
    from collections import defaultdict
    by_response = defaultdict(list)
    for c in claims:
        by_response[c["response_id"]].append(c)
    
    sample_ids = list(by_response.keys())[:10]
    
    # Load original responses for context
    train_path = DATA_PROCESSED_DIR / "train.jsonl"
    responses_lookup = {}
    if train_path.exists():
        with open(train_path, "r") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r["response_id"] in sample_ids:
                        responses_lookup[r["response_id"]] = r
    
    # Write review CSV
    with open(review_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "response_id", "response_text_preview",
            "num_claims", "generated_claims",
            "missed_claims", "merged_claims",
            "context_loss", "review_notes"
        ])
        
        for rid in sample_ids:
            resp = responses_lookup.get(rid, {})
            resp_text = resp.get("response_text", "")[:300] + "..."
            resp_claims = by_response[rid]
            claims_text = " | ".join(c["claim_text"][:100] for c in resp_claims[:5])
            
            writer.writerow([
                rid,
                resp_text,
                len(resp_claims),
                claims_text,
                "",  # missed_claims (for manual review)
                "",  # merged_claims (for manual review)
                "",  # context_loss (for manual review)
                "",  # review_notes (for manual review)
            ])
    
    print(f"  Saved: {review_path} ({len(sample_ids)} samples)")


if __name__ == "__main__":
    sys.exit(main())
