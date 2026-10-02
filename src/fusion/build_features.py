"""
TRUVI-EV — Feature Matrix Construction (Phase 10)
====================================================
Builds feature vectors from all computed signals for classification.

Features:
- NLI: max_entailment, max_contradiction, mean_entailment, mean_contradiction, mean_neutral (5)
- Similarity: max, mean, min, std (4)
- Reliability: top1_confidence, score_gap, mean_confidence, source_match_ratio (4)
- Agreement: nli_agreement, score_variance, entailment_ratio, contradiction_ratio (4)
Total: 17 features per claim
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

FEATURE_NAMES = [
    "nli_max_entailment", "nli_max_contradiction",
    "nli_mean_entailment", "nli_mean_contradiction", "nli_mean_neutral",
    "sim_max", "sim_mean", "sim_min", "sim_std",
    "rel_top1_confidence", "rel_score_gap", "rel_mean_confidence", "rel_source_match_ratio",
    "agr_nli_agreement", "agr_score_variance", "agr_entailment_ratio", "agr_contradiction_ratio",
]

LABEL_MAP = {"SUPPORTED": 0, "CONTRADICTED": 1, "UNVERIFIED": 2}
LABEL_NAMES = ["SUPPORTED", "CONTRADICTED", "UNVERIFIED"]


def load_jsonl(filepath):
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def extract_features(claim):
    """Extract feature vector from a claim record."""
    features = []
    for fname in FEATURE_NAMES:
        val = claim.get(fname, 0.0)
        if val is None:
            val = 0.0
        features.append(float(val))
    return features


def main():
    print("=" * 70)
    print("TRUVI-EV — Phase 10: Feature Matrix Construction")
    print("=" * 70)
    print(f"Features: {len(FEATURE_NAMES)}")
    print(f"Labels: {LABEL_NAMES}")
    print("=" * 70)

    for split in ["train", "val", "test"]:
        signals_path = DATA_PREDICTIONS_DIR / f"signals_{split}.jsonl"
        if not signals_path.exists():
            print(f"  [SKIP] {split}")
            continue

        claims = load_jsonl(signals_path)
        print(f"\n  {split}: {len(claims):,} claims")

        X = np.array([extract_features(c) for c in claims], dtype=np.float32)
        y = np.array([LABEL_MAP.get(c.get("derived_label", "SUPPORTED"), 0) for c in claims], dtype=np.int64)
        meta = [{"claim_id": c["claim_id"], "response_id": c["response_id"],
                 "derived_label": c.get("derived_label", "SUPPORTED")} for c in claims]

        # Replace NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

        # Save
        np.save(DATA_PREDICTIONS_DIR / f"X_{split}.npy", X)
        np.save(DATA_PREDICTIONS_DIR / f"y_{split}.npy", y)
        with open(DATA_PREDICTIONS_DIR / f"meta_{split}.json", "w") as f:
            json.dump(meta, f)

        print(f"    X shape: {X.shape}")
        print(f"    y distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Save feature names
    with open(DATA_PREDICTIONS_DIR / "feature_names.json", "w") as f:
        json.dump({"features": FEATURE_NAMES, "labels": LABEL_NAMES, "label_map": LABEL_MAP}, f, indent=2)

    print("\n" + "=" * 70)
    print("FEATURE MATRIX COMPLETE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
