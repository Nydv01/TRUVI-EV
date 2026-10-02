"""
TRUVI-EV — Phase 8: Evidence Reliability Signal
==================================================
Computes how reliable/trustworthy retrieved evidence is for each claim.

Does NOT invent arbitrary source trust values (e.g., Wikipedia=0.9, Blog=0.3).
Instead, reliability is derived from measurable information available in our
evidence corpus and retrieval results.

Components:
  1. Retrieval confidence — how strongly the retriever matched this evidence
  2. Source match — whether evidence comes from the claim's own source document
  3. Score gap — separation between top-1 and lower-ranked evidence
  4. NLI consistency — whether NLI gives confident (non-uniform) predictions

Formula:
  reliability = 0.3 * retrieval_confidence
              + 0.3 * source_match_ratio
              + 0.2 * score_gap_normalized
              + 0.2 * nli_confidence

All weights selected based on interpretability and equal weighting of
retrieval-based vs verification-based reliability components.
Final weights to be validated on development/validation data.

Research paper connection: Section IV-C (Evidence Reliability)
"""

import numpy as np
from typing import List, Dict, Any


def compute_reliability_signals(
    claims: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Compute evidence reliability for each claim.

    Reliability measures whether we should trust the retrieved evidence.
    High reliability = evidence is relevant, from the right source,
    well-separated from noise, and produces confident NLI predictions.

    Per-claim fields added:
      rel_top1_confidence  — retrieval score of top-1 evidence
      rel_score_gap        — gap between top-1 and top-2 scores
      rel_mean_confidence  — mean retrieval score across top-k
      rel_source_match     — count of evidence from claim's source
      rel_source_match_ratio — fraction of evidence from claim's source
      rel_nli_confidence   — average NLI prediction confidence
      evidence_reliability — combined reliability score [0, 1]

    Args:
        claims: List of claim dicts with retrieved_evidence (after NLI)

    Returns:
        claims with reliability signals added
    """
    for claim in claims:
        evidence = claim.get("retrieved_evidence", [])

        if not evidence:
            _set_reliability_defaults(claim)
            continue

        scores = [e.get("score", 0.0) for e in evidence]

        # Component 1: Retrieval confidence (top-1 score, already in [0,1] range)
        top1_confidence = float(scores[0])
        claim["rel_top1_confidence"] = top1_confidence

        # Component 2: Score gap (top-1 vs top-2)
        if len(scores) > 1:
            score_gap = float(scores[0] - scores[1])
        else:
            score_gap = 0.0
        claim["rel_score_gap"] = score_gap

        # Component 3: Mean retrieval confidence
        mean_confidence = float(np.mean(scores))
        claim["rel_mean_confidence"] = mean_confidence

        # Component 4: Source match
        claim_source = claim.get("source_id", "")
        source_matches = sum(
            1 for e in evidence
            if e.get("source_id", "") == claim_source
        )
        claim["rel_source_match"] = source_matches
        source_match_ratio = source_matches / len(evidence)
        claim["rel_source_match_ratio"] = float(source_match_ratio)

        # Component 5: NLI prediction confidence
        # High confidence = model is decisive (one class dominates)
        # Low confidence = uniform distribution (≈0.33 each)
        nli_confidences = []
        for ev in evidence:
            ent = ev.get("nli_entailment", 0.333)
            con = ev.get("nli_contradiction", 0.333)
            neu = ev.get("nli_neutral", 0.333)
            # Max probability as confidence measure
            max_prob = max(ent, con, neu)
            nli_confidences.append(max_prob)

        nli_confidence = float(np.mean(nli_confidences))
        claim["rel_nli_confidence"] = nli_confidence

        # Combined reliability score
        # Normalize score_gap to [0, 1] range (clamp to reasonable max)
        score_gap_norm = min(score_gap / 0.3, 1.0)  # 0.3 gap = full reliability

        reliability = (
            0.30 * top1_confidence +
            0.30 * source_match_ratio +
            0.20 * score_gap_norm +
            0.20 * nli_confidence
        )
        claim["evidence_reliability"] = float(np.clip(reliability, 0.0, 1.0))

    return claims


def _set_reliability_defaults(claim: Dict[str, Any]):
    """Set default reliability values when no evidence is available."""
    claim["rel_top1_confidence"] = 0.0
    claim["rel_score_gap"] = 0.0
    claim["rel_mean_confidence"] = 0.0
    claim["rel_source_match"] = 0
    claim["rel_source_match_ratio"] = 0.0
    claim["rel_nli_confidence"] = 0.0
    claim["evidence_reliability"] = 0.0
