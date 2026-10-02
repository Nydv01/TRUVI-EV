"""
TRUVI-EV — Phase 9: Evidence Agreement Signal
================================================
Measures consistency/agreement among top-k retrieved evidence passages.

For each claim, NLI is run on every retrieved passage. Agreement captures
whether the evidence tells a consistent story or conflicts.

High agreement example:
  e1 → entailment, e2 → entailment, e3 → entailment, e4 → neutral, e5 → entailment
  → Strong agreement toward entailment

Low agreement example:
  e1 → entailment, e2 → contradiction, e3 → neutral, e4 → contradiction, e5 → entailment
  → Conflicting evidence

Agreement metric: Maximum class proportion (majority ratio)
  agreement = max(count_entailment, count_contradiction, count_neutral) / k

This is simple, interpretable, and documented. Alternative: entropy-based.
We chose majority ratio for Phase 1 experiments.

Research paper connection: Section IV-C (Evidence Agreement)
"""

import numpy as np
from typing import List, Dict, Any
from collections import Counter


def compute_agreement_signals(
    claims: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Compute evidence agreement for each claim.

    Agreement measures how consistently the top-k evidence passages
    agree on the NLI verdict for a claim.

    Per-claim fields added:
      agr_nli_agreement       — majority class proportion [0, 1]
      agr_majority_label      — most common NLI label among evidence
      agr_majority_count      — count of majority label
      agr_unanimous           — whether all evidence agrees (bool)
      agr_entailment_ratio    — fraction of evidence predicting entailment
      agr_contradiction_ratio — fraction predicting contradiction
      agr_score_variance      — variance in NLI scores (higher = more conflict)
      evidence_agreement      — final agreement score [0, 1]

    Args:
        claims: List of claim dicts with retrieved_evidence (after NLI)

    Returns:
        claims with agreement signals added
    """
    for claim in claims:
        evidence = claim.get("retrieved_evidence", [])

        if len(evidence) < 2:
            _set_agreement_defaults(claim, evidence)
            continue

        # Determine NLI label for each evidence passage
        labels = []
        for ev in evidence:
            ent = ev.get("nli_entailment", 0.0)
            con = ev.get("nli_contradiction", 0.0)
            neu = ev.get("nli_neutral", 0.0)

            if ent >= con and ent >= neu:
                labels.append("entailment")
            elif con >= ent and con >= neu:
                labels.append("contradiction")
            else:
                labels.append("neutral")

        # Majority voting
        label_counts = Counter(labels)
        majority_label, majority_count = label_counts.most_common(1)[0]

        # Agreement = majority proportion
        agreement = majority_count / len(labels)

        # Class-specific ratios
        ent_count = sum(1 for l in labels if l == "entailment")
        con_count = sum(1 for l in labels if l == "contradiction")

        # Score variance (measures conflict intensity)
        ent_scores = [ev.get("nli_entailment", 0.0) for ev in evidence]
        con_scores = [ev.get("nli_contradiction", 0.0) for ev in evidence]
        score_variance = float(np.var(ent_scores) + np.var(con_scores))

        # Store signals
        claim["agr_nli_agreement"] = float(agreement)
        claim["agr_majority_label"] = majority_label
        claim["agr_majority_count"] = majority_count
        claim["agr_unanimous"] = bool(majority_count == len(labels))
        claim["agr_entailment_ratio"] = float(ent_count / len(labels))
        claim["agr_contradiction_ratio"] = float(con_count / len(labels))
        claim["agr_score_variance"] = score_variance

        # Final agreement score = majority proportion
        # (documented choice; entropy-based is alternative)
        claim["evidence_agreement"] = float(agreement)

    return claims


def _set_agreement_defaults(claim: Dict[str, Any], evidence: list):
    """Set default agreement values when < 2 evidence passages."""
    claim["agr_nli_agreement"] = 0.0
    claim["agr_majority_label"] = "neutral"
    claim["agr_majority_count"] = len(evidence)
    claim["agr_unanimous"] = len(evidence) <= 1
    claim["agr_entailment_ratio"] = 0.0
    claim["agr_contradiction_ratio"] = 0.0
    claim["agr_score_variance"] = 0.0
    claim["evidence_agreement"] = 0.0

    # If there's exactly 1 evidence, derive from its NLI
    if len(evidence) == 1:
        ev = evidence[0]
        ent = ev.get("nli_entailment", 0.0)
        con = ev.get("nli_contradiction", 0.0)
        neu = ev.get("nli_neutral", 0.0)

        if ent >= con and ent >= neu:
            claim["agr_majority_label"] = "entailment"
            claim["agr_entailment_ratio"] = 1.0
        elif con >= ent and con >= neu:
            claim["agr_majority_label"] = "contradiction"
            claim["agr_contradiction_ratio"] = 1.0
        else:
            claim["agr_majority_label"] = "neutral"

        claim["agr_nli_agreement"] = 1.0  # Only 1 passage, trivially "agrees"
        claim["evidence_agreement"] = 1.0
