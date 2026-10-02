"""
TRUVI-EV — Phase 6: NLI Signal Computation
=============================================
Uses cross-encoder/nli-deberta-v3-base to compute NLI probabilities
for every (claim, evidence) pair.

Model: https://huggingface.co/cross-encoder/nli-deberta-v3-base
Output order: [contradiction, entailment, neutral]

Stores:
  Per-evidence: nli_contradiction, nli_entailment, nli_neutral
  Per-claim:    nli_max_entailment, nli_max_contradiction,
                nli_mean_entailment, nli_mean_contradiction, nli_mean_neutral,
                nli_predicted_label

IMPORTANT: NLI is only ONE signal. It does NOT directly become the final
Supported/Contradicted/Unverified label.

Research paper connection: Section IV-C (NLI-based Verification Signal)
"""

import numpy as np
from typing import List, Dict, Any

# Model identifier — official HuggingFace model
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

# DeBERTa NLI output class order (from model card)
# Index 0: contradiction
# Index 1: entailment
# Index 2: neutral
NLI_LABEL_MAP = {0: "contradiction", 1: "entailment", 2: "neutral"}


def load_nli_model(device: str = None):
    """
    Load the NLI cross-encoder model.

    Args:
        device: 'mps', 'cuda', 'cpu', or None for auto-detect

    Returns:
        CrossEncoder model instance
    """
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(NLI_MODEL_NAME)

    # Set device if specified
    if device:
        import torch
        model.model.to(torch.device(device))

    return model


def compute_nli_for_pairs(
    pairs: List[tuple],
    nli_model,
    batch_size: int = 128,
    sub_batch_size: int = 1000,
) -> np.ndarray:
    """
    Compute NLI probabilities for a list of (premise, hypothesis) pairs.

    Args:
        pairs: List of (claim_text, evidence_text) tuples
        nli_model: Loaded CrossEncoder model
        batch_size: Batch size for model.predict()
        sub_batch_size: Process pairs in sub-batches to limit memory

    Returns:
        numpy array of shape (N, 3) with [contradiction, entailment, neutral] probs
    """
    if not pairs:
        return np.array([]).reshape(0, 3)

    all_probs = []
    for start in range(0, len(pairs), sub_batch_size):
        batch = pairs[start:start + sub_batch_size]
        # CrossEncoder.predict() returns raw logits
        raw_logits = nli_model.predict(
            batch,
            batch_size=batch_size,
            show_progress_bar=False
        )

        # Apply softmax to convert logits -> probabilities
        raw_logits = np.array(raw_logits)
        if raw_logits.ndim == 1:
            raw_logits = raw_logits.reshape(1, -1)

        exp_scores = np.exp(raw_logits - np.max(raw_logits, axis=1, keepdims=True))
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        all_probs.append(probs)

    return np.vstack(all_probs)


def compute_nli_for_claims(
    claims: List[Dict[str, Any]],
    nli_model,
    batch_size: int = 128,
) -> List[Dict[str, Any]]:
    """
    Compute NLI scores for a batch of claims with their retrieved evidence.

    For each claim:
      - Runs NLI on every (claim, evidence) pair
      - Stores per-evidence probabilities
      - Computes claim-level aggregates

    Args:
        claims: List of claim dicts with 'retrieved_evidence' field
        nli_model: Loaded CrossEncoder model
        batch_size: Batch size for NLI inference

    Returns:
        claims with NLI scores added
    """
    # Build flat list of (claim, evidence) pairs with mapping back to claims
    pairs = []
    pair_map = []  # (claim_index, evidence_index)

    for i, claim in enumerate(claims):
        for j, ev in enumerate(claim.get("retrieved_evidence", [])):
            text_a = claim["claim_text"]
            text_b = ev.get("chunk_text", "")
            if text_a and text_b:
                pairs.append((text_a, text_b))
                pair_map.append((i, j))

    if not pairs:
        # No pairs to process — set defaults
        for claim in claims:
            _set_nli_defaults(claim)
        return claims

    # Run NLI inference
    probs = compute_nli_for_pairs(pairs, nli_model, batch_size=batch_size)

    # Assign per-evidence scores
    for idx, (ci, ei) in enumerate(pair_map):
        ev = claims[ci]["retrieved_evidence"][ei]
        score = probs[idx]
        ev["nli_contradiction"] = float(score[0])
        ev["nli_entailment"] = float(score[1])
        ev["nli_neutral"] = float(score[2])
        # Per-evidence predicted label
        ev["nli_predicted_label"] = NLI_LABEL_MAP[int(np.argmax(score))]

    # Compute claim-level aggregates
    for claim in claims:
        evidence = claim.get("retrieved_evidence", [])
        if evidence and any("nli_entailment" in e for e in evidence):
            ent_scores = [e.get("nli_entailment", 0.0) for e in evidence]
            con_scores = [e.get("nli_contradiction", 0.0) for e in evidence]
            neu_scores = [e.get("nli_neutral", 0.0) for e in evidence]

            claim["nli_max_entailment"] = float(max(ent_scores))
            claim["nli_max_contradiction"] = float(max(con_scores))
            claim["nli_mean_entailment"] = float(np.mean(ent_scores))
            claim["nli_mean_contradiction"] = float(np.mean(con_scores))
            claim["nli_mean_neutral"] = float(np.mean(neu_scores))

            # Claim-level predicted NLI label (from the evidence with
            # highest max probability — NOT final verdict)
            all_maxes = [
                max(e.get("nli_entailment", 0), e.get("nli_contradiction", 0), e.get("nli_neutral", 0))
                for e in evidence
            ]
            best_ev_idx = int(np.argmax(all_maxes))
            best_ev = evidence[best_ev_idx]
            claim["nli_predicted_label"] = best_ev.get("nli_predicted_label", "neutral")
        else:
            _set_nli_defaults(claim)

    return claims


def _set_nli_defaults(claim: Dict[str, Any]):
    """Set default NLI values when no evidence is available."""
    claim["nli_max_entailment"] = 0.0
    claim["nli_max_contradiction"] = 0.0
    claim["nli_mean_entailment"] = 0.0
    claim["nli_mean_contradiction"] = 0.0
    claim["nli_mean_neutral"] = 0.0
    claim["nli_predicted_label"] = "neutral"
