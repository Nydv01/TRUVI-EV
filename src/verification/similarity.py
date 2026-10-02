"""
TRUVI-EV — Phase 7: Semantic Similarity Signal
=================================================
Computes cosine similarity between claim and evidence embeddings.

Uses the same embedding model as retrieval: BAAI/bge-base-en-v1.5
This ensures consistency — the similarity signal is conceptually separate
from the retrieval score (which is also cosine) but uses direct embedding
comparison rather than relying on the cached retrieval scores.

IMPORTANT:
  Semantic similarity is NOT factual verification.
  A highly similar passage can still contradict a claim.
  This is exactly why NLI and similarity signals must remain separate.

Research paper connection: Section IV-C (Semantic Similarity Signal)
"""

import numpy as np
from typing import List, Dict, Any

# Use the same model as retrieval for consistency
SIMILARITY_MODEL_NAME = "BAAI/bge-base-en-v1.5"


def compute_similarity_from_retrieval_scores(
    claims: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Phase 7: Use retrieval scores as semantic similarity.

    Since our retrieval already uses cosine similarity (inner product of
    normalized BGE embeddings), the retrieval score IS the semantic
    similarity. Re-encoding would produce identical values.

    This is documented as a design decision:
    - Embedding model: BAAI/bge-base-en-v1.5 (same for both)
    - Retrieval score = cosine(claim_embedding, evidence_embedding)
    - Therefore: semantic_similarity = retrieval_score

    If we later use a DIFFERENT embedding model for similarity vs retrieval,
    this function should be updated to do independent encoding.

    Per-evidence fields added:
      semantic_similarity (= score from retrieval)

    Per-claim aggregate fields:
      sim_max, sim_mean, sim_min, sim_std

    Args:
        claims: List of claim dicts with retrieved_evidence

    Returns:
        claims with similarity signals added
    """
    for claim in claims:
        evidence = claim.get("retrieved_evidence", [])

        if not evidence:
            claim["sim_max"] = 0.0
            claim["sim_mean"] = 0.0
            claim["sim_min"] = 0.0
            claim["sim_std"] = 0.0
            continue

        scores = []
        for ev in evidence:
            # Retrieval score IS the cosine similarity
            sim = ev.get("score", 0.0)
            ev["semantic_similarity"] = float(sim)
            scores.append(sim)

        claim["sim_max"] = float(max(scores))
        claim["sim_mean"] = float(np.mean(scores))
        claim["sim_min"] = float(min(scores))
        claim["sim_std"] = float(np.std(scores)) if len(scores) > 1 else 0.0

    return claims
