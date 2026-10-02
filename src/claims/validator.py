"""
TRUVI-EV — Claim Validator
============================
Validates and repairs extracted claim JSON.

Handles:
- JSON parsing with repair for common LLM output errors
- Claim deduplication
- Empty/invalid claim filtering
- Span alignment with original hallucination annotations
- 3-class label derivation (SUPPORTED / CONTRADICTED / UNVERIFIED)
"""

import re
import json


def parse_claims_json(raw_text):
    """
    Parse claims JSON from raw model output.
    Handles common issues: markdown fencing, trailing commas, etc.
    
    Returns:
        list of claim dicts, or empty list on failure
    """
    if not raw_text or not raw_text.strip():
        return []
    
    text = raw_text.strip()
    
    # Remove markdown code fences
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    
    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "claims" in data:
            return data["claims"]
        if isinstance(data, list):
            return data
        return []
    except json.JSONDecodeError:
        pass
    
    # Try fixing trailing commas
    fixed = re.sub(r',\s*([}\]])', r'\1', text)
    try:
        data = json.loads(fixed)
        if isinstance(data, dict) and "claims" in data:
            return data["claims"]
        if isinstance(data, list):
            return data
        return []
    except json.JSONDecodeError:
        pass
    
    # Try extracting JSON object from surrounding text
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            candidate = match.group()
            candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
            data = json.loads(candidate)
            if isinstance(data, dict) and "claims" in data:
                return data["claims"]
        except json.JSONDecodeError:
            pass
    
    return []


def validate_claim(claim_dict, index):
    """
    Validate a single claim dict.
    
    Returns:
        (is_valid, cleaned_claim_dict, error_message)
    """
    if not isinstance(claim_dict, dict):
        return False, None, f"Claim {index}: not a dict"
    
    # Get claim text
    claim_text = claim_dict.get("claim", "")
    if not claim_text or not claim_text.strip():
        return False, None, f"Claim {index}: empty claim text"
    
    claim_text = claim_text.strip()
    
    # Too short (likely not a real claim)
    if len(claim_text) < 5:
        return False, None, f"Claim {index}: too short ({len(claim_text)} chars)"
    
    # Too long (likely multiple claims merged)
    if len(claim_text) > 500:
        claim_text = claim_text[:500]  # truncate but keep
    
    # Get or generate claim_id
    claim_id = claim_dict.get("claim_id", f"c{index}")
    
    cleaned = {
        "claim_id": str(claim_id),
        "claim": claim_text,
    }
    
    return True, cleaned, None


def validate_claims_batch(raw_claims):
    """
    Validate a batch of claims.
    
    Returns:
        (valid_claims, errors)
    """
    valid = []
    errors = []
    seen_texts = set()
    
    for i, claim in enumerate(raw_claims, 1):
        is_valid, cleaned, error = validate_claim(claim, i)
        
        if not is_valid:
            errors.append(error)
            continue
        
        # Deduplication
        norm_text = cleaned["claim"].lower().strip()
        if norm_text in seen_texts:
            errors.append(f"Claim {i}: duplicate of existing claim")
            continue
        seen_texts.add(norm_text)
        
        valid.append(cleaned)
    
    return valid, errors


def align_claim_with_spans(claim_text, response_text, hallucination_spans):
    """
    Align an extracted claim with the original response text and
    check if it overlaps with any hallucination span.
    
    Strategy:
    1. Find where the claim text (or a close match) appears in the response
    2. Check if that region overlaps with any hallucination span
    3. Determine the derived label based on overlap
    
    Returns:
        dict with alignment info and derived label
    """
    if not hallucination_spans:
        return {
            "aligned": True,
            "claim_start": None,
            "claim_end": None,
            "overlapping_spans": [],
            "derived_label": "SUPPORTED",
            "overlap_label_types": [],
        }
    
    # Try to find claim in response
    claim_lower = claim_text.lower().strip()
    response_lower = response_text.lower()
    
    # Direct substring match
    idx = response_lower.find(claim_lower)
    
    if idx == -1:
        # Try matching significant words (at least 60% overlap)
        claim_words = set(claim_lower.split())
        best_overlap = 0
        best_span_overlaps = []
        
        for span in hallucination_spans:
            span_text = span.get("text", "").lower()
            span_words = set(span_text.split())
            
            if not claim_words or not span_words:
                continue
            
            overlap = len(claim_words & span_words) / len(claim_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_span_overlaps = [span]
            elif overlap == best_overlap and overlap > 0.3:
                best_span_overlaps.append(span)
        
        if best_overlap >= 0.3 and best_span_overlaps:
            label_types = [s.get("label_type", "") for s in best_span_overlaps]
            derived = _derive_label_from_types(label_types)
            return {
                "aligned": True,
                "claim_start": None,
                "claim_end": None,
                "overlapping_spans": best_span_overlaps,
                "derived_label": derived,
                "overlap_label_types": label_types,
                "alignment_method": "word_overlap",
                "overlap_score": round(best_overlap, 3),
            }
        
        # No overlap found — treat as SUPPORTED
        return {
            "aligned": False,
            "claim_start": None,
            "claim_end": None,
            "overlapping_spans": [],
            "derived_label": "SUPPORTED",
            "overlap_label_types": [],
            "alignment_method": "no_match",
        }
    
    # Found exact match — check span overlaps
    claim_start = idx
    claim_end = idx + len(claim_text)
    
    overlapping = []
    for span in hallucination_spans:
        span_start = span.get("start", 0)
        span_end = span.get("end", 0)
        
        # Check overlap
        if claim_start < span_end and claim_end > span_start:
            overlapping.append(span)
    
    if overlapping:
        label_types = [s.get("label_type", "") for s in overlapping]
        derived = _derive_label_from_types(label_types)
    else:
        derived = "SUPPORTED"
        label_types = []
    
    return {
        "aligned": True,
        "claim_start": claim_start,
        "claim_end": claim_end,
        "overlapping_spans": overlapping,
        "derived_label": derived,
        "overlap_label_types": label_types,
        "alignment_method": "exact_match",
    }


def _derive_label_from_types(label_types):
    """
    Derive 3-class label from hallucination span label types.
    
    Mapping (documented in Phase 1 report):
    - Evident Conflict / Subtle Conflict → CONTRADICTED
    - Evident Baseless Info / Subtle Baseless Info → UNVERIFIED
    - If both conflict and baseless → CONTRADICTED (stronger signal)
    """
    has_conflict = any("Conflict" in lt for lt in label_types)
    has_baseless = any("Baseless" in lt for lt in label_types)
    
    if has_conflict:
        return "CONTRADICTED"
    elif has_baseless:
        return "UNVERIFIED"
    else:
        return "UNVERIFIED"  # default for unknown hallucination types
