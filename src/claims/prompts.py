"""
TRUVI-EV — Claim Extraction Prompts
=====================================
Prompt templates for extracting factual claims from LLM responses.

Used by: src/claims/extractor.py
"""

CLAIM_EXTRACTION_SYSTEM = """You are a precise factual claim extractor. Your job is to break down a text into individual, independent factual claims.

Rules:
1. Each claim must be ONE atomic factual statement
2. Each claim must be independently verifiable
3. Preserve necessary context (names, dates, locations)
4. Do NOT include opinions, subjective statements, or hedging language
5. Do NOT combine multiple facts into one claim
6. Do NOT invent facts not present in the text
7. Do NOT include meta-statements like "the article says" or "according to"
8. Output valid JSON only"""

CLAIM_EXTRACTION_USER = """Extract all factual claims from the following text. Return ONLY valid JSON.

Text:
{response_text}

Return JSON in this exact format:
{{
  "claims": [
    {{"claim_id": "c1", "claim": "First factual claim here."}},
    {{"claim_id": "c2", "claim": "Second factual claim here."}}
  ]
}}"""

# Simpler prompt for smaller models
CLAIM_EXTRACTION_SIMPLE = """Extract factual claims from this text as JSON.

Text: {response_text}

Output format: {{"claims": [{{"claim_id": "c1", "claim": "..."}}, ...]}}
Only factual statements. One fact per claim. Valid JSON only."""
