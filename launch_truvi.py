#!/usr/bin/env python3
"""
TRUVI-EV Launcher & CLI Tool
============================
- Run without arguments: Launches the TRUVI Studio Desktop Panel & Clipboard HUD.
- Run with text argument: Instantly verifies the claim/paragraph in your terminal with
  full contradiction pinpointing, proving resource citations, and ground-truth corrections!
"""

import os
# Prevent tokenizer multi-threading deadlock & macOS segmentation faults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_cli_report(res: dict):
    is_para = "paragraph" in res
    v = res.get("overall_verdict" if is_para else "verdict", "UNVERIFIED")
    conf = res.get("overall_confidence_pct" if is_para else "confidence_pct", "")
    adv = res.get("action_advisory", "")
    rsn = res.get("overall_summary" if is_para else "short_reason", "")

    if v == "CONTRADICTED":
        badge = "❌ CONTRADICTED / HALLUCINATION DETECTED"
    elif v == "SUPPORTED":
        badge = "✅ SUPPORTED / VERIFIED FACTUAL"
    else:
        badge = "⚠️ UNVERIFIED / INSUFFICIENT EVIDENCE"

    print("\n" + "=" * 76)
    print("  TRUVI-EV FACT-CHECK AUDIT & FORENSIC CORRECTION REPORT")
    print("=" * 76)
    print(f"  Input:       \"{res.get('paragraph' if is_para else 'claim', '')}\"")
    print(f"  Verdict:     {badge} ({conf} Calibrated Confidence)")

    fact_pct = res.get("factuality_pct")
    hall_pct = res.get("hallucination_pct") or res.get("hallucination_severity_pct")
    if fact_pct or hall_pct:
        print(f"  Composition: 🟢 {fact_pct or '0%'} Factual  |  🔴 {hall_pct or '0%'} Hallucinated")

    print(f"  Advisory:    {adv}")
    print(f"  Reason:      {rsn}")

    # Forensic Contradiction Details (Single or Compound)
    if not is_para and v == "CONTRADICTED":
        c_part = res.get("contradicted_part", "")
        resrc = res.get("proving_resource", {}) or {}
        r_stmt = res.get("right_statement", "")

        print("\n  🚨 CONTRADICTION FORENSICS:")
        if c_part:
            print(f"  • Contradicted Part: \"{c_part}\"")
        if resrc:
            print(f"  • Proving Resource:  {resrc.get('source')} [{resrc.get('domain')}] ({resrc.get('authority_pct')} Authority)")
            print(f"  • Verbatim Evidence: \"{resrc.get('evidence_text')}\"")
        if r_stmt:
            print(f"\n  🟢 100% VERIFIED STATEMENT (USE THIS INSTEAD):")
            print(f"  \"{r_stmt}\"")

    # Sub-assertion Breakdown for Compound claims
    sub_assertions = res.get("sub_assertions") or res.get("sub_claims", [])
    if not is_para and len(sub_assertions) > 1:
        print("\n  🔬 ATOMIC ASSERTION FACTUALITY BREAKDOWN:")
        for sc in sub_assertions:
            txt = sc.get("assertion_text") or sc.get("claim")
            sc_v = sc.get("verdict", "UNVERIFIED")
            sc_ic = "✅" if sc_v == "SUPPORTED" else ("❌" if sc_v == "CONTRADICTED" else "⚠️")
            print(f"    {sc_ic} [{sc_v}] \"{txt}\"")
            if sc_v == "CONTRADICTED":
                pr = sc.get("proving_resource", {}) or {}
                if pr.get("source"):
                    print(f"       ↳ Refuted by: {pr.get('source')} ({pr.get('authority_pct', '99%')} Authority)")
                if sc.get("right_statement"):
                    print(f"       ↳ Right Statement: \"{sc.get('right_statement')}\"")

    # Paragraph Contradictions Breakdown & Full Correction
    if is_para:
        contradictions = res.get("contradictions", [])
        if contradictions:
            print(f"\n  🚨 CONTRADICTION BREAKDOWN ({len(contradictions)} CONTRADICTED CLAIMS):")
            for c in contradictions:
                resrc = c.get("proving_resource", {}) or {}
                f_pct = c.get("factuality_pct", "0.0%")
                h_pct = c.get("hallucination_pct", "100.0%")
                print(f"\n    [Claim #{c.get('claim_num')}] Original: \"{c.get('original_claim')}\"")
                print(f"    • Claim Composition: 🟢 {f_pct} Factual | 🔴 {h_pct} Hallucinated")
                print(f"    • Contradicted Part: \"{c.get('contradicted_part')}\"")
                print(f"    • Proving Resource:  {resrc.get('source')} ({resrc.get('authority_pct', '99%')} Authority)")
                print(f"    • Evidence Proof:    \"{resrc.get('evidence_text')}\"")
                print(f"    • Right Statement:   \"{c.get('right_statement')}\"")

        corr_para = res.get("corrected_paragraph", "")
        if corr_para:
            print("\n" + "-" * 76)
            print("  🟢 FULLY CORRECTED 100% FACTUAL PARAGRAPH (READY TO USE):")
            print("-" * 76)
            print(f"  \"{corr_para}\"")

    print("=" * 76 + "\n")


def main():
    args = sys.argv[1:]

    # CLI Fact-Check Mode
    if args and not (len(args) == 1 and args[0] in ["--gui", "-g"]):
        claim_input = " ".join(args).strip()
        print(f"\n[TRUVI-EV] Verifying via multi-signal pipeline: \"{claim_input}\"...")
        from src.engine import get_engine
        eng = get_engine()

        has_multiple_sentences = (
            "\n" in claim_input or
            len([s for s in claim_input.split(".") if len(s.strip()) > 10]) > 1
        )
        if has_multiple_sentences:
            res = eng.verify_paragraph(claim_input)
        else:
            res = eng.verify_claim(claim_input)

        print_cli_report(res)
        return

    # Native Desktop App Mode
    print("\n" + "=" * 60)
    print("  TRUVI-EV: Trustworthy Evidence-Aware Verifier")
    print("  Launching TRUVI Studio Desktop Panel & Clipboard HUD...")
    print("=" * 60 + "\n")
    from desktop_app import main as desktop_main
    desktop_main()


if __name__ == "__main__":
    main()
