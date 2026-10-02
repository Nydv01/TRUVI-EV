"""
TRUVI-EV — End-to-End Pipeline
================================
Runs all phases sequentially after retrieval and signals are computed.

Phase 10: Feature matrix construction
Phase 11-13: Baselines + TRUVI-EV training
Phase 15: Ablation study
Phase 16: Error analysis
Phase 17: Figure generation
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_module(module_name, description):
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Module:  {module_name}")
    print(f"{'='*70}")
    result = subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=str(PROJECT_ROOT),
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"[ERROR] {module_name} failed with code {result.returncode}")
        return False
    return True


def main():
    steps = [
        ("src.fusion.build_features", "Phase 10: Feature Matrix Construction"),
        ("src.fusion.train", "Phases 11-13: Baselines + TRUVI-EV Training"),
        ("src.evaluation.ablation", "Phase 15: Ablation Study"),
        ("src.evaluation.error_analysis", "Phase 16: Error Analysis"),
        ("src.evaluation.figures", "Phase 17: Figure Generation"),
    ]

    for module, desc in steps:
        success = run_module(module, desc)
        if not success:
            print(f"\n[STOPPED] Pipeline stopped at: {desc}")
            return 1

    print("\n" + "=" * 70)
    print("ALL PHASES COMPLETE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
