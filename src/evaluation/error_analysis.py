"""
TRUVI-EV — Error Analysis (Phase 16)
======================================
Analyzes model errors to understand failure patterns.

Outputs:
  - error_analysis.json           (summary statistics)
  - error_analysis_candidates.csv (detailed error records per master prompt Section 31)
"""

import sys
import json
import csv
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Import the TRUVI-EV model architecture
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fusion.train import ReliabilityGatedMLP

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

SEED = 42
LABEL_NAMES = ["SUPPORTED", "CONTRADICTED", "UNVERIFIED"]


def main():
    print("=" * 70)
    print("TRUVI-EV — Phase 16: Error Analysis")
    print("=" * 70)

    X_train = np.load(DATA_PREDICTIONS_DIR / "X_train.npy")
    y_train = np.load(DATA_PREDICTIONS_DIR / "y_train.npy")
    X_val = np.load(DATA_PREDICTIONS_DIR / "X_val.npy")
    y_val = np.load(DATA_PREDICTIONS_DIR / "y_val.npy")

    with open(DATA_PREDICTIONS_DIR / "meta_val.json") as f:
        meta_val = json.load(f)
    with open(DATA_PREDICTIONS_DIR / "feature_names.json") as f:
        finfo = json.load(f)
    feature_names = finfo["features"]

    # Load signals data for richer error records
    signals_path = DATA_PREDICTIONS_DIR / "signals_val.jsonl"
    signals_data = {}
    if signals_path.exists():
        with open(signals_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    signals_data[rec["claim_id"]] = rec

    # Train TRUVI-EV model (same as train.py)
    print("\n[1/3] Training TRUVI-EV for error analysis...")
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    classes = np.unique(y_train)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_v = scaler.transform(X_val)

    model = ReliabilityGatedMLP(17, (128, 64, 32), 3, 0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(cw.tolist()).to(device))

    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.LongTensor(y_train).to(device)

    best_f1 = -1
    best_state = None
    no_improve = 0

    for epoch in range(200):
        model.train()
        perm = torch.randperm(len(X_tr_t))
        for i in range(0, len(X_tr_t), 256):
            idx = perm[i:i+256]
            logits, _ = model(X_tr_t[idx])
            loss = criterion(logits, y_tr_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, val_gates = model(torch.FloatTensor(X_v).to(device))
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            from sklearn.metrics import f1_score
            val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 20:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_logits, val_gates = model(torch.FloatTensor(X_v).to(device))
        y_pred = val_logits.argmax(dim=1).cpu().numpy()
        probs = torch.softmax(val_logits, dim=1).cpu().numpy()
        gate_vals = val_gates.cpu().numpy()

    # [2/3] Classification report
    print("\n[2/3] Analysis...")
    report = classification_report(y_val, y_pred, target_names=LABEL_NAMES, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=LABEL_NAMES))

    cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])
    print("Confusion Matrix:")
    print(f"  {'':>15s} {'Pred_S':>8s} {'Pred_C':>8s} {'Pred_U':>8s}")
    for i, label in enumerate(LABEL_NAMES):
        print(f"  {'True_'+label:>15s} {cm[i][0]:>8d} {cm[i][1]:>8d} {cm[i][2]:>8d}")

    # Error analysis
    errors = []
    for i in range(len(y_val)):
        if y_val[i] != y_pred[i]:
            claim_id = meta_val[i]["claim_id"] if i < len(meta_val) else ""
            sig = signals_data.get(claim_id, {})
            evidence = sig.get("retrieved_evidence", [])

            err = {
                "claim_id": claim_id,
                "claim_text": sig.get("claim_text", ""),
                "evidence_text": evidence[0]["chunk_text"][:300] if evidence else "",
                "ground_truth": LABEL_NAMES[y_val[i]],
                "prediction": LABEL_NAMES[y_pred[i]],
                "confidence": round(float(probs[i].max()), 4),
                "nli_entailment": round(float(X_val[i, 2]), 4),
                "nli_contradiction": round(float(X_val[i, 3]), 4),
                "similarity": round(float(X_val[i, 6]), 4),
                "retrieval_score": round(float(X_val[i, 9]), 4),
                "reliability": round(float(X_val[i, 11]), 4),
                "agreement": round(float(X_val[i, 13]), 4),
                # Blank columns for manual annotation
                "error_category": "",  # To be filled: retrieval_failure, nli_error, etc.
            }
            errors.append(err)

    error_types = Counter((e["ground_truth"], e["prediction"]) for e in errors)
    print(f"\nTotal errors: {len(errors)} / {len(y_val)} ({len(errors)/len(y_val)*100:.1f}%)")
    print("\nError Type Distribution:")
    for (true_l, pred_l), count in error_types.most_common():
        print(f"  {true_l} -> {pred_l}: {count} ({count/len(errors)*100:.1f}%)")

    # Feature analysis
    print("\nFeature Analysis (mean values — correct vs error):")
    correct_mask = y_val == y_pred
    error_mask = ~correct_mask
    print(f"  {'Feature':<30s} {'Correct':>10s} {'Error':>10s} {'Diff':>10s}")
    print("  " + "-" * 60)
    for j, fname in enumerate(feature_names):
        corr_mean = np.mean(X_val[correct_mask, j])
        err_mean = np.mean(X_val[error_mask, j]) if error_mask.sum() > 0 else 0
        diff = err_mean - corr_mean
        print(f"  {fname:<30s} {corr_mean:>10.4f} {err_mean:>10.4f} {diff:>+10.4f}")

    # [3/3] Save
    print("\n[3/3] Saving error analysis...")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Error analysis candidates CSV (per master prompt Section 31)
    csv_path = TABLES_DIR / "error_analysis_candidates.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "claim_id", "claim_text", "evidence_text",
            "ground_truth", "prediction", "confidence",
            "nli_entailment", "nli_contradiction", "similarity",
            "retrieval_score", "reliability", "agreement",
            "error_category",
        ])
        writer.writeheader()
        for e in errors[:500]:  # Top 500 errors for review
            writer.writerow(e)
    print(f"  Saved: {csv_path} ({min(len(errors), 500)} candidates)")

    # Error type summary CSV
    csv_path2 = TABLES_DIR / "error_analysis.csv"
    with open(csv_path2, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Error_Type", "Count", "Percentage"])
        for (t, p), c in error_types.most_common():
            writer.writerow([f"{t}->{p}", c, round(c/len(errors)*100, 1)])

    # JSON summary
    error_summary = {
        "phase": "Phase 16 - Error Analysis",
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(y_val),
        "total_errors": len(errors),
        "error_rate": round(len(errors) / len(y_val) * 100, 2),
        "error_types": {f"{t}->{p}": c for (t, p), c in error_types.most_common()},
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    with open(REPORTS_DIR / "error_analysis.json", "w") as f:
        json.dump(error_summary, f, indent=2)

    print(f"  Saved: {REPORTS_DIR / 'error_analysis.json'}")
    print(f"  Saved: {csv_path2}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
