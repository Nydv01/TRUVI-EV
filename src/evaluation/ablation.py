"""
TRUVI-EV — Ablation Study (Phase 15)
======================================
Systematic removal of each signal group to measure contribution.

Ablation experiments (per master prompt Section 30):
  A1: Full TRUVI-EV (all 17 features with gate)
  A2: Without Evidence Reliability (remove 4 features)
  A3: Without Evidence Agreement (remove 4 features)
  A4: Without Semantic Similarity (remove 4 features)
  A5: Without NLI (remove 5 features)
  A6: Without Reliability Gate (all features, plain MLP — no gate)

All ablations use: same data, same test set, same evaluation metrics.
Only the specified component changes.
"""

import sys
import json
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

SEED = 42
LABEL_NAMES = ["SUPPORTED", "CONTRADICTED", "UNVERIFIED"]

# Feature indices
NLI_IDX = [0, 1, 2, 3, 4]
SIM_IDX = [5, 6, 7, 8]
REL_IDX = [9, 10, 11, 12]
AGR_IDX = [13, 14, 15, 16]
ALL_IDX = list(range(17))


class ReliabilityGatedMLP(nn.Module):
    """Same architecture as in train.py — with reliability gate."""
    def __init__(self, input_dim, hidden_dims=(128, 64, 32), num_classes=3, dropout=0.3):
        super().__init__()
        self.gate = nn.Linear(input_dim, input_dim)
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        g = torch.sigmoid(self.gate(x))
        z = g * x
        logits = self.classifier(z)
        return logits, g


class PlainMLP(nn.Module):
    """Same MLP but WITHOUT the reliability gate — for A6 ablation."""
    def __init__(self, input_dim, hidden_dims=(128, 64, 32), num_classes=3, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.classifier(x)
        return logits, None


def train_and_evaluate(X_train, y_train, X_val, y_val, feature_indices,
                       model_class, class_weights, name,
                       epochs=200, batch_size=256, patience=20):
    """Train a model on selected features and evaluate on val."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr = X_train[:, feature_indices]
    X_v = X_val[:, feature_indices]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_v = scaler.transform(X_v)

    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.LongTensor(y_train).to(device)
    X_v_t = torch.FloatTensor(X_v).to(device)

    weight_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    input_dim = len(feature_indices)
    model = model_class(input_dim, (128, 64, 32), 3, 0.3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_val_f1 = -1
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr_t))
        for i in range(0, len(X_tr_t), batch_size):
            idx = perm[i:i+batch_size]
            logits, _ = model(X_tr_t[idx])
            loss = criterion(logits, y_tr_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(X_v_t)
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_logits, _ = model(X_v_t)
        y_pred = val_logits.argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_val, y_pred)
    f1_mac = f1_score(y_val, y_pred, average="macro", zero_division=0)
    f1_w = f1_score(y_val, y_pred, average="weighted", zero_division=0)
    f1_per = f1_score(y_val, y_pred, average=None, zero_division=0)
    prec_mac = precision_score(y_val, y_pred, average="macro", zero_division=0)
    rec_mac = recall_score(y_val, y_pred, average="macro", zero_division=0)

    return {
        "name": name,
        "num_features": len(feature_indices),
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(prec_mac), 4),
        "recall_macro": round(float(rec_mac), 4),
        "f1_macro": round(float(f1_mac), 4),
        "f1_weighted": round(float(f1_w), 4),
        "f1_per_class": [round(float(x), 4) for x in f1_per],
    }


def main():
    print("=" * 70)
    print("TRUVI-EV — Phase 15: Ablation Study")
    print("=" * 70)
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 70)

    X_train = np.load(DATA_PREDICTIONS_DIR / "X_train.npy")
    y_train = np.load(DATA_PREDICTIONS_DIR / "y_train.npy")
    X_val = np.load(DATA_PREDICTIONS_DIR / "X_val.npy")
    y_val = np.load(DATA_PREDICTIONS_DIR / "y_val.npy")

    print(f"  Train: {X_train.shape} | Val: {X_val.shape}")

    classes = np.unique(y_train)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)

    ablations = [
        ("A1: Full TRUVI-EV",        ALL_IDX, ReliabilityGatedMLP),
        ("A2: w/o Reliability",       [i for i in ALL_IDX if i not in REL_IDX], ReliabilityGatedMLP),
        ("A3: w/o Agreement",         [i for i in ALL_IDX if i not in AGR_IDX], ReliabilityGatedMLP),
        ("A4: w/o Similarity",        [i for i in ALL_IDX if i not in SIM_IDX], ReliabilityGatedMLP),
        ("A5: w/o NLI",               [i for i in ALL_IDX if i not in NLI_IDX], ReliabilityGatedMLP),
        ("A6: w/o Gate (Plain MLP)",   ALL_IDX, PlainMLP),
    ]

    results = []
    for name, features, model_cls in ablations:
        print(f"\n  {name} ({len(features)} features, {model_cls.__name__})...")
        metrics = train_and_evaluate(
            X_train, y_train, X_val, y_val,
            features, model_cls, cw.tolist(), name,
        )
        results.append(metrics)
        fc = metrics["f1_per_class"]
        print(f"    Acc={metrics['accuracy']:.4f} F1_mac={metrics['f1_macro']:.4f} "
              f"F1_w={metrics['f1_weighted']:.4f}")
        print(f"    Per-class: S={fc[0]:.4f} C={fc[1]:.4f} U={fc[2]:.4f}")

    # Save CSV
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES_DIR / "ablation_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Experiment", "Features", "Accuracy", "Precision_Macro", "Recall_Macro",
                         "F1_Macro", "F1_Weighted", "F1_Supported", "F1_Contradicted", "F1_Unverified"])
        for r in results:
            fc = r["f1_per_class"]
            writer.writerow([r["name"], r["num_features"], r["accuracy"],
                            r["precision_macro"], r["recall_macro"],
                            r["f1_macro"], r["f1_weighted"],
                            fc[0], fc[1] if len(fc) > 1 else 0, fc[2] if len(fc) > 2 else 0])
    print(f"\n  Saved: {csv_path}")

    # Save JSON
    with open(REPORTS_DIR / "ablation_results.json", "w") as f:
        json.dump({"phase": "Phase 15 - Ablation", "timestamp": datetime.now().isoformat(),
                   "results": results}, f, indent=2)

    # Print table
    print("\n" + "=" * 100)
    print("ABLATION RESULTS (Validation Set)")
    print("=" * 100)
    print(f"{'Experiment':<30s} {'Feat':>4s} {'Acc':>6s} {'P_mac':>6s} {'R_mac':>6s} "
          f"{'F1_mac':>7s} {'F1_w':>6s} {'F1_S':>5s} {'F1_C':>5s} {'F1_U':>5s}")
    print("-" * 100)
    for r in results:
        fc = r["f1_per_class"]
        print(f"{r['name']:<30s} {r['num_features']:>4d} {r['accuracy']:>6.4f} "
              f"{r['precision_macro']:>6.4f} {r['recall_macro']:>6.4f} "
              f"{r['f1_macro']:>7.4f} {r['f1_weighted']:>6.4f} "
              f"{fc[0]:>5.4f} {fc[1] if len(fc)>1 else 0:>5.4f} {fc[2] if len(fc)>2 else 0:>5.4f}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
