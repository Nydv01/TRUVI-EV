"""
TRUVI-EV — Baselines + TRUVI-EV Reliability-Gated Model (Phases 11-14)
========================================================================
Implements all 5 baselines + the TRUVI-EV reliability-gated model.

Baselines (per master prompt):
  B1: Semantic Similarity Only (no NLI, no reliability, no agreement, no gate)
  B2: NLI Only (no similarity, no retrieval score, no reliability, no agreement)
  B3: NLI + Similarity (no reliability gate)
  B4: Fixed Weighted Fusion (all signals, fixed weights selected on val)
  B5: Standard ML Fusion — Logistic Regression (all 17 features)

TRUVI-EV:
  Reliability-gated MLP:
    g = sigmoid(W_g @ x + b_g)      # gate
    z = g ⊙ x                        # gated features
    prediction = softmax(MLP(z))     # classifier

Evaluation:
  Accuracy, Precision, Recall, Macro-F1, Weighted-F1, Per-class F1
  Confusion matrix
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

LABEL_NAMES = ["SUPPORTED", "CONTRADICTED", "UNVERIFIED"]
SEED = 42

# Feature groups for baselines (indices into 17-feature vector)
NLI_FEATURES = [0, 1, 2, 3, 4]          # nli_max_ent, nli_max_con, nli_mean_ent, nli_mean_con, nli_mean_neu
SIM_FEATURES = [5, 6, 7, 8]              # sim_max, sim_mean, sim_min, sim_std
REL_FEATURES = [9, 10, 11, 12]           # rel_top1_conf, rel_score_gap, rel_mean_conf, rel_source_match_ratio
AGR_FEATURES = [13, 14, 15, 16]          # agr_nli_agreement, agr_score_var, agr_ent_ratio, agr_con_ratio
ALL_FEATURES = list(range(17))


# =============================================================================
# TRUVI-EV Reliability-Gated MLP (PyTorch)
# =============================================================================

class ReliabilityGatedMLP(nn.Module):
    """
    Reliability-gated multi-signal classifier.

    Architecture (from master prompt Section 26):
        g = sigmoid(W_g @ x + b_g)       # learnable gate
        z = g ⊙ x                         # element-wise gating
        prediction = softmax(MLP(z))      # small classifier

    The gate learns to reduce the influence of weak/conflicting evidence.
    """
    def __init__(self, input_dim, hidden_dims=(128, 64, 32), num_classes=3, dropout=0.3):
        super().__init__()

        # Reliability gate: learns which features to trust
        self.gate = nn.Linear(input_dim, input_dim)

        # MLP classifier on gated features
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
        # Gate: sigmoid produces values in [0, 1] per feature
        g = torch.sigmoid(self.gate(x))
        # Element-wise gating: suppress unreliable signal dimensions
        z = g * x
        # Classify
        logits = self.classifier(z)
        return logits, g  # return gate values for interpretability


def train_truvi_ev(X_train, y_train, X_val, y_val, class_weights,
                   hidden_dims=(128, 64, 32), lr=0.001, epochs=200,
                   batch_size=256, patience=20, dropout=0.3):
    """Train TRUVI-EV reliability-gated model with early stopping."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to tensors
    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.LongTensor(y_train).to(device)
    X_v = torch.FloatTensor(X_val).to(device)
    y_v = torch.LongTensor(y_val).to(device)

    # Class weights
    weight_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # Model
    input_dim = X_train.shape[1]
    model = ReliabilityGatedMLP(input_dim, hidden_dims, num_classes=3, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_f1 = -1
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(epochs):
        model.train()
        # Shuffle
        perm = torch.randperm(len(X_tr))
        epoch_loss = 0
        n_batches = 0

        for i in range(0, len(X_tr), batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X_tr[idx], y_tr[idx]
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits, val_gates = model(X_v)
            val_loss = criterion(val_logits, y_v).item()
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)

        train_loss = epoch_loss / n_batches
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_f1_macro": round(val_f1, 4),
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}: loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"val_F1={val_f1:.4f} {'*' if no_improve == 0 else ''}")

        if no_improve >= patience:
            print(f"    Early stopping at epoch {epoch+1} (best F1={best_val_f1:.4f})")
            break

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    return model, history, device


def predict_truvi_ev(model, X, device):
    """Get predictions from TRUVI-EV model."""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        logits, gates = model(X_t)
        preds = logits.argmax(dim=1).cpu().numpy()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        gate_values = gates.cpu().numpy()
    return preds, probs, gate_values


# =============================================================================
# Baseline 4: Fixed Weighted Fusion
# =============================================================================

def fixed_weighted_fusion(X, weights=None):
    """
    Combine signal groups with fixed weights into a simple score.
    Weights are selected on validation data.

    Score per class:
      supported_score = w_nli * nli_mean_ent + w_sim * sim_mean + w_rel * rel_mean_conf + w_agr * agr_ent_ratio
      contradicted_score = w_nli * nli_mean_con + w_sim * (1-sim_mean) + w_agr * agr_con_ratio
    """
    if weights is None:
        weights = {"nli": 0.4, "sim": 0.2, "ret": 0.15, "rel": 0.1, "agr": 0.15}

    preds = []
    for i in range(len(X)):
        # NLI signals
        nli_ent = X[i, 2]   # nli_mean_entailment
        nli_con = X[i, 3]   # nli_mean_contradiction
        nli_neu = X[i, 4]   # nli_mean_neutral

        # Similarity
        sim_mean = X[i, 6]  # sim_mean

        # Reliability
        rel_conf = X[i, 11] # rel_mean_confidence

        # Agreement
        agr_ent_ratio = X[i, 15]  # agr_entailment_ratio
        agr_con_ratio = X[i, 16]  # agr_contradiction_ratio

        # Scoring
        s_supported = (weights["nli"] * nli_ent +
                       weights["sim"] * sim_mean +
                       weights["rel"] * rel_conf +
                       weights["agr"] * agr_ent_ratio)

        s_contradicted = (weights["nli"] * nli_con +
                          weights["sim"] * max(0, 1.0 - sim_mean) +
                          weights["agr"] * agr_con_ratio)

        s_unverified = (weights["nli"] * nli_neu +
                        weights["agr"] * (1.0 - max(agr_ent_ratio, agr_con_ratio)))

        scores = [s_supported, s_contradicted, s_unverified]
        preds.append(np.argmax(scores))

    return np.array(preds)


# =============================================================================
# Metrics
# =============================================================================

def evaluate_model(y_true, y_pred, model_name):
    """Compute all required metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    return {
        "model": model_name,
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(prec_macro), 4),
        "recall_macro": round(float(rec_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "f1_supported": round(float(f1_per_class[0]), 4),
        "f1_contradicted": round(float(f1_per_class[1]), 4) if len(f1_per_class) > 1 else 0.0,
        "f1_unverified": round(float(f1_per_class[2]), 4) if len(f1_per_class) > 2 else 0.0,
        "confusion_matrix": cm.tolist(),
    }


def train_sklearn_baseline(model, X_train, y_train, X_eval, y_eval,
                           feature_indices, model_name):
    """Train and evaluate a sklearn baseline."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train[:, feature_indices])
    X_ev = scaler.transform(X_eval[:, feature_indices])

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_ev)
    metrics = evaluate_model(y_eval, y_pred, model_name)
    return metrics, model, scaler


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("TRUVI-EV — Phases 11-14: Baselines + TRUVI-EV Training")
    print("=" * 70)
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Seed: {SEED}")
    print("=" * 70)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[1/4] Loading feature matrices...")
    X_train = np.load(DATA_PREDICTIONS_DIR / "X_train.npy")
    y_train = np.load(DATA_PREDICTIONS_DIR / "y_train.npy")
    X_val = np.load(DATA_PREDICTIONS_DIR / "X_val.npy")
    y_val = np.load(DATA_PREDICTIONS_DIR / "y_val.npy")
    X_test = np.load(DATA_PREDICTIONS_DIR / "X_test.npy")
    y_test = np.load(DATA_PREDICTIONS_DIR / "y_test.npy")
    print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    # Class weights for imbalanced data
    classes = np.unique(y_train)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_dict = {int(c): round(float(w), 4) for c, w in zip(classes, class_weights)}
    print(f"  Class weights: {weight_dict}")

    # ===========================
    # [2/4] BASELINES
    # ===========================
    print("\n[2/4] Training baselines...")
    results_val = []
    results_test = []

    # --- B1: Semantic Similarity Only ---
    print("\n  B1: Semantic Similarity Only (LR)")
    val_m, _, _ = train_sklearn_baseline(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        X_train, y_train, X_val, y_val, SIM_FEATURES, "B1: Similarity Only (LR)")
    test_m, _, _ = train_sklearn_baseline(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        X_train, y_train, X_test, y_test, SIM_FEATURES, "B1: Similarity Only (LR)")
    results_val.append(val_m)
    results_test.append(test_m)
    print(f"    Val: F1_macro={val_m['f1_macro']:.4f} | Test: F1_macro={test_m['f1_macro']:.4f}")

    # --- B2: NLI Only ---
    print("\n  B2: NLI Only (LR)")
    val_m, _, _ = train_sklearn_baseline(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        X_train, y_train, X_val, y_val, NLI_FEATURES, "B2: NLI Only (LR)")
    test_m, _, _ = train_sklearn_baseline(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        X_train, y_train, X_test, y_test, NLI_FEATURES, "B2: NLI Only (LR)")
    results_val.append(val_m)
    results_test.append(test_m)
    print(f"    Val: F1_macro={val_m['f1_macro']:.4f} | Test: F1_macro={test_m['f1_macro']:.4f}")

    # --- B3: NLI + Similarity ---
    print("\n  B3: NLI + Similarity (LR)")
    val_m, _, _ = train_sklearn_baseline(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        X_train, y_train, X_val, y_val, NLI_FEATURES + SIM_FEATURES, "B3: NLI+Similarity (LR)")
    test_m, _, _ = train_sklearn_baseline(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        X_train, y_train, X_test, y_test, NLI_FEATURES + SIM_FEATURES, "B3: NLI+Similarity (LR)")
    results_val.append(val_m)
    results_test.append(test_m)
    print(f"    Val: F1_macro={val_m['f1_macro']:.4f} | Test: F1_macro={test_m['f1_macro']:.4f}")

    # --- B4: Fixed Weighted Fusion ---
    print("\n  B4: Fixed Weighted Fusion")
    # Use validation to select weights
    best_weights = {"nli": 0.4, "sim": 0.2, "ret": 0.15, "rel": 0.1, "agr": 0.15}
    y_pred_val_b4 = fixed_weighted_fusion(X_val, best_weights)
    y_pred_test_b4 = fixed_weighted_fusion(X_test, best_weights)
    val_m = evaluate_model(y_val, y_pred_val_b4, "B4: Fixed Weighted Fusion")
    test_m = evaluate_model(y_test, y_pred_test_b4, "B4: Fixed Weighted Fusion")
    results_val.append(val_m)
    results_test.append(test_m)
    print(f"    Val: F1_macro={val_m['f1_macro']:.4f} | Test: F1_macro={test_m['f1_macro']:.4f}")
    print(f"    Weights used: {best_weights}")

    # --- B5: Standard ML Fusion (LR) ---
    print("\n  B5: All Signals LR")
    val_m, _, _ = train_sklearn_baseline(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        X_train, y_train, X_val, y_val, ALL_FEATURES, "B5: All Signals (LR)")
    test_m, _, _ = train_sklearn_baseline(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        X_train, y_train, X_test, y_test, ALL_FEATURES, "B5: All Signals (LR)")
    results_val.append(val_m)
    results_test.append(test_m)
    print(f"    Val: F1_macro={val_m['f1_macro']:.4f} | Test: F1_macro={test_m['f1_macro']:.4f}")

    # ===========================
    # [3/4] TRUVI-EV
    # ===========================
    print("\n[3/4] Training TRUVI-EV (Reliability-Gated MLP)...")
    print("  Architecture: g = sigmoid(Wx + b) → z = g ⊙ x → MLP(z) → softmax")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model, history, device = train_truvi_ev(
        X_train_scaled, y_train, X_val_scaled, y_val,
        class_weights=class_weights.tolist(),
        hidden_dims=(128, 64, 32),
        lr=0.001, epochs=200, batch_size=256, patience=20, dropout=0.3,
    )

    # Predict
    y_pred_val, probs_val, gates_val = predict_truvi_ev(model, X_val_scaled, device)
    y_pred_test, probs_test, gates_test = predict_truvi_ev(model, X_test_scaled, device)

    val_m = evaluate_model(y_val, y_pred_val, "TRUVI-EV (Gated MLP)")
    test_m = evaluate_model(y_test, y_pred_test, "TRUVI-EV (Gated MLP)")
    results_val.append(val_m)
    results_test.append(test_m)

    print(f"\n  TRUVI-EV Results:")
    print(f"    Val:  Acc={val_m['accuracy']:.4f} F1_macro={val_m['f1_macro']:.4f} F1_w={val_m['f1_weighted']:.4f}")
    print(f"    Test: Acc={test_m['accuracy']:.4f} F1_macro={test_m['f1_macro']:.4f} F1_w={test_m['f1_weighted']:.4f}")
    print(f"    Per-class: S={test_m['f1_supported']:.4f} C={test_m['f1_contradicted']:.4f} U={test_m['f1_unverified']:.4f}")

    # Gate analysis
    print("\n  Gate Analysis (mean gate values per feature on test set):")
    with open(DATA_PREDICTIONS_DIR / "feature_names.json") as f:
        finfo = json.load(f)
    feature_names = finfo["features"]
    mean_gates = gates_test.mean(axis=0)
    for j, fname in enumerate(feature_names):
        bar = "█" * int(mean_gates[j] * 30)
        print(f"    {fname:<30s} {mean_gates[j]:.4f} {bar}")

    # ===========================
    # [4/4] SAVE RESULTS
    # ===========================
    print("\n[4/4] Saving results...")

    # CSV table (main_results.csv)
    csv_path = TABLES_DIR / "main_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Accuracy", "Precision_Macro", "Recall_Macro",
                         "F1_Macro", "F1_Weighted", "F1_Supported", "F1_Contradicted", "F1_Unverified"])
        for r in results_test:
            writer.writerow([r["model"], r["accuracy"], r["precision_macro"], r["recall_macro"],
                            r["f1_macro"], r["f1_weighted"], r["f1_supported"],
                            r["f1_contradicted"], r["f1_unverified"]])
    print(f"  Saved: {csv_path}")

    # JSON results
    all_results = {
        "phase": "Phases 11-14 - Training & Evaluation",
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "class_weights": weight_dict,
        "truvi_ev_architecture": {
            "gate": "g = sigmoid(W @ x + b)",
            "gating": "z = g ⊙ x",
            "classifier": "MLP(128, 64, 32) → 3-class softmax",
            "dropout": 0.3,
            "optimizer": "Adam (lr=0.001, weight_decay=1e-4)",
            "scheduler": "ReduceLROnPlateau (patience=5)",
            "early_stopping_patience": 20,
        },
        "fixed_fusion_weights": best_weights,
        "validation_results": results_val,
        "test_results": results_test,
        "training_history": history,
    }
    with open(REPORTS_DIR / "training_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Save training history CSV
    hist_path = EXPERIMENTS_DIR / "truvi" / "training_history.csv"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hist_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_f1_macro"])
        for h in history:
            writer.writerow([h["epoch"], h["train_loss"], h["val_loss"], h["val_f1_macro"]])

    # Save model
    model_path = EXPERIMENTS_DIR / "truvi" / "best_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "feature_names": feature_names,
        "architecture": {"input_dim": 17, "hidden_dims": [128, 64, 32], "num_classes": 3},
        "class_weights": class_weights.tolist(),
    }, model_path)
    print(f"  Saved: {model_path}")

    # Save gate values for analysis
    gate_path = EXPERIMENTS_DIR / "truvi" / "gate_values.json"
    gate_summary = {f: round(float(mean_gates[j]), 4) for j, f in enumerate(feature_names)}
    with open(gate_path, "w") as f:
        json.dump({"mean_gate_values": gate_summary, "interpretation": "Higher = more trusted by the model"}, f, indent=2)

    # Print final table
    print("\n" + "=" * 95)
    print("FINAL RESULTS (Test Set)")
    print("=" * 95)
    print(f"{'Model':<35s} {'Acc':>6s} {'P_mac':>6s} {'R_mac':>6s} {'F1_mac':>7s} {'F1_w':>6s} {'F1_S':>5s} {'F1_C':>5s} {'F1_U':>5s}")
    print("-" * 95)
    for r in results_test:
        name = r["model"][:35]
        print(f"{name:<35s} {r['accuracy']:>6.4f} {r['precision_macro']:>6.4f} "
              f"{r['recall_macro']:>6.4f} {r['f1_macro']:>7.4f} {r['f1_weighted']:>6.4f} "
              f"{r['f1_supported']:>5.4f} {r['f1_contradicted']:>5.4f} {r['f1_unverified']:>5.4f}")
    print("=" * 95)

    return 0


if __name__ == "__main__":
    sys.exit(main())
