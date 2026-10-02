"""
TRUVI-EV — Figure Generation (Phase 17)
=========================================
Generates all required figures for the IEEE research paper.

Required figures (per master prompt Section 32):
  Fig 1: Label distribution
  Fig 2: Performance comparison (baselines vs TRUVI-EV)
  Fig 3: Reliability gate visualization
  Fig 4: Confusion matrix (TRUVI-EV)
  Fig 5: Ablation study
  Fig 6: Hallucination rate by LLM model
"""

import sys
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

LABEL_NAMES = ["SUPPORTED", "CONTRADICTED", "UNVERIFIED"]
COLORS = {"SUPPORTED": "#2ecc71", "CONTRADICTED": "#e74c3c", "UNVERIFIED": "#f39c12"}

# Style settings
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def fig1_label_distribution():
    """Pie chart of 3-class claim-level label distribution."""
    summary_path = REPORTS_DIR / "claim_extraction_summary.json"
    if not summary_path.exists():
        print("  [SKIP] fig1 — claim_extraction_summary.json not found")
        return
    with open(summary_path) as f:
        data = json.load(f)

    totals = {"SUPPORTED": 0, "CONTRADICTED": 0, "UNVERIFIED": 0}
    for split_data in data["splits"].values():
        for label, count in split_data["label_distribution"].items():
            totals[label] = totals.get(label, 0) + count

    labels = list(totals.keys())
    sizes = list(totals.values())
    colors = [COLORS[l] for l in labels]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=[f"{l}\n({s:,})" for l, s in zip(labels, sizes)],
        colors=colors, autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 11}, pctdistance=0.75
    )
    for t in autotexts:
        t.set_fontweight("bold")
    ax.set_title("Claim-Level Label Distribution (114,501 claims)",
                 fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_label_distribution.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig1_label_distribution.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved: fig1_label_distribution.png/pdf")


def fig2_performance_comparison():
    """Bar chart comparing all models — baselines vs TRUVI-EV on test set."""
    csv_path = TABLES_DIR / "main_results.csv"
    if not csv_path.exists():
        print("  [SKIP] fig2 — main_results.csv not found")
        return

    models, accs, f1_macs, f1_ws = [], [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Model"]
            # Short name for display
            if "TRUVI" in name:
                short = "TRUVI-EV"
            else:
                short = name.split(":")[0].strip()
            models.append(short)
            accs.append(float(row["Accuracy"]))
            f1_macs.append(float(row["F1_Macro"]))
            f1_ws.append(float(row["F1_Weighted"]))

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    b1 = ax.bar(x - width, accs, width, label="Accuracy", color="#3498db", edgecolor="white")
    b2 = ax.bar(x, f1_macs, width, label="F1 Macro", color="#e74c3c", edgecolor="white")
    b3 = ax.bar(x + width, f1_ws, width, label="F1 Weighted", color="#2ecc71", edgecolor="white")

    # Highlight TRUVI-EV
    for i, m in enumerate(models):
        if "TRUVI" in m:
            for b in [b1[i], b2[i], b3[i]]:
                b.set_edgecolor("black")
                b.set_linewidth(2)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Performance Comparison: Baselines vs TRUVI-EV (Test Set)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=10)
    ax.legend(loc="upper left")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_performance_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig2_performance_comparison.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved: fig2_performance_comparison.png/pdf")


def fig3_reliability_gate():
    """Reliability gate visualization — mean gate values per feature."""
    gate_path = EXPERIMENTS_DIR / "truvi" / "gate_values.json"
    if not gate_path.exists():
        print("  [SKIP] fig3 — gate_values.json not found")
        return
    with open(gate_path) as f:
        data = json.load(f)

    gate_vals = data["mean_gate_values"]
    features = list(gate_vals.keys())
    values = list(gate_vals.values())

    # Group colors
    group_colors = {}
    for f in features:
        if f.startswith("nli"): group_colors[f] = "#3498db"
        elif f.startswith("sim"): group_colors[f] = "#2ecc71"
        elif f.startswith("rel"): group_colors[f] = "#f39c12"
        elif f.startswith("agr"): group_colors[f] = "#e74c3c"
        else: group_colors[f] = "#95a5a6"

    colors = [group_colors[f] for f in features]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = range(len(features))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="white", height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.replace("_", " ") for f in features], fontsize=9)
    ax.set_xlabel("Mean Gate Value (0 = suppressed, 1 = fully trusted)", fontsize=11)
    ax.set_title("TRUVI-EV Reliability Gate — Learned Feature Importance",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="Threshold 0.5")

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8)

    # Legend
    legend_patches = [
        mpatches.Patch(color="#3498db", label="NLI"),
        mpatches.Patch(color="#2ecc71", label="Similarity"),
        mpatches.Patch(color="#f39c12", label="Reliability"),
        mpatches.Patch(color="#e74c3c", label="Agreement"),
    ]
    ax.legend(handles=legend_patches, loc="lower right")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_reliability_gate.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig3_reliability_gate.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved: fig3_reliability_gate.png/pdf")


def fig4_confusion_matrix():
    """Confusion matrix heatmap for TRUVI-EV."""
    report_path = REPORTS_DIR / "error_analysis.json"
    if not report_path.exists():
        # Try training_results.json as fallback
        report_path = REPORTS_DIR / "training_results.json"
        if not report_path.exists():
            print("  [SKIP] fig4 — no results file found")
            return
        with open(report_path) as f:
            data = json.load(f)
        # Find TRUVI-EV confusion matrix in test results
        cm = None
        for r in data.get("test_results", []):
            if "TRUVI" in r.get("model", ""):
                cm = np.array(r["confusion_matrix"])
                break
        if cm is None:
            print("  [SKIP] fig4 — TRUVI-EV results not found")
            return
        title_suffix = "Test Set"
    else:
        with open(report_path) as f:
            data = json.load(f)
        cm = np.array(data["confusion_matrix"])
        title_suffix = "Validation Set"

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(LABEL_NAMES, fontsize=11)
    ax.set_yticklabels(LABEL_NAMES, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"TRUVI-EV Confusion Matrix ({title_suffix})",
                 fontsize=14, fontweight="bold")

    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i][j]:,}", ha="center", va="center",
                    color="white" if cm[i][j] > cm.max()/2 else "black",
                    fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig4_confusion_matrix.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved: fig4_confusion_matrix.png/pdf")


def fig5_ablation():
    """Horizontal bar chart for ablation study."""
    csv_path = TABLES_DIR / "ablation_results.csv"
    if not csv_path.exists():
        print("  [SKIP] fig5 — ablation_results.csv not found")
        return

    exps, f1_macs, prec_macs, rec_macs = [], [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            exps.append(row["Experiment"])
            f1_macs.append(float(row["F1_Macro"]))
            prec_macs.append(float(row.get("Precision_Macro", 0)))
            rec_macs.append(float(row.get("Recall_Macro", 0)))

    colors = ["#3498db"] + ["#e74c3c"] * (len(exps) - 1)

    fig, ax = plt.subplots(figsize=(11, 7))
    y_pos = range(len(exps))
    bars = ax.barh(y_pos, f1_macs, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(exps, fontsize=10)
    ax.set_xlabel("F1 Macro Score", fontsize=12)
    ax.set_title("Ablation Study — Signal Contribution Analysis",
                 fontsize=14, fontweight="bold")

    for bar, val in zip(bars, f1_macs):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9, fontweight="bold")

    # Reference line for full model
    if f1_macs:
        ax.axvline(x=f1_macs[0], color="#3498db", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_ablation.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig5_ablation.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved: fig5_ablation.png/pdf")


def fig6_hallucination_by_model():
    """Hallucination rate by LLM model."""
    report_path = REPORTS_DIR / "dataset_report.json"
    if not report_path.exists():
        print("  [SKIP] fig6 — dataset_report.json not found")
        return
    with open(report_path) as f:
        data = json.load(f)

    by_model = data.get("response_analysis", {}).get("hallucination_by_model", {})
    if not by_model:
        print("  [SKIP] fig6 — no hallucination_by_model data")
        return

    models = sorted(by_model.keys())
    rates = []
    for m in models:
        w = by_model[m]["with"]
        wo = by_model[m]["without"]
        rates.append(w / (w + wo) * 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(models)))
    bars = ax.barh(models, rates, color=colors, edgecolor="white")
    ax.set_xlabel("Hallucination Rate (%)", fontsize=12)
    ax.set_title("Hallucination Rate by LLM Model (RAGTruth)",
                 fontsize=14, fontweight="bold")

    for bar, val in zip(bars, rates):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_hallucination_by_model.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig6_hallucination_by_model.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved: fig6_hallucination_by_model.png/pdf")


def fig7_training_curve():
    """TRUVI-EV training loss and validation F1 over epochs."""
    hist_path = EXPERIMENTS_DIR / "truvi" / "training_history.csv"
    if not hist_path.exists():
        print("  [SKIP] fig7 — training_history.csv not found")
        return

    epochs, train_loss, val_loss, val_f1 = [], [], [], []
    with open(hist_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            val_f1.append(float(row["val_f1_macro"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(epochs, train_loss, label="Train Loss", color="#3498db", linewidth=2)
    ax1.plot(epochs, val_loss, label="Val Loss", color="#e74c3c", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss", fontweight="bold")
    ax1.legend()

    # F1 curve
    ax2.plot(epochs, val_f1, label="Val F1 Macro", color="#2ecc71", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Macro")
    ax2.set_title("Validation F1 Macro Score", fontweight="bold")
    best_epoch = epochs[np.argmax(val_f1)]
    best_f1 = max(val_f1)
    ax2.axhline(y=best_f1, color="gray", linestyle="--", alpha=0.5)
    ax2.annotate(f"Best: {best_f1:.4f} (epoch {best_epoch})",
                 xy=(best_epoch, best_f1), fontsize=10,
                 arrowprops=dict(arrowstyle="->"), xytext=(best_epoch + 10, best_f1 - 0.02))

    plt.suptitle("TRUVI-EV Training Progress", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig7_training_curve.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig7_training_curve.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved: fig7_training_curve.png/pdf")


def main():
    print("=" * 70)
    print("TRUVI-EV — Phase 17: Figure Generation")
    print("=" * 70)
    print(f"Time: {datetime.now().isoformat()}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig1_label_distribution()
    fig2_performance_comparison()
    fig3_reliability_gate()
    fig4_confusion_matrix()
    fig5_ablation()
    fig6_hallucination_by_model()
    fig7_training_curve()

    print("\n" + "=" * 70)
    print("FIGURE GENERATION COMPLETE")
    print(f"All figures saved to: {FIGURES_DIR}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
