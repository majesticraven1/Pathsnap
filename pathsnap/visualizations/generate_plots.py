import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

OUT_DIR = Path("pathsnap/artifacts/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_cv_macro_f1():
    with open("pathsnap/artifacts/results/detailed_model_metrics.json") as f:
        metrics = json.load(f)
        
    models = list(metrics.keys())
    means = [metrics[m]["cv_macro_f1"] for m in models]
    stds = [metrics[m]["cv_std_f1"] for m in models]
    
    plt.figure(figsize=(9, 5))
    colors = ["#4a90e2", "#50e3c2", "#f5a623", "#7ed321", "#9013fe"]
    bars = plt.bar(models, means, yerr=stds, capsize=5, color=colors, edgecolor="black", alpha=0.85, width=0.55)
    
    plt.ylabel("Macro F1-Score (5-Fold CV)")
    plt.title("Cross-Validation Macro F1-Score Comparison Across 5 Model Families", pad=12, fontweight="bold")
    plt.ylim(0.65, 0.92)
    
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.012, f"{h:.4f}", ha="center", va="bottom", fontweight="bold")
        
    plt.tight_layout()
    out_file = OUT_DIR / "cv_macro_f1_comparison.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved {out_file}")


def plot_test_performance_comparison():
    df = pd.read_csv("pathsnap/artifacts/results/model_comparison.csv")
    
    models = df["Model"].tolist()
    # Parse Top-1 Accuracy percentage
    top1_acc = [float(x.replace("%", "")) / 100.0 for x in df["Top-1 Accuracy"]]
    macro_f1 = df["Student-Test Macro F1-score"].tolist()
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.figure(figsize=(10, 5.5))
    rects1 = plt.bar(x - width/2, top1_acc, width, label="Top-1 Accuracy", color="#2b5c8f", edgecolor="black", alpha=0.9)
    rects2 = plt.bar(x + width/2, macro_f1, width, label="Macro F1-Score", color="#27ae60", edgecolor="black", alpha=0.9)
    
    plt.ylabel("Score")
    plt.title("Student-Testing Performance: Top-1 Accuracy vs. Macro F1-Score", pad=12, fontweight="bold")
    plt.xticks(x, models, rotation=15, ha="right")
    plt.ylim(0.65, 0.95)
    plt.legend(frameon=True, loc="upper left")
    
    for rect in rects1:
        h = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., h + 0.008, f"{h*100:.1f}%", ha="center", va="bottom", fontsize=8.5)
    for rect in rects2:
        h = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., h + 0.008, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)
        
    plt.tight_layout()
    out_file = OUT_DIR / "test_top1_and_f1_comparison.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved {out_file}")


def plot_log_loss_comparison():
    df = pd.read_csv("pathsnap/artifacts/results/model_comparison.csv")
    models = df["Model"].tolist()
    log_losses = df["Log Loss"].tolist()
    
    plt.figure(figsize=(9, 4.8))
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db"]
    bars = plt.bar(models, log_losses, color=colors, edgecolor="black", alpha=0.85, width=0.5)
    
    plt.ylabel("Binary Cross-Entropy Log Loss (Lower is Better)")
    plt.title("Log Loss Comparison Across Candidate Classification Models", pad=12, fontweight="bold")
    plt.ylim(0.0, max(log_losses) * 1.25)
    
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.008, f"{h:.4f}", ha="center", va="bottom", fontweight="bold")
        
    plt.tight_layout()
    out_file = OUT_DIR / "log_loss_comparison.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved {out_file}")


def plot_confusion_matrices():
    with open("pathsnap/artifacts/results/detailed_model_metrics.json") as f:
        metrics = json.load(f)
        
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8))
    models = list(metrics.keys())
    
    for i, name in enumerate(models):
        cm = np.array(metrics[name]["confusion_matrix"])
        # Normalized
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", cbar=False, ax=axes[i],
                    xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"] if i == 0 else [])
        axes[i].set_title(name, fontsize=10, fontweight="bold")
        axes[i].set_xlabel("Predicted")
        if i == 0:
            axes[i].set_ylabel("Actual Label")
            
    plt.suptitle("Normalized Confusion Matrices on Student-Testing Dataset", y=1.02, fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_file = OUT_DIR / "confusion_matrices.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_file}")


def plot_bayesian_optimization():
    with open("pathsnap/artifacts/results/bayesian_opt_results.json") as f:
        data = json.load(f)
        
    trials = data["trials"]
    trial_nums = [t["trial_number"] for t in trials]
    scores = [t["cv_macro_f1"] for t in trials]
    best_running = np.maximum.accumulate(scores)
    
    plt.figure(figsize=(9, 4.8))
    plt.plot(trial_nums, scores, marker="o", linestyle="--", color="#3498db", label="Trial Score (Macro F1)", alpha=0.7)
    plt.plot(trial_nums, best_running, marker="s", color="#e74c3c", linewidth=2.2, label="Best Cumulative Score")
    
    plt.xlabel("Bayesian Optimization Trial")
    plt.ylabel("Group-CV Macro F1-Score")
    plt.title("Bayesian Hyperparameter Search Progression (Optuna TPE Sampler)", pad=12, fontweight="bold")
    plt.xticks(trial_nums)
    plt.ylim(min(scores) - 0.01, max(scores) + 0.015)
    plt.legend(frameon=True, loc="lower right")
    
    # Annotate best trial
    best_idx = int(np.argmax(scores))
    plt.annotate(
        f"Best: Trial #{best_idx+1} ({scores[best_idx]:.4f})",
        xy=(trial_nums[best_idx], scores[best_idx]),
        xytext=(trial_nums[best_idx] - 3, scores[best_idx] + 0.007),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold"
    )
    
    plt.tight_layout()
    out_file = OUT_DIR / "bayesian_opt_history.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved {out_file}")


def plot_hmm_vs_classifier():
    df = pd.read_csv("pathsnap/artifacts/results/hmm_vs_classifier_comparison.csv")
    
    trips = [str(t).zfill(8)[-4:] for t in df["Trip ID"]] # Last 4 digits for readability
    clf_acc = df["Classifier Top-1 (%)"].tolist()
    hmm_acc = df["HMM Top-1 (%)"].tolist()
    clf_cont = df["Classifier Continuity (%)"].tolist()
    hmm_cont = df["HMM Continuity (%)"].tolist()
    
    x = np.arange(len(trips))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    
    # Plot 1: Top-1 Accuracy
    ax1.bar(x - width/2, clf_acc, width, label="Classifier-Only Top-1", color="#e67e22", alpha=0.85, edgecolor="black")
    ax1.bar(x + width/2, hmm_acc, width, label="Classifier + HMM Top-1", color="#27ae60", alpha=0.85, edgecolor="black")
    ax1.set_ylabel("Top-1 Accuracy (%)")
    ax1.set_title("Pointwise Top-1 Accuracy: Classifier-Only vs. Classifier + HMM", fontweight="bold")
    ax1.set_ylim(55, 105)
    ax1.legend(loc="lower right", frameon=True)
    
    # Plot 2: Route Continuity
    ax2.bar(x - width/2, clf_cont, width, label="Classifier-Only Continuity", color="#e74c3c", alpha=0.85, edgecolor="black")
    ax2.bar(x + width/2, hmm_cont, width, label="Classifier + HMM Continuity", color="#2980b9", alpha=0.85, edgecolor="black")
    ax2.set_ylabel("Route Continuity (%)")
    ax2.set_title("Sequence Route Continuity: Classifier-Only vs. Classifier + HMM", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Trip {t}" for t in trips], rotation=15)
    ax2.set_ylim(75, 105)
    ax2.legend(loc="lower right", frameon=True)
    
    plt.tight_layout()
    out_file = OUT_DIR / "hmm_vs_classifier_comparison.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved {out_file}")


def plot_mlp_history():
    import joblib
    mlp = joblib.load("pathsnap/artifacts/trained_models/multi_layer_perceptron.joblib")
    loss_curve = mlp.loss_curve_
    
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(loss_curve) + 1), loss_curve, marker="o", color="#8e44ad", linewidth=2.0)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Training Loss")
    plt.title("Multi-Layer Perceptron (MLP) Training Loss Curve Across Epochs", pad=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    out_file = OUT_DIR / "mlp_training_history.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved {out_file}")


def generate_all_plots():
    print("Generating all visual charts for report and repository...")
    plot_cv_macro_f1()
    plot_test_performance_comparison()
    plot_log_loss_comparison()
    plot_confusion_matrices()
    plot_bayesian_optimization()
    plot_hmm_vs_classifier()
    plot_mlp_history()
    print("All figures successfully created in pathsnap/artifacts/figures/")


if __name__ == "__main__":
    generate_all_plots()
