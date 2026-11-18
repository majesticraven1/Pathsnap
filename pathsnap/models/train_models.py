import os
import time
import json
import joblib
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier


FEATURE_COLS = [
    "dist_to_seg", "heading_diff", "sin_hdiff", "cos_hdiff",
    "speed", "u_proj", "arc_len", "dist_to_start", "dist_to_end", "dt"
]


def compute_observation_top1(df: pd.DataFrame, probs: np.ndarray) -> float:
    """
    Compute observation-level Top-1 Accuracy:
    For each (trip_id, pt_idx), pick the candidate with highest predicted probability.
    Score 1 if that candidate is target == 1, else 0.
    """
    df_eval = df[["trip_id", "pt_idx", "arc_id", "target"]].copy()
    df_eval["prob"] = probs
    
    # Group by trip_id and pt_idx, find candidate with max prob
    idx_max = df_eval.groupby(["trip_id", "pt_idx"])["prob"].idxmax()
    top1_df = df_eval.loc[idx_max]
    
    top1_acc = float((top1_df["target"] == 1).mean())
    return top1_acc


def compute_route_continuity(df: pd.DataFrame, probs: np.ndarray, arcs_dict: Dict[str, np.ndarray]) -> float:
    """
    Compute sequential route continuity:
    Percentage of consecutive observation transitions where the top-1 candidate arcs
    are identical or topologically connected in the road network graph.
    """
    df_eval = df[["trip_id", "pt_idx", "arc_id"]].copy()
    df_eval["prob"] = probs
    idx_max = df_eval.groupby(["trip_id", "pt_idx"])["prob"].idxmax()
    top1_df = df_eval.loc[idx_max].sort_values(["trip_id", "pt_idx"])
    
    total_transitions = 0
    continuous_transitions = 0
    
    for tid, group in top1_df.groupby("trip_id"):
        arcs = group["arc_id"].values
        if len(arcs) < 2 or tid not in arcs_dict:
            continue
        trip_arcs = arcs_dict[tid]
        
        for i in range(len(arcs) - 1):
            a1 = arcs[i]
            a2 = arcs[i+1]
            total_transitions += 1
            if a1 == a2:
                continuous_transitions += 1
            elif a1 < len(trip_arcs) and a2 < len(trip_arcs):
                # a1.to_node == a2.from_node
                if trip_arcs[a1, 1] == trip_arcs[a2, 0]:
                    continuous_transitions += 1
                    
    return float(continuous_transitions / max(1, total_transitions))


def load_network_arcs(data_dir: str) -> Dict[str, np.ndarray]:
    arcs_dict = {}
    p = Path(data_dir)
    for tdir in p.iterdir():
        if tdir.is_dir():
            tid = tdir.name
            af = tdir / f"{tid}.arcs"
            if af.exists():
                arcs_dict[tid] = pd.read_csv(af, sep=r"\s+", header=None).values
    return arcs_dict


def get_model_definitions():
    models = {
        "Logistic Regression": {
            "family": "Linear and Probabilistic",
            "model": LogisticRegression(max_iter=500, C=1.0, class_weight="balanced", random_state=42),
            "needs_scale": True,
            "subsample_train": None,
            "params": {"C": 1.0, "solver": "lbfgs", "class_weight": "balanced"}
        },
        "K-Nearest Neighbors": {
            "family": "Instance-based",
            "model": KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1),
            "needs_scale": True,
            "subsample_train": 100000, # Subsample for tractable O(N*M) instance distance queries
            "params": {"n_neighbors": 15, "weights": "distance", "metric": "euclidean"}
        },
        "Random Forest": {
            "family": "Ensemble (Bagging)",
            "model": RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced_subsample", random_state=42, n_jobs=-1),
            "needs_scale": False,
            "subsample_train": 300000,
            "params": {"n_estimators": 100, "max_depth": 12, "class_weight": "balanced_subsample"}
        },
        "LightGBM": {
            "family": "Boosting (GBDT)",
            "model": LGBMClassifier(n_estimators=150, max_depth=7, learning_rate=0.08, scale_pos_weight=4.0, random_state=42, n_jobs=-1, verbose=-1),
            "needs_scale": False,
            "subsample_train": None,
            "params": {"n_estimators": 150, "max_depth": 7, "learning_rate": 0.08, "scale_pos_weight": 4.0}
        },
        "Multi-Layer Perceptron": {
            "family": "Neural Networks",
            "model": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=40, activation="relu", solver="adam", early_stopping=True, random_state=42),
            "needs_scale": True,
            "subsample_train": 300000,
            "params": {"hidden_layers": "(64, 32)", "activation": "relu", "solver": "adam"}
        }
    }
    return models


def train_and_evaluate_all():
    print("Loading feature datasets...")
    df_train = pd.read_parquet("pathsnap/artifacts/features_train.parquet")
    df_test = pd.read_parquet("pathsnap/artifacts/features_student_test.parquet")
    
    test_arcs_dict = load_network_arcs("Student folder/Student folder/student_test")
    
    X_train_full = df_train[FEATURE_COLS].values
    y_train_full = df_train["target"].values
    groups_train = df_train["trip_id"].values
    
    X_test_full = df_test[FEATURE_COLS].values
    y_test_full = df_test["target"].values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test_full)
    
    joblib.dump(scaler, "pathsnap/artifacts/trained_models/scaler.joblib")
    
    models = get_model_definitions()
    results_summary = []
    detailed_metrics = {}
    
    gkf = GroupKFold(n_splits=5)
    
    for name, config in models.items():
        print(f"\n==========================================")
        print(f"Training Model: {name} ({config['family']})")
        print(f"==========================================")
        
        # Cross-validation on training trips
        use_scaled = config["needs_scale"]
        X_all = X_train_scaled if use_scaled else X_train_full
        
        cv_f1s = []
        cv_t0 = time.time()
        
        # Use representative subsample of training data for CV speed if designated
        sub_n = config["subsample_train"]
        if sub_n is not None and len(df_train) > sub_n:
            np.random.seed(42)
            cv_indices = np.random.choice(len(df_train), sub_n, replace=False)
            cv_X = X_all[cv_indices]
            cv_y = y_train_full[cv_indices]
            cv_groups = groups_train[cv_indices]
        else:
            cv_X = X_all
            cv_y = y_train_full
            cv_groups = groups_train
            
        for fold, (train_idx, val_idx) in enumerate(gkf.split(cv_X, cv_y, groups=cv_groups)):
            fold_clf = joblib.load(joblib.dump(config["model"], "temp_clf.joblib")[0]) if os.path.exists("temp_clf.joblib") else config["model"]
            from sklearn.base import clone
            fold_clf = clone(config["model"])
            fold_clf.fit(cv_X[train_idx], cv_y[train_idx])
            val_preds = fold_clf.predict(cv_X[val_idx])
            fold_f1 = f1_score(cv_y[val_idx], val_preds, average="macro")
            cv_f1s.append(fold_f1)
            print(f"  Fold {fold+1} Macro F1: {fold_f1:.4f}")
            
        cv_macro_f1 = float(np.mean(cv_f1s))
        cv_std_f1 = float(np.std(cv_f1s))
        print(f"CV Macro F1: {cv_macro_f1:.4f} (+/- {cv_std_f1:.4f}) [Time: {time.time()-cv_t0:.1f}s]")
        
        # Train final model on full permitted training data (or scaled subsample)
        train_t0 = time.time()
        final_model = clone(config["model"])
        if sub_n is not None and len(df_train) > sub_n:
            np.random.seed(42)
            fit_idx = np.random.choice(len(df_train), sub_n, replace=False)
            final_model.fit(X_all[fit_idx], y_train_full[fit_idx])
        else:
            final_model.fit(X_all, y_train_full)
        train_time = time.time() - train_t0
        print(f"Final fit completed in {train_time:.2f}s")
        
        # Test evaluation
        X_test = X_test_scaled if use_scaled else X_test_full
        
        inf_t0 = time.time()
        test_probs = final_model.predict_proba(X_test)[:, 1]
        inf_time_total = time.time() - inf_t0
        inf_time_per_1k = (inf_time_total / len(X_test)) * 1000.0 * 1000.0 # ms per 1000 items
        
        test_preds = (test_probs >= 0.5).astype(int)
        test_macro_f1 = float(f1_score(y_test_full, test_preds, average="macro"))
        test_top1_acc = compute_observation_top1(df_test, test_probs)
        test_continuity = compute_route_continuity(df_test, test_probs, test_arcs_dict)
        test_logloss = float(log_loss(y_test_full, np.clip(test_probs, 1e-7, 1 - 1e-7)))
        
        # Compute precision, recall, confusion matrix
        from sklearn.metrics import precision_score, recall_score
        prec = float(precision_score(y_test_full, test_preds, zero_division=0))
        rec = float(recall_score(y_test_full, test_preds, zero_division=0))
        cm = confusion_matrix(y_test_full, test_preds).tolist()
        
        print(f"Student Test Results for {name}:")
        print(f"  Top-1 Accuracy: {test_top1_acc*100:.2f}%")
        print(f"  Macro F1:       {test_macro_f1:.4f}")
        print(f"  Precision:      {prec:.4f}")
        print(f"  Recall:         {rec:.4f}")
        print(f"  Route Contin.:  {test_continuity*100:.2f}%")
        print(f"  Log Loss:       {test_logloss:.4f}")
        print(f"  Inference Lat.: {inf_time_per_1k:.2f} ms / 1k observations")
        
        # Save model artifact
        save_name = name.lower().replace(" ", "_").replace("-", "_")
        joblib.dump(final_model, f"pathsnap/artifacts/trained_models/{save_name}.joblib")
        
        # Store for report table
        results_summary.append({
            "Model": name,
            "Algorithm Family": config["family"],
            "Key Configuration": ", ".join([f"{k}={v}" for k, v in config["params"].items()]),
            "CV Macro F1-score": round(cv_macro_f1, 4),
            "Top-1 Accuracy": f"{test_top1_acc*100:.2f}%",
            "Route continuity": f"{test_continuity*100:.2f}%",
            "Student-Test Macro F1-score": round(test_macro_f1, 4),
            "Log Loss": round(test_logloss, 4),
            "Inference Time": f"{inf_time_per_1k:.1f} ms"
        })
        
        detailed_metrics[name] = {
            "family": config["family"],
            "params": config["params"],
            "cv_macro_f1": cv_macro_f1,
            "cv_std_f1": cv_std_f1,
            "cv_folds": cv_f1s,
            "student_test_macro_f1": test_macro_f1,
            "student_test_top1_acc": test_top1_acc,
            "student_test_route_continuity": test_continuity,
            "student_test_log_loss": test_logloss,
            "student_test_precision": prec,
            "student_test_recall": rec,
            "confusion_matrix": cm,
            "training_time_s": train_time,
            "inference_time_per_1k_ms": inf_time_per_1k
        }
        
    # Save results table to CSV and JSON
    df_res = pd.DataFrame(results_summary)
    df_res.to_csv("pathsnap/artifacts/results/model_comparison.csv", index=False)
    with open("pathsnap/artifacts/results/detailed_model_metrics.json", "w") as f:
        json.dump(detailed_metrics, f, indent=2)
        
    print("\n==========================================")
    print("Consolidated Model Comparison Table:")
    print("==========================================")
    print(df_res.to_string(index=False))
    print("\nResults successfully saved to pathsnap/artifacts/results/model_comparison.csv")


if __name__ == "__main__":
    train_and_evaluate_all()
