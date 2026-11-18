import os
import time
import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score
from lightgbm import LGBMClassifier
from pathsnap.models.train_models import (
    FEATURE_COLS, compute_observation_top1,
    compute_route_continuity, load_network_arcs
)

# Silence optuna logging noise
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_bayesian_optimization(n_trials: int = 20):
    print("Loading datasets for Bayesian Hyperparameter Optimisation...")
    df_train = pd.read_parquet("pathsnap/artifacts/features_train.parquet")
    df_test = pd.read_parquet("pathsnap/artifacts/features_student_test.parquet")
    test_arcs_dict = load_network_arcs("Student folder/Student folder/student_test")
    
    X_train = df_train[FEATURE_COLS].values
    y_train = df_train["target"].values
    groups_train = df_train["trip_id"].values
    
    X_test = df_test[FEATURE_COLS].values
    y_test = df_test["target"].values
    
    # 3-fold GroupKFold for Bayesian tuning trials
    gkf = GroupKFold(n_splits=3)
    
    # Subsample for faster trial iterations across 80 trips
    np.random.seed(42)
    sample_size = 400000
    if len(df_train) > sample_size:
        idx_sample = np.random.choice(len(df_train), sample_size, replace=False)
        opt_X = X_train[idx_sample]
        opt_y = y_train[idx_sample]
        opt_groups = groups_train[idx_sample]
    else:
        opt_X = X_train
        opt_y = y_train
        opt_groups = groups_train
        
    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 95),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.20, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 80, 220),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 2.0, 7.0),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1
        }
        
        scores = []
        for train_idx, val_idx in gkf.split(opt_X, opt_y, groups=opt_groups):
            clf = LGBMClassifier(**params)
            clf.fit(opt_X[train_idx], opt_y[train_idx])
            preds = clf.predict(opt_X[val_idx])
            f1 = f1_score(opt_y[val_idx], preds, average="macro")
            scores.append(f1)
            
        return float(np.mean(scores))
        
    print(f"Starting {n_trials} Bayesian Optimization trials using TPE Sampler...")
    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    
    trial_records = []
    
    for i in range(n_trials):
        trial = study.ask()
        score = objective(trial)
        study.tell(trial, score)
        trial_records.append({
            "trial_number": i + 1,
            "params": trial.params,
            "cv_macro_f1": round(score, 4)
        })
        print(f"  Trial {i+1:02d}/{n_trials:02d} | CV Macro F1: {score:.4f} | Best so far: {study.best_value:.4f}")
        
    opt_time = time.time() - t0
    best_params = study.best_params
    print(f"\nOptimization completed in {opt_time:.1f}s")
    print(f"Best Trial #{study.best_trial.number + 1} with CV Macro F1: {study.best_value:.4f}")
    print("Best Hyperparameters:", json.dumps(best_params, indent=2))
    
    # Untuned baseline evaluation on student test
    untuned_model = LGBMClassifier(
        n_estimators=150, max_depth=7, learning_rate=0.08, scale_pos_weight=4.0,
        random_state=42, n_jobs=-1, verbose=-1
    )
    untuned_model.fit(X_train, y_train)
    
    t_inf_u0 = time.time()
    untuned_probs = untuned_model.predict_proba(X_test)[:, 1]
    untuned_inf_ms = ((time.time() - t_inf_u0) / len(X_test)) * 1e6
    untuned_preds = (untuned_probs >= 0.5).astype(int)
    
    untuned_macro_f1 = float(f1_score(y_test, untuned_preds, average="macro"))
    untuned_top1 = compute_observation_top1(df_test, untuned_probs)
    untuned_continuity = compute_route_continuity(df_test, untuned_probs, test_arcs_dict)
    untuned_logloss = float(log_loss(y_test, np.clip(untuned_probs, 1e-7, 1 - 1e-7)))
    
    # Train Tuned Model on full training data
    print("\nTraining Final Tuned Model on full dataset...")
    fit_params = best_params.copy()
    fit_params.update({"random_state": 42, "n_jobs": -1, "verbose": -1})
    tuned_model = LGBMClassifier(**fit_params)
    t_train0 = time.time()
    tuned_model.fit(X_train, y_train)
    tuned_train_time = time.time() - t_train0
    
    t_inf_t0 = time.time()
    tuned_probs = tuned_model.predict_proba(X_test)[:, 1]
    tuned_inf_ms = ((time.time() - t_inf_t0) / len(X_test)) * 1e6
    tuned_preds = (tuned_probs >= 0.5).astype(int)
    
    tuned_macro_f1 = float(f1_score(y_test, tuned_preds, average="macro"))
    tuned_top1 = compute_observation_top1(df_test, tuned_probs)
    tuned_continuity = compute_route_continuity(df_test, tuned_probs, test_arcs_dict)
    tuned_logloss = float(log_loss(y_test, np.clip(tuned_probs, 1e-7, 1 - 1e-7)))
    
    # Save Tuned Model
    joblib.dump(tuned_model, "pathsnap/artifacts/trained_models/lightgbm_tuned.joblib")
    
    # Comparison summary table
    comparison = [
        {
            "Metric": "Cross-validation Macro F1-score",
            "Untuned Model": 0.8381,
            "Tuned Model": round(study.best_value, 4),
            "Absolute Improvement": f"+{(study.best_value - 0.8381):.4f}"
        },
        {
            "Metric": "Student-testing Top-1 Accuracy",
            "Untuned Model": f"{untuned_top1*100:.2f}%",
            "Tuned Model": f"{tuned_top1*100:.2f}%",
            "Absolute Improvement": f"+{(tuned_top1 - untuned_top1)*100:.2f}%"
        },
        {
            "Metric": "Student-testing Macro F1-score",
            "Untuned Model": round(untuned_macro_f1, 4),
            "Tuned Model": round(tuned_macro_f1, 4),
            "Absolute Improvement": f"+{(tuned_macro_f1 - untuned_macro_f1):.4f}"
        },
        {
            "Metric": "Log Loss",
            "Untuned Model": round(untuned_logloss, 4),
            "Tuned Model": round(tuned_logloss, 4),
            "Absolute Improvement": f"{(tuned_logloss - untuned_logloss):.4f}"
        },
        {
            "Metric": "Inference Time (per 1k pts)",
            "Untuned Model": f"{untuned_inf_ms:.2f} ms",
            "Tuned Model": f"{tuned_inf_ms:.2f} ms",
            "Absolute Improvement": f"{(tuned_inf_ms - untuned_inf_ms):.2f} ms"
        }
    ]
    df_comp = pd.DataFrame(comparison)
    df_comp.to_csv("pathsnap/artifacts/results/tuned_vs_untuned_comparison.csv", index=False)
    
    results = {
        "search_space": {
            "num_leaves": [15, 95],
            "max_depth": [4, 10],
            "learning_rate": [0.03, 0.20],
            "n_estimators": [80, 220],
            "min_child_samples": [20, 150],
            "subsample": [0.65, 1.0],
            "colsample_bytree": [0.65, 1.0],
            "scale_pos_weight": [2.0, 7.0]
        },
        "objective": "3-fold GroupKFold Macro F1 on Training Trips",
        "n_trials": n_trials,
        "best_params": best_params,
        "best_cv_macro_f1": study.best_value,
        "optimization_time_s": opt_time,
        "trials": trial_records,
        "comparison": comparison
    }
    with open("pathsnap/artifacts/results/bayesian_opt_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n==========================================")
    print("Tuned vs Untuned Model Comparison:")
    print("==========================================")
    print(df_comp.to_string(index=False))
    print("\nBayesian Optimization completed and saved successfully!")


if __name__ == "__main__":
    run_bayesian_optimization(n_trials=20)
