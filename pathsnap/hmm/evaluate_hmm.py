import os
import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from pathsnap.data_processing.loader import load_trip
from pathsnap.feature_engineering.extractor import CandidateExtractor
from pathsnap.hmm.viterbi import HMMRouteReconstructor


def evaluate_all_student_test_trips():
    print("Evaluating Classifier vs Classifier + HMM across all 10 student_test trips...")
    test_dir = Path("Student folder/Student folder/student_test")
    test_trips = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
    
    extractor = CandidateExtractor(search_radius=40.0)
    reconstructor = HMMRouteReconstructor()
    model = joblib.load("pathsnap/artifacts/trained_models/lightgbm_tuned.joblib")
    
    trip_results = []
    
    for tid in test_trips:
        trip = load_trip(tid, str(test_dir))
        df_cand = extractor.extract_features(trip, include_target=True)
        res = reconstructor.decode(trip, df_cand, model)
        
        trip_results.append({
            "Trip ID": tid,
            "Observations": res["total_observations"],
            "Unique Arcs": res["unique_arcs_count"],
            "Classifier Top-1 (%)": round(res["top1_accuracy_classifier"], 2),
            "HMM Top-1 (%)": round(res["top1_accuracy_hmm"], 2),
            "Top-1 Improvement (%)": round(res["accuracy_improvement"], 2),
            "Classifier Continuity (%)": round(res["route_continuity_classifier"], 2),
            "HMM Continuity (%)": round(res["route_continuity_hmm"], 2),
            "Continuity Improvement (%)": round(res["continuity_improvement"], 2)
        })
        print(f"Trip {tid}: Clf Acc={res['top1_accuracy_classifier']:.2f}% -> HMM Acc={res['top1_accuracy_hmm']:.2f}% | Clf Cont={res['route_continuity_classifier']:.2f}% -> HMM Cont={res['route_continuity_hmm']:.2f}%")
        
    df_res = pd.DataFrame(trip_results)
    df_res.to_csv("pathsnap/artifacts/results/hmm_vs_classifier_comparison.csv", index=False)
    
    overall_clf_acc = df_res["Classifier Top-1 (%)"].mean()
    overall_hmm_acc = df_res["HMM Top-1 (%)"].mean()
    overall_clf_cont = df_res["Classifier Continuity (%)"].mean()
    overall_hmm_cont = df_res["HMM Continuity (%)"].mean()
    
    summary = {
        "trips": trip_results,
        "mean_classifier_top1_acc": round(overall_clf_acc, 2),
        "mean_hmm_top1_acc": round(overall_hmm_acc, 2),
        "mean_classifier_continuity": round(overall_clf_cont, 2),
        "mean_hmm_continuity": round(overall_hmm_cont, 2)
    }
    with open("pathsnap/artifacts/results/hmm_evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print("\n==========================================")
    print("HMM vs Classifier-Only Comparison across Student Test Trips:")
    print("==========================================")
    print(df_res.to_string(index=False))
    print(f"\nAverage Top-1 Accuracy:  Classifier={overall_clf_acc:.2f}%  -->  HMM={overall_hmm_acc:.2f}% (+{overall_hmm_acc-overall_clf_acc:.2f}%)")
    print(f"Average Continuity:      Classifier={overall_clf_cont:.2f}%  -->  HMM={overall_hmm_cont:.2f}% (+{overall_hmm_cont-overall_clf_cont:.2f}%)")


if __name__ == "__main__":
    evaluate_all_student_test_trips()
