import os
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathsnap.data_processing.loader import load_trip
from pathsnap.feature_engineering.extractor import CandidateExtractor


def build_feature_datasets():
    train_dir = Path("Student folder/Student folder/train")
    test_dir = Path("Student folder/Student folder/student_test")
    
    extractor = CandidateExtractor(search_radius=40.0)
    
    train_trips = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    test_trips = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
    
    print(f"Starting feature extraction for {len(train_trips)} train trips...")
    train_dfs = []
    t0 = time.time()
    
    train_stats = {
        "total_trips": len(train_trips),
        "trip_details": {},
        "total_gps_points": 0,
        "total_candidates": 0,
        "total_positives": 0
    }
    
    for tid in train_trips:
        trip = load_trip(tid, str(train_dir))
        df = extractor.extract_features(trip, include_target=True)
        train_dfs.append(df)
        
        pts = len(trip.track)
        cands = len(df)
        pos = int(df["target"].sum()) if "target" in df else 0
        train_stats["total_gps_points"] += pts
        train_stats["total_candidates"] += cands
        train_stats["total_positives"] += pos
        train_stats["trip_details"][tid] = {
            "pts": pts, "cands": cands, "pos": pos,
            "cands_per_pt": round(cands / max(1, pts), 2)
        }
        
    df_train = pd.concat(train_dfs, ignore_index=True)
    out_train = Path("pathsnap/artifacts/features_train.parquet")
    out_train.parent.mkdir(parents=True, exist_ok=True)
    df_train.to_parquet(out_train, index=False)
    print(f"Saved {len(df_train)} train candidates to {out_train} in {time.time()-t0:.2f}s")
    
    print(f"\nStarting feature extraction for {len(test_trips)} student_test trips...")
    test_dfs = []
    t1 = time.time()
    test_stats = {
        "total_trips": len(test_trips),
        "trip_details": {},
        "total_gps_points": 0,
        "total_candidates": 0,
        "total_positives": 0
    }
    
    for tid in test_trips:
        trip = load_trip(tid, str(test_dir))
        df = extractor.extract_features(trip, include_target=True)
        test_dfs.append(df)
        
        pts = len(trip.track)
        cands = len(df)
        pos = int(df["target"].sum()) if "target" in df else 0
        test_stats["total_gps_points"] += pts
        test_stats["total_candidates"] += cands
        test_stats["total_positives"] += pos
        test_stats["trip_details"][tid] = {
            "pts": pts, "cands": cands, "pos": pos,
            "cands_per_pt": round(cands / max(1, pts), 2)
        }
        
    df_test = pd.concat(test_dfs, ignore_index=True)
    out_test = Path("pathsnap/artifacts/features_student_test.parquet")
    df_test.to_parquet(out_test, index=False)
    print(f"Saved {len(df_test)} student_test candidates to {out_test} in {time.time()-t1:.2f}s")
    
    eda_summary = {
        "train": train_stats,
        "student_test": test_stats,
        "feature_names": [
            "dist_to_seg", "heading_diff", "sin_hdiff", "cos_hdiff",
            "speed", "u_proj", "arc_len", "dist_to_start", "dist_to_end", "dt"
        ]
    }
    
    with open("pathsnap/artifacts/results/eda_summary.json", "w") as f:
        json.dump(eda_summary, f, indent=2)
    print("EDA summary saved to pathsnap/artifacts/results/eda_summary.json")


if __name__ == "__main__":
    build_feature_datasets()
