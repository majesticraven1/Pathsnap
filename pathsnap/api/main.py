import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

from pathsnap.data_processing.loader import load_trip, TripData
from pathsnap.feature_engineering.extractor import CandidateExtractor
from pathsnap.hmm.viterbi import HMMRouteReconstructor

# Configure robust console logging for evaluator inspection
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PathSnap API] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("PathSnapAPI")

app = FastAPI(
    title="PathSnap: Map-Matching & Route Reconstruction API",
    description="Distributed Backend Server (Laptop 1) for GPS Trajectory Map-Matching & Route Evaluation",
    version="1.0.0"
)

# Enable CORS for cross-laptop network communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and engine singletons
MODEL_PATH = Path("pathsnap/artifacts/trained_models/lightgbm_tuned.joblib")
DATA_DIR = Path("Student folder/Student folder/student_test")

model = None
extractor = None
reconstructor = None


def load_artifacts():
    global model, extractor, reconstructor
    logger.info("Initializing PathSnap Inference Pipeline...")
    if not MODEL_PATH.exists():
        logger.error(f"Trained model not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model file missing: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    extractor = CandidateExtractor(search_radius=40.0)
    reconstructor = HMMRouteReconstructor()
    logger.info("Successfully loaded tuned LightGBM model and HMM Viterbi Decoder.")
    logger.info(f"Serving evaluation dataset from: {DATA_DIR.resolve()}")

# Load artifacts on startup
load_artifacts()


class EvaluationRequest(BaseModel):
    trip_id: str


@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status": "online",
        "service": "PathSnap Backend Server (Laptop 1)",
        "model_loaded": model is not None,
        "eval_trips_directory": str(DATA_DIR)
    }


@app.get("/api/trips")
def list_available_trips():
    """
    List all available reserved evaluation trips for Laptop 2 UI dropdown.
    """
    if not DATA_DIR.exists():
        raise HTTPException(status_code=500, detail="Evaluation trips directory not found.")
    trip_dirs = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    logger.info(f"Endpoint /api/trips queried. Returning {len(trip_dirs)} available trips.")
    return {
        "available_trips": trip_dirs,
        "count": len(trip_dirs)
    }


@app.post("/api/evaluate")
def evaluate_trip(req: EvaluationRequest):
    return process_trip_evaluation(req.trip_id)


@app.get("/api/evaluate/{trip_id}")
def evaluate_trip_get(trip_id: str):
    return process_trip_evaluation(trip_id)


def process_trip_evaluation(trip_id: str):
    """
    Execute full inference pipeline and server-side ground-truth evaluation.
    Logs each step to the console for the evaluator to inspect in real time.
    """
    t_start = time.time()
    logger.info(f"==> INCOMING EVALUATION REQUEST: Trip ID '{trip_id}'")
    
    trip_path = DATA_DIR / trip_id
    if not trip_path.exists():
        logger.warning(f"Trip '{trip_id}' not found in {DATA_DIR}")
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found.")
        
    try:
        # 1. Load Trip Data
        trip = load_trip(trip_id, str(DATA_DIR))
        T = len(trip.track)
        logger.info(f"Loaded Trip {trip_id}: {T} GPS observations, {len(trip.arcs)} road network arcs.")
        
        # 2. Candidate Generation and Feature Extraction (Strictly blind without ground truth!)
        t_feat = time.time()
        df_cand = extractor.extract_features(trip, include_target=False)
        feat_duration = time.time() - t_feat
        logger.info(f"Extracted {len(df_cand)} candidate pairs in {feat_duration*1000:.1f}ms ({len(df_cand)/max(1,T):.1f} cands/pt).")
        
        # 3. Model Scoring & HMM Viterbi Sequential Route Decoding
        t_decode = time.time()
        res = reconstructor.decode(trip, df_cand, model)
        decode_duration = time.time() - t_decode
        logger.info(f"HMM Viterbi decoding completed in {decode_duration*1000:.1f}ms.")
        
        # 4. Server-Side Ground Truth Evaluation (Labels remain server-side)
        top1_acc = 0.0
        if trip.route is not None:
            true_arcs = extractor.match_ground_truth(trip)
            decoded_arr = np.array(res["decoded_arcs"])
            correct_pts = np.sum(decoded_arr == true_arcs)
            top1_acc = float((correct_pts / max(1, T)) * 100.0)
            logger.info(f"SERVER-SIDE EVALUATION: Observation Top-1 Accuracy: {top1_acc:.2f}% ({correct_pts}/{T} correct).")
            
        cont = res["route_continuity_hmm"]
        logger.info(f"SERVER-SIDE EVALUATION: Reconstructed Route Continuity: {cont:.2f}%.")
        
        total_time_ms = (time.time() - t_start) * 1000.0
        logger.info(f"<== COMPLETED Trip '{trip_id}' | Latency: {total_time_ms:.1f}ms | Acc: {top1_acc:.2f}% | Cont: {cont:.2f}%\n")
        
        # Sample GPS points for UI rendering (up to 400 points)
        step = max(1, T // 400)
        sampled_gps = [
            {"lat": float(trip.track[i, 1]), "lon": float(trip.track[i, 0]), "t": float(trip.track[i, 2])}
            for i in range(0, T, step)
        ]
        
        # Ground truth polyline (lat, lon) for visual overlay on evaluator UI
        gt_polyline = []
        if trip.route is not None:
            seen_gt = []
            for a in trip.route:
                if not seen_gt or seen_gt[-1] != a:
                    seen_gt.append(a)
                    u, v = trip.arcs[a, 0], trip.arcs[a, 1]
                    gt_polyline.append([float(trip.nodes[u, 1]), float(trip.nodes[u, 0])])
                    gt_polyline.append([float(trip.nodes[v, 1]), float(trip.nodes[v, 0])])
                    
        return {
            "status": "success",
            "trip_id": trip_id,
            "total_observations": T,
            "unique_arcs_predicted": res["unique_arcs_count"],
            "top1_accuracy": round(top1_acc, 2),
            "route_continuity": round(cont, 2),
            "latency_ms": round(total_time_ms, 1),
            "predicted_arcs": res["decoded_arcs"][:100], # First 100 arcs
            "reconstructed_polyline": res["reconstructed_polyline"],
            "ground_truth_polyline": gt_polyline,
            "sampled_gps_points": sampled_gps
        }
        
    except Exception as e:
        logger.error(f"Error evaluating trip {trip_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Laptop 1 serves on 0.0.0.0:8000 so Laptop 2 can reach it over local Wi-Fi / LAN
    uvicorn.run("pathsnap.api.main:app", host="0.0.0.0", port=8000, reload=False)
