import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pathsnap.data_processing.loader import load_trip
from pathsnap.feature_engineering.extractor import CandidateExtractor
from pathsnap.hmm.viterbi import HMMRouteReconstructor
import joblib

def create_representative_map_figure(trip_id="00000000", data_dir="Student folder/Student folder/student_test"):
    print(f"Creating representative map visual for Trip {trip_id}...")
    trip = load_trip(trip_id, data_dir)
    model = joblib.load("pathsnap/artifacts/trained_models/lightgbm_tuned.joblib")
    extractor = CandidateExtractor(search_radius=40.0)
    df_cand = extractor.extract_features(trip, include_target=True)
    reconstructor = HMMRouteReconstructor()
    res = reconstructor.decode(trip, df_cand, model)

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)

    # 1. Nearby road network arcs
    nodes = trip.nodes
    arcs = trip.arcs
    lat_min, lat_max = trip.track[:, 1].min() - 0.003, trip.track[:, 1].max() + 0.003
    lon_min, lon_max = trip.track[:, 0].min() - 0.003, trip.track[:, 0].max() + 0.003

    step = max(1, len(arcs) // 3000)
    for a_idx in range(0, len(arcs), step):
        u, v = arcs[a_idx]
        p1 = [nodes[u, 0], nodes[u, 1]]
        p2 = [nodes[v, 0], nodes[v, 1]]
        if (lon_min <= p1[0] <= lon_max) and (lat_min <= p1[1] <= lat_max):
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#bdc3c7", linewidth=0.8, alpha=0.7, zorder=1)

    # Label network once for legend
    plt.plot([], [], color="#bdc3c7", linewidth=1.2, label="Road Network Segments")

    # 2. Noisy GPS points (blue dots)
    gps_lon = trip.track[:, 0]
    gps_lat = trip.track[:, 1]
    plt.scatter(gps_lon[::2], gps_lat[::2], color="#2980b9", s=14, alpha=0.6, label="Noisy GPS Observations", zorder=2)

    # 3. Ground truth reference route (solid green)
    if trip.route is not None:
        rx, ry = [], []
        for a_id in trip.route:
            u, v = trip.arcs[a_id]
            rx.extend([trip.nodes[u, 0], trip.nodes[v, 0]])
            ry.extend([trip.nodes[u, 1], trip.nodes[v, 1]])
        plt.plot(rx, ry, color="#27ae60", linewidth=3.5, alpha=0.85, label=f"Ground-Truth Route ({len(trip.route)} arcs)", zorder=3)

    # 4. Reconstructed predicted route (dashed red)
    pred_poly = np.array(res["reconstructed_polyline"]) # [lat, lon]
    plt.plot(pred_poly[:, 1], pred_poly[:, 0], color="#c0392b", linestyle="--", linewidth=2.2,
             label=f"PathSnap Reconstruction (Acc: {res['top1_accuracy_hmm']:.1f}%)", zorder=4)

    plt.xlabel("Longitude (°E)", fontsize=11, fontweight="bold")
    plt.ylabel("Latitude (°N)", fontsize=11, fontweight="bold")
    plt.title(f"Geographical Route Reconstruction - Trip {trip_id}\n(HMM Viterbi Map-Matching)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(frameon=True, facecolor="white", edgecolor="#bdc3c7", loc="best", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)

    out_file = Path("pathsnap/artifacts/figures/representative_map_view.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved {out_file}")

if __name__ == "__main__":
    create_representative_map_figure("00000000")
