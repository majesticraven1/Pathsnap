import os
from pathlib import Path
import folium
from folium import plugins
import numpy as np
import pandas as pd
from pathsnap.data_processing.loader import load_trip, TripData
from pathsnap.feature_engineering.extractor import CandidateExtractor
from pathsnap.hmm.viterbi import HMMRouteReconstructor
import joblib


def generate_trip_folium_map(
    trip_id: str,
    data_dir: str = "Student folder/Student folder/student_test",
    output_html: str = "pathsnap/artifacts/maps/map_trip_00000000.html",
    include_network_arcs: bool = True
) -> str:
    """
    Generate an interactive Folium map visualizing:
    - Raw GPS observations (blue circle markers)
    - Ground-truth reference route (green line)
    - Predicted reconstructed route (red dashed/solid line)
    - Road network geometry (light grey lines)
    """
    print(f"Generating Folium interactive map for Trip {trip_id}...")
    trip = load_trip(trip_id, data_dir)
    
    # Load trained model and decode route
    model = joblib.load("pathsnap/artifacts/trained_models/lightgbm_tuned.joblib")
    extractor = CandidateExtractor(search_radius=40.0)
    df_cand = extractor.extract_features(trip, include_target=True)
    
    reconstructor = HMMRouteReconstructor()
    res = reconstructor.decode(trip, df_cand, model)
    
    # Calculate map center
    mean_lat = float(np.mean(trip.track[:, 1]))
    mean_lon = float(np.mean(trip.track[:, 0]))
    
    m = folium.Map(
        location=[mean_lat, mean_lon],
        zoom_start=15,
        tiles="OpenStreetMap"
    )
    
    # Layer 1: Road Network Sub-graph (Optional nearby arcs)
    if include_network_arcs:
        fg_network = folium.FeatureGroup(name="Road Network (Nearby Arcs)", show=False)
        # Sample nearby arcs around trajectory bounding box to avoid heavy DOM
        nodes = trip.nodes
        arcs = trip.arcs
        lat_min, lat_max = trip.track[:, 1].min() - 0.005, trip.track[:, 1].max() + 0.005
        lon_min, lon_max = trip.track[:, 0].min() - 0.005, trip.track[:, 0].max() + 0.005
        
        sample_step = max(1, len(arcs) // 3000)
        for a_idx in range(0, len(arcs), sample_step):
            u, v = arcs[a_idx]
            p1 = [nodes[u, 1], nodes[u, 0]]
            p2 = [nodes[v, 1], nodes[v, 0]]
            if (lat_min <= p1[0] <= lat_max) and (lon_min <= p1[1] <= lon_max):
                folium.PolyLine(
                    locations=[p1, p2],
                    color="#9e9e9e",
                    weight=1.5,
                    opacity=0.6
                ).add_to(fg_network)
        fg_network.add_to(m)
        
    # Layer 2: Raw GPS Observations
    fg_gps = folium.FeatureGroup(name="Noisy GPS Observations", show=True)
    # Downsample points slightly if > 1500 to keep HTML smooth
    step = 1 if len(trip.track) < 1500 else 2
    for t in range(0, len(trip.track), step):
        lon, lat, ts = trip.track[t]
        folium.CircleMarker(
            location=[lat, lon],
            radius=3.5,
            color="#1976d2",
            fill=True,
            fill_color="#2196f3",
            fill_opacity=0.8,
            popup=f"GPS #{t} (t={ts:.0f}s)"
        ).add_to(fg_gps)
    fg_gps.add_to(m)
    
    # Layer 3: Ground-Truth Reference Route (if route provided)
    if trip.route is not None:
        fg_truth = folium.FeatureGroup(name="Ground-Truth Reference Route", show=True)
        truth_coords = []
        for a_id in trip.route:
            u, v = trip.arcs[a_id]
            truth_coords.append([trip.nodes[u, 1], trip.nodes[u, 0]])
            truth_coords.append([trip.nodes[v, 1], trip.nodes[v, 0]])
        folium.PolyLine(
            locations=truth_coords,
            color="#2e7d32",
            weight=5.5,
            opacity=0.85,
            popup=f"Ground-Truth Route ({len(trip.route)} arcs)"
        ).add_to(fg_truth)
        fg_truth.add_to(m)
        
    # Layer 4: Reconstructed Predicted Route
    fg_pred = folium.FeatureGroup(name="Reconstructed PathSnap Route", show=True)
    folium.PolyLine(
        locations=res["reconstructed_polyline"],
        color="#d32f2f",
        weight=3.5,
        dash_array="6, 6",
        opacity=0.95,
        popup=f"PathSnap Prediction (Top-1: {res['top1_accuracy_hmm']:.1f}%, Continuity: {res['route_continuity_hmm']:.1f}%)"
    ).add_to(fg_pred)
    fg_pred.add_to(m)
    
    # Add floating HTML Legend
    legend_html = f"""
    <div style="
        position: fixed; 
        bottom: 30px; left: 30px; width: 260px; height: 160px; 
        background-color: white; border:2px solid #ccc; z-index:9999; 
        font-size:13px; font-family: sans-serif; padding: 12px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3); border-radius: 6px;
    ">
        <b>PathSnap Route Reconstruction</b><br>
        <span style="font-size: 11px; color: #555;">Trip: {trip_id} | Acc: {res['top1_accuracy_hmm']:.1f}%</span><br><br>
        <i style="background: #2196f3; width: 10px; height: 10px; display: inline-block; border-radius: 50%;"></i>&nbsp; Raw GPS Observations<br>
        <i style="background: #2e7d32; width: 18px; height: 4px; display: inline-block; vertical-align: middle;"></i>&nbsp; Ground-Truth Route<br>
        <i style="border-top: 3px dashed #d32f2f; width: 18px; display: inline-block; vertical-align: middle;"></i>&nbsp; PathSnap Reconstruction<br>
        <i style="background: #9e9e9e; width: 18px; height: 2px; display: inline-block; vertical-align: middle;"></i>&nbsp; Road Network Arcs
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add Layer Control
    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    
    # Save map
    out_path = Path(output_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    print(f"Map successfully saved to {out_path}")
    return str(out_path)


if __name__ == "__main__":
    generate_trip_folium_map("00000000", output_html="pathsnap/artifacts/maps/map_trip_00000000.html")
    generate_trip_folium_map("00000040", output_html="pathsnap/artifacts/maps/map_trip_00000040.html")
