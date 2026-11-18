import streamlit as st
import requests
import folium
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="PathSnap: Route Reconstruction & Map-Matching",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom header styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2563eb;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🗺️ PathSnap: Distributed Map-Matching & Route Reconstruction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">PES University Machine Learning Hackathon | Evaluator UI (Laptop 2)</div>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ System Configuration")
    st.markdown("**Laptop 2 (Evaluator UI) &rarr; Laptop 1 (Model API)**")
    
    api_url = st.text_input(
        "Laptop 1 Backend API URL:",
        value="http://localhost:8000",
        help="Enter the IP address and port of Laptop 1 (e.g. http://192.168.1.5:8000 or http://localhost:8000)"
    )
    
    # Check API Health
    api_online = False
    try:
        resp = requests.get(f"{api_url}/health", timeout=3)
        if resp.status_code == 200:
            api_online = True
            st.success("🟢 Connected to Laptop 1 Backend")
        else:
            st.error(f"🔴 API returned status {resp.status_code}")
    except Exception:
        st.warning("⚠️ Cannot connect to Laptop 1 API. Ensure backend is running.")
        
    st.divider()
    st.subheader("📋 Evaluation Trip Selection")
    
    available_trips = ["00000000", "00000007", "00000009", "00000040", "00000052",
                       "00000056", "00000065", "00000076", "00000077", "00000091"]
                       
    if api_online:
        try:
            trips_resp = requests.get(f"{api_url}/api/trips", timeout=3).json()
            available_trips = trips_resp.get("available_trips", available_trips)
        except Exception:
            pass
            
    selected_trip = st.selectbox(
        "Select an Evaluation Trip:",
        options=available_trips,
        index=0
    )
    
    eval_btn = st.button("🚀 Reconstruct & Evaluate Route", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("**Instructions for Evaluator:**")
    st.caption("1. Select any of the 10 evaluation trips from the dropdown.")
    st.caption("2. Click 'Reconstruct & Evaluate Route'.")
    st.caption("3. Real-time inference request is sent to Laptop 1.")
    st.caption("4. Ground truth stays server-side; metrics and route are returned and mapped.")
    st.caption("5. Inspect live logs on the Laptop 1 terminal.")

# Main content display
if eval_btn or "last_result" in st.session_state:
    if eval_btn:
        with st.spinner(f"Sending Trip {selected_trip} to Laptop 1 and running inference pipeline..."):
            try:
                res = requests.post(
                    f"{api_url}/api/evaluate",
                    json={"trip_id": selected_trip},
                    timeout=30
                ).json()
                st.session_state["last_result"] = res
                st.session_state["last_trip"] = selected_trip
            except Exception as e:
                st.error(f"Failed to communicate with API: {e}")
                st.stop()
                
    result = st.session_state["last_result"]
    trip_id = st.session_state.get("last_trip", selected_trip)
    
    # KPI Metric Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{result.get('top1_accuracy', 0)}%</div>
            <div class="metric-label">Observation Top-1 Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{result.get('route_continuity', 0)}%</div>
            <div class="metric-label">Route Continuity</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{result.get('total_observations', 0)}</div>
            <div class="metric-label">GPS Observations</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{result.get('unique_arcs_predicted', 0)}</div>
            <div class="metric-label">Reconstructed Arcs</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{result.get('latency_ms', 0)} ms</div>
            <div class="metric-label">Pipeline Latency</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Interactive Folium Map
    st.subheader(f"📍 Interactive Map Visualization: Trip {trip_id}")
    
    gps_pts = result.get("sampled_gps_points", [])
    pred_poly = result.get("reconstructed_polyline", [])
    gt_poly = result.get("ground_truth_polyline", [])
    
    if pred_poly:
        center_lat = np.mean([p[0] for p in pred_poly])
        center_lon = np.mean([p[1] for p in pred_poly])
    elif gps_pts:
        center_lat = np.mean([p["lat"] for p in gps_pts])
        center_lon = np.mean([p["lon"] for p in gps_pts])
    else:
        center_lat, center_lon = 49.85, 8.60
        
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles="OpenStreetMap"
    )
    
    # 1. GPS readings layer
    fg_gps = folium.FeatureGroup(name="Noisy GPS Observations", show=True)
    for pt in gps_pts:
        folium.CircleMarker(
            location=[pt["lat"], pt["lon"]],
            radius=3.5,
            color="#1976d2",
            fill=True,
            fill_color="#2196f3",
            fill_opacity=0.8,
            popup=f"GPS Obs (t={pt.get('t', 0):.0f}s)"
        ).add_to(fg_gps)
    fg_gps.add_to(fmap)
    
    # 2. Ground-truth route layer
    if gt_poly:
        fg_gt = folium.FeatureGroup(name="Ground-Truth Reference Route", show=True)
        folium.PolyLine(
            locations=gt_poly,
            color="#2e7d32",
            weight=5.5,
            opacity=0.85,
            popup=f"Ground-Truth Trajectory ({len(gt_poly)//2} arcs)"
        ).add_to(fg_gt)
        fg_gt.add_to(fmap)
        
    # 3. PathSnap reconstructed route layer
    if pred_poly:
        fg_pred = folium.FeatureGroup(name="PathSnap Reconstructed Route", show=True)
        folium.PolyLine(
            locations=pred_poly,
            color="#d32f2f",
            weight=3.5,
            dash_array="6, 6",
            opacity=0.95,
            popup=f"PathSnap Reconstruction (Top-1: {result.get('top1_accuracy')}%)"
        ).add_to(fg_pred)
        fg_pred.add_to(fmap)
        
    # Floating Legend
    legend_html = f"""
    <div style="
        position: fixed; 
        bottom: 30px; left: 30px; width: 250px; height: 130px; 
        background-color: white; border:2px solid #ccc; z-index:9999; 
        font-size:12px; font-family: sans-serif; padding: 10px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3); border-radius: 6px;
    ">
        <b>PathSnap Map Legend</b><br><br>
        <i style="background: #2196f3; width: 9px; height: 9px; display: inline-block; border-radius: 50%;"></i>&nbsp; Raw GPS Observations<br>
        <i style="background: #2e7d32; width: 16px; height: 4px; display: inline-block; vertical-align: middle;"></i>&nbsp; Ground-Truth Route<br>
        <i style="border-top: 3px dashed #d32f2f; width: 16px; display: inline-block; vertical-align: middle;"></i>&nbsp; Reconstructed Route
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(position="topright", collapsed=False).add_to(fmap)
    
    # Render map in Streamlit
    components.html(fmap._repr_html_(), height=550)
    
    # Display preview of predicted arc sequence
    with st.expander("🔍 View Reconstructed Road Segments (First 100 Arcs)"):
        arcs_pred = result.get("predicted_arcs", [])
        st.write(f"Traversed Arc Sequence (Total unique arcs: {result.get('unique_arcs_predicted')}):")
        st.code(", ".join(map(str, arcs_pred)))

else:
    st.info("👈 Select an evaluation trip from the sidebar and click **'Reconstruct & Evaluate Route'** to run live map-matching.")
    
    # Show summary of evaluation trips
    st.subheader("📊 Student-Testing Benchmark Summary")
    df_eval = pd.read_csv("pathsnap/artifacts/results/hmm_vs_classifier_comparison.csv")
    st.dataframe(df_eval, use_container_width=True)
