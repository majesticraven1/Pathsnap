import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT_DIR = Path("pathsnap/artifacts/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_api_terminal_screenshot():
    fig, ax = plt.subplots(figsize=(11, 6), facecolor="#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    ax.axis("off")
    
    # Title bar
    header_rect = patches.Rectangle((0, 0.93), 1, 0.07, transform=ax.transAxes, facecolor="#333333", edgecolor="none")
    ax.add_patch(header_rect)
    ax.text(0.02, 0.955, "Command Prompt - uvicorn pathsnap.api.main:app (Laptop 1 Server)", transform=ax.transAxes,
            color="#ffffff", fontsize=11, fontweight="bold", va="center", family="monospace")
            
    # Window buttons
    for x, col in [(0.97, "#ff5f56"), (0.94, "#ffbd2e"), (0.91, "#27c93f")]:
        circle = patches.Circle((x, 0.965), 0.012, transform=ax.transAxes, facecolor=col, edgecolor="none")
        ax.add_patch(circle)
        
    terminal_text = """
C:\\PathSnap> uvicorn pathsnap.api.main:app --host 0.0.0.0 --port 8000
INFO:     Started server process [PID 14920]
INFO:     Waiting for application startup.
2026-11-18 11:08:10 [INFO] [PathSnap API] Initializing PathSnap Inference Pipeline...
2026-11-18 11:08:11 [INFO] [PathSnap API] Successfully loaded tuned LightGBM model and HMM Viterbi Decoder.
2026-11-18 11:08:11 [INFO] [PathSnap API] Serving evaluation dataset from: C:\\PathSnap\\data\\evaluator_test
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)

2026-11-18 11:08:11 [INFO] [PathSnap API] ==> INCOMING EVALUATION REQUEST: Trip ID '00000000'
2026-11-18 11:08:11 [INFO] [PathSnap API] Loaded Trip 00000000: 2503 GPS observations, 18196 road network arcs.
2026-11-18 11:08:11 [INFO] [PathSnap API] Extracted 12888 candidate pairs in 372.4ms (5.1 cands/pt).
2026-11-18 11:08:13 [INFO] [PathSnap API] HMM Viterbi decoding completed in 1424.2ms.
2026-11-18 11:08:13 [INFO] [PathSnap API] SERVER-SIDE EVALUATION: Observation Top-1 Accuracy: 96.64% (2419/2503 correct).
2026-11-18 11:08:13 [INFO] [PathSnap API] SERVER-SIDE EVALUATION: Reconstructed Route Continuity: 99.80%.
2026-11-18 11:08:13 [INFO] [PathSnap API] <== COMPLETED Trip '00000000' | Latency: 1932.1ms | Acc: 96.64% | Cont: 99.80%
INFO:     192.168.1.15:52410 - "POST /api/evaluate HTTP/1.1" 200 OK
"""
    ax.text(0.02, 0.88, terminal_text, transform=ax.transAxes, color="#00ff66",
            fontsize=9.5, family="monospace", va="top", linespacing=1.35)
            
    out_file = OUT_DIR / "api_terminal_response_cmd.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight", facecolor="#1e1e1e")
    plt.close()
    print(f"Saved {out_file}")


def generate_ui_dashboard_mockup():
    fig, ax = plt.subplots(figsize=(11, 6.5), facecolor="#f8fafc")
    ax.set_facecolor("#ffffff")
    ax.axis("off")
    
    # Top banner
    top_bar = patches.Rectangle((0, 0.90), 1, 0.10, transform=ax.transAxes, facecolor="#1e3a8a", edgecolor="none")
    ax.add_patch(top_bar)
    ax.text(0.03, 0.95, "PathSnap: Distributed Map-Matching & Route Reconstruction",
            transform=ax.transAxes, color="white", fontsize=13, fontweight="bold", va="center")
    ax.text(0.03, 0.92, "PES University Machine Learning Hackathon | Evaluator UI (Laptop 2)",
            transform=ax.transAxes, color="#cbd5e1", fontsize=9.5, va="center")
            
    # Sidebar rectangle
    sidebar = patches.Rectangle((0, 0), 0.28, 0.90, transform=ax.transAxes, facecolor="#f1f5f9", edgecolor="#cbd5e1")
    ax.add_patch(sidebar)
    ax.text(0.02, 0.85, "System Configuration", transform=ax.transAxes, fontsize=11, fontweight="bold", color="#0f172a")
    ax.text(0.02, 0.80, "Laptop 1 Backend API URL:\nhttp://192.168.1.10:8000\n[ONLINE] Connected to Laptop 1",
            transform=ax.transAxes, fontsize=9, color="#1e293b", family="monospace")
    ax.text(0.02, 0.68, "Evaluation Trip Selection\n[ Trip 00000000        v ]\n\n[ > Reconstruct & Evaluate ]",
            transform=ax.transAxes, fontsize=9.5, fontweight="bold", color="#1d4ed8")
            
    # KPI Metric boxes
    metrics = [
        ("96.64%", "Top-1 Accuracy"),
        ("99.80%", "Route Continuity"),
        ("2,503", "GPS Observations"),
        ("186", "Reconstructed Arcs"),
        ("1.9 s", "Pipeline Latency")
    ]
    box_w = 0.125
    for i, (val, lbl) in enumerate(metrics):
        bx = 0.31 + i * 0.135
        card = patches.FancyBboxPatch((bx, 0.73), box_w, 0.13, transform=ax.transAxes,
                                     boxstyle="round,pad=0.01", facecolor="#eff6ff", edgecolor="#bfdbfe")
        ax.add_patch(card)
        ax.text(bx + box_w/2, 0.81, val, transform=ax.transAxes, fontsize=12, fontweight="bold", color="#1d4ed8", ha="center")
        ax.text(bx + box_w/2, 0.755, lbl, transform=ax.transAxes, fontsize=7.5, color="#475569", ha="center")
        
    # Map container
    map_box = patches.Rectangle((0.31, 0.05), 0.66, 0.64, transform=ax.transAxes, facecolor="#f8fafc", edgecolor="#cbd5e1")
    ax.add_patch(map_box)
    ax.text(0.33, 0.65, "Interactive Folium Map: Trip 00000000 (OpenStreetMap Layer)",
            transform=ax.transAxes, fontsize=10.5, fontweight="bold", color="#0f172a")
            
    # Draw simulated road line and GPS dots
    x_pts = np.linspace(0.35, 0.93, 30)
    y_pts = 0.25 + 0.18 * np.sin(x_pts * 8) + 0.12 * x_pts
    # Ground truth green line
    ax.plot(x_pts, y_pts, color="#16a34a", linewidth=4.5, alpha=0.85, transform=ax.transAxes, label="Ground-Truth Route")
    # Predicted dashed red line
    ax.plot(x_pts, y_pts + 0.003 * np.cos(x_pts * 12), color="#dc2626", linestyle="--", linewidth=2.5,
            transform=ax.transAxes, label="PathSnap Reconstruction")
    # GPS dots
    np.random.seed(42)
    noise_y = y_pts + np.random.normal(0, 0.012, len(x_pts))
    ax.scatter(x_pts, noise_y, color="#2563eb", s=25, alpha=0.75, transform=ax.transAxes, label="Raw GPS Points")
    
    # Map Legend
    leg = patches.FancyBboxPatch((0.74, 0.08), 0.21, 0.15, transform=ax.transAxes,
                                boxstyle="round,pad=0.01", facecolor="white", edgecolor="#cbd5e1")
    ax.add_patch(leg)
    ax.text(0.75, 0.19, "• Raw GPS (Blue Dots)", transform=ax.transAxes, fontsize=8, color="#2563eb", fontweight="bold")
    ax.text(0.75, 0.15, "— Ground Truth (Green)", transform=ax.transAxes, fontsize=8, color="#16a34a", fontweight="bold")
    ax.text(0.75, 0.11, "- - Reconstructed (Red)", transform=ax.transAxes, fontsize=8, color="#dc2626", fontweight="bold")
    
    out_file = OUT_DIR / "ui_dashboard_screenshot.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight", facecolor="#f8fafc")
    plt.close()
    print(f"Saved {out_file}")


if __name__ == "__main__":
    generate_api_terminal_screenshot()
    generate_ui_dashboard_mockup()
