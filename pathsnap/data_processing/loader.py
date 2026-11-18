import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class TripData:
    trip_id: str
    nodes: np.ndarray          # shape (N, 2): [lon, lat]
    arcs: np.ndarray           # shape (M, 2): [from_node, to_node]
    track: np.ndarray          # shape (T, 3): [lon, lat, timestamp]
    route: Optional[np.ndarray] # shape (K,): arc_ids traversed
    proj_nodes: np.ndarray     # shape (N, 2): [x, y] in meters
    proj_track: np.ndarray     # shape (T, 2): [x, y] in meters
    mean_lat: float
    m_per_deg_lat: float
    m_per_deg_lon: float


def project_coordinates(lon: np.ndarray, lat: np.ndarray, mean_lat: float) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Project angular coordinates (lon, lat) to planar metric coordinates (x, y)
    using the WGS84 ellipsoidal local projection around the trajectory center.
    This preserves physical distances in meters accurately, avoiding Euclidean distortion on degrees.
    """
    rad_lat = np.radians(mean_lat)
    cos_lat = np.cos(rad_lat)
    
    # WGS-84 metric distance approximations per degree
    m_per_deg_lat = 111132.92 - 559.82 * np.cos(2 * rad_lat) + 1.175 * np.cos(4 * rad_lat)
    m_per_deg_lon = 111412.84 * cos_lat - 93.5 * np.cos(3 * rad_lat)
    
    x = lon * m_per_deg_lon
    y = lat * m_per_deg_lat
    return x, y, m_per_deg_lat, m_per_deg_lon


def unproject_coordinates(x: np.ndarray, y: np.ndarray, m_per_deg_lat: float, m_per_deg_lon: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert local metric coordinates back to (longitude, latitude).
    """
    lon = x / m_per_deg_lon
    lat = y / m_per_deg_lat
    return lon, lat


def load_trip(trip_id: str, data_dir: str = "Student folder/Student folder/train") -> TripData:
    """
    Load raw .nodes, .arcs, .track, and optional .route files for a given trip.
    """
    trip_path = Path(data_dir) / trip_id
    if not trip_path.exists():
        raise FileNotFoundError(f"Trip directory not found: {trip_path}")
        
    nodes_file = trip_path / f"{trip_id}.nodes"
    arcs_file = trip_path / f"{trip_id}.arcs"
    track_file = trip_path / f"{trip_id}.track"
    route_file = trip_path / f"{trip_id}.route"
    
    # Load nodes: [lon, lat]
    df_nodes = pd.read_csv(nodes_file, sep=r"\s+", header=None, names=["lon", "lat"])
    nodes = df_nodes[["lon", "lat"]].to_numpy(dtype=np.float64)
    
    # Load arcs: [from_node, to_node]
    df_arcs = pd.read_csv(arcs_file, sep=r"\s+", header=None, names=["from_node", "to_node"])
    arcs = df_arcs[["from_node", "to_node"]].to_numpy(dtype=np.int64)
    
    # Load track: [lon, lat, timestamp]
    df_track = pd.read_csv(track_file, sep=r"\s+", header=None, names=["lon", "lat", "timestamp"])
    track = df_track[["lon", "lat", "timestamp"]].to_numpy(dtype=np.float64)
    
    # Load route if exists
    route = None
    if route_file.exists():
        df_route = pd.read_csv(route_file, header=None, names=["arc_id"])
        route = df_route["arc_id"].to_numpy(dtype=np.int64)
        
    # Project coordinates
    mean_lat = float(np.mean(track[:, 1]))
    nx, ny, m_lat, m_lon = project_coordinates(nodes[:, 0], nodes[:, 1], mean_lat)
    proj_nodes = np.column_stack([nx, ny])
    
    tx, ty, _, _ = project_coordinates(track[:, 0], track[:, 1], mean_lat)
    proj_track = np.column_stack([tx, ty])
    
    return TripData(
        trip_id=trip_id,
        nodes=nodes,
        arcs=arcs,
        track=track,
        route=route,
        proj_nodes=proj_nodes,
        proj_track=proj_track,
        mean_lat=mean_lat,
        m_per_deg_lat=m_lat,
        m_per_deg_lon=m_lon
    )
