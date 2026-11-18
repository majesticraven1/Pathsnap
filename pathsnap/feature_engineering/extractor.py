import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from shapely.geometry import LineString, Point, box
from shapely.strtree import STRtree
from pathsnap.data_processing.loader import TripData


def compute_point_to_segment(
    px: np.ndarray, py: np.ndarray,
    x1: np.ndarray, y1: np.ndarray,
    x2: np.ndarray, y2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized computation of distance and projection scalar u for points to segments.
    """
    dx = x2 - x1
    dy = y2 - y1
    L2 = dx * dx + dy * dy
    L2_safe = np.maximum(1e-6, L2)
    
    # Projection parameter u in [0, 1]
    u = np.clip(((px - x1) * dx + (py - y1) * dy) / L2_safe, 0.0, 1.0)
    
    proj_x = x1 + u * dx
    proj_y = y1 + u * dy
    
    dist_to_seg = np.hypot(px - proj_x, py - proj_y)
    dist_to_start = np.hypot(px - x1, py - y1)
    dist_to_end = np.hypot(px - x2, py - y2)
    seg_len = np.sqrt(L2)
    
    return dist_to_seg, dist_to_start, dist_to_end, u, seg_len


class CandidateExtractor:
    def __init__(self, search_radius: float = 40.0, fallback_k: int = 3):
        self.search_radius = search_radius
        self.fallback_k = fallback_k

    def build_spatial_index(self, trip: TripData) -> Tuple[STRtree, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build an STRtree over the arcs in the bounding box of the GPS track.
        """
        u_nodes = trip.arcs[:, 0]
        v_nodes = trip.arcs[:, 1]
        
        ax1 = trip.proj_nodes[u_nodes, 0]
        ay1 = trip.proj_nodes[u_nodes, 1]
        ax2 = trip.proj_nodes[v_nodes, 0]
        ay2 = trip.proj_nodes[v_nodes, 1]
        
        tx = trip.proj_track[:, 0]
        ty = trip.proj_track[:, 1]
        
        margin = max(self.search_radius * 3, 200.0)
        min_x, max_x = tx.min() - margin, tx.max() + margin
        min_y, max_y = ty.min() - margin, ty.max() + margin
        
        arc_mask = (np.maximum(ax1, ax2) >= min_x) & (np.minimum(ax1, ax2) <= max_x) & \
                   (np.maximum(ay1, ay2) >= min_y) & (np.minimum(ay1, ay2) <= max_y)
        sub_arc_indices = np.where(arc_mask)[0]
        
        lines = [
            LineString([(ax1[i], ay1[i]), (ax2[i], ay2[i])])
            for i in sub_arc_indices
        ]
        tree = STRtree(lines)
        
        adx = ax2 - ax1
        ady = ay2 - ay1
        # Azimuth / heading in degrees: 0 = North, 90 = East
        arc_headings = (np.degrees(np.arctan2(adx, ady)) + 360.0) % 360.0
        
        return tree, sub_arc_indices, ax1, ay1, ax2, ay2, arc_headings

    def match_ground_truth(self, trip: TripData) -> np.ndarray:
        """
        Match each GPS observation to its reference route arc.
        """
        if trip.route is None or len(trip.route) == 0:
            return np.full(len(trip.track), -1, dtype=np.int64)
            
        u_nodes = trip.arcs[trip.route, 0]
        v_nodes = trip.arcs[trip.route, 1]
        rx1 = trip.proj_nodes[u_nodes, 0]
        ry1 = trip.proj_nodes[u_nodes, 1]
        rx2 = trip.proj_nodes[v_nodes, 0]
        ry2 = trip.proj_nodes[v_nodes, 1]
        
        route_lines = [
            LineString([(rx1[i], ry1[i]), (rx2[i], ry2[i])])
            for i in range(len(trip.route))
        ]
        route_tree = STRtree(route_lines)
        
        pts = [Point(trip.proj_track[t, 0], trip.proj_track[t, 1]) for t in range(len(trip.track))]
        nearest_route_sub_idx = route_tree.nearest(pts)
        true_arcs = trip.route[nearest_route_sub_idx]
        return true_arcs

    def extract_features(self, trip: TripData, include_target: bool = True) -> pd.DataFrame:
        """
        Extract candidate pairs and engineer geometric and kinematic features.
        """
        tree, sub_indices, ax1, ay1, ax2, ay2, arc_headings = self.build_spatial_index(trip)
        
        tx = trip.proj_track[:, 0]
        ty = trip.proj_track[:, 1]
        tt = trip.track[:, 2]
        T = len(trip.track)
        
        # Calculate trajectory kinematics
        gps_dx = np.zeros(T)
        gps_dy = np.zeros(T)
        gps_dt = np.ones(T)
        if T > 1:
            gps_dx[1:] = np.diff(tx)
            gps_dy[1:] = np.diff(ty)
            gps_dt[1:] = np.maximum(1e-3, np.diff(tt))
            gps_dx[0] = gps_dx[1]
            gps_dy[0] = gps_dy[1]
            
        gps_speeds = np.hypot(gps_dx, gps_dy) / gps_dt
        gps_headings = (np.degrees(np.arctan2(gps_dx, gps_dy)) + 360.0) % 360.0
        
        # Query candidates within search radius
        boxes = [
            box(tx[t] - self.search_radius, ty[t] - self.search_radius,
                tx[t] + self.search_radius, ty[t] + self.search_radius)
            for t in range(T)
        ]
        pt_indices, line_sub_indices = tree.query(boxes, predicate="intersects")
        
        # Check for any points without candidates and add fallback nearest arc
        pts_with_candidates = set(np.unique(pt_indices))
        missing_pts = [t for t in range(T) if t not in pts_with_candidates]
        if missing_pts:
            pts_to_query = [Point(tx[t], ty[t]) for t in missing_pts]
            nearest_sub = tree.nearest(pts_to_query)
            pt_indices = np.concatenate([pt_indices, np.array(missing_pts)])
            line_sub_indices = np.concatenate([line_sub_indices, nearest_sub])
            
        orig_arc_ids = sub_indices[line_sub_indices]
        
        # Vectorized feature calculations
        px = tx[pt_indices]
        py = ty[pt_indices]
        x1 = ax1[orig_arc_ids]
        y1 = ay1[orig_arc_ids]
        x2 = ax2[orig_arc_ids]
        y2 = ay2[orig_arc_ids]
        
        dist_to_seg, dist_to_start, dist_to_end, u, seg_len = compute_point_to_segment(
            px, py, x1, y1, x2, y2
        )
        
        pt_heading = gps_headings[pt_indices]
        cand_heading = arc_headings[orig_arc_ids]
        heading_diff = np.abs(pt_heading - cand_heading)
        heading_diff = np.minimum(heading_diff, 360.0 - heading_diff)
        
        speed = gps_speeds[pt_indices]
        dt = gps_dt[pt_indices]
        
        df = pd.DataFrame({
            "trip_id": trip.trip_id,
            "pt_idx": pt_indices,
            "arc_id": orig_arc_ids,
            "dist_to_seg": dist_to_seg,
            "heading_diff": heading_diff,
            "sin_hdiff": np.sin(np.radians(heading_diff)),
            "cos_hdiff": np.cos(np.radians(heading_diff)),
            "speed": speed,
            "u_proj": u,
            "arc_len": seg_len,
            "dist_to_start": dist_to_start,
            "dist_to_end": dist_to_end,
            "dt": dt
        })
        
        if include_target and trip.route is not None:
            true_arcs = self.match_ground_truth(trip)
            target = (orig_arc_ids == true_arcs[pt_indices]).astype(int)
            df["target"] = target
            
        return df.sort_values(by=["pt_idx", "dist_to_seg"]).reset_index(drop=True)
