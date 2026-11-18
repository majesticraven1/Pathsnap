from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from pathsnap.data_processing.loader import TripData
from pathsnap.models.train_models import FEATURE_COLS


class HMMRouteReconstructor:
    def __init__(
        self,
        sigma_gps: float = 10.0,
        beta: float = 15.0,
        stay_cost: float = 0.0,
        succ_cost: float = -1.0,
        two_hop_cost: float = -3.5,
        disconnected_penalty: float = -20.0
    ):
        self.sigma_gps = sigma_gps
        self.beta = beta
        self.stay_cost = stay_cost
        self.succ_cost = succ_cost
        self.two_hop_cost = two_hop_cost
        self.disconnected_penalty = disconnected_penalty

    def build_topology(self, arcs: np.ndarray) -> Tuple[Dict[int, List[int]], Dict[int, set], Dict[int, set]]:
        """
        Build outgoing node-to-arc index, 1-hop direct successors, and 2-hop successors.
        """
        num_arcs = len(arcs)
        from_nodes = arcs[:, 0]
        to_nodes = arcs[:, 1]
        
        node_out_arcs = defaultdict(list)
        for a_id in range(num_arcs):
            node_out_arcs[from_nodes[a_id]].append(a_id)
            
        arc_succs = defaultdict(set)
        for a_id in range(num_arcs):
            tn = to_nodes[a_id]
            for nxt in node_out_arcs[tn]:
                arc_succs[a_id].add(nxt)
                
        arc_two_hop = defaultdict(set)
        for a_id in list(arc_succs.keys()):
            succ_set = arc_succs[a_id]
            for s in succ_set:
                arc_two_hop[a_id].update(arc_succs.get(s, set()))
                
        return node_out_arcs, arc_succs, arc_two_hop

    def decode(
        self,
        trip: TripData,
        df_cand: pd.DataFrame,
        model: Any,
        use_scaled_features: bool = False,
        scaler: Any = None
    ) -> Dict[str, Any]:
        """
        Perform Viterbi decoding over candidate states for the entire GPS trajectory.
        """
        # Ensure candidate predictions
        if "prob" not in df_cand.columns:
            X = df_cand[FEATURE_COLS].values
            if use_scaled_features and scaler is not None:
                X = scaler.transform(X)
            probs = model.predict_proba(X)[:, 1]
            df_cand = df_cand.copy()
            df_cand["prob"] = probs
            
        T = len(trip.track)
        arcs = trip.arcs
        
        # Build network graph connectivity
        _, arc_succs, arc_two_hop = self.build_topology(arcs)
        
        # Group candidates by observation index
        grouped = dict(tuple(df_cand.groupby("pt_idx")))
        cand_groups = [grouped[t] for t in range(T) if t in grouped]
        
        # If any observation had zero candidates (rare fallback)
        if len(cand_groups) < T:
            # Re-index to ensure length matches
            cand_groups = [
                grouped.get(t, pd.DataFrame([{
                    "pt_idx": t, "arc_id": 0, "prob": 0.5, "dist_to_seg": 20.0, "target": 0
                }])) for t in range(T)
            ]
            
        # Viterbi Initialization at t=0
        first_cands = cand_groups[0]
        V_prev = np.log(first_cands["prob"].values + 1e-6) - 0.5 * (first_cands["dist_to_seg"].values / self.sigma_gps) ** 2
        prev_arcs = first_cands["arc_id"].values
        
        backpointers = []
        
        # Forward Viterbi recurrence
        for t in range(1, T):
            curr_cands = cand_groups[t]
            curr_arcs = curr_cands["arc_id"].values
            emiss = np.log(curr_cands["prob"].values + 1e-6) - 0.5 * (curr_cands["dist_to_seg"].values / self.sigma_gps) ** 2
            
            K_prev = len(prev_arcs)
            K_curr = len(curr_arcs)
            
            # Check time gap between consecutive observations
            dt = trip.track[t, 2] - trip.track[t-1, 2]
            gap_penalty_relax = min(15.0, max(0.0, (dt - 2.0) * 0.5))
            eff_penalty = self.disconnected_penalty + gap_penalty_relax
            
            trans_matrix = np.full((K_prev, K_curr), eff_penalty)
            
            for i in range(K_prev):
                a1 = prev_arcs[i]
                succs = arc_succs[a1]
                two_hops = arc_two_hop[a1]
                for j in range(K_curr):
                    a2 = curr_arcs[j]
                    if a1 == a2:
                        trans_matrix[i, j] = self.stay_cost
                    elif a2 in succs:
                        trans_matrix[i, j] = self.succ_cost
                    elif a2 in two_hops:
                        trans_matrix[i, j] = self.two_hop_cost
                        
            scores = V_prev[:, None] + trans_matrix
            best_prev_idx = np.argmax(scores, axis=0)
            V_curr = scores[best_prev_idx, np.arange(K_curr)] + emiss
            
            backpointers.append(best_prev_idx)
            V_prev = V_curr
            prev_arcs = curr_arcs
            
        # Backtracking
        best_last = np.argmax(V_prev)
        decoded_indices = [best_last]
        for bp in reversed(backpointers):
            best_last = bp[best_last]
            decoded_indices.append(best_last)
        decoded_indices.reverse()
        
        decoded_arcs = [int(cand_groups[t]["arc_id"].iloc[decoded_indices[t]]) for t in range(T)]
        
        # Classifier-only top-1 predictions for baseline comparison
        classifier_top1_arcs = []
        for t in range(T):
            cands = cand_groups[t]
            max_p_idx = cands["prob"].idxmax()
            classifier_top1_arcs.append(int(cands.loc[max_p_idx, "arc_id"]))
            
        # Compute evaluation metrics if ground-truth route is present
        top1_acc_clf = None
        top1_acc_hmm = None
        if "target" in df_cand.columns:
            clf_targets = [int(cand_groups[t].loc[cand_groups[t]["prob"].idxmax(), "target"]) for t in range(T)]
            hmm_targets = [int(cand_groups[t]["target"].iloc[decoded_indices[t]]) for t in range(T)]
            top1_acc_clf = float(np.mean(clf_targets) * 100.0)
            top1_acc_hmm = float(np.mean(hmm_targets) * 100.0)
            
        # Route Continuity calculation
        def get_continuity(arc_seq):
            if len(arc_seq) < 2:
                return 100.0
            consec = 0
            for i in range(len(arc_seq) - 1):
                a1, a2 = arc_seq[i], arc_seq[i+1]
                if a1 == a2 or arcs[a1, 1] == arcs[a2, 0]:
                    consec += 1
            return float(consec / (len(arc_seq) - 1) * 100.0)
            
        cont_clf = get_continuity(classifier_top1_arcs)
        cont_hmm = get_continuity(decoded_arcs)
        
        # Generate reconstructed route polyline coordinates (lat, lon) for mapping
        route_coords = []
        seen_arcs = []
        for a in decoded_arcs:
            if not seen_arcs or seen_arcs[-1] != a:
                seen_arcs.append(a)
                u = arcs[a, 0]
                v = arcs[a, 1]
                p1 = [float(trip.nodes[u, 1]), float(trip.nodes[u, 0])]
                p2 = [float(trip.nodes[v, 1]), float(trip.nodes[v, 0])]
                route_coords.append(p1)
                route_coords.append(p2)
                
        return {
            "trip_id": trip.trip_id,
            "total_observations": T,
            "decoded_arcs": decoded_arcs,
            "unique_arcs_count": len(set(decoded_arcs)),
            "classifier_top1_arcs": classifier_top1_arcs,
            "top1_accuracy_classifier": top1_acc_clf,
            "top1_accuracy_hmm": top1_acc_hmm,
            "route_continuity_classifier": cont_clf,
            "route_continuity_hmm": cont_hmm,
            "accuracy_improvement": (top1_acc_hmm - top1_acc_clf) if (top1_acc_hmm is not None and top1_acc_clf is not None) else None,
            "continuity_improvement": (cont_hmm - cont_clf),
            "reconstructed_polyline": route_coords
        }
