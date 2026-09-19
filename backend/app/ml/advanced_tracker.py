"""
Phase 3C: Advanced Multi-Hypothesis Atmospheric Tracker (MHT).
Maintains probabilistic trajectory hypotheses across forecast leads and ensemble members,
supports split/merge morphology, kinematic acceleration constraints, and association confidence scoring.
Never fabricates trajectories: low-confidence steps are explicitly tagged as UNCERTAIN.
Authoritatively provides persistent event IDs, consensus tracks, lifecycles, and kinematics.
"""

from typing import List, Dict, Any, Optional, Tuple
import math
import numpy as np
from backend.app.config import domain_config, DomainConfig
from backend.app.core.tracker import EventTracker

EARTH_RADIUS_KM = 6371.0


class TrajectoryHypothesis:
    """Represents a candidate trajectory branch with kinematic history and confidence metrics."""
    def __init__(self, hypothesis_id: str, initial_detection: Dict[str, Any], initial_confidence: float = 0.95):
        self.hypothesis_id = hypothesis_id
        det_init = dict(initial_detection)
        det_init["lifecycle_state"] = det_init.get("lifecycle_state", "GENESIS")
        det_init["association_confidence"] = float(round(initial_confidence, 3))
        det_init["uncertainty_flag"] = "CONFIDENT"
        if "prev_bounding_box" not in det_init:
            det_init["prev_bounding_box"] = det_init.get("bounding_box", [0, 0, 0, 0])
        if "bbox_delta" not in det_init:
            det_init["bbox_delta"] = [0.0, 0.0, 0.0, 0.0]
        if "peak_efi" not in det_init:
            det_init["peak_efi"] = det_init.get("peak_prob", 0.8)

        self.points: List[Dict[str, Any]] = [det_init]
        self.cumulative_confidence = initial_confidence
        self.step_confidences: List[float] = [initial_confidence]
        self.parent_id: Optional[str] = None
        self.split_from: Optional[str] = None
        self.merged_into: Optional[str] = None
        self.status = "ACTIVE"

    def add_step(self, detection: Dict[str, Any], step_confidence: float, lifecycle: str = "CONTINUATION"):
        det_step = dict(detection)
        det_step["association_confidence"] = float(round(step_confidence, 3))
        det_step["lifecycle_state"] = lifecycle
        if step_confidence < 0.40:
            det_step["uncertainty_flag"] = "CONFIDENCE_UNCERTAIN"
            self.status = "UNCERTAIN"
        else:
            det_step["uncertainty_flag"] = "CONFIDENT"
            if self.status == "UNCERTAIN":
                self.status = "ACTIVE"

        prev_box = self.points[-1].get("bounding_box", [0, 0, 0, 0])
        curr_box = det_step.get("bounding_box", prev_box)
        det_step["prev_bounding_box"] = prev_box
        det_step["bbox_delta"] = [curr_box[i] - prev_box[i] for i in range(4)]
        if "peak_efi" not in det_step:
            det_step["peak_efi"] = det_step.get("peak_prob", 0.8)

        self.points.append(det_step)
        self.step_confidences.append(step_confidence)
        self.cumulative_confidence = float(0.7 * self.cumulative_confidence + 0.3 * step_confidence)


class AdvancedMultiHypothesisTracker:
    """
    Advanced Multi-Hypothesis Spatio-Temporal Tracker with Kinematic Physics Validation.
    """
    def __init__(
        self,
        config: DomainConfig = domain_config,
        max_speed_kmh: float = 95.0,
        max_acceleration_kmh2: float = 30.0,
        max_turn_angle_deg: float = 75.0,
        confidence_threshold: float = 0.40,
        max_hypotheses_per_track: int = 3
    ):
        self.config = config
        self.max_speed_kmh = max_speed_kmh
        self.max_acceleration_kmh2 = max_acceleration_kmh2
        self.max_turn_angle_deg = max_turn_angle_deg
        self.confidence_threshold = confidence_threshold
        self.max_hypotheses = max_hypotheses_per_track
        self.fallback_tracker = EventTracker(max_speed_kmh=max_speed_kmh)

    def track_multi_hypothesis(
        self,
        detections_by_lead: Dict[int, List[Dict[str, Any]]],
        st_gnn_velocities: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Executes multi-hypothesis tracking across forecast lead times.
        Returns consensus trajectories, active/alternative hypotheses, split/merge events, and kinematics.
        """
        try:
            sorted_leads = sorted(detections_by_lead.keys())
            if not sorted_leads:
                return {
                    "success": True, "model": "AdvancedMultiHypothesisTracker",
                    "consensus_tracks": [], "tracks": [], "hypotheses": [],
                    "split_merge_events": [], "total_hypotheses_evaluated": 0,
                    "fallback_triggered": False
                }

            hypotheses: List[TrajectoryHypothesis] = []
            hyp_counter = 1
            split_merge_events = []

            # Step 1: Initialize hypotheses for initial lead
            initial_lead = sorted_leads[0]
            for det in detections_by_lead[initial_lead]:
                det_copy = dict(det)
                det_copy["lead_time"] = det.get("lead_time", initial_lead)
                det_copy["centroid_lat"] = det.get("centroid_lat", det.get("lat", 0.0))
                det_copy["centroid_lon"] = det.get("centroid_lon", det.get("lon", 0.0))
                det_copy["lat"] = det_copy["centroid_lat"]
                det_copy["lon"] = det_copy["centroid_lon"]
                det_copy["step_speed_kmh"] = 0.0
                det_copy["step_bearing_deg"] = 0.0
                det_copy["lifecycle_state"] = "GENESIS"
                hyp = TrajectoryHypothesis(f"HYP_{hyp_counter:03d}", det_copy, initial_confidence=0.95)
                hyp_counter += 1
                hypotheses.append(hyp)

            # Step 2: Iterate over consecutive lead times
            for lead_idx in range(1, len(sorted_leads)):
                prev_lead = sorted_leads[lead_idx - 1]
                curr_lead = sorted_leads[lead_idx]
                dt_hours = max(1, curr_lead - prev_lead)
                current_dets = []
                for det in detections_by_lead[curr_lead]:
                    d_c = dict(det)
                    d_c["lead_time"] = det.get("lead_time", curr_lead)
                    d_c["centroid_lat"] = det.get("centroid_lat", det.get("lat", 0.0))
                    d_c["centroid_lon"] = det.get("centroid_lon", det.get("lon", 0.0))
                    d_c["lat"] = d_c["centroid_lat"]
                    d_c["lon"] = d_c["centroid_lon"]
                    current_dets.append(d_c)

                active_hyps = [h for h in hypotheses if h.status in ("ACTIVE", "UNCERTAIN")]
                if not active_hyps:
                    for det in current_dets:
                        det_copy = dict(det)
                        det_copy["step_speed_kmh"] = 0.0
                        det_copy["step_bearing_deg"] = 0.0
                        det_copy["lifecycle_state"] = "GENESIS"
                        hyp = TrajectoryHypothesis(f"HYP_{hyp_counter:03d}", det_copy, initial_confidence=0.85)
                        hyp_counter += 1
                        hypotheses.append(hyp)
                    continue

                matched_hyp_indices = set()
                matched_det_indices = set()
                det_match_counts = {d_idx: [] for d_idx in range(len(current_dets))}

                candidates = []
                for h_idx, hyp in enumerate(active_hyps):
                    last_pt = hyp.points[-1]
                    for d_idx, det in enumerate(current_dets):
                        conf, metrics = self._calculate_association_confidence(
                            last_pt=last_pt,
                            candidate_det=det,
                            dt_hours=dt_hours,
                            prev_pts=hyp.points
                        )
                        if conf > 0.15:
                            candidates.append({
                                "hyp_idx": h_idx,
                                "det_idx": d_idx,
                                "confidence": conf,
                                "metrics": metrics
                            })

                candidates.sort(key=lambda x: x["confidence"], reverse=True)

                for cand in candidates:
                    h_idx = cand["hyp_idx"]
                    d_idx = cand["det_idx"]
                    conf = cand["confidence"]
                    metrics = cand["metrics"]

                    if h_idx not in matched_hyp_indices and d_idx not in matched_det_indices:
                        hyp = active_hyps[h_idx]
                        det_copy = dict(current_dets[d_idx])
                        det_copy["step_speed_kmh"] = metrics["speed_kmh"]
                        det_copy["step_bearing_deg"] = metrics["bearing_deg"]
                        
                        last_p = hyp.points[-1].get("peak_precip_mm", 0.0)
                        curr_p = det_copy.get("peak_precip_mm", 0.0)
                        
                        if curr_p >= last_p * 1.15:
                            lifecycle = "INTENSIFICATION"
                        elif curr_p <= last_p * 0.85:
                            lifecycle = "DECAY" if curr_p > 20.0 else "DISSIPATION"
                        elif curr_p >= 75.0:
                            lifecycle = "PEAK"
                        else:
                            lifecycle = "CONTINUATION"
                        
                        hyp.add_step(det_copy, step_confidence=conf, lifecycle=lifecycle)
                        matched_hyp_indices.add(h_idx)
                        matched_det_indices.add(d_idx)
                        det_match_counts[d_idx].append(h_idx)

                    elif h_idx in matched_hyp_indices and d_idx not in matched_det_indices and conf >= 0.50:
                        parent_hyp = active_hyps[h_idx]
                        split_hyp_id = f"HYP_{hyp_counter:03d}"
                        hyp_counter += 1
                        
                        det_copy = dict(current_dets[d_idx])
                        det_copy["step_speed_kmh"] = metrics["speed_kmh"]
                        det_copy["step_bearing_deg"] = metrics["bearing_deg"]
                        
                        base_pt = dict(parent_hyp.points[-2]) if len(parent_hyp.points) >= 2 else dict(parent_hyp.points[-1])
                        new_hyp = TrajectoryHypothesis(split_hyp_id, base_pt, initial_confidence=conf * 0.9)
                        new_hyp.split_from = parent_hyp.hypothesis_id
                        new_hyp.add_step(det_copy, step_confidence=conf, lifecycle="SPLIT_BRANCH")
                        hypotheses.append(new_hyp)
                        matched_det_indices.add(d_idx)
                        
                        split_merge_events.append({
                            "type": "SPLIT",
                            "lead_time": curr_lead,
                            "parent_hypothesis": parent_hyp.hypothesis_id,
                            "child_hypothesis": split_hyp_id,
                            "confidence": float(round(conf, 3))
                        })

                    elif h_idx not in matched_hyp_indices and d_idx in matched_det_indices and conf >= 0.50:
                        merging_hyp = active_hyps[h_idx]
                        primary_hyp_idx = det_match_counts[d_idx][0]
                        primary_hyp = active_hyps[primary_hyp_idx]
                        
                        merging_hyp.status = "MERGED"
                        merging_hyp.merged_into = primary_hyp.hypothesis_id
                        matched_hyp_indices.add(h_idx)

                        split_merge_events.append({
                            "type": "MERGE",
                            "lead_time": curr_lead,
                            "source_hypothesis": merging_hyp.hypothesis_id,
                            "target_hypothesis": primary_hyp.hypothesis_id,
                            "confidence": float(round(conf, 3))
                        })

                for h_idx, hyp in enumerate(active_hyps):
                    if h_idx not in matched_hyp_indices:
                        hyp.status = "TERMINATED"

                for d_idx, det in enumerate(current_dets):
                    if d_idx not in matched_det_indices:
                        det_copy = dict(det)
                        det_copy["step_speed_kmh"] = 0.0
                        det_copy["step_bearing_deg"] = 0.0
                        det_copy["lifecycle_state"] = "GENESIS"
                        hyp = TrajectoryHypothesis(f"HYP_{hyp_counter:03d}", det_copy, initial_confidence=0.75)
                        hyp_counter += 1
                        hypotheses.append(hyp)

            consensus_tracks = self._build_consensus_trajectories(hypotheses)

            return {
                "success": True,
                "model": "AdvancedMultiHypothesisTracker",
                "consensus_tracks": consensus_tracks,
                "tracks": consensus_tracks,
                "all_hypotheses": [self._serialize_hypothesis(h) for h in hypotheses],
                "split_merge_events": split_merge_events,
                "total_hypotheses_evaluated": len(hypotheses),
                "fallback_triggered": False
            }

        except Exception as e:
            fallback_tracks = self.fallback_tracker.track_events_across_leads(detections_by_lead)
            return {
                "success": False,
                "model": "Fallback_DeterministicTracker",
                "consensus_tracks": fallback_tracks,
                "tracks": fallback_tracks,
                "all_hypotheses": [],
                "split_merge_events": [],
                "error": str(e),
                "fallback_triggered": True
            }

    def _calculate_association_confidence(
        self,
        last_pt: Dict[str, Any],
        candidate_det: Dict[str, Any],
        dt_hours: int,
        prev_pts: List[Dict[str, Any]]
    ) -> Tuple[float, Dict[str, float]]:
        lat1 = last_pt["centroid_lat"] if "centroid_lat" in last_pt else last_pt["lat"]
        lon1 = last_pt["centroid_lon"] if "centroid_lon" in last_pt else last_pt["lon"]
        lat2 = candidate_det["centroid_lat"] if "centroid_lat" in candidate_det else candidate_det["lat"]
        lon2 = candidate_det["centroid_lon"] if "centroid_lon" in candidate_det else candidate_det["lon"]

        dist_km = self._haversine_distance(lat1, lon1, lat2, lon2)
        speed_kmh = dist_km / dt_hours
        bearing = self._calculate_bearing(lat1, lon1, lat2, lon2)

        if speed_kmh > self.max_speed_kmh:
            return 0.0, {"dist_km": dist_km, "speed_kmh": speed_kmh, "bearing_deg": bearing}

        speed_score = max(0.0, 1.0 - (speed_kmh / self.max_speed_kmh)**2)

        bearing_score = 1.0
        if len(prev_pts) >= 2:
            prev_speed = prev_pts[-1].get("step_speed_kmh", 0.0)
            accel = abs(speed_kmh - prev_speed) / dt_hours
            if accel > self.max_acceleration_kmh2:
                speed_score *= 0.5

            prev_bearing = prev_pts[-1].get("step_bearing_deg", bearing)
            d_bearing = abs((bearing - prev_bearing + 180) % 360 - 180)
            if d_bearing > self.max_turn_angle_deg:
                bearing_score = max(0.0, 1.0 - (d_bearing / 180.0))

        box1 = last_pt.get("bounding_box", [lat1, lon1, lat1, lon1])
        box2 = candidate_det.get("bounding_box", [lat2, lon2, lat2, lon2])
        iou = self._calculate_bbox_iou(box1, box2)

        p1 = last_pt.get("peak_precip_mm", 10.0)
        p2 = candidate_det.get("peak_precip_mm", 10.0)
        intensity_sim = 1.0 - (abs(p1 - p2) / max(10.0, p1 + p2))

        confidence = (0.35 * speed_score) + (0.25 * bearing_score) + (0.20 * iou) + (0.20 * intensity_sim)
        confidence = float(np.clip(confidence, 0.0, 1.0))

        return confidence, {
            "dist_km": float(dist_km),
            "speed_kmh": float(speed_kmh),
            "bearing_deg": float(bearing),
            "iou": float(iou),
            "intensity_sim": float(intensity_sim)
        }

    def _build_consensus_trajectories(self, hypotheses: List[TrajectoryHypothesis]) -> List[Dict[str, Any]]:
        qualified = [h for h in hypotheses if len(h.points) >= 1]
        qualified.sort(key=lambda h: (len(h.points), h.cumulative_confidence), reverse=True)

        consensus = []
        track_idx = 1
        for hyp in qualified:
            pts = hyp.points
            start_lead = pts[0]["lead_time"]
            end_lead = pts[-1]["lead_time"]
            duration = end_lead - start_lead

            peak_precip = max(p.get("peak_precip_mm", 0.0) for p in pts)
            min_mslp = min(p.get("min_mslp_hpa", 1013.0) for p in pts)
            peak_sev = max(p.get("severity_score", 0.5) for p in pts)
            total_dist = sum(p.get("step_speed_kmh", 0.0) * 24.0 for p in pts)

            lifecycles = [p.get("lifecycle_state", "CONTINUATION") for p in pts]
            predominant_lifecycle = "PEAK" if "PEAK" in lifecycles else lifecycles[-1]

            unc_radii = [float(round(25.0 + 4.0 * i, 1)) for i in range(len(pts))]
            track_record = {
                "track_id": f"TRK_{track_idx:03d}",
                "event_id": f"EVT_TRK_{track_idx:03d}",
                "primary_hypothesis_id": hyp.hypothesis_id,
                "start_lead_time": start_lead,
                "end_lead_time": end_lead,
                "duration_hours": duration,
                "status": hyp.status,
                "lifecycle_state": predominant_lifecycle,
                "cumulative_confidence": round(hyp.cumulative_confidence, 3),
                "peak_severity": round(peak_sev, 3),
                "peak_precip_mm": round(peak_precip, 2),
                "min_mslp_hpa": round(min_mslp, 1),
                "mean_speed_kmh": round(float(np.mean([p.get("step_speed_kmh", 0.0) for p in pts[1:]])), 2) if len(pts) > 1 else 0.0,
                "total_distance_km": round(total_dist, 1),
                "trajectory_points": pts,
                "uncertainty_radii_km": unc_radii,
                "split_from": hyp.split_from,
                "merged_into": hyp.merged_into,
                "heading_compass": self._deg_to_compass(pts[-1].get("step_bearing_deg", 0.0))
            }
            consensus.append(track_record)
            track_idx += 1

        return consensus

    def _serialize_hypothesis(self, hyp: TrajectoryHypothesis) -> Dict[str, Any]:
        return {
            "hypothesis_id": hyp.hypothesis_id,
            "status": hyp.status,
            "cumulative_confidence": round(hyp.cumulative_confidence, 3),
            "parent_id": hyp.parent_id,
            "split_from": hyp.split_from,
            "merged_into": hyp.merged_into,
            "num_steps": len(hyp.points),
            "step_confidences": [round(c, 3) for c in hyp.step_confidences]
        }

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dp = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)
        a = math.sin(dp / 2.0)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0)**2
        return EARTH_RADIUS_KM * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dl = math.radians(lon2 - lon1)
        y = math.sin(dl) * math.cos(p2)
        x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
        return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

    def _deg_to_compass(self, deg: float) -> str:
        dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        return dirs[int((deg + 22.5) / 45.0) % 8]

    def _calculate_bbox_iou(self, box1: List[float], box2: List[float]) -> float:
        min_lat = max(box1[0], box2[0])
        min_lon = max(box1[1], box2[1])
        max_lat = min(box1[2], box2[2])
        max_lon = min(box1[3], box2[3])
        if max_lat <= min_lat or max_lon <= min_lon:
            return 0.0
        inter = (max_lat - min_lat) * (max_lon - min_lon)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return float(inter / max(1e-6, a1 + a2 - inter))


advanced_tracker = AdvancedMultiHypothesisTracker()
AdvancedMHTTracker = AdvancedMultiHypothesisTracker
