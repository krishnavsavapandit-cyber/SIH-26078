"""
Spatio-Temporal Event Tracking Engine for SIH-26078.
Associates discrete spatial detections across forecast lead times into continuous trajectories,
computing kinematic velocities, bearing headings, and lifecycle transitions.
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional

class EventTracker:
    """
    Multi-lead-time object tracking engine using kinematic cost-matrix association.
    """
    def __init__(
        self,
        max_speed_kmh: float = 90.0,
        max_association_distance_km: float = 450.0,
        iou_weight: float = 0.35,
        dist_weight: float = 0.45,
        intensity_weight: float = 0.20
    ):
        self.max_speed_kmh = max_speed_kmh
        self.max_association_dist_km = max_association_distance_km
        self.iou_weight = iou_weight
        self.dist_weight = dist_weight
        self.intensity_weight = intensity_weight

    def track_events_across_leads(
        self,
        detections_by_lead: Dict[int, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Builds trajectories from lead-time detection dictionaries.
        """
        sorted_leads = sorted(detections_by_lead.keys())
        if not sorted_leads:
            return []

        active_tracks: List[Dict[str, Any]] = []
        finished_tracks: List[Dict[str, Any]] = []
        track_counter = 1

        for lead_time in sorted_leads:
            current_detections = detections_by_lead[lead_time]
            
            if not active_tracks:
                # Initialize tracks for first lead time
                for det in current_detections:
                    track_id = f"TRK_{track_counter:03d}"
                    track_counter += 1
                    track = self._create_new_track(track_id, det)
                    active_tracks.append(track)
                continue

            # Associate active tracks with current detections
            matched_pairs, unmatched_tracks, unmatched_detections = self._associate_step(
                active_tracks, current_detections, lead_time
            )

            # Update matched tracks
            for track_idx, det_idx in matched_pairs:
                track = active_tracks[track_idx]
                det = current_detections[det_idx]
                self._update_track(track, det, lead_time)

            # Mark unmatched tracks as terminated/inactive
            remaining_active = []
            for track_idx in range(len(active_tracks)):
                if track_idx in unmatched_tracks:
                    track = active_tracks[track_idx]
                    track["status"] = "TERMINATED"
                    finished_tracks.append(track)
                else:
                    remaining_active.append(active_tracks[track_idx])
            active_tracks = remaining_active

            # Spawn new tracks for unmatched detections
            for det_idx in unmatched_detections:
                track_id = f"TRK_{track_counter:03d}"
                track_counter += 1
                det = current_detections[det_idx]
                track = self._create_new_track(track_id, det)
                active_tracks.append(track)

        # Merge all tracks
        all_tracks = finished_tracks + active_tracks
        
        # Post-process summary metrics for each track
        for track in all_tracks:
            self._finalize_track_metrics(track)

        # Sort tracks by duration and peak severity descending
        all_tracks.sort(key=lambda t: (t["duration_hours"], t["peak_severity"]), reverse=True)
        return all_tracks

    def _associate_step(
        self,
        active_tracks: List[Dict[str, Any]],
        detections: List[Dict[str, Any]],
        current_lead: int
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Bipartite matching between existing tracks and incoming detections.
        """
        if not active_tracks or not detections:
            return [], list(range(len(active_tracks))), list(range(len(detections)))

        n_tracks = len(active_tracks)
        n_dets = len(detections)
        cost_matrix = np.full((n_tracks, n_dets), 1e6, dtype=np.float32)

        for i, track in enumerate(active_tracks):
            last_pt = track["trajectory_points"][-1]
            last_lead = last_pt["lead_time"]
            dt_hours = max(1, current_lead - last_lead)
            max_allowed_dist = min(self.max_association_dist_km, self.max_speed_kmh * dt_hours)

            for j, det in enumerate(detections):
                dist_km = self._haversine_distance(
                    last_pt["lat"], last_pt["lon"],
                    det["centroid_lat"], det["centroid_lon"]
                )
                
                if dist_km > max_allowed_dist:
                    continue # Exceeds physically plausible translation distance

                # Bounding box IoU
                iou = self._calculate_bbox_iou(last_pt["bounding_box"], det["bounding_box"])
                
                # Intensity difference
                intens_diff = abs(last_pt["peak_precip_mm"] - det["peak_precip_mm"]) / max(1.0, last_pt["peak_precip_mm"] + det["peak_precip_mm"])

                # Normalized composite cost
                norm_dist = dist_km / max_allowed_dist
                cost = (self.dist_weight * norm_dist + 
                        self.iou_weight * (1.0 - iou) + 
                        self.intensity_weight * intens_diff)
                
                cost_matrix[i, j] = cost

        # Greedy matching
        matched_pairs = []
        matched_tracks = set()
        matched_dets = set()

        while True:
            min_val = np.min(cost_matrix)
            if min_val >= 1.0: # Match threshold
                break
            min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            t_idx, d_idx = int(min_idx[0]), int(min_idx[1])
            
            matched_pairs.append((t_idx, d_idx))
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)
            
            cost_matrix[t_idx, :] = 1e6
            cost_matrix[:, d_idx] = 1e6

        unmatched_tracks = [i for i in range(n_tracks) if i not in matched_tracks]
        unmatched_dets = [j for j in range(n_dets) if j not in matched_dets]

        return matched_pairs, unmatched_tracks, unmatched_dets

    def _create_new_track(self, track_id: str, det: Dict[str, Any]) -> Dict[str, Any]:
        """Initializes a new trajectory record from a detection."""
        bbox = det["bounding_box"]
        pt = {
            "lead_time": det["lead_time"],
            "lat": det["centroid_lat"],
            "lon": det["centroid_lon"],
            "bounding_box": bbox,
            "prev_bounding_box": bbox,
            "bbox_delta": [0.0, 0.0, 0.0, 0.0],
            "area_km2": det["area_km2"],
            "prev_area_km2": det["area_km2"],
            "area_expansion_rate_km2h": 0.0,
            "footprint_evolution": "STABLE",
            "pixel_count": det.get("pixel_count", 0),
            "peak_precip_mm": det["peak_precip_mm"],
            "mean_precip_mm": det["mean_precip_mm"],
            "peak_efi": det["peak_efi"],
            "min_mslp_hpa": det["min_mslp_hpa"],
            "max_wind_ms": det["max_wind_ms"],
            "severity_score": det["severity_score"],
            "step_speed_kmh": 0.0,
            "step_bearing_deg": 0.0,
            "lifecycle_state": "INITIATION",
            "polygon": det["polygon"]
        }
        return {
            "track_id": track_id,
            "start_lead_time": det["lead_time"],
            "end_lead_time": det["lead_time"],
            "status": "ACTIVE",
            "trajectory_points": [pt],
            "peak_severity": det["severity_score"],
            "peak_precip_mm": det["peak_precip_mm"],
            "min_mslp_hpa": det["min_mslp_hpa"],
            "total_distance_km": 0.0,
            "duration_hours": 0
        }

    def _update_track(self, track: Dict[str, Any], det: Dict[str, Any], lead_time: int):
        """Appends step detection to active track and updates kinematics."""
        last_pt = track["trajectory_points"][-1]
        dt_hours = max(1, lead_time - last_pt["lead_time"])
        
        step_dist = self._haversine_distance(
            last_pt["lat"], last_pt["lon"],
            det["centroid_lat"], det["centroid_lon"]
        )
        step_speed = float(step_dist / dt_hours)
        step_bearing = float(self._calculate_bearing(
            last_pt["lat"], last_pt["lon"],
            det["centroid_lat"], det["centroid_lon"]
        ))

        # Dynamic Footprint Evolution
        curr_bbox = det["bounding_box"]
        prev_bbox = last_pt["bounding_box"]
        bbox_delta = [
            round(curr_bbox[0] - prev_bbox[0], 4),
            round(curr_bbox[1] - prev_bbox[1], 4),
            round(curr_bbox[2] - prev_bbox[2], 4),
            round(curr_bbox[3] - prev_bbox[3], 4),
        ]
        
        curr_area = det["area_km2"]
        prev_area = last_pt["area_km2"]
        area_expansion_rate = float((curr_area - prev_area) / dt_hours)

        if area_expansion_rate > 50.0:
            footprint_evolution = "EXPANDING"
        elif area_expansion_rate < -50.0:
            footprint_evolution = "CONTRACTING"
        else:
            footprint_evolution = "STEADY"

        # Lifecycle determination
        if det["peak_precip_mm"] >= last_pt["peak_precip_mm"] * 1.15:
            state = "INTENSIFICATION"
        elif det["peak_precip_mm"] <= last_pt["peak_precip_mm"] * 0.85:
            state = "DISSIPATION"
        else:
            state = "CONTINUATION"

        pt = {
            "lead_time": lead_time,
            "lat": det["centroid_lat"],
            "lon": det["centroid_lon"],
            "bounding_box": curr_bbox,
            "prev_bounding_box": prev_bbox,
            "bbox_delta": bbox_delta,
            "area_km2": curr_area,
            "prev_area_km2": prev_area,
            "area_expansion_rate_km2h": area_expansion_rate,
            "footprint_evolution": footprint_evolution,
            "pixel_count": det.get("pixel_count", 0),
            "peak_precip_mm": det["peak_precip_mm"],
            "mean_precip_mm": det["mean_precip_mm"],
            "peak_efi": det["peak_efi"],
            "min_mslp_hpa": det["min_mslp_hpa"],
            "max_wind_ms": det["max_wind_ms"],
            "severity_score": det["severity_score"],
            "step_speed_kmh": step_speed,
            "step_bearing_deg": step_bearing,
            "lifecycle_state": state,
            "polygon": det["polygon"]
        }

        track["trajectory_points"].append(pt)
        track["end_lead_time"] = lead_time
        track["total_distance_km"] += step_dist
        track["peak_severity"] = max(track["peak_severity"], det["severity_score"])
        track["peak_precip_mm"] = max(track["peak_precip_mm"], det["peak_precip_mm"])
        track["min_mslp_hpa"] = min(track["min_mslp_hpa"], det["min_mslp_hpa"])

    def _finalize_track_metrics(self, track: Dict[str, Any]):
        """Computes aggregated trajectory statistics."""
        pts = track["trajectory_points"]
        track["duration_hours"] = track["end_lead_time"] - track["start_lead_time"]
        
        if len(pts) > 1:
            speeds = [p["step_speed_kmh"] for p in pts[1:]]
            track["mean_speed_kmh"] = float(np.mean(speeds))
            track["max_speed_kmh"] = float(np.max(speeds))
            
            # Overall bearing from start to finish
            track["overall_bearing_deg"] = float(self._calculate_bearing(
                pts[0]["lat"], pts[0]["lon"],
                pts[-1]["lat"], pts[-1]["lon"]
            ))
            track["heading_compass"] = self._deg_to_compass(track["overall_bearing_deg"])
        else:
            track["mean_speed_kmh"] = 0.0
            track["max_speed_kmh"] = 0.0
            track["overall_bearing_deg"] = 0.0
            track["heading_compass"] = "STATIONARY"

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance in kilometers."""
        r = 6371.0 # Earth radius km
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        
        a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return r * c

    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculates forward azimuth bearing in degrees [0, 360)."""
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dlon = math.radians(lon2 - lon1)
        
        y = math.sin(dlon) * math.cos(phi2)
        x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
        bearing = math.degrees(math.atan2(y, x))
        return (bearing + 360.0) % 360.0

    def _deg_to_compass(self, deg: float) -> str:
        """Maps azimuth degrees to 8-point compass directions."""
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        idx = int((deg + 22.5) / 45.0) % 8
        return directions[idx]

    def _calculate_bbox_iou(self, box1: List[float], box2: List[float]) -> float:
        """Computes IoU between two bounding boxes [min_lat, min_lon, max_lat, max_lon]."""
        min_lat = max(box1[0], box2[0])
        min_lon = max(box1[1], box2[1])
        max_lat = min(box1[2], box2[2])
        max_lon = min(box1[3], box2[3])

        if max_lat <= min_lat or max_lon <= min_lon:
            return 0.0

        inter_area = (max_lat - min_lat) * (max_lon - min_lon)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = area1 + area2 - inter_area
        return float(inter_area / max(1e-6, union_area))
