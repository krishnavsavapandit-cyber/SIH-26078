/**
 * Typed API Client for SIH-26078 Weather Intelligence System.
 */

export interface TrajectoryPoint {
  lead_time: number;
  lat: number;
  lon: number;
  bounding_box: [number, number, number, number];
  area_km2: number;
  peak_precip_mm: number;
  mean_precip_mm: number;
  peak_efi: number;
  min_mslp_hpa: number;
  max_wind_ms: number;
  severity_score: number;
  step_speed_kmh: number;
  step_bearing_deg: number;
  lifecycle_state: string;
  polygon: [number, number][];
}

export interface ConePoint {
  lead_time: number;
  center_lat: number;
  center_lon: number;
  cone_radius_km: number;
  cone_polygon: [number, number][];
  precip_p10: number;
  precip_p50: number;
  precip_p90: number;
  mslp_p10: number;
  mslp_p50: number;
  mslp_p90: number;
  member_centroids: [number, number][];
}

export interface SpaghettiTrack {
  member_id: number;
  points: {
    lead_time: number;
    lat: number;
    lon: number;
    peak_precip_mm: number;
    min_mslp_hpa: number;
  }[];
}

export interface UncertaintyCone {
  track_id: string;
  cone_points: ConePoint[];
  spaghetti_tracks: SpaghettiTrack[];
}

export interface WeatherEvent {
  event_id: string;
  run_id: string;
  track_id: string;
  event_name: string;
  start_lead_time: number;
  end_lead_time: number;
  duration_hours: number;
  peak_severity: number;
  peak_precip_mm: number;
  min_mslp_hpa: number;
  mean_speed_kmh: number;
  total_distance_km: number;
  heading_compass: string;
  trajectory_points: TrajectoryPoint[];
  uncertainty_cone?: UncertaintyCone;
}

export interface VerificationReport {
  contingency: {
    true_positives: number;
    false_positives: number;
    false_negatives: number;
    true_negatives: number;
    precision: number;
    recall: number;
    f1_score: number;
    csi_threat_score: number;
    false_alarm_ratio: number;
    miss_rate: number;
    spatial_iou: number;
    dice_coefficient: number;
  };
  trajectory_error: {
    mean_displacement_km: number;
    max_displacement_km: number;
    mean_peak_precip_error_mm: number;
    track_points_evaluated: number;
  };
  field_metrics: Record<string, {
    rmse: number;
    mae: number;
    pearson_r: number;
  }>;
  lead_time_breakdowns: {
    lead_time: number;
    gt_lat: number;
    gt_lon: number;
    pred_lat?: number;
    pred_lon?: number;
    displacement_error_km?: number;
    gt_peak_precip_mm: number;
    pred_peak_precip_mm: number;
    precip_error_mm: number;
    step_iou: number;
  }[];
}

export interface AlertRecord {
  alert_id: string;
  run_id: string;
  event_id: string;
  alert_level: 'NORMAL' | 'WATCH' | 'HIGH' | 'SEVERE' | 'EXTREME';
  lead_time_onset: number;
  peak_lead_time: number;
  affected_regions: string[];
  primary_threat: string;
  trigger_evidence: {
    peak_efi: number;
    peak_precip_mm: number;
    min_mslp_hpa: number;
    heading: string;
    speed_kmh: number;
  };
  physics_audit_passed: boolean;
}

export interface ProvenanceRecord {
  run_id: string;
  dataset_sha256: string;
  random_seed: number;
  model_version: string;
  execution_duration_sec: number;
  pipeline_steps_executed: string[];
  parameters: Record<string, any>;
  created_at: string;
}

export interface ForecastFieldData {
  run_id: string;
  lead_time: number;
  variable: string;
  latitudes: number[];
  longitudes: number[];
  min_val: number;
  max_val: number;
  mean_grid: number[][];
  std_grid: number[][];
}

export interface EFIData {
  run_id: string;
  lead_time: number;
  variable: string;
  latitudes: number[];
  longitudes: number[];
  efi_grid: number[][];
  sot_grid: number[][];
  z_score_grid: number[][];
  peak_efi: number;
  max_sot: number;
}

const API_BASE = '/api';

export const api = {
  getHealth: async () => {
    const res = await fetch(`${API_BASE}/health`);
    return res.json();
  },
  getRuns: async () => {
    const res = await fetch(`${API_BASE}/runs`);
    return res.json();
  },
  getRunDetails: async (runId: string) => {
    const res = await fetch(`${API_BASE}/runs/${runId}`);
    return res.json();
  },
  getEvents: async (runId?: string): Promise<WeatherEvent[]> => {
    const url = runId ? `${API_BASE}/events?run_id=${runId}` : `${API_BASE}/events`;
    const res = await fetch(url);
    return res.json();
  },
  getEvent: async (eventId: string): Promise<WeatherEvent> => {
    const res = await fetch(`${API_BASE}/events/${eventId}`);
    return res.json();
  },
  getField: async (runId: string, leadTime: number, variable: string): Promise<ForecastFieldData> => {
    const res = await fetch(`${API_BASE}/fields/${runId}/${leadTime}/${variable}`);
    return res.json();
  },
  getEFI: async (runId: string, leadTime: number, variable: string = 'precipitation'): Promise<EFIData> => {
    const res = await fetch(`${API_BASE}/efi/${runId}/${leadTime}?variable=${variable}`);
    return res.json();
  },
  getVerification: async (runId: string): Promise<VerificationReport> => {
    const res = await fetch(`${API_BASE}/verification/${runId}`);
    return res.json();
  },
  getAlerts: async (runId: string): Promise<AlertRecord[]> => {
    const res = await fetch(`${API_BASE}/alerts/${runId}`);
    return res.json();
  },
  getProvenance: async (runId: string): Promise<ProvenanceRecord> => {
    const res = await fetch(`${API_BASE}/provenance/${runId}`);
    return res.json();
  },
  runPipeline: async (params: {
    run_id: string;
    scenario_type: string;
    seed: number;
    displacement_bias_km?: number;
    intensity_bias_pct?: number;
  }) => {
    const res = await fetch(`${API_BASE}/pipeline/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return res.json();
  }
};
