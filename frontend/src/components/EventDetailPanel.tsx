import React from 'react';
import { WeatherEvent, TrajectoryPoint } from '../services/api';
import { Activity, Compass, Wind, Droplets, MapPin, TrendingUp, ShieldAlert } from 'lucide-react';

interface EventDetailPanelProps {
  event: WeatherEvent | null;
  currentLeadTime: number;
}

export const EventDetailPanel: React.FC<EventDetailPanelProps> = ({ event, currentLeadTime }) => {
  if (!event) {
    return (
      <div className="glass-panel p-6 rounded-xl border border-slate-700 text-center text-slate-400">
        <Activity className="w-8 h-8 text-slate-600 mx-auto mb-2 animate-pulse" />
        <p className="text-sm">No extreme anomaly selected</p>
      </div>
    );
  }

  const currentPoint: TrajectoryPoint | undefined = 
    event.trajectory_points.find((p) => p.lead_time === currentLeadTime) || event.trajectory_points[0];

  const getSeverityBadge = (category: string) => {
    switch (category) {
      case 'EXTREME':
        return 'bg-rose-500/20 text-rose-300 border-rose-500/50';
      case 'SEVERE':
        return 'bg-amber-500/20 text-amber-300 border-amber-500/50';
      default:
        return 'bg-cyan-500/20 text-cyan-300 border-cyan-500/50';
    }
  };

  return (
    <div className="glass-panel p-5 rounded-xl border border-slate-700/80 shadow-xl space-y-4">
      {/* Header */}
      <div className="flex items-start justify-between border-b border-slate-700/60 pb-3">
        <div>
          <div className="flex items-center space-x-2">
            <span className="font-mono font-bold text-base text-cyan-300">{event.track_id}</span>
            <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${getSeverityBadge(currentPoint?.severity_score > 0.7 ? 'EXTREME' : 'SEVERE')}`}>
              {currentPoint?.lifecycle_state || 'ACTIVE'}
            </span>
          </div>
          <h3 className="text-sm font-semibold text-slate-200 mt-0.5">{event.event_name}</h3>
        </div>
        <div className="text-right">
          <div className="text-[10px] text-slate-400 uppercase tracking-wider">Severity Index</div>
          <div className="font-mono text-lg font-bold text-rose-400">
            {(currentPoint?.severity_score || event.peak_severity).toFixed(2)} / 1.0
          </div>
        </div>
      </div>

      {/* Primary Metrics Grid */}
      <div className="grid grid-cols-2 gap-3 font-mono text-xs">
        <div className="bg-slate-900/60 p-2.5 rounded-lg border border-slate-800">
          <div className="text-slate-400 text-[10px] flex items-center space-x-1 mb-1">
            <Droplets className="w-3.5 h-3.5 text-cyan-400" />
            <span>Peak Precip (6h)</span>
          </div>
          <div className="text-sm font-bold text-cyan-300">
            {currentPoint ? currentPoint.peak_precip_mm.toFixed(1) : event.peak_precip_mm.toFixed(1)} mm
          </div>
        </div>

        <div className="bg-slate-900/60 p-2.5 rounded-lg border border-slate-800">
          <div className="text-slate-400 text-[10px] flex items-center space-x-1 mb-1">
            <TrendingUp className="w-3.5 h-3.5 text-amber-400" />
            <span>Central MSLP</span>
          </div>
          <div className="text-sm font-bold text-amber-300">
            {currentPoint ? currentPoint.min_mslp_hpa.toFixed(1) : event.min_mslp_hpa.toFixed(1)} hPa
          </div>
        </div>

        <div className="bg-slate-900/60 p-2.5 rounded-lg border border-slate-800">
          <div className="text-slate-400 text-[10px] flex items-center space-x-1 mb-1">
            <Wind className="w-3.5 h-3.5 text-blue-400" />
            <span>Translation Speed</span>
          </div>
          <div className="text-sm font-bold text-slate-200">
            {currentPoint ? currentPoint.step_speed_kmh.toFixed(1) : event.mean_speed_kmh.toFixed(1)} km/h
          </div>
        </div>

        <div className="bg-slate-900/60 p-2.5 rounded-lg border border-slate-800">
          <div className="text-slate-400 text-[10px] flex items-center space-x-1 mb-1">
            <Compass className="w-3.5 h-3.5 text-emerald-400" />
            <span>Heading Direction</span>
          </div>
          <div className="text-sm font-bold text-emerald-300">
            {event.heading_compass} ({currentPoint ? currentPoint.step_bearing_deg.toFixed(0) : '290'}°)
          </div>
        </div>
      </div>

      {/* Position Coordinates & Area */}
      <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800 text-xs space-y-1 font-mono">
        <div className="flex justify-between text-slate-400">
          <span className="flex items-center space-x-1">
            <MapPin className="w-3.5 h-3.5 text-rose-400" />
            <span>Centroid Position:</span>
          </span>
          <span className="text-slate-200 font-bold">
            {currentPoint ? `${currentPoint.lat.toFixed(2)}°N, ${currentPoint.lon.toFixed(2)}°E` : 'N/A'}
          </span>
        </div>
        <div className="flex justify-between text-slate-400">
          <span>Spatial Impact Area:</span>
          <span className="text-cyan-300 font-semibold">
            {currentPoint ? `${Math.round(currentPoint.area_km2).toLocaleString()} km²` : 'N/A'}
          </span>
        </div>
        <div className="flex justify-between text-slate-400">
          <span>Track Lifespan:</span>
          <span className="text-slate-300">
            T+{event.start_lead_time}h to T+{event.end_lead_time}h ({event.duration_hours} hrs)
          </span>
        </div>
      </div>
    </div>
  );
};
