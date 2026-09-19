import React from 'react';
import { WeatherEvent, TrajectoryPoint } from '../services/api';
import {
  Activity,
  Compass,
  Wind,
  Droplets,
  MapPin,
  TrendingUp,
  ShieldAlert,
  ChevronDown,
  ChevronUp,
  Layers,
  Sparkles
} from 'lucide-react';

interface EventDetailPanelProps {
  event: WeatherEvent | null;
  currentLeadTime: number;
  isCollapsed?: boolean;
  onToggleCollapse?: () => void;
  onLayerSelect?: (layer: 'precipitation' | 'efi' | 'mslp' | 'wind') => void;
  showUncertaintyCone?: boolean;
  onToggleUncertaintyCone?: () => void;
  showSpaghetti?: boolean;
  onToggleSpaghetti?: () => void;
}

export const EventDetailPanel: React.FC<EventDetailPanelProps> = ({
  event,
  currentLeadTime,
  isCollapsed = false,
  onToggleCollapse,
  showUncertaintyCone = true,
  onToggleUncertaintyCone,
  showSpaghetti = true,
  onToggleSpaghetti
}) => {
  if (!event) {
    return (
      <div className="glass-panel p-5 rounded-2xl border border-slate-700/70 text-center text-slate-400">
        <Activity className="w-7 h-7 text-slate-600 mx-auto mb-2 animate-pulse" />
        <p className="text-xs font-mono">No extreme weather anomaly currently selected</p>
      </div>
    );
  }

  const currentPoint: TrajectoryPoint | undefined =
    event.trajectory_points?.find((p) => p.lead_time === currentLeadTime) || event.trajectory_points?.[0];

  const severityScore = currentPoint?.severity_score ?? event.peak_severity ?? 0.8;
  const isExtreme = severityScore > 0.65;

  return (
    <div className="glass-panel p-4 rounded-2xl border border-slate-700/80 shadow-2xl space-y-3">
      {/* Header Bar */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-lg bg-rose-500/20 text-rose-400 border border-rose-500/40">
            <ShieldAlert className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-mono font-bold text-sm text-cyan-300">{event.track_id}</span>
              <span
                className={`px-2 py-0.5 rounded text-[10px] font-bold border ${
                  isExtreme
                    ? 'bg-rose-500/20 text-rose-300 border-rose-500/50'
                    : 'bg-amber-500/20 text-amber-300 border-amber-500/50'
                }`}
              >
                {currentPoint?.lifecycle_state || 'INTENSIFYING'}
              </span>
            </div>
            <h4 className="text-xs font-semibold text-slate-200">{event.event_name}</h4>
          </div>
        </div>

        {onToggleCollapse && (
          <button
            onClick={onToggleCollapse}
            className="p-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-slate-200 transition"
          >
            {isCollapsed ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
          </button>
        )}
      </div>

      {!isCollapsed && (
        <>
          {/* Key Metrics Grid */}
          <div className="grid grid-cols-2 gap-2 font-mono text-xs">
            <div className="bg-slate-950/70 p-2.5 rounded-xl border border-slate-800/80">
              <div className="text-[10px] text-slate-400 flex items-center space-x-1">
                <Droplets className="w-3.5 h-3.5 text-cyan-400" />
                <span>Peak Rainfall (6h)</span>
              </div>
              <div className="text-sm font-bold text-cyan-300 mt-0.5">
                {currentPoint ? currentPoint.peak_precip_mm.toFixed(1) : event.peak_precip_mm.toFixed(1)} mm
              </div>
            </div>

            <div className="bg-slate-950/70 p-2.5 rounded-xl border border-slate-800/80">
              <div className="text-[10px] text-slate-400 flex items-center space-x-1">
                <TrendingUp className="w-3.5 h-3.5 text-rose-400" />
                <span>Central Pressure</span>
              </div>
              <div className="text-sm font-bold text-rose-300 mt-0.5">
                {currentPoint ? currentPoint.min_mslp_hpa.toFixed(0) : event.min_mslp_hpa.toFixed(0)} hPa
              </div>
            </div>

            <div className="bg-slate-950/70 p-2.5 rounded-xl border border-slate-800/80">
              <div className="text-[10px] text-slate-400 flex items-center space-x-1">
                <Compass className="w-3.5 h-3.5 text-emerald-400" />
                <span>Heading / Bearing</span>
              </div>
              <div className="text-sm font-bold text-emerald-300 mt-0.5">
                {event.heading_compass || 'WNW'} ({currentPoint ? currentPoint.step_bearing_deg.toFixed(0) : '290'}°)
              </div>
            </div>

            <div className="bg-slate-950/70 p-2.5 rounded-xl border border-slate-800/80">
              <div className="text-[10px] text-slate-400 flex items-center space-x-1">
                <Wind className="w-3.5 h-3.5 text-blue-400" />
                <span>Forward Speed</span>
              </div>
              <div className="text-sm font-bold text-slate-200 mt-0.5">
                {currentPoint ? currentPoint.step_speed_kmh.toFixed(1) : event.mean_speed_kmh.toFixed(1)} km/h
              </div>
            </div>
          </div>

          {/* Current Centroid & Bounding Footprint Coordinates */}
          <div className="bg-slate-950/80 p-2.5 rounded-xl border border-slate-800/90 text-xs font-mono space-y-1">
            <div className="flex justify-between items-center text-slate-400">
              <span className="flex items-center space-x-1">
                <MapPin className="w-3 h-3 text-cyan-400" />
                <span>Centroid Coordinates:</span>
              </span>
              <span className="font-bold text-slate-100">
                {currentPoint ? `${currentPoint.lat.toFixed(2)}°N, ${currentPoint.lon.toFixed(2)}°E` : 'N/A'}
              </span>
            </div>
            <div className="flex justify-between items-center text-slate-400 text-[11px]">
              <span>Dynamic Area Footprint:</span>
              <span className="text-cyan-300">
                {currentPoint ? `${currentPoint.area_km2.toLocaleString()} km²` : 'N/A'}
              </span>
            </div>
            <div className="flex justify-between items-center text-slate-400 text-[11px]">
              <span>Forecast Span:</span>
              <span className="text-slate-300">
                T+{event.start_lead_time}h → T+{event.end_lead_time}h ({event.duration_hours}h Horizon)
              </span>
            </div>
          </div>

          {/* Map Layer Overlay Toggles (Ensemble & Uncertainty) */}
          <div className="pt-1 border-t border-slate-800/80 flex items-center justify-between text-xs font-mono">
            <label className="flex items-center space-x-1.5 cursor-pointer text-slate-300 hover:text-white">
              <input
                type="checkbox"
                checked={showUncertaintyCone}
                onChange={onToggleUncertaintyCone}
                className="rounded bg-slate-900 border-slate-700 text-cyan-500 focus:ring-0 accent-cyan-500"
              />
              <span className="text-[11px]">Uncertainty Cone</span>
            </label>

            <label className="flex items-center space-x-1.5 cursor-pointer text-slate-300 hover:text-white">
              <input
                type="checkbox"
                checked={showSpaghetti}
                onChange={onToggleSpaghetti}
                className="rounded bg-slate-900 border-slate-700 text-cyan-500 focus:ring-0 accent-cyan-500"
              />
              <span className="text-[11px]">Ensemble Members</span>
            </label>
          </div>
        </>
      )}
    </div>
  );
};
