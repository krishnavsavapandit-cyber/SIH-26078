import React from 'react';
import { UncertaintyCone } from '../services/api';
import { ShieldCheck, GitCommit, Compass, Sparkles } from 'lucide-react';

interface TrajectoryConeViewerProps {
  uncertainty: UncertaintyCone | null | undefined;
  currentLead: number;
}

export const TrajectoryConeViewer: React.FC<TrajectoryConeViewerProps> = ({ uncertainty, currentLead }) => {
  if (!uncertainty || !uncertainty.cone_points || uncertainty.cone_points.length === 0) {
    return (
      <div className="glass-panel p-5 rounded-xl border border-slate-700 text-center text-slate-400">
        <p className="text-xs">No ensemble uncertainty cone available</p>
      </div>
    );
  }

  const curPoint = uncertainty.cone_points.find((p) => p.lead_time === currentLead) || uncertainty.cone_points[0];

  return (
    <div className="glass-panel p-5 rounded-xl border border-slate-700/80 shadow-xl space-y-4">
      <div className="flex items-center justify-between border-b border-slate-700/60 pb-3">
        <div className="flex items-center space-x-2">
          <ShieldCheck className="w-5 h-5 text-cyan-400" />
          <h3 className="text-sm font-semibold text-slate-200">Ensemble Uncertainty & Cone of Probability</h3>
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800">
          10 Ensemble Members
        </span>
      </div>

      {/* Uncertainty Radius & Quantiles */}
      <div className="grid grid-cols-3 gap-2.5 font-mono text-center">
        <div className="bg-slate-900/60 p-2 rounded-lg border border-slate-800">
          <div className="text-[10px] text-slate-400">Cone Radius (T+{currentLead}h)</div>
          <div className="text-sm font-bold text-cyan-400 mt-0.5">
            ±{curPoint.cone_radius_km.toFixed(1)} km
          </div>
        </div>
        <div className="bg-slate-900/60 p-2 rounded-lg border border-slate-800">
          <div className="text-[10px] text-slate-400">Precip Range (P10-P90)</div>
          <div className="text-xs font-bold text-slate-200 mt-0.5">
            {curPoint.precip_p10.toFixed(0)} - {curPoint.precip_p90.toFixed(0)} mm
          </div>
        </div>
        <div className="bg-slate-900/60 p-2 rounded-lg border border-slate-800">
          <div className="text-[10px] text-slate-400">MSLP Range (P10-P90)</div>
          <div className="text-xs font-bold text-amber-300 mt-0.5">
            {curPoint.mslp_p10.toFixed(0)} - {curPoint.mslp_p90.toFixed(0)} hPa
          </div>
        </div>
      </div>

      {/* Lead-Time Uncertainty Progression Table */}
      <div className="overflow-x-auto max-h-44">
        <table className="w-full text-[11px] font-mono text-left text-slate-300">
          <thead className="bg-slate-900/90 text-slate-400 sticky top-0">
            <tr>
              <th className="py-1.5 px-2">Lead</th>
              <th className="py-1.5 px-2">Center Position</th>
              <th className="py-1.5 px-2">Cone Radius</th>
              <th className="py-1.5 px-2">P50 Precip</th>
              <th className="py-1.5 px-2">P50 MSLP</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {uncertainty.cone_points.map((pt) => {
              const isSelected = pt.lead_time === currentLead;
              return (
                <tr
                  key={pt.lead_time}
                  className={`transition ${
                    isSelected ? 'bg-cyan-950/60 text-cyan-300 font-bold' : 'hover:bg-slate-800/40'
                  }`}
                >
                  <td className="py-1 px-2">+{pt.lead_time}h</td>
                  <td className="py-1 px-2">{pt.center_lat.toFixed(2)}°N, {pt.center_lon.toFixed(2)}°E</td>
                  <td className="py-1 px-2">{pt.cone_radius_km.toFixed(1)} km</td>
                  <td className="py-1 px-2">{pt.precip_p50.toFixed(1)} mm</td>
                  <td className="py-1 px-2">{pt.mslp_p50.toFixed(1)} hPa</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
