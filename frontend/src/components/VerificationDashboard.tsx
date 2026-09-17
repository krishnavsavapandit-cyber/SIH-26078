import React from 'react';
import { VerificationReport } from '../services/api';
import { CheckCircle2, AlertTriangle, Crosshair, BarChart3, Scale, Layers } from 'lucide-react';

interface VerificationDashboardProps {
  report: VerificationReport | null;
}

export const VerificationDashboard: React.FC<VerificationDashboardProps> = ({ report }) => {
  if (!report) {
    return (
      <div className="glass-panel p-6 rounded-xl border border-slate-700 text-center text-slate-400">
        <p className="text-sm">No verification telemetry available for this run</p>
      </div>
    );
  }

  const c = report.contingency;
  const te = report.trajectory_error;
  const fm = report.field_metrics;

  return (
    <div className="space-y-5">
      {/* Overview Banner */}
      <div className="glass-panel p-4 rounded-xl border border-cyan-500/30 flex items-center justify-between shadow-lg">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 rounded-lg bg-cyan-500/20 text-cyan-400 border border-cyan-500/40">
            <CheckCircle2 className="w-6 h-6" />
          </div>
          <div>
            <h3 className="font-bold text-slate-100 text-sm">Ground Truth Quantitative Verification</h3>
            <p className="text-xs text-slate-400">
              Evaluated against hidden deterministic synthetic ground truth ({te.track_points_evaluated} forecast steps)
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-4">
          <div className="text-right font-mono">
            <div className="text-[10px] text-slate-400 uppercase">Overall F1-Score</div>
            <div className="text-xl font-bold text-cyan-400">{(c.f1_score * 100).toFixed(1)}%</div>
          </div>
          <div className="text-right font-mono">
            <div className="text-[10px] text-slate-400 uppercase">Spatial IoU</div>
            <div className="text-xl font-bold text-emerald-400">{(c.spatial_iou * 100).toFixed(1)}%</div>
          </div>
        </div>
      </div>

      {/* Contingency Table & Spatial Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Spatial Contingency Matrix */}
        <div className="glass-panel p-4 rounded-xl border border-slate-700/80 shadow-xl space-y-3">
          <div className="flex items-center space-x-2 border-b border-slate-700/60 pb-2">
            <Crosshair className="w-4 h-4 text-cyan-400" />
            <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider">Spatial Detection Contingency</h4>
          </div>

          <div className="grid grid-cols-2 gap-2 text-center font-mono text-xs">
            <div className="bg-emerald-950/40 border border-emerald-500/30 p-2.5 rounded-lg">
              <div className="text-[10px] text-emerald-400">True Positives (TP)</div>
              <div className="text-base font-bold text-emerald-300 mt-0.5">{c.true_positives.toLocaleString()} px</div>
            </div>
            <div className="bg-rose-950/40 border border-rose-500/30 p-2.5 rounded-lg">
              <div className="text-[10px] text-rose-400">False Positives (FP)</div>
              <div className="text-base font-bold text-rose-300 mt-0.5">{c.false_positives.toLocaleString()} px</div>
            </div>
            <div className="bg-amber-950/40 border border-amber-500/30 p-2.5 rounded-lg">
              <div className="text-[10px] text-amber-400">False Negatives (FN)</div>
              <div className="text-base font-bold text-amber-300 mt-0.5">{c.false_negatives.toLocaleString()} px</div>
            </div>
            <div className="bg-slate-900 border border-slate-800 p-2.5 rounded-lg">
              <div className="text-[10px] text-slate-400">True Negatives (TN)</div>
              <div className="text-base font-bold text-slate-300 mt-0.5">{c.true_negatives.toLocaleString()} px</div>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-2 font-mono text-xs pt-1">
            <div className="text-center p-2 rounded bg-slate-900/60 border border-slate-800">
              <div className="text-[10px] text-slate-400">Precision</div>
              <div className="font-bold text-cyan-300 mt-0.5">{(c.precision * 100).toFixed(1)}%</div>
            </div>
            <div className="text-center p-2 rounded bg-slate-900/60 border border-slate-800">
              <div className="text-[10px] text-slate-400">Recall (POD)</div>
              <div className="font-bold text-cyan-300 mt-0.5">{(c.recall * 100).toFixed(1)}%</div>
            </div>
            <div className="text-center p-2 rounded bg-slate-900/60 border border-slate-800">
              <div className="text-[10px] text-slate-400">Threat (CSI)</div>
              <div className="font-bold text-cyan-300 mt-0.5">{(c.csi_threat_score * 100).toFixed(1)}%</div>
            </div>
          </div>
        </div>

        {/* Trajectory & Kinematic Errors */}
        <div className="glass-panel p-4 rounded-xl border border-slate-700/80 shadow-xl space-y-3">
          <div className="flex items-center space-x-2 border-b border-slate-700/60 pb-2">
            <Scale className="w-4 h-4 text-cyan-400" />
            <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider">Kinematic & Continuous Errors</h4>
          </div>

          <div className="space-y-2.5 text-xs font-mono">
            <div className="flex justify-between items-center p-2 rounded bg-slate-900/70 border border-slate-800">
              <span className="text-slate-400">Mean Centroid Displacement Error:</span>
              <span className="font-bold text-cyan-300">{te.mean_displacement_km.toFixed(2)} km</span>
            </div>
            <div className="flex justify-between items-center p-2 rounded bg-slate-900/70 border border-slate-800">
              <span className="text-slate-400">Max Track Displacement Error:</span>
              <span className="font-bold text-amber-300">{te.max_displacement_km.toFixed(2)} km</span>
            </div>
            <div className="flex justify-between items-center p-2 rounded bg-slate-900/70 border border-slate-800">
              <span className="text-slate-400">Peak Rainfall Intensity Error:</span>
              <span className="font-bold text-slate-200">{te.mean_peak_precip_error_mm.toFixed(1)} mm / 6h</span>
            </div>
            <div className="flex justify-between items-center p-2 rounded bg-slate-900/70 border border-slate-800">
              <span className="text-slate-400">Precipitation Field RMSE:</span>
              <span className="font-bold text-slate-200">
                {fm.precipitation ? fm.precipitation.rmse.toFixed(2) : '0.00'} mm
              </span>
            </div>
            <div className="flex justify-between items-center p-2 rounded bg-slate-900/70 border border-slate-800">
              <span className="text-slate-400">MSLP Field RMSE / Pearson r:</span>
              <span className="font-bold text-slate-200">
                {fm.mslp ? `${fm.mslp.rmse.toFixed(2)} hPa (r=${fm.mslp.pearson_r.toFixed(3)})` : 'N/A'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Step-by-Step Lead Time Breakdown */}
      <div className="glass-panel p-4 rounded-xl border border-slate-700/80 shadow-xl space-y-3">
        <div className="flex items-center space-x-2 border-b border-slate-700/60 pb-2">
          <BarChart3 className="w-4 h-4 text-cyan-400" />
          <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider">Lead-Time Step Breakdown vs Hidden Truth</h4>
        </div>

        <div className="overflow-x-auto max-h-56">
          <table className="w-full text-xs font-mono text-left text-slate-300">
            <thead className="bg-slate-900/90 text-slate-400 sticky top-0">
              <tr>
                <th className="py-2 px-2.5">Lead</th>
                <th className="py-2 px-2.5">GT Centroid</th>
                <th className="py-2 px-2.5">Pred Centroid</th>
                <th className="py-2 px-2.5">Disp Error</th>
                <th className="py-2 px-2.5">GT Peak Precip</th>
                <th className="py-2 px-2.5">Pred Peak Precip</th>
                <th className="py-2 px-2.5">Step IoU</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {report.lead_time_breakdowns.map((b) => (
                <tr key={b.lead_time} className="hover:bg-slate-800/40 transition">
                  <td className="py-1.5 px-2.5 font-bold text-cyan-400">+{b.lead_time}h</td>
                  <td className="py-1.5 px-2.5">{b.gt_lat.toFixed(2)}°N, {b.gt_lon.toFixed(2)}°E</td>
                  <td className="py-1.5 px-2.5">
                    {b.pred_lat ? `${b.pred_lat.toFixed(2)}°N, ${b.pred_lon?.toFixed(2)}°E` : <span className="text-rose-400">MISSED</span>}
                  </td>
                  <td className="py-1.5 px-2.5 text-cyan-300">
                    {b.displacement_error_km ? `${b.displacement_error_km.toFixed(1)} km` : '-'}
                  </td>
                  <td className="py-1.5 px-2.5">{b.gt_peak_precip_mm.toFixed(1)} mm</td>
                  <td className="py-1.5 px-2.5 text-slate-200">{b.pred_peak_precip_mm.toFixed(1)} mm</td>
                  <td className="py-1.5 px-2.5 font-semibold text-emerald-400">{(b.step_iou * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
