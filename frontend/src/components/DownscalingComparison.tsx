import React, { useEffect, useState } from 'react';
import { Sparkles, ArrowRight, Gauge, Layers, CheckCircle2 } from 'lucide-react';

interface DownscalingComparisonProps {
  runId: string;
  leadTime: number;
}

export const DownscalingComparison: React.FC<DownscalingComparisonProps> = ({ runId, leadTime }) => {
  const [data, setData] = useState<any | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    fetchData();
  }, [runId, leadTime]);

  const fetchData = async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/downscaling/${runId}/${leadTime}`);
      if (res.ok) {
        const json = await res.json();
        setData(json);
      }
    } catch (e) {
      console.warn('Downscaling load notice:', e);
    } finally {
      setLoading(false);
    }
  };

  if (loading || !data) {
    return (
      <div className="glass-panel p-6 rounded-xl border border-slate-700 text-center text-slate-400">
        <Sparkles className="w-8 h-8 text-cyan-400 mx-auto mb-2 animate-spin" />
        <p className="text-xs font-mono">Computing 5x Neural Super-Resolution Downscaling...</p>
      </div>
    );
  }

  const m = data.metrics;
  const bic = m.bicubic_baseline;
  const ml = m.ml_downscaler;
  const gt = m.ground_truth_stats;

  return (
    <div className="glass-panel p-5 rounded-xl border border-slate-700/80 shadow-xl space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-700/60 pb-3">
        <div className="flex items-center space-x-2">
          <Sparkles className="w-5 h-5 text-cyan-400" />
          <div>
            <h3 className="text-sm font-semibold text-slate-100">
              Learned Meteorological Downscaling (25km → 5km Super-Resolution)
            </h3>
            <p className="text-[11px] text-slate-400 font-mono">
              Bicubic Spline Baseline vs Physics-Aware Convolutional Downscaler (Lead: T+{leadTime}h)
            </p>
          </div>
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800">
          5x SPATIAL RESOLUTION
        </span>
      </div>

      {/* Comparative Metrics Leaderboard */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Bicubic Baseline Card */}
        <div className="bg-slate-900/80 p-3.5 rounded-xl border border-slate-800 space-y-2.5">
          <div className="flex justify-between items-center text-xs font-bold text-slate-300">
            <span>Bicubic Spline Baseline</span>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">
              Interpolation
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs font-mono">
            <div className="bg-slate-950/60 p-2 rounded border border-slate-800">
              <div className="text-[10px] text-slate-400">RMSE</div>
              <div className="font-bold text-slate-200 mt-0.5">{bic.rmse.toFixed(2)} mm</div>
            </div>
            <div className="bg-slate-950/60 p-2 rounded border border-slate-800">
              <div className="text-[10px] text-slate-400">PSNR</div>
              <div className="font-bold text-slate-200 mt-0.5">{bic.psnr_db.toFixed(1)} dB</div>
            </div>
            <div className="bg-slate-950/60 p-2 rounded border border-slate-800">
              <div className="text-[10px] text-slate-400">P99 Value Error</div>
              <div className="font-bold text-rose-400 mt-0.5">{(bic.p99_relative_error * 100).toFixed(1)}%</div>
            </div>
            <div className="bg-slate-950/60 p-2 rounded border border-slate-800">
              <div className="text-[10px] text-slate-400">Peak Intensity</div>
              <div className="font-bold text-slate-300 mt-0.5">{bic.peak_intensity.toFixed(1)} mm</div>
            </div>
          </div>
        </div>

        {/* ML Super-Resolution Downscaler Card */}
        <div className="bg-slate-900/80 p-3.5 rounded-xl border border-cyan-500/40 shadow-lg shadow-cyan-500/5 space-y-2.5">
          <div className="flex justify-between items-center text-xs font-bold text-cyan-300">
            <span className="flex items-center space-x-1.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>Learned CNN Downscaler</span>
            </span>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800">
              Trained Model
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs font-mono">
            <div className="bg-slate-950/60 p-2 rounded border border-cyan-900/40">
              <div className="text-[10px] text-slate-400">RMSE</div>
              <div className="font-bold text-emerald-400 mt-0.5">{ml.rmse.toFixed(2)} mm</div>
            </div>
            <div className="bg-slate-950/60 p-2 rounded border border-cyan-900/40">
              <div className="text-[10px] text-slate-400">PSNR</div>
              <div className="font-bold text-cyan-300 mt-0.5">{ml.psnr_db.toFixed(1)} dB</div>
            </div>
            <div className="bg-slate-950/60 p-2 rounded border border-cyan-900/40">
              <div className="text-[10px] text-slate-400">P99 Value Error</div>
              <div className="font-bold text-emerald-400 mt-0.5">{(ml.p99_relative_error * 100).toFixed(1)}%</div>
            </div>
            <div className="bg-slate-950/60 p-2 rounded border border-cyan-900/40">
              <div className="text-[10px] text-slate-400">Peak Intensity</div>
              <div className="font-bold text-cyan-300 mt-0.5">{ml.peak_intensity.toFixed(1)} mm</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
