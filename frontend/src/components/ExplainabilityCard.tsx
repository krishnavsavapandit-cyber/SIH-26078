import React, { useEffect, useState } from 'react';
import { HelpCircle, Sparkles, TrendingUp, BarChart, Layers } from 'lucide-react';

interface ExplainabilityCardProps {
  runId: string;
  eventId: string | undefined;
}

export const ExplainabilityCard: React.FC<ExplainabilityCardProps> = ({ runId, eventId }) => {
  const [data, setData] = useState<any | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    if (eventId) fetchExplainability();
  }, [runId, eventId]);

  const fetchExplainability = async () => {
    if (!eventId) return;
    setLoading(true);
    try {
      const res = await fetch(`/api/explainability/${runId}/${eventId}`);
      if (res.ok) {
        const json = await res.json();
        setData(json);
      }
    } catch (e) {
      console.warn('Explainability notice:', e);
    } finally {
      setLoading(false);
    }
  };

  if (loading || !data) {
    return (
      <div className="glass-panel p-5 rounded-xl border border-slate-700 text-center text-slate-400">
        <p className="text-xs font-mono">Computing attribution diagnostics...</p>
      </div>
    );
  }

  return (
    <div className="glass-panel p-5 rounded-xl border border-slate-700/80 shadow-xl space-y-4">
      <div className="flex items-center justify-between border-b border-slate-700/60 pb-3">
        <div className="flex items-center space-x-2">
          <HelpCircle className="w-5 h-5 text-amber-400" />
          <h3 className="text-sm font-semibold text-slate-200">
            Explainability & Feature Attribution Breakdown
          </h3>
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-950 text-amber-300 border border-amber-800">
          Dominant: {data.dominant_driver}
        </span>
      </div>

      <div className="space-y-3">
        {data.feature_contributions.map((fc: any, idx: number) => (
          <div key={idx} className="space-y-1 font-mono text-xs">
            <div className="flex justify-between text-slate-300">
              <span>{fc.feature}</span>
              <span className="font-bold text-cyan-400">{fc.contribution_pct.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-slate-900 rounded-full h-1.5 overflow-hidden">
              <div
                className="bg-gradient-to-r from-cyan-500 to-amber-400 h-full rounded-full transition-all duration-500"
                style={{ width: `${fc.contribution_pct}%` }}
              />
            </div>
            <div className="text-[10px] text-slate-400">{fc.evidence_value}</div>
          </div>
        ))}
      </div>
    </div>
  );
};
