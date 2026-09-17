import React, { useEffect, useState } from 'react';
import { ShieldCheck, AlertOctagon, CheckCircle2, Info, AlertTriangle } from 'lucide-react';

interface PhysicsAuditCardProps {
  runId: string;
  leadTime: number;
}

export const PhysicsAuditCard: React.FC<PhysicsAuditCardProps> = ({ runId, leadTime }) => {
  const [audit, setAudit] = useState<any | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    fetchAudit();
  }, [runId, leadTime]);

  const fetchAudit = async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/physics-audit/${runId}/${leadTime}`);
      if (res.ok) {
        const json = await res.json();
        setAudit(json);
      }
    } catch (e) {
      console.warn('Physics audit load notice:', e);
    } finally {
      setLoading(false);
    }
  };

  if (loading || !audit) {
    return (
      <div className="glass-panel p-5 rounded-xl border border-slate-700 text-center text-slate-400">
        <p className="text-xs font-mono">Running atmospheric invariant audit...</p>
      </div>
    );
  }

  return (
    <div className="glass-panel p-5 rounded-xl border border-slate-700/80 shadow-xl space-y-4">
      <div className="flex items-center justify-between border-b border-slate-700/60 pb-3">
        <div className="flex items-center space-x-2">
          <ShieldCheck className="w-5 h-5 text-emerald-400" />
          <h3 className="text-sm font-semibold text-slate-200">Atmospheric Physics & Conservation Audit</h3>
        </div>
        <span
          className={`text-[10px] font-mono px-2 py-0.5 rounded font-bold border ${
            audit.passed
              ? 'bg-emerald-950 text-emerald-300 border-emerald-700'
              : 'bg-rose-950 text-rose-300 border-rose-700'
          }`}
        >
          {audit.passed ? 'ALL CHECKS PASSED' : `${audit.violations_count} VIOLATIONS`}
        </span>
      </div>

      <div className="space-y-2">
        {audit.checks.map((c: any, idx: number) => (
          <div
            key={idx}
            className="flex items-center justify-between p-2.5 rounded-lg bg-slate-900/70 border border-slate-800 text-xs font-mono"
          >
            <div className="flex items-center space-x-2">
              {c.passed ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
              ) : (
                <AlertOctagon className="w-4 h-4 text-rose-400 flex-shrink-0" />
              )}
              <div>
                <div className="font-semibold text-slate-200">{c.check_name}</div>
                <div className="text-[10px] text-slate-400">
                  Observed: [{c.min_observed.toFixed(1)}, {c.max_observed.toFixed(1)}] | Allowed: [{c.allowed_range[0]}, {c.allowed_range[1]}]
                </div>
              </div>
            </div>

            <span
              className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                c.passed ? 'bg-emerald-950/60 text-emerald-400' : 'bg-rose-950/60 text-rose-400'
              }`}
            >
              {c.passed ? 'VALID' : 'FAIL'}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};
