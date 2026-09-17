import React from 'react';
import { ProvenanceRecord } from '../services/api';
import { Fingerprint, Lock, ShieldCheck, Cpu, Terminal, Clock, Sparkles } from 'lucide-react';

interface ProvenanceInspectorProps {
  provenance: ProvenanceRecord | null;
}

export const ProvenanceInspector: React.FC<ProvenanceInspectorProps> = ({ provenance }) => {
  if (!provenance) {
    return (
      <div className="glass-panel p-6 rounded-xl border border-slate-700 text-center text-slate-400">
        <p className="text-xs">No cryptographic provenance log recorded</p>
      </div>
    );
  }

  return (
    <div className="glass-panel p-5 rounded-xl border border-slate-700/80 shadow-xl space-y-4">
      <div className="flex items-center justify-between border-b border-slate-700/60 pb-3">
        <div className="flex items-center space-x-2">
          <Fingerprint className="w-5 h-5 text-cyan-400" />
          <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
            Scientific Provenance & Audit Trail
          </h3>
        </div>
        <div className="flex items-center space-x-1.5 px-2 py-0.5 rounded bg-emerald-950/80 text-emerald-300 border border-emerald-500/40 text-[10px] font-mono">
          <ShieldCheck className="w-3.5 h-3.5" />
          <span>REPRODUCIBLE RUN</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs font-mono">
        <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800 space-y-1">
          <div className="text-slate-400 text-[10px] flex items-center space-x-1">
            <Lock className="w-3.5 h-3.5 text-cyan-400" />
            <span>Dataset Cryptographic SHA-256</span>
          </div>
          <div className="text-cyan-300 font-bold break-all text-[11px]">
            {provenance.dataset_sha256}
          </div>
        </div>

        <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800 space-y-1">
          <div className="text-slate-400 text-[10px] flex items-center space-x-1">
            <Cpu className="w-3.5 h-3.5 text-amber-400" />
            <span>Model Version & Deterministic Seed</span>
          </div>
          <div className="text-slate-200 text-xs">
            Model: <strong className="text-amber-300">{provenance.model_version}</strong> | Seed: <strong className="text-cyan-400">{provenance.random_seed}</strong>
          </div>
        </div>
      </div>

      {/* Execution Pipeline Steps */}
      <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800 space-y-2 text-xs font-mono">
        <div className="text-slate-400 text-[10px] flex items-center justify-between">
          <span className="flex items-center space-x-1">
            <Terminal className="w-3.5 h-3.5 text-emerald-400" />
            <span>Pipeline Execution Sequence</span>
          </span>
          <span className="text-cyan-400 flex items-center space-x-1">
            <Clock className="w-3 h-3" />
            <span>{provenance.execution_duration_sec.toFixed(2)}s wall-clock</span>
          </span>
        </div>

        <div className="flex flex-wrap gap-1.5 pt-1">
          {provenance.pipeline_steps_executed.map((step, idx) => (
            <span
              key={idx}
              className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700 text-[10px]"
            >
              {idx + 1}. {step}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
};
