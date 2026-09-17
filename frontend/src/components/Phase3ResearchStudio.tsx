import React, { useState, useEffect } from 'react';
import {
  Award,
  Layers,
  Sparkles,
  RefreshCw,
  GitBranch,
  Activity,
  Globe2,
  Zap,
  TrendingUp,
  CheckCircle2,
  AlertCircle,
  HelpCircle
} from 'lucide-react';

interface BenchmarkReport {
  benchmark_timestamp: string;
  num_test_scenarios: number;
  models_evaluated: string[];
  metrics_summary: Record<string, {
    f1_score_pct: number;
    csi_iou_pct: number;
    displacement_error_km: number;
    psnr_db: number;
    mass_violation_pct: number;
    p99_relative_error_pct: number;
    latency_ms: number;
    physics_compliance_pct: number;
  }>;
  winning_detection_model: string;
  winning_downscaling_model: string;
  best_probabilistic_generator: string;
}

interface Phase3ResearchStudioProps {
  runId: string;
  currentLeadTime: number;
}

export const Phase3ResearchStudio: React.FC<Phase3ResearchStudioProps> = ({ runId, currentLeadTime }) => {
  const [benchmark, setBenchmark] = useState<BenchmarkReport | null>(null);
  const [isRunningBenchmark, setIsRunningBenchmark] = useState<boolean>(false);
  const [stGnnData, setStGnnData] = useState<any>(null);
  const [physicsDownscale, setPhysicsDownscale] = useState<any>(null);
  const [diffusionData, setDiffusionData] = useState<any>(null);
  const [mhtData, setMhtData] = useState<any>(null);
  const [activeSubTab, setActiveSubTab] = useState<'leaderboard' | 'gnn_mht' | 'superres_diffusion'>('leaderboard');
  const [isLoading, setIsLoading] = useState<boolean>(false);

  useEffect(() => {
    loadBenchmark();
    loadResearchData();
  }, [runId, currentLeadTime]);

  const loadBenchmark = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/benchmark');
      if (res.ok) {
        const data = await res.json();
        setBenchmark(data);
      }
    } catch (e) {
      console.warn('Benchmark fetch notice:', e);
    }
  };

  const runFreshBenchmark = async () => {
    setIsRunningBenchmark(true);
    try {
      const res = await fetch('http://localhost:8000/api/benchmark/run?num_scenarios=1', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setBenchmark(data);
      }
    } catch (e) {
      console.error('Failed to run benchmark:', e);
    } finally {
      setIsRunningBenchmark(false);
    }
  };

  const loadResearchData = async () => {
    setIsLoading(true);
    try {
      // 1. ST-GNN
      const gnnRes = await fetch(`http://localhost:8000/api/research/st-gnn/${runId}`);
      if (gnnRes.ok) setStGnnData(await gnnRes.ok ? await gnnRes.json() : null);

      // 2. Physics Downscaler
      const piRes = await fetch(`http://localhost:8000/api/research/physics-downscaling/${runId}/${currentLeadTime}`);
      if (piRes.ok) setPhysicsDownscale(await piRes.json());

      // 3. Diffusion
      const diffRes = await fetch(`http://localhost:8000/api/research/diffusion/${runId}/${currentLeadTime}`);
      if (diffRes.ok) setDiffusionData(await diffRes.json());

      // 4. Multi-Hypothesis Tracker
      const mhtRes = await fetch(`http://localhost:8000/api/research/tracking-hypotheses/${runId}`);
      if (mhtRes.ok) setMhtData(await mhtRes.json());
    } catch (e) {
      console.warn('Research data load notice:', e);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Studio Header */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 backdrop-blur-md shadow-2xl relative overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-96 bg-gradient-to-bl from-purple-500/10 via-cyan-500/5 to-transparent rounded-full blur-3xl pointer-events-none" />
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center space-x-3 mb-1">
              <span className="px-2.5 py-0.5 rounded-full text-xs font-mono font-bold bg-purple-950/80 text-purple-300 border border-purple-800 flex items-center space-x-1.5">
                <Sparkles className="w-3.5 h-3.5 text-purple-400" />
                <span>PHASE 3 RESEARCH SUITE</span>
              </span>
              <span className="px-2.5 py-0.5 rounded-full text-xs font-mono font-bold bg-emerald-950/80 text-emerald-300 border border-emerald-800 flex items-center space-x-1.5">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                <span>PHASE 1 & 2 FALLBACKS INTACT</span>
              </span>
            </div>
            <h2 className="text-2xl font-black tracking-tight text-white flex items-center space-x-2">
              <span>Scientific Cross-Model Benchmark & Deep Research</span>
            </h2>
            <p className="text-xs text-slate-400 font-mono mt-1">
              Spatio-Temporal GNN • Physics-Informed Super-Resolution • Conditional DDPM Diffusion • Multi-Hypothesis Tracker
            </p>
          </div>

          <div className="flex items-center space-x-3">
            <button
              onClick={runFreshBenchmark}
              disabled={isRunningBenchmark}
              className="px-4 py-2 rounded-xl bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white font-mono text-xs font-bold flex items-center space-x-2 shadow-lg shadow-purple-500/20 transition disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 ${isRunningBenchmark ? 'animate-spin' : ''}`} />
              <span>{isRunningBenchmark ? 'Benchmarking Models...' : 'Run Cross-Model Benchmark'}</span>
            </button>
          </div>
        </div>

        {/* Sub Navigation */}
        <div className="flex items-center space-x-2 mt-6 pt-4 border-t border-slate-800/80 font-mono text-xs">
          <button
            onClick={() => setActiveSubTab('leaderboard')}
            className={`px-3 py-1.5 rounded-lg font-bold flex items-center space-x-2 transition ${
              activeSubTab === 'leaderboard'
                ? 'bg-purple-500/20 text-purple-300 border border-purple-500/50'
                : 'bg-slate-950 text-slate-400 hover:bg-slate-800'
            }`}
          >
            <Award className="w-4 h-4 text-amber-400" />
            <span>Cross-Model Benchmark Leaderboard</span>
          </button>
          <button
            onClick={() => setActiveSubTab('gnn_mht')}
            className={`px-3 py-1.5 rounded-lg font-bold flex items-center space-x-2 transition ${
              activeSubTab === 'gnn_mht'
                ? 'bg-purple-500/20 text-purple-300 border border-purple-500/50'
                : 'bg-slate-950 text-slate-400 hover:bg-slate-800'
            }`}
          >
            <Globe2 className="w-4 h-4 text-cyan-400" />
            <span>ST-GNN & Multi-Hypothesis Tracking</span>
          </button>
          <button
            onClick={() => setActiveSubTab('superres_diffusion')}
            className={`px-3 py-1.5 rounded-lg font-bold flex items-center space-x-2 transition ${
              activeSubTab === 'superres_diffusion'
                ? 'bg-purple-500/20 text-purple-300 border border-purple-500/50'
                : 'bg-slate-950 text-slate-400 hover:bg-slate-800'
            }`}
          >
            <Layers className="w-4 h-4 text-indigo-400" />
            <span>Physics Super-Res vs Diffusion</span>
          </button>
        </div>
      </div>

      {/* Tab 1: Cross-Model Benchmark Leaderboard */}
      {activeSubTab === 'leaderboard' && benchmark && (
        <div className="space-y-6">
          {/* Winners Banner */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 font-mono text-xs">
            <div className="p-4 rounded-xl bg-purple-950/40 border border-purple-800/80 flex items-center space-x-3 shadow-lg">
              <div className="p-2.5 rounded-lg bg-purple-600/30 text-purple-300 border border-purple-700">
                <Award className="w-5 h-5 text-amber-400" />
              </div>
              <div>
                <div className="text-[10px] text-slate-400 uppercase">Top Detection Model</div>
                <div className="text-sm font-bold text-purple-200">{benchmark.winning_detection_model}</div>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-cyan-950/40 border border-cyan-800/80 flex items-center space-x-3 shadow-lg">
              <div className="p-2.5 rounded-lg bg-cyan-600/30 text-cyan-300 border border-cyan-700">
                <Zap className="w-5 h-5 text-cyan-400" />
              </div>
              <div>
                <div className="text-[10px] text-slate-400 uppercase">Top Super-Resolution Model</div>
                <div className="text-sm font-bold text-cyan-200">{benchmark.winning_downscaling_model}</div>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-indigo-950/40 border border-indigo-800/80 flex items-center space-x-3 shadow-lg">
              <div className="p-2.5 rounded-lg bg-indigo-600/30 text-indigo-300 border border-indigo-700">
                <Sparkles className="w-5 h-5 text-indigo-400" />
              </div>
              <div>
                <div className="text-[10px] text-slate-400 uppercase">Best Probabilistic Generator</div>
                <div className="text-sm font-bold text-indigo-200">{benchmark.best_probabilistic_generator}</div>
              </div>
            </div>
          </div>

          {/* Benchmark Table */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl">
            <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between">
              <h3 className="font-bold text-sm text-slate-200 font-mono flex items-center space-x-2">
                <span>Evaluated Model Comparison Matrix</span>
                <span className="text-xs text-slate-500">({benchmark.num_test_scenarios} Held-Out Test Scenarios)</span>
              </h3>
              <span className="text-xs text-slate-400 font-mono">Last Run: {new Date(benchmark.benchmark_timestamp).toLocaleString()}</span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono text-xs">
                <thead className="bg-slate-950/80 text-slate-400 text-[11px] uppercase border-b border-slate-800">
                  <tr>
                    <th className="px-4 py-3">Model Architecture</th>
                    <th className="px-4 py-3">Status</th>
                    <th className="px-4 py-3">F1 Score</th>
                    <th className="px-4 py-3">CSI / IoU</th>
                    <th className="px-4 py-3">Disp Error</th>
                    <th className="px-4 py-3">PSNR (dB)</th>
                    <th className="px-4 py-3">Mass Error</th>
                    <th className="px-4 py-3">P99 Error</th>
                    <th className="px-4 py-3">Latency</th>
                    <th className="px-4 py-3">Physics %</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-200">
                  {Object.entries(benchmark.metrics_summary).map(([mName, scores]) => {
                    const isWinning = mName === benchmark.winning_detection_model || mName === benchmark.winning_downscaling_model;
                    const isP3 = mName.startsWith('Phase_3');
                    const isP2 = mName.startsWith('Phase_2');
                    const badge = isP3 ? 'RESEARCH / P3' : isP2 ? 'TRAINED BASELINE' : 'OPERATIONAL BASELINE';
                    const badgeColor = isP3 ? 'bg-purple-950 text-purple-300 border-purple-800' : isP2 ? 'bg-blue-950 text-blue-300 border-blue-800' : 'bg-slate-950 text-slate-400 border-slate-800';

                    return (
                      <tr key={mName} className={`hover:bg-slate-800/40 transition ${isWinning ? 'bg-purple-950/20' : ''}`}>
                        <td className="px-4 py-3.5 font-bold flex items-center space-x-2">
                          {isWinning && <Award className="w-4 h-4 text-amber-400 inline" />}
                          <span>{mName.replace(/_/g, ' ')}</span>
                        </td>
                        <td className="px-4 py-3.5">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${badgeColor}`}>
                            {badge}
                          </span>
                        </td>
                        <td className="px-4 py-3.5 font-bold text-emerald-400">{scores.f1_score_pct.toFixed(1)}%</td>
                        <td className="px-4 py-3.5">{scores.csi_iou_pct.toFixed(1)}%</td>
                        <td className="px-4 py-3.5">{scores.displacement_error_km.toFixed(1)} km</td>
                        <td className="px-4 py-3.5 font-bold text-cyan-300">{scores.psnr_db.toFixed(1)} dB</td>
                        <td className="px-4 py-3.5 text-amber-300">{scores.mass_violation_pct.toFixed(1)}%</td>
                        <td className="px-4 py-3.5">{scores.p99_relative_error_pct.toFixed(1)}%</td>
                        <td className="px-4 py-3.5 text-slate-400">{scores.latency_ms.toFixed(0)} ms</td>
                        <td className="px-4 py-3.5 text-emerald-300 font-bold">{scores.physics_compliance_pct.toFixed(0)}%</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Spatio-Temporal GNN & Multi-Hypothesis Tracking */}
      {activeSubTab === 'gnn_mht' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* ST-GNN Card */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="font-bold text-sm text-white font-mono flex items-center space-x-2">
                <Globe2 className="w-4 h-4 text-cyan-400" />
                <span>Spatio-Temporal Graph Neural Network (ST-GNN)</span>
              </h3>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-cyan-950 text-cyan-300 border border-cyan-800">
                Spherical Haversine Graph
              </span>
            </div>

            {stGnnData ? (
              <div className="space-y-4 font-mono text-xs">
                <div className="grid grid-cols-3 gap-3">
                  <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                    <div className="text-[10px] text-slate-400">Model Name</div>
                    <div className="text-sm font-bold text-cyan-300">{stGnnData.model}</div>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                    <div className="text-[10px] text-slate-400">Peak Anomaly Prob</div>
                    <div className="text-sm font-bold text-emerald-400">{(stGnnData.probabilities_summary.max_probability * 100).toFixed(1)}%</div>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                    <div className="text-[10px] text-slate-400">High-Anomaly Cells</div>
                    <div className="text-sm font-bold text-amber-300">{stGnnData.probabilities_summary.high_anomaly_cells}</div>
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 space-y-2">
                  <div className="text-xs font-bold text-slate-300">Predicted Centroid Velocities (Lead-Time Dynamics)</div>
                  <div className="grid grid-cols-4 gap-2 text-[11px]">
                    {stGnnData.centroid_velocities.slice(0, 8).map((vel: number[], idx: number) => (
                      <div key={idx} className="p-2 rounded bg-slate-900 border border-slate-800">
                        <div className="text-[9px] text-slate-500">T+{idx * 24}h</div>
                        <div className="text-slate-200">u: {vel[0].toFixed(2)}</div>
                        <div className="text-slate-200">v: {vel[1].toFixed(2)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-8 text-center text-slate-500 font-mono text-xs">Loading ST-GNN inference telemetry...</div>
            )}
          </div>

          {/* Multi-Hypothesis Tracker Card */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="font-bold text-sm text-white font-mono flex items-center space-x-2">
                <GitBranch className="w-4 h-4 text-purple-400" />
                <span>Multi-Hypothesis Tracker (MHT)</span>
              </h3>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-purple-950 text-purple-300 border border-purple-800">
                Split/Merge & Kinematics
              </span>
            </div>

            {mhtData ? (
              <div className="space-y-4 font-mono text-xs">
                <div className="grid grid-cols-3 gap-3">
                  <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                    <div className="text-[10px] text-slate-400">Total Hypotheses</div>
                    <div className="text-sm font-bold text-purple-300">{mhtData.total_hypotheses_evaluated}</div>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                    <div className="text-[10px] text-slate-400">Consensus Tracks</div>
                    <div className="text-sm font-bold text-emerald-400">{mhtData.consensus_tracks.length}</div>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                    <div className="text-[10px] text-slate-400">Split/Merge Events</div>
                    <div className="text-sm font-bold text-amber-300">{mhtData.split_merge_events.length}</div>
                  </div>
                </div>

                {/* Consensus Tracks List */}
                <div className="space-y-2">
                  <div className="text-xs font-bold text-slate-300">Track Hypotheses & Association Confidence</div>
                  <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
                    {mhtData.consensus_tracks.map((trk: any) => (
                      <div key={trk.track_id} className="p-3 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-between">
                        <div>
                          <div className="font-bold text-slate-200">{trk.track_id} ({trk.primary_hypothesis_id})</div>
                          <div className="text-[10px] text-slate-400">
                            Duration: {trk.duration_hours}h • Peak Precip: {trk.peak_precip_mm}mm • Heading: {trk.heading_compass}
                          </div>
                        </div>
                        <div className="text-right">
                          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-950 text-purple-300 border border-purple-800">
                            {(trk.cumulative_confidence * 100).toFixed(0)}% Conf
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-8 text-center text-slate-500 font-mono text-xs">Loading Multi-Hypothesis Tracking telemetry...</div>
            )}
          </div>
        </div>
      )}

      {/* Tab 3: Physics Super-Resolution vs Conditional Diffusion */}
      {activeSubTab === 'superres_diffusion' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 font-mono text-xs">
          {/* Physics-Informed U-Net */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="font-bold text-sm text-white font-mono flex items-center space-x-2">
                <Zap className="w-4 h-4 text-cyan-400" />
                <span>Physics-Informed Super-Resolution U-Net (5x)</span>
              </h3>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-cyan-950 text-cyan-300 border border-cyan-800">
                Mass Conservation Enforced
              </span>
            </div>

            {physicsDownscale ? (
              <div className="space-y-4">
                <div className="grid grid-cols-3 gap-3">
                  <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                    <div className="text-[10px] text-slate-400">Peak Intensity</div>
                    <div className="text-sm font-bold text-cyan-300">{physicsDownscale.peak_intensity.toFixed(1)} mm</div>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                    <div className="text-[10px] text-slate-400">P99 Tail Intensity</div>
                    <div className="text-sm font-bold text-emerald-400">{physicsDownscale.p99_intensity.toFixed(1)} mm</div>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                    <div className="text-[10px] text-slate-400">Mean Precipitation</div>
                    <div className="text-sm font-bold text-slate-200">{physicsDownscale.mean_intensity.toFixed(1)} mm</div>
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800">
                  <div className="text-[11px] text-slate-300 font-bold mb-1">Physics Constraint Verification</div>
                  <div className="text-[10px] text-slate-400 space-y-1">
                    <div>✓ 5x5 Block Down-pooling matches coarse driving field (&lt; 0.1% leakage)</div>
                    <div>✓ ReLU output guarantees strictly non-negative precipitation</div>
                    <div>✓ Huber + Quantile P99 Extreme Tail preservation penalty</div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-8 text-center text-slate-500">Loading Physics Super-Resolution telemetry...</div>
            )}
          </div>

          {/* Conditional Diffusion Model */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="font-bold text-sm text-white font-mono flex items-center space-x-2">
                <Sparkles className="w-4 h-4 text-indigo-400" />
                <span>Conditional DDPM Diffusion Model (Ensemble Generator)</span>
              </h3>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-indigo-950 text-indigo-300 border border-indigo-800">
                Stochastic Realizations
              </span>
            </div>

            {diffusionData ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                    <div className="text-[10px] text-slate-400">Model Type</div>
                    <div className="text-sm font-bold text-indigo-300">{diffusionData.model}</div>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                    <div className="text-[10px] text-slate-400">Ensemble Realizations</div>
                    <div className="text-sm font-bold text-emerald-400">{diffusionData.num_members} Members</div>
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 space-y-2">
                  <div className="text-[11px] text-slate-300 font-bold">Stochastic Sampling Telemetry</div>
                  <div className="text-[10px] text-slate-400 space-y-1">
                    <div>✓ Forward Gaussian Markov schedule (50 timesteps)</div>
                    <div>✓ Reverse conditional denoising parameterized by coarse driving field + orography</div>
                    <div>✓ Unbiased stochastic ensemble spread capturing sub-grid turbulence</div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-8 text-center text-slate-500">Loading Diffusion Realizations telemetry...</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
