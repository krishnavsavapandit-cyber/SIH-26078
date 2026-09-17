import React, { useState, useEffect } from 'react';
import { api, WeatherEvent, ForecastFieldData, EFIData, VerificationReport, AlertRecord, ProvenanceRecord } from './services/api';
import { WeatherMap } from './components/WeatherMap';
import { TimelineSlider } from './components/TimelineSlider';
import { EventDetailPanel } from './components/EventDetailPanel';
import { TrajectoryConeViewer } from './components/TrajectoryConeViewer';
import { VerificationDashboard } from './components/VerificationDashboard';
import { AlertDesk } from './components/AlertDesk';
import { ProvenanceInspector } from './components/ProvenanceInspector';
import { Phase3ResearchStudio } from './components/Phase3ResearchStudio';
import {
  CloudLightning,
  Play,
  RotateCcw,
  Layers,
  ShieldCheck,
  AlertTriangle,
  FileCheck2,
  Cpu,
  RefreshCw,
  Sliders,
  CheckCircle2,
  Sparkles,
  Award,
  Info
} from 'lucide-react';

export const App: React.FC = () => {
  const [leadTimes, setLeadTimes] = useState<number[]>([
    0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120
  ]);
  const [currentLead, setCurrentLead] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playSpeed, setPlaySpeed] = useState<number>(1);
  const [activeTab, setActiveTab] = useState<'map' | 'verification' | 'alerts' | 'provenance' | 'research'>('map');

  // Active Map Layer
  const [activeLayer, setActiveLayer] = useState<'precipitation' | 'efi' | 'mslp' | 'wind'>('precipitation');
  const [showGroundTruth, setShowGroundTruth] = useState<boolean>(false);
  const [showUncertaintyCone, setShowUncertaintyCone] = useState<boolean>(true);
  const [showSpaghetti, setShowSpaghetti] = useState<boolean>(true);

  // State telemetry
  const [currentRunId, setCurrentRunId] = useState<string>('run_demo_monsoon_depression');
  const [events, setEvents] = useState<WeatherEvent[]>([]);
  const [selectedEvent, setSelectedEvent] = useState<WeatherEvent | null>(null);
  const [fieldData, setFieldData] = useState<ForecastFieldData | null>(null);
  const [efiData, setEfiData] = useState<EFIData | null>(null);
  const [verification, setVerification] = useState<VerificationReport | null>(null);
  const [alerts, setAlerts] = useState<AlertRecord[]>([]);
  const [provenance, setProvenance] = useState<ProvenanceRecord | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isRunningPipeline, setIsRunningPipeline] = useState<boolean>(false);

  // Initial Data Load
  useEffect(() => {
    loadRunData(currentRunId);
  }, [currentRunId]);

  // Lead Time Field Update
  useEffect(() => {
    if (!currentRunId) return;
    loadFieldAtLead(currentRunId, currentLead, activeLayer);
  }, [currentRunId, currentLead, activeLayer]);

  // Auto-play interval
  useEffect(() => {
    let interval: any = null;
    if (isPlaying) {
      interval = setInterval(() => {
        setCurrentLead((prev) => {
          const idx = leadTimes.indexOf(prev);
          if (idx >= leadTimes.length - 1) {
            return leadTimes[0];
          }
          return leadTimes[idx + 1];
        });
      }, 1200 / playSpeed);
    }
    return () => clearInterval(interval);
  }, [isPlaying, playSpeed, leadTimes]);

  const loadRunData = async (runId: string) => {
    setIsLoading(true);
    try {
      // 1. Fetch Events
      const evts = await api.getEvents(runId);
      setEvents(evts);
      if (evts.length > 0) {
        setSelectedEvent(evts[0]);
      }

      // 2. Fetch Verification Report
      try {
        const ver = await api.getVerification(runId);
        setVerification(ver);
      } catch (e) {
        console.warn('Verification not ready:', e);
      }

      // 3. Fetch Alerts
      try {
        const alts = await api.getAlerts(runId);
        setAlerts(alts);
      } catch (e) {
        console.warn('Alerts not ready:', e);
      }

      // 4. Fetch Provenance
      try {
        const prov = await api.getProvenance(runId);
        setProvenance(prov);
      } catch (e) {
        console.warn('Provenance not ready:', e);
      }

      // 5. Load Initial Field at Lead 0
      await loadFieldAtLead(runId, currentLead, activeLayer);
    } catch (err) {
      console.error('Failed to load weather run telemetry:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const loadFieldAtLead = async (runId: string, lead: number, layer: string) => {
    try {
      const varName = layer === 'wind' ? 'u_wind_850' : layer === 'precipitation' ? 'precipitation' : 'mslp';
      const fData = await api.getField(runId, lead, varName);
      setFieldData(fData);

      if (layer === 'efi') {
        const eData = await api.getEFI(runId, lead, 'precipitation');
        setEfiData(eData);
      }
    } catch (e) {
      console.warn('Field load notice:', e);
    }
  };

  const handleRunNewScenario = async (scenarioType: string) => {
    setIsRunningPipeline(true);
    const newRunId = `run_${scenarioType}_${Date.now().toString().slice(-4)}`;
    try {
      await api.runPipeline({
        run_id: newRunId,
        scenario_type: scenarioType,
        seed: Math.floor(Math.random() * 10000)
      });
      setCurrentRunId(newRunId);
    } catch (err) {
      console.error('Failed to execute pipeline run:', err);
    } finally {
      setIsRunningPipeline(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#070d19] text-slate-100 flex flex-col selection:bg-cyan-500 selection:text-white">
      {/* Top Navigation Bar */}
      <header className="border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md sticky top-0 z-40 px-6 py-3 flex items-center justify-between shadow-2xl">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 text-slate-950 shadow-lg shadow-cyan-500/20">
            <CloudLightning className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="font-extrabold text-base tracking-tight bg-gradient-to-r from-white via-slate-100 to-cyan-300 bg-clip-text text-transparent">
                SIH-26078 METEO-INTELLIGENCE
              </h1>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-cyan-950 text-cyan-400 border border-cyan-800">
                PHASE 1 BACKUP VERIFIED
              </span>
            </div>
            <p className="text-[11px] text-slate-400 font-mono">
              AI-Driven Spatio-Temporal Tracking of Extreme Weather Anomalies
            </p>
          </div>
        </div>

        {/* Global Action & Scenario Selectors */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center bg-slate-900/90 rounded-lg p-1 border border-slate-800 text-xs font-mono">
            <button
              onClick={() => handleRunNewScenario('monsoon_depression')}
              disabled={isRunningPipeline}
              className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-cyan-300 transition"
            >
              Monsoon Depression
            </button>
            <button
              onClick={() => handleRunNewScenario('coastal_cyclone')}
              disabled={isRunningPipeline}
              className="px-2.5 py-1 rounded hover:bg-slate-800 text-slate-300 transition"
            >
              Coastal Cyclone
            </button>
            <button
              onClick={() => handleRunNewScenario('convective_cluster')}
              disabled={isRunningPipeline}
              className="px-2.5 py-1 rounded hover:bg-slate-800 text-slate-300 transition"
            >
              Convective Storm
            </button>
          </div>

          <button
            onClick={() => loadRunData(currentRunId)}
            disabled={isLoading || isRunningPipeline}
            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition"
            title="Refresh Telemetry"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading || isRunningPipeline ? 'animate-spin text-cyan-400' : ''}`} />
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 p-6 space-y-6 max-w-[1700px] w-full mx-auto">
        {/* Navigation Tabs */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center space-x-2 font-mono text-xs">
            <button
              onClick={() => setActiveTab('map')}
              className={`px-4 py-2 rounded-lg font-bold flex items-center space-x-2 transition ${
                activeTab === 'map'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 shadow-lg shadow-cyan-500/10'
                  : 'bg-slate-900/60 text-slate-400 hover:bg-slate-800 hover:text-slate-200 border border-slate-800'
              }`}
            >
              <Layers className="w-4 h-4" />
              <span>4D Geospatial Radar & Trajectory</span>
            </button>

            <button
              onClick={() => setActiveTab('verification')}
              className={`px-4 py-2 rounded-lg font-bold flex items-center space-x-2 transition ${
                activeTab === 'verification'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 shadow-lg shadow-cyan-500/10'
                  : 'bg-slate-900/60 text-slate-400 hover:bg-slate-800 hover:text-slate-200 border border-slate-800'
              }`}
            >
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>Quantitative Verification Hub</span>
              {verification && (
                <span className="px-1.5 py-0.2 rounded bg-emerald-950 text-emerald-300 text-[10px] border border-emerald-800">
                  {(verification.contingency.f1_score * 100).toFixed(0)}% F1
                </span>
              )}
            </button>

            <button
              onClick={() => setActiveTab('alerts')}
              className={`px-4 py-2 rounded-lg font-bold flex items-center space-x-2 transition ${
                activeTab === 'alerts'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 shadow-lg shadow-cyan-500/10'
                  : 'bg-slate-900/60 text-slate-400 hover:bg-slate-800 hover:text-slate-200 border border-slate-800'
              }`}
            >
              <ShieldCheck className="w-4 h-4 text-rose-400" />
              <span>Disaster Alert Desk</span>
              {alerts.length > 0 && (
                <span className="px-1.5 py-0.2 rounded bg-rose-950 text-rose-300 text-[10px] border border-rose-800">
                  {alerts.length} Active
                </span>
              )}
            </button>

            <button
              onClick={() => setActiveTab('provenance')}
              className={`px-4 py-2 rounded-lg font-bold flex items-center space-x-2 transition ${
                activeTab === 'provenance'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 shadow-lg shadow-cyan-500/10'
                  : 'bg-slate-900/60 text-slate-400 hover:bg-slate-800 hover:text-slate-200 border border-slate-800'
              }`}
            >
              <Cpu className="w-4 h-4 text-amber-400" />
              <span>Provenance & Audit</span>
            </button>

            <button
              onClick={() => setActiveTab('research')}
              className={`px-4 py-2 rounded-lg font-bold flex items-center space-x-2 transition ${
                activeTab === 'research'
                  ? 'bg-purple-500/20 text-purple-300 border border-purple-500/50 shadow-lg shadow-purple-500/10'
                  : 'bg-slate-900/60 text-slate-400 hover:bg-slate-800 hover:text-slate-200 border border-slate-800'
              }`}
            >
              <Award className="w-4 h-4 text-purple-400" />
              <span>Phase 3 Research & Benchmark</span>
            </button>
          </div>

          {/* Active Layer Toggles */}
          {activeTab === 'map' && (
            <div className="flex items-center space-x-2 bg-slate-900/80 p-1 rounded-lg border border-slate-800 text-xs font-mono">
              <span className="text-[10px] text-slate-400 uppercase px-2">Layer:</span>
              <button
                onClick={() => setActiveLayer('precipitation')}
                className={`px-2.5 py-1 rounded transition ${
                  activeLayer === 'precipitation' ? 'bg-cyan-500 text-slate-950 font-bold' : 'text-slate-300 hover:bg-slate-800'
                }`}
              >
                Precipitation
              </button>
              <button
                onClick={() => setActiveLayer('efi')}
                className={`px-2.5 py-1 rounded transition ${
                  activeLayer === 'efi' ? 'bg-cyan-500 text-slate-950 font-bold' : 'text-slate-300 hover:bg-slate-800'
                }`}
              >
                EFI Heatmap
              </button>
              <button
                onClick={() => setActiveLayer('mslp')}
                className={`px-2.5 py-1 rounded transition ${
                  activeLayer === 'mslp' ? 'bg-cyan-500 text-slate-950 font-bold' : 'text-slate-300 hover:bg-slate-800'
                }`}
              >
                MSLP Pressure
              </button>
            </div>
          )}
        </div>

        {/* Tab 1: Live Interactive Geospatial Radar Map */}
        {activeTab === 'map' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Left 2 Cols: Main Canvas Map */}
              <div className="lg:col-span-2 space-y-4">
                <WeatherMap
                  currentLeadTime={currentLead}
                  activeEvent={selectedEvent}
                  fieldData={fieldData}
                  efiData={efiData}
                  activeLayer={activeLayer}
                  showGroundTruth={showGroundTruth}
                  showUncertaintyCone={showUncertaintyCone}
                  showSpaghetti={showSpaghetti}
                />

                {/* Timeline Player */}
                <TimelineSlider
                  leadTimes={leadTimes}
                  currentLead={currentLead}
                  onLeadChange={setCurrentLead}
                  isPlaying={isPlaying}
                  onTogglePlay={() => setIsPlaying(!isPlaying)}
                  speed={playSpeed}
                  onSpeedChange={setPlaySpeed}
                />
              </div>

              {/* Right Col: Active Event Telemetry & Trajectory Cone */}
              <div className="space-y-6">
                <EventDetailPanel event={selectedEvent} currentLeadTime={currentLead} />
                <TrajectoryConeViewer uncertainty={selectedEvent?.uncertainty_cone} currentLead={currentLead} />
              </div>
            </div>
          </div>
        )}

        {/* Tab 2: Ground Truth Quantitative Verification */}
        {activeTab === 'verification' && (
          <VerificationDashboard report={verification} />
        )}

        {/* Tab 3: Disaster Early Warning Alert Desk */}
        {activeTab === 'alerts' && (
          <AlertDesk alerts={alerts} />
        )}

        {/* Tab 4: Provenance & Cryptographic Audit */}
        {activeTab === 'provenance' && (
          <ProvenanceInspector provenance={provenance} />
        )}

        {/* Tab 5: Phase 3 Research Studio & Cross-Model Benchmarks */}
        {activeTab === 'research' && (
          <Phase3ResearchStudio runId={currentRunId} currentLeadTime={currentLead} />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950/90 py-3 px-6 text-center text-xs text-slate-500 font-mono flex items-center justify-between">
        <span>SIH-26078 Extreme Weather Intelligence System | 50-Hour Hardened Architecture</span>
        <span>Operational Backup: Phase 1 Validated | Zero Hardcoded Mock Data</span>
      </footer>
    </div>
  );
};
