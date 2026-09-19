import React, { useState, useEffect } from 'react';
import {
  api,
  WeatherEvent,
  ForecastFieldData,
  EFIData,
  VerificationReport,
  AlertRecord,
  ProvenanceRecord,
  DataModeStatus,
  DiscoveredDataset,
  DataSourceMode
} from './services/api';
import { WeatherMap } from './components/WeatherMap';
import { TimelineSlider } from './components/TimelineSlider';
import { EventDetailPanel } from './components/EventDetailPanel';
import { SystemWorkflowBar } from './components/SystemWorkflowBar';
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
  Info,
  Rocket,
  Database,
  Compass,
  Droplets,
  Gauge,
  Activity,
  X,
  FileText,
  Radio,
  Satellite,
  HardDrive
} from 'lucide-react';

export const App: React.FC = () => {
  const [leadTimes, setLeadTimes] = useState<number[]>([
    0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120
  ]);
  const [currentLead, setCurrentLead] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playSpeed, setPlaySpeed] = useState<number>(1);
  const [activeTab, setActiveTab] = useState<'map' | 'research' | 'verification' | 'alerts' | 'provenance'>('map');

  // Active Map Layer & Resolution
  const [activeLayer, setActiveLayer] = useState<'precipitation' | 'efi' | 'mslp' | 'wind'>('precipitation');
  const [resolutionMode, setResolutionMode] = useState<'12km' | '5km'>('5km');
  const [showGroundTruth, setShowGroundTruth] = useState<boolean>(false);
  const [showUncertaintyCone, setShowUncertaintyCone] = useState<boolean>(true);
  const [showSpaghetti, setShowSpaghetti] = useState<boolean>(true);
  const [isEventPanelCollapsed, setIsEventPanelCollapsed] = useState<boolean>(false);

  // State telemetry
  const [availableRuns, setAvailableRuns] = useState<any[]>([]);
  const [currentRunId, setCurrentRunId] = useState<string>('');
  const [events, setEvents] = useState<WeatherEvent[]>([]);
  const [selectedEvent, setSelectedEvent] = useState<WeatherEvent | null>(null);
  const [fieldData, setFieldData] = useState<ForecastFieldData | null>(null);
  const [efiData, setEfiData] = useState<EFIData | null>(null);
  const [downscalingData, setDownscalingData] = useState<any | null>(null);
  const [verification, setVerification] = useState<VerificationReport | null>(null);
  const [alerts, setAlerts] = useState<AlertRecord[]>([]);
  const [provenance, setProvenance] = useState<ProvenanceRecord | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isRunningPipeline, setIsRunningPipeline] = useState<boolean>(false);

  // Real Data & Discovery State
  const [dataModeStatus, setDataModeStatus] = useState<DataModeStatus | null>(null);
  const [discoveredDatasets, setDiscoveredDatasets] = useState<DiscoveredDataset[]>([]);

  // Flagship Demo & Data Source Modals
  const [flagshipModalOpen, setFlagshipModalOpen] = useState<boolean>(false);
  const [flagshipStages, setFlagshipStages] = useState<string[]>([]);
  const [flagshipStatus, setFlagshipStatus] = useState<'IDLE' | 'RUNNING' | 'COMPLETED' | 'FAILED'>('IDLE');
  const [adapterInfoOpen, setAdapterInfoOpen] = useState<boolean>(false);

  // Initial Runs & Data Load
  useEffect(() => {
    const initApp = async () => {
      try {
        const runs = await api.getRuns();
        if (Array.isArray(runs) && runs.length > 0) {
          setAvailableRuns(runs);
          setCurrentRunId(runs[0].run_id);
        } else {
          setCurrentRunId('run_flagship_monsoon');
        }
      } catch (err) {
        console.warn('Failed to fetch initial runs list:', err);
        setCurrentRunId('run_flagship_monsoon');
      }
    };
    initApp();
  }, []);

  // When currentRunId changes, load run telemetry
  useEffect(() => {
    if (currentRunId) {
      loadRunData(currentRunId);
    }
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
    if (!runId) return;
    setIsLoading(true);
    try {
      // 0. Fetch Mode Status & Discovered Datasets
      try {
        const modeStat = await api.getDataModeStatus();
        setDataModeStatus(modeStat);
        const disc = await api.discoverDatasets();
        setDiscoveredDatasets(disc);
      } catch (e) {
        console.warn('Mode status discovery notice:', e);
      }

      // 1. Fetch Events
      try {
        const evts = await api.getEvents(runId);
        setEvents(Array.isArray(evts) ? evts : []);
        if (Array.isArray(evts) && evts.length > 0) {
          setSelectedEvent(evts[0]);
        } else {
          setSelectedEvent(null);
        }
      } catch (e) {
        console.warn('Events notice:', e);
        setEvents([]);
        setSelectedEvent(null);
      }

      // 2. Fetch Verification Report
      try {
        const ver = await api.getVerification(runId);
        setVerification(ver);
      } catch (e) {
        console.warn('Verification notice:', e);
        setVerification(null);
      }

      // 3. Fetch Alerts
      try {
        const alts = await api.getAlerts(runId);
        setAlerts(Array.isArray(alts) ? alts : []);
      } catch (e) {
        console.warn('Alerts notice:', e);
        setAlerts([]);
      }

      // 4. Fetch Provenance
      try {
        const prov = await api.getProvenance(runId);
        setProvenance(prov);
      } catch (e) {
        console.warn('Provenance notice:', e);
        setProvenance(null);
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

  const handleRunFlagshipDemo = async (
    scenarioType: string = 'monsoon_depression',
    injectFailure: boolean = false,
    externalPath?: string
  ) => {
    setFlagshipModalOpen(true);
    setFlagshipStatus('RUNNING');
    setFlagshipStages([
      externalPath
        ? `1. Ingesting Real NWP Dataset (${externalPath.split(/[\/\\]/).pop()})`
        : '1. Initializing 4D Multivariable Forecast Array (Leads: 0-120h, 10 Members)',
      '2. Validating Strict Real-Data Compliance & Atmospheric Field Integrity',
      '3. Computing Multi-Variable EFI and Shift-of-Tails (SOT) Anomaly Fields',
      '4. Segmenting Dynamic Threat Footprints via Spherical ST-GNN',
      '5. Synthesizing Multi-Hypothesis Tracking (MHT) Probabilistic Trees & Cones',
      '6. Performing Physics-Informed 12km → 5km Super-Resolution Downscaling',
      '7. Evaluating Extreme Amplitude Preservation Scorecard (P95/P99/Max)',
      '8. Emitting Hyperlocal 5km Early Warning Advisories & Cryptographic Provenance'
    ]);

    const prefix = externalPath ? 'run_real_' : 'run_flagship_';
    const newRunId = `${prefix}${scenarioType}_${Date.now().toString().slice(-4)}`;
    try {
      await api.runFlagshipPipeline({
        run_id: newRunId,
        scenario_type: scenarioType,
        seed: 42,
        enable_phase3: true,
        inject_failure: injectFailure,
        data_source_mode: externalPath ? 'REAL' : 'SYNTHETIC',
        external_data_path: externalPath
      });
      setFlagshipStatus('COMPLETED');
      setCurrentRunId(newRunId);
      await loadRunData(newRunId);
    } catch (err) {
      console.error('Flagship pipeline execution failed:', err);
      setFlagshipStatus('FAILED');
    }
  };

  return (
    <div className="min-h-screen bg-[#060a14] text-slate-100 flex flex-col selection:bg-cyan-500 selection:text-white">
      {/* Top Header Bar */}
      <header className="border-b border-slate-800/80 bg-slate-950/90 backdrop-blur-md sticky top-0 z-40 px-6 py-2.5 flex items-center justify-between shadow-2xl">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 text-slate-950 shadow-lg shadow-cyan-500/20">
            <CloudLightning className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="font-extrabold text-sm tracking-tight bg-gradient-to-r from-white via-slate-100 to-cyan-300 bg-clip-text text-transparent">
                SIH-26078 METEO-INTELLIGENCE
              </h1>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-cyan-950 text-cyan-400 border border-cyan-800">
                SCIENTIFIC RESEARCH ENGINE
              </span>
            </div>
            <p className="text-[10px] text-slate-400 font-mono">
              AI-Driven Spatio-Temporal Tracking & Hyperlocal 5km Downscaling
            </p>
          </div>
        </div>

        {/* Action Controls & Data Source Indicator */}
        <div className="flex items-center space-x-2.5">
          {/* Live Data Source Mode Badge */}
          {provenance?.data_source_mode === 'REAL' || (!provenance && dataModeStatus?.active_mode === 'REAL') ? (
            <button
              onClick={() => setAdapterInfoOpen(true)}
              className="flex items-center space-x-2 px-2.5 py-1 rounded-lg bg-sky-950/80 border border-sky-500/60 hover:border-sky-400 text-xs font-mono text-sky-300 transition"
              title="Click to Inspect Cryptographic Real-Data Provenance & Datasets"
            >
              <span className="w-2 h-2 rounded-full bg-sky-400 animate-pulse" />
              <span className="font-bold">REAL DATA</span>
            </button>
          ) : (
            <button
              onClick={() => setAdapterInfoOpen(true)}
              className="flex items-center space-x-2 px-2.5 py-1 rounded-lg bg-emerald-950/80 border border-emerald-500/60 hover:border-emerald-400 text-xs font-mono text-emerald-300 transition"
              title="Click to Inspect Benchmark Generator & Available Real Datasets"
            >
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="font-bold">BENCHMARK MODE</span>
            </button>
          )}

          {/* Run Selector Dropdown */}
          {availableRuns.length > 0 && (
            <div className="flex items-center space-x-1.5 bg-slate-900/90 rounded-lg px-2 py-1 border border-slate-800 text-xs font-mono">
              <span className="text-[10px] text-slate-400">RUN:</span>
              <select
                value={currentRunId}
                onChange={(e) => setCurrentRunId(e.target.value)}
                className="bg-slate-950 text-cyan-300 text-xs rounded border border-slate-700 px-2 py-0.5 outline-none focus:border-cyan-500 max-w-[160px] truncate"
              >
                {availableRuns.map((r) => (
                  <option key={r.run_id} value={r.run_id}>
                    {r.run_id} ({r.data_source_mode === 'REAL' ? 'REAL' : 'SYN'})
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Benchmark Run Buttons */}
          <button
            onClick={() => handleRunFlagshipDemo('monsoon_depression')}
            disabled={isRunningPipeline || flagshipStatus === 'RUNNING'}
            className="flex items-center space-x-1.5 px-3 py-1 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold text-xs font-mono shadow-md shadow-cyan-500/20 transition active:scale-95"
          >
            <Rocket className="w-3.5 h-3.5" />
            <span>Run Pipeline</span>
          </button>

          <button
            onClick={() => loadRunData(currentRunId)}
            disabled={isLoading || isRunningPipeline}
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition"
            title="Refresh Telemetry"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin text-cyan-400' : ''}`} />
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 p-4 md:p-6 space-y-4 max-w-[1780px] w-full mx-auto">
        {/* Navigation Tabs */}
        <div className="flex items-center justify-between border-b border-slate-800/90 pb-2.5">
          <div className="flex items-center space-x-2 font-mono text-xs overflow-x-auto">
            <button
              onClick={() => setActiveTab('map')}
              className={`px-3.5 py-1.5 rounded-xl font-bold flex items-center space-x-2 transition ${
                activeTab === 'map'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 shadow-lg shadow-cyan-500/10'
                  : 'bg-slate-900/60 text-slate-400 hover:bg-slate-800 hover:text-slate-200 border border-slate-800'
              }`}
            >
              <Layers className="w-4 h-4 text-cyan-400" />
              <span>1. Mission Control & Weather Map</span>
            </button>

            <button
              onClick={() => setActiveTab('research')}
              className={`px-3.5 py-1.5 rounded-xl font-bold flex items-center space-x-2 transition ${
                activeTab === 'research'
                  ? 'bg-purple-500/20 text-purple-300 border border-purple-500/50 shadow-lg shadow-purple-500/10'
                  : 'bg-slate-900/60 text-slate-400 hover:bg-slate-800 hover:text-slate-200 border border-slate-800'
              }`}
            >
              <Award className="w-4 h-4 text-purple-400" />
              <span>2. Research Studio & Downscaling</span>
            </button>

            <button
              onClick={() => setActiveTab('verification')}
              className={`px-3.5 py-1.5 rounded-xl font-bold flex items-center space-x-2 transition ${
                activeTab === 'verification'
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/50 shadow-lg shadow-emerald-500/10'
                  : 'bg-slate-900/60 text-slate-400 hover:bg-slate-800 hover:text-slate-200 border border-slate-800'
              }`}
            >
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>3. Verification Hub</span>
              {verification?.contingency?.f1_score != null && (
                <span className="px-1.5 py-0.2 rounded bg-emerald-950 text-emerald-300 text-[10px] border border-emerald-800">
                  {(verification.contingency.f1_score * 100).toFixed(0)}% F1
                </span>
              )}
            </button>

            <button
              onClick={() => setActiveTab('alerts')}
              className={`px-3.5 py-1.5 rounded-xl font-bold flex items-center space-x-2 transition ${
                activeTab === 'alerts'
                  ? 'bg-rose-500/20 text-rose-300 border border-rose-500/50 shadow-lg shadow-rose-500/10'
                  : 'bg-slate-900/60 text-slate-400 hover:bg-slate-800 hover:text-slate-200 border border-slate-800'
              }`}
            >
              <ShieldCheck className="w-4 h-4 text-rose-400" />
              <span>4. Hyperlocal Alert Desk</span>
              {alerts.length > 0 && (
                <span className="px-1.5 py-0.2 rounded bg-rose-950 text-rose-300 text-[10px] border border-rose-800">
                  {alerts.length} Active
                </span>
              )}
            </button>

            <button
              onClick={() => setActiveTab('provenance')}
              className={`px-3.5 py-1.5 rounded-xl font-bold flex items-center space-x-2 transition ${
                activeTab === 'provenance'
                  ? 'bg-amber-500/20 text-amber-300 border border-amber-500/50 shadow-lg shadow-amber-500/10'
                  : 'bg-slate-900/60 text-slate-400 hover:bg-slate-800 hover:text-slate-200 border border-slate-800'
              }`}
            >
              <Cpu className="w-4 h-4 text-amber-400" />
              <span>5. Provenance & Cryptographic Audit</span>
            </button>
          </div>
        </div>

        {/* Tab 1: Live Interactive Research-Grade Weather Map (HERO AREA) */}
        {activeTab === 'map' && (
          <div className="space-y-4">
            {/* Judge Explainability Strip: DETECT -> TRACK -> REFINE -> ALERT */}
            <SystemWorkflowBar />

            {/* Hero Map & Compact Side Telemetry Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 items-start">
              {/* Main Interactive Map (Dominates 75% of visual layout) */}
              <div className="lg:col-span-3 space-y-3">
                <WeatherMap
                  currentLeadTime={currentLead}
                  activeEvent={selectedEvent}
                  fieldData={fieldData}
                  efiData={efiData}
                  downscalingData={downscalingData}
                  activeLayer={activeLayer}
                  onLayerChange={setActiveLayer}
                  showGroundTruth={showGroundTruth}
                  showUncertaintyCone={showUncertaintyCone}
                  showSpaghetti={showSpaghetti}
                  resolutionMode={resolutionMode}
                  onResolutionChange={setResolutionMode}
                />

                {/* Forecast Timeline Scrubber */}
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

              {/* Right Column: Compact Event Detail & Uncertainty Cone Panels */}
              <div className="space-y-3">
                <EventDetailPanel
                  event={selectedEvent}
                  currentLeadTime={currentLead}
                  isCollapsed={isEventPanelCollapsed}
                  onToggleCollapse={() => setIsEventPanelCollapsed(!isEventPanelCollapsed)}
                  showUncertaintyCone={showUncertaintyCone}
                  onToggleUncertaintyCone={() => setShowUncertaintyCone(!showUncertaintyCone)}
                  showSpaghetti={showSpaghetti}
                  onToggleSpaghetti={() => setShowSpaghetti(!showSpaghetti)}
                />

                <TrajectoryConeViewer uncertainty={selectedEvent?.uncertainty_cone} currentLead={currentLead} />
              </div>
            </div>
          </div>
        )}

        {/* Tab 2: Phase 3 Research Studio & Cross-Model Benchmarks */}
        {activeTab === 'research' && (
          <Phase3ResearchStudio runId={currentRunId} currentLeadTime={currentLead} />
        )}

        {/* Tab 3: Ground Truth Quantitative Verification */}
        {activeTab === 'verification' && (
          <VerificationDashboard report={verification} />
        )}

        {/* Tab 4: Disaster Early Warning Alert Desk */}
        {activeTab === 'alerts' && (
          <AlertDesk alerts={alerts} />
        )}

        {/* Tab 5: Provenance & Cryptographic Audit */}
        {activeTab === 'provenance' && (
          <ProvenanceInspector provenance={provenance} />
        )}
      </main>

      {/* Flagship Guided Run Progress Modal */}
      {flagshipModalOpen && (
        <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-2xl w-full p-6 shadow-2xl space-y-4 font-mono">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center space-x-2">
                <Rocket className="w-5 h-5 text-cyan-400" />
                <h3 className="font-bold text-sm text-slate-100 uppercase">
                  Flagship End-to-End Scientific Pipeline Execution
                </h3>
              </div>
              <button
                onClick={() => setFlagshipModalOpen(false)}
                className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-slate-200"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-2 max-h-[60vh] overflow-y-auto pr-2 text-xs">
              {flagshipStages.map((st, idx) => (
                <div key={idx} className="flex items-center space-x-3 p-2 rounded bg-slate-950/60 border border-slate-800/80">
                  <CheckCircle2
                    className={`w-4 h-4 flex-shrink-0 ${
                      flagshipStatus === 'COMPLETED' ? 'text-emerald-400' : 'text-cyan-400 animate-pulse'
                    }`}
                  />
                  <span className="text-slate-200">{st}</span>
                </div>
              ))}
            </div>

            <div className="flex justify-end pt-2 border-t border-slate-800">
              <button
                onClick={() => setFlagshipModalOpen(false)}
                className="px-4 py-1.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs"
              >
                {flagshipStatus === 'RUNNING' ? 'Running in Background...' : 'Close & View Telemetry'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
