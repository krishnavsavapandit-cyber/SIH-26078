import React, { useEffect, useRef, useState } from 'react';
import { WeatherEvent, TrajectoryPoint, ForecastFieldData, EFIData } from '../services/api';
import { Eye, Layers, Wind, Droplets, Gauge, AlertCircle } from 'lucide-react';

interface WeatherMapProps {
  currentLeadTime: number;
  activeEvent: WeatherEvent | null;
  fieldData: ForecastFieldData | null;
  efiData: EFIData | null;
  activeLayer: 'precipitation' | 'efi' | 'mslp' | 'wind';
  showGroundTruth: boolean;
  showUncertaintyCone: boolean;
  showSpaghetti: boolean;
}

export const WeatherMap: React.FC<WeatherMapProps> = ({
  currentLeadTime,
  activeEvent,
  fieldData,
  efiData,
  activeLayer,
  showGroundTruth,
  showUncertaintyCone,
  showSpaghetti,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hoverInfo, setHoverInfo] = useState<{ x: number; y: number; lat: number; lon: number; val: number } | null>(null);

  // Canvas coordinate transformations for India domain: Lat 6 to 38, Lon 68 to 98
  const latMin = 6.0, latMax = 38.0;
  const lonMin = 68.0, lonMax = 98.0;

  const toCanvasX = (lon: number, width: number) => ((lon - lonMin) / (lonMax - lonMin)) * width;
  const toCanvasY = (lat: number, height: number) => height - ((lat - latMin) / (latMax - latMin)) * height;
  const toGeoLon = (x: number, width: number) => lonMin + (x / width) * (lonMax - lonMin);
  const toGeoLat = (y: number, height: number) => latMin + ((height - y) / height) * (latMax - latMin);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // 1. Dark Ocean & Land Base Background
    ctx.fillStyle = '#070d19';
    ctx.fillRect(0, 0, width, height);

    // Grid lines (lat/lon 5 deg grid)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let lat = 10; lat <= 35; lat += 5) {
      const y = toCanvasY(lat, height);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
      ctx.fillStyle = 'rgba(255, 255, 255, 0.25)';
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.fillText(`${lat}°N`, 6, y - 4);
    }
    for (let lon = 70; lon <= 95; lon += 5) {
      const x = toCanvasX(lon, width);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
      ctx.fillStyle = 'rgba(255, 255, 255, 0.25)';
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.fillText(`${lon}°E`, x + 4, height - 8);
    }

    // India Outline & Coastline approximation
    drawSubcontinentCoastline(ctx, width, height);

    // 2. Render Meteorological Field Raster Layer
    if (activeLayer === 'efi' && efiData && efiData.efi_grid) {
      renderGridLayer(ctx, efiData.latitudes, efiData.longitudes, efiData.efi_grid, width, height, 'efi');
    } else if (fieldData && fieldData.mean_grid) {
      renderGridLayer(ctx, fieldData.latitudes, fieldData.longitudes, fieldData.mean_grid, width, height, activeLayer);
    }

    // 3. Render Uncertainty Cone
    if (showUncertaintyCone && activeEvent?.uncertainty_cone) {
      renderUncertaintyCone(ctx, activeEvent.uncertainty_cone, width, height, currentLeadTime);
    }

    // 4. Render Spaghetti Tracks
    if (showSpaghetti && activeEvent?.uncertainty_cone?.spaghetti_tracks) {
      renderSpaghettiTracks(ctx, activeEvent.uncertainty_cone.spaghetti_tracks, width, height);
    }

    // 5. Render Trajectory Track & Active Detection
    if (activeEvent && activeEvent.trajectory_points) {
      renderTrajectory(ctx, activeEvent.trajectory_points, width, height, currentLeadTime);
    }

  }, [fieldData, efiData, activeEvent, currentLeadTime, activeLayer, showGroundTruth, showUncertaintyCone, showSpaghetti]);

  const drawSubcontinentCoastline = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    // Schematic geographic coastline of India & Sri Lanka
    const coastPoints: [number, number][] = [
      [8.1, 77.5], [10.0, 75.8], [13.0, 74.8], [15.5, 73.8], [19.0, 72.8],
      [22.8, 69.5], [24.0, 68.8], [28.0, 70.0], [32.0, 74.5], [35.5, 77.5],
      [34.0, 80.0], [30.0, 81.0], [28.0, 84.5], [27.0, 88.5], [26.0, 92.5],
      [27.5, 96.0], [24.0, 94.0], [22.0, 91.8], [21.5, 87.0], [19.8, 85.8],
      [17.7, 83.3], [15.9, 80.5], [13.1, 80.3], [10.8, 79.8], [8.1, 77.5]
    ];

    ctx.strokeStyle = 'rgba(0, 210, 255, 0.4)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    coastPoints.forEach(([lat, lon], idx) => {
      const x = toCanvasX(lon, width);
      const y = toCanvasY(lat, height);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Fill land with subtle tint
    ctx.fillStyle = 'rgba(12, 21, 39, 0.4)';
    ctx.fill();
  };

  const renderGridLayer = (
    ctx: CanvasRenderingContext2D,
    lats: number[],
    lons: number[],
    grid: number[][],
    width: number,
    height: number,
    layerType: string
  ) => {
    const dLat = (lats[1] - lats[0]) || 0.25;
    const dLon = (lons[1] - lons[0]) || 0.25;
    const cellW = ((dLon / (lonMax - lonMin)) * width) + 1.2;
    const cellH = ((dLat / (latMax - latMin)) * height) + 1.2;

    for (let i = 0; i < lats.length; i++) {
      for (let j = 0; j < lons.length; j++) {
        const val = grid[i][j];
        if (layerType === 'precipitation' && val < 2.0) continue;
        if (layerType === 'efi' && val < 0.15) continue;

        ctx.fillStyle = getColorForValue(val, layerType);
        const x = toCanvasX(lons[j] - dLon / 2, width);
        const y = toCanvasY(lats[i] + dLat / 2, height);
        ctx.fillRect(x, y, cellW, cellH);
      }
    }
  };

  const getColorForValue = (val: number, layer: string): string => {
    if (layer === 'precipitation') {
      if (val >= 120) return 'rgba(255, 0, 128, 0.85)'; // Extreme violet
      if (val >= 70)  return 'rgba(255, 51, 51, 0.80)';  // Red
      if (val >= 40)  return 'rgba(255, 179, 0, 0.75)'; // Amber
      if (val >= 20)  return 'rgba(0, 230, 118, 0.65)'; // Green
      if (val >= 10)  return 'rgba(0, 210, 255, 0.55)'; // Cyan
      return 'rgba(0, 140, 255, 0.35)';                 // Blue
    } else if (layer === 'efi') {
      if (val >= 0.8) return 'rgba(255, 0, 90, 0.85)';
      if (val >= 0.6) return 'rgba(255, 120, 0, 0.75)';
      if (val >= 0.4) return 'rgba(255, 215, 0, 0.60)';
      return 'rgba(0, 200, 255, 0.35)';
    } else if (layer === 'mslp') {
      // Deep lows: < 996 hPa
      if (val <= 994) return 'rgba(255, 20, 100, 0.75)';
      if (val <= 998) return 'rgba(255, 120, 0, 0.65)';
      if (val <= 1004) return 'rgba(0, 210, 255, 0.40)';
      return 'rgba(100, 150, 255, 0.15)';
    }
    return 'rgba(0, 210, 255, 0.3)';
  };

  const renderUncertaintyCone = (
    ctx: CanvasRenderingContext2D,
    cone: any,
    width: number,
    height: number,
    currentLead: number
  ) => {
    const points = cone.cone_points || [];
    if (points.length < 2) return;

    // Draw shaded polygon for current lead cone
    const curCone = points.find((p: any) => p.lead_time === currentLead);
    if (curCone && curCone.cone_polygon) {
      ctx.fillStyle = 'rgba(0, 210, 255, 0.12)';
      ctx.strokeStyle = 'rgba(0, 210, 255, 0.5)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);

      ctx.beginPath();
      curCone.cone_polygon.forEach(([lat, lon]: [number, number], idx: number) => {
        const x = toCanvasX(lon, width);
        const y = toCanvasY(lat, height);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.setLineDash([]);
    }
  };

  const renderSpaghettiTracks = (
    ctx: CanvasRenderingContext2D,
    spaghetti: any[],
    width: number,
    height: number
  ) => {
    ctx.lineWidth = 1;
    spaghetti.forEach((member, mIdx) => {
      ctx.strokeStyle = `hsla(${170 + mIdx * 15}, 85%, 65%, 0.35)`;
      ctx.beginPath();
      member.points.forEach((pt: any, idx: number) => {
        const x = toCanvasX(pt.lon, width);
        const y = toCanvasY(pt.lat, height);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });
  };

  const renderTrajectory = (
    ctx: CanvasRenderingContext2D,
    pts: TrajectoryPoint[],
    width: number,
    height: number,
    currentLead: number
  ) => {
    // 1. Draw Full Trajectory Line
    ctx.strokeStyle = '#00d2ff';
    ctx.lineWidth = 3;
    ctx.beginPath();
    pts.forEach((pt, idx) => {
      const x = toCanvasX(pt.lon, width);
      const y = toCanvasY(pt.lat, height);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // 2. Draw Past Points
    pts.forEach((pt) => {
      const x = toCanvasX(pt.lon, width);
      const y = toCanvasY(pt.lat, height);
      const isCurrent = pt.lead_time === currentLead;

      if (!isCurrent) {
        ctx.fillStyle = '#0c1527';
        ctx.strokeStyle = '#00d2ff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    });

    // 3. Highlight Current Lead Time System Centroid with Radar Pulse
    const currentPt = pts.find((p) => p.lead_time === currentLead) || pts[pts.length - 1];
    if (currentPt) {
      const cx = toCanvasX(currentPt.lon, width);
      const cy = toCanvasY(currentPt.lat, height);

      // Radar pulse ring
      ctx.strokeStyle = '#ff3366';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(cx, cy, 18, 0, Math.PI * 2);
      ctx.stroke();

      ctx.fillStyle = '#ff3366';
      ctx.beginPath();
      ctx.arc(cx, cy, 6, 0, Math.PI * 2);
      ctx.fill();

      // Heading and intensity label
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 11px JetBrains Mono, monospace';
      ctx.fillText(`${currentPt.peak_precip_mm.toFixed(0)} mm/6h | ${currentPt.min_mslp_hpa.toFixed(0)} hPa`, cx + 12, cy - 8);
      ctx.fillStyle = '#00d2ff';
      ctx.font = '10px Inter, sans-serif';
      ctx.fillText(`Lead: ${currentPt.lead_time}h | ${currentPt.lifecycle_state}`, cx + 12, cy + 8);
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const lat = toGeoLat(y, canvas.height);
    const lon = toGeoLon(x, canvas.width);
    setHoverInfo({ x, y, lat, lon, val: 0 });
  };

  return (
    <div className="relative w-full h-[540px] rounded-xl overflow-hidden glass-panel border border-slate-700/60 shadow-2xl">
      <canvas
        ref={canvasRef}
        width={960}
        height={540}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoverInfo(null)}
        className="w-full h-full object-cover cursor-crosshair"
      />

      {/* Top Map Layer Indicator & Badge */}
      <div className="absolute top-4 left-4 flex items-center space-x-2 bg-slate-900/85 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-700/80 text-xs text-slate-300">
        <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>
        <span className="font-semibold text-white uppercase tracking-wider">{activeLayer}</span>
        <span className="text-slate-500">|</span>
        <span>Lead: <strong className="text-cyan-400">{currentLeadTime}h</strong></span>
      </div>

      {/* Hover Coordinates Telemetry */}
      {hoverInfo && (
        <div
          className="absolute pointer-events-none bg-slate-950/90 text-cyan-300 px-2.5 py-1 rounded text-xs font-mono border border-cyan-500/40 shadow-lg"
          style={{ left: Math.min(hoverInfo.x + 15, 800), top: Math.min(hoverInfo.y + 15, 480) }}
        >
          {hoverInfo.lat.toFixed(2)}°N, {hoverInfo.lon.toFixed(2)}°E
        </div>
      )}

      {/* Map Legend */}
      <div className="absolute bottom-4 right-4 bg-slate-900/90 backdrop-blur-md p-3 rounded-lg border border-slate-700 text-xs space-y-1.5 shadow-xl">
        <div className="font-semibold text-slate-200 flex items-center space-x-1.5">
          <Layers className="w-3.5 h-3.5 text-cyan-400" />
          <span>Scale ({activeLayer})</span>
        </div>
        {activeLayer === 'precipitation' ? (
          <div className="flex items-center space-x-1 font-mono text-[10px]">
            <span className="px-1.5 py-0.5 rounded bg-blue-600/60">10mm</span>
            <span className="px-1.5 py-0.5 rounded bg-emerald-500/70">20mm</span>
            <span className="px-1.5 py-0.5 rounded bg-amber-500/80">40mm</span>
            <span className="px-1.5 py-0.5 rounded bg-red-600/90">70mm</span>
            <span className="px-1.5 py-0.5 rounded bg-fuchsia-600 font-bold text-white">120+</span>
          </div>
        ) : (
          <div className="flex items-center space-x-1 font-mono text-[10px]">
            <span className="px-1.5 py-0.5 rounded bg-cyan-600/40">0.2</span>
            <span className="px-1.5 py-0.5 rounded bg-amber-500/60">0.5</span>
            <span className="px-1.5 py-0.5 rounded bg-orange-600/80">0.7</span>
            <span className="px-1.5 py-0.5 rounded bg-rose-600 font-bold text-white">0.9+</span>
          </div>
        )}
      </div>
    </div>
  );
};
