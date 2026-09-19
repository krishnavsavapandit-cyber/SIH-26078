import React, { useEffect, useRef, useState, useMemo } from 'react';
import { WeatherEvent, TrajectoryPoint, ForecastFieldData, EFIData } from '../services/api';
import {
  Layers,
  Eye,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Sparkles,
  Compass,
  Droplets,
  Wind,
  Gauge,
  Activity,
  AlertTriangle,
  Info,
  Maximize2
} from 'lucide-react';

interface WeatherMapProps {
  currentLeadTime: number;
  activeEvent: WeatherEvent | null;
  fieldData: ForecastFieldData | null;
  efiData: EFIData | null;
  downscalingData?: any | null;
  activeLayer: 'precipitation' | 'efi' | 'mslp' | 'wind';
  onLayerChange?: (layer: 'precipitation' | 'efi' | 'mslp' | 'wind') => void;
  showGroundTruth: boolean;
  showUncertaintyCone: boolean;
  showSpaghetti: boolean;
  resolutionMode: '12km' | '5km';
  onResolutionChange: (mode: '12km' | '5km') => void;
  onSelectTrackPoint?: (pt: TrajectoryPoint) => void;
}

// Detailed geographical boundary polygons for India subcontinent & surrounding South Asia
const INDIA_MAIN_COASTLINE: [number, number][] = [
  // West Coast (South to North)
  [8.08, 77.55], // Kanyakumari
  [8.48, 76.95], // Trivandrum
  [9.93, 76.26], // Kochi
  [11.25, 75.78], // Kozhikode
  [12.91, 74.85], // Mangaluru
  [15.30, 73.80], // Goa
  [16.99, 73.30], // Ratnagiri
  [18.96, 72.82], // Mumbai
  [20.55, 72.88], // Daman
  [21.17, 72.83], // Surat
  [21.70, 72.15], // Bhavnagar
  [20.75, 70.98], // Diu
  [20.90, 70.36], // Somnath
  [21.63, 69.60], // Porbandar
  [22.24, 68.97], // Dwarka
  [22.75, 69.70], // Gulf of Kutch inner
  [23.25, 68.60], // Lakhpat / Kutch border
  // Pakistan / Northern Borders
  [24.50, 68.20],
  [26.50, 70.50], // Thar
  [28.00, 70.30],
  [30.50, 72.50], // Punjab border
  [32.50, 74.50], // Jammu
  [34.50, 74.00], // Kashmir
  [36.80, 74.80], // Karakoram northern apex
  [35.50, 77.50], // Ladakh / Siachen
  [34.50, 78.80], // Aksai Chin border
  [32.80, 79.20], // HP border
  [31.00, 79.50], // Uttarakhand
  [30.20, 81.00], // Lipulekh / Nepal trijunction
  // Nepal Border (Northern arc)
  [28.80, 80.20],
  [27.70, 83.50],
  [26.80, 88.10], // East Nepal
  // Sikkim & Bhutan & North East
  [27.70, 88.60], // Sikkim
  [27.30, 88.90],
  [26.80, 89.90], // Bhutan South
  [27.80, 91.80], // Tawang / Arunachal
  [28.80, 94.50], // Arunachal North
  [28.20, 96.80], // Kibithu eastern apex
  [26.50, 97.00], // Myanmar border North
  [24.50, 94.50], // Nagaland/Manipur
  [22.00, 93.00], // Mizoram
  [21.20, 92.20], // Chittagong border
  // Bangladesh Loop & Sundarbans
  [23.00, 89.00],
  [25.00, 89.80], // Meghalaya South
  [25.80, 91.80], // Shillong
  [26.20, 90.00], // Assam
  [24.00, 88.50], // Bengal North
  [21.60, 88.00], // Sundarbans / Hooghly
  // East Coast (North to South)
  [21.48, 86.92], // Balasore / Chandipur
  [20.30, 86.70], // Paradip
  [19.80, 85.82], // Puri
  [19.30, 84.85], // Gopalpur
  [17.68, 83.21], // Visakhapatnam
  [16.98, 82.24], // Kakinada
  [15.82, 80.35], // Ongole / Machilipatnam
  [14.44, 80.00], // Nellore
  [13.08, 80.27], // Chennai
  [11.93, 79.83], // Puducherry
  [10.76, 79.84], // Nagapattinam
  [9.28, 79.31], // Rameswaram / Palk Strait
  [8.76, 78.13], // Tuticorin
  [8.08, 77.55]  // Kanyakumari loop close
];

// Sri Lanka coastline
const SRI_LANKA_COASTLINE: [number, number][] = [
  [9.80, 80.20], [9.20, 80.80], [8.57, 81.23], [7.70, 81.70], [6.90, 81.85],
  [6.00, 80.80], [6.05, 80.20], [6.93, 79.85], [7.95, 79.80], [8.55, 79.90],
  [9.80, 80.20]
];

// Major Indian state / regional division key lines for research orientation
const INTERNAL_GRID_LINES: [number, number][][] = [
  [[21.25, 69.00], [21.50, 74.00], [21.00, 79.00], [22.00, 87.00]], // Tropic-adjacent / Deccan divide
  [[15.00, 73.80], [15.50, 80.00]], // Southern peninsula line
  [[28.00, 74.00], [26.00, 84.00], [25.00, 88.00]], // Gangetic valley axis
  [[19.00, 72.80], [17.50, 78.50], [17.68, 83.21]]  // Mumbai-Hyderabad-Vizag cross section
];

// Key strategic coastal & research radar observatories for domain context
const RESEARCH_STATIONS = [
  { name: 'IMD New Delhi (HQ)', lat: 28.58, lon: 77.22, type: 'HQ' },
  { name: 'DWR Chennai', lat: 13.08, lon: 80.27, type: 'RADAR' },
  { name: 'DWR Visakhapatnam', lat: 17.68, lon: 83.21, type: 'RADAR' },
  { name: 'DWR Paradip', lat: 20.30, lon: 86.70, type: 'RADAR' },
  { name: 'DWR Mumbai', lat: 18.96, lon: 72.82, type: 'RADAR' },
  { name: 'NCMRWF Noida', lat: 28.62, lon: 77.36, type: 'NWP' },
  { name: 'INCOIS Hyderabad', lat: 17.55, lon: 78.38, type: 'OCEAN' }
];

export const WeatherMap: React.FC<WeatherMapProps> = ({
  currentLeadTime,
  activeEvent,
  fieldData,
  efiData,
  downscalingData,
  activeLayer,
  onLayerChange,
  showGroundTruth,
  showUncertaintyCone,
  showSpaghetti,
  resolutionMode,
  onResolutionChange,
  onSelectTrackPoint
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Zoom & Pan transformation state
  const [zoom, setZoom] = useState<number>(1.0);
  const [panOffset, setPanOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [hoverInfo, setHoverInfo] = useState<{
    x: number;
    y: number;
    lat: number;
    lon: number;
    val: number | null;
    stationName?: string;
  } | null>(null);

  // Geographic boundaries: India South Asia domain [Lat: 6-38, Lon: 68-98]
  const latMin = 6.0, latMax = 38.0;
  const lonMin = 68.0, lonMax = 98.0;

  // Transform coordinates accounting for zoom and pan
  const toCanvasX = (lon: number, width: number) => {
    const base = ((lon - lonMin) / (lonMax - lonMin)) * width;
    return (base - width / 2) * zoom + width / 2 + panOffset.x;
  };

  const toCanvasY = (lat: number, height: number) => {
    const base = height - ((lat - latMin) / (latMax - latMin)) * height;
    return (base - height / 2) * zoom + height / 2 + panOffset.y;
  };

  const toGeoLon = (x: number, width: number) => {
    const unpanned = (x - width / 2 - panOffset.x) / zoom + width / 2;
    return lonMin + (unpanned / width) * (lonMax - lonMin);
  };

  const toGeoLat = (y: number, height: number) => {
    const unpanned = (y - height / 2 - panOffset.y) / zoom + height / 2;
    return latMin + ((height - unpanned) / height) * (latMax - latMin);
  };

  // Redraw Canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // 1. Deep Oceanic & Atmospheric Slate Base Background
    ctx.fillStyle = '#060a14';
    ctx.fillRect(0, 0, width, height);

    // Subtle bathymetry & ocean current grid lines
    drawAtmosphericGrid(ctx, width, height);

    // 2. Geographic Landmass, Coastlines & Boundary Subdivisions
    drawSubcontinentGeography(ctx, width, height);

    // 3. Meteorological Continuous Field Layer (Precipitation, EFI, MSLP, Wind)
    if (activeLayer === 'efi' && efiData?.efi_grid) {
      renderContinuousField(ctx, efiData.latitudes, efiData.longitudes, efiData.efi_grid, width, height, 'efi');
    } else if (fieldData?.mean_grid) {
      renderContinuousField(ctx, fieldData.latitudes, fieldData.longitudes, fieldData.mean_grid, width, height, activeLayer);
    }

    // 4. Research Observatories & Radars
    drawResearchObservatories(ctx, width, height);

    // 5. Extreme Anomaly Dynamic Footprint & 5km Hyperlocal Threat Zone
    if (activeEvent) {
      renderExtremeFootprint(ctx, activeEvent, width, height, currentLeadTime, resolutionMode);
    }

    // 6. Ensemble Uncertainty Cone of Probability
    if (showUncertaintyCone && activeEvent?.uncertainty_cone) {
      renderUncertaintyCone(ctx, activeEvent.uncertainty_cone, width, height, currentLeadTime);
    }

    // 7. Multi-Hypothesis Spaghetti Tracks (Ensemble Members)
    if (showSpaghetti && activeEvent?.uncertainty_cone?.spaghetti_tracks) {
      renderSpaghettiTracks(ctx, activeEvent.uncertainty_cone.spaghetti_tracks, width, height);
    }

    // 8. Authoritative Spatio-Temporal Trajectory (Past vs Forecast with T+ markers)
    if (activeEvent?.trajectory_points && activeEvent.trajectory_points.length > 0) {
      renderScientificTrajectory(ctx, activeEvent.trajectory_points, width, height, currentLeadTime);
    }

  }, [
    fieldData,
    efiData,
    activeEvent,
    currentLeadTime,
    activeLayer,
    showGroundTruth,
    showUncertaintyCone,
    showSpaghetti,
    resolutionMode,
    zoom,
    panOffset
  ]);

  // Atmospheric Coordinate Grid (Lat/Lon parallels with clean scientific notation)
  const drawAtmosphericGrid = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    ctx.strokeStyle = 'rgba(0, 210, 255, 0.06)';
    ctx.lineWidth = 1;

    for (let lat = 10; lat <= 35; lat += 5) {
      const y = toCanvasY(lat, height);
      if (y >= 0 && y <= height) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();

        ctx.fillStyle = 'rgba(148, 163, 184, 0.4)';
        ctx.font = '9px JetBrains Mono, monospace';
        ctx.fillText(`${lat}°N`, 8, y - 3);
      }
    }

    for (let lon = 70; lon <= 95; lon += 5) {
      const x = toCanvasX(lon, width);
      if (x >= 0 && x <= width) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();

        ctx.fillStyle = 'rgba(148, 163, 184, 0.4)';
        ctx.font = '9px JetBrains Mono, monospace';
        ctx.fillText(`${lon}°E`, x + 4, height - 8);
      }
    }
  };

  // Geographic Subcontinent Landmass & Internal Divisions
  const drawSubcontinentGeography = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    // 1. Land Polygon Fill
    ctx.beginPath();
    INDIA_MAIN_COASTLINE.forEach(([lat, lon], idx) => {
      const x = toCanvasX(lon, width);
      const y = toCanvasY(lat, height);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();

    // Deep slate land tint with subtle gradient
    ctx.fillStyle = 'rgba(15, 23, 42, 0.75)';
    ctx.fill();

    // High-definition coastline stroke
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.45)';
    ctx.lineWidth = 1.6;
    ctx.stroke();

    // 2. Sri Lanka Coastline
    ctx.beginPath();
    SRI_LANKA_COASTLINE.forEach(([lat, lon], idx) => {
      const x = toCanvasX(lon, width);
      const y = toCanvasY(lat, height);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fillStyle = 'rgba(15, 23, 42, 0.75)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.45)';
    ctx.lineWidth = 1.2;
    ctx.stroke();

    // 3. Sub-regional & state boundaries
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.15)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    INTERNAL_GRID_LINES.forEach((line) => {
      ctx.beginPath();
      line.forEach(([lat, lon], idx) => {
        const x = toCanvasX(lon, width);
        const y = toCanvasY(lat, height);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });
    ctx.setLineDash([]);
  };

  // Continuous smooth meteorological grid rendering with bilinear color gradients
  const renderContinuousField = (
    ctx: CanvasRenderingContext2D,
    lats: number[],
    lons: number[],
    grid: number[][],
    width: number,
    height: number,
    layerType: string
  ) => {
    if (!lats || !lons || !grid || grid.length === 0) return;

    const dLat = Math.abs(lats[1] - lats[0]) || 0.25;
    const dLon = Math.abs(lons[1] - lons[0]) || 0.25;

    // Sub-sample rendering loop for speed & smoothness
    const step = zoom > 1.8 ? 1 : 1;

    for (let i = 0; i < lats.length; i += step) {
      for (let j = 0; j < lons.length; j += step) {
        const val = grid[i][j];

        // Atmospheric thresholds to avoid visual clutter on dry regions
        if (layerType === 'precipitation' && val < 2.5) continue;
        if (layerType === 'efi' && val < 0.15) continue;
        if (layerType === 'wind' && val < 4.0) continue;

        const x = toCanvasX(lons[j] - dLon / 2, width);
        const y = toCanvasY(lats[i] + dLat / 2, height);
        const cellW = Math.abs(toCanvasX(lons[j] + dLon / 2, width) - x) + 1.2;
        const cellH = Math.abs(toCanvasY(lats[i] - dLat / 2, height) - y) + 1.2;

        if (x + cellW < 0 || x > width || y + cellH < 0 || y > height) continue;

        ctx.fillStyle = getScientificColor(val, layerType);
        ctx.fillRect(x, y, cellW, cellH);
      }
    }
  };

  // Scientific palette mapping with meteorological color standards (WMO / ECMWF styling)
  const getScientificColor = (val: number, layer: string): string => {
    if (layer === 'precipitation') {
      // mm / 6h
      if (val >= 160) return 'rgba(217, 70, 239, 0.90)'; // Extreme Fuchsia/Purple
      if (val >= 100) return 'rgba(239, 68, 68, 0.85)';  // Very Heavy Red
      if (val >= 60)  return 'rgba(249, 115, 22, 0.80)'; // Heavy Orange
      if (val >= 35)  return 'rgba(234, 179, 8, 0.70)';  // Moderate Amber
      if (val >= 18)  return 'rgba(34, 197, 94, 0.60)';  // Light Green
      if (val >= 8)   return 'rgba(6, 182, 212, 0.50)';  // Cyan
      return 'rgba(59, 130, 246, 0.35)';                 // Light Blue
    } else if (layer === 'efi') {
      // Extreme Forecast Index (0.0 to 1.0)
      if (val >= 0.85) return 'rgba(225, 29, 72, 0.90)';  // Shift of Tails / Extreme Rose
      if (val >= 0.70) return 'rgba(234, 88, 12, 0.80)';  // Severe Orange
      if (val >= 0.50) return 'rgba(245, 158, 11, 0.65)'; // High Amber
      if (val >= 0.30) return 'rgba(56, 189, 248, 0.45)'; // Mild Cyan
      return 'rgba(14, 165, 233, 0.25)';
    } else if (layer === 'mslp') {
      // Mean Sea Level Pressure in hPa
      if (val <= 992) return 'rgba(225, 29, 72, 0.85)';  // Deep Depression / Cyclone eye
      if (val <= 998) return 'rgba(249, 115, 22, 0.70)';
      if (val <= 1004) return 'rgba(6, 182, 212, 0.45)';
      if (val <= 1010) return 'rgba(59, 130, 246, 0.30)';
      return 'rgba(99, 102, 241, 0.20)';
    } else if (layer === 'wind') {
      // 850 hPa Wind Speed in m/s
      if (val >= 30) return 'rgba(244, 63, 94, 0.90)';  // Gale / Cyclone Force
      if (val >= 22) return 'rgba(249, 115, 22, 0.80)';
      if (val >= 15) return 'rgba(234, 179, 8, 0.65)';
      if (val >= 8)  return 'rgba(14, 165, 233, 0.45)';
      return 'rgba(56, 189, 248, 0.25)';
    }
    return 'rgba(6, 182, 212, 0.4)';
  };

  // Research Radar & Meteorological Stations
  const drawResearchObservatories = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    RESEARCH_STATIONS.forEach((st) => {
      const x = toCanvasX(st.lon, width);
      const y = toCanvasY(st.lat, height);

      if (x >= 0 && x <= width && y >= 0 && y <= height) {
        ctx.fillStyle = st.type === 'RADAR' ? '#38bdf8' : '#a855f7';
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();

        if (zoom >= 1.4) {
          ctx.fillStyle = 'rgba(203, 213, 225, 0.7)';
          ctx.font = '9px Inter, sans-serif';
          ctx.fillText(st.name, x + 6, y + 3);
        }
      }
    });
  };

  // Dynamic Weather Anomaly Threat Footprint & 5km AI Microscope Box
  const renderExtremeFootprint = (
    ctx: CanvasRenderingContext2D,
    event: WeatherEvent,
    width: number,
    height: number,
    currentLead: number,
    resMode: '12km' | '5km'
  ) => {
    const curPt = event.trajectory_points?.find((p) => p.lead_time === currentLead) || event.trajectory_points?.[0];
    if (!curPt) return;

    const cx = toCanvasX(curPt.lon, width);
    const cy = toCanvasY(curPt.lat, height);

    // 1. Dynamic Bounding Box
    if (curPt.bounding_box) {
      const [minLat, minLon, maxLat, maxLon] = curPt.bounding_box;
      const bx1 = toCanvasX(minLon, width);
      const by1 = toCanvasY(maxLat, height);
      const bx2 = toCanvasX(maxLon, width);
      const by2 = toCanvasY(minLat, height);
      const bw = Math.abs(bx2 - bx1);
      const bh = Math.abs(by2 - by1);

      // Footprint boundary rectangle
      ctx.strokeStyle = resMode === '5km' ? 'rgba(168, 85, 247, 0.8)' : 'rgba(244, 63, 94, 0.7)';
      ctx.lineWidth = resMode === '5km' ? 2 : 1.5;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(bx1, by1, bw, bh);
      ctx.setLineDash([]);

      // Subtle filled threat region
      ctx.fillStyle = resMode === '5km' ? 'rgba(168, 85, 247, 0.08)' : 'rgba(244, 63, 94, 0.06)';
      ctx.fillRect(bx1, by1, bw, bh);

      // Footprint Tag
      ctx.fillStyle = resMode === '5km' ? '#c084fc' : '#fb7185';
      ctx.font = 'bold 10px JetBrains Mono, monospace';
      ctx.fillText(
        resMode === '5km' ? 'AI 5KM REFINED THREAT ZONE' : 'NWP 12KM THREAT FOOTPRINT',
        bx1 + 6,
        by1 - 6
      );
    }

    // 2. Continuous Anomaly Core Intensity Gradient
    const coreGradient = ctx.createRadialGradient(cx, cy, 2, cx, cy, resMode === '5km' ? 36 : 28);
    coreGradient.addColorStop(0, 'rgba(244, 63, 94, 0.85)');
    coreGradient.addColorStop(0.5, 'rgba(249, 115, 22, 0.45)');
    coreGradient.addColorStop(1, 'rgba(244, 63, 94, 0)');

    ctx.fillStyle = coreGradient;
    ctx.beginPath();
    ctx.arc(cx, cy, resMode === '5km' ? 36 : 28, 0, Math.PI * 2);
    ctx.fill();
  };

  // Ensemble Uncertainty Cone of Probability
  const renderUncertaintyCone = (
    ctx: CanvasRenderingContext2D,
    cone: any,
    width: number,
    height: number,
    currentLead: number
  ) => {
    const points = cone.cone_points || [];
    if (points.length < 2) return;

    // Draw shaded boundary cone across future forecast horizons
    const futurePoints = points.filter((p: any) => p.lead_time >= currentLead);
    if (futurePoints.length >= 2) {
      // Left and right boundary paths of the expanding cone
      const leftBoundary: [number, number][] = [];
      const rightBoundary: [number, number][] = [];

      futurePoints.forEach((p: any) => {
        const radDeg = (p.cone_radius_km || 40) / 111.0; // km to approx degrees
        leftBoundary.push([p.center_lat + radDeg * 0.7, p.center_lon - radDeg * 0.7]);
        rightBoundary.push([p.center_lat - radDeg * 0.7, p.center_lon + radDeg * 0.7]);
      });

      ctx.beginPath();
      leftBoundary.forEach(([lat, lon], idx) => {
        const x = toCanvasX(lon, width);
        const y = toCanvasY(lat, height);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      for (let i = rightBoundary.length - 1; i >= 0; i--) {
        const [lat, lon] = rightBoundary[i];
        const x = toCanvasX(lon, width);
        const y = toCanvasY(lat, height);
        ctx.lineTo(x, y);
      }
      ctx.closePath();

      // Shaded translucent cone
      ctx.fillStyle = 'rgba(56, 189, 248, 0.12)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.45)';
      ctx.lineWidth = 1.4;
      ctx.setLineDash([5, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  };

  // Spaghetti Ensemble Tracks
  const renderSpaghettiTracks = (
    ctx: CanvasRenderingContext2D,
    spaghetti: any[],
    width: number,
    height: number
  ) => {
    ctx.lineWidth = 1;
    spaghetti.forEach((member, mIdx) => {
      ctx.strokeStyle = `hsla(${175 + mIdx * 16}, 80%, 65%, 0.35)`;
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

  // Scientific Trajectory Track: Past vs Forecast with T+ Time Markers
  const renderScientificTrajectory = (
    ctx: CanvasRenderingContext2D,
    pts: TrajectoryPoint[],
    width: number,
    height: number,
    currentLead: number
  ) => {
    const pastPts = pts.filter((p) => p.lead_time <= currentLead);
    const futurePts = pts.filter((p) => p.lead_time >= currentLead);

    // 1. Past Trajectory (Solid Cyan Line)
    if (pastPts.length >= 2) {
      ctx.strokeStyle = '#06b6d4';
      ctx.lineWidth = 3;
      ctx.beginPath();
      pastPts.forEach((pt, idx) => {
        const x = toCanvasX(pt.lon, width);
        const y = toCanvasY(pt.lat, height);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    // 2. Future Forecast Trajectory (Dashed Sky-Blue Line)
    if (futurePts.length >= 2) {
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2.5;
      ctx.setLineDash([6, 5]);
      ctx.beginPath();
      futurePts.forEach((pt, idx) => {
        const x = toCanvasX(pt.lon, width);
        const y = toCanvasY(pt.lat, height);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 3. Track Position Markers & Lead Time Stamps
    pts.forEach((pt) => {
      const x = toCanvasX(pt.lon, width);
      const y = toCanvasY(pt.lat, height);
      const isCurrent = pt.lead_time === currentLead;
      const isKeyLead = pt.lead_time % 24 === 0 || pt.lead_time === 0 || pt.lead_time === 120;

      if (!isCurrent) {
        ctx.fillStyle = pt.lead_time < currentLead ? '#0e7490' : '#1e293b';
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(x, y, isKeyLead ? 4.5 : 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        // Forecast time label for key intervals
        if (isKeyLead && zoom >= 0.9) {
          ctx.fillStyle = '#94a3b8';
          ctx.font = '9px JetBrains Mono, monospace';
          ctx.fillText(`T+${pt.lead_time}h`, x + 6, y - 6);
        }
      }
    });

    // 4. Authoritative Current Position Centroid with Radar Pulse
    const currentPt = pts.find((p) => p.lead_time === currentLead) || pts[pts.length - 1];
    if (currentPt) {
      const cx = toCanvasX(currentPt.lon, width);
      const cy = toCanvasY(currentPt.lat, height);

      // Radar Pulse Rings
      ctx.strokeStyle = '#f43f5e';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(cx, cy, 18, 0, Math.PI * 2);
      ctx.stroke();

      ctx.fillStyle = '#f43f5e';
      ctx.beginPath();
      ctx.arc(cx, cy, 6, 0, Math.PI * 2);
      ctx.fill();

      // Meteorological Badge Callout
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 11px JetBrains Mono, monospace';
      ctx.fillText(
        `${currentPt.peak_precip_mm.toFixed(0)} mm/6h | ${currentPt.min_mslp_hpa.toFixed(0)} hPa`,
        cx + 12,
        cy - 6
      );

      ctx.fillStyle = '#38bdf8';
      ctx.font = '10px Inter, sans-serif';
      ctx.fillText(`T+${currentPt.lead_time}h | ${activeEvent?.heading_compass || 'WNW'}`, cx + 12, cy + 9);
    }
  };

  // Mouse Interactivity: Pan & Coordinates Telemetry
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - panOffset.x, y: e.clientY - panOffset.y });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    if (isDragging) {
      setPanOffset({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      });
    }

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const lat = toGeoLat(y, canvas.height);
    const lon = toGeoLon(x, canvas.width);

    // Check if near research station
    const nearStation = RESEARCH_STATIONS.find(
      (st) => Math.abs(st.lat - lat) < 0.6 && Math.abs(st.lon - lon) < 0.6
    );

    setHoverInfo({
      x,
      y,
      lat,
      lon,
      val: null,
      stationName: nearStation?.name
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const resetView = () => {
    setZoom(1.0);
    setPanOffset({ x: 0, y: 0 });
  };

  return (
    <div
      ref={containerRef}
      className="relative w-full h-[620px] rounded-2xl overflow-hidden glass-panel border border-slate-700/80 shadow-2xl flex flex-col select-none"
    >
      {/* Canvas Map Viewport */}
      <canvas
        ref={canvasRef}
        width={1120}
        height={620}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => {
          setIsDragging(false);
          setHoverInfo(null);
        }}
        className={`w-full h-full object-cover ${isDragging ? 'cursor-grabbing' : 'cursor-crosshair'}`}
      />

      {/* Top Left: Layer Selector & Resolution Switcher */}
      <div className="absolute top-4 left-4 flex flex-wrap items-center gap-2 z-20">
        {/* Layer Selector */}
        <div className="flex items-center bg-slate-950/90 backdrop-blur-md p-1 rounded-xl border border-slate-700/80 shadow-xl text-xs font-mono">
          <button
            onClick={() => onLayerChange?.('precipitation')}
            className={`px-3 py-1.5 rounded-lg flex items-center space-x-1.5 transition ${
              activeLayer === 'precipitation'
                ? 'bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/20'
                : 'text-slate-300 hover:bg-slate-800'
            }`}
          >
            <Droplets className="w-3.5 h-3.5" />
            <span>PRECIP</span>
          </button>

          <button
            onClick={() => onLayerChange?.('efi')}
            className={`px-3 py-1.5 rounded-lg flex items-center space-x-1.5 transition ${
              activeLayer === 'efi'
                ? 'bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/20'
                : 'text-slate-300 hover:bg-slate-800'
            }`}
          >
            <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
            <span>EFI</span>
          </button>

          <button
            onClick={() => onLayerChange?.('mslp')}
            className={`px-3 py-1.5 rounded-lg flex items-center space-x-1.5 transition ${
              activeLayer === 'mslp'
                ? 'bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/20'
                : 'text-slate-300 hover:bg-slate-800'
            }`}
          >
            <Gauge className="w-3.5 h-3.5" />
            <span>MSLP</span>
          </button>

          <button
            onClick={() => onLayerChange?.('wind')}
            className={`px-3 py-1.5 rounded-lg flex items-center space-x-1.5 transition ${
              activeLayer === 'wind'
                ? 'bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/20'
                : 'text-slate-300 hover:bg-slate-800'
            }`}
          >
            <Wind className="w-3.5 h-3.5" />
            <span>850 WIND</span>
          </button>
        </div>

        {/* 12km -> 5km Super-Resolution Toggle */}
        <div className="flex items-center bg-slate-950/90 backdrop-blur-md p-1 rounded-xl border border-purple-500/40 shadow-xl text-xs font-mono">
          <span className="text-[10px] text-purple-300 px-2 uppercase font-bold flex items-center space-x-1">
            <Sparkles className="w-3 h-3 text-purple-400" />
            <span>RESOLUTION:</span>
          </span>
          <button
            onClick={() => onResolutionChange('12km')}
            className={`px-2.5 py-1 rounded-lg transition ${
              resolutionMode === '12km'
                ? 'bg-slate-800 text-cyan-300 font-bold border border-slate-700'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            12 km NWP
          </button>
          <span className="text-slate-600 px-0.5">→</span>
          <button
            onClick={() => onResolutionChange('5km')}
            className={`px-2.5 py-1 rounded-lg transition ${
              resolutionMode === '5km'
                ? 'bg-gradient-to-r from-purple-600 to-indigo-600 text-white font-bold shadow-md shadow-purple-500/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
            title="AI-Refined Physics U-Net 5km Localized Threat Field"
          >
            5 km AI Refined
          </button>
        </div>
      </div>

      {/* Top Right: Zoom & View Reset Controls */}
      <div className="absolute top-4 right-4 flex items-center space-x-1.5 bg-slate-950/90 backdrop-blur-md p-1 rounded-xl border border-slate-700/80 shadow-xl z-20">
        <button
          onClick={() => setZoom((prev) => Math.min(prev + 0.25, 3.0))}
          className="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 hover:text-white transition"
          title="Zoom In"
        >
          <ZoomIn className="w-4 h-4" />
        </button>
        <button
          onClick={() => setZoom((prev) => Math.max(prev - 0.25, 0.75))}
          className="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 hover:text-white transition"
          title="Zoom Out"
        >
          <ZoomOut className="w-4 h-4" />
        </button>
        <button
          onClick={resetView}
          className="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 hover:text-cyan-400 transition"
          title="Reset Geographic Extent"
        >
          <RotateCcw className="w-4 h-4" />
        </button>
      </div>

      {/* Hover Coordinates Telemetry Display */}
      {hoverInfo && (
        <div
          className="absolute pointer-events-none bg-slate-950/95 text-cyan-300 px-3 py-1.5 rounded-lg text-xs font-mono border border-cyan-500/40 shadow-2xl z-30 space-y-0.5"
          style={{
            left: Math.min(hoverInfo.x + 16, 880),
            top: Math.min(hoverInfo.y + 16, 540)
          }}
        >
          <div className="font-bold flex items-center space-x-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse"></span>
            <span>{hoverInfo.lat.toFixed(2)}°N, {hoverInfo.lon.toFixed(2)}°E</span>
          </div>
          {hoverInfo.stationName && (
            <div className="text-[10px] text-purple-300">{hoverInfo.stationName}</div>
          )}
        </div>
      )}

      {/* Bottom Right: Dynamic Scientific Scale & Legend */}
      <div className="absolute bottom-4 right-4 bg-slate-950/90 backdrop-blur-md p-3 rounded-xl border border-slate-700/80 text-xs font-mono space-y-2 shadow-2xl z-20 max-w-xs">
        <div className="flex items-center justify-between text-slate-200">
          <div className="flex items-center space-x-1.5 font-bold text-[11px]">
            <Layers className="w-3.5 h-3.5 text-cyan-400" />
            <span className="uppercase">{activeLayer} SCALE</span>
          </div>
          <span className="text-[10px] text-slate-400">
            {activeLayer === 'precipitation' ? 'mm / 6h' : activeLayer === 'efi' ? 'EFI index' : activeLayer === 'mslp' ? 'hPa' : 'm/s'}
          </span>
        </div>

        {activeLayer === 'precipitation' && (
          <div className="space-y-1">
            <div className="h-2 w-52 rounded-full bg-gradient-to-r from-blue-600 via-cyan-400 via-emerald-400 via-amber-400 via-rose-500 to-fuchsia-500" />
            <div className="flex justify-between text-[9px] text-slate-400">
              <span>0 (Light)</span>
              <span>25mm</span>
              <span>60mm</span>
              <span>100mm</span>
              <span className="text-rose-400 font-bold">160+ (Extreme)</span>
            </div>
          </div>
        )}

        {activeLayer === 'efi' && (
          <div className="space-y-1">
            <div className="h-2 w-52 rounded-full bg-gradient-to-r from-sky-400 via-amber-400 to-rose-600" />
            <div className="flex justify-between text-[9px] text-slate-400">
              <span>0.0 (Normal)</span>
              <span>0.50</span>
              <span>0.70</span>
              <span className="text-rose-400 font-bold">0.85+ (Shift of Tails)</span>
            </div>
          </div>
        )}

        {activeLayer === 'mslp' && (
          <div className="space-y-1">
            <div className="h-2 w-52 rounded-full bg-gradient-to-r from-rose-600 via-amber-500 via-cyan-500 to-indigo-600" />
            <div className="flex justify-between text-[9px] text-slate-400">
              <span className="text-rose-400 font-bold">&lt; 992 (Deep Low)</span>
              <span>1000 hPa</span>
              <span>1012 hPa (Normal)</span>
            </div>
          </div>
        )}

        {activeLayer === 'wind' && (
          <div className="space-y-1">
            <div className="h-2 w-52 rounded-full bg-gradient-to-r from-cyan-400 via-amber-400 to-rose-600" />
            <div className="flex justify-between text-[9px] text-slate-400">
              <span>5 m/s</span>
              <span>15 m/s</span>
              <span className="text-rose-400 font-bold">30+ m/s (Gale)</span>
            </div>
          </div>
        )}
      </div>

      {/* Bottom Left: Live State & Anomaly Lead Info */}
      <div className="absolute bottom-4 left-4 bg-slate-950/90 backdrop-blur-md px-3 py-1.5 rounded-xl border border-slate-700/80 text-xs font-mono text-slate-300 flex items-center space-x-3 shadow-xl z-20">
        <div className="flex items-center space-x-1.5">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
          <span className="font-bold text-white uppercase">{activeEvent?.track_id || 'ACTIVE THREAT'}</span>
        </div>
        <span className="text-slate-600">|</span>
        <span>Forecast Lead: <strong className="text-cyan-400">T+{currentLeadTime}h</strong></span>
        <span className="text-slate-600">|</span>
        <span className="text-[10px] text-purple-300">
          {resolutionMode === '5km' ? '5km AI Microscope Active' : '12km NWP Grid'}
        </span>
      </div>
    </div>
  );
};
