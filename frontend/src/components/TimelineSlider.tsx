import React, { useEffect, useState } from 'react';
import { Play, Pause, RotateCcw, FastForward, SkipBack, SkipForward } from 'lucide-react';

interface TimelineSliderProps {
  leadTimes: number[];
  currentLead: number;
  onLeadChange: (lead: number) => void;
  isPlaying: boolean;
  onTogglePlay: () => void;
  speed: number;
  onSpeedChange: (speed: number) => void;
}

export const TimelineSlider: React.FC<TimelineSliderProps> = ({
  leadTimes,
  currentLead,
  onLeadChange,
  isPlaying,
  onTogglePlay,
  speed,
  onSpeedChange,
}) => {
  const currentIndex = leadTimes.indexOf(currentLead);

  const handleStep = (direction: 'prev' | 'next') => {
    if (direction === 'prev' && currentIndex > 0) {
      onLeadChange(leadTimes[currentIndex - 1]);
    } else if (direction === 'next' && currentIndex < leadTimes.length - 1) {
      onLeadChange(leadTimes[currentIndex + 1]);
    }
  };

  return (
    <div className="w-full glass-panel p-4 rounded-xl border border-slate-700/80 shadow-xl space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="px-2.5 py-1 rounded-md bg-cyan-950/80 border border-cyan-500/40 text-cyan-300 font-mono text-xs font-semibold">
            LEAD TIME: T+{currentLead}h
          </div>
          <span className="text-xs text-slate-400">
            (Step {currentIndex + 1} of {leadTimes.length} | Medium-Range Forecast Horizon)
          </span>
        </div>

        {/* Playback Controls */}
        <div className="flex items-center space-x-2">
          <button
            onClick={() => onLeadChange(leadTimes[0])}
            title="Reset to 0h"
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
          <button
            onClick={() => handleStep('prev')}
            disabled={currentIndex === 0}
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-300 transition"
          >
            <SkipBack className="w-4 h-4" />
          </button>
          <button
            onClick={onTogglePlay}
            className="px-3.5 py-1.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold flex items-center space-x-1.5 shadow-lg shadow-cyan-500/20 transition"
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            <span className="text-xs">{isPlaying ? 'PAUSE' : 'PLAY'}</span>
          </button>
          <button
            onClick={() => handleStep('next')}
            disabled={currentIndex === leadTimes.length - 1}
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-300 transition"
          >
            <SkipForward className="w-4 h-4" />
          </button>

          {/* Speed Toggle */}
          <button
            onClick={() => {
              const speeds = [1, 2, 4];
              const nextSpeed = speeds[(speeds.indexOf(speed) + 1) % speeds.length];
              onSpeedChange(nextSpeed);
            }}
            className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-xs font-mono text-cyan-400 border border-slate-700 transition"
          >
            {speed}x
          </button>
        </div>
      </div>

      {/* Scrub Slider with Tick Marks */}
      <div className="relative pt-2">
        <input
          type="range"
          min={0}
          max={leadTimes.length - 1}
          value={currentIndex >= 0 ? currentIndex : 0}
          onChange={(e) => onLeadChange(leadTimes[parseInt(e.target.value)])}
          className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400 focus:outline-none"
        />
        <div className="flex justify-between text-[10px] font-mono text-slate-500 mt-1.5">
          {leadTimes.map((lt, idx) => (
            <span
              key={lt}
              onClick={() => onLeadChange(lt)}
              className={`cursor-pointer transition ${
                lt === currentLead ? 'text-cyan-400 font-bold' : 'hover:text-slate-300'
              } ${idx % 2 === 0 ? 'inline' : 'hidden sm:inline'}`}
            >
              +{lt}h
            </span>
          ))}
        </div>
      </div>
    </div>
  );
};
