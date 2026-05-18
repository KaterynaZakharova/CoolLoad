import React, { useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload,
  MapPin,
  Cpu,
  Wind,
  Sun,
  Thermometer,
  Droplets,
  Zap,
  Trash2,
  Play,
  Loader2,
  Activity,
  Globe2,
  FileText,
  Gauge,
  X,
  Download,
  Eye,
  AlertTriangle,
  ImageIcon,
  BarChart3,
  Server,
  Bot,
  MessageSquare,
} from "lucide-react";

import { MapContainer, TileLayer, Marker, useMap } from "react-leaflet";

import L from "leaflet";
import "leaflet/dist/leaflet.css";

/* -------------------------------------------------------
   Local UI components
------------------------------------------------------- */

function Button({
  children,
  className = "",
  variant = "default",
  size,
  disabled,
  onClick,
  ...props
}) {
  const base =
    "inline-flex items-center justify-center rounded-md px-4 py-2 text-sm font-medium transition disabled:opacity-50 disabled:pointer-events-none";

  const styles =
    variant === "destructive"
      ? "bg-red-500 text-white hover:bg-red-400"
      : variant === "outline"
      ? "border border-white/10 bg-white/5 text-slate-200 hover:bg-white/10 hover:text-white"
      : variant === "ghost"
      ? "bg-transparent text-slate-300 hover:bg-white/10 hover:text-white"
      : "bg-cyan-400 text-slate-950 hover:bg-cyan-300";

  const iconSize = size === "icon" ? "h-9 w-9 p-0" : "";

  return (
    <button
      className={`${base} ${styles} ${iconSize} ${className}`}
      disabled={disabled}
      onClick={onClick}
      {...props}
    >
      {children}
    </button>
  );
}

function Card({ children, className = "" }) {
  return <div className={`rounded-3xl ${className}`}>{children}</div>;
}

function CardContent({ children, className = "" }) {
  return <div className={className}>{children}</div>;
}

/* -------------------------------------------------------
   Demo data
------------------------------------------------------- */

const IMAGE_PLACEHOLDERS = [
  "https://images.unsplash.com/photo-1558494949-ef010cbdcc31?auto=format&fit=crop&w=900&q=80",
  "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=900&q=80",
  "https://images.unsplash.com/photo-1544197150-b99a580bb7a8?auto=format&fit=crop&w=900&q=80",
  "https://images.unsplash.com/photo-1497366754035-f200968a6e72?auto=format&fit=crop&w=900&q=80",
  "https://images.unsplash.com/photo-1509395176047-4a66953fd231?auto=format&fit=crop&w=900&q=80",
];

const DEFAULT_WALL_PHYSICS = {
  material: "Concrete (built-in)",
  specific_heat_kj_per_kg_k: 0.88,
  density_kg_m3: 2400,
};

const INITIAL_CENTERS = [
  {
    id: "dc-toronto",
    name: "Toronto Edge DC",
    lat: 43.6532,
    lon: -79.3832,
    baseLoad: 68,
    optimalLoad: 68,
    weather: {
      temp: 21,
      humidity: 58,
      solar: 620,
      windSpeed: 4.2,
      windDirection: "NE",
    },
    image: IMAGE_PLACEHOLDERS[0],
    simulation: null,
    dirty: false,
    pdfResources: null,
    wallPhysics: { ...DEFAULT_WALL_PHYSICS },
  },
  {
    id: "dc-reykjavik",
    name: "Reykjavik Cold DC",
    lat: 64.1466,
    lon: -21.9426,
    baseLoad: 60,
    optimalLoad: 60,
    weather: {
      temp: 8,
      humidity: 71,
      solar: 410,
      windSpeed: 0.2,
      windDirection: "W",
    },
    image: IMAGE_PLACEHOLDERS[2],
    simulation: null,
    dirty: false,
    pdfResources: null,
    wallPhysics: { ...DEFAULT_WALL_PHYSICS },
  },
];

const PRESET_UPLOAD_RESULTS = [
  {
    name: "Phoenix AI Training DC",
    lat: 33.4484,
    lon: -112.074,
    baseLoad: 76,
    weather: {
      temp: 34,
      humidity: 22,
      solar: 920,
      windSpeed: 3.1,
      windDirection: "SW",
    },
    image: IMAGE_PLACEHOLDERS[1],
  },
  {
    name: "Helsinki Inference DC",
    lat: 60.1699,
    lon: 24.9384,
    baseLoad: 53,
    weather: {
      temp: 12,
      humidity: 67,
      solar: 480,
      windSpeed: 5.6,
      windDirection: "N",
    },
    image: IMAGE_PLACEHOLDERS[2],
  },
  {
    name: "Singapore Cloud DC",
    lat: 1.3521,
    lon: 103.8198,
    baseLoad: 81,
    weather: {
      temp: 30,
      humidity: 78,
      solar: 760,
      windSpeed: 2.4,
      windDirection: "SE",
    },
    image: IMAGE_PLACEHOLDERS[3],
  },
  {
    name: "Stockholm Green DC",
    lat: 59.3293,
    lon: 18.0686,
    baseLoad: 47,
    weather: {
      temp: 10,
      humidity: 64,
      solar: 390,
      windSpeed: 6.3,
      windDirection: "NW",
    },
    image: IMAGE_PLACEHOLDERS[4],
  },
];

/* -------------------------------------------------------
   Helpers
------------------------------------------------------- */

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function demoParsePdf(fileName, index) {
  const preset = PRESET_UPLOAD_RESULTS[index % PRESET_UPLOAD_RESULTS.length];
  const cleanName = fileName?.replace(/\.pdf$/i, "").replace(/[_-]/g, " ");

  return {
    ...preset,
    name: cleanName ? `${cleanName} DC` : preset.name,
  };
}

/** Base URL for simulation API (no trailing slash). Empty = same origin; Vite dev proxies /api → 127.0.0.1:8765. */
const SIM_API_BASE = String(import.meta.env.VITE_SIM_API_BASE ?? "")
  .trim()
  .replace(/\/+$/, "");

function wallPhysicsOptionsForCenter(center) {
  const opts = [
    {
      label: "Concrete (built-in): Cp 0.88 kJ/kg·K, ρ 2400 kg/m³",
      value: {
        material: "Concrete (built-in)",
        specific_heat_kj_per_kg_k: 0.88,
        density_kg_m3: 2400,
      },
    },
  ];
  const rows = center.pdfResources?.thermal_capacities ?? [];
  rows.forEach((r, i) => {
    const cp = Number(r.specific_heat_kj_per_kg_k);
    if (!Number.isFinite(cp)) return;
    const name = r.matched_material || r.material || `Material ${i + 1}`;
    opts.push({
      label: `${name}: Cp ${cp} kJ/kg·K (density from model lookup)`,
      value: {
        material: name,
        specific_heat_kj_per_kg_k: cp,
        density_kg_m3: null,
      },
    });
  });
  return opts;
}

function wallPhysicsMatches(a, b) {
  if (!a || !b) return false;
  return (
    (a.material || "") === (b.material || "") &&
    Number(a.specific_heat_kj_per_kg_k) === Number(b.specific_heat_kj_per_kg_k) &&
    (a.density_kg_m3 ?? null) === (b.density_kg_m3 ?? null)
  );
}

function defaultWallPhysicsFromExtract(data) {
  const rows = data?.thermal_capacities ?? [];
  const row = rows.find((r) => Number.isFinite(Number(r.specific_heat_kj_per_kg_k)));
  if (!row) return null;
  return {
    material: row.matched_material || row.material || "From PDF",
    specific_heat_kj_per_kg_k: Number(row.specific_heat_kj_per_kg_k),
    density_kg_m3: null,
  };
}

function riskFromMetrics(m) {
  const maxT = Number(m.max_temp_C);
  if (Number.isNaN(maxT)) return "Low";
  if (maxT > 45) return "High";
  if (maxT > 38) return "Medium";
  return "Low";
}

function statusFromRisk(risk) {
  if (risk === "High") return "Reduce load or increase cooling";
  if (risk === "Medium") return "Acceptable with monitoring";
  return "Good thermal envelope";
}

function buildSimulationFromBayesSite(siteOut, cacheBust, runId) {
  const m = siteOut.metrics;
  const raw = siteOut.assets || {};
  const assets = Object.fromEntries(
    Object.entries(raw).map(([k, url]) => {
      const sep = String(url).includes("?") ? "&" : "?";
      return [k, `${url}${sep}t=${cacheBust}`];
    })
  );
  const risk = riskFromMetrics(m);
  const dAmb = Number(m.central_building_anomaly_C);
  const peak = Number(m.max_temp_C);
  const optMw = Number(siteOut.optimal_load_mw);
  const ex = Number(siteOut.assigned_extra_mw);
  return {
    deltaT: Number(dAmb.toFixed(2)),
    maxTemp: Number(peak.toFixed(2)),
    risk,
    status: statusFromRisk(risk),
    gifLabel: assets.gif ? "04_heat_plume_animation.gif" : "GIF unavailable",
    finalResult: `Bayes-optimal load ${optMw.toFixed(
      2
    )} MW (extra ${ex.toFixed(2)} MW vs baseline). Central ΔT ${dAmb.toFixed(
      2
    )}°C; domain peak ${peak.toFixed(2)}°C.`,
    assignedExtra: Number(ex.toFixed(2)),
    assets,
    engine: "bayes",
    runId,
    siteObjective: siteOut.site_objective,
  };
}

function buildSimulationFromPhysics(center, extraLoad, apiResult) {
  const m = apiResult.metrics;
  const raw = apiResult.assets || {};
  const t = Date.now();
  const assets = Object.fromEntries(
    Object.entries(raw).map(([k, url]) => {
      const sep = String(url).includes("?") ? "&" : "?";
      return [k, `${url}${sep}t=${t}`];
    })
  );
  const risk = riskFromMetrics(m);
  const dAmb = Number(m.central_building_anomaly_C);
  const peak = Number(m.max_temp_C);
  return {
    deltaT: Number(dAmb.toFixed(2)),
    maxTemp: Number(peak.toFixed(2)),
    risk,
    status: statusFromRisk(risk),
    gifLabel: assets.gif ? "04_heat_plume_animation.gif" : "GIF not generated",
    finalResult: `Physical model: central building ΔT above ambient ${dAmb.toFixed(
      2
    )}°C; domain peak ${peak.toFixed(2)}°C.`,
    assignedExtra: Number(extraLoad.toFixed(1)),
    assets,
    engine: "physical",
  };
}

function generateReport(centers, newLoad, optimizationMeta = null) {
  const totalBase = centers.reduce((s, c) => s + Number(c.baseLoad || 0), 0);
  const totalOptimal = centers.reduce(
    (s, c) => s + Number(c.optimalLoad || 0),
    0
  );

  return {
    title: "Data Center Thermal Load Redistribution Report",
    generatedAt: new Date().toISOString(),
    newLoadToRedistributeMW: newLoad,
    totalBaseLoadMW: Number(totalBase.toFixed(1)),
    totalOptimalLoadMW: Number(totalOptimal.toFixed(1)),
    optimization: optimizationMeta,
    centers: centers.map((c) => ({
      id: c.id,
      name: c.name,
      location: {
        lat: c.lat,
        lon: c.lon,
      },
      baseLoadMW: c.baseLoad,
      optimalLoadMW: c.optimalLoad,
      weather: c.weather,
      simulation: c.simulation,
      agentVerdict: c.agentVerdict ?? null,
    })),
  };
}

function downloadJson(filename, data) {
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: "application/json",
  });

  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

async function downloadBlobFromUrl(url, filename) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Download failed: ${res.status}`);
  const blob = await res.blob();
  const href = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = href;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(href);
}

/*
  IMPORTANT:
  The previous marker appeared to move because the inner HTML was manually
  translated while Leaflet also applied an icon anchor. This version lets
  Leaflet control positioning and only uses iconAnchor: [21, 42].
*/
function makeMarkerIcon(active) {
  return L.divIcon({
    className: "dc-leaflet-marker",
    html: `
      <div style="
        position: relative;
        width: 42px;
        height: 42px;
      ">
        <div style="
          position: absolute;
          left: 50%;
          top: 50%;
          width: ${active ? "46px" : "34px"};
          height: ${active ? "46px" : "34px"};
          transform: translate(-50%, -50%);
          border-radius: 999px;
          background: ${
            active ? "rgba(251,146,60,.32)" : "rgba(34,211,238,.25)"
          };
          filter: blur(8px);
        "></div>

        <svg width="42" height="42" viewBox="0 0 24 24" fill="${
          active ? "#fb923c" : "#22d3ee"
        }" stroke="${
      active ? "#fed7aa" : "#cffafe"
    }" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="position:relative; filter: drop-shadow(0 10px 18px rgba(0,0,0,.55));">
          <path d="M20 10c0 4.993-5.539 10.193-7.399 11.799a1 1 0 0 1-1.202 0C9.539 20.193 4 14.993 4 10a8 8 0 0 1 16 0"/>
          <circle cx="12" cy="10" r="3" fill="#020617"/>
        </svg>
      </div>
    `,
    iconSize: [42, 42],
    iconAnchor: [21, 42],
    popupAnchor: [0, -42],
  });
}

/* -------------------------------------------------------
   Visual components
------------------------------------------------------- */

function Metric({ icon: Icon, label, value, unit }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-3">
      <div className="mb-1 flex items-center gap-2 text-xs text-slate-400">
        <Icon className="h-3.5 w-3.5" />
        {label}
      </div>
      <div className="text-lg font-semibold text-white">
        {value}
        {unit ? (
          <span className="ml-1 text-xs font-normal text-slate-400">
            {unit}
          </span>
        ) : null}
      </div>
    </div>
  );
}

function SiteImage({ src, label }) {
  return (
    <div className="relative h-32 overflow-hidden rounded-2xl border border-white/10 bg-slate-900">
      <img
        src={src}
        alt={label}
        className="h-full w-full object-cover opacity-75"
        onError={(e) => {
          e.currentTarget.style.display = "none";
        }}
      />
      <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent" />
      <div className="absolute bottom-3 left-3 flex items-center gap-2 text-xs font-medium text-white">
        <ImageIcon className="h-4 w-4 text-cyan-300" />
        {label}
      </div>
    </div>
  );
}

function LoadBar({ label, value, max, tone = "cyan" }) {
  const pct = clamp((Number(value || 0) / Math.max(max, 1)) * 100, 0, 100);
  const color =
    tone === "orange"
      ? "from-orange-300 to-red-400"
      : "from-cyan-300 to-blue-400";

  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-[11px] text-slate-400">
        <span>{label}</span>
        <span className="font-semibold text-white">{value} MW</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-white/10">
        <div
          className={`h-full rounded-full bg-gradient-to-r ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

const AGENT_PHASES = new Set([
  "agent_review",
  "agent_site",
  "agent_orchestrator",
]);

function buildAgentConversationEntries(bayesEvents, agentHistory) {
  const entries = [];
  let seq = 0;
  const push = (entry) => entries.push({ id: seq++, ...entry });

  for (const ev of bayesEvents || []) {
    if (ev.type !== "log" || !AGENT_PHASES.has(ev.phase)) continue;
    if (ev.phase === "agent_site") {
      push({
        kind: "site",
        round: ev.agent_round ?? 0,
        siteIndex: ev.site_index,
        siteName: ev.site_name,
        verdict: ev.verdict,
        loadMw: ev.load_mw,
        reasons: ev.reasons || [],
        concerns: ev.concerns || [],
        source: ev.source,
        centralDeltaT: ev.central_delta_t_c,
        message: ev.message,
      });
    } else if (ev.phase === "agent_orchestrator") {
      push({
        kind: "orchestrator",
        round: ev.agent_round ?? 0,
        message: ev.orchestrator_notes || ev.message,
      });
    } else {
      push({
        kind: "system",
        round: ev.agent_round ?? 0,
        message: ev.message,
      });
    }
  }

  if (entries.length === 0 && agentHistory?.length) {
    for (const round of agentHistory) {
      const ri = round.round ?? 0;
      push({
        kind: "system",
        round: ri,
        message: `Review round ${ri + 1} — ${
          round.all_accepted ? "all sites accepted" : "orchestrator adjusting loads"
        }`,
      });
      for (const r of round.reviews || []) {
        push({
          kind: "site",
          round: ri,
          siteIndex: r.site_index,
          siteName: r.site_name,
          verdict: r.verdict,
          loadMw: r.load_mw,
          reasons: r.reasons || [],
          concerns: r.concerns || [],
          source: r.source,
          centralDeltaT: r.central_delta_t_c,
          message: r.reasons?.join(" ") || "",
        });
      }
    }
  }

  return entries;
}

function AgentConversationPanel({
  bayesEvents,
  agentHistory,
  centers,
  isRunning,
}) {
  const entries = useMemo(
    () => buildAgentConversationEntries(bayesEvents, agentHistory),
    [bayesEvents, agentHistory]
  );
  const scrollRef = useRef(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [entries.length, isRunning]);

  const siteColor = (name) => {
    const idx = centers.findIndex((c) => c.name === name);
    const hues = [
      "border-cyan-400/40 bg-cyan-500/10",
      "border-violet-400/40 bg-violet-500/10",
      "border-emerald-400/40 bg-emerald-500/10",
      "border-amber-400/40 bg-amber-500/10",
      "border-rose-400/40 bg-rose-500/10",
      "border-sky-400/40 bg-sky-500/10",
    ];
    return hues[(idx >= 0 ? idx : name?.length || 0) % hues.length];
  };

  return (
    <Card className="border-violet-400/25 bg-white/[0.04] shadow-2xl shadow-black/30 backdrop-blur-xl">
      <CardContent className="p-5">
        <motion.div className="mb-4 flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="flex items-center gap-2 text-lg font-bold text-white">
              <MessageSquare className="h-5 w-5 text-violet-300" />
              Datacenter agent conversation
            </h2>
            <p className="mt-1 max-w-2xl text-sm text-slate-400">
              Local site agents review plume / anomaly imagery; the orchestrator
              replies when loads must be redistributed.
            </p>
          </div>
          {isRunning && (
            <div className="flex items-center gap-2 rounded-full border border-violet-400/30 bg-violet-500/10 px-3 py-1.5 text-xs text-violet-200">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Live
            </div>
          )}
        </motion.div>

        <div
          ref={scrollRef}
          className="min-h-[22rem] max-h-[min(52vh,520px)] overflow-y-auto rounded-2xl border border-white/10 bg-slate-950/80 p-4 md:min-h-[26rem] md:p-5"
        >
          {entries.length === 0 ? (
            <div className="flex h-full min-h-[18rem] flex-col items-center justify-center text-center text-sm text-slate-500">
              <Bot className="mb-3 h-10 w-10 text-slate-600" />
              <p>No agent messages yet.</p>
              <p className="mt-1 text-xs">
                Run optimization to see per-datacenter accept / reject dialogue and
                orchestrator responses.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {entries.map((entry) => {
                if (entry.kind === "orchestrator") {
                  return (
                    <div
                      key={entry.id}
                      className="rounded-2xl border border-violet-500/35 bg-violet-500/10 p-4"
                    >
                      <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-violet-200">
                        <Bot className="h-4 w-4" />
                        Global orchestrator
                        {entry.round > 0 && (
                          <span className="font-normal text-violet-300/80">
                            · after round {entry.round}
                          </span>
                        )}
                      </div>
                      <p className="text-sm leading-relaxed text-slate-200">
                        {entry.message}
                      </p>
                    </div>
                  );
                }
                if (entry.kind === "system") {
                  return (
                    <div
                      key={entry.id}
                      className="mx-auto max-w-[95%] rounded-xl border border-white/10 bg-white/5 px-4 py-2.5 text-center text-xs text-slate-400"
                    >
                      {entry.round > 0 && (
                        <span className="mr-2 font-mono text-slate-500">
                          R{entry.round + 1}
                        </span>
                      )}
                      {entry.message}
                    </div>
                  );
                }
                const accept = entry.verdict === "accept";
                return (
                  <div
                    key={entry.id}
                    className={`rounded-2xl border p-4 ${siteColor(entry.siteName)}`}
                  >
                    <div className="mb-2 flex flex-wrap items-center gap-2">
                      <Server className="h-4 w-4 text-cyan-300" />
                      <span className="font-semibold text-white">
                        {entry.siteName}
                      </span>
                      <span
                        className={`rounded-full px-2.5 py-0.5 text-[11px] font-semibold uppercase ${
                          accept
                            ? "bg-emerald-500/20 text-emerald-200"
                            : "bg-amber-500/20 text-amber-200"
                        }`}
                      >
                        {entry.verdict}
                      </span>
                      {entry.loadMw != null && (
                        <span className="font-mono text-xs text-slate-400">
                          {Number(entry.loadMw).toFixed(2)} MW
                        </span>
                      )}
                      {entry.source && (
                        <span className="text-[10px] text-slate-500">
                          via {entry.source}
                        </span>
                      )}
                    </div>
                    {entry.centralDeltaT != null && (
                      <p className="mb-2 text-xs text-slate-400">
                        Central ΔT:{" "}
                        <span className="font-mono text-slate-200">
                          {Number(entry.centralDeltaT).toFixed(2)} °C
                        </span>
                      </p>
                    )}
                    {entry.concerns?.length > 0 && (
                      <div className="mb-2 flex flex-wrap gap-1.5">
                        {entry.concerns.map((c) => (
                          <span
                            key={c}
                            className="rounded-md bg-black/30 px-2 py-0.5 text-[10px] text-amber-200/90"
                          >
                            {c.replace(/_/g, " ")}
                          </span>
                        ))}
                      </div>
                    )}
                    {(entry.reasons?.length > 0 || entry.message) && (
                      <ul className="list-inside list-disc space-y-1 text-sm leading-relaxed text-slate-300">
                        {(entry.reasons?.length ? entry.reasons : [entry.message])
                          .filter(Boolean)
                          .map((line, i) => (
                            <li key={i}>{line}</li>
                          ))}
                      </ul>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function DataCenterFrame({
  center,
  selected,
  maxLoad,
  onSelect,
  onRemove,
  onBaseLoadChange,
}) {
  return (
    <motion.div
      layout
      onClick={onSelect}
      className={`min-w-[330px] cursor-pointer rounded-3xl border p-4 transition ${
        selected
          ? "border-cyan-300/70 bg-cyan-300/10 shadow-lg shadow-cyan-950/40"
          : "border-white/10 bg-slate-950/55 hover:border-white/20 hover:bg-white/[0.06]"
      }`}
    >
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 font-semibold text-white">
            <Server className="h-4 w-4 text-cyan-300" />
            {center.name}
          </div>
          <div className="mt-1 text-xs text-slate-500">
            {center.lat.toFixed(2)}, {center.lon.toFixed(2)}
          </div>
        </div>

        <button
          className="rounded-xl p-2 text-slate-500 hover:bg-red-500/10 hover:text-red-300"
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
        >
          <Trash2 className="h-4 w-4" />
        </button>
      </div>

      <SiteImage src={center.image} label="Data center placeholder" />

      {center.simulation?.assets?.final && !center.simulation?.error && (
        <div className="mt-3 overflow-hidden rounded-2xl border border-cyan-400/30 bg-black/40 shadow-lg shadow-cyan-950/20">
          <img
            src={center.simulation.assets.final}
            alt="Final temperature field"
            className="h-28 w-full object-cover object-center"
          />
          <div className="px-2 py-1 text-[10px] text-cyan-200/90">
            Latest physics run — final temperature
          </div>
        </div>
      )}

      {center.dirty && (
        <div className="mt-3 flex items-center gap-2 rounded-2xl border border-orange-300/20 bg-orange-400/10 p-3 text-xs text-orange-100">
          <AlertTriangle className="h-4 w-4 shrink-0" />
          Load changed. Rerun simulation.
        </div>
      )}

      <div className="mt-4 space-y-3">
        <LoadBar
          label="Base load"
          value={center.baseLoad}
          max={maxLoad}
          tone="cyan"
        />
        <LoadBar
          label="Optimal load"
          value={center.optimalLoad}
          max={maxLoad}
          tone="orange"
        />
      </div>

      <div className="mt-4 flex items-center justify-between text-xs text-slate-400">
        <span>Adjust baseline load</span>
        <span>{center.baseLoad} MW</span>
      </div>

      <input
        type="range"
        min="5"
        max="150"
        value={center.baseLoad}
        onClick={(e) => e.stopPropagation()}
        onChange={(e) => onBaseLoadChange(Number(e.target.value))}
        className="mt-2 w-full accent-cyan-300"
      />

      <div className="mt-3 grid grid-cols-4 gap-2 text-[11px] text-slate-300">
        <div className="rounded-xl bg-black/20 p-2">
          <Thermometer className="mb-1 h-3.5 w-3.5 text-orange-300" />
          {center.weather.temp}°C
        </div>

        <div className="rounded-xl bg-black/20 p-2">
          <Droplets className="mb-1 h-3.5 w-3.5 text-blue-300" />
          {center.weather.humidity}%
        </div>

        <div className="rounded-xl bg-black/20 p-2">
          <Sun className="mb-1 h-3.5 w-3.5 text-yellow-300" />
          {center.weather.solar}
        </div>

        <div className="rounded-xl bg-black/20 p-2">
          <Wind className="mb-1 h-3.5 w-3.5 text-cyan-300" />
          {center.weather.windSpeed} {center.weather.windDirection}
        </div>
      </div>

      {center.simulation && !center.simulation.error && (
        <div className="mt-3 rounded-2xl bg-black/20 p-3 text-xs text-slate-300">
          ΔT {center.simulation.deltaT}°C · max{" "}
          {center.simulation.maxTemp}°C · {center.simulation.status}
        </div>
      )}
      {center.simulation?.error && (
        <div className="mt-3 rounded-2xl border border-red-400/25 bg-red-500/10 p-2 text-[11px] text-red-100">
          Run failed: {String(center.simulation.error).slice(0, 220)}
          {String(center.simulation.error).length > 220 ? "…" : ""}
        </div>
      )}
    </motion.div>
  );
}

function FlyToSelected({ center }) {
  const map = useMap();

  React.useEffect(() => {
    if (!center) return;

    map.flyTo([center.lat, center.lon], Math.max(map.getZoom(), 4), {
      duration: 0.85,
    });
  }, [center, map]);

  return null;
}

function MinimalWorldMap({ centers, selectedId, setSelectedId }) {
  const selectedCenter = centers.find((c) => c.id === selectedId) || null;

  return (
    <Card className="relative overflow-hidden border-white/10 bg-slate-950/80 shadow-2xl shadow-cyan-950/40 backdrop-blur">
      <CardContent className="p-0">
        <div className="absolute left-6 top-6 z-[500] flex items-center gap-3 rounded-2xl border border-white/10 bg-black/55 px-4 py-3 backdrop-blur-xl">
          <Globe2 className="h-5 w-5 text-cyan-300" />
          <div>
            <div className="text-sm font-semibold text-white">
              Minimal World Map
            </div>
            <div className="text-xs text-slate-400">
              Fixed geospatial pins. Click pins or frames.
            </div>
          </div>
        </div>

        <div className="h-[690px] w-full overflow-hidden rounded-3xl">
          <MapContainer
            center={[35, -30]}
            zoom={2}
            minZoom={2}
            maxZoom={8}
            scrollWheelZoom
            worldCopyJump={false}
            maxBounds={[
              [-85, -180],
              [85, 180],
            ]}
            maxBoundsViscosity={1.0}
            className="h-full w-full bg-black"
            zoomControl={false}
          >
            <TileLayer
              attribution='&copy; OpenStreetMap &copy; CARTO'
              url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
              noWrap
              bounds={[
                [-85, -180],
                [85, 180],
              ]}
            />

            <FlyToSelected center={selectedCenter} />

            {centers.map((center) => {
              const active = selectedId === center.id;

              return (
                <Marker
                  key={center.id}
                  position={[center.lat, center.lon]}
                  icon={makeMarkerIcon(active)}
                  eventHandlers={{
                    click: () => setSelectedId(center.id),
                  }}
                />
              );
            })}
          </MapContainer>
        </div>
      </CardContent>
    </Card>
  );
}

function SelectedCenterPanel({ center, onClose, onWallPhysicsChange }) {
  if (!center) {
    return (
      <Card className="border-white/10 bg-white/[0.04] shadow-2xl shadow-black/30 backdrop-blur-xl">
        <CardContent className="p-5">
          <div className="rounded-3xl border border-dashed border-white/10 p-8 text-center text-sm text-slate-500">
            Select a data center from the map or from the frames below the map.
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="sticky top-6 border-white/10 bg-white/[0.04] shadow-2xl shadow-black/30 backdrop-blur-xl">
      <CardContent className="p-5">
        <div className="mb-4 flex items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 text-xl font-bold text-white">
              <Cpu className="h-5 w-5 text-cyan-300" />
              {center.name}
            </div>
            <div className="mt-1 text-xs text-slate-400">
              {center.lat.toFixed(3)}, {center.lon.toFixed(3)} · base load{" "}
              {center.baseLoad} MW
            </div>
          </div>

          <Button
            size="icon"
            variant="ghost"
            className="h-8 w-8"
            onClick={onClose}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        <SiteImage src={center.image} label="Selected data center" />

        <div className="mt-4">
          <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">
            Envelope material (physics)
          </label>
          <select
            className="w-full rounded-xl border border-white/10 bg-black/40 px-3 py-2 text-sm text-white outline-none focus:border-cyan-400/50"
            value={(() => {
              const opts = wallPhysicsOptionsForCenter(center);
              const ix = opts.findIndex((o) =>
                wallPhysicsMatches(o.value, center.wallPhysics)
              );
              return String(Math.max(0, ix));
            })()}
            onChange={(e) => {
              const opts = wallPhysicsOptionsForCenter(center);
              const sel = opts[Number(e.target.value)];
              if (sel && onWallPhysicsChange) {
                onWallPhysicsChange(center.id, { ...sel.value });
              }
            }}
          >
            {wallPhysicsOptionsForCenter(center).map((o, i) => (
              <option key={i} value={String(i)}>
                {o.label}
              </option>
            ))}
          </select>
        </div>

        {center.pdfResources && (
          <div className="mt-4 space-y-3 rounded-2xl border border-white/10 bg-black/20 p-3">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
              <FileText className="h-3.5 w-3.5" />
              PDF resources
            </div>
            {center.pdfResources.warnings?.length > 0 && (
              <div className="rounded-xl border border-amber-400/20 bg-amber-500/10 p-2 text-xs text-amber-100">
                {center.pdfResources.warnings.map((w, i) => (
                  <div key={i}>{w}</div>
                ))}
              </div>
            )}
            <div className="grid gap-2 text-xs text-slate-300">
              {center.pdfResources.specs?.address && (
                <div>
                  <span className="text-slate-500">Address: </span>
                  {center.pdfResources.specs.address}
                </div>
              )}
              {center.pdfResources.specs?.maximum_power_load_kw != null && (
                <div>
                  <span className="text-slate-500">Max power (doc): </span>
                  {center.pdfResources.specs.maximum_power_load_kw} kW
                </div>
              )}
              {center.pdfResources.specs?.cooling_configuration && (
                <div>
                  <span className="text-slate-500">Cooling: </span>
                  {center.pdfResources.specs.cooling_configuration}
                </div>
              )}
              {Array.isArray(center.pdfResources.specs?.building_materials) &&
                center.pdfResources.specs.building_materials.length > 0 && (
                  <div>
                    <span className="text-slate-500">Materials: </span>
                    {center.pdfResources.specs.building_materials.join(", ")}
                  </div>
                )}
            </div>
            {center.pdfResources.thermal_capacities?.length > 0 && (
              <div>
                <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-slate-500">
                  Thermal capacities (Cp, kJ/kg·K)
                </div>
                <div className="max-h-40 overflow-auto rounded-lg border border-white/5 text-[11px]">
                  <table className="w-full text-left text-slate-300">
                    <thead className="sticky top-0 bg-slate-900/90 text-slate-500">
                      <tr>
                        <th className="p-2">Material</th>
                        <th className="p-2">Cp</th>
                        <th className="p-2">Phase</th>
                      </tr>
                    </thead>
                    <tbody>
                      {center.pdfResources.thermal_capacities.map((r, i) => (
                        <tr key={i} className="border-t border-white/5">
                          <td className="p-2">
                            {r.matched_material || r.material || "—"}
                          </td>
                          <td className="p-2 font-mono">
                            {r.specific_heat_kj_per_kg_k ?? "—"}
                          </td>
                          <td className="p-2">{r.phase || "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {center.dirty && (
          <div className="mt-4 flex items-center gap-2 rounded-2xl border border-orange-300/20 bg-orange-400/10 p-3 text-sm text-orange-100">
            <AlertTriangle className="h-4 w-4 shrink-0" />
            Baseline load changed. Rerun simulation to update this site.
          </div>
        )}

        <div className="mt-4 grid grid-cols-2 gap-3">
          <Metric
            icon={Cpu}
            label="Base load"
            value={center.baseLoad}
            unit="MW"
          />
          <Metric
            icon={Zap}
            label="Optimal load"
            value={center.optimalLoad}
            unit="MW"
          />
          <Metric
            icon={Thermometer}
            label="Air temp"
            value={center.weather.temp}
            unit="°C"
          />
          <Metric
            icon={Droplets}
            label="Humidity"
            value={center.weather.humidity}
            unit="%"
          />
          <Metric
            icon={Sun}
            label="Solar"
            value={center.weather.solar}
            unit="W/m²"
          />
          <Metric
            icon={Wind}
            label="Wind"
            value={`${center.weather.windSpeed} m/s`}
            unit={center.weather.windDirection}
          />
        </div>

        {center.simulation?.error && (
          <div className="mt-4 rounded-2xl border border-red-400/30 bg-red-500/10 p-4 text-xs text-red-100">
            <div className="mb-2 font-semibold text-red-200">Simulation error</div>
            <pre className="max-h-48 overflow-auto whitespace-pre-wrap font-mono text-[11px]">
              {center.simulation.error}
            </pre>
          </div>
        )}

        {center.simulation && !center.simulation.error && (
          <div className="mt-4 rounded-2xl border border-cyan-400/20 bg-cyan-400/5 p-4">
            <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-cyan-100">
              <Activity className="h-4 w-4" />
              {center.simulation.engine === "bayes"
                ? "Bayes optimization + full physics"
                : center.simulation.engine === "physical"
                ? "Physics output"
                : "Physics preview"}
            </div>

            <div className="grid grid-cols-3 gap-2 text-center">
              <div className="rounded-xl bg-white/[0.04] p-2">
                <div className="text-[10px] text-slate-400">ΔT</div>
                <div className="font-bold text-white">
                  {center.simulation.deltaT}°C
                </div>
              </div>

              <div className="rounded-xl bg-white/[0.04] p-2">
                <div className="text-[10px] text-slate-400">Max</div>
                <div className="font-bold text-white">
                  {center.simulation.maxTemp}°C
                </div>
              </div>

              <div className="rounded-xl bg-white/[0.04] p-2">
                <div className="text-[10px] text-slate-400">Risk</div>
                <div className="font-bold text-white">
                  {center.simulation.risk}
                </div>
              </div>
            </div>

            <div className="mt-3 text-xs text-slate-300">
              {center.simulation.finalResult}
            </div>

            {center.simulation.assets?.gif ? (
              <div className="mt-3">
                <div className="mb-1 text-[10px] uppercase tracking-wide text-slate-500">
                  Heat plume ({center.simulation.gifLabel})
                </div>
                <img
                  src={center.simulation.assets.gif}
                  alt="Heat plume animation"
                  className="w-full rounded-xl border border-white/10 shadow-lg shadow-black/40"
                />
              </div>
            ) : (
              <div className="mt-1 text-xs text-slate-500">
                GIF: {center.simulation.gifLabel}
              </div>
            )}

            {center.simulation.assets && (
              <div className="mt-4 space-y-2">
                <div className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">
                  Static frames
                </div>
                <div className="grid grid-cols-2 gap-2">
                  {center.simulation.assets.masks && (
                    <a
                      href={center.simulation.assets.masks}
                      target="_blank"
                      rel="noreferrer"
                      className="block overflow-hidden rounded-xl border border-white/10 bg-black/30"
                    >
                      <img
                        src={center.simulation.assets.masks}
                        alt="Building masks"
                        className="h-28 w-full object-cover"
                      />
                      <div className="px-2 py-1 text-[10px] text-slate-400">Masks</div>
                    </a>
                  )}
                  {center.simulation.assets.final && (
                    <a
                      href={center.simulation.assets.final}
                      target="_blank"
                      rel="noreferrer"
                      className="block overflow-hidden rounded-xl border border-white/10 bg-black/30"
                    >
                      <img
                        src={center.simulation.assets.final}
                        alt="Final temperature"
                        className="h-28 w-full object-cover"
                      />
                      <div className="px-2 py-1 text-[10px] text-slate-400">Final T</div>
                    </a>
                  )}
                  {center.simulation.assets.anomaly && (
                    <a
                      href={center.simulation.assets.anomaly}
                      target="_blank"
                      rel="noreferrer"
                      className="col-span-2 block overflow-hidden rounded-xl border border-white/10 bg-black/30"
                    >
                      <img
                        src={center.simulation.assets.anomaly}
                        alt="Temperature anomaly"
                        className="h-32 w-full object-cover"
                      />
                      <div className="px-2 py-1 text-[10px] text-slate-400">Anomaly</div>
                    </a>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {!center.simulation && (
          <div className="mt-4 rounded-2xl border border-dashed border-white/10 p-4 text-sm text-slate-400">
            Run simulation to generate heat GIF, final field, ΔT, max
            temperature, and optimal load.
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function LoadControlPanel({
  totalBase,
  totalOptimal,
  hasDirtyCenters,
  newLoad,
  setNewLoad,
  bayesianLoops,
  setBayesianLoops,
  isRunning,
  centersLength,
  runSimulation,
  resetSimulationKeepCenters,
  report,
  setShowReport,
  handleLoadReportFile,
  simulationError,
  bayesEvents,
  optimizationProgress,
  lastOptimizationBundle,
  onDownloadOptimizationHtml,
  onDownloadOptimizationJson,
}) {
  return (
    <Card className="sticky top-6 border-white/10 bg-white/[0.04] shadow-2xl shadow-black/30 backdrop-blur-xl">
      <CardContent className="p-5">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-bold text-white">
              Master Load Control
            </h2>
            <p className="text-xs text-slate-400">
              Global redistribution input and simulation controls.
            </p>
          </div>
          <Zap className="h-5 w-5 text-orange-300" />
        </div>

        {hasDirtyCenters && (
          <div className="mb-4 flex items-center gap-2 rounded-2xl border border-orange-300/20 bg-orange-400/10 p-3 text-xs text-orange-100">
            <AlertTriangle className="h-4 w-4 shrink-0" />
            One or more baseline loads changed. Rerun simulation.
          </div>
        )}

        {simulationError && (
          <div className="mb-4 max-h-40 overflow-y-auto rounded-2xl border border-red-400/30 bg-red-500/10 p-3 text-xs text-red-100">
            <div className="mb-1 font-semibold text-red-200">Simulation server</div>
            <pre className="whitespace-pre-wrap font-mono text-[11px] text-red-100/90">
              {simulationError}
            </pre>
            <div className="mt-2 text-[11px] text-red-200/80">
              Start the API:{" "}
              <span className="text-white">
                uvicorn simulation_api_server:app --host 127.0.0.1 --port 8765
              </span>
              . With <span className="text-white">npm run dev</span>,{" "}
              <span className="text-white">/api</span> is proxied to that port
              (set <span className="text-white">VITE_SIM_API_BASE</span> only if
              the API runs elsewhere).
            </div>
          </div>
        )}

        {(isRunning || bayesEvents.length > 0) && (
          <div className="mb-4 rounded-2xl border border-cyan-400/20 bg-slate-950/70 p-3">
            <div className="mb-2 flex items-center justify-between text-xs text-cyan-200">
              <span className="font-semibold">Bayes optimizer log</span>
              {optimizationProgress?.stepsRemaining != null && (
                <span className="font-mono text-[11px] text-slate-400">
                  ~{optimizationProgress.stepsRemaining} steps left
                </span>
              )}
            </div>
            {optimizationProgress?.message && (
              <div className="mb-2 text-[11px] text-slate-400">
                {optimizationProgress.phase ? `[${optimizationProgress.phase}] ` : ""}
                {optimizationProgress.message}
              </div>
            )}
            <div className="max-h-40 overflow-y-auto font-mono text-[10px] leading-relaxed text-slate-300">
              {bayesEvents
                .filter(
                  (ev) =>
                    ev.type !== "log" || !AGENT_PHASES.has(ev.phase)
                )
                .slice(-40)
                .map((ev, i) => (
                <div key={i} className="border-b border-white/5 py-1 last:border-0">
                  {ev.type === "log" && <span className="text-cyan-300/90">log</span>}
                  {ev.type === "progress" && (
                    <span className="text-orange-300/90">progress</span>
                  )}
                  {ev.type === "plan" && <span className="text-emerald-300/90">plan</span>}
                  {": "}
                  {ev.message ||
                    (ev.phase === "global"
                      ? `Candidate ${ev.index}/${ev.total}`
                      : ev.phase === "refine"
                      ? `Refine ${ev.index}/${ev.total}`
                      : JSON.stringify(ev))}
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="grid grid-cols-2 gap-3">
          <Metric
            icon={Cpu}
            label="Total base"
            value={totalBase.toFixed(1)}
            unit="MW"
          />
          <Metric
            icon={Activity}
            label="Optimal total"
            value={totalOptimal.toFixed(1)}
            unit="MW"
          />
        </div>

        <div className="mt-4 rounded-2xl border border-white/10 bg-slate-950/60 p-4">
          <div className="mb-2 flex items-center justify-between text-sm">
            <span className="font-medium text-slate-200">
              New load to redistribute
            </span>
            <span className="font-bold text-orange-200">{newLoad} MW</span>
          </div>

          <input
            type="range"
            min="0"
            max="160"
            value={newLoad}
            onChange={(e) => setNewLoad(Number(e.target.value))}
            className="w-full accent-orange-400"
          />

          <input
            type="number"
            value={newLoad}
            onChange={(e) => setNewLoad(Number(e.target.value))}
            className="mt-3 w-full rounded-xl border border-white/10 bg-black/30 px-3 py-2 text-sm text-white outline-none focus:border-orange-300/60"
          />
        </div>

        <div className="mt-4 rounded-2xl border border-white/10 bg-slate-950/60 p-4">
          <div className="mb-2 flex items-center justify-between text-sm">
            <span className="font-medium text-slate-200">
              Bayesian candidate runs
            </span>
            <span className="font-mono font-bold text-cyan-200">{bayesianLoops}</span>
          </div>
          <input
            type="range"
            min={4}
            max={48}
            value={bayesianLoops}
            onChange={(e) => setBayesianLoops(Number(e.target.value))}
            className="w-full accent-cyan-400"
          />
          <p className="mt-2 text-[11px] leading-snug text-slate-500">
            Number of global load splits evaluated before picking seeds for local
            refinement. Higher is slower but explores more combinations.
          </p>
        </div>

        <div className="mt-4 grid grid-cols-1 gap-3">
          <Button
            className="rounded-2xl bg-orange-400 text-slate-950 hover:bg-orange-300"
            onClick={runSimulation}
            disabled={isRunning || !centersLength}
          >
            {isRunning ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <Play className="mr-2 h-4 w-4" />
            )}
            Run simulation
          </Button>

          <Button
            variant="outline"
            className="rounded-2xl"
            onClick={resetSimulationKeepCenters}
          >
            Reset results
          </Button>
        </div>

        <div className="mt-3 grid grid-cols-1 gap-3">
          <Button
            variant="outline"
            disabled={!report}
            onClick={() => setShowReport(true)}
          >
            <Eye className="mr-2 h-4 w-4" />
            View report
          </Button>

          <Button
            variant="outline"
            disabled={!lastOptimizationBundle}
            onClick={onDownloadOptimizationHtml}
          >
            <Download className="mr-2 h-4 w-4" />
            Download HTML report
          </Button>

          <Button
            variant="outline"
            disabled={!lastOptimizationBundle}
            onClick={onDownloadOptimizationJson}
          >
            <Download className="mr-2 h-4 w-4" />
            Download optimal JSON
          </Button>

          <Button
            variant="outline"
            disabled={!report}
            onClick={() => downloadJson("thermal_load_report.json", report)}
          >
            <Download className="mr-2 h-4 w-4" />
            Download summary JSON
          </Button>

          <label className="inline-flex cursor-pointer items-center justify-center rounded-md border border-white/10 bg-white/5 px-4 py-2 text-sm font-medium text-slate-200 transition hover:bg-white/10 hover:text-white">
            <FileText className="mr-2 h-4 w-4" />
            Load report
            <input
              type="file"
              accept="application/json"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                handleLoadReportFile(file);
                e.target.value = "";
              }}
            />
          </label>
        </div>
      </CardContent>
    </Card>
  );
}

function ReportModal({ report, onClose, apiBase = "" }) {
  const opt = report.optimization || {};
  const chartUrls = opt.chartUrls || opt.chart_urls || {};
  const chartSrc = (key) =>
    chartUrls[key] ? `${apiBase}${chartUrls[key]}` : null;
  const runId = opt.runId || opt.run_id;
  const reportHtmlUrl = opt.reportHtmlUrl || opt.report_html_url || "";
  const [panelTab, setPanelTab] = useState("summary");

  useEffect(() => {
    setPanelTab("summary");
  }, [report.generatedAt]);

  const CHART_ORDER = [
    "improvement_trace",
    "gp_surrogate",
    "loads",
    "load_heatmap",
    "delta_mw",
    "site_scores",
    "refinement",
  ];
  const CHART_LABELS = {
    improvement_trace: "Search progress (objective vs eval #)",
    gp_surrogate: "GP surrogate vs observations (±σ)",
    loads: "MW allocation (baseline, optimal, spread)",
    load_heatmap: "Best trials: MW split heatmap",
    delta_mw: "Δ MW per site (optimal − baseline)",
    site_scores: "Per-site diagnostic score at optimum",
    refinement: "Local refinement (before → after)",
  };
  const chartKeys = [
    ...CHART_ORDER.filter((k) => chartUrls[k]),
    ...Object.keys(chartUrls).filter(
      (k) => !CHART_ORDER.includes(k) && chartUrls[k]
    ),
  ];

  const fullReportSrc =
    reportHtmlUrl &&
    (reportHtmlUrl.startsWith("http")
      ? reportHtmlUrl
      : `${apiBase}${reportHtmlUrl}`);

  return (
    <motion.div
      role="presentation"
      className="fixed inset-0 z-[9999] flex flex-col bg-slate-950"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <motion.div
        role="dialog"
        aria-modal="true"
        className="flex h-full min-h-0 w-full flex-col overflow-hidden border border-white/10 bg-slate-950 shadow-2xl"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.2 }}
        onClick={(e) => e.stopPropagation()}
      >
        <motion.div className="flex shrink-0 flex-col gap-3 border-b border-white/10 px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-xl font-bold text-white">
              <BarChart3 className="h-5 w-5 shrink-0 text-cyan-300" />
              Simulation report
            </div>
            <div className="mt-1 text-xs text-slate-500">
              {report.generatedAt}
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-end gap-2">
            {reportHtmlUrl ? (
              <div className="mr-auto flex rounded-lg border border-white/10 bg-black/35 p-0.5 sm:mr-0">
                <button
                  type="button"
                  className={`rounded-md px-3 py-1.5 text-xs font-medium transition ${
                    panelTab === "full"
                      ? "bg-cyan-500/25 text-cyan-100"
                      : "text-slate-400 hover:text-white"
                  }`}
                  onClick={() => setPanelTab("full")}
                >
                  Full HTML report
                </button>
                <button
                  type="button"
                  className={`rounded-md px-3 py-1.5 text-xs font-medium transition ${
                    panelTab === "summary"
                      ? "bg-cyan-500/25 text-cyan-100"
                      : "text-slate-400 hover:text-white"
                  }`}
                  onClick={() => setPanelTab("summary")}
                >
                  Summary panel
                </button>
              </div>
            ) : null}
            <Button
              type="button"
              variant="outline"
              onClick={() => downloadJson("thermal_load_report.json", report)}
            >
              <Download className="mr-2 h-4 w-4" />
              JSON
            </Button>
            <Button type="button" variant="ghost" onClick={onClose}>
              <X className="mr-2 h-4 w-4" />
              Close
            </Button>
          </div>
        </motion.div>

        {panelTab === "full" && fullReportSrc ? (
          <iframe
            title="Optimization HTML report"
            src={fullReportSrc}
            className="min-h-0 w-full flex-1 border-0 bg-[#0b1220]"
          />
        ) : (
          <motion.div className="min-h-0 flex-1 overflow-y-auto p-5 md:p-8">
            <div className="mb-6 grid grid-cols-1 gap-3 md:grid-cols-3">
              <Metric
                icon={Zap}
                label="New load to place"
                value={report.newLoadToRedistributeMW}
                unit="MW"
              />
              <Metric
                icon={Cpu}
                label="Total base"
                value={report.totalBaseLoadMW}
                unit="MW"
              />
              <Metric
                icon={Activity}
                label="Total optimal"
                value={report.totalOptimalLoadMW}
                unit="MW"
              />
            </div>

            {(opt.agentHistory || opt.agent_history)?.length > 0 && (
              <motion.div className="mb-8 rounded-2xl border border-violet-500/25 bg-violet-500/5 p-5">
                <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-violet-200">
                  Local agent review
                </h3>
                <div className="space-y-2 text-xs text-slate-300">
                  {(opt.agentHistory || opt.agent_history).map((round, ri) => (
                    <motion.div key={ri} className="rounded-xl border border-white/10 bg-black/25 p-3">
                      <div className="mb-2 font-medium text-violet-200">
                        Round {round.round + 1}
                        {round.all_accepted ? " · all accepted" : " · adjustments requested"}
                      </div>
                      <ul className="space-y-1">
                        {(round.reviews || []).map((r) => (
                          <li key={r.site_index}>
                            <span
                              className={
                                r.verdict === "accept"
                                  ? "text-emerald-300"
                                  : "text-amber-300"
                              }
                            >
                              {r.site_name}: {r.verdict}
                            </span>
                            {r.reasons?.[0] ? ` — ${r.reasons[0]}` : ""}
                          </li>
                        ))}
                      </ul>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            )}

            {runId && chartKeys.length > 0 && (
              <div className="mb-8 rounded-2xl border border-cyan-500/20 bg-cyan-500/5 p-5">
                <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-cyan-200">
                  Optimization charts
                </h3>
                <p className="mb-4 text-xs text-slate-400">
                  Run <span className="font-mono text-cyan-300">{runId}</span>
                  {(opt.bayesianLoopCount ?? opt.bayesian_loop_count) != null && (
                    <>
                      {" "}
                      · {opt.bayesianLoopCount ?? opt.bayesian_loop_count} global
                      candidates
                    </>
                  )}
                  {(opt.topKRefine ?? opt.top_k_refine) != null && (
                    <> · top {opt.topKRefine ?? opt.top_k_refine} refined</>
                  )}
                  {opt.bestObjective != null && (
                    <>
                      {" "}
                      · best objective (full physics){" "}
                      <span className="text-white">
                        {Number(
                          opt.bestObjective ?? opt.best_objective
                        ).toFixed(4)}
                      </span>
                    </>
                  )}
                </p>
                <div className="grid gap-4 md:grid-cols-2">
                  {chartKeys.map((key) => {
                    const src = chartSrc(key);
                    if (!src) return null;
                    const label =
                      CHART_LABELS[key] ||
                      key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
                    const wide =
                      key === "refinement" ||
                      key === "improvement_trace" ||
                      key === "gp_surrogate";
                    return (
                      <div
                        key={key}
                        className={`overflow-hidden rounded-xl border border-white/10 bg-black/30 ${
                          wide ? "md:col-span-2" : ""
                        }`}
                      >
                        <div className="px-3 py-2 text-[11px] font-medium text-slate-400">
                          {label}
                        </div>
                        <img
                          src={src}
                          alt={label}
                          className={
                            wide
                              ? "max-h-96 w-full object-contain"
                              : "w-full object-contain"
                          }
                        />
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            <h3 className="mb-3 text-sm font-semibold text-slate-300">Sites</h3>
            <div className="space-y-4">
              {report.centers.map((center) => (
                <div
                  key={center.id}
                  className="rounded-2xl border border-white/10 bg-white/[0.04] p-4"
                >
                  <div className="mb-3 flex flex-wrap items-start justify-between gap-2">
                    <div>
                      <div className="font-semibold text-white">{center.name}</div>
                      <div className="text-xs text-slate-500">
                        {center.location.lat.toFixed(3)},{" "}
                        {center.location.lon.toFixed(3)}
                      </div>
                    </div>
                    <motion.div className="flex flex-wrap gap-2">
                      {center.agentVerdict && (
                        <div
                          className={`rounded-full px-3 py-1 text-xs ${
                            center.agentVerdict === "accept"
                              ? "bg-emerald-500/15 text-emerald-200"
                              : "bg-amber-500/15 text-amber-200"
                          }`}
                        >
                          Agent: {center.agentVerdict}
                        </div>
                      )}
                      <div
                        className={`rounded-full px-3 py-1 text-xs ${
                          center.simulation?.risk === "High"
                            ? "bg-red-500/15 text-red-200"
                            : center.simulation?.risk === "Medium"
                            ? "bg-orange-500/15 text-orange-200"
                            : "bg-emerald-500/15 text-emerald-200"
                        }`}
                      >
                        {center.simulation?.risk || "Not run"}
                      </div>
                    </motion.div>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-xs md:grid-cols-5">
                    <div className="rounded-xl bg-black/25 p-2">
                      <div className="text-slate-500">Base</div>
                      <div className="font-bold text-white">{center.baseLoadMW} MW</div>
                    </div>
                    <div className="rounded-xl bg-black/25 p-2">
                      <div className="text-slate-500">Optimal</div>
                      <div className="font-bold text-orange-200">
                        {center.optimalLoadMW} MW
                      </div>
                    </div>
                    <div className="rounded-xl bg-black/25 p-2">
                      <div className="text-slate-500">ΔT</div>
                      <div className="font-bold text-white">
                        {center.simulation?.deltaT ?? "—"} °C
                      </div>
                    </div>
                    <div className="rounded-xl bg-black/25 p-2">
                      <div className="text-slate-500">Max temp</div>
                      <div className="font-bold text-white">
                        {center.simulation?.maxTemp ?? "—"} °C
                      </div>
                    </div>
                    <div className="rounded-xl bg-black/25 p-2">
                      <div className="text-slate-500">Extra</div>
                      <div className="font-bold text-cyan-200">
                        {center.simulation?.assignedExtra ?? "—"} MW
                      </div>
                    </div>
                  </div>

                  {center.simulation?.assets && !center.simulation?.error && (
                    <div className="mt-4 grid grid-cols-2 gap-2 md:grid-cols-4">
                      {["masks", "final", "anomaly"].map((k) => {
                        const u = center.simulation.assets[k];
                        if (!u) return null;
                        const src = u.startsWith("http") ? u : `${apiBase}${u}`;
                        return (
                          <a
                            key={k}
                            href={src}
                            target="_blank"
                            rel="noreferrer"
                            className="block overflow-hidden rounded-xl border border-white/10 bg-black/40"
                          >
                            <img
                              src={src}
                              alt={k}
                              className="h-24 w-full object-cover"
                            />
                            <div className="px-2 py-1 text-[10px] capitalize text-slate-400">
                              {k === "final" ? "Final T" : k}
                            </div>
                          </a>
                        );
                      })}
                      {center.simulation.assets.gif && (
                        <div className="col-span-2 overflow-hidden rounded-xl border border-white/10 bg-black/40 md:col-span-4">
                          <div className="px-2 py-1 text-[10px] text-slate-400">
                            Heat GIF
                          </div>
                          <img
                            src={
                              center.simulation.assets.gif.startsWith("http")
                                ? center.simulation.assets.gif
                                : `${apiBase}${center.simulation.assets.gif}`
                            }
                            alt="gif"
                            className="max-h-48 w-full object-contain"
                          />
                        </div>
                      )}
                    </div>
                  )}

                  <div className="mt-2 text-xs text-slate-500">
                    {center.simulation?.status || "—"}
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </motion.div>
    </motion.div>
  );
}

/* -------------------------------------------------------
   Main app
------------------------------------------------------- */

export default function App() {
  const [centers, setCenters] = useState(INITIAL_CENTERS);
  const [selectedId, setSelectedId] = useState("dc-toronto");
  const [newLoad, setNewLoadValue] = useState(36);
  const [uploadIndex, setUploadIndex] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [report, setReport] = useState(null);
  const [showReport, setShowReport] = useState(false);
  const [simulationError, setSimulationError] = useState(null);
  const [bayesEvents, setBayesEvents] = useState([]);
  const [agentHistory, setAgentHistory] = useState([]);
  const [optimizationProgress, setOptimizationProgress] = useState(null);
  const [lastOptimizationBundle, setLastOptimizationBundle] = useState(null);
  const [bayesianLoops, setBayesianLoops] = useState(12);
  const [pdfUploadBusy, setPdfUploadBusy] = useState(false);

  const selectedCenter = centers.find((c) => c.id === selectedId) || null;

  const totalBase = useMemo(
    () => centers.reduce((sum, c) => sum + Number(c.baseLoad || 0), 0),
    [centers]
  );

  const totalOptimal = useMemo(
    () => centers.reduce((sum, c) => sum + Number(c.optimalLoad || 0), 0),
    [centers]
  );

  const maxLoad = useMemo(() => {
    const values = centers.flatMap((c) => [
      Number(c.baseLoad || 0),
      Number(c.optimalLoad || 0),
    ]);
    return Math.max(150, ...values);
  }, [centers]);

  const hasDirtyCenters = centers.some((c) => c.dirty);

  function setNewLoad(value) {
    setNewLoadValue(value);
    setReport(null);
    setLastOptimizationBundle(null);
  }

  function addDemoDataCenter(fileName = "uploaded_specification.pdf") {
    const parsed = demoParsePdf(fileName, uploadIndex);
    const id = `dc-${Date.now()}-${uploadIndex}`;

    const center = {
      id,
      ...parsed,
      optimalLoad: parsed.baseLoad,
      simulation: null,
      dirty: false,
      pdfResources: null,
      wallPhysics: { ...DEFAULT_WALL_PHYSICS },
    };

    setCenters((prev) => [...prev, center]);
    setSelectedId(id);
    setUploadIndex((i) => i + 1);
    setReport(null);
    setLastOptimizationBundle(null);
  }

  function updateWallPhysics(id, wallPhysics) {
    setCenters((prev) =>
      prev.map((c) =>
        c.id === id ? { ...c, wallPhysics, dirty: true, simulation: null } : c
      )
    );
    setReport(null);
    setLastOptimizationBundle(null);
  }

  async function addDataCenterFromPdf(file) {
    if (!file) return;
    setPdfUploadBusy(true);
    setSimulationError(null);
    try {
      const url = `${SIM_API_BASE}/api/pdf-extract`;
      console.info("[pdf-extract] POST", url, "file=", file.name, file.size, "bytes");
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(url, {
        method: "POST",
        body: fd,
      });
      console.info("[pdf-extract] response", res.status, res.statusText);
      const data = await res.json().catch(() => ({}));
      console.info("[pdf-extract] body", {
        ok: data.ok,
        error: data.error,
        warnings: data.warnings,
      });
      if (!res.ok || data.ok === false) {
        throw new Error(data.error || data.message || `HTTP ${res.status}`);
      }
      const specs = data.specs || {};
      const loc = data.location || {};
      const lat = parseFloat(loc.latitude);
      const lon = parseFloat(loc.longitude);
      const name =
        specs.data_center_name ||
        file.name.replace(/\.pdf$/i, "").replace(/[_-]/g, " ") ||
        "Imported DC";
      let baseLoadMw = 55;
      if (specs.maximum_power_load_kw != null) {
        const kw = Number(specs.maximum_power_load_kw);
        if (Number.isFinite(kw)) {
          baseLoadMw = clamp(kw / 1000, 5, 150);
        }
      }
      const id = `dc-${Date.now()}-${uploadIndex}`;
      const wp = defaultWallPhysicsFromExtract(data);
      const center = {
        id,
        name,
        lat: Number.isFinite(lat) ? lat : 43.6532,
        lon: Number.isFinite(lon) ? lon : -79.3832,
        baseLoad: baseLoadMw,
        optimalLoad: baseLoadMw,
        weather: {
          temp: 22,
          humidity: 55,
          solar: 600,
          windSpeed: 4,
          windDirection: "NE",
        },
        image: IMAGE_PLACEHOLDERS[uploadIndex % IMAGE_PLACEHOLDERS.length],
        simulation: null,
        dirty: false,
        pdfResources: {
          specs,
          thermal_capacities: data.thermal_capacities || [],
          warnings: data.warnings || [],
        },
        wallPhysics:
          wp || {
            material: "Concrete (built-in)",
            specific_heat_kj_per_kg_k: 0.88,
            density_kg_m3: 2400,
          },
      };
      setCenters((prev) => [...prev, center]);
      setSelectedId(id);
      setUploadIndex((i) => i + 1);
      setReport(null);
      setLastOptimizationBundle(null);
    } catch (e) {
      setSimulationError(e?.message || String(e));
    } finally {
      setPdfUploadBusy(false);
    }
  }

  function removeCenter(id) {
    const remaining = centers.filter((c) => c.id !== id);

    setCenters(remaining);

    setSelectedId((current) => {
      if (current !== id) return current;
      return remaining[0]?.id || null;
    });

    setReport(null);
    setLastOptimizationBundle(null);
  }

  function updateBaseLoad(id, value) {
    setCenters((prev) =>
      prev.map((c) =>
        c.id === id
          ? {
              ...c,
              baseLoad: Number(value),

              /*
                Important change:
                optimalLoad is NOT modified here anymore.
                It updates only after running simulation.
              */
              simulation: null,
              dirty: true,
            }
          : c
      )
    );

    setSelectedId(id);
    setReport(null);
    setLastOptimizationBundle(null);
  }

  function resetSimulationKeepCenters() {
    setCenters((prev) =>
      prev.map((c) => ({
        ...c,
        optimalLoad: c.baseLoad,
        simulation: null,
        dirty: false,
      }))
    );

    setReport(null);
    setSimulationError(null);
    setBayesEvents([]);
    setAgentHistory([]);
    setOptimizationProgress(null);
    setLastOptimizationBundle(null);
  }

  async function onDownloadOptimizationHtml() {
    if (!lastOptimizationBundle) return;
    try {
      const url = `${SIM_API_BASE}${lastOptimizationBundle.reportHtmlUrl}`;
      await downloadBlobFromUrl(
        url,
        `thermal_optimization_${lastOptimizationBundle.runId}.html`
      );
    } catch (e) {
      setSimulationError(e?.message || String(e));
    }
  }

  async function onDownloadOptimizationJson() {
    if (!lastOptimizationBundle) return;
    try {
      const url = `${SIM_API_BASE}${lastOptimizationBundle.optimalJsonUrl}`;
      await downloadBlobFromUrl(
        url,
        `optimal_data_${lastOptimizationBundle.runId}.json`
      );
    } catch (e) {
      setSimulationError(e?.message || String(e));
    }
  }

  async function runSimulation() {
    if (!centers.length) return;

    setIsRunning(true);
    setSimulationError(null);
    setBayesEvents([]);
    setAgentHistory([]);
    setOptimizationProgress(null);
    setLastOptimizationBundle(null);

    const sitesPayload = centers.map((center) => {
      const row = {
        id: center.id,
        name: center.name,
        lat: center.lat,
        lon: center.lon,
        base_load_mw: Number(center.baseLoad),
        weather: {
          temp: center.weather.temp,
          humidity: center.weather.humidity,
          solar: center.weather.solar,
          windSpeed: center.weather.windSpeed,
          windDirection: center.weather.windDirection,
        },
      };
      const wp = center.wallPhysics;
      if (
        wp &&
        wp.specific_heat_kj_per_kg_k != null &&
        Number.isFinite(Number(wp.specific_heat_kj_per_kg_k))
      ) {
        const physics = {
          material: wp.material ?? null,
          specific_heat_kj_per_kg_k: Number(wp.specific_heat_kj_per_kg_k),
        };
        if (
          wp.density_kg_m3 != null &&
          Number.isFinite(Number(wp.density_kg_m3))
        ) {
          physics.density_kg_m3 = Number(wp.density_kg_m3);
        }
        row.physics = physics;
      }
      return row;
    });

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 1_800_000);

    try {
      let res;
      try {
        res = await fetch(`${SIM_API_BASE}/api/optimize-run`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sites: sitesPayload,
            extra_total_load_mw: Number(newLoad),
            bayesian_loop_count: Math.min(80, Math.max(4, Number(bayesianLoops) || 12)),
            top_k_refine: 2,
            agent_max_delta_t_with_concerns_c: 9.5,
          }),
          signal: controller.signal,
        });
      } finally {
        clearTimeout(timeoutId);
      }

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }

      if (!res.body) {
        throw new Error("No response body (streaming).");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let completeData = null;
      const logs = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n");
        buffer = parts.pop() ?? "";
        for (const line of parts) {
          if (!line.trim()) continue;
          let ev;
          try {
            ev = JSON.parse(line);
          } catch {
            continue;
          }
          if (ev.type === "error") {
            throw new Error(ev.message || "Optimization failed");
          }
          if (ev.type === "complete") {
            completeData = ev.data;
            continue;
          }
          logs.push(ev);
          setBayesEvents([...logs]);
          if (ev.steps_remaining != null || ev.message) {
            setOptimizationProgress({
              stepsRemaining: ev.steps_remaining,
              message: ev.message || "",
              phase: ev.phase || "",
            });
          }
        }
      }

      if (buffer.trim()) {
        try {
          const ev = JSON.parse(buffer.trim());
          if (ev.type === "complete") completeData = ev.data;
          else if (ev.type === "error") throw new Error(ev.message);
        } catch {
          /* ignore trailing partial */
        }
      }

      if (!completeData?.sites?.length) {
        throw new Error("Optimization finished without site results.");
      }

      const t = Date.now();
      const runId = completeData.run_id;
      const byId = Object.fromEntries(completeData.sites.map((s) => [s.id, s]));

      const updated = centers.map((center) => {
        const s = byId[center.id];
        if (!s) {
          return { ...center, dirty: false };
        }
        return {
          ...center,
          optimalLoad: Number(Number(s.optimal_load_mw).toFixed(2)),
          simulation: buildSimulationFromBayesSite(s, t, runId),
          agentVerdict: s.agent_verdict ?? null,
          dirty: false,
        };
      });

      const bundle = {
        runId,
        reportHtmlUrl: completeData.report_html_url,
        optimalJsonUrl: completeData.optimal_json_url,
        bestObjective: completeData.best_objective,
      };
      setLastOptimizationBundle(bundle);

      const optMeta = {
        runId: completeData.run_id,
        bestObjective: completeData.best_objective,
        reportHtmlUrl: completeData.report_html_url,
        optimalJsonUrl: completeData.optimal_json_url,
        chartUrls: completeData.chart_urls || {},
        globalSearch: completeData.global_slim,
        refinement: completeData.refined_slim,
        bayesianLoopCount: completeData.bayesian_loop_count,
        topKRefine: completeData.top_k_refine,
        agentHistory: completeData.agent_history || [],
      };

      setAgentHistory(completeData.agent_history || []);

      const generated = generateReport(updated, newLoad, optMeta);
      setCenters(updated);
      setReport(generated);
    } catch (e) {
      const msg =
        e?.name === "AbortError"
          ? "Request timed out (Bayes search + full runs can take many minutes)."
          : e?.message || String(e);
      setSimulationError(msg);
    } finally {
      setIsRunning(false);
    }
  }

  function handleLoadReportFile(file) {
    if (!file) return;

    const reader = new FileReader();

    reader.onload = (event) => {
      try {
        const parsed = JSON.parse(event.target.result);
        setReport(parsed);
        setShowReport(true);
      } catch {
        alert("Could not parse report JSON.");
      }
    };

    reader.readAsText(file);
  }

  return (
    <div className="min-h-screen bg-[#020617] p-6 text-white">
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -left-32 -top-32 h-96 w-96 rounded-full bg-cyan-500/15 blur-3xl" />
        <div className="absolute right-0 top-1/4 h-[32rem] w-[32rem] rounded-full bg-orange-500/10 blur-3xl" />
        <div className="absolute bottom-0 left-1/3 h-96 w-96 rounded-full bg-emerald-500/10 blur-3xl" />
      </div>

      <div className="relative mx-auto max-w-[1900px] space-y-6">
        <header>
          <Card className="border-white/10 bg-white/[0.04] shadow-2xl shadow-black/30 backdrop-blur-xl">
            <CardContent className="p-5">
              <div className="flex flex-col justify-between gap-5 lg:flex-row lg:items-center">
                <div className="flex items-center gap-4">
                  <div className="rounded-2xl bg-cyan-400/10 p-3">
                    <Gauge className="h-7 w-7 text-cyan-300" />
                  </div>

                  <div>
                    <h1 className="text-3xl font-black tracking-tight text-white">
                      Data Center Thermal Orchestrator
                    </h1>
                    <p className="mt-1 text-sm text-slate-400">
                      Bayes load split (streaming logs) · full physics + GIFs on
                      the best MW plan · HTML / JSON exports
                    </p>
                  </div>
                </div>

                <div className="flex flex-wrap gap-3">
                  <label
                    className={`inline-flex cursor-pointer items-center justify-center rounded-md bg-cyan-400 px-4 py-2 text-sm font-medium text-slate-950 transition hover:bg-cyan-300 ${
                      pdfUploadBusy ? "pointer-events-none opacity-70" : ""
                    }`}
                  >
                    {pdfUploadBusy ? (
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    ) : (
                      <Upload className="mr-2 h-4 w-4" />
                    )}
                    {pdfUploadBusy ? "Extracting PDF…" : "Upload PDF spec"}
                    <input
                      type="file"
                      accept="application/pdf"
                      className="hidden"
                      disabled={pdfUploadBusy}
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) addDataCenterFromPdf(file);
                        e.target.value = "";
                      }}
                    />
                  </label>

                  <Button variant="outline" onClick={() => addDemoDataCenter()}>
                    <MapPin className="mr-2 h-4 w-4" />
                    Add demo site
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </header>

        <section className="grid grid-cols-1 gap-6 xl:grid-cols-[340px_1fr_430px]">
          <aside>
            <LoadControlPanel
              totalBase={totalBase}
              totalOptimal={totalOptimal}
              hasDirtyCenters={hasDirtyCenters}
              newLoad={newLoad}
              setNewLoad={setNewLoad}
              bayesianLoops={bayesianLoops}
              setBayesianLoops={setBayesianLoops}
              isRunning={isRunning}
              centersLength={centers.length}
              runSimulation={runSimulation}
              resetSimulationKeepCenters={resetSimulationKeepCenters}
              report={report}
              setShowReport={setShowReport}
              handleLoadReportFile={handleLoadReportFile}
              simulationError={simulationError}
              bayesEvents={bayesEvents}
              optimizationProgress={optimizationProgress}
              lastOptimizationBundle={lastOptimizationBundle}
              onDownloadOptimizationHtml={onDownloadOptimizationHtml}
              onDownloadOptimizationJson={onDownloadOptimizationJson}
            />
          </aside>

          <main className="space-y-5">
            <MinimalWorldMap
              centers={centers}
              selectedId={selectedId}
              setSelectedId={setSelectedId}
            />

            <Card className="border-white/10 bg-white/[0.04] shadow-2xl shadow-black/30 backdrop-blur-xl">
              <CardContent className="p-5">
                <div className="mb-4 flex items-center justify-between">
                  <div>
                    <h2 className="text-lg font-bold text-white">
                      Data Center Load Frames
                    </h2>
                    <p className="text-xs text-slate-400">
                      Click any frame to open full info. Adjust base load
                      directly here.
                    </p>
                  </div>
                  <Server className="h-5 w-5 text-cyan-300" />
                </div>

                <div className="flex gap-4 overflow-x-auto pb-2">
                  {centers.map((center) => (
                    <DataCenterFrame
                      key={center.id}
                      center={center}
                      selected={selectedId === center.id}
                      maxLoad={maxLoad}
                      onSelect={() => setSelectedId(center.id)}
                      onRemove={() => removeCenter(center.id)}
                      onBaseLoadChange={(value) =>
                        updateBaseLoad(center.id, value)
                      }
                    />
                  ))}

                  {!centers.length && (
                    <div className="w-full rounded-3xl border border-dashed border-white/10 p-8 text-center text-sm text-slate-500">
                      No data centers added. Upload a PDF spec to place a new
                      pin on the map.
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            <AgentConversationPanel
              bayesEvents={bayesEvents}
              agentHistory={agentHistory}
              centers={centers}
              isRunning={isRunning}
            />
          </main>

          <aside>
            <SelectedCenterPanel
              center={selectedCenter}
              onClose={() => setSelectedId(null)}
              onWallPhysicsChange={updateWallPhysics}
            />
          </aside>
        </section>
      </div>

      <AnimatePresence>
        {showReport && report && (
          <ReportModal
            key="thermal-report"
            report={report}
            apiBase={SIM_API_BASE}
            onClose={() => setShowReport(false)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}