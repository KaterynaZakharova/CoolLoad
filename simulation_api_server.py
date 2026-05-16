"""
HTTP API: Bayes load optimization (streaming) + optional direct physics batch.

Run from repo root:

    pip install -r requirements-api.txt
    uvicorn simulation_api_server:app --host 127.0.0.1 --port 8765 --reload
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from bayes_optimizer import (
    OUTPUT_ROOT as OPT_OUTPUT_ROOT,
    build_per_site_bounds,
    clear_load_optimization_output_dir,
    collect_final_run_asset_paths,
    compute_single_datacenter_objective,
    compute_total_objective,
    optimize_datacenter_loads_iter,
    project_loads_to_bounds,
    rerun_best_with_full_outputs,
    write_optimization_report_bundle,
)
from pdf_extractor import extract_pdf_resources_bundle
from run_one_physical_simulation import run_physical_simulation_for_params

ROOT = Path(__file__).resolve().parent
SIM_OUTPUT_ROOT = ROOT / "simulation_outputs"


class WeatherIn(BaseModel):
    temp: float
    humidity: float
    solar: float
    windSpeed: float
    windDirection: str = "N"


class PhysicsWallIn(BaseModel):
    """Opaque wall + slab volumetric heat capacity via rho * Cp (Cp in kJ/kg·K)."""

    material: str | None = None
    specific_heat_kj_per_kg_k: float | None = None
    density_kg_m3: float | None = None


class SiteOpt(BaseModel):
    id: str
    name: str
    lat: float
    lon: float
    base_load_mw: float = Field(..., description="Current site IT load in MW")
    weather: WeatherIn
    physics: PhysicsWallIn | None = None


class OptimizeRequest(BaseModel):
    sites: list[SiteOpt]
    extra_total_load_mw: float = Field(
        ...,
        description="Total additional MW to spread across sites (Bayes search).",
    )
    bayesian_loop_count: int = Field(
        12,
        ge=4,
        le=80,
        description="Number of global load-split candidates (Dirichlet + structured seeds).",
    )
    top_k_refine: int = Field(
        2,
        ge=1,
        le=8,
        description="How many best global candidates receive local coordinate refinement.",
    )


class SiteIn(BaseModel):
    id: str
    lat: float
    lon: float
    load_mw: float = Field(
        ...,
        description="IT load in MW for this site (e.g. base + redistributed extra).",
    )
    weather: WeatherIn
    physics: PhysicsWallIn | None = None


class BatchRequest(BaseModel):
    sites: list[SiteIn]


def safe_site_subdir(site_id: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", site_id).strip("_")
    return s or "site"


def _json_line(obj: Any) -> bytes:
    return (json.dumps(obj, default=str) + "\n").encode("utf-8")


def _physics_wall_to_dc_keys(p: PhysicsWallIn | None) -> dict[str, Any]:
    if p is None:
        return {}
    out: dict[str, Any] = {}
    if p.material is not None:
        out["wall_material"] = str(p.material)
    if p.specific_heat_kj_per_kg_k is not None:
        out["wall_specific_heat_kj_per_kg_k"] = float(p.specific_heat_kj_per_kg_k)
    if p.density_kg_m3 is not None:
        out["wall_density_kg_m3"] = float(p.density_kg_m3)
    return out


def _site_to_dc(site: SiteOpt) -> dict[str, Any]:
    return {
        "name": site.name,
        "lat": site.lat,
        "lon": site.lon,
        "temp_c": float(site.weather.temp),
        "humidity": float(site.weather.humidity),
        "solar_wm2": float(site.weather.solar),
        "wind_speed_m_s": float(site.weather.windSpeed),
        "wind_direction": str(site.weather.windDirection),
        **_physics_wall_to_dc_keys(site.physics),
    }


app = FastAPI(title="DC thermal + Bayes optimization")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SIM_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
OPT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=str(SIM_OUTPUT_ROOT)), name="outputs")
app.mount("/opt-out", StaticFiles(directory=str(OPT_OUTPUT_ROOT)), name="opt_out")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/pdf-extract")
def pdf_extract(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload a PDF; return specs, thermal capacity rows, optional geocode, warnings."""
    name = (file.filename or "").lower()
    if not name.endswith(".pdf"):
        return {"ok": False, "error": "Expected a .pdf file.", "specs": {}, "thermal_capacities": [], "location": None, "warnings": []}

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)
    finally:
        try:
            file.file.close()
        except Exception:
            pass

    try:
        if not tmp_path:
            return {"ok": False, "error": "Could not store upload.", "specs": {}, "thermal_capacities": [], "location": None, "warnings": []}
        sz = Path(tmp_path).stat().st_size
        try: 
            out = extract_pdf_resources_bundle(tmp_path, try_geocode=True)
            return out
        except Exception as exc:
            return {
                "ok": False,
                "error": str(exc),
                "specs": {},
                "thermal_capacities": [],
                "location": None,
                "warnings": [traceback.format_exc()],
            }
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def _optimize_stream(req: OptimizeRequest) -> Iterator[bytes]:
    dcs = [_site_to_dc(s) for s in req.sites]
    n = len(dcs)
    if n == 0:
        yield _json_line({"type": "error", "message": "No sites."})
        return

    base_loads = np.array([float(s.base_load_mw) for s in req.sites], dtype=float)
    extra = float(req.extra_total_load_mw)
    min_l, max_l = build_per_site_bounds(
        base_loads_mw=base_loads,
        extra_total_load_mw=extra,
        min_floor_mw=5.0,
        max_cap_mw=150.0,
    )

    run_id = f"opt_{uuid.uuid4().hex[:14]}"

    try:
        yield _json_line(
            {
                "type": "log",
                "phase": "init",
                "message": "Clearing previous results in load_optimization_outputs/ …",
            }
        )
        clear_load_optimization_output_dir()

        n_loops = int(req.bayesian_loop_count)
        top_k = int(req.top_k_refine)

        for event in optimize_datacenter_loads_iter(
            dcs,
            base_loads_mw=base_loads,
            extra_total_load_mw=extra,
            min_loads_mw=min_l,
            max_loads_mw=max_l,
            run_id=run_id,
            n_global_candidates=n_loops,
            top_k_refine=top_k,
        ):
            if event.get("type") == "complete":
                sites_out: list[dict[str, Any]] = []
                full = event["result"]
                best_loads = np.asarray(full["best_loads_mw"], dtype=float)
                target_total_mw = float(base_loads.sum() + extra)
                best_loads = project_loads_to_bounds(
                    np.maximum(best_loads, base_loads),
                    min_l,
                    max_l,
                    target_total_mw,
                )
                yield _json_line(
                    {
                        "type": "log",
                        "phase": "finalize",
                        "message": "Best split found. Running full physics + plots + GIFs for the optimal MW allocation…",
                    }
                )
                final_results = rerun_best_with_full_outputs(
                    datacenters=dcs,
                    best_loads_mw=best_loads,
                    run_id=run_id,
                    save_gif=True,
                    verbose=False,
                )
                obj_final = float(compute_total_objective(final_results))
                assets = collect_final_run_asset_paths(run_id, dcs, final_results)
                run_root = OPT_OUTPUT_ROOT / run_id
                _, _, chart_paths = write_optimization_report_bundle(
                    run_root=run_root,
                    run_id=run_id,
                    datacenters=dcs,
                    base_loads_mw=base_loads,
                    extra_total_load_mw=extra,
                    global_slim=full["global_slim"],
                    refined_slim=full["refined_slim"],
                    best_loads_mw=best_loads,
                    best_objective=obj_final,
                    best_results=final_results,
                    final_asset_rel=assets,
                )

                scratch = OPT_OUTPUT_ROOT / run_id / "_scratch_eval"
                if scratch.is_dir():
                    shutil.rmtree(scratch, ignore_errors=True)

                chart_urls = {
                    k: f"/opt-out/{run_id}/{fn}"
                    for k, fn in (chart_paths or {}).items()
                    if fn
                }
                for i, site in enumerate(req.sites):
                    od = Path(final_results[i]["output_dir"])
                    rel_to_opt = od.relative_to(OPT_OUTPUT_ROOT).as_posix()
                    base = f"/opt-out/{rel_to_opt}".replace("\\", "/")
                    m = final_results[i]["metrics"]
                    sites_out.append(
                        {
                            "id": site.id,
                            "name": site.name,
                            "base_load_mw": float(base_loads[i]),
                            "optimal_load_mw": float(best_loads[i]),
                            "assigned_extra_mw": float(best_loads[i] - base_loads[i]),
                            "site_objective": float(compute_single_datacenter_objective(m)),
                            "metrics": m,
                            "assets": {
                                "masks": f"{base}/01_building_masks.png",
                                "final": f"{base}/02_final_temperature.png",
                                "anomaly": f"{base}/03_final_anomaly.png",
                                **(
                                    {"gif": f"{base}/04_heat_plume_animation.gif"}
                                    if (od / "04_heat_plume_animation.gif").is_file()
                                    else {}
                                ),
                            },
                        }
                    )

                yield _json_line(
                    {
                        "type": "complete",
                        "data": {
                            "run_id": run_id,
                            "extra_total_load_mw": extra,
                            "best_objective": obj_final,
                            "report_html_url": f"/opt-out/{run_id}/report.html",
                            "optimal_json_url": f"/opt-out/{run_id}/optimal_data.json",
                            "sites": sites_out,
                            "global_slim": full["global_slim"],
                            "refined_slim": full["refined_slim"],
                            "chart_urls": chart_urls,
                            "bayesian_loop_count": n_loops,
                            "top_k_refine": top_k,
                        },
                    }
                )
            else:
                yield _json_line(event)
    except Exception as exc:
        yield _json_line(
            {
                "type": "error",
                "message": str(exc),
                "trace": traceback.format_exc(),
            }
        )


@app.post("/api/optimize-run")
def optimize_run(req: OptimizeRequest) -> StreamingResponse:
    return StreamingResponse(
        _optimize_stream(req),
        media_type="application/x-ndjson",
    )


@app.post("/api/simulate-batch")
def simulate_batch(req: BatchRequest) -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    for site in req.sites:
        sub = safe_site_subdir(site.id)
        out_dir = SIM_OUTPUT_ROOT / sub
        sim_extras: dict[str, Any] = {}
        if site.physics is not None:
            p = site.physics
            if p.specific_heat_kj_per_kg_k is not None:
                sim_extras["wall_specific_heat_kj_per_kg_k"] = float(p.specific_heat_kj_per_kg_k)
            if p.density_kg_m3 is not None:
                sim_extras["wall_density_kg_m3"] = float(p.density_kg_m3)
            if p.material is not None:
                sim_extras["wall_material"] = str(p.material)

        try:
            run_physical_simulation_for_params(
                lat=site.lat,
                lon=site.lon,
                load_mw=site.load_mw,
                temp_c=site.weather.temp,
                humidity=site.weather.humidity,
                solar_wm2=site.weather.solar,
                wind_speed_m_s=site.weather.windSpeed,
                wind_direction=site.weather.windDirection,
                output_dir=out_dir,
                verbose=False,
                **sim_extras,
            )

            base = f"/outputs/{sub}"
            gif_path = out_dir / "04_heat_plume_animation.gif"
            assets = {
                "masks": f"{base}/01_building_masks.png",
                "final": f"{base}/02_final_temperature.png",
                "anomaly": f"{base}/03_final_anomaly.png",
            }
            if gif_path.is_file():
                assets["gif"] = f"{base}/04_heat_plume_animation.gif"

            with open(out_dir / "metrics.json", "r", encoding="utf-8") as f:
                metrics = json.load(f)

            results.append(
                {
                    "id": site.id,
                    "ok": True,
                    "error": None,
                    "metrics": metrics,
                    "assets": assets,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "id": site.id,
                    "ok": False,
                    "error": f"{exc}\n{traceback.format_exc()}",
                    "metrics": None,
                    "assets": None,
                }
            )

    return {"results": results}
