"""Orchestrate one isolated normal/wedge, static/history run."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import RunConfig, derive_scales
from .evaluate import cycle_metrics, snapshot_metrics, thermal_balance_metrics
from .history import HistoryField
from .model import TemperaturePINN
from .physics import moser_properties, source_power_error
from .trainer import train_stage


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_case(
    cfg: RunConfig,
    geometry: str,
    mode: str,
    preset: str,
    output_name: str | None = None,
) -> Path:
    if mode not in {"static", "history"}:
        raise ValueError("mode must be 'static' or 'history'")
    sc = derive_scales(cfg, geometry)
    _seed_everything(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = output_name or f"{preset}_{geometry}_{mode}_{stamp}"
    out = (RUNS / name).resolve()
    if RUNS.resolve() not in out.parents:
        raise ValueError("output must remain under phase_history_pinn/runs")
    out.mkdir(parents=True, exist_ok=False)

    manifest = {
        "label": "SMOKE_ONLY" if preset == "smoke" else "SCIENTIFIC_RUN",
        "preset": preset,
        "geometry": geometry,
        "mode": mode,
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "git_commit": _git_commit(),
        "config": cfg.to_dict(),
        "scales": asdict(sc),
        "moser": moser_properties(cfg),
        "source_check": source_power_error(cfg, sc),
    }
    _write_json(out / "manifest.json", manifest)

    model = TemperaturePINN(cfg, sc).to(device)
    history = HistoryField.zeros(cfg, sc)
    outer_records: list[dict[str, object]] = []
    max_outer = 1 if mode == "static" else cfg.training.outer_iterations
    started = time.perf_counter()

    for outer in range(max_outer):
        result = train_stage(model, history, cfg, sc, device, outer)
        checkpoint = out / f"model_outer_{outer:02d}.pt"
        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "geometry": geometry,
                "mode": mode,
                "outer": outer,
            },
            checkpoint,
        )
        record: dict[str, object] = {
            "outer": outer,
            "training": result.to_dict(),
            "checkpoint": checkpoint.name,
        }
        thermal_gate = thermal_balance_metrics(model, history, cfg, sc, device)
        record["thermal_gate"] = thermal_gate
        if mode == "history":
            if cfg.training.enforce_thermal_gate and not thermal_gate["passed"]:
                record["history_update_skipped"] = True
                outer_records.append(record)
                _write_json(out / "progress.json", outer_records)
                break
            updated, diagnostics = history.update_from_model(model, cfg, sc, device)
            history_path = out / f"history_outer_{outer:02d}.npz"
            updated.save(history_path)
            record["history"] = diagnostics.to_dict()
            record["history_file"] = history_path.name
            history = updated
            outer_records.append(record)
            _write_json(out / "progress.json", outer_records)
            if diagnostics.converged and outer > 0:
                break
        else:
            history.save(out / "history_static_zero.npz")
            outer_records.append(record)
            _write_json(out / "progress.json", outer_records)

    indicators = {
        "snapshot": snapshot_metrics(model, cfg, sc, device),
        "cycle": cycle_metrics(model, cfg, sc, device),
    }
    summary = {
        "label": manifest["label"],
        "geometry": geometry,
        "mode": mode,
        "outer_completed": len(outer_records),
        "elapsed_seconds": time.perf_counter() - started,
        "indicators_are_model_internal": True,
        "verified_solution": False,
        "thermal_gate_passed": bool(
            outer_records and outer_records[-1]["thermal_gate"]["passed"]
        ),
        "outer": outer_records,
        "indicators": indicators,
    }
    _write_json(out / "summary.json", summary)
    return out

