"""Staged C1 optimizer qualification with the final physics unchanged."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .config import PhysicsOptions, corrective_config, derive_scales
from .evaluate import cycle_metrics, snapshot_metrics, thermal_balance_metrics
from .history import HistoryField
from .model import TemperaturePINN
from .physics import moser_properties, source_power_error
from .run import RUNS
from .trainer import train_stage


STAGES = (
    ("A", PhysicsOptions(False, False, False, False), 3000, 3e-4, 1e-4, 0.1),
    ("B1", PhysicsOptions(False, True, False, False), 2000, 1e-4, 5e-5, 0.5),
    ("B2", PhysicsOptions(False, True, True, False), 2000, 1e-4, 5e-5, 0.5),
    ("B3", PhysicsOptions(False, True, True, True), 2000, 1e-4, 5e-5, 0.5),
    ("B4", PhysicsOptions(True, True, True, True), 5000, 1e-4, 2e-5, 1.0),
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_curriculum_c1(output_name: str | None = None) -> Path:
    base = corrective_config()
    sc = derive_scales(base, "normal")
    np.random.seed(base.training.seed)
    torch.manual_seed(base.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = (RUNS / (output_name or f"curriculum_C1_{stamp}")).resolve()
    if RUNS.resolve() not in out.parents:
        raise ValueError("output must remain under phase_history_pinn/runs")
    out.mkdir(parents=True, exist_ok=False)

    model = TemperaturePINN(base, sc).to(device)
    history = HistoryField.zeros(base, sc)
    records: list[dict[str, object]] = []
    started = time.perf_counter()
    manifest = {
        "label": "CURRICULUM_PILOT_C1",
        "device": str(device),
        "base_config": base.to_dict(),
        "scales": vars(sc),
        "moser": moser_properties(base),
        "source_check": source_power_error(base, sc),
        "stages": [
            {
                "name": name,
                "physics": vars(physics),
                "steps": steps,
                "learning_rate": lr,
                "final_learning_rate": final_lr,
                "weight_energy": energy_weight,
            }
            for name, physics, steps, lr, final_lr, energy_weight in STAGES
        ],
    }
    _write_json(out / "manifest.json", manifest)

    for index, (name, physics, steps, lr, final_lr, energy_weight) in enumerate(STAGES):
        training = replace(
            base.training,
            adam_steps_per_outer=steps,
            learning_rate=lr,
            final_learning_rate=final_lr,
            weight_energy=energy_weight,
            energy_warmup_steps=min(500, steps // 4),
            energy_ramp_steps=max(steps // 2, 1),
            report_every=250,
        )
        cfg = replace(base, physics=physics, training=training)
        result = train_stage(model, history, cfg, sc, device, index)
        gate = thermal_balance_metrics(model, history, cfg, sc, device)
        checkpoint = out / f"model_{name}.pt"
        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "stage": name,
                "physics": vars(physics),
            },
            checkpoint,
        )
        record = {
            "stage": name,
            "physics": vars(physics),
            "training": result.to_dict(),
            "thermal_gate": gate,
            "checkpoint": checkpoint.name,
        }
        records.append(record)
        _write_json(out / "progress.json", records)

    final_gate = records[-1]["thermal_gate"]
    summary = {
        "label": "CURRICULUM_PILOT_C1",
        "elapsed_seconds": time.perf_counter() - started,
        "thermal_gate_passed": final_gate["passed"],
        "stages": records,
        "indicators": {
            "snapshot": snapshot_metrics(model, base, sc, device),
            "cycle": cycle_metrics(model, base, sc, device),
        },
        "verified_solution": False,
    }
    _write_json(out / "summary.json", summary)
    return out
