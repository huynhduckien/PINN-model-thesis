"""Gated conductivity continuation from the accepted curriculum B3 state."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .config import corrective_config, derive_scales
from .evaluate import cycle_metrics, snapshot_metrics, thermal_balance_metrics
from .history import HistoryField
from .model import TemperaturePINN
from .run import RUNS
from .trainer import train_stage


STAGES = (
    ("K1", 1.0, 3000, 5e-5, 1e-5),
    ("K03", 0.3, 3000, 5e-5, 1e-5),
    ("K01", 0.1, 3000, 3e-5, 5e-6),
    ("K005", 0.05, 3000, 2e-5, 5e-6),
    ("KM", None, 5000, 2e-5, 2e-6),
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_conductivity_continuation(
    initial_checkpoint: str | Path,
    output_name: str | None = None,
) -> Path:
    base = corrective_config()
    sc = derive_scales(base, "normal")
    np.random.seed(base.training.seed)
    torch.manual_seed(base.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    initial_checkpoint = Path(initial_checkpoint).resolve()
    initial = torch.load(initial_checkpoint, map_location=device, weights_only=True)
    model = TemperaturePINN(base, sc).to(device)
    model.load_state_dict(initial["state_dict"])
    history = HistoryField.zeros(base, sc)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = (RUNS / (output_name or f"conductivity_C1_{stamp}")).resolve()
    if RUNS.resolve() not in out.parents:
        raise ValueError("output must remain under phase_history_pinn/runs")
    out.mkdir(parents=True, exist_ok=False)
    _write_json(
        out / "manifest.json",
        {
            "label": "GATED_CONDUCTIVITY_CONTINUATION_C1",
            "initial_checkpoint": str(initial_checkpoint),
            "moser_ratio": sc.k_powder_ratio,
            "base_config": base.to_dict(),
            "stages": [
                {
                    "name": name,
                    "powder_k_ratio": sc.k_powder_ratio if ratio is None else ratio,
                    "steps": steps,
                    "learning_rate": lr,
                    "final_learning_rate": final_lr,
                }
                for name, ratio, steps, lr, final_lr in STAGES
            ],
        },
    )

    records: list[dict[str, object]] = []
    started = time.perf_counter()
    for index, (name, ratio, steps, lr, final_lr) in enumerate(STAGES):
        actual_ratio = sc.k_powder_ratio if ratio is None else ratio
        physics = replace(
            base.physics,
            powder_conductivity=True,
            powder_k_ratio_override=actual_ratio,
        )
        training = replace(
            base.training,
            adam_steps_per_outer=steps,
            learning_rate=lr,
            final_learning_rate=final_lr,
            weight_energy=1.0,
            energy_warmup_steps=0,
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
                "powder_k_ratio": actual_ratio,
            },
            checkpoint,
        )
        records.append(
            {
                "stage": name,
                "powder_k_ratio": actual_ratio,
                "training": result.to_dict(),
                "thermal_gate": gate,
                "checkpoint": checkpoint.name,
            }
        )
        _write_json(out / "progress.json", records)
        if not gate["passed"]:
            break

    last_ratio = records[-1]["powder_k_ratio"]
    last_cfg = replace(
        base,
        physics=replace(base.physics, powder_k_ratio_override=last_ratio),
    )
    reached_moser = bool(
        len(records) == len(STAGES)
        and abs(last_ratio - sc.k_powder_ratio) < 1e-12
    )
    summary = {
        "label": "GATED_CONDUCTIVITY_CONTINUATION_C1",
        "elapsed_seconds": time.perf_counter() - started,
        "stages_completed": len(records),
        "stopped_after": records[-1]["stage"],
        "thermal_gate_passed": records[-1]["thermal_gate"]["passed"],
        "reached_moser_ratio": reached_moser,
        "records": records,
        "indicators_at_last_stage": {
            "snapshot": snapshot_metrics(model, last_cfg, sc, device),
            "cycle": cycle_metrics(model, last_cfg, sc, device),
        },
        "verified_solution": False,
    }
    _write_json(out / "summary.json", summary)
    return out
