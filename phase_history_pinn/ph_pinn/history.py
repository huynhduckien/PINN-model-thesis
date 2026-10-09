"""Irreversible four-dimensional phase-history field and outer update."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import RunConfig, Scales


def _axis(lo: float, hi: float, spacing: float, minimum: int = 2) -> np.ndarray:
    n = max(minimum, int(np.ceil((hi - lo) / spacing)) + 1)
    return np.linspace(lo, hi, n, dtype=np.float64)


@dataclass
class HistoryDiagnostics:
    max_abs_change: float
    changed_fraction: float
    consolidated_fraction_old: float
    consolidated_fraction_new: float
    consolidated_fraction_change: float
    monotone_time: bool
    monotone_outer: bool
    converged: bool

    def to_dict(self) -> dict[str, float | bool]:
        return vars(self).copy()


class HistoryField:
    """M(t,x,y,z) in [0,1], interpolated linearly at PINN points.

    The tensor is frozen during a training stage. Index selection is discrete,
    while interpolation weights retain gradients with respect to coordinates;
    this supplies the piecewise-linear coefficient gradients needed by the
    divergence operator.
    """

    def __init__(
        self,
        t_axis: np.ndarray,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        z_axis: np.ndarray,
        values: np.ndarray,
    ) -> None:
        expected = (len(t_axis), len(x_axis), len(y_axis), len(z_axis))
        if values.shape != expected:
            raise ValueError(f"history shape {values.shape} != {expected}")
        self.t_axis = np.asarray(t_axis, dtype=np.float64)
        self.x_axis = np.asarray(x_axis, dtype=np.float64)
        self.y_axis = np.asarray(y_axis, dtype=np.float64)
        self.z_axis = np.asarray(z_axis, dtype=np.float64)
        self.values = np.asarray(values, dtype=np.float32)
        self._cache: dict[tuple[str, torch.dtype], tuple[list[torch.Tensor], torch.Tensor]] = {}
        self.validate()

    @classmethod
    def zeros(cls, cfg: RunConfig, sc: Scales) -> "HistoryField":
        hr = cfg.history
        axes = (
            np.linspace(0.0, sc.t_end_s, hr.n_time, dtype=np.float64),
            _axis(-sc.lx_s / 2, sc.lx_s / 2, hr.dx / sc.lc),
            _axis(-sc.ly_s / 2, sc.ly_s / 2, hr.dy / sc.lc),
            _axis(0.0, sc.lz_s, hr.dz / sc.lc),
        )
        values = np.zeros(tuple(len(a) for a in axes), dtype=np.float32)
        return cls(*axes, values)

    def clone(self) -> "HistoryField":
        return HistoryField(
            self.t_axis.copy(),
            self.x_axis.copy(),
            self.y_axis.copy(),
            self.z_axis.copy(),
            self.values.copy(),
        )

    def validate(self) -> None:
        if not np.isfinite(self.values).all():
            raise ValueError("history contains non-finite values")
        if self.values.min() < -1e-7 or self.values.max() > 1 + 1e-7:
            raise ValueError("history is outside [0,1]")
        if np.any(np.diff(self.values, axis=0) < -1e-6):
            raise ValueError("history decreases in time")

    def _torch_data(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        key = (str(x.device), x.dtype)
        if key not in self._cache:
            axes = [
                torch.as_tensor(a, dtype=x.dtype, device=x.device)
                for a in (self.t_axis, self.x_axis, self.y_axis, self.z_axis)
            ]
            values = torch.as_tensor(self.values, dtype=x.dtype, device=x.device)
            self._cache[key] = (axes, values)
        return self._cache[key]

    @staticmethod
    def _bracket(axis: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.searchsorted(axis, q.contiguous(), right=True) - 1
        idx = idx.clamp(0, axis.numel() - 2)
        lo, hi = axis[idx], axis[idx + 1]
        frac = ((q - lo) / (hi - lo)).clamp(0.0, 1.0)
        return idx, frac

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """Return M for scaled PINN coordinates [x,y,z,t]."""
        axes, values = self._torch_data(x)
        queries = (x[:, 3], x[:, 0], x[:, 1], x[:, 2])
        brackets = [self._bracket(a, q) for a, q in zip(axes, queries)]
        out = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for bt in (0, 1):
            for bx in (0, 1):
                for by in (0, 1):
                    for bz in (0, 1):
                        bits = (bt, bx, by, bz)
                        inds = [i + b for (i, _), b in zip(brackets, bits)]
                        weight = torch.ones_like(out)
                        for (_, f), b in zip(brackets, bits):
                            weight = weight * (f if b else 1.0 - f)
                        out = out + weight * values[inds[0], inds[1], inds[2], inds[3]]
        return out[:, None]

    def update_from_model(
        self,
        model: torch.nn.Module,
        cfg: RunConfig,
        sc: Scales,
        device: torch.device,
    ) -> tuple["HistoryField", HistoryDiagnostics]:
        """Evaluate T on the grid and apply cumulative liquid-fraction history."""
        from .model import temperature_from_theta
        from .physics import liquid_fraction, powder_index

        xx, yy, zz = np.meshgrid(
            self.x_axis, self.y_axis, self.z_axis, indexing="ij"
        )
        spatial = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        powder_parts: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(spatial), cfg.history.evaluation_chunk):
                xyz = spatial[start : start + cfg.history.evaluation_chunk]
                pts = np.column_stack([xyz, np.zeros(len(xyz), dtype=np.float64)])
                xt = torch.as_tensor(pts, dtype=torch.float32, device=device)
                powder_parts.append(powder_index(xt, cfg, sc).cpu().numpy()[:, 0])
        powder_weight = np.concatenate(powder_parts).astype(np.float32)
        new = np.empty_like(self.values)
        running = np.zeros(spatial.shape[0], dtype=np.float32)
        model.eval()
        with torch.no_grad():
            for it, time_value in enumerate(self.t_axis):
                frac_parts: list[np.ndarray] = []
                for start in range(0, len(spatial), cfg.history.evaluation_chunk):
                    xyz = spatial[start : start + cfg.history.evaluation_chunk]
                    pts = np.column_stack(
                        [xyz, np.full(len(xyz), time_value, dtype=np.float64)]
                    )
                    xt = torch.as_tensor(pts, dtype=torch.float32, device=device)
                    temp = temperature_from_theta(model(xt), cfg)
                    frac_parts.append(liquid_fraction(temp, cfg).cpu().numpy()[:, 0])
                # M records thermal phase history everywhere. Conductivity uses
                # it only through s*(1-M), so original solid is unaffected and
                # the smooth powder interface is not applied twice.
                instant = np.concatenate(frac_parts).astype(np.float32)
                running = np.maximum(running, instant)
                old_slice = self.values[it].reshape(-1)
                new[it] = np.maximum(old_slice, running).reshape(new.shape[1:])

        new = np.maximum.accumulate(new, axis=0)
        updated = HistoryField(
            self.t_axis, self.x_axis, self.y_axis, self.z_axis, new
        )
        delta = updated.values - self.values
        active = np.broadcast_to(powder_weight.reshape(1, *new.shape[1:]) >= 0.5, new.shape)
        old_fraction = float(np.mean(self.values[active] >= 0.5))
        new_fraction = float(np.mean(updated.values[active] >= 0.5))
        max_abs = float(np.max(np.abs(delta[active])))
        changed = float(np.mean(np.abs(delta[active]) > 0.05))
        fraction_change = new_fraction - old_fraction
        diagnostics = HistoryDiagnostics(
            max_abs_change=max_abs,
            changed_fraction=changed,
            consolidated_fraction_old=old_fraction,
            consolidated_fraction_new=new_fraction,
            consolidated_fraction_change=float(fraction_change),
            monotone_time=bool(np.all(np.diff(updated.values, axis=0) >= -1e-6)),
            monotone_outer=bool(np.all(delta >= -1e-6)),
            converged=bool(
                max_abs <= 0.02
                and changed <= 0.05
                and abs(fraction_change) <= 0.02
            ),
        )
        return updated, diagnostics

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            t_axis=self.t_axis,
            x_axis=self.x_axis,
            y_axis=self.y_axis,
            z_axis=self.z_axis,
            values=self.values,
        )

    @classmethod
    def load(cls, path: str | Path) -> "HistoryField":
        with np.load(path) as data:
            return cls(
                data["t_axis"],
                data["x_axis"],
                data["y_axis"],
                data["z_axis"],
                data["values"],
            )

