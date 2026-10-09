"""CLI for an isolated phase-history PINN case."""

from __future__ import annotations

import argparse

from phase_history_pinn.ph_pinn.config import (
    corrective_config,
    default_config,
    smoke_config,
)
from phase_history_pinn.ph_pinn.run import run_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset", choices=("smoke", "default", "corrective"), default="smoke"
    )
    parser.add_argument("--geometry", choices=("normal", "wedge"), required=True)
    parser.add_argument("--mode", choices=("static", "history"), required=True)
    parser.add_argument("--output-name")
    args = parser.parse_args()
    configs = {
        "smoke": smoke_config,
        "default": default_config,
        "corrective": corrective_config,
    }
    cfg = configs[args.preset]()
    out = run_case(cfg, args.geometry, args.mode, args.preset, args.output_name)
    print(out)


if __name__ == "__main__":
    main()

