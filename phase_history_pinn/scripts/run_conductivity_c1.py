"""Run gated C1 powder-conductivity continuation."""

from __future__ import annotations

import argparse

from phase_history_pinn.ph_pinn.continuation import run_conductivity_continuation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial-checkpoint",
        default="phase_history_pinn/runs/curriculum_top_C1/model_B3.pt",
    )
    parser.add_argument("--output-name")
    args = parser.parse_args()
    print(run_conductivity_continuation(args.initial_checkpoint, args.output_name))


if __name__ == "__main__":
    main()
