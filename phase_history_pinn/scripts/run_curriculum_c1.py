"""Run the preregistered C1 physics curriculum."""

from __future__ import annotations

import argparse

from phase_history_pinn.ph_pinn.curriculum import run_curriculum_c1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name")
    args = parser.parse_args()
    print(run_curriculum_c1(args.output_name))


if __name__ == "__main__":
    main()
