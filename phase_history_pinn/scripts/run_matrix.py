"""Run the preregistered C1-C4 matrix sequentially."""

from __future__ import annotations

import argparse
import json
import time

from phase_history_pinn.ph_pinn.config import corrective_config, default_config, smoke_config
from phase_history_pinn.ph_pinn.run import RUNS, run_case


CASES = (
    ("C1", "normal", "static"),
    ("C2", "wedge", "static"),
    ("C3", "normal", "history"),
    ("C4", "wedge", "history"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset", choices=("smoke", "default", "corrective"), default="smoke"
    )
    args = parser.parse_args()
    configs = {
        "smoke": smoke_config,
        "default": default_config,
        "corrective": corrective_config,
    }
    cfg = configs[args.preset]()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    index = {"preset": args.preset, "created": stamp, "cases": {}}
    for label, geometry, mode in CASES:
        out = run_case(
            cfg,
            geometry,
            mode,
            args.preset,
            output_name=f"{args.preset}_{stamp}_{label}_{geometry}_{mode}",
        )
        index["cases"][label] = str(out.relative_to(RUNS))
        print(f"{label}: {out}", flush=True)
    index_path = RUNS / f"matrix_{args.preset}_{stamp}.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"matrix index: {index_path}")


if __name__ == "__main__":
    main()

