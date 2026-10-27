"""Run the complete reproducible workflow in order.

    python -m src.run_all             # data -> features -> EDA -> training/evaluation -> figures
    python -m src.run_all --offline   # use cached raw data only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

STEPS = [
    ("Data pipeline", ["-m", "src.data_pipeline"]),
    ("EDA figures", ["-m", "src.eda"]),
    ("Training, tuning, ablation, evaluation", ["-m", "src.train"]),
    ("Report figures", ["-m", "src.visualizations"]),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    for name, cmd in STEPS:
        if name == "Data pipeline":
            cmd = cmd + (["--offline"] if args.offline else []) + (["--refresh"] if args.refresh else [])
        print(f"\n=== {name} ===", flush=True)
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, *cmd])
        if r.returncode != 0:
            sys.exit(f"Step '{name}' failed with exit code {r.returncode}")
        print(f"=== {name} done in {time.perf_counter() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
