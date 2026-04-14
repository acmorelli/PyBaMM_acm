"""Launch all radius sweep simulations (can be run as a batch)."""
import subprocess
import sys

RADII = [30, 50, 100, 300, 500, 1000]
MODES = ["charge", "discharge"]
MODELS = [[], ["core_shell"]]  # baseline, then core-shell

for R_nm in RADII:
    for mode in MODES:
        for extra in MODELS:
            tag = "core_shell" if extra else "baseline"
            args = [sys.executable, "run_radius_sweep.py", str(R_nm), mode] + extra
            print(f"\n{'='*60}")
            print(f"  Running: R={R_nm}nm  {mode}  {tag}")
            print(f"{'='*60}")
            result = subprocess.run(args, cwd=".")
            if result.returncode != 0:
                print(f"  *** FAILED (exit {result.returncode}) ***")
