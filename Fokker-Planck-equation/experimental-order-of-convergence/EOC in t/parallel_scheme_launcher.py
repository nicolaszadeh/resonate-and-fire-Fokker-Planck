# -*- coding: utf-8 -*-

import subprocess
import os
import sys
from pathlib import Path
from datetime import datetime

base_dir = Path(__file__).resolve().parent

scripts = [
    base_dir / "semi_implicit_upwind.py",
    base_dir / "explicit_upwind.py",
    base_dir / "explicit_centered.py",
    base_dir / "semi_implicit_centered.py",
]

env = os.environ.copy()

# avoid oversubscription
env["OMP_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"
env["OPENBLAS_NUM_THREADS"] = "1"
env["NUMEXPR_NUM_THREADS"] = "1"

# isolate Python from user-site / foreign Python installs
env["PYTHONNOUSERSITE"] = "1"
env.pop("PYTHONPATH", None)
env.pop("PYTHONHOME", None)

log_dir = base_dir / "parallel_logs"
log_dir.mkdir(exist_ok=True)

procs = []
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

for script_path in scripts:
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    log_file = log_dir / f"{script_path.stem}_{timestamp}.log"
    f = open(log_file, "w", encoding="utf-8")

    print(f"Launching: {script_path.name}")

    p = subprocess.Popen(
        [sys.executable, "-I", str(script_path)],
        cwd=str(base_dir),
        env=env,
        stdout=f,
        stderr=subprocess.STDOUT,
    )

    procs.append((script_path.name, p, f, log_file))

print("\nAll jobs launched.\n")

for name, p, f, log_file in procs:
    code = p.wait()
    f.close()
    print(f"{name} finished with return code {code}")
    print(f"Log: {log_file}\n")

print("Done.")