"""Run the whole pipeline in LITE mode on a local CPU, headless.

Executes notebooks 01-04 in order with POLLINATOR_LITE=1 and saves the
executed copies (with all outputs and figures) to outputs_lite/executed/.
Stops at the first failing notebook and prints which one failed.

Usage (from the repo root):
    python scripts/run_lite.py            # all four notebooks
    python scripts/run_lite.py 02 03      # only some (earlier outputs must exist)
    python scripts/run_lite.py --clean    # delete data_lite/ and outputs_lite/ first

Requires: pip install -r requirements.txt   (tensorflow, jupyter, ...)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("0[1-4]_*.ipynb"))
OUT = ROOT / "outputs_lite" / "executed"


def main(argv: list[str]) -> int:
    if "--clean" in argv:
        for d in (ROOT / "data_lite", ROOT / "outputs_lite"):
            shutil.rmtree(d, ignore_errors=True)
            print("removed", d)
        argv = [a for a in argv if a != "--clean"]
    selected = [nb for nb in NOTEBOOKS if not argv or nb.name[:2] in argv]
    if not selected:
        print("no notebooks matched", argv)
        return 1

    env = dict(os.environ, POLLINATOR_LITE="1", TF_CPP_MIN_LOG_LEVEL="2", MPLBACKEND="Agg")
    OUT.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    for nb in selected:
        print(f"\n=== {nb.name} (lite) ===", flush=True)
        t0 = time.time()
        cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
               "--ExecutePreprocessor.timeout=-1", "--output-dir", str(OUT), str(nb)]
        result = subprocess.run(cmd, cwd=nb.parent, env=env)
        if result.returncode != 0:
            print(f"\nFAILED: {nb.name}. Open {OUT / nb.name} (if written) or re-run it in Jupyter "
                  f"with LITE_MODE = True to see the failing cell.")
            return result.returncode
        print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)

    print(f"\nAll {len(selected)} notebook(s) passed in lite mode in {(time.time() - t_all) / 60:.1f} min.")
    print(f"Executed notebooks: {OUT}")
    print(f"Figures:            {ROOT / 'outputs_lite' / 'figures'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
