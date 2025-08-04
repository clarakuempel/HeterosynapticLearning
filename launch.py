import subprocess
import os
from textwrap import dedent

# --- Config ---
SWEEP_CONFIG = "grid"
PROJECT = f"test-gpt-{SWEEP_CONFIG}"

# Optimization parameter grids
lr_schdl = ["cosine", "steplr", "None"]
optimizers = ["md"]

# --- Launch Loop ---
for sch in lr_schdl:
    for opt in optimizers:
        study_name = f"study_{sch}_{opt}"
        group = f"{sch}_{opt}"
        name = f"{sch}_{opt}"

        print(f"\nLaunching local sweep: {name}")
        
        setup_msg = dedent(f"""\
            === Running: {name} ===
            Project: {PROJECT}
            Study: {study_name}
            Group: {group}
        """)
        print(setup_msg)

        cmd = [
            "python", "src/train.py", "-m",
            f"optimizer.update_alg={opt}",
            f"optimizer.lr_scheduler={sch}",
            f"hydra.sweeper.study_name={study_name}",
            f"logger.group={group}",
            f"hparams_search={SWEEP_CONFIG}",
            f"logger.project={PROJECT}",
        ]

        print("Running command:", " ".join(cmd))
        subprocess.run(cmd)
