import subprocess
import os
from textwrap import dedent

# --- Config ---
SWEEP_CONFIG = "grid"
PROJECT = f"test-hydra-sweeps-{SWEEP_CONFIG}"

# Optimization parameter grids
corruption_types = ["identity", "full_dense", "block_diagonal"]
optimizers = ["md"]

# --- Launch Loop ---
for corruption in corruption_types:
    for opt in optimizers:
        study_name = f"study_{corruption}_{opt}"
        group = f"{corruption}_{opt}"
        name = f"{corruption}_{opt}"

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
            f"corruption.corruption_type={corruption}",
            f"optimizer.update_alg={opt}",
            f"hydra.sweeper.study_name={study_name}",
            f"logger.group={group}",
            f"hparams_search={SWEEP_CONFIG}",
            f"logger.project={PROJECT}",
        ]

        print("Running command:", " ".join(cmd))
        subprocess.run(cmd)
        break
    break
