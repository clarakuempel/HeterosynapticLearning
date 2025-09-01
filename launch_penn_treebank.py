import subprocess
import os
from textwrap import dedent

# --- Config ---
PROJECT = "penn-treebank-test"

print(f"\nLaunching Penn Treebank training")

setup_msg = dedent(f"""\
    === Running Penn Treebank Language Modeling ===
    Project: {PROJECT}
    Task: Language modeling on Penn Treebank dataset
    Model: nanoGPT
""")
print(setup_msg)

cmd = [
    "python", "src/train.py",
    "data=penn_treebank",
    "model=nanoGPT",
    "optimizer.update_alg=gd",  # Test GD with stronger coupling
    "optimizer.alpha=0.5",  # Stronger heterosynaptic coupling
    "optimizer.lr=0.03",  # Higher learning rate to amplify differences
    "trainer.max_epochs=2",  # Two epochs
    f"logger.project={PROJECT}",
    "logger.group=debug_gd_lr03",
    "seed=43"
]

print("Running command:", " ".join(cmd))
subprocess.run(cmd)