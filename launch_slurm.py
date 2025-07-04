import subprocess
from textwrap import dedent
import os
CONDA_ENV_NAME = "HL-env"
PROJECT = "test-hydra-sweeps"
REPO_DIR = os.path.abspath(".")  # adjust if needed
SWEEP_CONFIG = "optuna"

# Parameters that represent each unique optimisation space
corruption_types = ["identity", "full_dense", "block_diagonal"]
optimizers = ["md", "gd"]

for corruption in corruption_types:
    for opt in optimizers:
        study_name = f"study_{corruption}_{opt}"
        group = f"{corruption}_{opt}"
        name = f"hp_search_{corruption}_{opt}"
        
        # Create the batch script as a multi-line string
        template_script = dedent(f"""\
            #!/bin/bash
            #SBATCH --job-name={name}
            #SBATCH --output=slurm-logs/{name}_%j.out
            #SBATCH --error=slurm-logs/{name}_%j.err
            #SBATCH --time=4:00:00
            #SBATCH --partition=gpu
            #SBATCH --gres=gpu:1
            #SBATCH --mem=16G
            #SBATCH --cpus-per-task=4

            module load miniforge
            conda activate $HOME/{CONDA_ENV_NAME}

            export CUDA_DEVICE_ORDER=PCI_BUS_ID

            LOGGING="$SCRATCH/{PROJECT}/{study_name}"

            mkdir -p "$LOGGING"
            CHKP="$LOGGING/last.ckpt"

            cd {REPO_DIR}
            echo "Copying data from {REPO_DIR}/data into $TMP_SHARED/data"
            cp -r "{REPO_DIR}/data" "$TMP_SHARED/data"

        """)

        cmd = [
            "python", "src/train.py",
            f"corruption.corruption_type={corruption}",
            f"optimizer.update_alg={opt}",
            f"hydra.sweeper.study_name={study_name}",
            f"logger.group={group}",
            f"hparams_search={SWEEP_CONFIG}",
            f"logger.project={PROJECT}"
        ]

        # Add the command to run the script
        batch_script = template_script + "\n" + " ".join(cmd) + "\n" + "echo 'Job completed.'\n"

        # Write the script to a temp file (can be named uniquely)
        script_filename = f"tmp.sh"

        with open(script_filename, "w") as f:
            f.write(batch_script)

        # Launch the job using sbatch
        subprocess.run(["sbatch", script_filename])
        break
    break
