import subprocess
from textwrap import dedent
import os
CONDA_ENV_NAME = "HL-env"
REPO_DIR = os.path.abspath(".")  # adjust if needed
SWEEP_CONFIG = "optuna"
PROJECT = f"hydra-sweeps-{SWEEP_CONFIG}"

# Parameters that represent each unique optimisation space
corruption_types = ["identity", "full_dense", "block_diagonal"]
optimizers = ["md", "gd"]
alphas = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

for corruption in corruption_types:
    for opt in optimizers:
        for alpha in alphas:
            study_name = f"study_{corruption}_{opt}_{alpha}"
            group = f"{corruption}_{opt}_{alpha}"
            name = f"{corruption}_{opt}_{alpha}"

            data_dir = "$TMP_SHARED"
            # Create the batch script as a multi-line string
            template_script = dedent(f"""\
                #!/bin/bash
                #SBATCH --job-name={name}
                #SBATCH --output=slurm-logs/{PROJECT}/{name}_%j.out
                #SBATCH --error=slurm-logs/{PROJECT}/{name}_%j.err
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

                cd $LOGGING
                echo "Copying data from {REPO_DIR}/data into {data_dir}/data"
                cp -r "{REPO_DIR}/data" "{data_dir}/data"

            """)

            cmd = [
                "python", f"{REPO_DIR}/src/train.py", "-m", 
                f"corruption.corruption_type={corruption}",
                f"optimizer.update_alg={opt}",
                f"optimizer.alpha={alpha}",
                f"hydra.sweeper.study_name={study_name}",
                f"hparams_search={SWEEP_CONFIG}",

                f"logger.group={group}",
                f"logger.save_dir=$LOGGING",
                f"logger.project={PROJECT}",
                f"data.data_dir={data_dir}/data"
            ]

            if opt == "md":
                # disable wd for mirror descent
                cmd.append(f"hydra.sweeper.params.optimizer.weight_decay=choice(0)")

            # Add the command to run the script
            batch_script = template_script + "\n" + " ".join(cmd) + "\n" + "echo 'Job completed.'\n"

            # Write the script to a temp file (can be named uniquely)
            script_filename = f"tmp.sh"

            with open(script_filename, "w") as f:
                f.write(batch_script)

            # Launch the job using sbatch
            subprocess.run(["sbatch", script_filename])
