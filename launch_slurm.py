import subprocess
from textwrap import dedent
import os
CONDA_ENV_NAME = "HL-env"
REPO_DIR = os.path.abspath(".")  # adjust if needed
SWEEP_CONFIG = "grid"
PROJECT = f"hydra-sweeps-{SWEEP_CONFIG}"

# Parameters that represent each unique optimisation space
# corruption_types = ["identity", "full_dense", "block_diagonal"]
corruption_types = ["identity"]
alphas = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99] # 8
lrs = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0] # 7
optimizers = ['gd', 'md'] # 2
weight_decays = [0.0001, 0.001, 0.01, 0.1, 1] # 5


# TODO This can be made cleaner and more elegant with kwargs
def launch_job(opt, lr, alpha, weight_decay):
    """
    Launch a job on SLURM with the specified parameters.
    """
    study_name = f"study_{lr}_{opt}_{alpha}"
    group = f"{lr}_{opt}_{alpha}"
    name = f"{lr}_{opt}_{alpha}"

    data_dir = "$TMP_SHARED"
    # Create the batch script as a multi-line string
    template_script = dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={name}
        #SBATCH --output=slurm-logs/{PROJECT}/{name}_%j.out
        #SBATCH --error=slurm-logs/{PROJECT}/{name}_%j.err
        #SBATCH --time=9:00:00
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

        f"optimizer.lr={lr}",
        f"optimizer.update_alg={opt}",
        f"optimizer.alpha={alpha}",
        f"optimizer.weight_decay={weight_decay}",

        f"hydra.sweeper.study_name={study_name}",
        f"hparams_search={SWEEP_CONFIG}",
        f"logger.group={group}",
        f"save_dir=$LOGGING",
        f"logger.project={PROJECT}",
        f"data.data_dir={data_dir}/data",
    ]

    # Add the command to run the script
    batch_script = template_script + "\n" + " ".join(cmd) + "\n" + "echo 'Job completed.'\n"

    # Write the script to a temp file (can be named uniquely)
    script_filename = f"tmp.sh"

    with open(script_filename, "w") as f:
        f.write(batch_script)

    # Launch the job using sbatch
    subprocess.run(["sbatch", script_filename])


for opt in optimizers:
    for lr in lrs:
        if opt == 'md':
            for alpha in alphas:
                launch_job(opt, lr, alpha, 0.0)
        else:  # 'gd'
            for wd in weight_decays:
                launch_job(opt, lr, 0.0, wd)
