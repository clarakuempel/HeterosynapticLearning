import subprocess
from textwrap import dedent
from itertools import product
import os
CONDA_ENV_NAME = "HL-env"
REPO_DIR = os.path.abspath(".")  # adjust if needed
SWEEP_CONFIG = "grid"
PROJECT = f"rerun-corruption-{SWEEP_CONFIG}"

# Parameters that represent each unique optimisation space
grid = {
    "default": {
        "corruption.corruption_type": ["block_diagonal", "full_dense"],
        "optimizer.lr": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
    },
        
    "md": {
        "optimizer.update_alg": ['md'],
        "optimizer.block_size": [2, 4, 6, 8],
        "optimizer.alpha": [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
    },
    "gd": {
        "optimizer.update_alg": ['gd'],
        "optimizer.weight_decay": [0.0001, 0.001, 0.01],
    }
}
    

def launch_job(**hp):
    """
    Launch a job on SLURM with the specified parameters.

    args == hyper params
    """
    name = "_".join([str(hp[k]) for k in sorted(hp)])
    study_name = f"study_{name}"
    group = name

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
        f"hydra.sweeper.study_name={study_name}",
        f"hparams_search={SWEEP_CONFIG}",
        f"logger.group={group}",
        f"save_dir=$LOGGING",
        f"logger.project={PROJECT}",
        f"data.data_dir={data_dir}/data",
    ]

    # the keu is the name of the hyperparameter, the value is the value to set it to
    for key, value in hp.items():
        cmd.append(f"{key}={value}")

    # Add the command to run the script
    batch_script = template_script + "\n" + " ".join(cmd) + "\n" + "echo 'Job completed.'\n"

    # Write the script to a temp file (can be named uniquely)
    script_filename = f"tmp.sh"

    with open(script_filename, "w") as f:
        f.write(batch_script)

    # Launch the job using sbatch
    subprocess.run(["sbatch", script_filename])


for space, params in grid.items():
    if space == "default":
        continue

    # Add the default parameters to the grid
    full_params = {**grid["default"], **params}
    keys = sorted(full_params.keys())
    values_list = [full_params[key] for key in keys]

    for values in product(*values_list):
        hp = {
            key: value
            for key, value in zip(keys, values)
        }
        # Launch the job with the hyperparameters
        launch_job(**hp)
