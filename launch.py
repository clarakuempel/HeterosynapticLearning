import subprocess

# Parameters that represent each unique optimisation space
corruption_types = ["identity", "full_dense", "block_diagonal"]
optimizers = ["md", "gd"]


sweep_config = "corruptions"
for corruption in corruption_types:
    for opt in optimizers:
        study_name = f"study_{corruption}_{opt}"
        print(f"Launching sweep for corruption={corruption}, optimizer={opt}, study_name={study_name}")
        group = f"{corruption}_{opt}"
        
        cmd = [
            "python", "src/train.py",
            f"corruption.corruption_type={corruption}",
            f"optimizer.update_alg={opt}",
            f"hydra.sweeper.study_name={study_name}",
            f"logger.group={group}",
            f"hparams_search={sweep_config}"
        ]

        # print the command for debugging
        print("Running command:", " ".join(cmd))
        
        subprocess.run(cmd)
