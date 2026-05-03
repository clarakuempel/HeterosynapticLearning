# Heterosynaptic Learning in Artificial Neural Networks

This repository implements heterosynaptic plasticity in artificial neural networks, a biologically-inspired learning mechanism where synaptic connections are competing in the weight space. We implement a custom optimizer that incorporates cross-parameter coupling through block-diagonal Hessian approximations.

## Project Overview

Heterosynaptic plasticity allows neural connections to be influenced by the activity of neighboring synapses, not just their own direct input-output relationships. This project explores how incorporating such mechanisms into artificial neural networks can improve learning and performance.


**Supported Tasks**: MNIST, Fashion-MNIST, Selective Copying, PennTreebank (Work In Progress) 

**Corruption Types**: Identity, Full Dense, Block Diagonal 

**Supported Models**: Basic MLP, NanoGPT

## Code Structure

```
src/
├── train.py                 # Main entry point
├── models/
│   ├── mlp_module.py        # MLP Lightning module
│   ├── gpt_module.py        # GPT Lightning module
│   └── components/
│       ├── dense.py         # MLP with corruption matrices
│       └── nanoGPT.py       # GPT implementation
├── optimizer/
│   └── md.py                # HP_SGD optimizer (mirror descent)
├── data/                    # Data modules for each task
└── utils/
    ├── corruptions.py       # Corruption matrix generation
    └── prune.py             # N:M pruning

analysis/                    # Jupyter notebooks for results visualization
conf/                        # Hydra configuration files
```

**Key files:**
- `src/optimizer/md.py` - The heterosynaptic plasticity optimizer
- `src/models/components/dense.py` - Corruption mechanism implementation

## Installation

### Prerequisites

- Python 3.11
- PyTorch Lightning
- Hydra for configuration management
- Weights & Biases Account for experiment tracking

### Setup

```bash
# Clone the repository
git clone git@github.com:clarakuempel/HeterosynapticLearning.git
cd HeterosynapticLearning

conda create -p $HOME/HL-env python=3.11 -y
conda activate $HOME/HL-env
conda install pytorch==2.2.1 -c pytorch
pip install -r requirements.txt
```

### Weights & Biases Setup

This project uses [Weights & Biases](https://wandb.ai) for experiment tracking.

```bash
wandb login
```

By default, logs go to the `hp-learning-rules` team. To change this, edit `conf/logger/wandb.yaml` or run offline:
```bash
python src/train.py logger.offline=True
```

## Usage

### Basic Training

You can train a model with the default configuration. To change the dataset, model and other hyperparams, use Hydra config. **Key configuration files are located in `conf/`:**

- `config.yaml`: Main configuration file
- `model/`: Model architectures (basic_mlp, nanoGPT)
- `data/`: Dataset configurations (mnist, fmnist, penn_treebank, selective_copying)
- `task/`: Task-specific configurations (mnist, fmnist, penn_treebank, selective_copying)
- `optimizer/`: Optimizer settings including heterosynaptic parameters
- `trainer/`: PyTorch Lightning trainer configurations
- `corruption/`: Corruption mechanisms for robustness studies
- `pruning/`: Neural pruning configurations
- `logger/`: Weights & Biases logging settings
- `hparams_search/`: Hyperparameter search configurations (grid, optuna)

Just change config if needed and run:
```bash 
python src/train.py
```

### Hyperparameter Sweeps

For large-scale experiments, there are currently two scripts:

```bash
# Corruption type sweep (identity, full_dense, block_diagonal) with adamW
python launch_corruption_sweep.py

# Fashion-MNIST alpha parameter grid search (local or SLURM)
python launch_slurm.py
```

### Key Parameters

#### Heterosynaptic Plasticity Settings

- `optimizer.update_alg`: Choose between "gd" (gradient descent), "md" (mirror descent), "adam", or "adamW"
- `optimizer.block_size`: Size of blocks in the Hessian matrix (default: 4)
- `optimizer.alpha`: Coupling strength for heterosynaptic interactions (default: 0.1)

#### Dataset Selection

- `task=mnist`: MNIST digit classification
- `task=fmnist`: Fashion-MNIST classification
- `task=selective_copying model=nanoGPT`: Sequential copying task (requires GPT model)
- `task=penn_treebank`: Character-level language modeling (WIP - torchtext API issues, results not yet competitive)
