# Heterosynaptic Learning in Artificial Neural Networks

This project implements heterosynaptic plasticity in artificial neural networks, a biologically-inspired learning mechanism where synaptic connections can be modified through indirect mechanisms beyond traditional Hebbian learning. The key innovation is a custom optimizer that incorporates cross-parameter coupling through block-diagonal Hessian approximations.

## Project Overview

Heterosynaptic plasticity allows neural connections to be influenced by the activity of neighboring synapses, not just their own direct input-output relationships. This project explores how incorporating such mechanisms into artificial neural networks can improve learning and performance.

### Key Features

- **Custom HP_SGD Optimizer**: Implements heterosynaptic plasticity using mirror descent with block-diagonal Hessian matrices
- **Configurable Coupling**: Adjustable alpha parameter controls the strength of heterosynaptic interactions
- **Multiple Architectures**: Supports both MLPs and GPT-style transformer models
- **Neural Pruning**: N:M structured sparsity patterns for efficient inference
- **Comprehensive Logging**: Weights & Biases integration for experiment tracking
- **Corruption Analysis**: Built-in support for studying robustness with different corruption types

### Supported Tasks

- **MNIST Classification**: Standard benchmark for image classification
- **Fashion-MNIST Classification**: Fashion item classification benchmark
- **Selective Copying**: Sequential task requiring selective attention and memory
- **Penn Treebank Language Modeling**: Character-level language modeling task

### Corruption Types

The project supports various corruption mechanisms for studying robustness:
- **Identity**: No corruption (baseline)
- **Full Dense**: Dense coupling matrices
- **Block Diagonal**: Structured block-diagonal coupling

## Installation

### Prerequisites

- Python 3.11
- PyTorch Lightning
- Hydra for configuration management
- Weights & Biases for experiment tracking

### Setup

```bash
# Clone the repository
git clone git@github.com:clarakuempel/HeterosynapticLearning.git
cd HeterosynapticLearning

# Install dependencies
pip install -r requirements.txt
```

Or using conda:

```bash
conda create -p $HOME/HL-env python=3.11 -y
conda activate $HOME/HL-env
conda install pytorch==2.2.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install numpy==1.26.4
```

## Usage

### Basic Training

Train a model with default configuration:

```bash
python src/train.py
```

### Configuration Options

The project uses Hydra for configuration management. Key configuration files are located in `conf/`:

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

### Hyperparameter Sweeps

For large-scale experiments, use the launch scripts:

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
- `task=selective_copying`: Sequential copying task
- `task=penn_treebank`: Character-level language modeling (Still Work in Progress)



## Optimizer Details

The HP_SGD optimizer implements heterosynaptic plasticity through:

1. **Block-diagonal Hessian**: Parameters are grouped into blocks with coupling matrices
2. **Mirror Descent Updates**: Uses inverse Hessian for parameter updates instead of raw gradients
3. **Configurable Coupling**: Alpha parameter controls interaction strength between parameters

The update rule for mirror descent is:
```
θ_new = θ_old - lr * H^(-1) * ∇L
```

where H is the block-diagonal Hessian approximation with coupling strength α.