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

### Supported Tasks

- **MNIST Classification**: Standard benchmark for image classification
- **Selective Copying**: Sequential task requiring selective attention and memory

## Installation

### Prerequisites

- Python 3.11
- PyTorch Lightning
- Hydra for configuration management
- Weights & Biases for experiment tracking

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd HeterosynapticLearning

# Install dependencies
pip install -r requirements.txt
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
- `model/`: Model architectures (MLP, nanoGPT)
- `data/`: Dataset configurations (MNIST, selective copying)
- `optimizer/`: Optimizer settings including heterosynaptic parameters
- `trainer/`: PyTorch Lightning trainer configurations

### Hyperparameter Sweeps

Run hyperparameter optimization with Optuna:

```bash
python src/train.py -m hparams_search=mnist_optuna
```

Run grid search sweeps:

```bash
python src/train.py -m hparams_search=grid
```

### Key Parameters

#### Heterosynaptic Plasticity Settings

- `optimizer.update_alg`: Choose between "gd" (gradient descent), "md" (mirror descent), "adam", or "adamW"
- `optimizer.block_size`: Size of blocks in the Hessian matrix (default: 4)
- `optimizer.alpha`: Coupling strength for heterosynaptic interactions (default: 0.1)

#### Model Configuration

```bash
# Train with heterosynaptic plasticity
python src/train.py optimizer.update_alg=md optimizer.alpha=0.1 optimizer.block_size=4

# Train with standard gradient descent
python src/train.py optimizer.update_alg=gd

# Change model architecture
python src/train.py model=nanoGPT data=selective_copying
```

### Pruning

Enable neural network pruning:

```bash
python src/train.py pruning.enable=true pruning.N=2 pruning.M=4
```

This applies 2:4 structured sparsity (2 out of every 4 weights are pruned).

## Project Structure

```
src/
├── train.py              # Main training script
├── eval.py              # Evaluation utilities
├── data/                # Dataset implementations
│   ├── mnist_datamodule.py
│   └── selectivecopying_datamodule.py
├── models/              # Model architectures
│   ├── mlp_module.py    # MLP with heterosynaptic learning
│   └── gpt_module.py    # Transformer model
├── optimizer/           # Custom optimizers
│   └── md.py           # HP_SGD optimizer implementation
└── utils/              # Utility functions
    ├── corruptions.py  # Data corruption utilities
    └── prune.py       # Neural pruning functions
```

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

## Logging and Monitoring

The project integrates with Weights & Biases for comprehensive experiment tracking:

- Training/validation/test metrics
- Hyperparameter configurations
- Model checkpoints
- Pruning statistics

Configure logging in `conf/logger/wandb.yaml`.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{heterosynaptic-learning,
  title={Heterosynaptic Learning in Artificial Neural Networks},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo}}
}
``` 