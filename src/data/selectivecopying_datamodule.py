from typing import Any, Dict, Optional, Tuple

import lightning as L
import torch
import random
import numpy as np
import torch.nn.functional as F
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


def torch_copying_data(L, M, A, variable=False, variable_length=False, batch_shape=(), one_hot=False, reverse=False):
    """
    Generate a dataset for a sequence copying task.
    This code is adopted from the copying.py script in the S4 repository. The original code can be found at:
    https://github.com/state-spaces/s4/blob/e757cef57d89e448c413de7325ed5601aceaac13/src/dataloaders/datasets/copying.py

    Parameters:
    L (int): Number of padding tokens
    M (int): Number of tokens to memorize
    A (int): Alphabet size
    variable (bool): If True, selective copying task
    variable_length (bool): If True, randomize number of tokens to memorize
    batch_shape (tuple): Shape of the batch
    one_hot (bool): If True, convert the input sequence into a one-hot encoded tensor
    reverse (bool): If True, reverse the order of the target sequence

    Returns:
    tuple: Generated input sequence and target sequence
    """
    if variable_length:
        M = int(random.random() * M) + 1
    tokens = torch.randint(low=1, high=A-1, size=batch_shape+(M,))
    if variable:
        total_batch = int(np.prod(batch_shape))
        inds = torch.stack([
            torch.randperm(L+M)[:M]
            for _ in range(total_batch) 
            ], 0)
        inds = inds.reshape(batch_shape+(M,))
        inds, _ = inds.sort()
    else:
        inds = torch.arange(M).repeat(batch_shape+(1,))
    zeros_x = torch.zeros(batch_shape+(M+L,), dtype=torch.long)
    zeros_x.scatter_(-1, inds, tokens)
    markers = (A-1) * torch.ones(batch_shape+(M,), dtype=torch.long)

    x_ = torch.cat([zeros_x, markers], dim=-1)
    y_ = torch.cat([tokens], dim=-1)
    if reverse: 
        y_ = y_.flip(-1)
    if one_hot: 
        x = F.one_hot(x_, A).float()
    else: 
        x = x_
    y = y_
    return x, y


class SelectiveCopyingDataset(Dataset):
    """
    Dataset for selective copying task that generates data on-the-fly.
    """
    def __init__(self, 
                 length: int,
                 l_noise: int = 100, 
                 l_memorize: int = 10, 
                 n_tokens: int = 10,
                 variable: bool = True,
                 variable_length: bool = False,
                 one_hot: bool = False,
                 reverse: bool = False,
                 seed: Optional[int] = None):
        """
        Args:
            length: Number of samples in the dataset
            l_noise: Number of padding/noise tokens (L)
            l_memorize: Number of tokens to memorize (M)
            n_tokens: Alphabet size (A)
            variable: If True, selective copying task
            variable_length: If True, randomize number of tokens to memorize
            one_hot: If True, convert input to one-hot encoding
            reverse: If True, reverse the target sequence
            seed: Random seed for reproducibility
        """
        self.length = length
        self.l_noise = l_noise
        self.l_memorize = l_memorize
        self.n_tokens = n_tokens
        self.variable = variable
        self.variable_length = variable_length
        self.one_hot = one_hot
        self.reverse = reverse
        
        # Set seed for reproducibility
        if seed is not None:
            self.rng = torch.Generator()
            self.rng.manual_seed(seed)
        else:
            self.rng = None
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Set seed based on index for reproducible samples
        if self.rng is not None:
            old_state = torch.get_rng_state()
            torch.manual_seed(self.rng.initial_seed() + idx)
        
        x, y = torch_copying_data(
            L=self.l_noise,
            M=self.l_memorize,
            A=self.n_tokens,
            variable=self.variable,
            variable_length=self.variable_length,
            batch_shape=(),
            one_hot=self.one_hot,
            reverse=self.reverse
        )
        
        # Restore random state
        if self.rng is not None:
            torch.set_rng_state(old_state)
        
        return x, y


class SelectiveCopyingDataModule(L.LightningDataModule):
    """
    Lightning DataModule for the selective copying task.
    """
    def __init__(self, 
                 l_noise: int = 100,
                 l_memorize: int = 10,
                 n_tokens: int = 10,
                 variable: bool = True,
                 variable_length: bool = False,
                 one_hot: bool = False,
                 reverse: bool = False,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 train_size: int = 10000,
                 val_size: int = 1000,
                 test_size: int = 1000,
                 seed: int = 42):
        """
        Args:
            l_noise: Number of padding/noise tokens (L)
            l_memorize: Number of tokens to memorize (M)
            n_tokens: Alphabet size (A)
            variable: If True, selective copying task
            variable_length: If True, randomize number of tokens to memorize
            one_hot: If True, convert input to one-hot encoding
            reverse: If True, reverse the target sequence
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            train_size: Number of training samples
            val_size: Number of validation samples
            test_size: Number of test samples
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.l_noise = l_noise
        self.l_memorize = l_memorize
        self.n_tokens = n_tokens
        self.variable = variable
        self.variable_length = variable_length
        self.one_hot = one_hot
        self.reverse = reverse
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed
        
        # Store datasets
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.predict_dataset: Optional[Dataset] = None
        
        self.save_hyperparameters()
    
    def prepare_data(self):
        # Nothing to download or prepare for synthetic data
        pass
    
    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = SelectiveCopyingDataset(
                length=self.train_size,
                l_noise=self.l_noise,
                l_memorize=self.l_memorize,
                n_tokens=self.n_tokens,
                variable=self.variable,
                variable_length=self.variable_length,
                one_hot=self.one_hot,
                reverse=self.reverse,
                seed=self.seed
            )
            
            self.val_dataset = SelectiveCopyingDataset(
                length=self.val_size,
                l_noise=self.l_noise,
                l_memorize=self.l_memorize,
                n_tokens=self.n_tokens,
                variable=self.variable,
                variable_length=self.variable_length,
                one_hot=self.one_hot,
                reverse=self.reverse,
                seed=self.seed + 1  # Different seed for validation
            )
        
        if stage in ["test", "validate"]:
            self.test_dataset = SelectiveCopyingDataset(
                length=self.test_size,
                l_noise=self.l_noise,
                l_memorize=self.l_memorize,
                n_tokens=self.n_tokens,
                variable=self.variable,
                variable_length=self.variable_length,
                one_hot=self.one_hot,
                reverse=self.reverse,
                seed=self.seed + 2  # Different seed for test
            )
        
        if stage == "predict":
            self.predict_dataset = SelectiveCopyingDataset(
                length=self.test_size,
                l_noise=self.l_noise,
                l_memorize=self.l_memorize,
                n_tokens=self.n_tokens,
                variable=self.variable,
                variable_length=self.variable_length,
                one_hot=self.one_hot,
                reverse=self.reverse,
                seed=self.seed + 3  # Different seed for prediction
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def teardown(self, stage: str):
        # Clean up if needed
        pass
    
    @property
    def sequence_length(self):
        """Total sequence length (input length)"""
        return self.l_noise + self.l_memorize + self.l_memorize
    
    @property
    def input_size(self):
        """Input size (vocabulary size or one-hot dimension)"""
        return self.n_tokens if self.one_hot else 1
    
    @property
    def output_size(self):
        """Output size (vocabulary size)"""
        return self.n_tokens
    
    @property
    def target_length(self):
        """Target sequence length"""
        return self.l_memorize


if __name__ == "__main__":
    # Test the datamodule
    dm = SelectiveCopyingDataModule(
        l_noise=10,
        l_memorize=5,
        n_tokens=10,
        variable=True,
        batch_size=4,
        train_size=8,
        val_size=4,
        test_size=4
    )
    
    # Setup and test
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    
    # Get a batch
    batch = next(iter(train_loader))
    x, y = batch
    
    print(f"Input shape: {x.shape}")  # [batch size, sequence_length]
    print(f"Target shape: {y.shape}") # [batch size, positions to predict]
    print(f"Sequence length: {dm.sequence_length}") # total input sequence length
    print(f"Input size: {dm.input_size}")   # Input vocabulary size, 1
    print(f"Output size: {dm.output_size}") # Output vocabulary size
    
    print("\nSample input sequence:")
    print(x[0])
    print("\nSample target sequence:")
    print(y[0])
