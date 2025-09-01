from typing import Dict, Optional, List
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchtext import data, datasets


class PennTreebankDataset(Dataset):
    """
    Dataset for Penn Treebank language modeling task.
    """
    def __init__(self, tokens: List[int], seq_len: int = 35):
        """
        Args:
            tokens: List of token indices
            seq_len: Sequence length for language modeling (BPTT length)
        """
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len
        
    def __len__(self):
        return max(1, (len(self.tokens) - 1) // self.seq_len)
    
    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = min(start_idx + self.seq_len, len(self.tokens) - 1)
        
        # Input sequence
        x = self.tokens[start_idx:end_idx]
        # Target sequence (shifted by 1)
        y = self.tokens[start_idx + 1:end_idx + 1]
        
        # Pad if necessary
        if len(x) < self.seq_len:
            pad_length = self.seq_len - len(x)
            x = F.pad(x, (0, pad_length), value=0)
            y = F.pad(y, (0, pad_length), value=0)
            
        return x, y


class PennTreebankDataModule(L.LightningDataModule):
    """
    Lightning DataModule for Penn Treebank language modeling task.
    """
    
    def __init__(self, 
                 data_dir: str = "./data/penn-treebank",
                 seq_len: int = 35,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 seed: int = 42,
                 n_tokens: int = 10000,
                 debug_mode: bool = False):
        """
        Args:
            data_dir: Directory to store/load the data (not used, torchtext handles caching)
            seq_len: Sequence length for language modeling (BPTT length)
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            seed: Random seed for reproducibility
            n_tokens: Estimated vocabulary size (will be overridden by actual vocab size)
        """
        super().__init__()
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.n_tokens = n_tokens  # Will be updated when vocab is built
        
        # Vocabulary and datasets
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.vocab_size: int = 0
        
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.predict_dataset: Optional[Dataset] = None
        
        self.save_hyperparameters()
        
    def prepare_data(self):
        """Download Penn Treebank data using torchtext."""
        # Create text field for processing
        text_field = data.Field(lower=True)
        
        # Load Penn Treebank dataset - this will download if not present
        train_data, valid_data, test_data = datasets.PennTreebank.splits(text_field)
        
        # Build vocabulary from training data
        text_field.build_vocab(train_data)
    
    def _build_vocab_from_iter(self, data_iter):
        """Build vocabulary from data iterator."""
        word_counts = {}
        
        for line in data_iter:
            words = line.strip().split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Create vocabulary (sort by frequency for consistency)
        vocab = ['<pad>', '<unk>'] + sorted(word_counts.keys())
        
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(vocab)
        
        print(f"Built vocabulary with {self.vocab_size} tokens")
    
    def _build_vocab_from_field(self, text_field):
        """Build vocabulary from torchtext field."""
        # Map torchtext vocab to our format
        self.word_to_idx = text_field.vocab.stoi
        self.idx_to_word = text_field.vocab.itos
        self.vocab_size = len(text_field.vocab)
        self.n_tokens = self.vocab_size  # Update n_tokens to actual vocab size
        
        print(f"Built vocabulary with {self.vocab_size} tokens")
    
    def _tokenize_dataset(self, dataset) -> List[int]:
        """Tokenize data from torchtext dataset and return list of token indices."""
        tokens = []
        for example in dataset:
            # Each example.text contains word tokens, convert to indices
            for word in example.text:
                token_idx = self.word_to_idx.get(word, self.word_to_idx.get('<unk>', 1))
                tokens.append(token_idx)
        return tokens
    
    def setup(self, stage: str):
        """Setup datasets for different stages."""
        # Create text field for processing
        text_field = data.Field(lower=True)
        
        # Load Penn Treebank dataset splits
        train_data, valid_data, test_data = datasets.PennTreebank.splits(text_field)
        
        # Build vocabulary if not already done
        if self.vocab_size == 0:
            text_field.build_vocab(train_data)
            self._build_vocab_from_field(text_field)
        
        if stage == "fit":
            # Training data
            train_tokens = self._tokenize_dataset(train_data)
            self.train_dataset = PennTreebankDataset(train_tokens, self.seq_len)
            
            # Validation data
            val_tokens = self._tokenize_dataset(valid_data)
            self.val_dataset = PennTreebankDataset(val_tokens, self.seq_len)
            
        if stage in ["test", "validate"]:
            # Test data
            test_tokens = self._tokenize_dataset(test_data)
            self.test_dataset = PennTreebankDataset(test_tokens, self.seq_len)
            
        if stage == "predict":
            # Use test data for prediction
            test_tokens = self._tokenize_dataset(test_data)
            self.predict_dataset = PennTreebankDataset(test_tokens, self.seq_len)
    
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
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def teardown(self, stage: str):
        del stage
    
    @property
    def sequence_length(self):
        """Sequence length for language modeling"""
        return self.seq_len
    
    @property
    def input_size(self):
        """Input vocabulary size"""
        return self.vocab_size
    
    @property
    def output_size(self):
        """Output vocabulary size (same as input for language modeling)"""
        return self.vocab_size
    
    def get_n_tokens(self):
        """Number of tokens in vocabulary (for config compatibility)"""
        return self.vocab_size if hasattr(self, 'vocab_size') and self.vocab_size > 0 else self.n_tokens


if __name__ == "__main__":
    # Test the datamodule
    dm = PennTreebankDataModule(
        data_dir="../../data/penn-treebank",
        seq_len=10,  # Short sequence for testing
        batch_size=4
    )
    
    # Prepare and setup
    dm.prepare_data()
    dm.setup("fit")
    
    # Test dataloaders
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    
    # Get a batch
    train_batch = next(iter(train_loader))
    x, y = train_batch
    
    print(f"Vocabulary size: {dm.vocab_size}")
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Sequence length: {dm.sequence_length}")
    
    # Show sample tokens
    print("\nSample input tokens:")
    print(x[0])
    print("Sample target tokens:")
    print(y[0])
    
    # Show decoded text for first sample
    input_words = [dm.idx_to_word[idx.item()] for idx in x[0]]
    target_words = [dm.idx_to_word[idx.item()] for idx in y[0]]
    
    print("\nDecoded input text:")
    print(' '.join(input_words))
    print("Decoded target text:")
    print(' '.join(target_words))