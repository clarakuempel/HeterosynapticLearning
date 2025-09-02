from typing import Dict, Optional, List
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchtext.datasets import PennTreebank
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator



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
        super().__init__()
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.n_tokens = n_tokens

        # Vocabulary and datasets
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: List[str] = []
        self.vocab_size: int = 0

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.predict_dataset: Optional[Dataset] = None

        self.tokenizer = get_tokenizer("basic_english")

        self.save_hyperparameters()
        
    def prepare_data(self):
        # This will download the data if not already present
        PennTreebank(root=self.data_dir, split=("train", "valid", "test"))

    def _yield_tokens(self, data_iter):
        for line in data_iter:
            yield self.tokenizer(line)
    
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
        # Load dataset splits
        train_iter, val_iter, test_iter = PennTreebank(
            root=self.data_dir, split=("train", "valid", "test")
        )

        # Build vocab if not already built
        if self.vocab_size == 0:
            vocab = build_vocab_from_iterator(
                self._yield_tokens(train_iter),
                specials=["<unk>", "<pad>"]
            )
            vocab.set_default_index(vocab["<unk>"])
            self.word_to_idx = vocab.get_stoi()
            self.idx_to_word = vocab.get_itos()
            self.vocab_size = len(vocab)
            self.n_tokens = self.vocab_size

            print(f"Built vocabulary with {self.vocab_size} tokens")

        # reload iterators (they are exhausted after use)
        train_iter, val_iter, test_iter = PennTreebank(
            root=self.data_dir, split=("train", "valid", "test")
        )

        def encode(data_iter):
            tokens = []
            for line in data_iter:
                for word in self.tokenizer(line):
                    tokens.append(self.word_to_idx.get(word, self.word_to_idx["<unk>"]))
            return tokens

        if stage == "fit":
            self.train_dataset = PennTreebankDataset(encode(train_iter), self.seq_len)
            self.val_dataset = PennTreebankDataset(encode(val_iter), self.seq_len)

        if stage in ["test", "validate"]:
            self.test_dataset = PennTreebankDataset(encode(test_iter), self.seq_len)

        if stage == "predict":
            self.predict_dataset = PennTreebankDataset(encode(test_iter), self.seq_len)
    
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
