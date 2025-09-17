from typing import Any, Dict, Optional, Tuple

import lightning as L
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import os

class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir 
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mnist_train: Optional[Dataset] = None
        self.mnist_val: Optional[Dataset] = None
        self.mnist_test: Optional[Dataset] = None
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            torch.nn.Flatten(start_dim=0)
        ])
        self.batch_size = batch_size   
        self.save_hyperparameters()
    
    def prepare_data(self):
        os.makedirs(self.data_dir, exist_ok=True)
        # create data dir and download it into there
        MNIST(root=self.data_dir, train=True, download=True)
        MNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        if stage == "fit":
          
            mnist_full = MNIST(self.data_dir, train=True, download=False, 
                            transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator()
            )
            
        if stage in ["test", "validate"]:
            self.mnist_test = MNIST(self.data_dir, train=False, download=False,
                                transform=self.transform)
        
        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, download=False,
                                    transform=self.transform)
    

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, num_workers=self.num_workers)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        # TODO: have a look into this for cleaning up & multiruns
        pass

from torchvision.datasets import FashionMNIST

class FashionMNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mnist_train: Optional[Dataset] = None
        self.mnist_val: Optional[Dataset] = None
        self.mnist_test: Optional[Dataset] = None
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            torch.nn.Flatten(start_dim=0)
        ])
        self.save_hyperparameters()

    def prepare_data(self):
        os.makedirs(self.data_dir, exist_ok=True)
        FashionMNIST(root=self.data_dir, train=True, download=True)
        FashionMNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        if stage == "fit":
            dataset = FashionMNIST(self.data_dir, train=True, download=False, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(dataset, [55000, 5000])

        if stage in ["test", "validate"]:
            self.mnist_test = FashionMNIST(self.data_dir, train=False, download=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = FashionMNIST(self.data_dir, train=False, download=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, num_workers=self.num_workers)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        # TODO: have a look into this for cleaning up & multiruns
        pass

if __name__ == "__main__":
    _ = MNISTDataModule()
