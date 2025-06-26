import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch




class MLP_Simple(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # TODO: fill

    def forward(self, x):
        # TODO: fill
        return x

    def training_step(self, batch, batch_idx):
        # TODO: fill
        return loss

    def configure_optimizers(self):
        # TODO: fill
        return  torch.optim.Adam(self.parameters(), lr=0.02)