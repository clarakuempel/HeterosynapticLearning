import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Tuple
from lightning import LightningModule

from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from utils.corruptions import create_corruption_nmatrix  


class MLP_Simple(LightningModule):
    def __init__(self, net, corruption, pruning, optimizer):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        model = net
        cfg_model = model
        self.cfg_model = cfg_model
        cfg_corruption = corruption
        self.cfg_corruption = cfg_corruption
        self.cfg_pruning = pruning
        self.cfg_optimizer = optimizer
        self.save_hyperparameters(logger=False)
            
        self.fc1 = nn.Linear(cfg_model.input_size, cfg_model.hidden_size1)
        self.fc2 = nn.Linear(cfg_model.hidden_size1, cfg_model.hidden_size2)
        self.fc3 = nn.Linear(cfg_model.hidden_size2, cfg_model.hidden_size3)
        self.fc4 = nn.Linear(cfg_model.hidden_size3, cfg_model.output_size)
        self.relu = nn.ReLU()

        # setup corruption matrices if specified
        self.setup_corruption(cfg_corruption.corruption_type, cfg_corruption.alpha, cfg_corruption.block_size)

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()


    def setup_corruption(self, corruption_type, alpha, block_size):
        """Setup the corruption matrices based on the corruption type."""
        C1 = create_corruption_nmatrix(self.cfg_model.hidden_size1, corruption_type, alpha, block_size)
        C2 = create_corruption_nmatrix(self.cfg_model.hidden_size2, corruption_type, alpha, block_size)
        C3 = create_corruption_nmatrix(self.cfg_model.hidden_size3, corruption_type, alpha, block_size)

        # Register buffers to ensure they are part of the model state
        self.register_buffer('C1', C1)
        self.register_buffer('C2', C2)
        self.register_buffer('C3', C3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = x @ self.C1.T  
        x = self.relu(self.fc2(x))
        x = x @ self.C2.T  
        x = self.relu(self.fc3(x))
        x = x @ self.C3.T  #
        x = self.fc4(x)
        return x

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y
 
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss
    
    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single validation step on a batch of data from the validation set."""
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single test step on a batch of data from the test set."""
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        acc = self.val_acc.compute()  
        self.val_acc_best(acc)  
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
