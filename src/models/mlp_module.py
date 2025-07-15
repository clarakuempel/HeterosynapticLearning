import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Tuple
from lightning import LightningModule
from src.optimizer.md import HP_SGD

from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.utils import apply_nm_pruning

class MLP_module(LightningModule):
    """
    A lightning module for a simple MLP model with optional corruption and pruning.
    """
    def __init__(self, net, pruning, optimizer):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.cfg_pruning = pruning
        self.cfg_optimizer = optimizer
        # Net
        self.net = net
            
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
        self.save_hyperparameters(ignore=['net', 'optimizer', 'pruning'])        

    def prune(self):
        # TODO: implement the pruning rounds
        apply_nm_pruning(
            self.net,
            self.cfg_pruning['N'],
            self.cfg_pruning['M'],
        )

    def forward(self, x):
        return self.net(x)

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
        # optim alg
        if self.cfg_optimizer['update_alg'] == "md":
            # use the HP_SGD optimizer
            return HP_SGD(
                params=self.parameters(),
                lr=self.cfg_optimizer['lr'],
                momentum=self.cfg_optimizer['momentum'],
                dampening=self.cfg_optimizer['dampening'],
                weight_decay=self.cfg_optimizer['weight_decay'],
                nesterov=self.cfg_optimizer['nesterov'],
                block_size=self.cfg_optimizer['block_size'],
                alpha=self.cfg_optimizer['alpha'],
            )
        elif self.cfg_optimizer['update_alg'] == "gd":
            # use the SGD optimizer
            return torch.optim.SGD(
                params=self.parameters(),
                lr=self.cfg_optimizer['lr'],
                momentum=self.cfg_optimizer['momentum'],
                dampening=self.cfg_optimizer['dampening'],
                weight_decay=self.cfg_optimizer['weight_decay'],
                nesterov=self.cfg_optimizer['nesterov'],
            )
        elif self.cfg_optimizer['update_alg'] == "adam":
            return torch.optim.Adam(
                params=self.parameters(),
                lr=self.cfg_optimizer['lr'],
                weight_decay=self.cfg_optimizer['weight_decay'],
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.cfg_optimizer['update_alg']}")
