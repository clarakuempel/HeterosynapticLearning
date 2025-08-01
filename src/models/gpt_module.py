import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Tuple
from lightning import LightningModule
from src.optimizer.md import HP_SGD

from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.utils import apply_nm_pruning

import hydra

class GPT_module(LightningModule):
    """
    A lightning module for a nanoGPT
    """
    def __init__(self, net, optimizer): # for now only net, future add pruning and corruption
        super().__init__()
        self.net = net
        self.cfg_optimizer = optimizer

        # criterion 
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        
        # metric objects for calculating and averaging accuracy across batches
        vocab_size = net.config.vocab_size
        self.train_acc = Accuracy(task="multiclass", num_classes=vocab_size)
        self.val_acc = Accuracy(task="multiclass", num_classes=vocab_size)
        self.test_acc = Accuracy(task="multiclass", num_classes=vocab_size)
        
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.save_hyperparameters(ignore=['net'])        

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

        x_size = x.size(-1)
        l_memorize = y.size(-1)
        batch_size = x.size(0)
        x = torch.cat((x, torch.zeros(batch_size, l_memorize, dtype=x.dtype, device=x.device)), dim=-1)
        y = torch.cat((torch.zeros(batch_size, x_size, dtype=y.dtype, device=y.device), y), dim=-1)

        logits = self.net(x)

        # mask out the x part of the logits
        loss_mask = torch.ones_like(x, device=x.device, dtype=torch.bool)
        loss_mask[:, :x_size] = False

        logits = logits[loss_mask].view(batch_size, -1, logits.size(-1))  # B, T, C
        y = y[loss_mask].view(batch_size, -1)  # B, T

        # print(f"logits: {logits.shape}, y: {y.shape}")
        # Flatten logits and y for loss calculation
        loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))

        preds = torch.argmax(logits, dim=-1) # B, T
        # print(f"preds: {preds.shape}, y: {y.shape}")

        return loss, preds, y

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
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
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)

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
        # if the optimizer is not MD we need to separate params into decay and nodecay
        decay_params, nodecay_params = self.net.get_params_for_optimizer()

        optim_groups = [
            {"params": decay_params, "weight_decay": self.cfg_optimizer['weight_decay']},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]


        optimizer = None
        if self.cfg_optimizer['update_alg'] == "md":
            # use the HP_SGD optimizer
            optimizer =  HP_SGD(
                params=self.parameters(),
                lr=self.cfg_optimizer['lr'],
                block_size=self.cfg_optimizer['block_size'],
                alpha=self.cfg_optimizer['alpha'],
                momentum=self.cfg_optimizer['momentum'],
            )
        elif self.cfg_optimizer['update_alg'] == "gd":
            # use the SGD optimizer
            optimizer = torch.optim.SGD(
                optim_groups,
                lr=self.cfg_optimizer['lr'],
                momentum=self.cfg_optimizer['momentum'],
                dampening=self.cfg_optimizer['dampening'],
                weight_decay=self.cfg_optimizer['weight_decay'],
                nesterov=self.cfg_optimizer['nesterov'],
            )

        elif self.cfg_optimizer['update_alg'] == "adam":
            optimizer = torch.optim.Adam(
                optim_groups,
                lr=self.cfg_optimizer['lr'],
                weight_decay=self.cfg_optimizer['weight_decay'],
                betas=(self.cfg_optimizer['beta1'], self.cfg_optimizer['beta2']),
            )

        elif self.cfg_optimizer['update_alg'] == "adamW":
            optimizer = torch.optim.AdamW(
                optim_groups,
                lr=self.cfg_optimizer['lr'],
                weight_decay=self.cfg_optimizer['weight_decay'],
                betas=(self.cfg_optimizer['beta1'], self.cfg_optimizer['beta2']),
            )

        else:
            raise ValueError(f"Unknown optimizer: {self.cfg_optimizer['update_alg']}")

        lr_scheduler = None
        if self.cfg_optimizer['lr_scheduler'] == "cosine":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.cfg_optimizer['T_max'],
                eta_min=self.cfg_optimizer['eta_min']
            )
        elif self.cfg_optimizer['lr_scheduler'] == "steplr":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=self.cfg_optimizer['step_size'],
                gamma=self.cfg_optimizer['gamma']
            )
        elif self.cfg_optimizer['lr_scheduler'] != "None":
            raise ValueError(f"Unknown lr_scheduler: {self.cfg_optimizer['lr_scheduler']}")

        config = {
            "optimizer": optimizer,
        }
            
        if lr_scheduler is not None:
            config["lr_scheduler"] = {
                "scheduler": lr_scheduler,
                "monitor": "val/acc_best"
            }

        return config
