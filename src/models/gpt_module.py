import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Tuple
from lightning import LightningModule
from src.optimizer.md import HP_SGD

from torchmetrics import MaxMetric, MeanMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy

from src.utils import apply_nm_pruning


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

        # PPL
        self.train_ppl = MeanMetric()
        self.val_ppl = MeanMetric()
        self.test_ppl = MeanMetric()

        # Accuracy
        self.train_acc = Accuracy(task="multiclass", num_classes=vocab_size)
        self.val_acc = Accuracy(task="multiclass", num_classes=vocab_size)
        self.test_acc = Accuracy(task="multiclass", num_classes=vocab_size)
        
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_ppl_best = MinMetric()
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
        
        # Auto-detect task type based on sequence lengths and vocab size
        # Language modeling: x and y have same length, large vocab (>1000)
        # Selective copying: y is much shorter than x, small vocab (<100)
        is_language_modeling = (x.size(-1) == y.size(-1)) and (self.net.config.vocab_size > 1000)
        
        if is_language_modeling:
            # Language modeling: predict next token at each position
            # x: [B, T], y: [B, T] where y[i] = x[i+1]
            logits = self.net(x, num_last_tokens=0)  # Get logits for all positions
        else:
            # Selective copying: predict only the target sequence
            # x = noise + mem + delimiters, y = target sequence only
            logits = self.net(x, num_last_tokens=y.size(-1))

        loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))

        preds = torch.argmax(logits, dim=-1) # B, T

        return loss, preds, y

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        ppl = torch.exp(loss.detach())
        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_ppl(ppl)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/ppl", self.train_ppl, on_step=True, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single validation step on a batch of data from the validation set."""
        loss, preds, targets = self.model_step(batch)

        ppl = torch.exp(loss.detach())
        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_ppl(ppl)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/ppl", self.val_ppl, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single test step on a batch of data from the test set."""
        loss, preds, targets = self.model_step(batch)

        ppl = torch.exp(loss.detach())
        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_ppl(ppl)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/ppl", self.test_ppl, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        acc = self.val_acc.compute()
        ppl = self.val_ppl.compute()
        self.val_acc_best(acc)
        self.val_ppl_best(ppl)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=False)
        self.log("val/ppl_best", self.val_ppl_best.compute(), sync_dist=True, prog_bar=False)

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
                block_size=self.cfg_optimizer['block_size'],
                alpha=self.cfg_optimizer['alpha'],
                momentum=self.cfg_optimizer['momentum'],
            )
        # if the optimizer is not MD we need to separate params into decay and nodecay
        decay_params, nodecay_params = self.net.get_params_for_optimizer()

        optim_groups = [
            {"params": decay_params, "weight_decay": self.cfg_optimizer['weight_decay']},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]

        if self.cfg_optimizer['update_alg'] == "gd":
            # use the SGD optimizer
            return torch.optim.SGD(
                optim_groups,
                lr=self.cfg_optimizer['lr'],
                momentum=self.cfg_optimizer['momentum'],
                dampening=self.cfg_optimizer['dampening'],
                weight_decay=self.cfg_optimizer['weight_decay'],
                nesterov=self.cfg_optimizer['nesterov'],
            )

        elif self.cfg_optimizer['update_alg'] == "adam":
            return torch.optim.Adam(
                optim_groups,
                lr=self.cfg_optimizer['lr'],
                weight_decay=self.cfg_optimizer['weight_decay'],
                betas=(self.cfg_optimizer['beta1'], self.cfg_optimizer['beta2']),
            )

        elif self.cfg_optimizer['update_alg'] == "adamW":
            return torch.optim.AdamW(
                optim_groups,
                lr=self.cfg_optimizer['lr'],
                weight_decay=self.cfg_optimizer['weight_decay'],
                betas=(self.cfg_optimizer['beta1'], self.cfg_optimizer['beta2']),
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.cfg_optimizer['update_alg']}")
