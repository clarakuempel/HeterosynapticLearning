import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf
from lightning import Trainer
from lightning.pytorch.core.module import LightningModule
from lightning.pytorch.core.datamodule import LightningDataModule
from lightning.pytorch.loggers import Logger
import rootutils
from typing import Optional
from utils.log_config import prepare_config
import wandb
from lightning.pytorch.loggers.wandb import WandbLogger

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def train(cfg : DictConfig) -> Optional[float]:
    """
    Trains the model
    """

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    print(f"Instantiating logger ...")
    logger: Logger = hydra.utils.instantiate(cfg.logger)
    full_config = OmegaConf.to_container(cfg, resolve=True)
    logger.experiment.config.update(prepare_config(full_config))

    print(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    print(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    print(f"Instantiating trainer ...")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    trainer.fit(model, datamodule)

    trainer.validate(model, datamodule)

    # If pruning is enabled prune and re-train
    # NOTE: for now no pruning rounds
    if cfg['pruning']['enable']:
        model.prune()
        trainer.validate(model, datamodule)
        print("Re-initializing trainer for pruning...")
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
        trainer.fit(model, datamodule)

    trainer.validate(model, datamodule)
    trainer.test(model, datamodule)

    metric_value = model.val_acc_best.compute().item()
    wandb.finish()
    return metric_value


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # print(OmegaConf.to_yaml(cfg))
    return train(cfg) 



if __name__ == "__main__":
    main()
