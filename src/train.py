import hydra

from omegaconf import DictConfig, OmegaConf
from lightning import Trainer
from data.mnist_datamodule import MNISTDataModule
from lightning.pytorch.loggers import Logger
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def train(cfg : DictConfig) -> None:
    """
    Trains the model
    """

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    print(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    print(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    print(f"Instantiating logger ...")
    logger: Logger = hydra.utils.instantiate(cfg.logger)

    print(f"Instantiating trainer ...")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    trainer.fit(model, datamodule)

    # TODO: fix this
    trainer.validate(model, datamodule)
    trainer.test(model, datamodule)

    return None    


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    print(OmegaConf.to_yaml(cfg))
    train(cfg)
    return None



if __name__ == "__main__":
    main()
