import hydra

from omegaconf import DictConfig, OmegaConf
from lightning import Trainer
from models.mlp_simple import MLP_Simple
from data.mnist_datamodule import MNISTDataModule
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def train(cfg : DictConfig) -> None:
    datamodule = MNISTDataModule(data_dir = cfg.data.data_dir, batch_size = cfg.data.batch_size)
    # model: LightningModule = MLP_Simple(cfg)
    model = hydra.utils.instantiate(cfg.model)

    trainer = Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        max_epochs=cfg.max_epochs,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch
    )

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
