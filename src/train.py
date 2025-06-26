import hydra
from omegaconf import DictConfig, OmegaConf
from lightning import Callback, LightningDataModule, LightningModule, Trainer




def train(cfg : DictConfig) -> None:

    # example usage
    train_loader = DataLoader(MNIST(os.getcwd(), download=True, transform=transforms.ToTensor()))
    trainer = pl.Trainer(max_epochs=1)
    model = MLP_Simple()
    trainer.fit(model, train_dataloaders=train_loader)

    return None    



@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    print(OmegaConf.to_yaml(cfg))

    train(cfg)


    # return optimized metric
    return metric_value



if __name__ == "__main__":
    main()