import hydra
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig



def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    
    return None # metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """

    evaluate(cfg)

if __name__ == "__main__":
    main()
