from omegaconf import OmegaConf

from data.dataloader import DataloaderFetcher
from trainer.trainer import Trainer

if __name__ == "__main__":
    config_path = "config/main_config.yaml"
    config = OmegaConf.load(config_path)
    fetcher = DataloaderFetcher(config)
    train_dl = fetcher.train_dataloader()
    val_dl = fetcher.val_dataloader()
    trainer = Trainer(config, train_dl, val_dl)
