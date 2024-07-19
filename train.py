from omegaconf import OmegaConf

from data.dataloader import DataloaderFetcher


if __name__=="__main__":
    config_path = "config/main_config.yaml"
    config = OmegaConf.load(config_path)
    fetcher = DataloaderFetcher(config)
    train_dl = fetcher.train_dataloader()
    val_dl = fetcher.val_dataloader()

    
