from data.dataloader import DataloaderFetcher
from trainer.argparser import parse_config
from trainer.trainer import Trainer

if __name__ == "__main__":
    config_path = "config/main_config.yaml"
    config = parse_config(config_path)
    fetcher = DataloaderFetcher(config)
    train_dl = fetcher.train_dataloader()
    val_dl = fetcher.val_dataloader()
    trainer = Trainer(config, train_dl, val_dl, perform_sanity_check=False)

    # trainer.sanity_check()
    trainer.run_training()
    # trainer.validation()

    pass

from tqdm import tqdm

for item in tqdm(val_dl):
    pass
