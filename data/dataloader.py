from functools import partial

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class DataloaderFetcher:
    def __init__(self, config):
        self.seed = 42
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model,
            trust_remote_code=True,
            padding_side="right",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.complete_collate_fn = partial(
            self.collate_fn,
            text_key=config.data.text_key,
            tokenizer=self.tokenizer,
            max_len=config.model.sequence_length,
        )

        self.dataloader = None
        self._train_dataloader = None
        self._val_dataloader = None
        self.setup()

    @staticmethod
    def collate_fn(batch, text_key, tokenizer, max_len):
        inputs = tokenizer.batch_encode_plus(
            [s[text_key] for s in batch],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_len,
        )
        return inputs

    def prepare_dataset(self):
        train_ds = load_dataset(
            path=self.config.data.train_name_or_path,
            name=self.config.data.train_ds_subset,
        )["train"]
        train_ds = train_ds.shuffle(self.seed)
        self.train_dataset = train_ds

        val_ds = load_dataset(
            path=self.config.data.val_name_or_path,
            name=self.config.data.val_ds_subset,
        )["test"]
        self.val_dataset = val_ds.shuffle(self.seed)

    def setup(self):
        self.prepare_dataset()
        self._train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.train.train_batch_size,
            collate_fn=self.complete_collate_fn,
            num_workers=self.config.general.num_workers,
        )
        self._val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.train.train_batch_size,
            collate_fn=self.complete_collate_fn,
            num_workers=self.config.general.num_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self._train_dataloader

    def val_dataloader(self) -> DataLoader:
        return self._val_dataloader
