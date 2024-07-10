from datasets import load_dataset
from torch.utils.data import DataLoader
from functools import partial
from utils import process_batch_template

def get_datasets(tokenizer, batch_size, max_seq_length, seed=42, max_val_samples=None):

    train_dataset = load_dataset("openwebtext")["train"].shuffle(seed=seed)
    val_dataset = load_dataset("wikitext", "wikitext-103-raw-v1")["test"].shuffle(seed=seed)

    if max_val_samples is not None:
        val_dataset = val_dataset.select(range(max_val_samples))

    process_batch = partial(
        process_batch_template,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=process_batch,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=process_batch,
        shuffle=True,
    )

    return train_loader, val_loader