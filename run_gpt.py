from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config
)
import os
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import wandb
from tqdm import tqdm
# import torch.nn.init as init

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Define the model and tokenizer
# model = GPT2LMHeadModel.from_pretrained("gpt2")
config = GPT2Config.from_pretrained('gpt2')
model = GPT2LMHeadModel(config)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load the wikitext dataset
from datasets import load_dataset

datapath = "/mnt/data2/galimzyanov/megatron/wikitext"
outpath = "/mnt/data2/galimzyanov/megatron/gpt2_checkpoints"
prefix = "wiki-text-103-raw-"
train_dataset = load_dataset(
    "json", data_files=os.path.join(datapath, prefix + "train.jsonl"))["train"]
val_dataset = load_dataset(
    "json", data_files=os.path.join(datapath, prefix + "val.jsonl"))["train"]
test_dataset = load_dataset(
    "json", data_files=os.path.join(datapath, prefix + "test.jsonl"))["train"]

max_seq_length = 1024

tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token


def process_batch(batch):
    texts = [item["text"] for item in batch]
    inputs = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
        return_tensors="pt",
    )

    inputs["labels"] = inputs.input_ids.contiguous()#[:, 1:]
    inputs["input_ids"] = inputs.input_ids.contiguous()#[:, :-1]

    return inputs


batch_size = 6
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, collate_fn=process_batch
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=process_batch)

# Move the model to the selected device
model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)
val_interval = 100

run = wandb.init(project="GPT2", entity="timur-galimzyanov")

def train_step(batch, model, optimizer):
    inputs = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    optimizer.zero_grad()
    outputs = model(input_ids=inputs, labels=labels)
    loss = outputs[0]
    # Log the loss
    wandb.log({"loss": loss.item()})
    loss.backward()
    optimizer.step()

def validate(val_loader, model):
    total_eval_loss=0
    model.eval()
    for batch_idx, batch in enumerate(val_loader, start=1):
        with torch.no_grad():
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=inputs, labels=labels)
            eval_loss = outputs[0]
            total_eval_loss += eval_loss.item()
    val_loss = total_eval_loss/(batch_idx)
    print(f"Validation loss = {val_loss:.2f}")
    return val_loss


# Training loop
model.train()
for epoch in range(100):  # number of epochs
    for batch_idx, batch in tqdm(enumerate(train_loader, start=1)):
        train_step(batch, model, optimizer)

        if batch_idx%val_interval==0:
            loss_val = validate(val_loader, model)
            wandb.log({"val/loss": loss_val})
            model.save_pretrained(os.path.join(outpath,f"gpt2_e{epoch}"))
            model.train()