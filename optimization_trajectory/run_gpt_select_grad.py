import os
from collections import deque

import torch
import wandb

# from transformers import AdamW, SGD
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

from .utils import filter_grad, track_params

"""
Making a optimizer step only along those weignts,
which grad direction is largest/smallest
"""

# import torch.nn.init as init

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Define the model and tokenizer
# model = GPT2LMHeadModel.from_pretrained("gpt2")
config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel(config)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load the wikitext dataset
from datasets import load_dataset

datapath = "/mnt/data2/galimzyanov/megatron/wikitext"
outpath = "/mnt/data2/galimzyanov/megatron/gpt2_checkpoints"
prefix = "wiki-text-103-raw-"

args = {
    "max_seq_length": 1024,
    "type": "largest",
    "optimizer": "AdamW",  # "SGD"
    "lr": 1e-4,
    "val_interval": 200,
    "mini_batch_size": 6,
    "batch_accum_size": 6,
    "threshold": 0.2,
    "epochs": 4,
    "fix_mask_samples": 100000,
}

# model = GPT2LMHeadModel.from_pretrained(os.path.join(outpath, 'gpt2_batch6_grad_select_e3'))

train_dataset = load_dataset(
    "json", data_files=os.path.join(datapath, prefix + "train.jsonl")
)["train"]
val_dataset = load_dataset(
    "json", data_files=os.path.join(datapath, prefix + "val.jsonl")
)["train"]
test_dataset = load_dataset(
    "json", data_files=os.path.join(datapath, prefix + "test.jsonl")
)["train"]

max_seq_length = args["max_seq_length"]

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

    inputs["labels"] = inputs.input_ids.contiguous()  # [:, 1:]
    inputs["input_ids"] = inputs.input_ids.contiguous()  # [:, :-1]

    return inputs


to_log = True
batch_size = args["mini_batch_size"]
batch_accum = args["batch_accum_size"]
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, collate_fn=process_batch, shuffle=True
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=process_batch)

# Move the model to the selected device
model.to(device)
if args["optimizer"] == "SGD":
    optimizer = SGD(model.parameters(), lr=args["lr"])
if args["optimizer"] == "AdamW":
    optimizer = AdamW(model.parameters(), lr=args["lr"])
val_interval = args["val_interval"]

if to_log:
    run = wandb.init(project="GPT2", entity="timur-galimzyanov")
    wandb.define_metric("samples")
    wandb.define_metric("loss vs samples", step_metric="samples")
    wandb.define_metric("batch accum vs samples", step_metric="samples")
    wandb.define_metric("val/loss vs samples", step_metric="samples")
    wandb.define_metric("grad_part", step_metric="samples")
    wandb.config.update(args)


def train_step(batch, model, mask_dict, optimizer, batch_accum, consumed_batches):
    inputs = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    batch_size = inputs.shape[0]
    consumed_samples = batch_size * consumed_batches
    apply_saved_mask = False

    if batch_accum < batch_size:
        split_inputs = torch.split(inputs, batch_accum, dim=0)
        split_labels = torch.split(labels, batch_accum, dim=0)
        log_dict = {}
        for i, (inputs, labels) in enumerate(zip(split_inputs, split_labels), start=1):
            optimizer.zero_grad()
            loss = model(input_ids=inputs, labels=labels)[0]
            loss.backward()
            # if consumed_samples > args["fix_mask_samples"]:
            #     apply_saved_mask = True
            # grad_part, mask_dict = filter_grad(
            #     model, mask_dict, args["threshold"], args["type"], apply_saved_mask
            # )
            grad_part = filter_grad(
                model, mask_dict, args["threshold"], args["type"], apply_saved_mask
            )
            log_dict = {"grad_part": grad_part}
            optimizer.step()
            if to_log:
                log_dict.update(
                    {"loss vs samples": loss.item(), "samples": consumed_samples}
                )
                wandb.log(log_dict, commit=True)
    else:
        loss = model(input_ids=inputs, labels=labels)[0]
        loss.backward()

        if consumed_samples % batch_accum == 0:
            log_dict = {}
            # if consumed_samples % (batch_accum*10) == 0:
            if consumed_samples > args["fix_mask_samples"]:
                apply_saved_mask = True
            # grad_part, mask_dict = filter_grad(
            #     model, mask_dict, args["threshold"], args["type"], apply_saved_mask
            # )
            grad_part = filter_grad(
                model, mask_dict, args["threshold"], args["type"], apply_saved_mask
            )
            log_dict = {"grad_part": grad_part}
            optimizer.step()
            optimizer.zero_grad()
            if to_log:
                log_dict.update(
                    {"loss vs samples": loss.item(), "samples": consumed_samples}
                )
                wandb.log(log_dict, commit=True)
    return mask_dict


def validate(val_loader, model):
    total_eval_loss = 0
    model.eval()
    for n, batch in enumerate(val_loader, start=1):
        with torch.no_grad():
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=inputs, labels=labels)
            eval_loss = outputs[0]
            total_eval_loss += eval_loss.item()
    val_loss = total_eval_loss / n
    print(f"Validation loss = {val_loss:.2f}")
    return val_loss


# Training loop
model.train()
consumed_batches = 0
mask_dict = dict()
for epoch in range(args["epochs"]):  # number of epochs
    for batch in tqdm(train_loader):
        consumed_batches += 1
        consumed_samples = batch_size * consumed_batches
        if to_log:
            wandb.log(
                {"batch accum vs samples": batch_accum, "samples": consumed_samples},
                commit=True,
            )
        mask_dict = train_step(
            batch, model, mask_dict, optimizer, batch_accum, consumed_batches
        )

        if (consumed_batches - 1) % val_interval == 0:
            loss_val = validate(val_loader, model)
            if to_log:
                wandb.log(
                    {"val/loss vs samples": loss_val, "samples": consumed_samples},
                    commit=True,
                )
            thr_str = str(args["threshold"]).replace(".", "_")
            model.save_pretrained(
                os.path.join(
                    outpath,
                    f"gpt2_batch{args['batch_accum_size']}_grad_select_switch{args['fix_mask_samples']}_threshold={thr_str}_e{epoch}",
                )
            )
            model.train()
