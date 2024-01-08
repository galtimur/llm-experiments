from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import os
import torch
import copy
from torch.utils.data import DataLoader
from functools import partial

from utils import process_batch_template, validate, general_train_step

# from transformers import AdamW, SGD
from torch.optim import SGD, AdamW
import wandb
from tqdm import tqdm

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
num_params = sum(p.numel() for p in model.parameters())
print(f"Model size in M = {num_params//1e6:.2f}")

# Load the wikitext dataset
from datasets import load_dataset

datapath = "/mnt/data2/galimzyanov/megatron/wikitext"
outpath = "/mnt/data2/galimzyanov/megatron/gpt2_checkpoints"
prefix = "wiki-text-103-raw-"
filter_gradients = False

args = {
    "max_seq_length": 1024,
    "optimizer": "AdamW",  # "SGD", "AdamW"
    "lr": 2e-4,  # 1e-4, 1e-3, 1e-2
    "val_interval": 240,
    "mini_batch_size": 6,
    "batch_accum_size": 6,
    "epochs": 10,
    "threshold": 0.2,
    "type": "largest",
    "model_size": num_params,
    "model_size_M": f"{num_params/1e6:.2f}",
}

max_seq_length = args["max_seq_length"]
to_log = True
track_loss_change = True
batch_size = args["mini_batch_size"]
batch_accum = args["batch_accum_size"]

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

tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token


def adjust_model_mask(model, model_prev, mask_dict):
    for (_, param_prev), (name, param_curr) in zip(
        model_prev.named_parameters(), model.named_parameters()
    ):
        mask = ~mask_dict[name]
        param_curr.data[mask] = param_prev.data[mask]


process_batch = partial(
    process_batch_template, tokenizer=tokenizer, max_seq_length=max_seq_length
)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, collate_fn=process_batch, shuffle=True
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=process_batch)

# Move the model to the selected device
model.to(device)
model_start = copy.deepcopy(model)
model_start1 = copy.deepcopy(model)

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
    wandb.define_metric("distance", step_metric="samples")
    wandb.define_metric("distance from 1 epoch", step_metric="samples")
    wandb.define_metric("grad step", step_metric="samples")
    wandb.define_metric("std distance", step_metric="samples")
    wandb.define_metric("std step", step_metric="samples")
    wandb.define_metric("grad part", step_metric="samples")
    wandb.define_metric("part common", step_metric="samples")
    wandb.define_metric("learning rate", step_metric="samples")
    wandb.define_metric("loss change", step_metric="samples")
    wandb.config.update(args)

train_step = partial(
    general_train_step,
    model=model,
    optimizer=optimizer,
    model_start=model_start,
    args=args,
    device=device,
    filter_gradients=filter_gradients,
    to_log=to_log,
    track_loss_change=track_loss_change,
)

# Training loop
model.train()
consumed_batches = 0
mask_dict = dict()
num_batches = len(train_loader)

for epoch in range(args["epochs"]):  # number of epochs
    if epoch == 1:
        print("Copying model for distance calc")
        model_start1 = copy.deepcopy(model)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args["lr"],
        pct_start=0.05,
        total_steps=num_batches,
        cycle_momentum = False,
    )
    for batch in tqdm(train_loader):
        consumed_batches += 1
        consumed_samples = batch_size * consumed_batches
        if to_log:
            wandb.log(
                {
                    "batch accum vs samples": batch_accum,
                    "learning rate": optimizer.param_groups[0]["lr"],
                    "samples": consumed_samples,
                },
                commit=True,
            )
        mask_dict = train_step(
            batch=batch,
            model_start1=model_start1,
            mask_dict=mask_dict,
            batch_accum=batch_accum,
            consumed_batches=consumed_batches,
            epoch=epoch,
        )

        if (consumed_batches - 0) % val_interval == 0:
            loss_val = validate(val_loader, model, device)
            if to_log:
                wandb.log(
                    {"val/loss vs samples": loss_val, "samples": consumed_samples},
                    commit=True,
                )
                model.train()
        scheduler.step()

    model.save_pretrained(
        os.path.join(outpath, f"gpt2_cycle_no_moment_batch{batch_accum}_e{epoch}")
    )
