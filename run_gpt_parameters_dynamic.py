from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import os
import torch
import copy
from torch.utils.data import DataLoader
from functools import partial

from utils import process_batch_template, validate, general_train_step, init_wandb

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

from datasets import load_dataset

datapath = "/mnt/data2/galimzyanov/megatron/wikitext"
outpath = "/mnt/data2/galimzyanov/megatron/gpt2_checkpoints"
prefix = "wiki-text-103-raw-"
filter_gradients = False
project_name = "GPT2"
entity_name = "timur-galimzyanov"

args = {
    "max_seq_length": 1024,
    "optimizer": "AdamW",  # "SGD", "AdamW"
    "lr": 2e-4,  # 1e-4, 1e-3, 1e-2
    "val_interval": 240,
    "mini_batch_size": 6,
    "batch_accum_size": 6,
    "epochs": 10,
    "loss change period": 10,
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

# val_set = set(val_dataset["text"])
# test_set = set(test_dataset["text"])
# train_set = set(train_dataset["text"])
# intersection_val = train_set.intersection(val_set)
# intersection_test = train_set.intersection(test_set)
# intersection_test_val = test_set.intersection(val_set)

datapath_out = "/mnt/data2/huggingface/datasets"
val_ood_dataset = load_dataset(
    "json", data_files=os.path.join(datapath_out, "bookcorpus", "splits", "val.json")
)["train"]

# lengths = [len(item["text"]) for item in val_dataset]
# print(sum(lengths) / len(lengths))
#
# lengths = [len(item["text"]) for item in val_ood_dataset]
# print(sum(lengths) / len(lengths))

tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token


process_batch = partial(
    process_batch_template, tokenizer=tokenizer, max_seq_length=max_seq_length
)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, collate_fn=process_batch, shuffle=True
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=process_batch)
val_OOD_loader = DataLoader(
    val_ood_dataset, batch_size=batch_size, collate_fn=process_batch
)

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
    init_wandb(project_name, entity_name, args)

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
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args["lr"],
        pct_start=0.05,
        total_steps=num_batches,
        cycle_momentum=False,
    )
    if epoch == 1:
        print("Copying model for distance calc")
        model_start1 = copy.deepcopy(model)
    batch_prev = None
    skip = True
    for batch in tqdm(train_loader):
        if skip:
            skip = False
            batch_prev = batch
            continue
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
        mask_dict, loss = train_step(
            batch=batch_prev,
            batch_next=batch,
            model_start1=model_start1,
            mask_dict=mask_dict,
            batch_accum=batch_accum,
            consumed_batches=consumed_batches,
            epoch=epoch,
        )

        if (consumed_batches - 0) % val_interval == 0:
            loss_val = validate(val_loader, model, device)
            loss_ood_val = validate(val_OOD_loader, model, device)
            if to_log:
                wandb.log(
                    {
                        "val/loss vs samples": loss_val,
                        "val/loss OOD vs samples": loss_ood_val,
                        "samples": consumed_samples,
                    },
                    commit=True,
                )
                model.train()
        scheduler.step()
        batch_prev = batch

    # model.save_pretrained(
    #     os.path.join(outpath, f"gpt2_batch{batch_accum}_e{epoch}")
    # )
