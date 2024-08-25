import copy
import os

import torch
from datasets import load_dataset

# from transformers import AdamW, SGD
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from utils import track_params

import wandb

"""
Making a optimizer step only along those weights,
which grad direction is largest/smallest
"""

# import torch.nn.init as init

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Define the model and tokenizer
# model = GPT2LMHeadModel.from_pretrained("gpt2")
config = GPT2Config.from_pretrained("gpt2")
config.n_embd = 24
config.n_head = 1
model = GPT2LMHeadModel(config)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


datapath = "/mnt/data2/galimzyanov/megatron/wikitext"
outpath = "/mnt/data2/galimzyanov/gpt2/checkpoints"
prefix = "wiki-text-103-raw-"
num_params = sum(p.numel() for p in model.parameters())
print(f"Model size = {num_params}")
print(f"Model size in M = {num_params/1e6:.2f}")

args = {
    "max_seq_length": 1024,
    "optimizer": "AdamW",  # "SGD", "AdamW"
    "lr": 1e-4,  # 1e-4, 1e-3, 1e-2
    "val_interval": 240,
    "mini_batch_size": 24,
    "batch_accum_size": 24,
    "epochs": 4,
    "threshold": 0.2,
    "type": "largest",
    "model_size": num_params,
    "model_size_M": f"{num_params/1e6:.2f}",
}

max_seq_length = args["max_seq_length"]
to_log = False
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


train_loader = DataLoader(
    train_dataset, batch_size=batch_size, collate_fn=process_batch, shuffle=True
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=process_batch)

val_interval = args["val_interval"]

if to_log:
    run = wandb.init(project="GPT2", entity="timur-galimzyanov")
    wandb.define_metric("samples")
    wandb.define_metric("loss vs samples", step_metric="samples")
    wandb.define_metric("batch accum vs samples", step_metric="samples")
    wandb.define_metric("val/loss vs samples", step_metric="samples")
    wandb.define_metric("distance", step_metric="samples")
    wandb.define_metric("grad step", step_metric="samples")
    wandb.define_metric("std distance", step_metric="samples")
    wandb.define_metric("std step", step_metric="samples")
    # wandb.define_metric("grad part", step_metric="samples")
    # wandb.define_metric("part common", step_metric="samples")
    wandb.config.update(args)


def train_step(batch, model, optimizer, batch_accum, consumed_batches, model_start):
    inputs = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    batch_size = inputs.shape[0]
    consumed_samples = batch_size * consumed_batches

    loss = model(input_ids=inputs, labels=labels)[0]
    loss.backward()
    v = dict()
    if consumed_samples % batch_accum == 0:
        model_prev = copy.deepcopy(model)
        optimizer.step()
        if track_params:
            dist, grad_step, std_step, std_dist = track_params(
                model, model_start, model_prev
            )
            optimizer.zero_grad()
            log_dict = {
                "distance": dist,
                "grad step": grad_step,
                "std distance": std_dist,
                "std step": std_step,
            }
        else:
            log_dict = {}
        if to_log:
            log_dict.update(
                {"loss vs samples": loss.item(), "samples": consumed_samples}
            )
            wandb.log(log_dict, commit=True)


def validate(val_loader, model):
    total_eval_loss = 0
    model.eval()
    with torch.no_grad():
        for n, batch in enumerate(val_loader, start=1):
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=inputs, labels=labels)
            eval_loss = outputs[0]
            total_eval_loss += eval_loss.item()
    val_loss = total_eval_loss / n
    print(f"Validation loss = {val_loss:.2f}")
    return val_loss


start_idx = 24
for j in range(8):
    i = j + start_idx
    model.to(device)
    model_start = copy.deepcopy(model)

    if args["optimizer"] == "SGD":
        optimizer = SGD(model.parameters(), lr=args["lr"])
    if args["optimizer"] == "AdamW":
        optimizer = AdamW(model.parameters(), lr=args["lr"])

    model.train()
    consumed_batches = 0
    mask_dict = dict()
    for epoch in range(args["epochs"]):
        for batch in tqdm(train_loader):
            consumed_batches += 1
            consumed_samples = batch_size * consumed_batches
            if to_log:
                wandb.log(
                    {
                        "batch accum vs samples": batch_accum,
                        "samples": consumed_samples,
                    },
                    commit=True,
                )
            mask_dict = train_step(
                batch, model, optimizer, batch_accum, consumed_batches, model_start
            )

            # if (consumed_batches - 0) % val_interval == 0:
            #     loss_val = validate(val_loader, model)
            #     if to_log:
            #         wandb.log(
            #             {"val/loss vs samples": loss_val, "samples": consumed_samples},
            #             commit=True,
            #         )
            #     model.train()
        loss_val = validate(val_loader, model)
        model.train()
        print(40 * "-")
        print(f"Model {i}")
        print(f"Validation loss = {loss_val}")
        print(40 * "-")
    model.save_pretrained(
        os.path.join(
            outpath, f"gpt2_ensemble_d={config.n_embd}_batch{batch_accum}_ver{i}"
        )
    )
    with open(os.path.join(outpath, f"val_loss_ver{i}_{loss_val:.2f}"), "w") as f:
        f.write(f"{loss_val}")
