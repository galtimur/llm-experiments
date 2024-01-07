from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import os
import torch
import copy
from torch.utils.data import DataLoader

from utils import track_params, filter_grad

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
    "lr": 1e-4,  # 1e-4, 1e-3, 1e-2
    "val_interval": 240,
    "mini_batch_size": 6,
    "batch_accum_size": 6,
    "epochs": 10,
    "threshold": 0.2,
    "type": "largest",
    "model_size": num_params,
    "model_size_M": f"{num_params//1e6:.2f}",
}

max_seq_length = args["max_seq_length"]
to_log = True
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


def process_batch(batch):
    texts = [item["text"] for item in batch]
    inputs = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
        return_tensors="pt",
    )

    input_ids = inputs.input_ids
    labels = input_ids.clone().contiguous()
    labels[labels == tokenizer.pad_token_id] = -100
    attn_mask = labels != -100
    inputs["labels"] = labels  # [:, 1:]
    inputs["input_ids"] = inputs.input_ids.contiguous()  # [:, :-1]
    inputs["labels"][inputs["input_ids"] == tokenizer.pad_token_id] = -100
    inputs["attn_mask"] = attn_mask

    return inputs


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
    wandb.config.update(args)


def train_step(
    batch,
    model,
    model_start,
    model_start1,
    mask_dict,
    optimizer,
    batch_accum,
    consumed_batches,
    epoch,
):
    inputs = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attn_mask = batch["attn_mask"].to(device)
    batch_size = inputs.shape[0]
    consumed_samples = batch_size * consumed_batches

    loss = model(input_ids=inputs, labels=labels, attention_mask=attn_mask)[0]
    loss.backward()

    if consumed_samples % batch_accum == 0:
        log_dict = {}
        model_prev = copy.deepcopy(model)
        if filter_gradients:
            grad_part, part_common, mask_dict = filter_grad(
                model,
                mask_dict,
                args["threshold"],
                args["type"],
                apply_saved_mask=False,
            )
            log_dict = {"grad part": grad_part, "part common": part_common}
        else:
            log_dict = {}
        optimizer.step()
        if args["optimizer"] == "AdamW":
            # adjust_model_mask(model, model_prev, mask_dict)
            pass
        dist, dist1, grad_step, std_step, std_dist = track_params(
            model, model_start, model_prev, model_start1
        )
        optimizer.zero_grad()
        if epoch == 0:
            dist1 = 0
        log_dict.update(
            {
                "distance": dist,
                "distance from 1 epoch": dist1,
                "grad step": grad_step,
                "std distance": std_dist,
                "std step": std_step,
            }
        )
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
            attn_mask = batch["attn_mask"].to(device)

            outputs = model(input_ids=inputs, labels=labels, attention_mask=attn_mask)
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
    if epoch == 1:
        print("Copying model for distance calc")
        model_start1 = copy.deepcopy(model)
    for batch in tqdm(train_loader):
        consumed_batches += 1
        consumed_samples = batch_size * consumed_batches
        if to_log:
            wandb.log(
                {"batch accum vs samples": batch_accum, "samples": consumed_samples},
                commit=True,
            )
        mask_dict = train_step(
            batch,
            model,
            model_start,
            model_start1,
            mask_dict,
            optimizer,
            batch_accum,
            consumed_batches,
            epoch,
        )

        if (consumed_batches - 0) % val_interval == 0:
            loss_val = validate(val_loader, model)
            if to_log:
                wandb.log(
                    {"val/loss vs samples": loss_val, "samples": consumed_samples},
                    commit=True,
                )
                model.train()

    model.save_pretrained(os.path.join(outpath, f"gpt2_batch{batch_accum}_e{epoch}"))
