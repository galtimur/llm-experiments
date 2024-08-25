import copy
import os

import torch
import torch.nn as nn
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

datapath = "/mnt/data2/galimzyanov/megatron/wikitext"
outpath = "/mnt/data2/galimzyanov/gpt2/checkpoints"
# Define the model and tokenizer
# model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token

args = {
    "max_seq_length": 1024,
    "optimizer": "AdamW",  # "SGD", "AdamW"
    "lr": 1e-4,  # 1e-4, 1e-3, 1e-2
    "val_interval": 240,
    "mini_batch_size": 12,
    "batch_accum_size": 24,
    "epochs": 4,
    "threshold": 0.2,
    "type": "largest",
    # "model_size": num_params,
    # "model_size_M": f"{num_params/1e6:.2f}",
}

max_seq_length = args["max_seq_length"]
to_log = False
batch_size = args["mini_batch_size"]
batch_accum = args["batch_accum_size"]

prefix = "wiki-text-103-raw-"
train_dataset = load_dataset(
    "json", data_files=os.path.join(datapath, prefix + "train.jsonl")
)["train"]
val_dataset = load_dataset(
    "json", data_files=os.path.join(datapath, prefix + "val.jsonl")
)["train"]
test_dataset = load_dataset(
    "json", data_files=os.path.join(datapath, prefix + "test.jsonl")
)["train"]


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


def shift_inputs(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().view(-1)
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))

    return shift_logits, shift_labels


def read_files_val_value(folder):
    files = [
        os.path.join(folder, file_name)
        for file_name in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, file_name))
    ]
    val_lst = []
    for file in files:
        with open(file, "r") as f:
            val_lst.append(float(f.read().strip()))

    return sum(val_lst) / len(val_lst)


train_loader = DataLoader(
    train_dataset, batch_size=batch_size, collate_fn=process_batch, shuffle=True
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=process_batch)

basename = "gpt2_ensemble_d=24_batch24_ver"


def read_model_list(main_folder, basename, num):
    model_list = []
    for i in range(num):
        checkpoint_name = f"{basename}{i}"
        checkpoint_folder = os.path.join(main_folder, checkpoint_name)
        if os.path.exists(checkpoint_folder):
            model_list.append(
                GPT2LMHeadModel.from_pretrained(checkpoint_folder).to(device).eval()
            )

    return model_list


class GPT_ensamble(nn.Module):
    def __init__(self, model_list):
        self.model_list = model_list

    def forward(self, batch):
        outputs = []
        for model in self.model_list:
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=inputs, labels=labels)


model_list = read_model_list(outpath, basename, 32)
mean_val = read_files_val_value(outpath)
model_list = model_list
loss_fun = nn.CrossEntropyLoss()  # ignore_index=tokenizer.pad_token_id

with torch.no_grad():
    loss = 0
    loss_orig = 0
    for n, batch in tqdm(enumerate(val_loader, start=1)):
        logits_agg = None
        inputs = batch["input_ids"].to(device)
        for model in model_list:
            output = model(input_ids=inputs, labels=inputs)
            if logits_agg is None:
                logits_agg = output.logits  # (batch, seq_len, vocab_size)
                loss_orig += output.loss
            else:
                logits_agg += output.logits  # (batch, seq_len, vocab_size)
                # logits_agg = torch.where(torch.abs(logits_agg) >= torch.abs(output.logits), logits_agg, output.logits)
                # logits_agg = torch.max(logits_agg, output.logits)
        logits_agg = logits_agg / len(model_list)
        shift_logits, shift_labels = shift_inputs(logits_agg, inputs)
        loss += loss_fun(shift_logits, shift_labels).item()
    loss = loss / n
    loss_orig = loss_orig / n

print(f"Mean agg loss = {loss}")
# print(loss_orig)
print(f"Mean original loss = {mean_val}")
pass
