from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import os
import torch
from torch.utils.data import DataLoader

# from transformers import AdamW, SGD
from torch.optim import SGD, AdamW
import wandb
from tqdm import tqdm
from collections import deque

'''
Making a step by each matrix step by step.
i.e. several steps on 1 layer, than 2, than 3 
'''

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

# model = GPT2LMHeadModel.from_pretrained(os.path.join(outpath, 'gpt2_batch_6_e0'))

train_dataset = load_dataset(
    "json", data_files=os.path.join(datapath, prefix + "train.jsonl")
)["train"]
val_dataset = load_dataset(
    "json", data_files=os.path.join(datapath, prefix + "val.jsonl")
)["train"]
test_dataset = load_dataset(
    "json", data_files=os.path.join(datapath, prefix + "test.jsonl")
)["train"]

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

    inputs["labels"] = inputs.input_ids.contiguous()  # [:, 1:]
    inputs["input_ids"] = inputs.input_ids.contiguous()  # [:, :-1]

    return inputs

def split_parameters(model, group_size):
    param_groups = []
    current_group_params = 0
    current_group = []
    for param in model.parameters():
        param_size = param.numel()
        if current_group_params + param_size <= group_size:
            current_group.append(param)
            current_group_params += param_size
        else:
            if current_group:
                param_groups.append(current_group)
            current_group = [param]
            current_group_params = param_size

    if current_group:
        param_groups.append(current_group)

    return param_groups

def switch_group(parameter_group, on_index, off_index=None):
    if off_index is not None:
        for param in parameter_group[off_index]:
            param.requires_grad = False
    for param in parameter_group[on_index]:
        param.requires_grad = True

to_log = True
batch_size = 6
batch_accum = 6
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, collate_fn=process_batch, shuffle=True
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=process_batch)
val_interval = 200

if to_log:
    run = wandb.init(project="GPT2", entity="timur-galimzyanov")
    wandb.define_metric("samples")
    wandb.define_metric("loss vs samples", step_metric="samples")
    wandb.define_metric("batch accum vs samples", step_metric="samples")
    wandb.define_metric("val/loss vs samples", step_metric="samples")
    wandb.define_metric("grad_part", step_metric="samples")

def train_step(batch, model, optimizer, batch_accum, consumed_batches, optimizer_step):
    inputs = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    batch_size = inputs.shape[0]
    consumed_samples = batch_size*consumed_batches

    loss = model(input_ids=inputs, labels=labels)[0]
    loss.backward()

    if consumed_samples % batch_accum == 0:
        log_dict = {}
        # if consumed_samples % (batch_accum*10) == 0:
        optimizer.step()
        optimizer.zero_grad()
        if to_log:
            log_dict.update({"loss vs samples": loss.item(), "samples": consumed_samples})
            wandb.log(log_dict, commit=True)
    return optimizer_step + 1

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
model.to(device)
model.train()
for param in model.parameters():
    param.requires_grad = False
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params/1e6:.0f}M")

# optimizer = SGD(model.parameters(), lr=1e-3)
group_size = total_params/100
param_groups = split_parameters(model, group_size)
num_groups = len(param_groups)

# optimizer = torch.optim.SGD([
#     {'params': param_group, 'lr': 1e-3} for param_group in param_groups
# ])

optimizers = [torch.optim.SGD(params=param_group, lr=1e-3) for param_group in param_groups]

consumed_batches = 0
optimizer_step = 0
group_index = 0
switch_group(param_groups, on_index = group_index)
optimizer = optimizers[group_index]

for epoch in range(4):  # number of epochs
    for batch in tqdm(train_loader):
        consumed_batches += 1
        consumed_samples = batch_size*consumed_batches
        if to_log:
            wandb.log({"batch accum vs samples": batch_accum, "samples": consumed_samples}, commit=True)

        optimizer_step = train_step(batch, model, optimizer, batch_accum, consumed_batches, optimizer_step)
        if optimizer_step%100==0:
            group_index_new = (group_index + 1)%num_groups
            switch_group(
                param_groups,
                on_index = group_index_new,
                off_index = group_index
            )
            optimizer = optimizers[group_index_new]
            group_index = group_index_new

        if consumed_batches % val_interval == 0:
            loss_val = validate(val_loader, model)
            if to_log:
                wandb.log({"val/loss vs samples": loss_val, "samples": consumed_samples}, commit=True)
            # model.save_pretrained(os.path.join(outpath, f"gpt2_select_grad_e{epoch}"))
            model.train()
