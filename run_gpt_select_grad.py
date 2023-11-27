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
Making a optimizer step only along those weignts,
which grad direction is largest/smallest
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

def filter_grad(model, threshold, type):
    num_el = 0
    num_grad = 0
    for name, param in model.named_parameters():
        num_el += param.grad.data.numel()
        # grad_norm = torch.norm(param.grad.data, p=1)/ param.grad.data.numel()
        param_abs = torch.abs(param.grad.data)
        grad_norm = torch.mean(param_abs)
        grad_max = torch.max(param_abs)
        grad_min = torch.min(param_abs)
        if type == 'largest':
            thresh = grad_norm/threshold
            if thresh > grad_max:
                thresh = (1-threshold)*grad_max
            mask = param_abs > thresh
        if type == 'smallest':
            thresh = grad_norm*threshold
            if thresh < grad_min:
                thresh = (1+threshold)*grad_min
            mask = param_abs < thresh
        param.grad.data = param.grad.data * mask.float()
        param.grad.data = param.grad.data/torch.mean(torch.abs(param.grad.data))*grad_norm/threshold
        num_grad += torch.sum(mask)

    return num_grad/num_el

to_log = True
batch_size = 6
batch_accum = 1
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, collate_fn=process_batch, shuffle=True
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=process_batch)

# Move the model to the selected device
model.to(device)
optimizer = SGD(model.parameters(), lr=1e-3)
val_interval = 200

if to_log:
    run = wandb.init(project="GPT2", entity="timur-galimzyanov")
    wandb.define_metric("samples")
    wandb.define_metric("loss vs samples", step_metric="samples")
    wandb.define_metric("batch accum vs samples", step_metric="samples")
    wandb.define_metric("val/loss vs samples", step_metric="samples")
    wandb.define_metric("grad_part", step_metric="samples")

def train_step(batch, model, optimizer, batch_accum, consumed_batches):
    inputs = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    batch_size = inputs.shape[0]
    consumed_samples = batch_size*consumed_batches

    if batch_accum < batch_size:
        split_inputs = torch.split(inputs, batch_accum, dim=0)
        split_labels = torch.split(labels, batch_accum, dim=0)
        log_dict = {}

        for i, (inputs, labels) in enumerate(zip(split_inputs, split_labels), start=1):
            optimizer.zero_grad()
            loss = model(input_ids=inputs, labels=labels)[0]
            loss.backward()
            grad_part = filter_grad(model, threshold=0.2, type = 'largest')
            log_dict = {"grad_part": grad_part}
            optimizer.step()
            if to_log:
                log_dict.update({"loss vs samples": loss.item(), "samples": consumed_samples})
                wandb.log(log_dict, commit=True)
    else:

        loss = model(input_ids=inputs, labels=labels)[0]
        loss.backward()

        if consumed_samples % batch_accum == 0:
            log_dict = {}
            # if consumed_samples % (batch_accum*10) == 0:
            if consumed_batches > 0:
                grad_part = filter_grad(model, threshold=0.2, type = 'largest')
                log_dict = {"grad_part": grad_part}
            optimizer.step()
            optimizer.zero_grad()
            if to_log:
                log_dict.update({"loss vs samples": loss.item(), "samples": consumed_samples})
                wandb.log(log_dict, commit=True)

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
for epoch in range(4):  # number of epochs
    for batch in tqdm(train_loader):
        consumed_batches += 1
        consumed_samples = batch_size*consumed_batches
        if to_log:
            wandb.log({"batch accum vs samples": batch_accum, "samples": consumed_samples}, commit=True)
        train_step(batch, model, optimizer, batch_accum, consumed_batches)

        if consumed_batches % val_interval == 0:
            loss_val = validate(val_loader, model)
            if to_log:
                wandb.log({"val/loss vs samples": loss_val, "samples": consumed_samples}, commit=True)
            # model.save_pretrained(os.path.join(outpath, f"gpt2_select_grad_e{epoch}"))
            model.train()
