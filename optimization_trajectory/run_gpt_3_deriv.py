from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import os
import torch
from torch.utils.data import DataLoader

# from transformers import AdamW, SGD
from torch.optim import SGD, AdamW
import wandb
from tqdm import tqdm
from collections import deque

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

model = GPT2LMHeadModel.from_pretrained(os.path.join(outpath, "gpt2_batch_6_e0"))

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


grad_list = deque(maxlen=3)
param_list = deque(maxlen=3)
grads = dict()
params = dict()
g3 = dict()


def add_3_deriv(model, g3):
    for name, param in model.named_parameters():
        grads[name] = param.grad.clone()
        params[name] = param.data.clone()
    grad_list.append(grads.copy())
    param_list.append(params.copy())

    if len(grad_list) == 3:  # and batch_idx < 10000:
        for name, param in model.named_parameters():
            dx = (param_list[2][name] - param_list[1][name]) * (
                param_list[1][name] - param_list[0][name]
            )
            g3[name] = (
                grad_list[2][name] - 2 * grad_list[1][name] + grad_list[0][name]
            ) / (dx + 0.0001)
            g3[name] = torch.clamp(g3[name], min=-100, max=100)
            g3_norm = torch.norm(g3[name])
            grad_norm = torch.norm(param.grad.data)
            param.grad.data = param.grad.data + grad_norm / g3_norm * g3[name]  #


to_log = True
batch_size = 6
# batch_sizes = [1, 2, 6, 12, 24, 48, 96, 192]
# batch_accum_step = 6000
batch_sizes = [6, 6]
batch_accum_step = 400000000
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
    wandb.define_metric(f"loss vs samples", step_metric="samples")
    wandb.define_metric(f"batch accum vs samples", step_metric="samples")
    wandb.define_metric(f"val/loss vs samples", step_metric="samples")


def train_step(batch, model, optimizer, batch_accum, consumed_batches):
    inputs = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    batch_size = inputs.shape[0]
    consumed_samples = batch_size * consumed_batches

    if batch_accum < batch_size:
        split_inputs = torch.split(inputs, batch_accum, dim=0)
        split_labels = torch.split(labels, batch_accum, dim=0)

        for i, (inputs, labels) in enumerate(zip(split_inputs, split_labels), start=1):
            optimizer.zero_grad()
            loss = model(input_ids=inputs, labels=labels)[0]
            loss.backward()
            add_3_deriv(model, g3)
            optimizer.step()
            if to_log:
                wandb.log(
                    {
                        "loss vs samples": loss.item(),
                        "samples": consumed_samples - batch_size + i * batch_accum,
                    },
                    commit=True,
                )
            # wandb.log(
            #     {"loss vs samples": 1, "samples": consumed_samples - batch_size + i * batch_accum},
            #     commit=True,
            # )
    else:
        loss = model(input_ids=inputs, labels=labels)[0]
        loss.backward()

        if consumed_samples % batch_accum == 0:
            add_3_deriv(model, g3)
            optimizer.step()
            optimizer.zero_grad()
            if to_log:
                wandb.log(
                    {"loss vs samples": loss.item(), "samples": consumed_samples},
                    commit=True,
                )
            # wandb.log({"loss vs samples": 1, "samples": consumed_samples}, commit=True)


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
        consumed_samples = batch_size * consumed_batches
        if consumed_batches // batch_accum_step < len(batch_sizes):
            batch_accum = batch_sizes[consumed_batches // batch_accum_step]
        else:
            batch_accum = batch_sizes[-1]
        if to_log:
            wandb.log(
                {"batch accum vs samples": batch_accum, "samples": consumed_samples},
                commit=True,
            )
        train_step(batch, model, optimizer, batch_accum, consumed_batches)

        if consumed_batches % val_interval == 0:
            loss_val = validate(val_loader, model)
            if to_log:
                wandb.log(
                    {"val/loss vs samples": loss_val, "samples": consumed_samples},
                    commit=True,
                )
            model.save_pretrained(os.path.join(outpath, f"gpt2_alter_grad_e{epoch}"))
            model.train()
