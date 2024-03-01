from torch.optim import SGD, AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from functools import partial

from omegaconf import OmegaConf
import copy

from utils import process_batch_template, validate, general_train_step, init_wandb

"""
Learning embedding
"""


def calc_traj_length(input_ids, embeddings):
    emb = embeddings(input_ids)
    emb = F.normalize(emb, p=2, dim=1)
    distances = torch.norm(emb[:, 1:] - emb[:, :-1], dim=-1)
    distance = distances.sum()

    return distance


def calc_dist(input_ids1, input_ids2, embeddings):
    emb1 = embeddings(input_ids1)
    emb2 = embeddings(input_ids2)
    emb1 = F.normalize(emb1, p=2, dim=1)
    emb2 = F.normalize(emb2, p=2, dim=1)

    distances = torch.norm(emb1 - emb2, dim=-1)
    distance = distances.sum()

    return distance


def validate_dist(
    embeddings,
    embeddings_pretrained,
    vocab_size,
    train_loader=None,
    input_ids=None,
    verbose=True,
):
    if train_loader is not None:
        for i, item in enumerate(train_loader):
            input_ids = item["input_ids"].to(device)
            break

    rand_ids = torch.randint_like(input_ids, 0, vocab_size)
    dist = calc_traj_length(input_ids, embeddings)
    dist_pre = calc_traj_length(input_ids, embeddings_pretrained)

    dist_rnd = calc_traj_length(rand_ids, embeddings)
    dist_rnd_pre = calc_traj_length(rand_ids, embeddings_pretrained)

    # print(
    #     f"sent, sent pre | rnd, rnd pre"
    # )
    # print(
    #     f"{dist.item():.0f}, {dist_pre.item():.0f} | {dist_rnd.item():.0f}, {dist_rnd_pre.item():.0f}"
    # )
    if verbose:
        print(f"sent | rnd")
        print(f"{(dist/dist_pre).item():.2f} | {(dist_rnd/dist_rnd_pre).item():.2f}")

    return {
        "val/dist ratio": (dist / dist_pre).item(),
        "val/dist random ratio": (dist_rnd / dist_rnd_pre).item(),
        "val/dist sent": dist.item(),
        "val/dist sent pre": dist_pre.item(),
        "val/dist rnd": dist_rnd.item(),
        "val/dist rnd pre": dist_rnd_pre.item(),
    }


def test_generate(model, tokenizer, device):
    prompt = "Once upon a time"
    model = model.to(device)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(
        input_ids,
        max_length=100,
        num_beams=5,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def setup_train():
    config_path = "config/embed.yaml"
    config = OmegaConf.load(config_path)
    # config = OmegaConf.to_container(config, resolve=True)

    config_model = GPT2Config.from_pretrained("gpt2")
    model = GPT2LMHeadModel(config_model)
    model_pretrained = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    # test_generate(model, tokenizer, device)

    embeddings = copy.deepcopy(model.get_input_embeddings()).to(device)
    embeddings_pretrained = copy.deepcopy(model_pretrained.get_input_embeddings()).to(
        device
    )
    num_params = embeddings.weight.numel()
    print(f"Number of parameters = {num_params}")
    if config.optimizer == "SGD":
        optimizer = SGD(embeddings.parameters(), lr=config.learning_rate)
    if config.optimizer == "AdamW":
        optimizer = AdamW(embeddings.parameters(), lr=config.learning_rate)

    train_dataset = load_dataset("openwebtext")["train"]
    val_dataset = load_dataset("wikitext", "wikitext-103-raw-v1")["test"]

    process_batch = partial(
        process_batch_template,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=process_batch,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=process_batch,
        shuffle=True,
    )

    return (
        config,
        embeddings,
        embeddings_pretrained,
        optimizer,
        tokenizer,
        train_loader,
        val_loader,
    )


def maximize_dist(embeddings, embeddings_pretrained, optimizer, input_ids, tokenizer):
    i = 0
    while True:
        rand_ids = torch.randint(tokenizer.vocab_size, (3000,), device=device)
        emb = embeddings(rand_ids)
        emb = F.normalize(emb, p=2, dim=1)
        dists = torch.cdist(emb, emb, p=2).sum()
        loss = -dists
        loss.backward()
        if (i + 1) % 100 == 0:
            print(dists.item())
            validate_dist(
                embeddings,
                embeddings_pretrained,
                tokenizer.vocab_size - 1,
                input_ids=input_ids,
            )
        optimizer.step()
        optimizer.zero_grad()
        if i > 1000:
            break

        i += 1
