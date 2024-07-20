import torch
from embedding_learn_utils import calc_traj_length
from omegaconf import OmegaConf
from torch.optim import SGD, AdamW
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

from data import get_datasets

"""
Learning embedding
"""
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def setup_train(config_path):
    config = OmegaConf.load(config_path)
    # config = OmegaConf.to_container(config, resolve=True)

    config_model = GPT2Config.from_pretrained("gpt2")
    model = GPT2LMHeadModel(config_model)
    model.to(device)
    # model_pretrained = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    # num_params = model.weight.numel()
    # print(f"Number of parameters = {num_params}")
    if config.optimizer == "SGD":
        optimizer = SGD(model.parameters(), lr=config.learning_rate)
    if config.optimizer == "AdamW":
        optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    train_loader, val_loader = get_datasets(
        tokenizer,
        config.batch_size,
        config.max_seq_length,
        seed=config.shuffle_seed,
        max_val_samples=config.max_val_samples,
    )

    return (
        config,
        model,
        # embeddings_pretrained,
        optimizer,
        tokenizer,
        train_loader,
        val_loader,
    )


def validate(model, dataloader):
    total_loss = 0
    total_len = 0
    model.eval()
    with torch.no_grad():
        for i, item in tqdm(enumerate(dataloader, start=1), disable=True):
            out = model(
                input_ids=item["input_ids"].to(device),
                labels=item["labels"].to(device),
                attention_mask=item["attn_mask"].to(device),
            )
            loss = out["loss"]
            filtered_labels = item["labels"][item["labels"] != -100].to(device)
            traj_len = calc_traj_length(
                filtered_labels.view(-1).unsqueeze(0), model.get_input_embeddings()
            )

            total_loss += loss.item()
            total_len += traj_len

    av_loss = total_loss / i
    av_len = total_len / i
    model.train()

    return av_loss, av_len
