import os.path

import torch
from gpt_utils import setup_train, validate
from omegaconf import OmegaConf
from tqdm import tqdm

import wandb

"""
Learning embedding
"""

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    config_path = "config/GPT.yaml"
    (
        config,
        model,
        optimizer,
        tokenizer,
        train_loader,
        val_loader,
    ) = setup_train(config_path)

    emb_ckpt_path = "/mnt/data2/galimzyanov/gpt2/embedings_ckpts/beta_0.5_AdamW_lr0.001/checkpoint-100000.pth"
    embeddings_state_dict = torch.load(emb_ckpt_path)
    model.get_input_embeddings().load_state_dict(embeddings_state_dict)

    run_name = f"GPT_embed_batch_{config.batch_size}_{config.optimizer}_lr{config.learning_rate}"
    run = wandb.init(
        project=config.project_name,
        entity=config.entity_name,
        name=run_name,
        config=OmegaConf.to_container(config, resolve=True),
    )
    config.out_folder = os.path.join(config.out_folder, run_name)
    os.makedirs(config.out_folder, exist_ok=True)

    loss_val, traj_len = validate(model, val_loader)
    wandb.log({"val/loss": loss_val, "val/traj_len": traj_len})

    for i, item in tqdm(enumerate(train_loader, start=1)):
        out = model(
            input_ids=item["input_ids"].to(device),
            labels=item["labels"].to(device),
            attention_mask=item["attn_mask"].to(device),
        )
        loss = out["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        wandb.log({"loss": loss.item()})

        if i % config.validate_per == 0:
            loss_val, traj_len = validate(model, val_loader)
            wandb.log({"val/loss": loss_val, "val/traj_len": traj_len})

        if i % config.save_period == 0:
            ckpt_path = os.path.join(config.out_folder, f"checkpoint-{i}")
            model.save_pretrained(ckpt_path)

        if i > config.max_steps:
            break

    pass
    print(1)


if __name__ == "__main__":
    main()
