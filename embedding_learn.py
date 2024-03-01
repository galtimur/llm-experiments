import torch
import wandb
from omegaconf import OmegaConf

from embedding_learn_utils import (
    setup_train,
    validate_dist,
    maximize_dist,
    calc_traj_length,
    calc_dist,
)

"""
Learning embedding
"""

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    (
        config,
        embeddings,
        embeddings_pretrained,
        optimizer,
        tokenizer,
        train_loader,
        val_loader,
    ) = setup_train()

    input_ids = next(iter(train_loader))["input_ids"].to(device)

    beta = config.beta
    run_name = f"beta_{beta}"
    run = wandb.init(
        project=config.project_name,
        entity=config.entity_name,
        name=run_name,
        config=OmegaConf.to_container(config, resolve=True),
    )

    validate_dist(
        embeddings, embeddings_pretrained, tokenizer.vocab_size - 1, input_ids=input_ids
    )

    # maximize_dist(embeddings, embeddings_pretrained, optimizer, input_ids, tokenizer)
    for i, item in enumerate(train_loader):
        input_ids = item["input_ids"].to(device)
        rand_ids = torch.randint_like(
            input_ids, 0, tokenizer.vocab_size - 1, device=device
        )
        dist = calc_traj_length(input_ids, embeddings)
        dist_rnd = calc_dist(input_ids, rand_ids, embeddings)
        loss = dist - beta * dist_rnd
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        wandb.log(
            {"dist_sent": dist.item(), "dist_rnd": dist_rnd.item(), "loss": loss.item()}
        )

        if (i + 1) % 200 == 0:
            # print(dist.item())
            with torch.no_grad():
                result = validate_dist(
                    embeddings,
                    embeddings_pretrained,
                    tokenizer.vocab_size - 1,
                    input_ids=input_ids,
                    verbose=False,
                )

                wandb.log(result)

        if i > 100000:
            break

    pass
    print(1)


if __name__ == "__main__":
    main()
