import torch
import gc
from tqdm import tqdm
import numpy as np

def batch_list(lst, batch_size):
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


def calc_loss(x, self_emb, X, mask, epsilon=1e-4):

    xself = torch.einsum("ij,ij->i", x, self_emb)
    xX = x @ X.T

    xA = xself[:, None] - xX
    xA = xA * mask
    loss = torch.sum(torch.relu(-xA + epsilon))

    return loss


def calc_bad_embeds(x_optim, self_emb, embeddings, mask):

    xself = torch.einsum("ij,ij->i", x_optim, self_emb)
    xX = x_optim @ embeddings.T
    xA = xself[:, None] - xX
    xA = xA + 1e10 * (1 - mask)
    is_good_embed = torch.all(xA > 0, dim=1)
    bad_embeds = len(is_good_embed) - sum(is_good_embed).item()
    bad_embeds_ratio = bad_embeds / len(is_good_embed)

    return bad_embeds, bad_embeds_ratio, np.array(is_good_embed.cpu())


def train_vectors(n_lst, embeddings, x_optim_start=None, n_steps=100, verbose=False, use_tqdm=True):
    min_bad = len(n_lst)
    X = embeddings
    self_emb = X[n_lst]
    mask = torch.ones((len(n_lst), len(X)), requires_grad=False, device=X.device)
    indices = torch.arange(len(n_lst))
    mask[indices, n_lst] = 0

    if x_optim_start is None:
        x_optim = self_emb.detach().clone()
    else:
        x_optim = x_optim_start.detach().clone()
    x_optim.requires_grad = True
    optimizer = torch.optim.AdamW([x_optim], lr=0.01)

    with torch.no_grad():
        loss = calc_loss(x_optim, self_emb, X, mask)
        bad_embeds, bad_embeds_ratio, is_good_embed = calc_bad_embeds(
            x_optim, self_emb, embeddings, mask
        )
        if verbose:
            print(f"Initial\nloss = {loss.item()}")
            print(f"Bad embeds = {bad_embeds}/{len(n_lst)}, ratio = {bad_embeds_ratio}")

    pbar = tqdm(range(n_steps), disable=not use_tqdm)
    for step in pbar:
        optimizer.zero_grad()

        loss = calc_loss(x_optim, self_emb, X, mask)
        loss.backward()
        optimizer.step()
        if (step + 1) % 10 == 0:
            with torch.no_grad():
                bad_embeds, bad_embeds_ratio, is_good_embed = calc_bad_embeds(
                    x_optim, self_emb, embeddings, mask
                )
            pbar.set_postfix_str(f"Bad embeds: {bad_embeds}/{len(n_lst)}")
            if bad_embeds < min_bad:
                min_bad = bad_embeds
            if bad_embeds_ratio == 0.0:
                break
            # if verbose:
            #     print(f"Step {step + 1}, Loss: {loss.item()}, bad embeds = {bad_embeds}/{len(n_lst)}, ratio = {bad_embeds_ratio}")

    with torch.no_grad():
        loss = calc_loss(x_optim, self_emb, X, mask)
        bad_embeds, bad_embeds_ratio, is_good_embed = calc_bad_embeds(
            x_optim, self_emb, embeddings, mask
        )
    if verbose:
        print("Final")
        print(f"steps = {step+1}, loss = {loss.item()}")
        print(
            f"Bad embeds = {bad_embeds}/{len(n_lst)}, ratio = {bad_embeds_ratio:.4f}, Minimal bad = {min_bad}"
        )

    torch.cuda.empty_cache()
    gc.collect()

    return loss, x_optim, self_emb, mask, is_good_embed