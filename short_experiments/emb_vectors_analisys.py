"""
Looking on the matrix transformation of the input embedding layer into the lm_head.
How close we can approximate them with linear transformation?
"""

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_transform_matrix(A, B):
    AtA_inv = torch.inverse(A.T @ A)
    # print(f"matrix inversion time = {time() - start_time:.2f}")

    AtB = A.T @ B
    C = AtA_inv @ AtB

    return C


def check_transform(A, B, C):
    B_pred = A @ C
    residual = B_pred - B
    residual_norm = torch.norm(residual, p="fro")

    return residual_norm / torch.norm(B, p="fro")


def get_norms(tensor):

    norms = torch.norm(tensor, p=2, dim=1)
    print(norms.shape)
    average_norm = norms.mean().item()
    std_norm = norms.std()
    print(f"Average Norm: {average_norm}")
    print(f"Standard Deviation of Norms: {std_norm}")
    print(f"std/mead: {std_norm/average_norm}")


def find_self_embeds(embedding):
    start = 0
    end = len(embedding)
    all_indices = np.array(range(end))
    batch_size = 10000
    batches = [
        range(i, min(i + batch_size, end)) for i in range(start, end, batch_size)
    ]
    eqs = []
    embedding = embedding.cuda()

    for batch in tqdm(batches):
        indices = np.array(list(batch))
        X = embedding[batch]
        w = np.array(torch.argmax(X @ embedding.T, dim=-1).cpu())
        eq = indices == w
        eqs.extend(eq)
    eqs = np.array(eqs)
    print(f"failed matching = {sum(~eqs)}")
    fail_indices = all_indices[~eqs]

    return fail_indices


# %%

model_name = "meta-llama/Meta-Llama-3.1-8B"  # "meta-llama/Meta-Llama-3.1-8B" , "gpt2", "meta-llama/Llama-2-7b-hf" # Example: using GPT-2
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# %%
# Loading embeddings:
embedding = model.model.embed_tokens.weight  # Embedding layer weights
head = model.lm_head.weight  # Output head weights

embedding_cut = embedding[: embedding.size(1)]
head_cut = head[: head.size(1)]
embedding_norm = embedding / torch.norm(embedding)
head_norm = head / torch.norm(head)

# %%
# Checking that the matrices are different
shared = torch.all(embedding == head)
norm_diff = torch.norm(embedding - head)
print(
    f"Relative difference norm between head and emb = {norm_diff/torch.norm(embedding):.4f}"
)

norm_diff = torch.norm(embedding_norm - head_norm)
print(f"Relative difference norm between normalized head and emb = {norm_diff:.4f}")
# %%

trans = get_transform_matrix(embedding, head)
resid = check_transform(embedding, head, trans)
print(f"relative residual norm = {resid:.2f}")

# %%

trans = get_transform_matrix(embedding_cut, head_cut)
resid = check_transform(embedding_cut, head_cut, trans)
print(f"relative residual norm for cut = {resid:.2f}")

# %%

rnd_mat = torch.randn_like(embedding)
test_mat = embedding + 1 * torch.norm(embedding) / torch.norm(rnd_mat) * rnd_mat

trans = get_transform_matrix(embedding, test_mat)
resid = check_transform(embedding, test_mat, trans)
print(f"relative residual norm = {resid:.4f}")

# %%
embedding_norm = (embedding.T / torch.norm(embedding, p=2, dim=1)).T
head_norm = (head.T / torch.norm(head, p=2, dim=1)).T
# %%
get_norms(embedding)
get_norms(head)

# %%

failed_emb = find_self_embeds(embedding)
failed_head = find_self_embeds(head)

# %%

failed_emb_toks = [tokenizer.decode(idx) for idx in failed_emb]
failed_head_toks = [tokenizer.decode(idx) for idx in failed_head]

joined_set = set(failed_emb_toks) | set(failed_head_toks)
