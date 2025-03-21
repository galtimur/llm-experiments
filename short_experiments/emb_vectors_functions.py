import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def find_self_embeds(
    embedding: torch.Tensor, tokenizer, query_matrix: torch.Tensor | None = None
) -> tuple[list[int], list[int], list]:
    """
    For each vector of embedding we find the closest vector in a matrix.
    Ideally it should be the same vector. If not, we count and return number of such discrepancies.
    We do it in batches, since all vectors can not fit the GPU
    """

    start = 0
    end = len(embedding)
    all_indices = np.array(range(end))
    result_indices = []
    batch_size = 10000
    batches = [
        range(i, min(i + batch_size, end)) for i in range(start, end, batch_size)
    ]
    eqs = []
    if query_matrix is None:
        query_matrix = embedding
    query_matrix = query_matrix.cuda()
    embedding = embedding.cuda()

    for batch in tqdm(batches):
        # indices = np.array(list(batch))
        X = query_matrix[batch]
        w = torch.argmax(X @ embedding.T, dim=-1).cpu()  # .numpy()
        result_indices.extend(w)
    result_indices = np.array(result_indices)
    eqs = all_indices == result_indices
    fail_indices = all_indices[~eqs]
    fail_result_indices = result_indices[~eqs]

    failed_emb_toks = [tokenizer.decode(idx) for idx in fail_indices]
    failed_res_emb_toks = [tokenizer.decode(idx) for idx in fail_result_indices]
    failed_pairs = list(zip(failed_emb_toks, failed_res_emb_toks))

    return fail_indices.tolist(), fail_result_indices.tolist(), failed_pairs


def get_shadow_ratios(
    fail_indices: list[int], embeddings: torch.Tensor
) -> list[tuple[int, float, float, float]]:
    shadow_ratios = []
    for ind in fail_indices:
        max_sims = torch.sort(embeddings[ind] @ embeddings.T, descending=True)[0][:3]
        max_sim = max_sims[0]
        true_sim = embeddings[ind] @ embeddings[ind]
        true_ratio = true_sim / max_sim
        max_ratio = max_sims[1] / max_sim
        max_ratio_2 = max_sims[2] / max_sim
        shadow_ratios.append(
            (ind, true_ratio.item(), max_ratio.item(), max_ratio_2.item())
        )

    return shadow_ratios
#
from huggingface_hub import snapshot_download
model_name = "meta-llama/Llama-3.1-70B"
model_path = snapshot_download(repo_id=model_name)
print(model_path)
# model, embedding, head, model_norm, mean_norm, tokenizer = get_model_and_embed(
#     model_name
# )
# print(model)