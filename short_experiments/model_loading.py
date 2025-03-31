import gc
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_key_for_target(
    data: dict[str, list[str]], target: str
) -> tuple[str | None, str | None]:
    for key, values in data.items():
        for item in values:
            if target in item:
                return key, item
    raise KeyError(f"Key {target} was not found")


def get_model_path(model_name):
    model_path = snapshot_download(repo_id=model_name)
    model_path = Path(model_path)
    return model_path


def get_model_file_keys(model_path: Path) -> dict[str, list[str]]:

    file_paths_bin = list(model_path.rglob("*.bin"))
    file_paths_safetensor = list(model_path.rglob("*.safetensors"))
    if len(file_paths_safetensor) > 0:
        is_safetensors = True
        model_files = file_paths_safetensor
    elif len(file_paths_bin) > 0:
        is_safetensors = False
        model_files = file_paths_bin
    else:
        raise FileExistsError("No model files found")

    key_lists = dict()
    for model_file in model_files:
        if is_safetensors:
            # For safetensors, we can directly get keys without loading tensors
            with safe_open(model_file, framework="pt") as f:  # type: ignore
                keys = list(f.keys())
            key_lists[str(model_file)] = keys
        else:
            with torch.device("meta"):
                state_dict = torch.load(model_file, weights_only=True)
            keys = list(state_dict.keys())
            key_lists[str(model_file)] = keys
            del state_dict
            gc.collect()

    return key_lists


def _get_weight_by_name(model_name: Path, target: str) -> torch.Tensor:

    model_path = get_model_path(model_name)
    model_file_keys = get_model_file_keys(model_path)
    file_path, target_key = find_key_for_target(model_file_keys, target)
    file_ext = file_path.split(".")[-1]
    if file_ext == "safetensors":
        state_dict = load_file(file_path)
    elif file_ext == ".bin":
        state_dict = torch.load(file_path)
    else:
        raise ValueError(f"{file_ext} is not supported")
    weights = state_dict[target_key]

    return weights


def get_weight_by_name(model_name: Path, target: str) -> torch.Tensor:

    try:
        weights = _get_weight_by_name(model_name, target)
    except KeyError:
        print(f"{target} was not found. Trying to load the full model.")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        if "embed" in target:
            weights = model.model.embed_tokens.weight
        else:
            weights = model.lm_head.weight
        del model
        gc.collect()

    return weights.detach()


# Not used
def get_model_and_embed(model_name):

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for param in model.parameters():
        param.requires_grad = False

    embedding = model.model.embed_tokens.weight.cuda()  # Embedding layer weights
    head = model.lm_head.weight.cuda()
    model_norm = model.model.norm.cuda()

    emb_norms = torch.norm(embedding, dim=1)
    filtered_norms = emb_norms[emb_norms >= 10e-5]
    mean_norm = torch.mean(filtered_norms)

    return model, embedding, head, model_norm, mean_norm, tokenizer
