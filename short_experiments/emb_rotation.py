import torch

from data.dataloader import DataloaderFetcher
from trainer.argparser import parse_config
from trainer.trainer import Trainer
import torch.nn.functional as F
from tqdm import tqdm

start_pos = 0
end_pos = 256

"""
Projections of activations x of each layer on the head
(SoftMax(Head(RMSNorm(x)) = vector of dictionary dimension) 
and calculated cosine similarity with the last layer.
Averaged over 1000 samples, activations were taken on the 256th token.
"""


class HeadProjector(torch.nn.Module):
    def __init__(self, model):
        super(HeadProjector, self).__init__()
        self.norm = model.base_model.norm
        self.head = model.lm_head
        pass

    def forward(self, batch):
        batch = self.norm(batch)
        batch = self.head(batch)
        batch = F.softmax(batch, dim=-1)
        return batch


def get_emb_rotation(hidden_states: tuple[torch.Tensor]):

    hidden_states_proj = [
        projector(hidden_state[:, start_pos:end_pos]) for hidden_state in hidden_states
    ]
    start_vectors = hidden_states_proj[-1]
    hidden_states_tens = torch.stack(hidden_states_proj, dim=0)
    cosine_sim = F.cosine_similarity(hidden_states_tens, start_vectors, dim=-1)
    cosine_sim_adjusted = (cosine_sim - cosine_sim[:1]) / (
        cosine_sim[-1:] - cosine_sim[:1]
    )

    return cosine_sim


if __name__ == "__main__":
    config_path = "config/main_config.yaml"
    config = parse_config(config_path)
    fetcher = DataloaderFetcher(config)
    train_dl = fetcher.train_dataloader()
    val_dl = fetcher.val_dataloader()
    # item = next(iter(val_dl))
    trainer = Trainer(config, train_dl, val_dl, perform_sanity_check=False)
    model = trainer.model
    model.eval()
    projector = HeadProjector(model)
    embed_rotations = []
    i = 0
    for batch in tqdm(train_dl):
        batch = {k: v.to(trainer.device) for k, v in batch.items()}
        if batch["input_ids"].size(1) != config.model.sequence_length:
            continue
        with torch.no_grad():
            outputs = model.base_model(
                **batch, output_hidden_states=True, use_cache=False, return_dict=True
            )
            hidden_states = outputs.hidden_states
            embed_rotation = get_emb_rotation(hidden_states)
            embed_rotations.append(embed_rotation)
            i += 1
            if i > 1000:
                break
    embed_rotations = torch.cat(embed_rotations, dim=1)
    embed_rotations_mean = embed_rotations.mean(dim=1)
    embed_rotations_std = embed_rotations.std(dim=1)
    pass
