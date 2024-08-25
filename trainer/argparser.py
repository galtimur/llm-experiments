import os
from dataclasses import dataclass, field

import transformers
import yaml

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


@dataclass
class GeneralArguments:
    checkpoints_path: str
    s3_bucket: str | None = field(default=None)
    wandb_project: str | None = field(default=None)
    wandb_entity: str | None = field(default=None)
    wandb_url: str | None = field(default=None)
    upload_to_s3: bool = field(default=False)
    s3_checkpoint_path: str | None = field(default=None)


@dataclass
class ModelArguments:
    sequence_length: int = field()
    name: str | None = field(default=None)
    pretrained: bool = field(default=True)
    use_flash_attn: bool = field(default=True)
    parameters: dict = field(default=None)


@dataclass
class StepsArguments:
    num_epochs: int = field(default=1)
    max_train_samples: int | None = field(default=None)
    max_train_steps: int | None = field(default=None)
    warmup_samples: int | None = field(default=None)
    warmup_steps_or_ratio: int | float | None = field(default=None)
    val_every_sample: int | None = field(default=None)
    val_every_step: int | None = field(default=None)
    max_val_samples: int | None = field(default=None)
    max_val_steps: int | None = field(default=None)
    save_every_sample: int | None = field(default=None)
    save_every_step: int | None = field(default=None)


@dataclass
class LoraArguments:
    lora: bool = field(default=False)
    lora_r: int = field(default=16, metadata={"help": "lora rank"})
    lora_alpha: int = field(default=16)
    lora_bias: str = field(default="none")
    lora_dropout: float = field(default=0.05)
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "o_proj", "k_proj"]
    )


@dataclass
class TrainArguments:
    steps: StepsArguments
    lora: LoraArguments
    scheduler: str
    max_lr: float
    optimiser: str = field(default="adamw")
    weight_decay: float = field(default=0)
    train_mini_batch_size: int = field(default=1)
    train_batch_size: int = field(default=1)
    val_batch_size: int = field(default=1)
    num_gpus: int = field(default=1)
    precision: str = field(default="bf16")
    max_norm: int = field(default=10000)
    do_sanity_check: bool = field(default=True)


@dataclass
class DataArguments:
    train_name_or_path: str
    val_name_or_path: str
    processed_folder: str
    train_ds_subset: str | None = field(default=None)
    val_ds_subset: str | None = field(default=None)
    text_key: str = field(default="text")
    num_workers: int = field(default=0)
    s3_path: str | None = field(default=None)
    seed: int = field(default=42)


@dataclass
class ConfigArguments:
    general: GeneralArguments
    model: ModelArguments
    train: TrainArguments
    data: DataArguments


def process_args(config: ConfigArguments) -> ConfigArguments:

    pass

    return config


def parse_config(config_path: str) -> ConfigArguments:
    with open(config_path, "r") as f:
        config_raw = yaml.safe_load(f)

    config_dict = dict()
    for outer_key, outer_value in config_raw.items():
        for inner_key, inner_value in outer_value.items():
            config_dict[inner_key] = inner_value

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainArguments, GeneralArguments)
    )
    model_args, data_args, train_args, general_args = parser.parse_dict(config_dict)
    train_args.steps = StepsArguments(**train_args.steps)
    train_args.lora = LoraArguments(**train_args.lora)

    config = ConfigArguments(
        general=general_args, model=model_args, train=train_args, data=data_args
    )

    config = process_args(config)

    return config
